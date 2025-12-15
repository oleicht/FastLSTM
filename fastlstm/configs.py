import torch
import triton


SM_count = torch.cuda.get_device_properties("cuda").multi_processor_count
shared_memory = torch.cuda.get_device_properties("cuda").shared_memory_per_block


def get_graph_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_H": h,
                "BLOCK_SIZE_B": b,
                "BLOCK_SIZE_K": k,
                "GROUP_SIZE_B": 8,
            },
            num_warps=w,
            num_stages=s,
        )
        for s in [1, 2, 3, 4, 5, 6, 7, 8]
        for w in [1, 2, 4, 8]
        for k in [32, 64]
        for h in [16, 32, 64, 128, 256]
        for b in [16, 32, 64, 128]
    ]


def compute_batch_layout(hidden_size, BLOCK_SIZE_H, batch_size, BLOCK_SIZE_B):
    max_grid_size = SM_count
    num_pid_h = triton.cdiv(hidden_size, BLOCK_SIZE_H)
    num_pid_b = triton.cdiv(batch_size, BLOCK_SIZE_B)
    batch_chunks = 1
    if num_pid_h * num_pid_b > max_grid_size:
        num_pid_b = triton.next_power_of_2(num_pid_b) // 2
        batch_chunks *= 2
        while num_pid_h * num_pid_b > max_grid_size:
            num_pid_b //= 2
            batch_chunks *= 2

    return {
        "batch_chunks": batch_chunks,
        "num_pid_b": num_pid_b,
        "BLOCK_SIZE_B": BLOCK_SIZE_B,
        "BLOCK_SIZE_H": BLOCK_SIZE_H,
    }


def compute_persistent_grid_dim(kwargs):
    num_pid_b = kwargs["num_pid_b"]
    if "num_pid_h" in kwargs:
        num_pid_h = kwargs["num_pid_h"]
    else:
        num_pid_h = triton.cdiv(kwargs["hidden_size"], kwargs["BLOCK_SIZE_H"])

    assert num_pid_b * num_pid_h <= SM_count

    return (num_pid_b * num_pid_h,)


def naive_smem_estimation(n_bytes, K, M, N, num_stages):
    """the accumualtor lives in the registers"""
    return num_stages * n_bytes * (K * M + K * N) / shared_memory


def prune_graph_configs(configs, named_args, **kwargs):
    bwd = "d_out_ptr" in kwargs

    if bwd:
        k_infl = 4
        h_infl = 1
    else:
        k_infl = 1
        h_infl = 4

    hidden_size = kwargs["hidden_size"]
    batch_size = kwargs["batch_size"]

    configs = [
        c
        for c in configs
        if (
            (c.kwargs["BLOCK_SIZE_H"] < 2 * hidden_size)
            or (hidden_size < 16 and c.kwargs["BLOCK_SIZE_H"] < 32)
        )
    ]
    configs_new = []
    for c in configs:
        if not (
            (c.kwargs["BLOCK_SIZE_B"] < 2 * batch_size)
            or (batch_size < 8 and c.kwargs["BLOCK_SIZE_B"] < 32)
        ):
            continue

        rmem = naive_smem_estimation(
            n_bytes=2 if kwargs["dtype"].endswith("16") else 4,
            K=k_infl * c.kwargs["BLOCK_SIZE_K"],
            M=h_infl * c.kwargs["BLOCK_SIZE_H"],
            N=c.kwargs["BLOCK_SIZE_B"],
            num_stages=c.all_kwargs()["num_stages"],
        )

        if rmem > 3:
            continue

        tile_size = h_infl * c.kwargs["BLOCK_SIZE_H"] * c.kwargs["BLOCK_SIZE_B"]

        if tile_size <= 256:
            target_num_warps = [1]
            target_num_stages = [4, 5, 6]
        elif tile_size <= 1024:
            target_num_warps = [2]
            target_num_stages = [4, 5]
        elif tile_size <= 2048:
            target_num_warps = [4]
            target_num_stages = [2, 3, 4]
        elif tile_size <= 4096:
            target_num_warps = [4, 8]
            target_num_stages = [3, 4]
        else:
            target_num_warps = [8]
            target_num_stages = [1, 2, 3]

        if c.all_kwargs()["num_warps"] not in target_num_warps:
            continue

        if c.all_kwargs()["num_stages"] not in target_num_stages:
            continue

        configs_new += [c]

    def _mm_intensity(c):
        M = h_infl * c.kwargs["BLOCK_SIZE_H"]
        N = c.kwargs["BLOCK_SIZE_B"]
        K = (
            triton.cdiv(k_infl * hidden_size, c.kwargs["BLOCK_SIZE_K"])
            * c.kwargs["BLOCK_SIZE_K"]
        )
        return M * N * K / (M * N + K * M + K * N)

    max_intensity = max(_mm_intensity(c) for c in configs_new)
    configs_new = [c for c in configs_new if _mm_intensity(c) >= max_intensity / 4]
    return configs_new


def prune_persistent_configs(configs, named_args, **kwargs):
    """
    - select from a big grid of configs the relevant ones
    - dynamically compute and overwrite some attributes
    """

    hidden_size = kwargs["hidden_size"]
    batch_size = kwargs["batch_size"]
    fully_fused_pers = "W_x_ptr" in kwargs

    configs = [
        c
        for c in configs
        if (
            (
                triton.cdiv(hidden_size, c.kwargs["BLOCK_SIZE_H"]) <= SM_count
            )  # prevent deadlocks
            and (
                (c.kwargs["BLOCK_SIZE_H"] < 2 * hidden_size)
                or (hidden_size < 16 and c.kwargs["BLOCK_SIZE_H"] < 32)
                or fully_fused_pers
            )  # these configs do unnecessary computations
        )
    ]
    assert len(configs) > 0, "Problem with block_size_h filter"
    h_infl = 4
    k_infl = 1
    # standard persistent kernel
    if "RELOAD_WEIGHTS" in kwargs:
        for c in configs:
            c.kwargs["k_steps"] = triton.cdiv(
                k_infl * hidden_size, c.kwargs["BLOCK_SIZE_K"]
            )

        # RELOAD means weights are reloaded every time-step
        # this reduces shared memory pressues and relies on cache instead
        if not kwargs["RELOAD_WEIGHTS"]:
            # kernel only implements up to 4 weight chunks
            configs = [c for c in configs if c.kwargs["k_steps"] <= 4]

    # fully fused persistent -- ie Wx is part of it!
    elif fully_fused_pers:
        configs = [
            c
            for c in configs
            if (
                c.kwargs["BLOCK_SIZE_K"] == c.kwargs["BLOCK_SIZE_H"]
                and (
                    c.kwargs["BLOCK_SIZE_H"]
                    >= triton.next_power_of_2(max(hidden_size, kwargs["input_size"]))
                )
            )
        ]

        assert len(configs) > 0
    else:
        # that's the bwd pass
        h_infl = 1
        k_infl = 4
        pass

    # filter batch-size related things
    configs_new = []
    for c in configs:
        if not (
            (c.kwargs["BLOCK_SIZE_B"] < 2 * batch_size)
            or (batch_size < 8 and c.kwargs["BLOCK_SIZE_B"] < 32)
        ):
            continue

        rmem = naive_smem_estimation(
            n_bytes=2 if kwargs["dtype"].endswith("16") else 4,
            K=k_infl * c.kwargs["BLOCK_SIZE_K"],
            M=h_infl * c.kwargs["BLOCK_SIZE_H"],
            N=c.kwargs["BLOCK_SIZE_B"],
            num_stages=c.all_kwargs()["num_stages"],
        )

        if rmem > 3:
            continue

        tile_size = h_infl * c.kwargs["BLOCK_SIZE_H"] * c.kwargs["BLOCK_SIZE_B"]

        if tile_size <= 256:
            target_num_warps = [1]
            target_num_stages = [4, 5, 6]
        elif tile_size <= 1024:
            target_num_warps = [2]
            target_num_stages = [4, 5]
        elif tile_size <= 2048:
            target_num_warps = [4]
            target_num_stages = [2, 3, 4]
        elif tile_size <= 4096:
            target_num_warps = [4, 8]
            target_num_stages = [3, 4]
        elif tile_size <= 32768:
            target_num_warps = [8]
            target_num_stages = [3]
        else:
            continue

        if c.all_kwargs()["num_warps"] not in target_num_warps:
            continue

        if c.all_kwargs()["num_stages"] not in target_num_stages:
            continue

        update_params = compute_batch_layout(
            hidden_size,
            c.kwargs["BLOCK_SIZE_H"],
            kwargs["batch_size"],
            c.kwargs["BLOCK_SIZE_B"],
        )
        c.kwargs["batch_chunks"] = update_params["batch_chunks"]
        c.kwargs["num_pid_b"] = update_params["num_pid_b"]
        configs_new += [c]

    def _mm_intensity(c):
        M = h_infl * c.kwargs["BLOCK_SIZE_H"]
        N = c.kwargs["BLOCK_SIZE_B"]
        K = (
            c.kwargs.get(
                "k_steps", triton.cdiv(k_infl * hidden_size, c.kwargs["BLOCK_SIZE_K"])
            )
            * c.kwargs["BLOCK_SIZE_K"]
        )
        return M * N * K / (M * N + K * M + K * N)

    max_intensity = max(_mm_intensity(c) for c in configs_new)
    configs_new = [c for c in configs_new if _mm_intensity(c) >= max_intensity / 4]
    return configs_new


def get_persistent_autotune_configs():
    """
    Define a huge grid of possibly interesting configs.

    prune_persistent_configs then selects the reasonable ones
    and modifies `num_pid_b` and `batch_chunks` accordingly
    """
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_K": k,
                "BLOCK_SIZE_H": h,
                "BLOCK_SIZE_B": b,
                "num_pid_b": 1,
                "batch_chunks": 1,
            },
            num_warps=w,
            num_stages=s,
        )
        for k in [32, 64]
        for w in [1, 2, 4, 8]
        for s in list(range(1, 9))
        for h in [8, 16, 32, 64, 128, 256]
        for b in [1, 8, 16, 32, 64, 128]
    ]
    return configs


def get_persistent_fwd_v2_autotune_configs():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_H": h,
                "BLOCK_SIZE_B": b,
                "num_pid_h": nh,
                "num_pid_b": nb,
            },
            num_warps=w,
            num_stages=s,
        )
        for w in [1, 2, 4, 8]
        for s in list(range(1, 9))
        for h in [8, 16, 32, 64, 128]
        for b in [8, 16, 32, 64, 128]
        for nh, nb in [
            (11, 2),
            # (8, 2)
        ]
    ]
    return configs
