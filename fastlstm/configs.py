import torch
import triton


SM_count = torch.cuda.get_device_properties("cuda").multi_processor_count


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
        for s in [4, 6, 8]
        for w in [4, 8]
        for k in [32, 64]
        for h in [32, 64]
        for b in [32, 64]
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


def prune_persistent_configs(configs, named_args, **kwargs):
    """
    - select from a big grid of configs the relevant ones
    - dynamically compute and overwrite some attributes

    ToDo:
    - introduce logic to keep the number of tested configs small
    """

    hidden_size = kwargs["hidden_size"]
    configs = [
        c
        for c in configs
        if (
            (triton.cdiv(hidden_size, c.kwargs["BLOCK_SIZE_H"]) <= SM_count)
            and (c.kwargs["BLOCK_SIZE_H"] < 2 * hidden_size)
            and c.kwargs["BLOCK_SIZE_H"] in [32, 64, 128]
        )
    ]

    if "RELOAD_WEIGHTS" in kwargs:
        for c in configs:
            k_steps = triton.cdiv(hidden_size, c.kwargs["BLOCK_SIZE_K"])
            c.kwargs["k_steps"] = k_steps
        if not kwargs["RELOAD_WEIGHTS"]:
            configs = [c for c in configs if c.kwargs["k_steps"] < 4]

    if "W_x_ptr" in kwargs:
        configs = [
            c
            for c in configs
            if (
                (c.kwargs["BLOCK_SIZE_K"] == triton.next_power_of_2(hidden_size))
                and (c.kwargs["BLOCK_SIZE_H"] == triton.next_power_of_2(hidden_size))
            )
        ]

        assert len(configs) > 0

    configs = [c for c in configs if (c.kwargs["BLOCK_SIZE_B"] in [32])]

    configs_new = []
    for c in configs:
        update_params = compute_batch_layout(
            hidden_size,
            c.kwargs["BLOCK_SIZE_H"],
            kwargs["batch_size"],
            c.kwargs["BLOCK_SIZE_B"],
        )
        c.kwargs["batch_chunks"] = update_params["batch_chunks"]
        c.kwargs["num_pid_b"] = update_params["num_pid_b"]
        configs_new += [c]
    assert len(configs) < 25, (
        f"Maybe worth trying to prune configs further! {len(configs)}"
    )
    return configs_new


def get_persistent_autotune_configs():
    """
    Define a huge grid of possibly interesting configs.

    prune_persistent_bwd_configs then selects the reasonable ones
    and modified `num_pid_b` and `batch_chunks` accordingly
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
        for w in [4, 8]
        for s in [4, 6]
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
        for w in [2, 4]
        for s in [2, 4, 6]
        for h in [8, 16, 32, 64, 128]
        for b in [8, 16, 32, 64, 128]
        for nh, nb in [
            (11, 2),
            # (8, 2)
        ]
    ]
    return configs
