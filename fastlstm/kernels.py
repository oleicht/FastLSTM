import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

from fastlstm import configs


#######################################################################################
#################################### fwd kernels ######################################
#######################################################################################
@triton.autotune(
    configs=configs.get_graph_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
)
@triton.jit
def one_step_fwd(
    ifgo_ptr,
    cell_ptr,
    h_ptr,
    W_h_ptr,
    offset_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
    dtype: tl.constexpr,
):
    if dtype == "fp16":
        target = tl.float16

    elif dtype == "bf16":
        target = tl.bfloat16

    num_pid_b = tl.cdiv(batch_size, BLOCK_SIZE_B)
    num_pid_h = tl.cdiv(hidden_size, BLOCK_SIZE_H)

    pid = tl.program_id(axis=0)
    num_pid_in_group = GROUP_SIZE_B * num_pid_h
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_SIZE_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_SIZE_B)
    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_h = (pid % num_pid_in_group) // group_size_b

    tl.assume(pid_b >= 0)
    tl.assume(pid_h >= 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)

    tl.assume(BLOCK_SIZE_B > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(BLOCK_SIZE_H > 0)
    tl.assume(GROUP_SIZE_B > 0)
    offset = tl.load(offset_ptr)

    offs_ab = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_bh = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    ifgo_ptrs = ifgo_ptr + (offs_ab[:, None] * 4 * hidden_size + offs_bh[None, :] * 1)
    cell_ptrs = cell_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)
    h_write_ptrs = h_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)

    mask = (offs_ab[:, None] < batch_size) & (offs_bh[None, :] < hidden_size)

    cell_ptrs += offset * batch_size * hidden_size
    c = tl.load(cell_ptrs, mask=mask, other=0.0)
    cell_ptrs += batch_size * hidden_size

    h_mm_ptrs = h_ptr + (offs_ab[:, None] * hidden_size + offs_k[None, :] * 1)
    W_h_ptrs = W_h_ptr + (offs_k[:, None] * 1 + offs_bh[None, :] * hidden_size)

    ifgo_ptrs += offset * batch_size * 4 * hidden_size
    i = tl.load(ifgo_ptrs + 0 * hidden_size, mask=mask, other=0.0)
    f = tl.load(ifgo_ptrs + 1 * hidden_size, mask=mask, other=0.0)
    g = tl.load(ifgo_ptrs + 2 * hidden_size, mask=mask, other=0.0)
    o = tl.load(ifgo_ptrs + 3 * hidden_size, mask=mask, other=0.0)

    if dtype != "fp32":
        i = i.cast(tl.float32)
        f = f.cast(tl.float32)
        g = g.cast(tl.float32)
        o = o.cast(tl.float32)

    h_write_ptrs += (offset + 1) * batch_size * hidden_size
    h_mm_ptrs += offset * batch_size * hidden_size

    for k in range(tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        h_0 = tl.load(
            h_mm_ptrs,
            mask=offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K,
            other=0.0,
        )

        w_mask = offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K
        W_i = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
        W_f = tl.load(W_h_ptrs + hidden_size * hidden_size, mask=w_mask, other=0.0)
        W_g = tl.load(W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0)
        W_o = tl.load(W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0)

        i = tl.dot(h_0, W_i, i)
        f = tl.dot(h_0, W_f, f)
        g = tl.dot(h_0, W_g, g)
        o = tl.dot(h_0, W_o, o)

        h_mm_ptrs += BLOCK_SIZE_K
        W_h_ptrs += BLOCK_SIZE_K

    # reset accumulator pointers for next iteration

    if dtype != "fp32":
        tl.store(ifgo_ptrs + 0 * hidden_size, i.cast(target), mask=mask)
        tl.store(ifgo_ptrs + 1 * hidden_size, f.cast(target), mask=mask)
        tl.store(ifgo_ptrs + 2 * hidden_size, g.cast(target), mask=mask)
        tl.store(ifgo_ptrs + 3 * hidden_size, o.cast(target), mask=mask)

    else:
        tl.store(ifgo_ptrs + 0 * hidden_size, i, mask=mask)
        tl.store(ifgo_ptrs + 1 * hidden_size, f, mask=mask)
        tl.store(ifgo_ptrs + 2 * hidden_size, g, mask=mask)
        tl.store(ifgo_ptrs + 3 * hidden_size, o, mask=mask)

    # step 2: compute c and h
    # update the pointers first, so the write goes to i+1 element
    c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
    h = tl.sigmoid(o) * libdevice.tanh(c)

    if dtype != "fp32":
        tl.store(cell_ptrs, c.cast(target), mask=mask)
        tl.store(h_write_ptrs, h.cast(target), mask=mask)

    else:
        tl.store(cell_ptrs, c, mask=mask)
        tl.store(h_write_ptrs, h, mask=mask)


@triton.autotune(
    configs=configs.get_persistent_fwd_v2_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
)
@triton.jit(do_not_specialize=["seq_len"])
def persistent_fwd_kernel_v2(
    ifgo_ptr,
    cell_ptr,
    h_ptr,
    W_h_ptr,
    seq_len,  # : tl.constexpr,
    global_sync_ptr,
    num_pid_h: tl.constexpr,
    num_pid_b: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    dtype: tl.constexpr,
):
    if dtype == "fp16":
        target = tl.float16

    elif dtype == "bf16":
        target = tl.bfloat16

    total_num_pid_b = tl.cdiv(batch_size, BLOCK_SIZE_B)
    total_num_pid_h = tl.cdiv(hidden_size, BLOCK_SIZE_H)

    # kill configs where sms would be idle

    # old
    pid = tl.program_id(axis=0)
    pid_b = pid // num_pid_h
    pid_h = pid % num_pid_h

    batch_hidden_4 = batch_size * 4 * hidden_size
    hidden_4 = 4 * hidden_size

    for pb in range(pid_b, total_num_pid_b, num_pid_b):
        global_sync_ptrl = global_sync_ptr + pb

        for sid in range(seq_len):
            if sid > 0 and num_pid_h > 1:
                while tl.atomic_add(global_sync_ptrl, 0, sem="acquire") < num_pid_h:
                    pass

            # loop over hidden patches
            for ph in range(pid_h, total_num_pid_h, num_pid_h):
                i_ptrs = tl.make_block_ptr(
                    ifgo_ptr + sid * batch_hidden_4,
                    (batch_size, 4, hidden_size),
                    (hidden_4, hidden_size, 1),
                    (pb * BLOCK_SIZE_B, 0, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                    (0, 1, 2),
                )
                i = tl.load(i_ptrs, boundary_check=(0, 2)).reshape(
                    BLOCK_SIZE_B, BLOCK_SIZE_H
                )

                f_ptrs = tl.make_block_ptr(
                    ifgo_ptr + sid * batch_hidden_4,
                    (batch_size, 4, hidden_size),
                    (hidden_4, hidden_size, 1),
                    (pb * BLOCK_SIZE_B, 1, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                    (0, 1, 2),
                )
                f = tl.load(f_ptrs, boundary_check=(0, 2)).reshape(
                    BLOCK_SIZE_B, BLOCK_SIZE_H
                )

                g_ptrs = tl.make_block_ptr(
                    ifgo_ptr + sid * batch_hidden_4,
                    (batch_size, 4, hidden_size),
                    (hidden_4, hidden_size, 1),
                    (pb * BLOCK_SIZE_B, 2, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                    (0, 1, 2),
                )

                g = tl.load(g_ptrs, boundary_check=(0, 2)).reshape(
                    BLOCK_SIZE_B, BLOCK_SIZE_H
                )

                o_ptrs = tl.make_block_ptr(
                    ifgo_ptr + sid * batch_hidden_4,
                    (batch_size, 4, hidden_size),
                    (hidden_4, hidden_size, 1),
                    (pb * BLOCK_SIZE_B, 3, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                    (0, 1, 2),
                )

                o = tl.load(o_ptrs, boundary_check=(0, 2)).reshape(
                    BLOCK_SIZE_B, BLOCK_SIZE_H
                )
                # note! it's W_h.T not W_h!!
                # (batch_size, hidden_size) x (hidden_size, 4 x hidden_size)
                if dtype != "fp32":
                    i = i.cast(tl.float32)
                    f = f.cast(tl.float32)
                    g = g.cast(tl.float32)
                    o = o.cast(tl.float32)

                for k in range(tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                    h_mm_ptrs = tl.make_block_ptr(
                        h_ptr + sid * batch_size * hidden_size,
                        (batch_size, hidden_size),
                        (hidden_size, 1),
                        (pb * BLOCK_SIZE_B, k * BLOCK_SIZE_K),
                        (BLOCK_SIZE_B, BLOCK_SIZE_K),
                        (0, 1),
                    )

                    h_0 = tl.load(h_mm_ptrs, boundary_check=(0, 1))

                    W_i_ptr = tl.make_block_ptr(
                        W_h_ptr,
                        (hidden_size, hidden_size, 4),
                        (1, hidden_size, hidden_size * hidden_size),
                        (k * BLOCK_SIZE_K, ph * BLOCK_SIZE_H, 0),
                        (BLOCK_SIZE_K, BLOCK_SIZE_H, 1),
                        (2, 1, 0),
                    )

                    W_i = tl.load(W_i_ptr, boundary_check=(1, 2)).reshape(
                        BLOCK_SIZE_K, BLOCK_SIZE_H
                    )

                    W_f_ptr = tl.make_block_ptr(
                        W_h_ptr,
                        (hidden_size, hidden_size, 4),
                        (1, hidden_size, hidden_size * hidden_size),
                        (k * BLOCK_SIZE_K, ph * BLOCK_SIZE_H, 1),
                        (BLOCK_SIZE_K, BLOCK_SIZE_H, 1),
                        (2, 1, 0),
                    )
                    W_f = tl.load(W_f_ptr, boundary_check=(1, 2)).reshape(
                        BLOCK_SIZE_K, BLOCK_SIZE_H
                    )

                    W_g_ptr = tl.make_block_ptr(
                        W_h_ptr,
                        (hidden_size, hidden_size, 4),
                        (1, hidden_size, hidden_size * hidden_size),
                        (k * BLOCK_SIZE_K, ph * BLOCK_SIZE_H, 2),
                        (BLOCK_SIZE_K, BLOCK_SIZE_H, 1),
                        (2, 1, 0),
                    )
                    W_g = tl.load(W_g_ptr, boundary_check=(1, 2)).reshape(
                        BLOCK_SIZE_K, BLOCK_SIZE_H
                    )

                    W_o_ptr = tl.make_block_ptr(
                        W_h_ptr,
                        (hidden_size, hidden_size, 4),
                        (1, hidden_size, hidden_size * hidden_size),
                        (k * BLOCK_SIZE_K, ph * BLOCK_SIZE_H, 3),
                        (BLOCK_SIZE_K, BLOCK_SIZE_H, 1),
                        (2, 1, 0),
                    )
                    W_o = tl.load(W_o_ptr, boundary_check=(1, 2)).reshape(
                        BLOCK_SIZE_K, BLOCK_SIZE_H
                    )

                    i = tl.dot(h_0, W_i, i)
                    f = tl.dot(h_0, W_f, f)
                    g = tl.dot(h_0, W_g, g)
                    o = tl.dot(h_0, W_o, o)

                # reset accumulator pointers for next iteration
                if dtype != "fp32":
                    tl.store(
                        i_ptrs,
                        i.cast(target).reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        f_ptrs,
                        f.cast(target).reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        g_ptrs,
                        g.cast(target).reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        o_ptrs,
                        o.cast(target).reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                else:
                    tl.store(
                        i_ptrs,
                        i.reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        f_ptrs,
                        f.reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        g_ptrs,
                        g.reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )
                    tl.store(
                        o_ptrs,
                        o.reshape(BLOCK_SIZE_B, 1, BLOCK_SIZE_H),
                        boundary_check=(0, 2),
                    )

                # step 2: compute c and h
                # update the pointers first, so the write goes to i+1 element
                cell_ptrs = tl.make_block_ptr(
                    cell_ptr + sid * batch_size * hidden_size,
                    (batch_size, hidden_size),
                    (hidden_size, 1),
                    (pb * BLOCK_SIZE_B, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, BLOCK_SIZE_H),
                    (0, 1),
                )

                c = tl.load(cell_ptrs, boundary_check=(0, 1))

                h_write_ptrs = tl.make_block_ptr(
                    h_ptr + (sid + 1) * batch_size * hidden_size,
                    (batch_size, hidden_size),
                    (hidden_size, 1),
                    (pb * BLOCK_SIZE_B, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, BLOCK_SIZE_H),
                    (0, 1),
                )

                cell_ptrs = tl.make_block_ptr(
                    cell_ptr + (sid + 1) * batch_size * hidden_size,
                    (batch_size, hidden_size),
                    (hidden_size, 1),
                    (pb * BLOCK_SIZE_B, ph * BLOCK_SIZE_H),
                    (BLOCK_SIZE_B, BLOCK_SIZE_H),
                    (0, 1),
                )

                c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
                h = tl.sigmoid(o) * libdevice.tanh(c)

                if dtype != "fp32":
                    tl.store(cell_ptrs, c.cast(target), boundary_check=(0, 1))
                    tl.store(h_write_ptrs, h.cast(target), boundary_check=(0, 1))
                else:
                    tl.store(cell_ptrs, c, boundary_check=(0, 1))
                    tl.store(h_write_ptrs, h, boundary_check=(0, 1))

            # synchronize within block -> h vector is updated
            tl.debug_barrier(sem="release")
            # update global counter
            if num_pid_h > 1:
                global_sync_ptrl += total_num_pid_b
                tl.atomic_add(global_sync_ptrl, 1, sem="release")


@triton.autotune(
    configs=configs.get_persistent_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
    prune_configs_by={"early_config_prune": configs.prune_persistent_configs},
)
@triton.jit(do_not_specialize=["seq_len"])
def persistent_fwd_kernel(
    ifgo_ptr,
    cell_ptr,
    h_ptr,
    W_h_ptr,
    seq_len,  # : tl.constexpr,
    global_sync_ptr,
    batch_chunks: tl.constexpr,
    num_pid_b: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    dtype: tl.constexpr,
    k_steps: tl.constexpr = 1,
    RELOAD_WEIGHTS: tl.constexpr = False,
):
    total_num_pid_b = tl.cdiv(batch_size, BLOCK_SIZE_B)
    num_pid_h = tl.cdiv(hidden_size, BLOCK_SIZE_H)

    # old
    pid = tl.program_id(axis=0)

    pid_b = pid // num_pid_h
    pid_h = pid % num_pid_h

    tl.assume(pid_b >= 0)
    tl.assume(pid_h >= 0)
    tl.assume(seq_len > 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)

    tl.assume(BLOCK_SIZE_B > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(BLOCK_SIZE_H > 0)

    if dtype == "fp32":
        target = tl.float32
    elif dtype == "fp16":
        target = tl.float16
    elif dtype == "bf16":
        target = tl.bfloat16

    # k_steps = tl.cdiv(hidden_size, BLOCK_SIZE_K)
    offs_bh = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for _ in range(batch_chunks):
        global_sync_ptrl = global_sync_ptr + pid_b

        offs_ab = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        ifgo_ptrs = ifgo_ptr + (
            offs_ab[:, None] * 4 * hidden_size + offs_bh[None, :] * 1
        )
        cell_ptrs = cell_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)
        h_write_ptrs = h_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)

        mask = (offs_ab[:, None] < batch_size) & (offs_bh[None, :] < hidden_size)

        c = tl.load(cell_ptrs, mask=mask, other=0.0)
        if dtype != "fp32":
            c = c.cast(tl.float32)

        h_mm_ptrs = h_ptr + (offs_ab[:, None] * hidden_size + offs_k[None, :] * 1)

        W_h_ptrs = W_h_ptr + (offs_k[:, None] * 1 + offs_bh[None, :] * hidden_size)

        if not RELOAD_WEIGHTS:
            tl.assume(k_steps <= 4)
            w_mask = offs_k[:, None] < hidden_size
            W_i0 = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
            W_f0 = tl.load(
                W_h_ptrs + 1 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_g0 = tl.load(
                W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_o0 = tl.load(
                W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_h_ptrs += BLOCK_SIZE_K

        if not RELOAD_WEIGHTS and k_steps > 1:
            w_mask = offs_k[:, None] < hidden_size - BLOCK_SIZE_K
            W_i1 = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
            W_f1 = tl.load(
                W_h_ptrs + 1 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_g1 = tl.load(
                W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_o1 = tl.load(
                W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_h_ptrs += BLOCK_SIZE_K
        if not RELOAD_WEIGHTS and k_steps > 2:
            w_mask = offs_k[:, None] < hidden_size - 2 * BLOCK_SIZE_K
            W_i2 = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
            W_f2 = tl.load(
                W_h_ptrs + 1 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_g2 = tl.load(
                W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_o2 = tl.load(
                W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_h_ptrs += BLOCK_SIZE_K
        if not RELOAD_WEIGHTS and k_steps > 3:
            w_mask = offs_k[:, None] < hidden_size - 3 * BLOCK_SIZE_K
            W_i3 = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
            W_f3 = tl.load(
                W_h_ptrs + 1 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_g3 = tl.load(
                W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_o3 = tl.load(
                W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0
            )
            W_h_ptrs += BLOCK_SIZE_K

        for ss in range(seq_len):
            i = tl.load(ifgo_ptrs + 0 * hidden_size, mask=mask, other=0.0)
            f = tl.load(ifgo_ptrs + 1 * hidden_size, mask=mask, other=0.0)
            g = tl.load(ifgo_ptrs + 2 * hidden_size, mask=mask, other=0.0)
            o = tl.load(ifgo_ptrs + 3 * hidden_size, mask=mask, other=0.0)
            if dtype != "fp32":
                i = i.cast(tl.float32)
                f = f.cast(tl.float32)
                g = g.cast(tl.float32)
                o = o.cast(tl.float32)

            # note! it's W_h.T not W_h!!
            # (batch_size, hidden_size) x (hidden_size, 4 x hidden_size)

            if ss > 0 and num_pid_h > 1:
                while tl.atomic_add(global_sync_ptrl, 0, sem="acquire") < num_pid_h:
                    pass

            if RELOAD_WEIGHTS:
                for k in range(k_steps):
                    h_0 = tl.load(
                        h_mm_ptrs,
                        mask=offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    h_mm_ptrs += BLOCK_SIZE_K
                    w_mask = offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K
                    W_i = tl.load(W_h_ptrs, mask=w_mask, other=0.0)

                    W_f = tl.load(
                        W_h_ptrs + hidden_size * hidden_size, mask=w_mask, other=0.0
                    )
                    W_g = tl.load(
                        W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0
                    )
                    W_o = tl.load(
                        W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0
                    )
                    W_h_ptrs += BLOCK_SIZE_K

                    i = tl.dot(h_0, W_i, i)
                    f = tl.dot(h_0, W_f, f)
                    g = tl.dot(h_0, W_g, g)
                    o = tl.dot(h_0, W_o, o)
                W_h_ptrs = W_h_ptrs - k_steps * BLOCK_SIZE_K

            else:
                # compiler wants us to manually unroll the loop
                h_0 = tl.load(
                    h_mm_ptrs,
                    mask=offs_k[None, :] < hidden_size,
                    other=0.0,
                )
                h_mm_ptrs += BLOCK_SIZE_K
                i = tl.dot(h_0, W_i0, i)
                f = tl.dot(h_0, W_f0, f)
                g = tl.dot(h_0, W_g0, g)
                o = tl.dot(h_0, W_o0, o)

                if k_steps > 1:
                    h_0 = tl.load(
                        h_mm_ptrs,
                        mask=offs_k[None, :] < hidden_size - BLOCK_SIZE_K,
                        other=0.0,
                    )
                    h_mm_ptrs += BLOCK_SIZE_K
                    i = tl.dot(h_0, W_i1, i)
                    f = tl.dot(h_0, W_f1, f)
                    g = tl.dot(h_0, W_g1, g)
                    o = tl.dot(h_0, W_o1, o)
                if k_steps > 2:
                    h_0 = tl.load(
                        h_mm_ptrs,
                        mask=offs_k[None, :] < hidden_size - 2 * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    h_mm_ptrs += BLOCK_SIZE_K
                    i = tl.dot(h_0, W_i2, i)
                    f = tl.dot(h_0, W_f2, f)
                    g = tl.dot(h_0, W_g2, g)
                    o = tl.dot(h_0, W_o2, o)
                if k_steps > 3:
                    h_0 = tl.load(
                        h_mm_ptrs,
                        mask=offs_k[None, :] < hidden_size - 3 * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    h_mm_ptrs += BLOCK_SIZE_K
                    i = tl.dot(h_0, W_i3, i)
                    f = tl.dot(h_0, W_f3, f)
                    g = tl.dot(h_0, W_g3, g)
                    o = tl.dot(h_0, W_o3, o)

            # reset accumulator pointers for next iteration
            h_mm_ptrs = h_mm_ptrs - k_steps * BLOCK_SIZE_K + batch_size * hidden_size

            if dtype != "fp32":
                tl.store(ifgo_ptrs + 0 * hidden_size, i.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 1 * hidden_size, f.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 2 * hidden_size, g.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 3 * hidden_size, o.cast(target), mask=mask)
            else:
                tl.store(ifgo_ptrs + 0 * hidden_size, i, mask=mask)
                tl.store(ifgo_ptrs + 1 * hidden_size, f, mask=mask)
                tl.store(ifgo_ptrs + 2 * hidden_size, g, mask=mask)
                tl.store(ifgo_ptrs + 3 * hidden_size, o, mask=mask)

            ifgo_ptrs += batch_size * 4 * hidden_size

            # step 2: compute c and h
            # update the pointers first, so the write goes to i+1 element
            cell_ptrs += batch_size * hidden_size
            h_write_ptrs += batch_size * hidden_size

            c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
            h = tl.sigmoid(o) * libdevice.tanh(c)
            if dtype != "fp32":
                tl.store(cell_ptrs, c.cast(target), mask=mask)
                tl.store(h_write_ptrs, h.cast(target), mask=mask)
            else:
                tl.store(cell_ptrs, c, mask=mask)
                tl.store(h_write_ptrs, h, mask=mask)

            # synchronize within block -> h vector is updated
            # update global counter
            global_sync_ptrl += total_num_pid_b
            tl.atomic_add(global_sync_ptrl, 1, sem="release")

        pid_b += num_pid_b


@triton.autotune(
    configs=configs.get_persistent_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
    prune_configs_by={"early_config_prune": configs.prune_persistent_configs},
)
@triton.jit(do_not_specialize=["seq_len"])
def fully_fused_persistent_fwd_kernel(
    ifgo_ptr,
    x_ptr,
    cell_ptr,
    h_ptr,
    W_h_ptr,
    W_x_ptr,
    b_h_ptr,
    b_x_ptr,
    seq_len,  # : tl.constexpr,
    global_sync_ptr,
    batch_chunks: tl.constexpr,
    num_pid_b: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    input_size: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    dtype: tl.constexpr,
    fully_fused: tl.constexpr = True,
):
    tl.device_assert(tl.cdiv(hidden_size, BLOCK_SIZE_H) == 1)
    tl.device_assert(tl.cdiv(max(hidden_size, input_size), BLOCK_SIZE_K) == 1)
    tl.device_assert(tl.cdiv(batch_size, BLOCK_SIZE_B) <= num_pid_b * batch_chunks)

    pid_b = tl.program_id(axis=0)

    tl.assume(pid_b >= 0)
    tl.assume(seq_len > 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)

    tl.assume(BLOCK_SIZE_B > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(BLOCK_SIZE_H > 0)

    if dtype == "fp32":
        target = tl.float32
    elif dtype == "fp16":
        target = tl.float16
    elif dtype == "bf16":
        target = tl.bfloat16

    # load entire RHS in at once
    offs_bh = tl.arange(0, BLOCK_SIZE_H)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    W_x_ptrs = W_x_ptr + (offs_k[:, None] * 1 + offs_bh[None, :] * input_size)
    w_mask = (offs_k[:, None] < input_size) & (offs_bh[None, :] < hidden_size)
    Wx_i = tl.load(W_x_ptrs, mask=w_mask, other=0.0)
    Wx_f = tl.load(W_x_ptrs + input_size * hidden_size, mask=w_mask, other=0.0)
    Wx_g = tl.load(W_x_ptrs + 2 * input_size * hidden_size, mask=w_mask, other=0.0)
    Wx_o = tl.load(W_x_ptrs + 3 * input_size * hidden_size, mask=w_mask, other=0.0)

    W_h_ptrs = W_h_ptr + (offs_k[:, None] * 1 + offs_bh[None, :] * hidden_size)
    w_mask = (offs_k[:, None] < hidden_size) & (offs_bh[None, :] < hidden_size)
    Wh_i = tl.load(W_h_ptrs, mask=w_mask, other=0.0)
    Wh_f = tl.load(W_h_ptrs + hidden_size * hidden_size, mask=w_mask, other=0.0)
    Wh_g = tl.load(W_h_ptrs + 2 * hidden_size * hidden_size, mask=w_mask, other=0.0)
    Wh_o = tl.load(W_h_ptrs + 3 * hidden_size * hidden_size, mask=w_mask, other=0.0)

    b_mask = offs_bh < hidden_size
    b_i = tl.load(b_x_ptr + offs_bh, mask=b_mask) + tl.load(
        b_h_ptr + offs_bh, mask=b_mask
    )
    b_f = tl.load(b_x_ptr + hidden_size + offs_bh, mask=b_mask) + tl.load(
        b_h_ptr + hidden_size + offs_bh, mask=b_mask
    )
    b_g = tl.load(b_x_ptr + offs_bh + 2 * hidden_size, mask=b_mask) + tl.load(
        b_h_ptr + 2 * hidden_size + offs_bh, mask=b_mask
    )
    b_o = tl.load(b_x_ptr + offs_bh + 3 * hidden_size, mask=b_mask) + tl.load(
        b_h_ptr + 3 * hidden_size + offs_bh, mask=b_mask
    )

    for _ in range(batch_chunks):
        offs_ab = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        ifgo_ptrs = ifgo_ptr + (
            offs_ab[:, None] * 4 * hidden_size + offs_bh[None, :] * 1
        )

        mask = (offs_ab[:, None] < batch_size) & (offs_bh[None, :] < hidden_size)

        cell_ptrs = cell_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)
        c = tl.load(cell_ptrs, mask=mask, other=0.0)
        if dtype != "fp32":
            c = c.cast(tl.float32)

        h_ptrs = h_ptr + (offs_ab[:, None] * hidden_size + offs_bh[None, :] * 1)
        h = tl.load(
            h_ptrs,
            mask=mask,
            other=0.0,
        )

        x_mm_ptrs = x_ptr + (offs_ab[:, None] * input_size + offs_k[None, :] * 1)

        for _ in range(seq_len):
            i = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32) + b_i
            f = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32) + b_f
            g = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32) + b_g
            o = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32) + b_o

            i = tl.dot(h, Wh_i, i)
            f = tl.dot(h, Wh_f, f)
            g = tl.dot(h, Wh_g, g)
            o = tl.dot(h, Wh_o, o)

            x = tl.load(
                x_mm_ptrs,
                mask=(offs_ab[:, None] < batch_size) & (offs_k[None, :] < input_size),
                other=0.0,
            )
            i = tl.dot(x, Wx_i, i)
            f = tl.dot(x, Wx_f, f)
            g = tl.dot(x, Wx_g, g)
            o = tl.dot(x, Wx_o, o)

            if dtype != "fp32":
                tl.store(ifgo_ptrs + 0 * hidden_size, i.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 1 * hidden_size, f.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 2 * hidden_size, g.cast(target), mask=mask)
                tl.store(ifgo_ptrs + 3 * hidden_size, o.cast(target), mask=mask)
            else:
                tl.store(ifgo_ptrs + 0 * hidden_size, i, mask=mask)
                tl.store(ifgo_ptrs + 1 * hidden_size, f, mask=mask)
                tl.store(ifgo_ptrs + 2 * hidden_size, g, mask=mask)
                tl.store(ifgo_ptrs + 3 * hidden_size, o, mask=mask)

            ifgo_ptrs += batch_size * 4 * hidden_size
            x_mm_ptrs += batch_size * input_size
            cell_ptrs += batch_size * hidden_size
            h_ptrs += batch_size * hidden_size

            c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
            h = tl.sigmoid(o) * libdevice.tanh(c)

            if dtype != "fp32":
                h = h.cast(target)
                tl.store(cell_ptrs, c.cast(target), mask=mask)
                tl.store(h_ptrs, h, mask=mask)
            else:
                tl.store(cell_ptrs, c, mask=mask)
                tl.store(h_ptrs, h, mask=mask)

        pid_b += num_pid_b


@triton.jit
def triton_lstm_cell_fwd_phase2(
    ifgo_ptr,
    c0_ptr,
    h_out_ptr,
    c_out_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    # 2d tiles to make index work easier
    # x-dim ->
    pid_h = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    tl.assume(pid_h >= 0)
    tl.assume(pid_b >= 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)

    hidden_start = pid_h * BLOCK_SIZE_H
    batch_start = pid_b * BLOCK_SIZE_B
    offsets_h = (hidden_start + tl.arange(0, BLOCK_SIZE_H))[None]
    offsets_b = (batch_start + tl.arange(0, BLOCK_SIZE_B))[:, None]

    ifgo_indices = tl.ravel(offsets_h + 4 * hidden_size * offsets_b)
    c0_indices = tl.ravel(offsets_h + hidden_size * offsets_b)
    mask = tl.ravel((offsets_h < hidden_size) & (offsets_b < batch_size))
    # 4 * hidden_dim *
    i = tl.load(ifgo_ptr + ifgo_indices, mask=mask)
    f = tl.load(ifgo_ptr + ifgo_indices + hidden_size, mask=mask)
    g = tl.load(ifgo_ptr + ifgo_indices + 2 * hidden_size, mask=mask)
    o = tl.load(ifgo_ptr + ifgo_indices + 3 * hidden_size, mask=mask)
    c0 = tl.load(c0_ptr + c0_indices)

    c1 = tl.sigmoid(f) * c0 + tl.sigmoid(i) * libdevice.tanh(g)
    h1 = tl.sigmoid(o) * libdevice.tanh(c1)

    tl.store(c_out_ptr + c0_indices, c1, mask=mask)
    tl.store(h_out_ptr + c0_indices, h1, mask=mask)


#######################################################################################
####################### bwd kernels - some are experimental/dead ends #################
#######################################################################################
@triton.jit
def lstm_ifgo_bwd(
    d_out_ptr,
    d_h_ptr,
    d_c_ptr,
    ifgo_ptr,
    cell_ptr,
    d_ifgo_ptr,
    offset_ptr,
    batch_size,
    hidden_size,
    d_ifgo_stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    dtype: tl.constexpr,
):
    if dtype == "fp16":
        target = tl.float16
    elif dtype == "bf16":
        target = tl.bfloat16

    # shape: (batch, channel, ifgo)
    pid_h = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    seq_offset = tl.load(offset_ptr)

    d_ifgo_ptr += seq_offset * d_ifgo_stride

    offsets_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[None]
    offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None]

    ifgo_indices = offsets_h + 4 * hidden_size * offsets_b
    c0_indices = offsets_h + hidden_size * offsets_b

    mask = (offsets_h < hidden_size) & (offsets_b < batch_size)

    ifgo_ptr = ifgo_ptr + seq_offset * 4 * hidden_size * batch_size

    i = tl.load(ifgo_ptr + ifgo_indices, mask=mask)
    f = tl.load(ifgo_ptr + ifgo_indices + hidden_size, mask=mask)
    g = tl.load(ifgo_ptr + ifgo_indices + 2 * hidden_size, mask=mask)
    o = tl.load(ifgo_ptr + ifgo_indices + 3 * hidden_size, mask=mask)

    c0_ptr = cell_ptr + seq_offset * hidden_size * batch_size
    c1_ptr = cell_ptr + (seq_offset + 1) * hidden_size * batch_size

    c0 = tl.load(c0_ptr + c0_indices, mask=mask)
    c1 = tl.load(c1_ptr + c0_indices, mask=mask)

    dh = tl.load(d_h_ptr + c0_indices, mask=mask) + tl.load(
        d_out_ptr + seq_offset * hidden_size * batch_size + c0_indices, mask=mask
    )
    dc1 = tl.load(d_c_ptr + c0_indices, mask=mask)

    if dtype != "fp32":
        i = i.cast(tl.float32)
        f = f.cast(tl.float32)
        g = g.cast(tl.float32)
        o = o.cast(tl.float32)
        c0 = c0.cast(tl.float32)
        c1 = c1.cast(tl.float32)
        dc1 = dc1.cast(tl.float32)

    dc1 += dh * tl.sigmoid(o) * (1.0 - libdevice.tanh(c1) * libdevice.tanh(c1))

    d_o = dh * libdevice.tanh(c1) * tl.sigmoid(o) * (1 - tl.sigmoid(o))

    # step 2: c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
    d_c0 = dc1 * tl.sigmoid(f)

    d_f = dc1 * c0 * tl.sigmoid(f) * (1 - tl.sigmoid(f))
    d_i = dc1 * libdevice.tanh(g) * tl.sigmoid(i) * (1.0 - tl.sigmoid(i))
    d_g = dc1 * tl.sigmoid(i) * (1.0 - libdevice.tanh(g) * libdevice.tanh(g))

    if dtype != "fp32":
        d_o = d_o.cast(target)
        d_f = d_f.cast(target)
        d_i = d_i.cast(target)
        d_g = d_g.cast(target)
        d_c0 = d_c0.cast(target)

    tl.store(d_c_ptr + c0_indices, d_c0, mask=mask)
    tl.store(d_ifgo_ptr + ifgo_indices, d_i, mask=mask)
    tl.store(d_ifgo_ptr + ifgo_indices + hidden_size, d_f, mask=mask)
    tl.store(d_ifgo_ptr + ifgo_indices + 2 * hidden_size, d_g, mask=mask)
    tl.store(d_ifgo_ptr + ifgo_indices + 3 * hidden_size, d_o, mask=mask)


@triton.jit
def lstm_Dgrad(
    d_ifgo_ptr,
    Wx_ptr,
    Wh_ptr,
    d_x_ptr,
    dh_n_ptr,
    offset_ptr,
    hidden_size,
    input_size,
    batch_size,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Compute:
    (d_x[s], dh_n) = d_ifgo @ (Wx, Wh)
    Shapes:
    (batch, input|hidden) = (batch, 4 hidden) x (4 hidden, input|hidden)
    """
    K = 4 * hidden_size
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n_x = tl.cdiv(input_size, BLOCK_SIZE_N)
    num_pid_n_h = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_x + num_pid_n_h
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    seq_offset = tl.load(offset_ptr)

    if pid_n < num_pid_n_x:
        b_ptr = Wx_ptr
        c_ptr = d_x_ptr + seq_offset * batch_size * input_size
        N = input_size

    else:
        b_ptr = Wh_ptr
        c_ptr = dh_n_ptr
        N = hidden_size
        pid_n -= num_pid_n_x

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % batch_size
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = d_ifgo_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < batch_size) & (offs_cn[None, :] < N)

    # always accumulate in single precision!
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    # if you want to fuse an activation in, do it here! should be done in fp32!
    c = accumulator

    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def lstm_Wgrad(
    d_ifgo_ptr,
    x_ptr,
    h_ptr,
    dWx_ptr,
    dWh_ptr,
    db_ptr,
    offset_ptr,
    hidden_size,
    input_size,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute:
    (dWx, dWh) += d_ifgo.T @ (x[s], h[s])
    Shapes:
    (4 hidden, input|hidden) = (4 hidden, batch) x (batch, input|hidden)

    Note: auto-tune is a pain with non-idemnpotent kernels!!

    """

    K = batch_size
    M = 4 * hidden_size
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(4 * hidden_size, BLOCK_SIZE_M)
    num_pid_n_x = tl.cdiv(input_size, BLOCK_SIZE_N)
    num_pid_n_h = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_x + num_pid_n_h
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    seq_offset = tl.load(offset_ptr)
    mod = seq_offset % 2

    if pid_n < num_pid_n_x:
        b_ptr = x_ptr
        c_ptr = dWx_ptr
        N = input_size

    else:
        pid_n -= num_pid_n_x
        b_ptr = h_ptr
        c_ptr = dWh_ptr
        N = hidden_size

    b_ptr += seq_offset * batch_size * N
    c_ptr += mod * M * N
    db_ptr += mod * M
    c_read = (1 - 2 * mod) * M * N
    db_read = (1 - 2 * mod) * M

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = d_ifgo_ptr + (offs_am[:, None] + offs_k[None, :] * M)
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # always accumulate in single precision!
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    db = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if pid_n == 0:
            db += tl.sum(a, axis=1)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * M
        b_ptrs += BLOCK_SIZE_K * N
    # if you want to fuse an activation in, do it here! should be done in fp32!
    c = accumulator + tl.load(c_ptrs + c_read, mask=c_mask)

    tl.store(c_ptrs, c, mask=c_mask)
    if pid_n == 0:
        db += tl.load(db_ptr + db_read + offs_am)
        tl.store(db_ptr + offs_cm, db, mask=offs_cm < M)


@triton.jit
def lstm_h_grad(
    d_ifgo_ptr,
    Wh_ptr,
    dh_n_ptr,
    hidden_size,
    batch_size,
    offset_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    dtype: tl.constexpr,
):
    """Compute:
    dh_n = d_ifgo @ Wh
    Shapes:
    (batch, hidden) = (batch, 4 hidden) x (4 hidden, hidden)
    """
    K = 4 * hidden_size
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    seq_offset = tl.load(offset_ptr)
    d_ifgo_ptr += seq_offset * batch_size * 4 * hidden_size
    N = hidden_size
    b_ptr = Wh_ptr
    c_ptr = dh_n_ptr

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % batch_size
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = d_ifgo_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < batch_size) & (offs_cn[None, :] < N)

    # always accumulate in single precision!
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if dtype == "fp16":
            b = b.cast(tl.float16)
        elif dtype == "bf16":
            b = b.cast(tl.bfloat16)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    # if you want to fuse an activation in, do it here! should be done in fp32!
    c = accumulator
    if dtype == "fp16":
        c = c.cast(tl.float16)
    elif dtype == "bf16":
        c = c.cast(tl.bfloat16)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=configs.get_persistent_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
    prune_configs_by={"early_config_prune": configs.prune_persistent_configs},
)
@triton.jit(do_not_specialize=["seq_len"])
def lstm_persistent_seq_bwd(
    d_out_ptr,
    d_h_ptr,
    d_c_ptr,
    ifgo_ptr,
    cell_ptr,
    d_ifgo_ptr,
    Wh_ptr,
    sync_ptr,
    seq_len,
    # num_batch_iter,
    batch_chunks: tl.constexpr,
    num_pid_b: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr,
    LESS_IO: tl.constexpr = False,  # False is the faster vesion
):
    pid = tl.program_id(axis=0)
    num_pid_h = tl.cdiv(hidden_size, BLOCK_SIZE_H)
    total_num_pid_b = tl.cdiv(batch_size, BLOCK_SIZE_B)

    if dtype == "fp16":
        target = tl.float16
    elif dtype == "bf16":
        target = tl.bfloat16
    else:
        target = tl.float32

    pid_h = pid % num_pid_h
    pid_b = pid // num_pid_h

    tl.assume(pid >= 0)
    tl.assume(num_pid_h > 0)
    tl.assume(total_num_pid_b > 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)
    tl.assume(seq_len > 0)
    tl.assume(BLOCK_SIZE_H > 0)
    tl.assume(BLOCK_SIZE_B > 0)
    tl.assume(BLOCK_SIZE_K > 0)

    ifgo_ptr += (seq_len - 1) * 4 * hidden_size * batch_size
    d_ifgo_ptr += (seq_len - 1) * 4 * hidden_size * batch_size
    cell_ptr += (seq_len - 1) * hidden_size * batch_size
    d_out_ptr += (seq_len - 1) * hidden_size * batch_size

    offsets_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[None]
    for _ in range(batch_chunks):
        if pid_b < total_num_pid_b:
            s = sync_ptr + pid_b

            offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None]
            c_mask = (offsets_b < batch_size) & (offsets_h < hidden_size)
            ifgo_indices = offsets_h + 4 * hidden_size * offsets_b
            c0_indices = offsets_h + hidden_size * offsets_b

            if LESS_IO:
                dh = tl.load(d_h_ptr + c0_indices, mask=c_mask)
                dc1 = tl.load(d_c_ptr + c0_indices, mask=c_mask)
                c1 = tl.load(
                    cell_ptr + hidden_size * batch_size + c0_indices, mask=c_mask
                )
                if dtype != "fp32":
                    c1 = c1.cast(tl.float32)
                    dc1 = dc1.cast(tl.float32)
            for _ in tl.range(seq_len, num_stages=1):
                i = tl.load(ifgo_ptr + ifgo_indices, mask=c_mask)
                f = tl.load(
                    ifgo_ptr + ifgo_indices + hidden_size,
                    mask=c_mask,
                )
                g = tl.load(
                    ifgo_ptr + ifgo_indices + 2 * hidden_size,
                    mask=c_mask,
                )
                o = tl.load(
                    ifgo_ptr + ifgo_indices + 3 * hidden_size,
                    mask=c_mask,
                )

                c0 = tl.load(cell_ptr + c0_indices, mask=c_mask)
                if not LESS_IO:
                    c1 = tl.load(
                        cell_ptr + hidden_size * batch_size + c0_indices, mask=c_mask
                    ).cast(tl.float32)
                    dc1 = tl.load(d_c_ptr + c0_indices, mask=c_mask).cast(tl.float32)
                    dh = tl.load(d_h_ptr + c0_indices, mask=c_mask).cast(tl.float32)
                dh += tl.load(d_out_ptr + c0_indices, mask=c_mask)

                if dtype != "fp32":
                    i = i.cast(tl.float32)
                    f = f.cast(tl.float32)
                    g = g.cast(tl.float32)
                    o = o.cast(tl.float32)
                    c0 = c0.cast(tl.float32)

                dc1 += (
                    dh * tl.sigmoid(o) * (1.0 - libdevice.tanh(c1) * libdevice.tanh(c1))
                )

                d_o = dh * libdevice.tanh(c1) * tl.sigmoid(o) * (1 - tl.sigmoid(o))

                # step 2: c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)

                d_f = dc1 * c0 * tl.sigmoid(f) * (1 - tl.sigmoid(f))

                d_i = dc1 * libdevice.tanh(g) * tl.sigmoid(i) * (1.0 - tl.sigmoid(i))

                d_g = (
                    dc1 * tl.sigmoid(i) * (1.0 - libdevice.tanh(g) * libdevice.tanh(g))
                )
                dc0 = dc1 * tl.sigmoid(f)
                if LESS_IO:
                    dc1 = dc0
                    c1 = c0
                if dtype != "fp32":
                    tl.store(d_ifgo_ptr + ifgo_indices, d_i.cast(target), mask=c_mask)
                    tl.store(
                        d_ifgo_ptr + ifgo_indices + hidden_size,
                        d_f.cast(target),
                        mask=c_mask,
                    )
                    tl.store(
                        d_ifgo_ptr + ifgo_indices + 2 * hidden_size,
                        d_g.cast(target),
                        mask=c_mask,
                    )
                    tl.store(
                        d_ifgo_ptr + ifgo_indices + 3 * hidden_size,
                        d_o.cast(target),
                        mask=c_mask,
                    )
                    if not LESS_IO:
                        tl.store(
                            d_c_ptr + c0_indices,
                            dc0.cast(target),
                            mask=c_mask,
                        )
                else:
                    tl.store(d_ifgo_ptr + ifgo_indices, d_i, mask=c_mask)
                    tl.store(d_ifgo_ptr + ifgo_indices + hidden_size, d_f, mask=c_mask)
                    tl.store(
                        d_ifgo_ptr + ifgo_indices + 2 * hidden_size, d_g, mask=c_mask
                    )
                    tl.store(
                        d_ifgo_ptr + ifgo_indices + 3 * hidden_size, d_o, mask=c_mask
                    )
                    if not LESS_IO:
                        tl.store(d_c_ptr + c0_indices, dc0, mask=c_mask)

                tl.atomic_add(s, 1, sem="release")
                # before the channel mixing, make sure all channels are ready
                while tl.atomic_add(s, 0, sem="acquire") < num_pid_h:
                    pass

                s += total_num_pid_b
                accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)
                K = 4 * hidden_size
                N = hidden_size
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                Wh_ptrs = Wh_ptr + (offs_k[:, None] * N + (offsets_h % hidden_size))
                d_ifgo_ptrs = d_ifgo_ptr + (offsets_b * K + offs_k[None, :])

                for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    d = tl.load(
                        d_ifgo_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    W = tl.load(
                        Wh_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                    if dtype != "fp32":
                        W = W.cast(target)

                    accumulator = tl.dot(d, W, accumulator)
                    d_ifgo_ptrs += BLOCK_SIZE_K
                    Wh_ptrs += BLOCK_SIZE_K * N

                if LESS_IO:
                    dh = accumulator.cast(target)
                else:
                    tl.store(
                        d_h_ptr + c0_indices, accumulator.cast(target), mask=c_mask
                    )

                d_ifgo_ptr -= batch_size * 4 * hidden_size
                ifgo_ptr -= batch_size * 4 * hidden_size
                cell_ptr -= batch_size * hidden_size
                d_out_ptr -= batch_size * hidden_size

            if LESS_IO:
                tl.store(d_h_ptr + c0_indices, dh, mask=c_mask)
                tl.store(
                    d_c_ptr + c0_indices,
                    dc1.cast(target),
                    mask=c_mask,
                )
        ifgo_ptr += seq_len * batch_size * 4 * hidden_size
        d_ifgo_ptr += seq_len * batch_size * 4 * hidden_size
        cell_ptr += seq_len * batch_size * hidden_size
        d_out_ptr += seq_len * batch_size * hidden_size

        pid_b += num_pid_b


@triton.jit
def lstm_full_Wgrad(
    d_ifgoT_ptr,
    x_ptr,
    h_ptr,
    dWx_ptr,
    dWh_ptr,
    db_ptr,
    hidden_size,
    input_size,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute:
    (dWx, dWh) = d_ifgoT @ (x, h)
    Shapes:
    (4 hidden, input|hidden) = (4 hidden, batch) x (batch, input|hidden)
    """

    K = batch_size * seq_len
    M = 4 * hidden_size
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_x = tl.cdiv(input_size, BLOCK_SIZE_N)
    num_pid_n_h = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_x + num_pid_n_h
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_n < num_pid_n_x:
        b_ptr = x_ptr
        c_ptr = dWx_ptr
        N = input_size

    else:
        pid_n -= num_pid_n_x
        b_ptr = h_ptr
        c_ptr = dWh_ptr
        N = hidden_size

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = d_ifgoT_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # always accumulate in single precision!
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    db = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if pid_n == 0:
            db += tl.sum(a, axis=1)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    # if you want to fuse an activation in, do it here! should be done in fp32

    tl.store(c_ptrs, accumulator, mask=c_mask)
    if pid_n == 0:
        tl.store(db_ptr + offs_cm, db, mask=offs_cm < M)


@triton.autotune(
    configs=configs.get_graph_autotune_configs(),
    key=["batch_size", "hidden_size", "dtype"],
)
@triton.jit
def lstm_overlap_bwd(
    ifgo_ptr,
    cell_ptr,
    d_c_ptr,
    d_ifgo_ptr,
    d_out_ptr,
    Wh_ptr,
    dh_n_ptr,
    hidden_size,
    batch_size,
    seq_len,
    offset_ptr,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_B: tl.constexpr,  #
    dtype: tl.constexpr,
):
    """Compute:
    Part 1:
    dh_n = d_ifgo @ Wh
    Shapes:
    (batch, hidden) = (batch, 4 hidden) x (4 hidden, hidden)
    Part 2:
    d_ifgo as a function of dh, ..
    """

    if dtype == "fp16":
        target = tl.float16
    elif dtype == "bf16":
        target = tl.bfloat16
    else:
        target = tl.float32

    #######################################################################
    ################ compute dh_{n-1} using d_ifgo_n ######################
    #######################################################################
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(hidden_size, BLOCK_SIZE_H)
    num_pid_in_group = GROUP_SIZE_B * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_B
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_B)
    pid_b = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_h = (pid % num_pid_in_group) // group_size_m

    seq_offset = tl.load(offset_ptr)

    tl.assume(pid_b >= 0)
    tl.assume(pid_h >= 0)
    tl.assume(hidden_size > 0)
    tl.assume(batch_size > 0)
    tl.assume(seq_len > 0)

    tl.assume(BLOCK_SIZE_B > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(BLOCK_SIZE_H > 0)
    tl.assume(GROUP_SIZE_B > 0)
    # tl.assume(seq_offset >= 0)

    K = 4 * hidden_size
    N = hidden_size
    offs_am = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_bn = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    c_ptrs = dh_n_ptr + N * offs_am[:, None] + offs_bn[None, :]
    c_mask = (offs_am[:, None] < batch_size) & (offs_bn[None, :] < N)
    d_ifgo_ptr += seq_offset * batch_size * 4 * hidden_size
    if seq_offset < seq_len:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = d_ifgo_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = Wh_ptr + (offs_k[:, None] * N + offs_bn[None, :])

        dh_n = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)
        for k in tl.range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            if dtype != "fp32":
                b = b.cast(target)

            dh_n = tl.dot(a, b, dh_n)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * N

        if dtype != "fp32":
            dh_n = dh_n.cast(target)
        tl.store(c_ptrs, dh_n, mask=c_mask)
    else:
        dh_n = tl.load(c_ptrs, mask=c_mask)
        if dtype != "fp32":
            dh_n = dh_n.cast(target)

    #######################################################################
    ############## compute d_ifgo_{n-1} using dh_{n-1}#####################
    #######################################################################
    seq_offset -= 1
    if seq_offset >= 0:
        ifgo_indices = offs_am[:, None] * K + offs_bn[None]
        c0_indices = offs_am[:, None] * hidden_size + offs_bn[None]

        ifgo_ptr += seq_offset * 4 * hidden_size * batch_size

        i = tl.load(ifgo_ptr + ifgo_indices, mask=c_mask, cache_modifier=".cv")
        f = tl.load(
            ifgo_ptr + ifgo_indices + hidden_size, mask=c_mask, cache_modifier=".cv"
        )
        g = tl.load(
            ifgo_ptr + ifgo_indices + 2 * hidden_size, mask=c_mask, cache_modifier=".cv"
        )
        o = tl.load(
            ifgo_ptr + ifgo_indices + 3 * hidden_size, mask=c_mask, cache_modifier=".cv"
        )

        c0_ptr = cell_ptr + seq_offset * hidden_size * batch_size

        c0 = tl.load(c0_ptr + c0_indices, mask=c_mask)
        c1 = tl.load(c0_ptr + hidden_size * batch_size + c0_indices, mask=c_mask)
        dc1 = tl.load(d_c_ptr + c0_indices, mask=c_mask)
        dh = tl.load(
            d_out_ptr + seq_offset * hidden_size * batch_size + c0_indices, mask=c_mask
        )

        if dtype != "fp32":
            i = i.cast(tl.float32)
            f = f.cast(tl.float32)
            g = g.cast(tl.float32)
            o = o.cast(tl.float32)
            c0 = c0.cast(tl.float32)
            c1 = c1.cast(tl.float32)
            dc1 = dc1.cast(tl.float32)

        dh += dh_n
        dc1 += dh * tl.sigmoid(o) * (1.0 - libdevice.tanh(c1) * libdevice.tanh(c1))

        d_o = dh * libdevice.tanh(c1) * tl.sigmoid(o) * (1 - tl.sigmoid(o))

        # step 2: c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
        d_c0 = dc1 * tl.sigmoid(f)
        d_f = dc1 * c0 * tl.sigmoid(f) * (1 - tl.sigmoid(f))
        d_i = dc1 * libdevice.tanh(g) * tl.sigmoid(i) * (1.0 - tl.sigmoid(i))
        d_g = dc1 * tl.sigmoid(i) * (1.0 - libdevice.tanh(g) * libdevice.tanh(g))

        d_ifgo_ptr -= batch_size * 4 * hidden_size
        if dtype != "fp32":
            tl.store(d_ifgo_ptr + ifgo_indices, d_i.cast(target), mask=c_mask)
            tl.store(
                d_ifgo_ptr + ifgo_indices + hidden_size, d_f.cast(target), mask=c_mask
            )
            tl.store(
                d_ifgo_ptr + ifgo_indices + 2 * hidden_size,
                d_g.cast(target),
                mask=c_mask,
            )
            tl.store(
                d_ifgo_ptr + ifgo_indices + 3 * hidden_size,
                d_o.cast(target),
                mask=c_mask,
            )
            tl.store(d_c_ptr + c0_indices, d_c0, mask=c_mask)
        else:
            tl.store(d_ifgo_ptr + ifgo_indices, d_i, mask=c_mask)
            tl.store(d_ifgo_ptr + ifgo_indices + hidden_size, d_f, mask=c_mask)
            tl.store(d_ifgo_ptr + ifgo_indices + 2 * hidden_size, d_g, mask=c_mask)
            tl.store(d_ifgo_ptr + ifgo_indices + 3 * hidden_size, d_o, mask=c_mask)
            tl.store(d_c_ptr + c0_indices, d_c0, mask=c_mask)


@triton.jit
def lstm_one_step_phase1_bwd(
    d_out_ptr,
    d_h_ptr,
    d_c_ptr,
    ifgo_ptr,
    cell_ptr,
    d_ifgo_ptr,
    offset_ptr,
    batch_size,
    hidden_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    # shape: (batch, channel, ifgo)
    pid_h = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    seq_offset = tl.load(offset_ptr)

    offsets_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[None]
    offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None]

    ifgo_indices = offsets_h + 4 * hidden_size * offsets_b
    c0_indices = offsets_h + hidden_size * offsets_b

    mask = (offsets_h < hidden_size) & (offsets_b < batch_size)

    ifgo_ptr = ifgo_ptr + seq_offset * 4 * hidden_size * batch_size

    i = tl.load(ifgo_ptr + ifgo_indices, mask=mask)
    f = tl.load(ifgo_ptr + ifgo_indices + hidden_size, mask=mask)
    g = tl.load(ifgo_ptr + ifgo_indices + 2 * hidden_size, mask=mask)
    o = tl.load(ifgo_ptr + ifgo_indices + 3 * hidden_size, mask=mask)

    c0_ptr = cell_ptr + seq_offset * hidden_size * batch_size
    c1_ptr = cell_ptr + (seq_offset + 1) * hidden_size * batch_size

    c0 = tl.load(c0_ptr + c0_indices, mask=mask)
    c1 = tl.load(c1_ptr + c0_indices, mask=mask)

    dh = tl.load(d_h_ptr + c0_indices, mask=mask)
    dh += tl.load(
        d_out_ptr + seq_offset * hidden_size * batch_size + c0_indices, mask=mask
    )

    dc1 = tl.load(d_c_ptr + c0_indices, mask=mask)

    dc1 += dh * tl.sigmoid(o) * (1.0 - libdevice.tanh(c1) * libdevice.tanh(c1))

    d_o = dh * libdevice.tanh(c1) * tl.sigmoid(o) * (1 - tl.sigmoid(o))
    tl.store(d_ifgo_ptr + ifgo_indices + 3 * hidden_size, d_o, mask=mask)

    # step 2: c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
    d_c0 = dc1 * tl.sigmoid(f)
    tl.store(d_c_ptr + c0_indices, d_c0, mask=mask)

    d_f = dc1 * c0 * tl.sigmoid(f) * (1 - tl.sigmoid(f))
    tl.store(d_ifgo_ptr + ifgo_indices + hidden_size, d_f, mask=mask)

    d_i = dc1 * libdevice.tanh(g) * tl.sigmoid(i) * (1.0 - tl.sigmoid(i))
    tl.store(d_ifgo_ptr + ifgo_indices, d_i, mask=mask)

    d_g = dc1 * tl.sigmoid(i) * (1.0 - libdevice.tanh(g) * libdevice.tanh(g))
    tl.store(d_ifgo_ptr + ifgo_indices + 2 * hidden_size, d_g, mask=mask)


#######################################################################################
# matmul kernels - modified versions of ###############################################
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
########################################################################################
def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    fp32: tl.constexpr,
    ACTIVATION: tl.constexpr,  #
    accumulate: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if accumulate:
        accumulator = tl.load(c_ptrs, mask=c_mask).cast(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator
    if not fp32:
        c = c.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def my_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    offset_ptr,
    a_off_stride,
    b_off_stride,
    c_off_stride,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    db_ptr,
    c_idemn,
    db_idemn,
    row_acc: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    dtype: tl.constexpr,
    accumulate: tl.constexpr,
):
    """from: triton tutorial"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    op = tl.load(offset_ptr)
    mod = op % 2
    if accumulate:
        c_ptr += mod * c_idemn
        c_read = (1 - 2 * mod) * c_idemn

    if row_acc:
        db_ptr += mod * db_idemn
        db_read = (1 - 2 * mod) * db_idemn

    a_offset = op * a_off_stride
    b_offset = op * b_off_stride
    c_offset = op * c_off_stride

    tl.assume(c_idemn >= 0)
    tl.assume(db_idemn >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(a_offset >= 0)
    tl.assume(b_offset >= 0)
    tl.assume(c_offset >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr + a_offset + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        b_ptr + b_offset + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr + c_offset + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # always accumulate in single precision!
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    db = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if row_acc and (pid_n == 0):
            db += tl.sum(a, axis=1)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # if you want to fuse an activation in, do it here! should be done in fp32!
    c = accumulator
    if dtype == "fp16":
        c = c.to(tl.float16)
    elif dtype == "bf16":
        c = c.to(tl.bfloat16)

    if accumulate:
        c += tl.load(c_ptrs + c_read, mask=c_mask)

    tl.store(c_ptrs, c, mask=c_mask)
    if row_acc and (pid_n == 0):
        db += tl.load(db_ptr + db_read + offs_am)
        tl.store(db_ptr + offs_cm, db, mask=offs_cm < M)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.

    fp32 = a.dtype == torch.float32
    # artifact from triton tutorials
    c = torch.empty(
        (M, N), device=a.device, dtype=torch.float32 if fp32 else torch.float16
    )
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        fp32=fp32,
        ACTIVATION=activation,  #
        accumulate=False,
    )
    return c


def matmul_v2(a, b, c, accumulate=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.

    fp32 = a.dtype == torch.float32
    # artifact from triton tutorials

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        fp32=fp32,
        ACTIVATION="",  #
        accumulate=accumulate,
    )
    return c
