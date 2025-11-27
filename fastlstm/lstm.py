from collections import defaultdict
from functools import partial
from importlib import reload

import torch
import torch.nn as nn
import triton

import fastlstm.kernels as kernels
import fastlstm.configs as configs
from flashrnn import flashrnn

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"

TRACK_AUTOTUNE_RUNTIMES = False
CONFIG_RES = defaultdict(list)

dtype_str = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

def lstm_persistent_fwd(x, h0, c0, Wx, bx, Wh, bh, triton_config=None, version=1):
    torch.cuda.nvtx.range_push("fwd setup")
    if x.dim() == 2:
        x = x[None]

    seq_len = x.shape[0]
    batch_size = x.shape[1]
    hidden_size = Wh.shape[1]
    assert x.is_contiguous()
    assert Wh.is_contiguous()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("ifgo")
    ifgo = torch.addmm(
        bx + bh, x.view(seq_len * batch_size, -1), Wx.T, beta=1.0, alpha=1.0
    ).view(seq_len, batch_size, -1)

    assert ifgo.stride(1) == 4 * hidden_size
    assert ifgo.stride(2) == 1
    assert Wh.stride(0) == hidden_size
    assert Wh.stride(1) == 1

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("out-cell-init")
    out, cell = torch.zeros(
        (2, seq_len + 1, batch_size, hidden_size), device=x.device, dtype=x.dtype
    ).unbind(0)

    if h0 is not None:
        out[0] = h0
        cell[0] = c0

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("kernel setup")
    # if triton_config is None:
    #     # RTX 2000 Ada
    #     BLOCK_SIZE_H = 8
    #     BLOCK_SIZE_B = 32
    #     BLOCK_SIZE_K = 32
    #     GROUP_SIZE_B = 8
    #     num_warps = 2

    #     if hidden_size <= 256:
    #         num_stages = 6
    #     elif hidden_size <= 512:
    #         num_stages = 4
    #     else:
    #         num_stages = 2

    #     H100
    #     BLOCK_SIZE_H = 32
    #     BLOCK_SIZE_B = 32
    #     BLOCK_SIZE_K = 32
    #     GROUP_SIZE_B = 8
    #     num_warps = 2
    #     num_stages = 6

    # else:
    #     BLOCK_SIZE_H = triton_config["BLOCK_SIZE_H"]
    #     BLOCK_SIZE_B = triton_config["BLOCK_SIZE_B"]
    #     BLOCK_SIZE_K = triton_config["BLOCK_SIZE_K"]
    #     GROUP_SIZE_B = triton_config["GROUP_SIZE_B"]
    #     num_warps = triton_config["num_warps"]
    #     num_stages = triton_config["num_stages"]


    grid = configs.compute_persistent_grid_dim 
    global_sync = torch.zeros(
            seq_len * batch_size,
        dtype=torch.int,
        device=x.device,
    )

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("run kernel")

    dtype = dtype_str[ifgo.dtype]

    kernel = kernels.persistent_fwd_kernel_v2 if version==2 else kernels.persistent_fwd_kernel
 
    if not any((batch_size, hidden_size, dtype) == k[:3] for k in kernel.cache):
        # triton.heuristics can't be used here
        # we need to modify the triton.Configs as a function of the inputs which isn't supported
        CurrentShape = configs.PersistentData(BATCH_SIZE=batch_size, HIDDEN_SIZE=hidden_size)
        if version==1 and (configs.ProblemShape != CurrentShape):
            configs.ProblemShape = CurrentShape
            reload(kernels)  # reload the kernel with dynamically adjusted shapes etc.
            kernel = kernels.persistent_fwd_kernel

        kernel[grid](
            ifgo_ptr=torch.randn_like(ifgo),  # ifgo gets overwritten
            cell_ptr=cell,
            h_ptr=out,
            W_h_ptr=Wh,
            seq_len=6,
            batch_size=batch_size,
            hidden_size=hidden_size,
            global_sync_ptr=torch.zeros_like(global_sync),
            dtype=dtype_str[ifgo.dtype],
        )
        if TRACK_AUTOTUNE_RUNTIMES:
            for k, v in kernel.configs_timings.items():
                CONFIG_RES[f"persistent-h{hidden_size}-b{batch_size}-{dtype}"] += [(str(k), v)]
    
    kernel[grid](
        ifgo_ptr=ifgo,
        cell_ptr=cell,
        h_ptr=out,
        W_h_ptr=Wh,
        seq_len=seq_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        global_sync_ptr=global_sync,
        dtype=dtype,
    )

    torch.cuda.nvtx.range_pop()

    return out, cell, ifgo


def lstm_graph_fwd(x, h0, c0, Wx, bx, Wh, bh, triton_config=None):
    torch.cuda.nvtx.range_push("fwd_start")
    if x.dim() == 2:
        x = x[None]

    seq_len, batch_size, input_size = x.shape
    hidden_size = Wh.shape[1]

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("ifgo_start")
    ifgo = torch.addmm(
        bx + bh, x.view(seq_len * batch_size, -1), Wx.T, beta=1.0, alpha=1.0
    ).view(seq_len, batch_size, -1)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("out-cell-init")
    out = torch.empty(
        (seq_len + 1, batch_size, hidden_size), device=x.device, dtype=x.dtype
    )
    cell = torch.empty(
        (seq_len + 1, batch_size, hidden_size), device=x.device, dtype=x.dtype
    )

    if h0 is not None:
        out[0] = h0
        cell[0] = c0
    else:
        out[0] = 0.0
        cell[0] = 0.0

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("triton_start")
    # if triton_config is None:
    #     RTX 2000 Ada
    #     BLOCK_SIZE_H = 8
    #     BLOCK_SIZE_B = 32
    #     BLOCK_SIZE_K = 32
    #     GROUP_SIZE_B = 8
    #     num_warps = 2
    #     num_stages = 1

    #     if batch_size > 32:
    #         BLOCK_SIZE_B = 64

    #     # H100
    #     BLOCK_SIZE_H = 32
    #     BLOCK_SIZE_B = 64
    #     BLOCK_SIZE_K = 64
    #     GROUP_SIZE_B = 8
    #     num_warps = 4
    #     num_stages = 6


    # else:
    #     BLOCK_SIZE_H = triton_config["BLOCK_SIZE_H"]
    #     BLOCK_SIZE_B = triton_config["BLOCK_SIZE_B"]
    #     BLOCK_SIZE_K = triton_config["BLOCK_SIZE_K"]
    #     GROUP_SIZE_B = triton_config["GROUP_SIZE_B"]
    #     num_warps = triton_config["num_warps"]
    #     num_stages = triton_config["num_stages"]
    # grid = (
    #     triton.cdiv(batch_size, BLOCK_SIZE_B) * triton.cdiv(hidden_size, BLOCK_SIZE_H),
    # )

    grid = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_SIZE_B"]) * triton.cdiv(hidden_size, META["BLOCK_SIZE_H"]),
    )

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("warmup")

    offset = torch.zeros((1,), device=x.device, dtype=torch.int)
    dtype = dtype_str[ifgo.dtype]

    # if the config hasn't run yet: autotune the kernel
    if not any((batch_size, hidden_size, dtype) == k[:3] for k in kernels.one_step_fwd.cache):
        kernels.one_step_fwd[grid](
            ifgo_ptr=torch.randn_like(ifgo),  # ifgo gets overwritten
            cell_ptr=cell,
            h_ptr=out,
            W_h_ptr=Wh,
            offset_ptr=offset,
            batch_size=batch_size,
            hidden_size=hidden_size,
            # BLOCK_SIZE_B=BLOCK_SIZE_B,
            # BLOCK_SIZE_K=BLOCK_SIZE_K,
            # BLOCK_SIZE_H=BLOCK_SIZE_H,
            # GROUP_SIZE_B=GROUP_SIZE_B,
            # num_warps=num_warps,
            # num_stages=num_stages,
            dtype=dtype,
        )
        if TRACK_AUTOTUNE_RUNTIMES:
            for k, v in kernels.one_step_fwd.configs_timings.items():
                CONFIG_RES[f"graph-h{hidden_size}-b{batch_size}-{dtype}"] += [(str(k), v)]
    torch.cuda.nvtx.range_pop()


    torch.cuda.nvtx.range_push("graph capture")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        kernels.one_step_fwd[grid](
            ifgo_ptr=ifgo,
            cell_ptr=cell,
            h_ptr=out,
            W_h_ptr=Wh,
            offset_ptr=offset,
            batch_size=batch_size,
            hidden_size=hidden_size,
            # BLOCK_SIZE_B=BLOCK_SIZE_B,
            # BLOCK_SIZE_K=BLOCK_SIZE_K,
            # BLOCK_SIZE_H=BLOCK_SIZE_H,
            # GROUP_SIZE_B=GROUP_SIZE_B,
            # num_warps=num_warps,
            # num_stages=num_stages,
            dtype=dtype,
        )
        offset.add_(1)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("replay")
    for _ in range(seq_len):
        g.replay()
    torch.cuda.nvtx.range_pop()

    return out, cell, ifgo


def lstm_persistent_bwd(dh, dc_n, dh_n, x, h, cell, ifgo, Wx, Wh, triton_config=None):
    torch.cuda.nvtx.range_push("persistent bwd init")
    d_ifgo = torch.empty_like(ifgo)

    seq_len, batch_size, hidden_size = dh.shape

    dh = dh.contiguous()

    if triton_config is not None:
        BLOCK_SIZE_H = triton_config["BLOCK_SIZE_H"]
        BLOCK_SIZE_B = triton_config["BLOCK_SIZE_B"]
        BLOCK_SIZE_K = triton_config["BLOCK_SIZE_K"]
        num_warps = triton_config["num_warps"]
        num_stages = triton_config["num_stages"]

        BLOCK_SIZE_H2 = triton_config["BLOCK_SIZE_H2"]
        BLOCK_SIZE_B2 = triton_config["BLOCK_SIZE_B2"]
        BLOCK_SIZE_K2 = triton_config["BLOCK_SIZE_K2"]
        num_warps2 = triton_config["num_warps2"]
        num_stages2 = triton_config["num_stages2"]

    else:
        BLOCK_SIZE_H = 32
        BLOCK_SIZE_B = 32
        BLOCK_SIZE_K = 64
        num_warps = 4
        if hidden_size < 512:
            num_stages = 6
        elif hidden_size < 1024:
            num_stages = 4
        else:
            num_stages = 3

        BLOCK_SIZE_H2 = 32
        BLOCK_SIZE_B2 = 32
        BLOCK_SIZE_K2 = 64
        num_warps2 = 4
        num_stages2 = 2

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("persistent bwd grid setup")

    # num_b_splits = triton.cdiv(batch_size, BLOCK_SIZE_B)
    # max_grid_size = torch.cuda.get_device_properties("cuda").multi_processor_count

    # while triton.cdiv(hidden_size, BLOCK_SIZE_H) > max_grid_size:
    #     BLOCK_SIZE_H *= 2
    # num_h_splits = triton.cdiv(hidden_size, BLOCK_SIZE_H)

    # num_batch_iter = 1
    # while (
    #     num_h_splits * triton.cdiv(batch_size, num_batch_iter * BLOCK_SIZE_B)
    #     > max_grid_size
    # ):
    #     num_batch_iter += 1

    # Pgrid = (num_h_splits * triton.cdiv(batch_size, num_batch_iter * BLOCK_SIZE_B),)
    Pgrid = configs.compute_persistent_grid_dim

    sync = torch.zeros((seq_len * batch_size), dtype=torch.int, device=x.device)

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("persistent bwd kernel")
    dtype = dtype_str[ifgo.dtype]
    kernel = kernels.lstm_persistent_seq_bwd

    if not any((batch_size, hidden_size, dtype) == k[:3] for k in kernel.cache):
        # triton.heuristics can't be used here
        # we need to modify the triton.Configs as a function of the inputs which isn't supported
        CurrentShape = configs.PersistentData(BATCH_SIZE=batch_size, HIDDEN_SIZE=hidden_size)
        if (configs.ProblemShape != CurrentShape):
            configs.ProblemShape = CurrentShape
            reload(kernels)  # reload the kernel with dynamically adjusted shapes etc.
            kernel = kernels.lstm_persistent_seq_bwd

        kernel[Pgrid](
            d_out_ptr=dh,
            d_h_ptr=torch.randn_like(dh_n),
            d_c_ptr=torch.randn_like(dc_n),
            d_ifgo_ptr=d_ifgo,
            ifgo_ptr=ifgo,
            cell_ptr=cell,
            Wh_ptr=Wh,
            sync_ptr=torch.zeros_like(sync),
            batch_size=batch_size,
            hidden_size=hidden_size,
            seq_len=6,
            dtype=dtype
            )
        if TRACK_AUTOTUNE_RUNTIMES:
            for k, v in kernel.configs_timings.items():
                CONFIG_RES[f"persistentBWD-h{hidden_size}-b{batch_size}-{dtype}"] += [(str(k), v)]


    kernel[Pgrid](
        d_out_ptr=dh,
        d_h_ptr=dh_n,
        d_c_ptr=dc_n,
        d_ifgo_ptr=d_ifgo,
        ifgo_ptr=ifgo,
        cell_ptr=cell,
        Wh_ptr=Wh,
        sync_ptr=sync,
        batch_size=batch_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        # num_batch_iter=num_batch_iter,
        # BLOCK_SIZE_H=BLOCK_SIZE_H,
        # BLOCK_SIZE_B=BLOCK_SIZE_B,
        # BLOCK_SIZE_K=BLOCK_SIZE_K,
        # num_warps=num_warps,
        # num_stages=num_stages,
        dtype=dtype,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("matmuls for gradients")

    d_x = d_ifgo @ Wx

    if True:
        dbx, dbh, dWx, dWh = Wgrad(d_ifgo, x, h)
    else:
        # this is not competitive
        d_ifgoT = d_ifgo.view(-1, d_ifgo.shape[-1]).T.contiguous()
        input_size = Wx.shape[1]

        dWx = torch.empty_like(Wx)
        dWh = torch.empty_like(Wh)
        db = torch.empty((4 * hidden_size,), device=x.device, dtype=x.dtype)
        wGrid = (
            triton.cdiv(4 * hidden_size, BLOCK_SIZE_B2)
            * (
                triton.cdiv(hidden_size, BLOCK_SIZE_H2)
                + triton.cdiv(input_size, BLOCK_SIZE_H2)
            ),
        )
        kernels.lstm_full_Wgrad[wGrid](
            d_ifgoT_ptr=d_ifgoT,
            x_ptr=x,
            h_ptr=h,
            dWx_ptr=dWx,
            dWh_ptr=dWh,
            db_ptr=db,
            hidden_size=hidden_size,
            input_size=input_size,
            batch_size=batch_size,
            seq_len=seq_len,
            BLOCK_SIZE_M=BLOCK_SIZE_B2,
            BLOCK_SIZE_K=BLOCK_SIZE_K2,
            BLOCK_SIZE_N=BLOCK_SIZE_H2,
            GROUP_SIZE_M=8,
            num_warps=num_warps2,
            num_stages=num_stages2,
        )

        dbx = dbh = db
    torch.cuda.nvtx.range_pop()
    return d_x, dh_n, dc_n, dWx, dbx, dWh, dbh


def lstm_graph_bwd(
    dh,
    dc_n,
    dh_n,
    x,
    h,
    cell,
    ifgo,
    Wx,
    Wh,
    triton_config=None,
):
    torch.cuda.nvtx.range_push("graph bwd init")
    d_ifgo = torch.empty_like(ifgo)

    seq_len, batch_size, hidden_size = dh.shape
    input_size = x.shape[-1]

    offset = torch.empty((1,), dtype=torch.int, device=x.device)
    offset[0] = seq_len

    dh = dh.contiguous()

    assert ifgo.is_contiguous()
    assert cell.is_contiguous()
    assert ifgo.shape == (seq_len, batch_size, 4 * hidden_size)
    assert dh.shape == (seq_len, batch_size, hidden_size)
    assert cell.shape == (seq_len + 1, batch_size, hidden_size)
    assert dc_n.shape == (batch_size, hidden_size)
    assert Wx.shape == (4 * hidden_size, input_size)

    if triton_config is not None:
        BLOCK_SIZE_H = triton_config["BLOCK_SIZE_H"]
        BLOCK_SIZE_B = triton_config["BLOCK_SIZE_B"]
        BLOCK_SIZE_K = triton_config["BLOCK_SIZE_K"]
        num_warps = triton_config["num_warps"]
        num_stages = triton_config["num_stages"]
        overlap_version = triton_config.get("overlap_version", True)

    else:
        BLOCK_SIZE_H = 32
        BLOCK_SIZE_B = 32
        BLOCK_SIZE_K = 64
        num_warps = 4
        num_stages = 4
        overlap_version = True  # makes small problems ~5% faster and large ones <1%

    grid = (
        triton.cdiv(hidden_size, BLOCK_SIZE_H) * triton.cdiv(batch_size, BLOCK_SIZE_B),
    )

    point_grid = (
        triton.cdiv(hidden_size, 32),
        triton.cdiv(batch_size, 32),
    )

    def run():
        if overlap_version:
            kernels.lstm_overlap_bwd[grid](
                ifgo_ptr=ifgo,
                cell_ptr=cell,
                d_c_ptr=dc_n,
                d_ifgo_ptr=d_ifgo,
                d_out_ptr=dh,
                Wh_ptr=Wh,
                dh_n_ptr=dh_n,
                hidden_size=hidden_size,
                batch_size=batch_size,
                seq_len=seq_len,
                offset_ptr=offset,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=8,
                num_warps=num_warps,
                num_stages=num_stages,
                dtype=dtype_str[ifgo.dtype],
            )
        offset.add_(-1)
        if not overlap_version:
            kernels.lstm_ifgo_bwd[point_grid](
                d_out_ptr=dh,
                d_h_ptr=dh_n,
                cell_ptr=cell,
                d_c_ptr=dc_n,
                ifgo_ptr=ifgo,
                d_ifgo_ptr=d_ifgo,
                d_ifgo_stride=d_ifgo.stride(0),
                offset_ptr=offset,
                batch_size=batch_size,
                hidden_size=hidden_size,
                BLOCK_SIZE_H=32,
                BLOCK_SIZE_B=32,
                dtype=dtype_str[ifgo.dtype],
            )
            kernels.lstm_h_grad[grid](
                d_ifgo_ptr=d_ifgo,
                Wh_ptr=Wh,
                dh_n_ptr=dh_n,
                hidden_size=hidden_size,
                batch_size=batch_size,
                offset_ptr=offset,
                BLOCK_SIZE_M=BLOCK_SIZE_H,
                BLOCK_SIZE_N=BLOCK_SIZE_B,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=8,
                num_warps=num_warps,
                num_stages=num_stages,
                dtype=dtype_str[ifgo.dtype],
            )

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("graph bwd warmup")
    run()
    torch.cuda.nvtx.range_pop()
    if seq_len > 1:
        torch.cuda.nvtx.range_push("graph bwd capture kernel")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("replay kernel")
        for _ in range(seq_len - 1 + int(overlap_version)):
            g.replay()
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("matmuls for gradients")
    d_x = d_ifgo @ Wx

    if True:
        dbx, dbh, dWx, dWh = Wgrad(d_ifgo, x, h)
    else:
        # fused version is slower
        d_ifgoT = d_ifgo.view(-1, d_ifgo.shape[-1]).T.contiguous()
        wGrid = 1
        dWx = torch.empty_like(Wx)
        dWh = torch.empty_like(Wh)
        db = torch.empty((4 * hidden_size,), device=x.device, dtype=x.dtype)
        wGrid = (
            triton.cdiv(4 * hidden_size, BLOCK_SIZE_B)
            * (
                triton.cdiv(hidden_size, BLOCK_SIZE_H)
                + triton.cdiv(input_size, BLOCK_SIZE_H)
            ),
        )
        kernels.lstm_full_Wgrad[wGrid](
            d_ifgoT_ptr=d_ifgoT,
            x_ptr=x,
            h_ptr=h,
            dWx_ptr=dWx,
            dWh_ptr=dWh,
            db_ptr=db,
            hidden_size=hidden_size,
            input_size=input_size,
            batch_size=batch_size,
            seq_len=seq_len,
            BLOCK_SIZE_M=BLOCK_SIZE_B,
            BLOCK_SIZE_K=32,
            BLOCK_SIZE_N=BLOCK_SIZE_H,
            GROUP_SIZE_M=8,
        )

        dbx = dbh = db
    torch.cuda.nvtx.range_pop()

    return d_x, dh_n, dc_n, dWx, dbx, dWh, dbh


def Wgrad(d_ifgo, x, h):
    d_ifgoT = d_ifgo.view(-1, d_ifgo.shape[-1]).T.contiguous()

    db = d_ifgoT.sum(dim=1)
    dWx = d_ifgoT @ x.view(-1, x.shape[-1])
    dWh = d_ifgoT @ h[:-1].view(-1, h.shape[-1])

    return db, db, dWx, dWh


class PersistentLSTMfn(torch.autograd.Function):
    FWD_TRITON_CONFIG = None
    BWD_TRITON_CONFIG = None

    @staticmethod
    def forward(ctx, x, h_0, c_0, Wx, bx, Wh, bh):
        out, cell, ifgo = lstm_persistent_fwd(
            x,
            h_0,
            c_0,
            Wx,
            bx,
            Wh,
            bh,
            triton_config=PersistentLSTMfn.FWD_TRITON_CONFIG,
        )
        ctx.h0_is_nan = h_0 is None
        ctx.save_for_backward(x, out, cell, ifgo, Wx, Wh)
        # first one is initial condition
        return out[1:], (out[-1], cell[-1])

    @staticmethod
    def backward(ctx, dh, d_out_cell):
        x, out, cell, ifgo, Wx, Wh = ctx.saved_tensors
        if d_out_cell is None:
            dc_n, dh_n = torch.zeros_like(cell[:2]).unbind(0)
        else:
            raise ValueError("Not implemented")
        d_x, dh_0, dc_0, dWx, dbx, dWh, dbh = lstm_persistent_bwd(
            dh,
            dc_n,
            dh_n,
            x,
            out,
            cell,
            ifgo,
            Wx,
            Wh,
            triton_config=PersistentLSTMfn.BWD_TRITON_CONFIG,
        )
        if ctx.h0_is_nan:
            dh_0 = None
            dc_0 = None
        return d_x, dh_0, dc_0, dWx, dbx, dWh, dbh


class GraphLSTMfn(torch.autograd.Function):
    FWD_TRITON_CONFIG = None
    BWD_TRITON_CONFIG = None

    @staticmethod
    def forward(ctx, x, h_0, c_0, Wx, bx, Wh, bh):
        torch.cuda.nvtx.range_push("graph_fwd")
        out, cell, ifgo = lstm_graph_fwd(
            x, h_0, c_0, Wx, bx, Wh, bh, triton_config=GraphLSTMfn.FWD_TRITON_CONFIG
        )
        ctx.save_for_backward(x, out, cell, ifgo, Wx, Wh)
        ctx.h0_is_nan = h_0 is None
        out = out[1:]  # first one is initial condition
        torch.cuda.nvtx.range_pop()
        return out, (out[-1], cell[-1])

    @staticmethod
    def backward(ctx, dh, d_out_cell):
        torch.cuda.nvtx.range_push("graph_bwd")
        x, out, cell, ifgo, Wx, Wh = ctx.saved_tensors
        if d_out_cell is None:
            dc_n, dh_n = torch.zeros_like(cell[:2]).unbind(0)
        else:
            raise ValueError("Not implemented")

        d_x, dh_0, dc_0, dWx, dbx, dWh, dbh = lstm_graph_bwd(
            dh,
            dc_n,
            dh_n,
            x,
            out,
            cell,
            ifgo,
            Wx,
            Wh,
            triton_config=GraphLSTMfn.BWD_TRITON_CONFIG,
        )
        if ctx.h0_is_nan:
            dh_0 = None
            dc_0 = None
        torch.cuda.nvtx.range_pop()
        return d_x, dh_0, dc_0, dWx, dbx, dWh, dbh


class LSTMfn(torch.autograd.Function):
    FWD_TRITON_CONFIG = None
    BWD_TRITON_CONFIG = None

    @staticmethod
    def forward(ctx, x, h_0, c_0, Wx, bx, Wh, bh):
        # select kernel
        _, batch_size, hidden_size = x.shape
        fn = lstm_graph_fwd
        if hidden_size >= 1024:
            fn = lstm_graph_fwd
        elif (batch_size < 64) or (hidden_size <= 64):
            fn = lstm_persistent_fwd
        elif (hidden_size / 64) + (batch_size / 64) < 3.5:
            fn = lstm_persistent_fwd

        torch.cuda.nvtx.range_push("graph_fwd")
        out, cell, ifgo = fn(
            x, h_0, c_0, Wx, bx, Wh, bh, triton_config=LSTMfn.FWD_TRITON_CONFIG
        )
        ctx.save_for_backward(x, out, cell, ifgo, Wx, Wh)
        ctx.h0_is_nan = h_0 is None
        out = out[1:]  # first one is initial condition
        torch.cuda.nvtx.range_pop()
        return out, (out[-1], cell[-1])

    @staticmethod
    def backward(ctx, dh, d_out_cell):
        # select kernel
        if dh.shape[-1] < 1500:
            fn = lstm_persistent_bwd
        else:
            fn = lstm_graph_bwd

        torch.cuda.nvtx.range_push("graph_bwd")
        x, out, cell, ifgo, Wx, Wh = ctx.saved_tensors
        if d_out_cell is None:
            dc_n, dh_n = torch.zeros_like(cell[:2]).unbind(0)
        else:
            raise ValueError("Not implemented")
        d_x, dh_0, dc_0, dWx, dbx, dWh, dbh = fn(
            dh,
            dc_n,
            dh_n,
            x,
            out,
            cell,
            ifgo,
            Wx,
            Wh,
            triton_config=LSTMfn.BWD_TRITON_CONFIG,
        )
        if ctx.h0_is_nan:
            dh_0 = None
            dc_0 = None
        torch.cuda.nvtx.range_pop()
        return d_x, dh_0, dc_0, dWx, dbx, dWh, dbh


#######################################################################################
################################ nn.Module wrapper ####################################
#######################################################################################
class FastLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        version="fast",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.version = version
        self.device = device
        self.dtype = dtype
        sqrt_k_inv = 1 / hidden_size**0.5
        for layer in range(num_layers):
            setattr(
                self,
                f"weight_ih_l{layer}",
                nn.Parameter(
                    torch.rand(
                        4 * self.hidden_size,
                        self.input_size if layer == 0 else self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    * sqrt_k_inv
                    - sqrt_k_inv
                ),
            )
            setattr(
                self,
                f"weight_hh_l{layer}",
                nn.Parameter(
                    torch.rand(
                        4 * self.hidden_size,
                        self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    * sqrt_k_inv
                    - sqrt_k_inv
                ),
            )
            setattr(
                self,
                f"bias_ih_l{layer}",
                nn.Parameter(
                    torch.rand(
                        4 * self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    * sqrt_k_inv
                    - sqrt_k_inv
                ),
            )
            setattr(
                self,
                f"bias_hh_l{layer}",
                nn.Parameter(
                    torch.rand(
                        4 * self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2
                    * sqrt_k_inv
                    - sqrt_k_inv
                ),
            )

    def forward(self, x, h_c_0=None):
        if h_c_0 is None:
            h_0, c_0 = [None] * self.num_layers, [None] * self.num_layers
        else:
            h_0, c_0 = h_c_0

        h_n = []
        c_n = []

        match self.version:
            case "naive-pt":
                fn = NaiveLSTMfn.apply
            case "persistent":
                fn = PersistentLSTMfn.apply
            case "graph":
                fn = GraphLSTMfn.apply
            case "fast":
                fn = LSTMfn.apply
            case "v1":
                fn = V1LSTMfn.apply
            case _:
                raise ValueError(f"Not Implemnted: {self.version}")

        for layer in range(self.num_layers):
            x, (h_i, c_i) = fn(
                x,
                h_0[layer],
                c_0[layer],
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
                getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_hh_l{layer}"),
            )
            h_n += [h_i]
            c_n += [c_i]

        return x, (torch.stack(h_n), torch.stack(c_n))


class FlashLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, backend, dtype=None, device="cuda"):
        super().__init__()
        self.gate_in = nn.Linear(input_size, 4 * hidden_size, dtype=dtype, device=device, bias=False)
        sqrt_k_inv = 1 / hidden_size**0.5
        self.R = nn.Parameter(torch.randn([4, 1, hidden_size, hidden_size], device=device, dtype=dtype)* 2
                    * sqrt_k_inv
                    - sqrt_k_inv)
        self.b = nn.Parameter(torch.randn([4, 1, hidden_size], device=device, dtype=dtype)* 2
                    * sqrt_k_inv
                    - sqrt_k_inv)
        self.hidden_size = hidden_size
        self.backend = backend
        self.dtype = {torch.float16: "float16",
                      torch.bfloat16: "bfloat16",
                       torch.float32: "float32" }[dtype]

    def forward(self, x):
        R = self.R
        # convert to batch_first
        Wx = self.gate_in(x.transpose(0,1))
        Wx = Wx.reshape(
                Wx.shape[0], Wx.shape[1], R.shape[0], R.shape[1], R.shape[2]
            )

        h_frnn, hlast_frnn = flashrnn(
            Wx=Wx,
            R=R,
            b=self.b,
            states=None,
            function="lstm",
            backend=self.backend,
            dtype=self.dtype,
            )

        return h_frnn[0].transpose(0, 1).squeeze(-2), hlast_frnn

#######################################################################################
################# versions below were experimental and aren't performant ##############
#######################################################################################
def lstm_v1_fwd(
    x,
    h0,
    c0,
    Wx,
    bx,
    Wh,
    bh,
):
    if x.dim() == 2:
        x = x[None]

    seq_len = x.shape[0]
    batch_size = x.shape[1]
    hidden_size = Wh.shape[0] // 4

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_B = 1

    grid = (
        triton.cdiv(hidden_size, BLOCK_SIZE_H),
        triton.cdiv(batch_size, BLOCK_SIZE_B),
    )

    out = torch.zeros((seq_len + 1, batch_size, hidden_size), device=x.device)
    cell = torch.zeros((seq_len + 1, batch_size, hidden_size), device=x.device)

    if h0 is not None:
        out[0] = h0
        cell[0] = c0

    torch.cuda.nvtx.range_push("triton fwd")

    ifgo = x @ Wx.T + (bx + bh)

    for i in range(seq_len):
        # phase 1; compute ifgo
        ifgo[i] += kernels.matmul(out[i], Wh.T)

        # phase 2: compute h1 & c1
        kernels.triton_lstm_cell_fwd_phase2[grid](
            ifgo[i],
            cell[i],
            out[i + 1],
            cell[i + 1],
            batch_size,
            hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

    torch.cuda.nvtx.range_pop()
    return out, cell, ifgo


def lstm_v1_bwd(dh, dc_n, dh_n, x, h, cell, ifgo, Wx, Wh, mode=2):
    assert mode in [0, 1, 2]
    dWx = torch.zeros((2, Wx.shape[0], Wx.shape[1]), device=Wx.device, dtype=Wx.dtype)
    dWh = torch.zeros((2, Wh.shape[0], Wh.shape[1]), device=Wh.device, dtype=Wh.dtype)
    db = torch.zeros((2, Wx.shape[0]), device=Wx.device, dtype=Wx.dtype)

    d_x = torch.empty_like(x)
    d_ifgo = torch.empty_like(ifgo[0])

    seq_len, batch_size, hidden_size = dh.shape
    input_size = x.shape[-1]

    offset = torch.empty((1,), dtype=torch.int, device=x.device)
    offset[0] = seq_len

    assert ifgo.is_contiguous()
    assert cell.is_contiguous()

    dh = dh.contiguous()

    assert ifgo.shape == (seq_len, batch_size, 4 * hidden_size)
    assert dh.shape == (seq_len, batch_size, hidden_size)
    assert cell.shape == (seq_len + 1, batch_size, hidden_size)
    assert dc_n.shape == (batch_size, hidden_size)
    assert Wx.shape == (4 * hidden_size, input_size)

    # Dgrad
    DgradGrid1 = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(hidden_size, META["BLOCK_SIZE_N"]),
    )

    DgradGrid2 = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(input_size, META["BLOCK_SIZE_N"]),
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    DgradGrid = (
        triton.cdiv(batch_size, BLOCK_SIZE_M)
        * (
            triton.cdiv(input_size, BLOCK_SIZE_N)
            + triton.cdiv(hidden_size, BLOCK_SIZE_N)
        ),
    )

    # Wgrad
    WgradGrid1 = lambda META: (
        triton.cdiv(Wx.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(Wx.shape[1], META["BLOCK_SIZE_N"]),
    )

    WgradGrid2 = lambda META: (
        triton.cdiv(Wh.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(Wh.shape[1], META["BLOCK_SIZE_N"]),
    )

    WgradGrid = (
        triton.cdiv(Wh.shape[0], BLOCK_SIZE_M)
        * (
            triton.cdiv(Wh.shape[1], BLOCK_SIZE_N)
            + triton.cdiv(Wx.shape[1], BLOCK_SIZE_N)
        ),
    )

    # phase 1: compute dpo
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_B = 1
    grid = (
        triton.cdiv(hidden_size, BLOCK_SIZE_H),
        triton.cdiv(batch_size, BLOCK_SIZE_B),
    )

    def run():
        offset.add_(-1)
        kernels.lstm_ifgo_bwd[grid](
            d_out_ptr=dh,
            d_h_ptr=dh_n,
            cell_ptr=cell,
            d_c_ptr=dc_n,
            ifgo_ptr=ifgo,
            d_ifgo_ptr=d_ifgo,
            d_ifgo_stride=0,
            offset_ptr=offset,
            batch_size=batch_size,
            hidden_size=hidden_size,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            dtype=dtype_str[ifgo.dtype],
        )

        if mode == 0:
            kernels.matmul_v2(d_ifgo, Wh, dh_n)
            kernels.matmul_v2(d_ifgo, Wx, d_x[offset.item()])
        elif mode == 1:
            kernels.my_matmul_kernel[DgradGrid1](
                a_ptr=d_ifgo,
                b_ptr=Wh,
                c_ptr=dh_n,
                M=batch_size,
                N=hidden_size,
                K=4 * hidden_size,
                a_off_stride=0,
                b_off_stride=0,
                c_off_stride=0,
                offset_ptr=offset,
                dtype="float32",
                accumulate=False,
                row_acc=False,
                db_ptr=offset,
                c_idemn=0,
                db_idemn=0,
                stride_am=d_ifgo.stride(0),
                stride_ak=d_ifgo.stride(1),
                stride_bk=Wh.stride(0),
                stride_bn=Wh.stride(1),
                stride_cm=dh_n.stride(0),
                stride_cn=dh_n.stride(1),
            )

            kernels.my_matmul_kernel[DgradGrid2](
                a_ptr=d_ifgo,
                b_ptr=Wx,
                c_ptr=d_x,
                M=batch_size,
                N=input_size,
                K=4 * hidden_size,
                a_off_stride=0,
                b_off_stride=0,
                c_off_stride=batch_size * input_size,
                offset_ptr=offset,
                dtype="float32",
                accumulate=False,
                row_acc=False,
                db_ptr=offset,
                c_idemn=0,
                db_idemn=0,
                stride_am=d_ifgo.stride(0),
                stride_ak=d_ifgo.stride(1),
                stride_bk=Wx.stride(0),
                stride_bn=Wx.stride(1),
                stride_cm=d_x.stride(1),
                stride_cn=d_x.stride(2),
            )
        elif mode == 2:
            kernels.lstm_Dgrad[DgradGrid](
                d_ifgo_ptr=d_ifgo,
                Wh_ptr=Wh,
                Wx_ptr=Wx,
                dh_n_ptr=dh_n,
                d_x_ptr=d_x,
                offset_ptr=offset,
                hidden_size=hidden_size,
                input_size=input_size,
                batch_size=batch_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=8,
            )

        if mode == 0:
            dWx.add_(kernels.matmul(x[offset[0]].T.contiguous(), d_ifgo).T)
            dWh.add_(kernels.matmul(h[offset[0]].T.contiguous(), d_ifgo).T)
            db.add_(d_ifgo.sum(dim=0))

        elif mode == 1:
            kernels.my_matmul_kernel[WgradGrid1](
                a_ptr=d_ifgo,
                b_ptr=x,
                c_ptr=dWx,
                M=4 * hidden_size,
                N=input_size,
                K=batch_size,
                a_off_stride=0,
                b_off_stride=x.stride(0),
                c_off_stride=0,
                offset_ptr=offset,
                dtype="float32",
                accumulate=True,
                row_acc=True,
                db_ptr=db,
                c_idemn=dWx.stride(0),
                db_idemn=db.stride(0),
                stride_am=d_ifgo.stride(1),
                stride_ak=d_ifgo.stride(0),
                stride_bk=x.stride(1),
                stride_bn=x.stride(2),
                stride_cm=dWx.stride(1),
                stride_cn=dWx.stride(2),
            )
            kernels.my_matmul_kernel[WgradGrid2](
                a_ptr=d_ifgo,
                b_ptr=h,
                c_ptr=dWh,
                M=4 * hidden_size,
                N=hidden_size,
                K=batch_size,
                a_off_stride=0,
                b_off_stride=h.stride(0),
                c_off_stride=0,
                offset_ptr=offset,
                dtype="float32",
                accumulate=True,
                row_acc=False,
                db_ptr=offset,
                c_idemn=dWh.stride(0),
                db_idemn=0,
                stride_am=d_ifgo.stride(1),
                stride_ak=d_ifgo.stride(0),
                stride_bk=h.stride(1),
                stride_bn=h.stride(2),
                stride_cm=dWh.stride(1),
                stride_cn=dWh.stride(2),
            )
        elif mode == 2:
            kernels.lstm_Wgrad[WgradGrid](
                d_ifgo_ptr=d_ifgo,
                x_ptr=x,
                h_ptr=h,
                dWx_ptr=dWx,
                dWh_ptr=dWh,
                db_ptr=db,
                offset_ptr=offset,
                hidden_size=hidden_size,
                input_size=input_size,
                batch_size=batch_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                GROUP_SIZE_M=8,
            )

    run()
    if seq_len > 1:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()

        for _ in range(seq_len - 1):
            g.replay()

    idemn_selector = 0
    dbx = db[idemn_selector]
    dbh = db[idemn_selector]
    return d_x, dh_n, dc_n, dWx[idemn_selector], dbx, dWh[idemn_selector], dbh


class V1LSTMfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_0, c_0, Wx, bx, Wh, bh):
        torch.cuda.nvtx.range_push("graph_fwd")
        out, cell, ifgo = lstm_v1_fwd(
            x,
            h_0,
            c_0,
            Wx,
            bx,
            Wh,
            bh,
        )
        ctx.save_for_backward(x, out, cell, ifgo, Wx, Wh, h_0)
        out = out[1:]  # first one is initial condition
        torch.cuda.nvtx.range_pop()
        return out, (out[-1], cell[-1])

    @staticmethod
    def backward(ctx, dh, d_out_cell):
        torch.cuda.nvtx.range_push("graph_bwd")
        x, out, cell, ifgo, Wx, Wh, h0 = ctx.saved_tensors
        assert d_out_cell is None, "Not implemented"
        dc_n = torch.zeros_like(cell[0])
        dh_n = torch.zeros_like(cell[0])
        d_x, dh_0, dc_0, dWx, dbx, dWh, dbh = lstm_v1_bwd(
            dh, dc_n, dh_n, x, out, cell, ifgo, Wx, Wh
        )
        if h0 is None:
            dh_0 = None
            dc_0 = None
        torch.cuda.nvtx.range_pop()
        return d_x, dh_0, dc_0, dWx, dbx, dWh, dbh


#######################################################################################
############################ basic pytorch implementations ############################
#######################################################################################


def naive_lstm_cell_fwd(
    x,
    h0,
    c0,
    Wx,
    bx,
    Wh,
    bh,
):
    if h0 is None:
        batch_size = x.shape[0]
        h0 = torch.zeros((batch_size, Wh.shape[1]), device=x.device)
        assert c0 is None
        c0 = torch.zeros((batch_size, Wh.shape[1]), device=x.device)
    i, f, g, o = torch.chunk(x @ Wx.T + bx + h0 @ Wh.T + bh, 4, dim=-1)

    c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
    h1 = torch.sigmoid(o) * torch.tanh(c1)
    return h1, c1, (i, f, g, o)


def naive_lstm_fwd(
    x: torch.Tensor,
    h0,
    c0,
    Wx,
    bx,
    Wh,
    bh,
):
    """assume batch_first=False layout"""
    if x.dim() == 2:
        x = x[None]

    run_step = partial(naive_lstm_cell_fwd, Wx=Wx, bx=bx, Wh=Wh, bh=bh)

    if h0 is None:
        h0 = torch.zeros((x.shape[1], Wh.shape[0] // 4), device=x.device)
        c0 = torch.zeros((x.shape[1], Wh.shape[0] // 4), device=x.device)

    out = [h0]
    cell = [c0]
    ifgo_list = []
    for t in x:
        h0, c0, ifgo = run_step(t, h0, c0)
        out += [h0]
        cell += [c0]
        ifgo_list += [torch.stack(ifgo, dim=-1)]

    return torch.stack(out), torch.stack(cell), torch.stack(ifgo_list)


def naive_lstm_cell_bwd(dh, dc1, x, h0, c0, c1, ifgo: torch.Tensor, Wx, Wh):
    """Goal: return gradients of all params and dx, dh0 + dc0"""
    # h1, c1, (pi, pf, pg, po) = naive_lstm_cell_fwd(x, h0, c0, Wx, bx, Wh, bh)

    pi, pf, pg, po = ifgo.unbind(-1)

    # step 1: exploid h = o * tanh(c)
    # dc1 += dL/dh dh/dc1 = dh o (1 - tanh**2(c))
    dc1 += dh * torch.sigmoid(po) * (1 - torch.tanh(c1) ** 2)

    do = dh * torch.tanh(c1)
    # o = sigmoid(dpo) -> dL/dpo = dL/do do/dpo
    dpo = do * torch.sigmoid(po) * (1 - torch.sigmoid(po))

    # step 2: c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
    dc0 = dc1 * torch.sigmoid(pf)

    if c0 is None:
        dpf = torch.zeros_like(dc0)
    else:
        df = dc1 * c0
        dpf = df * torch.sigmoid(pf) * (1 - torch.sigmoid(pf))

    di = dc1 * torch.tanh(pg)
    dpi = di * torch.sigmoid(pi) * (1 - torch.sigmoid(pi))

    dg = dc1 * torch.sigmoid(pi)
    dpg = dg * (1 - torch.tanh(pg) ** 2)

    dp = torch.concat([dpi, dpf, dpg, dpo], dim=-1)

    # dL/dx = dL/dp dp/dx
    dh0 = dp @ Wh
    dx = dp @ Wx

    # weights are shared!! that means we need to sum over batch (& seq)
    dWx = dp.T @ x
    if h0 is None:
        dWh = torch.zeros_like(Wh)
    else:
        dWh = dp.T @ h0

    dbx = dp.sum(dim=0)
    dbh = dp.sum(dim=0)

    return dx, dh0, dc0, dWx, dbx, dWh, dbh


def naive_lstm_bwd(
    dh,
    dc_n,
    x,
    h,
    c,
    ifgo,
    Wx,
    Wh,
):
    dWx = torch.zeros_like(Wx)
    dbx = torch.zeros(Wx.shape[0], device=Wx.device, dtype=Wx.dtype)
    dWh = torch.zeros_like(Wh)
    dbh = torch.zeros(Wh.shape[0], device=Wx.device, dtype=Wx.dtype)
    dx_out = torch.zeros_like(x)
    dc1 = dc_n
    dh0 = 0
    for s in range(dh.shape[0] - 1, -1, -1):
        dx, dh0, dc1, dWx_tmp, dbx_tmp, dWh_tmp, dbh_tmp = naive_lstm_cell_bwd(
            dh=dh[s] + dh0,  # gradient from output and prior timestep
            dc1=dc1,
            x=x[s],
            h0=h[s],
            c0=c[s],
            c1=c[s + 1],
            ifgo=ifgo[s],
            Wx=Wx,
            Wh=Wh,
        )

        dx_out[s] = dx
        dWx += dWx_tmp
        dbx += dbx_tmp
        dWh += dWh_tmp
        dbh += dbh_tmp

    return dx_out, dh0, dc1, dWx, dbx, dWh, dbh


class NaiveLSTMfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_0, c_0, Wx, bx, Wh, bh):
        out, cell, ifgo = naive_lstm_fwd(x, h_0, c_0, Wx, bx, Wh, bh)
        ctx.save_for_backward(x, out, cell, ifgo, Wx, Wh, h_0)
        out = out[1:]  # first one is initial condition
        return out, (out[-1], cell[-1])

    @staticmethod
    def backward(ctx, dh, d_out_cell):
        x, out, cell, ifgo, Wx, Wh, h0 = ctx.saved_tensors
        assert d_out_cell is None, "Not implemented"
        dc_n = torch.zeros_like(cell[0])
        dx_out, dh0, dc1, dWx, dbx, dWh, dbh = naive_lstm_bwd(
            dh, dc_n, x, out, cell, ifgo, Wx, Wh
        )
        if h0 is None:
            dh0 = None
            dc1 = None
        return dx_out, dh0, dc1, dWx, dbx, dWh, dbh
