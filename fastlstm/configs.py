from functools import partial

import torch
import triton


# persistent fwd args
HIDDEN_SIZE=128
BATCH_SIZE=128
BLOCK_SIZE_B=32

def get_graph_fwd_autotune_configs():
    return [
    triton.Config(
            {
                "BLOCK_SIZE_H": 8,
                "BLOCK_SIZE_B": 8,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_B": 8,
            },
            num_warps=1,
            num_stages=s,
        ) for s in [4, 6, 8] ] + [

        triton.Config(
            {
                "BLOCK_SIZE_H": 8,
                "BLOCK_SIZE_B": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_B": 8,
            },
            num_warps=2,
            num_stages=6,
        ),

        triton.Config(
            {
                "BLOCK_SIZE_H": 32,
                "BLOCK_SIZE_B": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_B": 8,
            },
            num_warps=4,
            num_stages=6,
        ),
    ]


def persistent_layout(hidden_size, BLOCK_SIZE_H, batch_size, BLOCK_SIZE_B):
    max_grid_size = torch.cuda.get_device_properties("cuda").multi_processor_count

    # modify BLOCK_SIZES to prevent deadlocks. Assumption: one program per SM
    while triton.cdiv(hidden_size, BLOCK_SIZE_H) > max_grid_size:
        BLOCK_SIZE_H *= 2

    num_pid_h = triton.cdiv(hidden_size, BLOCK_SIZE_H)
    num_pid_b = triton.cdiv(batch_size, BLOCK_SIZE_B)
    batch_chunks = 1
    if num_pid_h * num_pid_b > max_grid_size:
        num_pid_b = triton.next_power_of_2(num_pid_b) // 2
        batch_chunks *= 2
        while num_pid_h * num_pid_b > max_grid_size:
            num_pid_b //= 2
            batch_chunks *= 2

    return {"BLOCK_SIZE_H": BLOCK_SIZE_H,
            "BLOCK_SIZE_B": BLOCK_SIZE_B,
            "batch_chunks": batch_chunks,
            "num_pid_b": num_pid_b
            }

def compute_persistent_grid_dim(kwargs):
    num_pid_b = kwargs["num_pid_b"]
    num_pid_h = triton.cdiv(kwargs["hidden_size"], kwargs["BLOCK_SIZE_H"])
    assert num_pid_b * num_pid_h  < torch.cuda.get_device_properties("cuda").multi_processor_count
    return (num_pid_b * num_pid_h, )

def get_persistent_fwd_autotune_configs():
    """
    Be careful here! The setup relies on global state:
    - HIDDEN_SIZE and BATCH_SIZE passed from lstm.py so that no deadlocks occur
    - BLOCK_SIZE_B can't be automatically tuned at the time
    """
    return [
    triton.Config(
            {
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_B": 8,
            } | persistent_layout(BLOCK_SIZE_H=8,
                                  hidden_size=HIDDEN_SIZE,
                                  batch_size=BATCH_SIZE,
                                  BLOCK_SIZE_B=BLOCK_SIZE_B),
            num_warps=2,
            num_stages=s,
        ) for s in [2, 4, 6] ]
