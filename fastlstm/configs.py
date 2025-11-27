from dataclasses import dataclass

import torch
import triton


@dataclass
class PersistentData:
    BATCH_SIZE: int = 128
    HIDDEN_SIZE: int = 128

ProblemShape = PersistentData()

SM_count =  torch.cuda.get_device_properties("cuda").multi_processor_count

def get_graph_fwd_autotune_configs():
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
        ) for s in [4, 6, 8]
          for w in [1, 2, 4]
          for k in [32, 64]
          for h in [8, 16]
          for b in [8, 16, 32, 64]
        ][:2]


def compute_batch_layout(hidden_size, BLOCK_SIZE_H, batch_size, BLOCK_SIZE_B):
    max_grid_size = SM_count
    num_pid_h=triton.cdiv(hidden_size, BLOCK_SIZE_H)
    num_pid_b = triton.cdiv(batch_size, BLOCK_SIZE_B)
    batch_chunks = 1
    if num_pid_h * num_pid_b > max_grid_size:
        num_pid_b = triton.next_power_of_2(num_pid_b) // 2
        batch_chunks *= 2
        while num_pid_h * num_pid_b > max_grid_size:
            num_pid_b //= 2
            batch_chunks *= 2

    return {"batch_chunks": batch_chunks,
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

    assert num_pid_b * num_pid_h  <= SM_count
        
    return (num_pid_b * num_pid_h, )


def get_persistent_autotune_configs(pfd: PersistentData):
    """
    Be careful here! The setup relies on global state:
    - HIDDEN_SIZE and BATCH_SIZE passed from lstm.py so that no deadlocks occur
    """
    # step 1:
    # restrict to block_sizes_h that lead to non-deadlocked configs
    hidden_block_sizes = []
    for block_size in [8, 16, 32, 64, 128, 256]:
        if triton.cdiv(pfd.HIDDEN_SIZE, block_size) > SM_count:
            continue
        elif block_size >= 2 * pfd.HIDDEN_SIZE: 
            continue

        hidden_block_sizes += [block_size]
    assert len(hidden_block_sizes) > 0, f"BLOCK_SIZE_H not large enough to support hidden_size {pfd.HIDDEN_SIZE} on {SM_count} many SMs."

    batch_block_sizes = []
    for block_size in [1, 8, 16, 32, 64, 128]:
        if (block_size >= 2 * pfd.BATCH_SIZE):
            continue
        if block_size == 1 and pfd.BATCH_SIZE >=12:
            continue

        batch_block_sizes += [block_size]


    configs =  [
    triton.Config(
            {
                "BLOCK_SIZE_K": 32,
            } | compute_batch_layout(hidden_size=pfd.HIDDEN_SIZE,
                                     BLOCK_SIZE_H=h,
                                     batch_size=pfd.BATCH_SIZE,
                                     BLOCK_SIZE_B=b),
            num_warps=w,
            num_stages=s,
        )
        for w in [2, 4]
        for s in [2, 4, 6, 8]
        for h in hidden_block_sizes
        for b in batch_block_sizes
    ]
    
    best_sm_ratio = max([c.kwargs["num_pid_b"] * (triton.cdiv(pfd.HIDDEN_SIZE, c.kwargs["BLOCK_SIZE_H"]) ) / SM_count for c in configs])
    # remove configs that utilize too few SMs
    configs = [c for c in configs if c.kwargs["num_pid_b"] * (triton.cdiv(pfd.HIDDEN_SIZE, c.kwargs["BLOCK_SIZE_H"])) / SM_count > 0.75 * best_sm_ratio]

    assert len(configs)>0
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
        for nh, nb in [(11, 2),
                       # (8, 2)
                       ]
    ]
    return configs
