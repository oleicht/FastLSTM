from itertools import product
import json
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from triton.testing import do_bench
from tqdm import tqdm

import fastlstm.lstm as flstm


def generate_configs():
    """
    - seq_len = [64 ... 2048]
    - batch_size = [4 ... 128]
    - hidden_size = [64 ... 2048]
    """
    configs = {}
    for seq_len, bs, hs, layer in product(range(3, 4), range(5,-1,-1), range(5,-1,-1), range(1)):
        name = f"s{seq_len}_b{bs}_h{hs}_l{layer}"
        configs[name] = (64 << seq_len, 4 << bs, 64 << hs, 1 + layer)

    return configs


def run_benchmark(config, models, mode, dtype=None):
    medians = []

    seq_len, batch_size, hidden_size, num_layers = config

    x = torch.randn((seq_len, batch_size, hidden_size), device="cuda", dtype=dtype)
    for model in models:
        if model.lower() in ["graph", "persistent", "fast"]:
            m = flstm.FastLSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                device="cuda",
                num_layers=num_layers,
                version=model,
                dtype=dtype,
            )
        elif model.lower() in ["cuda", "cuda_fused", "triton_fused"]:
            m = flstm.FlashLSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                device="cuda",
                dtype=dtype,
                backend=model.lower()
            )
        else:
            m = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device="cuda",
                dtype=dtype,
            )
        # with torch.no_grad():
        match mode:
            case "fwd":
                fn = lambda: m.forward(x)
            case "full":
                fn = lambda: m.forward(x)[0].sum().backward()
            case "bwd":
                assert model.lower() in ["graph", "persistent"], (
                    f"{model} doesn't have separate bwd"
                )
                helper = (
                    flstm.lstm_graph_bwd
                    if (model.startswith("graph"))
                    else flstm.lstm_persistent_bwd
                )

                fn = lambda: helper(
                    dh=torch.randn(
                        (seq_len, batch_size, hidden_size), device="cuda", dtype=dtype
                    ),
                    dc_n=torch.randn(
                        (batch_size, hidden_size), device="cuda", dtype=dtype
                    ),
                    dh_n=torch.randn(
                        (batch_size, hidden_size), device="cuda", dtype=dtype
                    ),
                    x=x,
                    h=torch.randn(
                        (seq_len + 1, batch_size, hidden_size),
                        device="cuda",
                        dtype=dtype,
                    ),
                    cell=torch.randn(
                        (seq_len + 1, batch_size, hidden_size),
                        device="cuda",
                        dtype=dtype,
                    ),
                    ifgo=torch.randn(
                        (seq_len, batch_size, 4 * hidden_size),
                        device=x.device,
                        dtype=dtype,
                    ),
                    Wx=m.weight_ih_l0,
                    Wh=m.weight_hh_l0,
                )
            case _:
                raise ValueError(f"Unknown mode {mode}")

        # adjust warmups and repetitions for small problems to get reliabe measurements
        warmup = 40
        rep = 160
        if seq_len < 128:
            warmup *= 1.5
            rep *= 3

        if hidden_size < 128:
            warmup *= 1.5
            rep *= 3

        if batch_size < 32:
            warmup *= 1.5
            rep *= 3

        t = do_bench(fn, return_mode="median")  # , warmup=int(warmup), rep=int(rep))

        # try:
        #     t = do_bench(fn, return_mode="median", warmup=int(warmup), rep=int(rep))
        # except (torch.OutOfMemoryError, torch.AcceleratorError,
        #         CompilationError, OutOfResources):
        #     t = np.nan
        medians += [t]
    return medians


if __name__ == "__main__":
    fname = "graph"
    overwrite = True
    dtype = [None, torch.bfloat16, torch.float16][-1]

    flstm.TRACK_AUTOTUNE_RUNTIMES = True

    for mode in ["fwd", "full", "bwd"][:1]:
        p = Path(f"{mode}_{fname}.parquet")
        if not overwrite and p.exists():
            raise ValueError(f"File {mode}_{fname} exists")
        models = [
            "graph",
            # "lstm"
            # "persistent",
        ]

        if False:
            models += [
                "cuda",
                "cuda_fused",
                "triton_fused",
            ]

        if False and mode != "bwd":
            models += [
                "lstm",
                "fast",
            ]

        bench = partial(run_benchmark, models=models, mode=mode, dtype=dtype)
        res = {}
        configs = generate_configs()
        for name, config in tqdm(generate_configs().items(), total=len(configs)):
            res[tuple(name.split("_"))] = bench(config)

            # overwrite existing file every time a config finishes
            pd.DataFrame(res, index=models).T.to_parquet(p)

            if flstm.TRACK_AUTOTUNE_RUNTIMES:
                with open(f"tuning_{mode}_{fname}.json", "w") as fh:
                    json.dump(flstm.CONFIG_RES, fh, default=str)
