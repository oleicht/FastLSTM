import pandas as pd
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import trange

from fastlstm.lstm import FastLSTM


def run_overfit(
    models, seq_len=128, batch_size=16, hidden_size=256, precision=torch.bfloat16
):
    y = torch.randn((seq_len, batch_size, 1), device="cuda")
    x = torch.randn((seq_len, batch_size, hidden_size), device="cuda")

    losses = {}

    amp_enabled = precision in [torch.float16, torch.bfloat16]

    for model in models:
        torch.manual_seed(123)
        onet = nn.Linear(in_features=hidden_size, out_features=1, device="cuda")
        match model:
            case "lstm":
                snet = nn.LSTM(hidden_size, hidden_size, device="cuda")
            case _:
                snet = FastLSTM(hidden_size, hidden_size, version=model, device="cuda")

        opt = Adam(list(onet.parameters()) + list(snet.parameters()), lr=1e-3)
        losses[model] = []

        for iter in trange(1_000):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=precision, enabled=amp_enabled):
                yhat = onet(snet(x)[0])
                loss = ((yhat - y) ** 2).mean()
                loss.backward()
                opt.step()

            if iter % 50 == 0:
                losses[model] += [loss.item()]

    return pd.DataFrame(losses)


if __name__ == "__main__":
    df = run_overfit(
        ["fast", "lstm", "persistent", "graph"],
        precision=[None, torch.float16, torch.bfloat16][0],
    )
    df.to_parquet("bf16_losses.parquet")
