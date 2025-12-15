import torch
import torch.nn as nn
from triton.testing import do_bench

from fastlstm.lstm import FastLSTM

seq_len = 128
batch_size = 64
hidden_size = 64
num_layers = 1

x = torch.randn((seq_len, batch_size, hidden_size), device="cuda")

if True:
    m = FastLSTM(
        hidden_size,
        hidden_size,
        num_layers=num_layers,
        version="persistent",
        device="cuda",
    )
else:
    m = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, device="cuda")

ms, min_ms, max_ms = do_bench(
    lambda: m.forward(x), quantiles=[0.5, 0.2, 0.8], warmup=100, rep=1000
)
