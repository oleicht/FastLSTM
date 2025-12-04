import pytest

import torch
import torch.nn as nn

from fastlstm.lstm import (
    naive_lstm_cell_fwd,
    naive_lstm_fwd,
    lstm_v1_fwd,
    naive_lstm_cell_bwd,
    FastLSTM,
    FlashLSTM,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rtol = 0.0
atol = 1e-3


def test_naive_cell_fwd():
    batch_dim = 12
    channel_dim = 128
    hidden_dim = 213
    x = torch.randn((batch_dim, channel_dim), device=DEVICE)

    ref_lstm = nn.LSTMCell(
        input_size=channel_dim, hidden_size=hidden_dim, device=x.device
    )

    h_ref, c_ref = ref_lstm(x)
    h_naive, c_naive, _ = naive_lstm_cell_fwd(
        x=x,
        h0=None,
        c0=None,
        Wx=ref_lstm.weight_ih,
        bx=ref_lstm.bias_ih,
        Wh=ref_lstm.weight_hh,
        bh=ref_lstm.bias_hh,
    )

    torch.testing.assert_close(h_ref, h_naive)
    torch.testing.assert_close(c_ref, c_naive)


def test_naive_lstm_fn_fwd():
    batch_dim = 12
    seq_dim = 63
    channel_dim = 128
    hidden_dim = 213
    x = torch.randn((seq_dim, batch_dim, channel_dim), device=DEVICE)

    ref_lstm = nn.LSTM(input_size=channel_dim, hidden_size=hidden_dim, device=x.device)

    out_ref, _ = ref_lstm(x)
    out_naive, *_ = naive_lstm_fwd(
        x=x,
        h0=None,
        c0=None,
        Wx=ref_lstm.weight_ih_l0,
        bx=ref_lstm.bias_ih_l0,
        Wh=ref_lstm.weight_hh_l0,
        bh=ref_lstm.bias_hh_l0,
    )

    out_naive = out_naive[
        1:
    ]  # time-step 0 is the initial condition, needed for backprop
    torch.testing.assert_close(out_ref, out_naive, rtol=rtol, atol=atol)


def test_naive_lstm_cell_bwd():
    batch_dim = 12
    channel_dim = 128
    hidden_dim = 213
    x = torch.randn((batch_dim, channel_dim), device=DEVICE)

    ref_lstm = nn.LSTMCell(
        input_size=channel_dim, hidden_size=hidden_dim, device=x.device
    )

    h_ref, c_ref = ref_lstm(x)

    dh = h_ref.sum()
    h_ref.retain_grad()
    dh.backward()

    _, c1, ifgo = naive_lstm_cell_fwd(
        x,
        None,
        None,
        ref_lstm.weight_ih,
        ref_lstm.bias_ih,
        ref_lstm.weight_hh,
        ref_lstm.bias_hh,
    )

    dx, dh0, dc0, dWx, dbx, dWh, dbh = naive_lstm_cell_bwd(
        dh=h_ref.grad,
        dc1=torch.zeros(12, 213, device=h_ref.device),
        x=x,
        h0=None,
        c0=None,
        c1=c1,
        ifgo=torch.stack(ifgo, dim=-1),
        Wx=ref_lstm.weight_ih,
        Wh=ref_lstm.weight_hh,
    )

    torch.testing.assert_close(dbx, ref_lstm.bias_ih.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dbh, ref_lstm.bias_hh.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dWx, ref_lstm.weight_ih.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dWh, ref_lstm.weight_hh.grad, atol=atol, rtol=rtol)


def test_naive_lstm_init():
    torch.manual_seed(123)
    m1 = FastLSTM(8, 3, num_layers=2)
    torch.manual_seed(123)
    m2 = nn.LSTM(8, 3, num_layers=2)

    for n, p in m1.named_parameters():
        try:
            torch.testing.assert_close(p, getattr(m2, n))
        except Exception as e:
            print("*******************************")
            print(f"Problem with weight {n}")
            print("*******************************")
            raise e


def test_lstm_v1_fwd():
    if not DEVICE == "cuda":
        pytest.skip("No gpu detected but triton needs it. Test skipped")
    batch_dim = 12
    seq_dim = 63
    channel_dim = 128
    hidden_dim = 213
    x = torch.randn((seq_dim, batch_dim, channel_dim), device="cuda")

    ref_lstm = nn.LSTM(input_size=channel_dim, hidden_size=hidden_dim, device=x.device)

    out_ref, _ = ref_lstm(x)
    out_naive, *_ = lstm_v1_fwd(
        x=x,
        h0=None,
        c0=None,
        Wx=ref_lstm.weight_ih_l0,
        bx=ref_lstm.bias_ih_l0,
        Wh=ref_lstm.weight_hh_l0,
        bh=ref_lstm.bias_hh_l0,
    )

    out_naive = out_naive[
        1:
    ]  # time-step 0 is the initial condition, needed for backprop
    torch.testing.assert_close(out_ref, out_naive, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "version",
    [
        "naive-pt",
        "graph",
        "persistent",
        "fast",
    ],
)
@pytest.mark.parametrize(
    "seq_size",
    [10, 100][:1],
)
@pytest.mark.parametrize(
    "batch_size",
    [4, 44, 100][:1],
)
@pytest.mark.parametrize(
    "hidden_size",
    [16, 65, 257, 2030],
)
def test_fastlstm(version, seq_size, batch_size, hidden_size):
    """I believe the implementation is correct. However, errors of the gradients
    grow with both seq-size and batch-size."""
    if not DEVICE == "cuda":
        pytest.skip("No gpu detected but triton needs it. Test skipped")
    input_size = 123
    num_layers = 2
    torch.manual_seed(123)
    m1 = FastLSTM(
        input_size,
        hidden_size,
        num_layers=num_layers,
        version=version,
        device="cuda",
    )
    torch.manual_seed(123)
    m2 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, device="cuda")

    x = torch.randn((seq_size, batch_size, input_size), device="cuda")

    y1, (hn1, cn1) = m1(x)
    y2, (hn2, cn2) = m2(x)
    # check fwd
    torch.testing.assert_close(y1, y2, atol=10 * atol, rtol=rtol)
    torch.testing.assert_close(hn1, hn2, atol=10 * atol, rtol=rtol)
    torch.testing.assert_close(cn1, cn2, atol=10 * atol, rtol=rtol)

    (y1.sum() + y2.sum()).backward()

    # check bwd
    for n, p in m1.named_parameters():
        torch.testing.assert_close(
            p.grad,
            getattr(m2, n).grad,
            atol=max(seq_size / 4, 1)
            * max(batch_size / 8, 1)
            * max(1, hidden_size / 512)
            * atol
            * 10,
            rtol=0.03,
        )


def test_flashlstm_fwd(hidden_size=256, batch_size=16, seq_len=1024):
    if not DEVICE == "cuda":
        pytest.skip("No gpu detected - test skipped")

    torch.manual_seed(123)
    lstm_t = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        device="cuda",
        dtype=torch.float16,
    )

    lstm_f = FlashLSTM(hidden_size, hidden_size, dtype=torch.float16, backend="cuda")
    lstm_f.b.data.copy_(
        (lstm_t.bias_ih_l0 + lstm_t.bias_hh_l0).reshape(4, 1, hidden_size)
    )
    lstm_f.R.data.copy_((lstm_t.weight_hh_l0).reshape(4, 1, hidden_size, hidden_size))
    lstm_f.gate_in.weight.data.copy_(lstm_t.weight_ih_l0)

    x = torch.randn(
        (seq_len, batch_size, hidden_size), device="cuda", dtype=torch.float16
    )
    o1 = lstm_f(x)
    o2 = lstm_t(x)

    torch.testing.assert_close(o1[0], o2[0], atol=1e-3, rtol=0)
