import torch
import torch.nn.functional as F
import pytest

from model.transformer import SwiGLU


def test_swiglu_forward_shape_and_dtype():
    """Verify that the forward pass preserves input shapes [batch, seq_len, d_model] and dtypes"""
    d_model = 64
    hidden_dim = 192
    batch_size, seq_len = 2, 32

    swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, d_model)

    out = swiglu(x)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_swiglu_forward_computation():
    """Verify forward computation explicitly follows:
    down_proj( SiLU(gate_proj(x)) * content_proj(x) )
    """
    d_model = 4
    hidden_dim = 8

    swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
    x = torch.randn(2, 3, d_model)

    # manual analytical calculation using linear weight matrices
    gate = F.silu(F.linear(x, swiglu.gate_proj.weight))
    content = F.linear(x, swiglu.content_proj.weight)
    expected = F.linear(gate * content, swiglu.down_proj.weight)

    out = swiglu(x)

    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_swiglu_backward_pass_gradients():
    """Verify that gradients flow back to inputs and all three weight matrices
    (gate_proj, content_proj, down_proj) without NaNs or zero gradients"""
    d_model = 128
    hidden_dim = 340
    swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)

    x = torch.randn(4, 16, d_model, requires_grad=True)

    out = swiglu(x)
    loss = out.sum()
    loss.backward()

    # check input gradient
    assert x.grad is not None, "Gradient for input x should not be None."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."

    # check parameter gradients
    for name, param in swiglu.named_parameters():
        assert param.grad is not None, f"Gradient for parameter '{name}' should not be None."
        assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaN values."
        assert (param.grad != 0).any(), f"Gradient for '{name}' is completely zero."
