import torch
import pytest

from model.transformer import Derf


def test_derf_forward_shape_and_dtype():
    """Verify that the forward pass preserves input shapes and dtypes"""
    d_model = 768
    batch_size, seq_len = 2, 32

    derf = Derf(d_model=d_model, init_alpha=0.4, init_shift=0.5)
    x = torch.randn(batch_size, seq_len, d_model)

    out = derf(x)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_derf_forward_computation():
    """Verify forward computation matches the exact mathematical formula:
    y = weight * erf(alpha * x + shift) + bias
    """
    d_model = 4
    init_alpha, init_shift = 0.4, 0.5

    derf = Derf(d_model=d_model, init_alpha=init_alpha, init_shift=init_shift)
    x = torch.tensor([[1.0, -1.0, 0.5, 0.0]])

    # Expected analytical result
    expected = 1.0 * torch.erf(init_alpha * x + init_shift) + 0.0
    out = derf(x)

    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_derf_backward_pass_gradients():
    """Verify that gradients flow back to both inputs AND all 4 learnable parameters
    (weight, bias, alpha, shift) without returning None or NaN"""
    d_model = 128
    derf = Derf(d_model=d_model, init_alpha=0.4, init_shift=0.5)

    x = torch.randn(4, 16, d_model, requires_grad=True)

    out = derf(x)
    loss = out.sum()
    loss.backward()

    # check input gradient
    assert x.grad is not None, "Gradient for input x should not be None."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."

    # check parameter gradients
    for name, param in derf.named_parameters():
        assert param.grad is not None, f"Gradient for parameter '{name}' should not be None."
        assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaN values."
        assert (param.grad != 0).any(), f"Gradient for '{name}' is completely zero."
