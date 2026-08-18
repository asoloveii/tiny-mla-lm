from dataclasses import dataclass

import torch
import pytest

from model.transformer import MLA, TinyConfig, precompute_rope_embeddings


@pytest.fixture
def config():
    return TinyConfig(
        d_model=64,
        n_heads=4,
        kv_latent_dim=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_batch_size=8,
        max_seq_len=128,
    )


def test_mla_forward_shape_and_dtype(config):
    """Verify that the forward pass preserves the input sequence shape [batch, seq_len, d_model]"""
    bsz, seq_len = 2, 16
    mla = MLA(config)

    x = torch.randn(bsz, seq_len, config.d_model)
    rope_embs = precompute_rope_embeddings(config.qk_rope_head_dim, config.max_seq_len, config.theta)

    out = mla(x, rope_embs, start_pos=0)

    assert out.shape == (bsz, seq_len, config.d_model), f"Expected {(bsz, seq_len, config.d_model)}, got {out.shape}"
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_mla_kv_cache_population(config):
    """Verify that the compressed KV cache buffers are updated correctly over time"""
    bsz = 2
    mla = MLA(config)

    # prefill Step (start_pos = 0, seq_len = 4)
    prefill_len = 4
    x_prefill = torch.randn(bsz, prefill_len, config.d_model)
    rope_prefill = precompute_rope_embeddings(config.qk_rope_head_dim, prefill_len, config.theta)

    _ = mla(x_prefill, rope_prefill, start_pos=0)

    # check cache slice is non-zero after prefill
    assert not torch.all(mla.cache_c_kv[:bsz, :prefill_len] == 0)
    assert not torch.all(mla.cache_k_rope[:bsz, :prefill_len] == 0)

    # check remaining cache memory is untouched
    assert torch.all(mla.cache_c_kv[:bsz, prefill_len:] == 0)
    assert torch.all(mla.cache_k_rope[:bsz, prefill_len:] == 0)


def test_mla_backward_pass_gradients(config):
    """Verify gradients flow properly back to inputs and all projection weights without returning NaNs"""
    bsz, seq_len = 2, 8
    mla = MLA(config)

    x = torch.randn(bsz, seq_len, config.d_model, requires_grad=True)
    rope_embs = precompute_rope_embeddings(config.qk_rope_head_dim, config.max_seq_len, config.theta)

    out = mla(x, rope_embs, start_pos=0)
    loss = out.sum()
    loss.backward()

    # check input gradient
    assert x.grad is not None, "Gradient for input x should not be None."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."

    # check parameter gradients
    for name, param in mla.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for parameter '{name}' should not be None."
            assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaN values."
            assert (param.grad != 0).any(), f"Gradient for '{name}' is completely zero."
