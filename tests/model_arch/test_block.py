import torch
import pytest

from model.transformer import Block, TinyConfig, precompute_rope_embeddings


@pytest.fixture
def config():
    return TinyConfig(
        d_model=64,
        max_seq_len=128,
        max_batch_size=8,
        n_heads=4,
        n_layers=2,
        kv_latent_dim=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        hidden_dim=128,
        init_alpha=0.4,
        init_shift=0.5,
    )


def test_block_forward_shape_and_dtype(config):
    """Verify that the block preserves input tensor shape and data type"""
    bsz, seq_len = 2, 16
    block = Block(config)

    x = torch.randn(bsz, seq_len, config.d_model)
    rope_embs = precompute_rope_embeddings(config.qk_rope_head_dim, seq_len, config.theta)

    out = block(x, rope_embs)

    assert out.shape == (bsz, seq_len, config.d_model), f"Expected {(bsz, seq_len, config.d_model)}, got {out.shape}"
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_block_gradients_flow_to_residual_alphas(config):
    """Verify that gradients properly propagate to alpha1, alpha2, and input x"""
    bsz, seq_len = 2, 8
    block = Block(config)

    x = torch.randn(bsz, seq_len, config.d_model, requires_grad=True)
    rope_embs = precompute_rope_embeddings(config.qk_rope_head_dim, seq_len, config.theta)

    out = block(x, rope_embs)
    loss = out.pow(2).sum()
    loss.backward()

    # check input gradient
    assert x.grad is not None, "Gradient for input x should not be None."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaNs."

    # check learnable residual parameters
    assert block.alpha1.grad is not None, "Gradient for alpha1 should not be None."
    assert block.alpha2.grad is not None, "Gradient for alpha2 should not be None."
    assert not torch.isnan(block.alpha1.grad).any(), "alpha1 gradient contains NaNs."
    assert not torch.isnan(block.alpha2.grad).any(), "alpha2 gradient contains NaNs."
    assert (block.alpha1.grad != 0).any(), "alpha1 gradient is completely zero."
    assert (block.alpha2.grad != 0).any(), "alpha2 gradient is completely zero."


def test_block_identity_when_alphas_zero(config):
    """Verify that when alpha1 and alpha2 are set to 0, the block acts as an identity function"""
    block = Block(config)

    with torch.no_grad():
        block.alpha1.zero_()
        block.alpha2.zero_()

    x = torch.randn(2, 8, config.d_model)
    rope_embs = precompute_rope_embeddings(config.qk_rope_head_dim, 8, config.theta)

    out = block(x, rope_embs)

    assert torch.allclose(out, x, atol=1e-6), "Block output did not match input when residual scaling is zero."
