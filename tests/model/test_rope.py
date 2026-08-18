import torch 
import pytest 
from model.transformer import apply_rotary_pos_emb, precompute_rope_embeddings


def test_rope_output_shapes():
    """Verify that outputs match expected shapes across standard transformer tensor layouts"""
    head_dim = 64
    max_seq_len = 128
    theta = 10000.0
    
    # precomputed embedding table shape (max_seq_len, head_dim)
    emb = precompute_rope_embeddings(head_dim, max_seq_len, theta)
    assert emb.shape == (max_seq_len, head_dim), f"Expected shape {(max_seq_len, head_dim)}, got {emb.shape}"

    # input tensor shape: (batch_size, seq_len, n_heads, head_dim)
    batch_size, seq_len, n_heads = 2, 32, 8
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    out = apply_rotary_pos_emb(x, emb)
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}"


def test_rope_identity_at_position_zero():
    """At position 0, angle m*theta = 0, so cos(0) = 1 and sin(0) = 0.
    The first token's embedding should remain mathematically unchanged"""
    head_dim = 32
    max_seq_len = 16
    theta = 10000.0

    emb = precompute_rope_embeddings(head_dim, max_seq_len, theta)
    
    # batch size 1, seq_len 1 (position 0), 1 head, head_dim
    x_pos0 = torch.randn(1, 1, 1, head_dim)
    out_pos0 = apply_rotary_pos_emb(x_pos0, emb)

    torch.testing.assert_close(out_pos0, x_pos0, rtol=1e-5, atol=1e-5)


def test_rope_norm_preservation():
    """Rotations preserve vector magnitude (Euclidean norm / L2 norm).
    The L2 norm of each head vector before and after RoPE must be equal."""
    head_dim = 64
    seq_len = 16
    
    emb = precompute_rope_embeddings(head_dim, seq_len, 10000.0)
    x = torch.randn(2, seq_len, 4, head_dim)
    out = apply_rotary_pos_emb(x, emb)

    norm_x = torch.norm(x, dim=-1)
    norm_out = torch.norm(out, dim=-1)

    torch.testing.assert_close(norm_out, norm_x, rtol=1e-5, atol=1e-5)


def test_rope_relative_position_property():
    """The key mathematical property of RoPE: dot product <RoPE(q, m), RoPE(k, n)>
    depends ONLY on the relative distance (m - n)."""
    head_dim = 64
    max_seq_len = 100
    emb = precompute_rope_embeddings(head_dim, max_seq_len, 10000.0)

    # random query and key single vectors (shape: [1, 1, 1, head_dim])
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # test pair 1: positions m=10, n=5 (distance = 5)
    q_pos10 = apply_rotary_pos_emb(q, emb[10:11])
    k_pos5 = apply_rotary_pos_emb(k, emb[5:6])
    dot_product_1 = (q_pos10 * k_pos5).sum()

    # test pair 2: shifted positions m=40, n=35 (same distance = 5)
    q_pos40 = apply_rotary_pos_emb(q, emb[40:41])
    k_pos35 = apply_rotary_pos_emb(k, emb[35:36])
    dot_product_2 = (q_pos40 * k_pos35).sum()

    torch.testing.assert_close(dot_product_1, dot_product_2, rtol=1e-4, atol=1e-4)
