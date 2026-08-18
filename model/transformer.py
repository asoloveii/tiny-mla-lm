import torch 
import torch.nn.functional as F


def precompute_rope_embeddings(head_dim: int, max_seq_len: int, theta: float):
    # calculate theta_i = 1 / (base ^ (2i/d))
    inv_freq = 1 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # generate position indices m
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: m * theta
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # repeat for cos and sin
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb


def apply_rotary_pos_emb(x, rope_emb):
    # slice embeddings to sequence length
    seq_len = x.shape[1]
    rope_emb_sliced = rope_emb[:seq_len, :]

    # reshape for broadcasting
    emb = rope_emb_sliced.unsqueeze(0).unsqueeze(2)

    cos, sin = emb.cos(), emb.sin()

    # create partner (-y, x)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos + x_partner * sin).type_as(x)
    
