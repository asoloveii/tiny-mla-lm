import torch 
import torch.nn as nn
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


class Derf(nn.Module):
    """Dynamic erf layer.
    First introduced in Stronger Normalization-Free Transformers, replaces usual
    LayerNorm/RMSNorm layer by erf(x) function with learnable alpha and shift parameters:
    y = erf(alpha * x + shift)
    
    Args:
        d_model (int): embedding dimension
        init_alpha (float): initial value for alpha parameter
        init_shift (float): initial value for shift parameter
    """

    def __init__(self, d_model: int, init_alpha: float, init_shift: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(d_model), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones(1) * init_alpha, requires_grad=True)
        self.shift = nn.Parameter(torch.ones(1) * init_shift, requires_grad=True)

    def forward(self, x: torch.Tensor):
        return self.weight * torch.erf(self.alpha * x + self.shift) + self.bias


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit.
    Projects input embedding into a higher-dimensional input/value gate and 
    a swish gate (projections+silu activation), then takes its elemenwise product
    and projects back onto an original dimension.
    
    Args:
        d_model (int): embedding dimension
        hidden_dim (int): hidden embedding dimension
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        # gated up projections
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.content_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # output down projections
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.content_proj(x))
