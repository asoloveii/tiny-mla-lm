import yaml 
from dataclasses import dataclass

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


@dataclass
class TinyConfig:
    d_model: int = 576
    vocab_size: int = 50256
    max_seq_len: int = 2048
    max_batch_size: int = 512
    n_heads: int = 12
    n_layers: int = 10
    
    kv_latent_dim: int = 192
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64
    
    hidden_dim: int = 1536
    init_alpha: float = 0.4
    init_shift: float = 0.5
    theta: float = 10000.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TinyConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)["model"]
        return cls(**config_dict)


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


class MLA(nn.Module):
    """Multi-Head Latent Attention (MLA) with Latent KV Caching and Exclusive Attention (XSA).
    
    Args:
        args (TinyConfig): Configuration object containing model hyper-parameters
    """

    def __init__(self, args: TinyConfig):
        super().__init__()

        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.kv_latent_dim = args.kv_latent_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        # query projection
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.qk_head_dim, bias=False)
        # compressed kv projections onto a latent space + keys for rope
        self.down_kv_proj = nn.Linear(self.d_model, self.kv_latent_dim+self.qk_rope_head_dim, bias=False)
        self.up_kv_proj = nn.Linear(self.kv_latent_dim, self.n_heads*(self.qk_nope_head_dim + self.v_head_dim), bias=False)
        # output projections
        self.out_proj = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)

        self.register_buffer(
            "cache_c_kv",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_latent_dim),
            persistent=False,
        )
        self.register_buffer(
            "cache_k_rope",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim),
            persistent=False,
        )

    def forward(self, 
                x: torch.Tensor, 
                rope_embs: torch.Tensor, 
                start_pos: int = 0,) -> torch.Tensor:
        bsz, seq, dim = x.shape

        # queries projections
        q = self.q_proj(x).reshape(bsz, seq, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary_pos_emb(q_rope, rope_embs)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        # latent space + keys for pos embs
        C_kv = self.down_kv_proj(x)
        C_kv, k_rope = torch.split(C_kv, [self.kv_latent_dim, self.qk_rope_head_dim], dim=-1)
        k_rope = apply_rotary_pos_emb(k_rope.unsqueeze(2), rope_embs).squeeze(2)

        with torch.no_grad():
            # store in pre-allocated kv caches
            self.cache_c_kv[:bsz, start_pos : start_pos + seq] = C_kv
            self.cache_k_rope[:bsz, start_pos : start_pos + seq] = k_rope

        if start_pos > 0:
            # retrieve cached sequences up to current length
            cached_c_kv = torch.cat([self.cache_c_kv[:bsz, :start_pos], C_kv], dim=1)
            cached_k_rope = torch.cat([self.cache_k_rope[:bsz, :start_pos], k_rope], dim=1)
        else:
            cached_c_kv = C_kv
            cached_k_rope = k_rope
            
        total_seq = start_pos + seq

        # up projections for keys nope and values
        kv = self.up_kv_proj(cached_c_kv).view(bsz, total_seq, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, cached_k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # flash-attention
        if start_pos == 0 and seq == total_seq:
            attn_outs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            i = torch.arange(seq, device=x.device).unsqueeze(1) + start_pos
            j = torch.arange(total_seq, device=x.device).unsqueeze(0)
            causal_mask = j <= i
            attn_outs = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

        attn_outs = attn_outs.transpose(1, 2)

        # xsa (orthogonality to self-value direction)
        v_local = v.transpose(1, 2)[:, start_pos:total_seq, :, :]
        v_norm = F.normalize(v_local, dim=-1)
        z = attn_outs - (attn_outs * v_norm).sum(dim=-1, keepdim=True) * v_norm
        z = z.contiguous().view(bsz, seq, dim)

        return self.out_proj(z)


class Block(nn.Module):
    """Transformer block with normalization-free residual paths and learnable scaling.

    Combines Multi-Head Latent Attention (MLA) and SwiGLU feed-forward layer 
    using learnable scalar residual multipliers (ReZero/LayerScale strategy) 
    and Derf normalization.

    Args:
        args (TinyConfig): Configuration object containing model hyper-parameters
    """

    def __init__(self, args):
        super().__init__()

        self.alpha1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.norm1 = Derf(args.d_model, args.init_alpha, args.init_shift)
        self.mla = MLA(args)

        self.norm2 = Derf(args.d_model, args.init_alpha, args.init_shift)
        self.swiglu = SwiGLU(args.d_model, args.hidden_dim)

    def forward(self, 
                x: torch.Tensor, 
                rope_embs: torch.Tensor, 
                start_pos: int = 0) -> torch.Tensor:
        x = x + self.alpha1 * self.mla(self.norm1(x), rope_embs, start_pos)
        x = x + self.alpha2 * self.swiglu(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """TinyLM Causal Language Model.

    A decoder-only architecture utilizing Multi-Head Latent Attention (MLA),
    gated SwiGLU activations, dynamic error function (Derf) normalization, 
    and learnable residual scaling.

    Args:
        args (TinyConfig): Configuration parameters for the model layout and hyperparameters
    """

    def __init__(self, args: TinyConfig):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.d_model)

        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])

        self.norm = nn.LayerNorm(args.d_model)
        self.out_proj = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # cache precomputed rotary pos embeddings for attention
        self.register_buffer(
            "rope_embs",
            precompute_rope_embeddings(args.qk_rope_head_dim, args.max_seq_len, args.theta),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        bsz, seq = x.shape

        x = self.embeddings(x)
        rope_embs = self.rope_embs[start_pos : start_pos + seq, :]

        for block in self.blocks:
            x = block(x, rope_embs, start_pos=start_pos)

        return self.out_proj(self.norm(x))

    @torch.inference_mode
    def generate(self,
                 ids: torch.Tensor,
                 max_tokens: int,
                 temperature: float = 1.0,
                 top_k: int = -1):

        self.eval()
        bsz, prompt_len = ids.shape
        start_pos = 0

        # prefill prompt sequence into KV caches
        logits = self(ids, start_pos=start_pos)  # (bsz, prompt_len, vocab_size)
        start_pos += prompt_len
        # select logits for the last token position
        next_token_logits = logits[:, -1, :]

        for _ in range(max_tokens):
            # apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # apply top-k filtering
            if top_k > 0:
                top_logits, _ = torch.topk(next_token_logits, k=top_k)
                next_token_logits[next_token_logits < top_logits[:, [-1]]] = float("-inf")

            # sample next token from probability distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (bsz, 1)
            # append generated token to complete history
            ids = torch.cat([ids, next_token], dim=1)
            #decode single step using incremented start_pos
            logits = self(next_token, start_pos=start_pos)  # (bsz, 1, vocab_size)
            next_token_logits = logits[:, -1, :]
            start_pos += 1

        return ids


if __name__ == "__main__":
    model = TinyLM(TinyConfig())
    print(f"number of params: {sum([p.numel() for p in model.parameters()]):,}")
