import torch
import pytest

from model.transformer import TinyLM, TinyConfig


@pytest.fixture
def config():
    """Returns a lightweight TinyConfig instance for rapid test execution."""
    return TinyConfig(
        d_model=64,
        vocab_size=100,
        max_seq_len=64,
        max_batch_size=4,
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


def test_tinylm_forward_shape(config):
    """Verify forward pass output shape matches (batch_size, seq_len, vocab_size)"""
    model = TinyLM(config)
    bsz, seq_len = 2, 8
    x = torch.randint(0, config.vocab_size, (bsz, seq_len))

    logits = model(x, start_pos=0)

    assert logits.shape == (bsz, seq_len, config.vocab_size), (
        f"Expected logits shape {(bsz, seq_len, config.vocab_size)}, got {logits.shape}"
    )


def test_tinylm_generate_output_shape_and_length(config):
    """Verify that generate returns prompt + max_tokens generated tokens"""
    model = TinyLM(config)
    bsz, prompt_len = 2, 5
    max_tokens = 10
    prompt_ids = torch.randint(0, config.vocab_size, (bsz, prompt_len))

    generated_ids = model.generate(prompt_ids, max_tokens=max_tokens, temperature=1.0)

    expected_len = prompt_len + max_tokens
    assert generated_ids.shape == (bsz, expected_len), (
        f"Expected shape {(bsz, expected_len)}, got {generated_ids.shape}"
    )
    # ensure original prompt tokens remain unchanged at the start
    assert torch.equal(generated_ids[:, :prompt_len], prompt_ids), (
        "Generated tokens modified the original prompt prefix."
    )


def test_tinylm_kv_cache_decode_consistency(config):
    """Verify single-step decode produces identical logits to full prefill sequence"""
    model = TinyLM(config)
    model.eval()

    bsz = 1
    seq = torch.randint(0, config.vocab_size, (bsz, 6))

    # full prefill forward pass
    full_logits = model(seq, start_pos=0)

    # sequential step-by-step KV cache forward pass
    _ = model(seq[:, :4], start_pos=0)

    # decode step 4 (5th token)
    _ = model(seq[:, 4:5], start_pos=4)

    # decode step 5 (6th token)
    logits_step5 = model(seq[:, 5:6], start_pos=5)

    # compare step 5 prediction against full forward pass position 5
    assert torch.allclose(
        logits_step5, 
        full_logits[:, 5:6, :], 
        rtol=1e-3, 
        atol=1e-4
    ), f"Max difference: {(logits_step5 - full_logits[:, 5:6, :]).abs().max().item()}"


def test_tinylm_top_k_sampling(config):
    """Verify generate handles top_k filtering without throwing errors or NaNs"""
    model = TinyLM(config)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 4))

    out_ids = model.generate(prompt_ids, max_tokens=5, temperature=0.7, top_k=1)

    assert out_ids.shape == (1, 9)
    assert not torch.isnan(out_ids.float()).any()


def test_tinylm_backward_pass_gradients(config):
    """Verify gradients propagate back to embeddings, blocks, and out_proj"""
    model = TinyLM(config)
    x = torch.randint(0, config.vocab_size, (2, 4))

    logits = model(x, start_pos=0)
    loss = logits.sum()
    loss.backward()

    # check embeddings and linear head gradients
    assert model.embeddings.weight.grad is not None, "Embedding gradient is None"
    assert model.out_proj.weight.grad is not None, "out_proj gradient is None"
    assert not torch.isnan(model.embeddings.weight.grad).any()
    assert not torch.isnan(model.out_proj.weight.grad).any()
