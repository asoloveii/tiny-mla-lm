import os
import argparse
import torch
from transformers import PretrainedConfig, AutoTokenizer


class TinyLMConfig(PretrainedConfig):
    """Hugging Face compatible configuration for TinyLM"""
    model_type = "tiny_lm"

    def __init__(self,
                d_model: int = 768,
                vocab_size: int = 50256,
                max_seq_len: int = 2048,
                n_heads: int = 12,
                n_layers: int = 12,
                kv_latent_dim: int = 256,
                qk_rope_head_dim: int = 32,
                qk_nope_head_dim: int = 64,
                v_head_dim: int = 64,
                hidden_dim: int = 2048,
                init_alpha: float = 0.4,
                init_shift: float = 0.5,
                theta: float = 10000.0,
                pad_token_id: int = 50256,
                bos_token_id: int = 50256,
                eos_token_id: int = 50256,
                **kwargs,):
        super().__init__(pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        **kwargs,)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kv_latent_dim = kv_latent_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.hidden_dim = hidden_dim
        self.init_alpha = init_alpha
        self.init_shift = init_shift
        self.theta = theta


def clean_state_dict(state_dict: dict) -> dict:
    """Removes DDP wrapper ('_orig_mod.', 'module.') prefixes and compiled wrappers"""
    cleaned = {}
    for key, val in state_dict.items():
        new_key = key
        # remove torch.compile prefix
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        # remove DDP module prefix
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = val
    return cleaned



def export_checkpoint(checkpoint_path: str,
                      output_dir: str,
                      model_config_path: str = "./configs/model_config.yaml",):
    """Converts a local raw checkpoint file into HF format for vLLM usage"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw checkpoint from: {checkpoint_path}")

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # extract state dict 
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
    else:
        raw_state_dict = ckpt

    cleaned_weights = clean_state_dict(raw_state_dict)

    # instantiate hf config
    print("Building Hugging Face configuration...")
    if os.path.exists(model_config_path):
        import yaml
        with open(model_config_path, "r") as f:
            cfg_dict = yaml.safe_load(f).get("model", {})
        hf_config = TinyLMConfig(**cfg_dict)
    else:
        print(f"Warning: {model_config_path} not found. Falling back to default TinyLMConfig.")
        hf_config = TinyLMConfig()

    model_bin_path = os.path.join(output_dir, "pytorch_model.bin")
    print(f"Saving model weights to {model_bin_path}...")
    torch.save(cleaned_weights, model_bin_path)

    # save hf config.json
    hf_config.save_pretrained(output_dir)

    # export standard gpt2 tokenizer 
    print("Initializing and saving HF GPT-2 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    print(f"\nSuccessfully exported HF-style model directory to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to HF directory for vLLM.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/model_latest.pt")
    parser.add_argument("--output_dir", type=str, default="exported_hf_model")
    parser.add_argument("--model_config", type=str, default="./configs/model_config.yaml")

    args = parser.parse_args()
    export_checkpoint(checkpoint_path=args.checkpoint_path,
                      output_dir=args.output_dir,
                      model_config_path=args.model_config,)
