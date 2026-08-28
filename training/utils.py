import os 
import re
import glob
from omegaconf import OmegaConf
import torch
import torch.distributed as dist


def setup_ddp():
    """Initializes DDP process group using environment variables set by torchrun"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def get_latest_checkpoint(ckpt_dir):
    """Finds the checkpoint with the highest step number in ckpt_dir"""
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt"))
    if not ckpts:
        return None
    # sort by step number extracted from filename
    return max(ckpts, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively removes DDP and torch.compile wrappers"""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def load_checkpoint(cfg, model, optimizer, lr_scheduler, scaler, device, rank):
    is_sft = cfg.get("is_sft", False)
    sft_ckpt_dir = cfg.training.get("ckpt_dir", "./ckpts/sft")
    latest_sft_ckpt = os.path.join(sft_ckpt_dir, "checkpoint_latest.pt")

    # resume an ongoing sft run if a checkpoint already exists in ckpt_dir
    if is_sft and os.path.exists(latest_sft_ckpt) and cfg.training.resume_from == "auto":
        if rank == 0:
            print(f"Resuming existing SFT run from {latest_sft_ckpt}")
        checkpoint = torch.load(latest_sft_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["step"], checkpoint["epoch"]

    # cold start sft using pretrained weights
    if is_sft:
        pretrained_path = cfg.training.get("pretrained_ckpt_path", None)
        assert pretrained_path and os.path.exists(pretrained_path), (
            f"Pretrained checkpoint not found at: {pretrained_path}"
        )
        if rank == 0:
            print(f"Loading pretrained weights for SFT from {pretrained_path}...")
        
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # extract model weights state dict
        model_weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(model_weights)
        
        # reset training counters and optimizer states for a fresh SFT start
        return 0, 0

    # pretraining resume or cold start from scratch
    pretrain_latest = os.path.join(cfg.training.ckpt_dir, "checkpoint_latest.pt")
    if os.path.exists(pretrain_latest) and cfg.training.resume_from == "auto":
        if rank == 0:
            print(f"Resuming pretraining from {pretrain_latest}")
        checkpoint = torch.load(pretrain_latest, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["step"], checkpoint["epoch"]

    # cold start pretraining from scratch
    return 0, 0


def save_checkpoint(cfg, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler, 
                    scaler, 
                    step: int, 
                    epoch: int, 
                    rank: int = 0):
    """Saves un-wrapped model and training state on rank 0 with DDP barrier"""
    if rank == 0:
        os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
        raw_model = unwrap_model(model)

        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        ckpt_path = os.path.join(cfg.training.ckpt_dir, f"ckpt_step_{step+1:05d}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"\n[Rank 0] Saved checkpoint to {ckpt_path}")

    if dist.is_initialized():
        dist.barrier()


def extract_answer(text: str) -> str:
    """Extract final numeric answer from generated CoT string ('#### 42')"""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    # fallback to finding the last standalone number in text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""
