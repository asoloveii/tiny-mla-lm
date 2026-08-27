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


def load_checkpoint(cfg, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler, 
                    scaler, 
                    device: torch.device, 
                    rank: int = 0) -> tuple[int, int]:
    """Loads state dicts before DDP wrapping. Returns (start_step, start_epoch)"""
    resume_path = getattr(cfg.training, "resume_from", None)
    if resume_path == "auto":
        resume_path = get_latest_checkpoint(cfg.training.ckpt_dir)

    if not resume_path or not os.path.exists(resume_path):
        return 0, 0

    if rank == 0:
        print(f"\n[Checkpoint] Resuming from: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)

    # remove DDP/compile key prefixes if loading legacy state dicts
    state_dict = checkpoint["model"]
    clean_state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v 
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if "scaler" in checkpoint and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])

    start_step = checkpoint["step"] + 1
    start_epoch = checkpoint.get("epoch", 0)

    if rank == 0:
        print(f"[Checkpoint] Success. Resuming step {start_step}, epoch {start_epoch}\n")

    return start_step, start_epoch


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
