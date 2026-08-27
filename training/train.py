import os
import hydra
import wandb
import tqdm
import torch 
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf

from model import TinyConfig, TinyLM
from data.prepare import get_dataloaders
from utils import *


@hydra.main(version_base=None, config_path="configs", config_name="pretrain_config")
def train(cfg: DictConfig):
    rank, local_rank, world_size, device = setup_ddp()

    if rank == 0:
        wandb.init(project="tiny-mla-lm", config=OmegaConf.to_container(cfg, resolve=True)) 

    torch.manual_seed(cfg.training.seed + rank)

    model_config = TinyConfig.from_yaml("./configs/model_config.yml")

    train_loader, val_loader, train_sampler = get_dataloaders(
        data_dir=cfg.data.data_dir,
        seq_len=model_config.max_seq_len,
        batch_size=cfg.training.device_batch_size,
    )

    pt_dtype = getattr(torch, cfg.training.mixed_precision)

    model = TinyLM(model_config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
    s1 = optim.lr_scheduler.LinearLR(optimizer, total_iters=cfg.scheduler.warmup_steps)
    s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_steps-cfg.scheduler.warmup_steps)
    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[cfg.scheduler.warmup_steps])

    scaler = torch.GradScaler(device, enabled=(pt_dtype == torch.float16))

    start_step, start_epoch = load_checkpoint(
        cfg, model, optimizer, lr_scheduler, scaler, device, rank
    )

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    step, epoch = start_step, start_epoch

    while step < cfg.training.max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for xs, ys in pbar:
            if step >= cfg.training.max_steps:
                break

            loss = train_one_batch(model, xs, ys, step, optimizer, scaler, cfg, device, pt_dtype)

            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"train/lr": current_lr, "train/loss": loss}, step=step)

            if (step + 1) % cfg.training.val_every == 0:
                val_loss = validate(model, val_loader, cfg, device, pt_dtype)
                if rank == 0:
                    wandb.log({"val/loss": val_loss}, step=step)

            if (step + 1) % cfg.training.ckpt_every == 0 or (step + 1) == cfg.training.max_steps:
                save_checkpoint(cfg, model, optimizer, lr_scheduler, scaler, step, epoch, rank)

            lr_scheduler.step()
            step += 1

        epoch += 1

    if rank == 0:
        wandb.finish()

    cleanup_ddp()


def train_one_batch(model, 
                    xs, 
                    ys,
                    step, 
                    optimizer,
                    scaler,
                    cfg,
                    device,
                    pt_dtype):
    model.train()
    xs, ys = xs.to(device), ys.to(device)

    is_accumulating = (step + 1) % cfg.training.gradient_accumulation_steps != 0
    context = model.no_sync() if is_accumulating else torch.enable_grad()

    with context:
        with torch.autocast(device_type=device.type, dtype=pt_dtype):
            logits = model(xs)
            loss = F.cross_entropy(logits.flatten(0, 1), ys.flatten(), ignore_index=-100)
            accum_loss = loss / cfg.training.gradient_accumulation_steps

        scaler.scale(accum_loss).backward()

    if not is_accumulating:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # average loss across all GPU ranks 
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    return loss.item()


def validate(model, val_loader, cfg, device, pt_dtype):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0.0, device=device)

    with torch.inference_mode():
        for xs, ys in val_loader:
            xs, ys = xs.to(device), ys.to(device)
            with torch.autocast(device_type=device.type, dtype=pt_dtype):
                logits = model(xs)
                loss = F.cross_entropy(logits.flatten(0, 1), ys.flatten(), ignore_index=-100)
                total_loss += loss
                total_batches += 1.0

    # all-reduce validation totals across all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    return (total_loss / total_batches).item()


if __name__ == "__main__":
    train()