import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model import TinyConfig, TinyLM
from data.prepare.gsm8k import get_gsm8k_dataloaders
from utils import *


def compute_rewards(completions: list[str], ground_truths: list[str]) -> torch.Tensor:
    """computes a binary/format reward vector across group outputs."""
    rewards = []
    for comp, target in zip(completions, ground_truths):
        extracted = extract_answer(comp)
        target_ans = extract_answer(target) or target.strip()
        
        r = 0.0
        if "####" in comp:
            r += 0.1
        if extracted and extracted == target_ans:
            r += 1.0
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


@hydra.main(version_base=None, config_path="configs", config_name="grpo_config")
def main(cfg: DictConfig):
    rank, local_rank, world_size, device = setup_ddp()

    if rank == 0:
        wandb.init(project="tiny-mla-lm", config=OmegaConf.to_container(cfg, resolve=True)) 

    torch.manual_seed(cfg.training.seed + rank)
    model_config = TinyConfig.from_yaml("./configs/model_config.yaml")
    pt_dtype = getattr(torch, cfg.training.mixed_precision)

    train_loader, val_loader, train_sampler = get_gsm8k_dataloaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.device_batch_size,
        num_workers=cfg.data.num_workers,
    )

    # actor (policy model)
    model = TinyLM(model_config).to(device)

    start_step, start_epoch = load_checkpoint(
        cfg, model, optimizer, scheduler, scaler, device, rank
    )

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # frozen ref model for kl penalty 
    ref_model = TinyLM(model_config).to(device)
    ref_model.load_state_dict(model.module.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
    s1 = optim.lr_scheduler.LinearLR(optimizer, total_iters=cfg.scheduler.warmup_steps)
    s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.max_steps - cfg.scheduler.warmup_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[s1, s2], milestones=[cfg.scheduler.warmup_steps]
    )

    scaler = torch.GradScaler(device, enabled=(pt_dtype == torch.float16))

    step, epoch = start_step, start_epoch

    while step < cfg.training.max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"GRPO Epoch {epoch}", disable=(rank != 0))
        for prompts, targets in pbar:
            if step >= cfg.training.max_steps:
                break

            loss, mean_reward = grpo_step(
                model=model,
                ref_model=ref_model,
                prompts=prompts,
                targets=targets,
                step=step,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=device,
                pt_dtype=pt_dtype,
            )

            if (step + 1) % cfg.training.ckpt_every == 0 or (step + 1) == cfg.training.max_steps:
                save_checkpoint(cfg, model, optimizer, scheduler, scaler, step, epoch, rank)

            if rank == 0:
                wandb.log({
                    "train/loss": loss,
                    "train/mean_reward": mean_reward,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=step)

            scheduler.step()
            step += 1

        epoch += 1

    if rank == 0:
        wandb.finish()

    cleanup_ddp()


def grpo_step(model, 
              ref_model, 
              prompts, 
              targets, 
              step, 
              optimizer, 
              scaler, 
              cfg, 
              device, 
              pt_dtype):
    """Executes group rollouts, computes group-relative advantage normalization, and optimizes"""
    model.train()
    group_size = cfg.grpo.group_size
    kl_coeff = cfg.grpo.kl_coeff
    clip_eps = cfg.grpo.clip_eps

    # rollout: repeat prompts G times to form trajectory groups, 
    # shape (B, prompt_len) -> (B * G, prompt_len)
    prompt_tokens = prompts.repeat_interleave(group_size, dim=0).to(device)
    extended_targets = [target for target in targets for _ in range(group_size)]

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=pt_dtype):
            seq_sequences = model.module.generate(
                prompt_tokens,
                max_new_tokens=cfg.grpo.max_gen_len,
                temperature=cfg.grpo.temperature,
            )

    prompt_len = prompt_tokens.shape[1]
    gen_tokens = seq_sequences[:, prompt_len:]  # (B * G, gen_len)
    
    # decode string completions for rule-based reward evaluator
    decoded_completions = [
        model.module.tokenizer.decode(g.tolist()) for g in gen_tokens
    ]
    
    # compute group-normalized rewards and advantages
    raw_rewards = compute_rewards(decoded_completions, extended_targets).to(device) # (B * G)
    rewards_grouped = raw_rewards.view(-1, group_size)

    # standardize rewards per group (A = (R - mean) / std)
    mean_r = rewards_grouped.mean(dim=-1, keepdim=True)
    std_r = rewards_grouped.std(dim=-1, keepdim=True) + 1e-8
    advantages = ((rewards_grouped - mean_r) / std_r).view(-1) 

    # forward on actor & ref models
    is_accumulating = (step + 1) % cfg.training.gradient_accumulation_steps != 0
    context = model.no_sync() if is_accumulating else torch.enable_grad()

    with context:
        with torch.autocast(device_type=device.type, dtype=pt_dtype):
            # compute current policy log probs
            logits = model(seq_sequences)
            log_probs = F.log_softmax(logits[:, prompt_len - 1 : -1, :], dim=-1)
            token_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

            # compute frozen reference policy log probs
            with torch.no_grad():
                ref_logits = ref_model(seq_sequences)
                ref_log_probs = F.log_softmax(ref_logits[:, prompt_len - 1 : -1, :], dim=-1)
                ref_token_log_probs = ref_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

            # ratio and kl divergence
            ratio = torch.exp(token_log_probs - ref_token_log_probs.detach())
            kl_div = torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1

            # ppo-style clipped Loss + kl div penalty
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.unsqueeze(-1)
            
            policy_loss = -torch.min(surr1, surr2) + kl_coeff * kl_div
            loss = policy_loss.mean() / cfg.training.gradient_accumulation_steps

        scaler.scale(loss).backward()

    if not is_accumulating:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    dist.all_reduce(raw_rewards, op=dist.ReduceOp.SUM)
    global_mean_reward = (raw_rewards.sum() / len(raw_rewards)).item()

    return loss.item() * cfg.training.gradient_accumulation_steps, global_mean_reward


if __name__ == "__main__":
    main()
