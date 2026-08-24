import os
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset


DEFAULT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it step by step. The assistant puts the final answer at the end of the response "
    "after '#### '.\n"
)


class GSM8KDataset(Dataset):

    def __init__(self,
                split: str = "train",
                encoding_name: str = "gpt2",
                max_prompt_len: int = 512,
                system_prompt: str = DEFAULT_SYSTEM_PROMPT,):
        super().__init__()

        self.max_prompt_len = max_prompt_len
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.pad_token_id = self.tokenizer.eot_token

        hf_ds = load_dataset("openai/gsm8k", "main", split=split)
        
        self.samples = []
        for item in hf_ds:
            formatted_prompt = f"{system_prompt}\nUser: {item['question']}\nAssistant:"
            self.samples.append({
                "prompt_text": formatted_prompt,
                "target_answer": item["answer"],  
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        input_ids = self.tokenizer.encode_ordinary(sample["prompt_text"])

        # truncate (keep trailing context if prompt exceeds max_prompt_len)
        if len(input_ids) > self.max_prompt_len:
            input_ids = input_ids[-self.max_prompt_len:]

        # manual left-padding
        pad_len = self.max_prompt_len - len(input_ids)
        padded_input_ids = [self.pad_token_id] * pad_len + input_ids

        return {
            "prompt_tokens": torch.tensor(padded_input_ids, dtype=torch.long),
            "target_text": sample["target_answer"],
        }

def collate_gsm8k_fn(batch: list[dict]) -> tuple[torch.Tensor, list[str]]:
    """Custom collate function separating prompt tensors from target strings"""
    prompt_tokens = torch.stack([b["prompt_tokens"] for b in batch])
    targets = [b["target_text"] for b in batch]
    return prompt_tokens, targets


def get_gsm8k_dataloaders(data_dir: str = "./data/processed/gsm8k",
                          batch_size: int = 2,
                          num_workers: int = 2,
                          max_prompt_len: int = 512,
                          tokenizer_name: str = "gpt2",) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Builds training and validation DataLoaders with DistributedSampler integration"""

    train_dataset = GSM8KDataset(
        split="train",
        tokenizer_name=tokenizer_name,
        max_prompt_len=max_prompt_len,
    )
    
    val_dataset = GSM8KDataset(
        split="test",
        tokenizer_name=tokenizer_name,
        max_prompt_len=max_prompt_len,
    )

    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_gsm8k_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_gsm8k_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler