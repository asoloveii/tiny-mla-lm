import os
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataloaders(data_dir,
                    seq_len,
                    batch_size,
                    val_batch_size=None,
                    train_pattern="*_train_*.bin",
                    val_pattern="*_val_*.bin",
                    num_workers=4,
                    seed=42,):
    """Build train/val DataLoaders (+ train sampler) ready for DDP.

    Works identically whether torch.distributed is initialized or not:
    falls back to rank=0, world_size=1 otherwise. 

    Returns:
        train_loader, val_loader, train_sampler
    """
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    train_paths = sorted(glob.glob(os.path.join(data_dir, train_pattern)))
    val_paths = sorted(glob.glob(os.path.join(data_dir, val_pattern)))
    if not train_paths:
        raise FileNotFoundError(f"No train shards matching {train_pattern!r} in {data_dir}")
    if not val_paths:
        raise FileNotFoundError(f"No val shards matching {val_pattern!r} in {data_dir}")

    train_ds = TokenDataset(train_paths, seq_len)
    val_ds = TokenDataset(val_paths, seq_len)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size or batch_size, sampler=val_sampler,
        drop_last=False, **loader_kwargs,
    )

    return train_loader, val_loader, train_sampler


class TokenDataset(Dataset):

    def __init__(self, bin_paths, seq_len):
        self.seq_len = seq_len
        self.bin_paths = sorted(bin_paths)
        self._mmaps = None

        itemsize = np.dtype(np.uint16).itemsize
        shard_lengths = [os.path.getsize(p) // itemsize for p in self.bin_paths]

        # non-overlapping (seq_len+1)-token windows; remainder per shard is dropped
        self.samples_per_shard = [max(0, (n - 1) // seq_len) for n in shard_lengths]
        self.cum_samples = np.cumsum([0] + self.samples_per_shard)

    def __len__(self):
        return int(self.cum_samples[-1])

    def _lazy_init(self):
        if self._mmaps is None:  
            self._mmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.bin_paths]

    def _locate(self, idx):
        shard_i = int(np.searchsorted(self.cum_samples, idx, side="right") - 1)
        local_i = idx - self.cum_samples[shard_i]
        offset = local_i * self.seq_len
        return shard_i, offset

    def __getitem__(self, idx):
        self._lazy_init()
        shard_i, offset = self._locate(idx)
        chunk = self._mmaps[shard_i][offset : offset + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
