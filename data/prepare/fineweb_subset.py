import os
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


_enc = None 
_eot = None

def _init_worker():
    global _enc, _eot
    _enc = tiktoken.get_encoding("gpt2")
    _eot = _enc._special_tokens["<|endoftext|>"]

def process_batch(batch_texts):
    tokens_list = _enc.encode_ordinary_batch(batch_texts, num_threads=1)
    all_tokens = []
    for tokens in tokens_list:
        tokens.append(_eot)
        all_tokens.extend(tokens)
    return np.array(all_tokens, dtype=np.uint16)


def prepare_fineweb_shards(output_dir="./data/processed/fineweb", 
                           shard_size=100_000_000, 
                           target_tokens=3_000_000_000, 
                           val_shards=1,
                           chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    num_cpus = mp.cpu_count()
    print(f"Parallelizing tokenization across {num_cpus} CPU cores...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    pool = mp.Pool(processes=num_cpus, initializer=_init_worker)

    shard_idx = 0
    total_tokens = 0
    buffer_chunks, buffer_len = [], 0   
    batch_buffer = []
    pbar = tqdm(total=target_tokens, unit="tok")

    def write_shard(tokens):
        nonlocal shard_idx
        split = "val" if shard_idx < val_shards else "train"
        shard_path = os.path.join(output_dir, f"fineweb_{split}_{shard_idx:04d}.bin")
        tokens.tofile(shard_path)
        shard_idx += 1

    def dispatch(batch):
        nonlocal buffer_chunks, buffer_len, total_tokens
        chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]
        for tokens in pool.map(process_batch, chunks):
            buffer_chunks.append(tokens)
            buffer_len += len(tokens)

        if buffer_len >= shard_size:
            merged = np.concatenate(buffer_chunks)
            while len(merged) >= shard_size:
                write_shard(merged[:shard_size])
                total_tokens += shard_size
                pbar.update(shard_size)
                merged = merged[shard_size:]
            buffer_chunks, buffer_len = [merged], len(merged)

    for entry in dataset:
        if total_tokens >= target_tokens:
            break
        batch_buffer.append(entry["text"])
        if len(batch_buffer) >= chunk_size * num_cpus:
            dispatch(batch_buffer)
            batch_buffer = []

    if batch_buffer:
        dispatch(batch_buffer)
    if buffer_len > 0:
        remainder = np.concatenate(buffer_chunks)
        write_shard(remainder)
        total_tokens += len(remainder)
        pbar.update(len(remainder))

    pool.close()
    pool.join()
    pbar.close()
    print(f"Finished! {shard_idx} shards, ~{total_tokens/1e9:.2f}B tokens in {output_dir}")


class FineWebDataset(Dataset):

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./data/processed/fineweb")
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--target_tokens", type=int, default=3_000_000_000)
    parser.add_argument("--val_shards", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1000)

    args = parser.parse_args()

    prepare_fineweb_shards(
        args.output_dir, 
        args.shard_size, 
        args.target_tokens, 
        args.val_shards, 
        args.chunk_size
    )
