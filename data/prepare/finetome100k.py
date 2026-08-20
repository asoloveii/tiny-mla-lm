import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def format_conversation(turns):
    parts = []
    for turn in turns:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        parts.append(f"<|{role}|>\n{turn['value']}")
    return "\n".join(parts)


def prepare_finetome_shards(output_dir="./data/processed/finetome100k",
                            shard_size=50_000_000,   
                            val_fraction=0.01,
                            tokenize_batch_size=1000,):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")  
    eot_token = enc._special_tokens["<|endoftext|>"]

    dataset = load_dataset("mlabonne/FineTome-100k", split="train")  
    texts = [format_conversation(ex["conversations"]) for ex in dataset]

    n_val = int(len(texts) * val_fraction)
    splits = [("val", texts[:n_val]), ("train", texts[n_val:])]

    for split_name, split_texts in splits:
        shard_index = 0
        token_buffer = []

        def flush_shard(buf):
            nonlocal shard_index
            arr = np.array(buf, dtype=np.uint16)
            path = os.path.join(output_dir, f"finetome_{split_name}_{shard_index:04d}.bin")
            arr.tofile(path)   # direct buffer write, no bytes-copy intermediate
            print(f"saved {path} ({len(buf):,} tokens)")
            shard_index += 1

        for i in tqdm(range(0, len(split_texts), tokenize_batch_size), desc=f"tokenizing {split_name}"):
            batch = split_texts[i : i + tokenize_batch_size]
            for tokens in enc.encode_ordinary_batch(batch, num_threads=os.cpu_count()):
                tokens.append(eot_token)   
                token_buffer.extend(tokens)

            while len(token_buffer) >= shard_size:
                flush_shard(token_buffer[:shard_size])
                token_buffer = token_buffer[shard_size:]

        if token_buffer:
            flush_shard(token_buffer)

    print(f"Done. Shards written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./data/processed/finetome100k")
    parser.add_argument("--shard_size", type=int, default=50_000_000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)

    args = parser.parse_args()

    prepare_finetome_shards(
        args.output_dir, 
        args.shard_size, 
        args.val_fraction, 
        args.tokenize_batch_size
    )
