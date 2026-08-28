import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def tokenize_and_mask_conversation(turns, enc, eot_token):
    """Returns two lists: tokens and labels. 
    Labels are -100 for system/user tokens, and actual token IDs for assistant tokens"""
    tokens = []
    labels = []

    for turn in turns:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        formatted_turn = f"<|{role}|>\n{turn['value']}\n"
        turn_tokens = enc.encode_ordinary(formatted_turn)

        tokens.extend(turn_tokens)

        # mask user and system roles with -100
        if role == "assistant":
            labels.extend(turn_tokens)
        else:
            labels.extend([-100] * len(turn_tokens))

    tokens.append(eot_token)
    labels.append(eot_token)

    return tokens, labels


def prepare_finetome_shards(output_dir="./data/processed/finetome100k",
                            shard_size=50_000_000,   
                            val_fraction=0.01,):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")  
    eot_token = enc._special_tokens["<|endoftext|>"]

    dataset = load_dataset("mlabonne/FineTome-100k", split="train")  

    n_val = int(len(dataset) * val_fraction)
    splits = [("val", dataset.select(range(n_val))), ("train", dataset.select(range(n_val, len(dataset))))]

    for split_name, split_dataset in splits:
        shard_index = 0
        token_buffer = []
        label_buffer = []

        def flush_shard(t_buf, l_buf):
            nonlocal shard_index
            t_arr = np.array(t_buf, dtype=np.int32) 
            l_arr = np.array(l_buf, dtype=np.int32)

            t_path = os.path.join(output_dir, f"finetome_{split_name}_{shard_index:04d}_tokens.bin")
            l_path = os.path.join(output_dir, f"finetome_{split_name}_{shard_index:04d}_labels.bin")
            
            t_arr.tofile(t_path)
            l_arr.tofile(l_path)
            print(f"saved {t_path} and labels ({len(t_buf):,} tokens)")
            shard_index += 1

        for ex in tqdm(split_dataset, desc=f"processing {split_name}"):
            tokens, labels = tokenize_and_mask_conversation(ex["conversations"], enc, eot_token)
            token_buffer.extend(tokens)
            label_buffer.extend(labels)

            while len(token_buffer) >= shard_size:
                flush_shard(token_buffer[:shard_size], label_buffer[:shard_size])
                token_buffer = token_buffer[shard_size:]
                label_buffer = label_buffer[shard_size:]

        if token_buffer:
            flush_shard(token_buffer, label_buffer)

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
