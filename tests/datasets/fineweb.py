import os
import shutil
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.prepare.fineweb_subset import FineWebDataset, prepare_fineweb_shards


@pytest.fixture(scope="module")
def processed_data_dir(tmp_path_factory):
    """Generates real mini-shards from a small slice of FineWeb-Edu.

    Runs once per test session and cleans up automatically afterwards.
    """
    temp_dir = tmp_path_factory.mktemp("fineweb_mini")

    prepare_fineweb_shards(
        output_dir=str(temp_dir),
        shard_size=1_000,
        target_tokens=5_000,
        val_shards=1,
        chunk_size=10
    )

    yield temp_dir


def test_fineweb_shards_creation(processed_data_dir):
    """Verify that actual binary shards are generated on disk with correct uint16 formatting"""
    all_files = sorted([f for f in os.listdir(processed_data_dir) if f.endswith(".bin")])

    assert len(all_files) > 0, "No .bin shards were generated."

    # filter out val and train shards explicitly
    val_files = [f for f in all_files if "val" in f]
    train_files = [f for f in all_files if "train" in f]

    # with val_shards=1, exactly one validation shard starting at index 0000 should exist
    assert len(val_files) == 1, f"Expected 1 val shard, found {len(val_files)}"
    assert val_files[0] == "fineweb_val_0000.bin"

    # if extra shards were flushed, they must be formatted as train shards
    if train_files:
        assert train_files[0] == "fineweb_train_0001.bin"

    # verify content byte size aligns with 16-bit integers (2 bytes per token)
    for fname in all_files:
        fpath = os.path.join(processed_data_dir, fname)
        file_size_bytes = os.path.getsize(fpath)
        assert file_size_bytes % 2 == 0, f"File {fname} is not aligned to uint16 (2 bytes)."
        
        tokens = np.fromfile(fpath, dtype=np.uint16)
        assert len(tokens) > 0
        assert tokens.max() < 50257  # gpt2 tokenizer vocabulary limit


def test_token_dataset_with_real_shards(processed_data_dir):
    """Verify TokenDataset correctly loads and formats real token arrays into training pairs"""
    bin_paths = [
        os.path.join(processed_data_dir, f)
        for f in os.listdir(processed_data_dir)
        if f.endswith(".bin")
    ]

    seq_len = 128
    dataset = FineWebDataset(bin_paths=bin_paths, seq_len=seq_len)

    # dataset should contain at least 1 valid sample
    assert len(dataset) > 0

    x, y = dataset[0]

    # verify tensor properties
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)

    # verify standard language model autoregressive target shift: y[t] == x[t + 1]
    mmap_first_shard = np.memmap(sorted(bin_paths)[0], dtype=np.uint16, mode="r")
    raw_sample = mmap_first_shard[: seq_len + 1].astype(np.int64)

    torch.testing.assert_close(x, torch.from_numpy(raw_sample[:-1]))
    torch.testing.assert_close(y, torch.from_numpy(raw_sample[1:]))


def test_dataloader_batching_real_data(processed_data_dir):
    """Verify PyTorch DataLoader successfully batches samples for training"""
    bin_paths = [
        os.path.join(processed_data_dir, f)
        for f in os.listdir(processed_data_dir)
        if f.endswith(".bin")
    ]

    seq_len = 64
    dataset = FineWebDataset(bin_paths=bin_paths, seq_len=seq_len)
    
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (batch_size, seq_len)
    assert batch_y.shape == (batch_size, seq_len)
