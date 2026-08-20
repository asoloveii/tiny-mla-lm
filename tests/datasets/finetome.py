import os
import numpy as np
import pytest
import torch
from unittest.mock import patch

from data.prepare.finetome100k import format_conversation, prepare_finetome_shards
from data.prepare.utils import TokenDataset


def test_format_conversation_role_mapping():
    """Verify ShareGPT turns map 'human' and 'gpt' roles to '<|user|>' and '<|assistant|>'"""
    turns = [
        {"from": "human", "value": "Hello!"},
        {"from": "gpt", "value": "Hi there!"},
        {"from": "system", "value": "You are a helpful assistant."},
    ]
    
    formatted = format_conversation(turns)
    expected = "<|user|>\nHello!\n<|assistant|>\nHi there!\n<|system|>\nYou are a helpful assistant."
    
    assert formatted == expected


def test_format_conversation_unknown_role():
    """Verify custom or unexpected roles pass through unmapped"""
    turns = [{"from": "custom_agent", "value": "Executing task..."}]
    formatted = format_conversation(turns)
    
    assert formatted == "<|custom_agent|>\nExecuting task..."


@patch("data.prepare.finetome100k.load_dataset")
def test_prepare_finetome_shards_splitting_and_eot(mock_load_dataset, tmp_path):
    """Verify train/val splitting logic, shard buffer flush, and EOT token appending"""
    # 10 dummy conversation samples
    dummy_conversations = [
        {"conversations": [{"from": "human", "value": f"Question {i}"}, {"from": "gpt", "value": f"Answer {i}"}]}
        for i in range(10)
    ]
    mock_load_dataset.return_value = dummy_conversations

    output_dir = tmp_path / "finetome_test"
    shard_size = 50 
    val_fraction = 0.2  

    prepare_finetome_shards(
        output_dir=str(output_dir),
        shard_size=shard_size,
        val_fraction=val_fraction,
        tokenize_batch_size=2,
    )

    files = sorted(os.listdir(output_dir))
    val_files = [f for f in files if "val" in f]
    train_files = [f for f in files if "train" in f]

    assert len(val_files) >= 1
    assert len(train_files) >= 1
    assert val_files[0] == "finetome_val_0000.bin"
    assert train_files[0] == "finetome_train_0000.bin"

    # verify uint16 binary token validity
    val_tokens = np.fromfile(output_dir / val_files[0], dtype=np.uint16)
    assert len(val_tokens) > 0
    assert val_tokens.max() < 50257  # gpt2 vocab limit


@pytest.fixture(scope="module")
def processed_finetome_data(tmp_path_factory):
    """Downloads a tiny slice of FineTome-100k to test shard building and dataset loading"""
    temp_dir = tmp_path_factory.mktemp("finetome_mini")

    # mock load_dataset to pull a fast, streaming sample from finetome-100k
    from datasets import load_dataset as real_load_dataset
    dataset_slice = list(real_load_dataset("mlabonne/FineTome-100k", split="train", streaming=True).take(20))

    with patch("data.prepare.finetome100k.load_dataset", return_value=dataset_slice):
        prepare_finetome_shards(
            output_dir=str(temp_dir),
            shard_size=200,
            val_fraction=0.1,
            tokenize_batch_size=5,
        )

    yield temp_dir


def test_token_dataset_integration_with_finetome(processed_finetome_data):
    """Verify TokenDataset correctly loads generated FineTome binary shards"""
    bin_paths = [
        os.path.join(processed_finetome_data, f)
        for f in os.listdir(processed_finetome_data)
        if f.endswith(".bin")
    ]

    seq_len = 32
    dataset = TokenDataset(bin_paths=bin_paths, seq_len=seq_len)

    assert len(dataset) > 0

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # target alignment check (y is x shifted right by 1)
    mmap_shard = np.memmap(sorted(bin_paths)[0], dtype=np.uint16, mode="r")
    raw = mmap_shard[: seq_len + 1].astype(np.int64)

    torch.testing.assert_close(x, torch.from_numpy(raw[:-1]))
    torch.testing.assert_close(y, torch.from_numpy(raw[1:]))
