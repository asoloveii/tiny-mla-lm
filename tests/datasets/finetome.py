import os
import tiktoken
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from data.prepare.finetome100k import tokenize_and_mask_conversation, prepare_finetome_shards
from data.prepare.utils import TokenDataset


def test_tokenize_and_mask_conversation_prompt_masking():
    """Verify user/system tokens get masked with -100 and assistant tokens retain IDs"""
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc._special_tokens["<|endoftext|>"]

    turns = [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello!"},
    ]

    tokens, labels = tokenize_and_mask_conversation(turns, enc, eot_token)

    assert len(tokens) == len(labels)
    assert tokens[-1] == eot_token
    assert labels[-1] == eot_token

    # extract assistant text tokens and verify masking split
    assistant_str = "<|assistant|>\nHello!\n"
    assistant_tokens = enc.encode_ordinary(assistant_str)
    num_assistant_tokens = len(assistant_tokens)

    # all non-assistant tokens (system + user) must be masked with -100
    non_assistant_len = len(tokens) - num_assistant_tokens - 1  # -1 for EOT
    assert labels[:non_assistant_len] == [-100] * non_assistant_len

    # assistant tokens must match actual token IDs
    assert labels[non_assistant_len : non_assistant_len + num_assistant_tokens] == assistant_tokens


@patch("data.prepare.finetome100k.load_dataset")
def test_prepare_finetome_shards_generation(mock_load_dataset, tmp_path):
    """Verify train/val splitting and paired int32 (_tokens.bin, _labels.bin) file creation"""
    dummy_conversations = [
        {"conversations": [{"from": "human", "value": f"Question {i}"}, {"from": "gpt", "value": f"Answer {i}"}]}
        for i in range(10)
    ]
    
    # mock huggingface dataset object and select behavior
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(dummy_conversations)
    mock_ds.select.side_effect = lambda r: dummy_conversations[r.start : r.stop]
    mock_load_dataset.return_value = mock_ds

    output_dir = tmp_path / "finetome_test"
    prepare_finetome_shards(
        output_dir=str(output_dir),
        shard_size=50,
        val_fraction=0.2,
    )

    files = sorted(os.listdir(output_dir))
    val_tokens = [f for f in files if "val" in f and f.endswith("_tokens.bin")]
    val_labels = [f for f in files if "val" in f and f.endswith("_labels.bin")]
    train_tokens = [f for f in files if "train" in f and f.endswith("_tokens.bin")]
    train_labels = [f for f in files if "train" in f and f.endswith("_labels.bin")]

    assert len(val_tokens) >= 1
    assert len(val_labels) == len(val_tokens)
    assert len(train_tokens) >= 1
    assert len(train_labels) == len(train_tokens)

    # verify int32 binary dtype and presence of -100 masks in labels
    labels_arr = np.fromfile(output_dir / val_labels[0], dtype=np.int32)
    tokens_arr = np.fromfile(output_dir / val_tokens[0], dtype=np.int32)

    assert len(labels_arr) == len(tokens_arr)
    assert -100 in labels_arr


@pytest.fixture(scope="module")
def processed_finetome_data(tmp_path_factory):
    """Generates a tiny slice of FineTome shards for SFT integration tests"""
    temp_dir = tmp_path_factory.mktemp("finetome_mini")

    from datasets import load_dataset as real_load_dataset
    dataset_slice = list(real_load_dataset("mlabonne/FineTome-100k", split="train", streaming=True).take(20))

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(dataset_slice)
    mock_ds.select.side_effect = lambda r: dataset_slice[r.start : r.stop]

    with patch("data.prepare.finetome100k.load_dataset", return_value=mock_ds):
        prepare_finetome_shards(
            output_dir=str(temp_dir),
            shard_size=200,
            val_fraction=0.1,
        )

    yield temp_dir


def test_token_dataset_sft_mode_masked_targets(processed_finetome_data):
    """Verify TokenDataset correctly loads paired SFT tokens/labels and yields -100 targets"""
    token_paths = [
        os.path.join(processed_finetome_data, f)
        for f in os.listdir(processed_finetome_data)
        if f.endswith("_tokens.bin")
    ]

    seq_len = 32
    dataset = TokenDataset(bin_paths=token_paths, seq_len=seq_len)

    assert dataset.has_masks is True
    assert dataset.dtype == np.int32
    assert len(dataset) > 0

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Verify target alignment against label mmap
    label_path = sorted(token_paths)[0].replace("_tokens.bin", "_labels.bin")
    mmap_label = np.memmap(label_path, dtype=np.int32, mode="r")
    raw_y = mmap_label[1 : seq_len + 1].astype(np.int64)

    torch.testing.assert_close(y, torch.from_numpy(raw_y))
    assert -100 in y.tolist()
