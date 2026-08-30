# TinyLM

A from-scratch decoder-only language model exploring efficient attention
and normalization techniques, paired with a document-grounded RAG API for
serving it.

## Overview

TinyLM is trained in three stages: pretraining on a slice of FineWebEDU,
then supervised fine-tuning on FineTome-100k and post-training on GSM8K with Group-Relative Policy Optimization objective. The project is also served behind a FastAPI application that can answer questions either directly or grounded in a set
of ingested documents via retrieval-augmented generation (RAG).

This repository covers the full pipeline: tokenizing and sharding
pretraining/SFT/GRPO data, DDP-ready data loading, and a demo
serving layer (vLLM + FastAPI + LangChain-based retrieval).

## Architecture

TinyLM is a decoder-only transformer with:

- **Multi-Head Latent Attention (MLA)** — queries/keys/values are projected
  through a low-rank latent bottleneck rather than full-size K/V
  projections, with a decoupled RoPE key path so positional information
  doesn't have to survive compression. KV caching stores only the
  compressed latent and the decoupled rope key, not full per-head K/V,
  which is the main point of MLA over standard multi-head attention.
- **Rotary positional embeddings (RoPE)** — applied to queries and the
  decoupled key path.
- **Derf normalization** — a learnable, dynamic error-function-based
  normalization used in place of LayerNorm at each sub-block.
- **SwiGLU feed-forward layers** — gated activation in the MLP block.
- **ReZero/LayerScale-style residual scaling** — each residual branch is
  scaled by a learnable scalar, initialized near identity, rather than
  relying on pre-norm/post-norm alone.
- **Exclusive attention (XSA)** — attention output is projected to be
  orthogonal to the local value direction before the output projection.

## Repository structure

```
tiny_mla_lm/
│   ├── pretrain_config.yaml       # LR schedule, batch size, tokens budget
│   ├── sft_config.yaml
│   ├── grpo_config.yaml           # group size, KL coeff, reward settings
│   └── model_config.yaml          # 100M arch: n_layer, n_head, d_model
│
├── data/
│   ├── raw/                       # cached FineWebEDU / FineTome / GSM8K shards
│   ├── processed/                 # tokenized .bin/.parquet shards per stage
│   └── prepare/
│       ├── fineweb_subset.py      # stream, filter, tokenize, pack into blocks
│       ├── finetome100k.py        # apply chat template, mask prompt tokens
│       └── gsm8k.py               # format Q/A, extract gold answers for reward
│       └── utils.py               # custom Dataset class for pretraining and sft
│
├── model/
│   ├── transformer.py             # architecture implementation
│   └── hf_export.py               # convert raw checkpoint -> HF-style dir 
│
├── training/
│   ├── train.py                 # stage 1 and 2: pretraining and SFT
│   └── grpo_train.py            # stage 3: GRPO loop on GSM8K
│
├── checkpoints/
│   ├── pretrain/
│   ├── sft/
│   └── grpo/                       # final model -> exported to HF format here
│
├── api/
│   ├── main.py                     # FastAPI app entrypoint
│   ├── routers/
│   │   ├── generate.py             # /generate — plain prompt passthrough to vLLM
│   │   └── rag_query.py            # /ask — RAG-augmented endpoint
│   ├── schemas.py                  # Pydantic request/response models
│   └── client.py                   # OpenAI-client wrapper pointed at vLLM :8000
│
├── rag/
│   ├── ingest.py                   # load docs -> chunk -> embed -> write to vector store
│   ├── vectorstore.py              # LangChain FAISS/Chroma init & persistence
│   ├── retriever.py                # LangChain retriever (+ optional reranker)
│   └── chain.py                    # LangChain chain: retrieve -> build augmented prompt -> call api/client.py
│
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline

### Pretraining data (FineWeb-Edu)

`data/prepare_fineweb.py` streams the `HuggingFaceFW/fineweb-edu` dataset (sample-10BT subset), tokenizes text with `tiktoken`'s gpt2 encoding via `encode_ordinary_batch`, and appends an `<|endoftext|>` token (50256) after each document. High throughput is achieved by batching documents into chunks and distributing them across a `multiprocessing.Pool` of CPU workers initialized with worker-local tokenizers.

```bash
python data/prepare_fineweb.py   # defaults to 2B tokens
```

### SFT data (FineTome-100k)

`data/prepare_finetome.py` processes `mlabonne/FineTome-100k` in memory, converting ShareGPT-style conversations into `<|user|>` / `<|assistant|>` templated string turns. To support assistant-only loss masking, the script produces two synchronized parallel binary streams: a tokens stream (`*_tokens.bin`) containing the tokenized sequence using gpt2 encoding ending with `<|endoftext|>`, and a labels stream (`*_labels.bin`) containing loss targets where non-assistant tokens (system and user prompts) are masked out with -100 while assistant tokens retain their actual token IDs.

```bash
python data/prepare_finetome.py
```

### Dataloaders

`get_dataloaders()` in `data/dataloaders.py` returns DDP-ready train/val
`DataLoader`s (backed by `DistributedSampler`) from a directory of `.bin`
shards. It auto-detects whether `torch.distributed` is initialized, so the
same call works unchanged on a single GPU or under `torchrun`.

```python
train_loader, val_loader, train_sampler = get_dataloaders(
    data_dir="./data/raw/fineweb10B", seq_len=1024, batch_size=32,
)
```
