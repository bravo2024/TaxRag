#!/usr/bin/env python3
"""Pre-embed the TaxRAG corpus with a chosen retriever and cache it to disk.

The app (app.py) normally embeds the corpus on the fly the first time a model is
selected, which is slow on CPU. Running this once per model writes
`out/emb_<model-slug>.npy` (normalized embeddings aligned to out/index.npz's
chunk order); app.py then loads that instantly instead of re-embedding.

Usage:
    python build_model_index.py                      # default: Granite-97m
    python build_model_index.py --model vivekkopthsd/tax-retriever-bge-small
    python build_model_index.py --all                # all app defaults
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import load_index, embed_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

DEFAULT_MODEL = "vivekkopthsd/tax-retriever-bge-small"
MAX_SEQ_LENGTH = 512  # passages are 200–500 tokens; keeps CPU embedding fast
ALL_MODELS = [
    "vivekkopthsd/tax-retriever-bge-small",
    "vivekkopthsd/fiqa-retriever-bge-small",
]


def emb_path(model_name: str) -> str:
    """Where the cached embeddings for a model live. Must match app.py."""
    return os.path.join(OUT, "emb_" + model_name.replace("/", "__") + ".npy")


def build(model_name: str, chunks) -> None:
    path = emb_path(model_name)
    t0 = time.time()
    print(f"[build] {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH  # raise 256->512 (no truncation) / cap long-context
    emb = embed_chunks(chunks, model).astype("float32")
    np.save(path, emb)
    print(f"  saved {path}  ({emb.shape[0]} vectors, dim {emb.shape[1]}, "
          f"{time.time() - t0:.1f}s)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-embed the corpus per retriever")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--all", action="store_true", help="build every app default model")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    _, chunks = load_index(OUT)
    print(f"Loaded {len(chunks)} chunks from {OUT}\n")

    for m in (ALL_MODELS if args.all else [args.model]):
        build(m, chunks)


if __name__ == "__main__":
    main()
