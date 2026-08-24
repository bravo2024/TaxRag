#!/usr/bin/env python3
"""Re-embed the bilingual TaxRAG corpus with the new 2026 encoder.

Reads the existing bilingual metadata (out/index_m3_bilingual.npz: EN passages)
plus the cached Hindi translations (out/passages_hi.json), re-embeds all 2,474
passages (EN + HI) with the new encoder, and writes the new bilingual index:

  index_gemma_bilingual.npz     if google/embeddinggemma-300m loads
  index_qwen3_bilingual.npz     if it falls back to Qwen/Qwen3-Embedding-0.6B

Schema matches index_m3_bilingual.npz (embeddings, chunk_texts, chunk_titles,
chunk_sources, chunk_ids, languages) so hybrid_retriever can consume it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("taxrag.build_gemma")

OUT = "./out"
GEMMA_ENCODER = "google/embeddinggemma-300m"
QWEN_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
MIN_FREE_GPU_GB = 3.0


def _pick_device() -> str:
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            if free_gb >= MIN_FREE_GPU_GB:
                log.info("CUDA free %.1fGB; using GPU", free_gb)
                return "cuda"
            log.warning("Only %.1fGB GPU free; running on CPU", free_gb)
        except Exception as exc:  # noqa: BLE001
            log.warning("GPU mem check failed (%s); CPU", exc)
    return "cpu"


def load_corpus(out: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Return (en_chunks, hi_chunks, languages) for the full bilingual set."""
    src = os.path.join(out, "index_m3_bilingual.npz")
    d = np.load(src, allow_pickle=True)
    n_en = int((np.array([str(x) for x in d["languages"]]) == "en").sum())
    en_chunks = []
    for i in range(n_en):
        en_chunks.append({
            "text": str(d["chunk_texts"][i]),
            "title": str(d["chunk_titles"][i]),
            "source": str(d["chunk_sources"][i]),
            "chunk_id": str(d["chunk_ids"][i]),
        })
    log.info("Loaded %d English passages from %s", len(en_chunks), src)

    hi_path = os.path.join(out, "passages_hi.json")
    with open(hi_path, encoding="utf-8") as f:
        passages = json.load(f)
    by_id = {p["id"]: p for p in passages}
    hi_chunks = []
    for c in en_chunks:
        p = by_id.get(c["chunk_id"])
        if p is None:
            raise KeyError(f"No Hindi passage for chunk {c['chunk_id']}")
        hi_chunks.append({
            "text": p["hi_text"],
            "title": c["title"],
            "source": c["source"],
            "chunk_id": c["chunk_id"],
        })
    log.info("Loaded %d Hindi passages from %s", len(hi_chunks), hi_path)
    return en_chunks, hi_chunks, ["en"] * len(en_chunks) + ["hi"] * len(hi_chunks)


def embed(model, texts: List[str], batch: int = 16) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        v = model.encode(chunk, prompt_name="document",
                         normalize_embeddings=True, batch_size=batch)
        vecs.append(v)
        log.info("embedded %d/%d", min(i + batch, len(texts)), len(texts))
    return np.vstack(vecs).astype("float32")


def save_index(path: str, embeddings: np.ndarray, chunks: List[Dict[str, Any]],
               languages: List[str]) -> None:
    np.savez_compressed(
        path,
        embeddings=embeddings,
        chunk_texts=np.array([c["text"] for c in chunks], dtype=object),
        chunk_titles=np.array([c["title"] for c in chunks], dtype=object),
        chunk_sources=np.array([c["source"] for c in chunks], dtype=object),
        chunk_ids=np.array([c["chunk_id"] for c in chunks], dtype=object),
        languages=np.array(languages, dtype=object),
    )
    log.info("Saved %s (%d vectors, dim %d)", path, len(chunks), embeddings.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-embed bilingual corpus with 2026 encoder")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=os.environ.get("TAXRAG_GEMMA_ENCODER", GEMMA_ENCODER),
                    help="encoder to embed with (HF id or local dir); env TAXRAG_GEMMA_ENCODER")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    en_chunks, hi_chunks, languages = load_corpus(out)
    all_chunks = en_chunks + hi_chunks
    device = args.device or _pick_device()

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    primary = args.model
    encoder, encoder_tag = primary, ("gemma" if "gemma" in primary.lower() else "qwen3")
    t0 = time.time()
    try:
        log.info("Loading encoder %s on %s", primary, device)
        model = SentenceTransformer(primary, device=device)
    except Exception as exc:  # noqa: BLE001
        log.warning("Encoder %s failed to load (%s); falling back to %s",
                    primary, exc, QWEN_ENCODER)
        encoder, encoder_tag = QWEN_ENCODER, "qwen3"
        model = SentenceTransformer(QWEN_ENCODER, device=device)

    texts = [c["text"] for c in all_chunks]
    log.info("Embedding %d passages (%d EN + %d HI) with %s",
             len(texts), len(en_chunks), len(hi_chunks), encoder)
    emb = embed(model, texts, batch=args.batch)
    log.info("Re-embedding done in %.1fs -> %s", time.time() - t0, emb.shape)

    out_name = "index_gemma_bilingual.npz" if encoder_tag == "gemma" \
        else "index_qwen3_bilingual.npz"
    save_index(os.path.join(out, out_name), emb, all_chunks, languages)

    meta_path = os.path.join(out, "multilingual_build.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        meta["new_stack_encoder"] = encoder
        meta["new_stack_encoder_tag"] = encoder_tag
        meta["new_stack_embedding_dim"] = int(emb.shape[1])
        meta["new_stack_index"] = out_name
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info("Updated %s", meta_path)
    log.info("ENCODER_LOADED=%s index=%s", encoder, out_name)


if __name__ == "__main__":
    main()
