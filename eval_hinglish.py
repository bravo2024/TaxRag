#!/usr/bin/env python3
"""Compare tax-retriever-bge-small vs hinglish-st-33m on English vs Hinglish
queries over the same 100-query hard gold set used by eval_rerank.py.

Reads hinglish_queries.json (list of {gold, en, hi}) built from the identical
query-selection logic in eval_rerank.py (SUBS/SYN/paraphrase/must/pool/chosen),
via extract_queries.py + build_hinglish.py.

For each model x each query-set (English, Hinglish):
  - embed corpus fresh (unless a cached .npy already exists for that model)
  - embed queries, cosine top-5 retrieval against corpus
  - report Recall@5, MRR@5, nDCG@5 (single relevant doc)

Run:
    PYTHONUTF8=1 python eval_hinglish.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
K = 5

MODELS = [
    # (label, hf_id, max_seq_length)
    ("tax-bge-small",   "vivekkopthsd/tax-retriever-bge-small", 512),
    ("hinglish-st-33m", "vivekkopthsd/hinglish-st-33m",          192),
]


def log(msg):
    print(msg, flush=True)


def emb_cache_path(model_name: str) -> str:
    return os.path.join(OUT, "emb_" + model_name.replace("/", "__") + ".npy")


# ── load corpus ───────────────────────────────────────────────────────────
d = np.load(os.path.join(OUT, "index.npz"), allow_pickle=True)
titles = [str(t) for t in d["chunk_titles"]]
texts = [str(t) for t in d["chunk_texts"]]
n = len(texts)
chunk_sec = np.array([int(m.group(1)) if (m := re.match(r"Section (\d+):", t)) else -1
                      for t in titles])
log(f"Corpus: {n} passages")

# ── load 100-query hard set (English + Hinglish) ─────────────────────────
with open(os.path.join(HERE, "hinglish_queries.json"), encoding="utf-8") as f:
    QROWS = json.load(f)
assert len(QROWS) == 100, f"expected 100 queries, got {len(QROWS)}"
log(f"Loaded {len(QROWS)} gold queries (English + Hinglish)")


def metrics_for(gold_sec: int, top_secs: list[int]):
    if gold_sec in top_secs:
        rank = top_secs.index(gold_sec) + 1
        return 1.0, 1.0 / rank, 1.0 / math.log2(rank + 1)
    return 0.0, 0.0, 0.0


def eval_model(label: str, hf_id: str, max_seq_len: int):
    log(f"\n=== Loading model: {label} ({hf_id}) max_seq_length={max_seq_len} ===")
    t0 = time.time()
    model = SentenceTransformer(hf_id, device="cpu")
    model.max_seq_length = max_seq_len
    log(f"  loaded in {time.time()-t0:.1f}s")

    cache = emb_cache_path(hf_id)
    if os.path.exists(cache):
        corpus_emb = np.load(cache).astype("float32")
        log(f"  loaded cached corpus embeddings {corpus_emb.shape} from {os.path.basename(cache)}")
    else:
        log(f"  embedding {n} corpus passages fresh (this takes a few minutes on CPU)...")
        t0 = time.time()
        corpus_emb = model.encode(texts, normalize_embeddings=True, batch_size=32,
                                   show_progress_bar=True).astype("float32")
        log(f"  corpus embedded in {time.time()-t0:.1f}s")
        np.save(cache, corpus_emb)
        log(f"  cached to {cache}")
    corpus_emb /= (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)

    results = {}
    for qset in ("en", "hi"):
        R = MRR = NDCG = 0.0
        t0 = time.time()
        for row in QROWS:
            gold = row["gold"]
            q = row[qset]
            qv = model.encode([q], normalize_embeddings=True)[0].astype("float32")
            sims = corpus_emb @ qv
            top_idx = np.argsort(-sims)[:K]
            top_secs = [int(chunk_sec[i]) for i in top_idx]
            r, rr, ndcg = metrics_for(gold, top_secs)
            R += r; MRR += rr; NDCG += ndcg
        nq = len(QROWS)
        dt = (time.time() - t0) / nq
        results[qset] = (R / nq, MRR / nq, NDCG / nq, dt)
        log(f"  [{label}] {qset:>2} queries -> Recall@5={R/nq:.3f} MRR@5={MRR/nq:.3f} "
            f"nDCG@5={NDCG/nq:.3f} ({dt*1000:.1f} ms/query)")

    del model
    return results


all_results = {}
for label, hf_id, max_seq_len in MODELS:
    all_results[label] = eval_model(label, hf_id, max_seq_len)

# ── 2x2 table ─────────────────────────────────────────────────────────────
log("\n" + "=" * 78)
log("RESULTS: Recall@5 / MRR@5 / nDCG@5")
log("=" * 78)
header = f"{'Model':<20} {'English queries':>28} {'Hinglish queries':>28}"
log(header)
log("-" * len(header))
for label, _, _ in MODELS:
    en = all_results[label]["en"]
    hi = all_results[label]["hi"]
    en_s = f"R={en[0]:.3f} MRR={en[1]:.3f} nDCG={en[2]:.3f}"
    hi_s = f"R={hi[0]:.3f} MRR={hi[1]:.3f} nDCG={hi[2]:.3f}"
    log(f"{label:<20} {en_s:>28} {hi_s:>28}")
log("=" * 78)

out_json = os.path.join(HERE, "hinglish_eval_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({label: {qset: list(vals) for qset, vals in res.items()}
               for label, res in all_results.items()}, f, indent=2)
log(f"\nSaved raw results to {out_json}")
log("DONE")
