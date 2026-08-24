#!/usr/bin/env python3
"""Two-stage retrieval ablation for TaxRAG.

Clean experiment requested: does a second-stage reranker (or a lexical BM25
signal) improve on the fine-tuned dense retriever — which stays the first stage
in every arm? The reranker only REORDERS the fine-tuned retriever's candidates;
it never replaces it.

Pipeline under test:
    Query -> fine-tuned dense retriever -> Top-N candidates
          -> [optional BM25 fusion] -> [optional cross-encoder rerank]
          -> Top-5 -> (downstream LLM, phase 2)

Arms:
    1. Dense                 (fine-tuned bge-small, first stage only)
    2. Dense + BM25          (RRF fusion, dense-primary)
    3. Dense + Reranker      (dense Top-N -> cross-encoder -> Top-5)
    4. Dense + BM25 + Reranker

Metrics (gold = the section a query was paraphrased from):
    Recall@5, MRR@5, nDCG@5   — averaged over the 100-query hard set.

The 100-query set replicates eval_retrieval.py (seed 7) so numbers are
comparable across scripts.

Run:
    python eval_rerank.py
Env:
    RERANKER   cross-encoder id (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
    CAND_N     first-stage candidate pool before rerank/fusion (default 20)
"""
from __future__ import annotations

import math
import os
import re
import random
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DENSE_MODEL = "vivekkopthsd/tax-retriever-bge-small"   # the fine-tuned retriever
RERANKER = os.environ.get("RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CAND_N = int(os.environ.get("CAND_N", "20"))
MAX_SEQ_LENGTH = 512
K = 5                                                  # metrics cutoff
RRF_K = 60
random.seed(7)


# ── load corpus + fine-tuned dense embeddings ────────────────────────────────
def emb_cache_path(model_name: str) -> str:
    return os.path.join(OUT, "emb_" + model_name.replace("/", "__") + ".npy")


d = np.load(os.path.join(OUT, "index.npz"), allow_pickle=True)
titles = [str(t) for t in d["chunk_titles"]]
texts = [str(t) for t in d["chunk_texts"]]
n = len(texts)
chunk_sec = np.array([int(m.group(1)) if (m := re.match(r"Section (\d+):", t)) else -1
                      for t in titles])
sec = {}
for t in titles:
    m = re.match(r"Section (\d+): (.+)", t)
    if m:
        sec.setdefault(int(m.group(1)), m.group(2))

print(f"Loading fine-tuned dense retriever: {DENSE_MODEL}")
model = SentenceTransformer(DENSE_MODEL, device="cpu")
model.max_seq_length = MAX_SEQ_LENGTH

cache = emb_cache_path(DENSE_MODEL)
if os.path.exists(cache):
    corpus_emb = np.load(cache).astype("float32")
    print(f"  loaded pre-built corpus embeddings {corpus_emb.shape} from {os.path.basename(cache)}")
else:
    print("  embedding corpus (no cache found)…")
    corpus_emb = model.encode(texts, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=True).astype("float32")
# ensure normalized for cosine == dot
corpus_emb /= (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)

print(f"Building BM25 over {n} passages…")
bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", t.lower()) for t in texts])

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker: {RERANKER}")
        _reranker = CrossEncoder(RERANKER, device="cpu", max_length=512)
    return _reranker


# ── stage-1 rankers -> ordered list of corpus indices ────────────────────────
def dense_rank(query: str) -> np.ndarray:
    qv = model.encode([query], normalize_embeddings=True)[0].astype("float32")
    sims = corpus_emb @ qv
    return np.argsort(-sims)  # best first


def bm25_rank(query: str) -> np.ndarray:
    scores = bm25.get_scores(re.findall(r"[a-z0-9]+", query.lower()))
    return np.argsort(-scores)


def rrf_fuse(*ranked_lists: np.ndarray, limit: int) -> list[int]:
    """Reciprocal-rank fusion of several ordered index arrays."""
    score: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst):
            score[int(idx)] = score.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])][:limit]


def rerank(query: str, cand_idx: list[int], top_k: int) -> list[int]:
    ce = get_reranker()
    pairs = [(query, texts[i]) for i in cand_idx]
    scores = ce.predict(pairs, batch_size=32)
    order = np.argsort(-np.asarray(scores))
    return [cand_idx[i] for i in order[:top_k]]


# ── the four arms -> Top-K corpus indices ────────────────────────────────────
def arm_dense(q):
    return list(dense_rank(q)[:K])

def arm_dense_bm25(q):
    return rrf_fuse(dense_rank(q), bm25_rank(q), limit=K)

def arm_dense_rerank(q):
    cand = list(dense_rank(q)[:CAND_N])
    return rerank(q, cand, K)

def arm_dense_bm25_rerank(q):
    cand = rrf_fuse(dense_rank(q), bm25_rank(q), limit=CAND_N)
    return rerank(q, cand, K)

ARMS = {
    "Dense (fine-tuned)":        arm_dense,
    "Dense + BM25":              arm_dense_bm25,
    "Dense + Reranker":          arm_dense_rerank,
    "Dense + BM25 + Reranker":   arm_dense_bm25_rerank,
}


# ── 100-query hard set (identical to eval_retrieval.py) ───────────────────────
SUBS = [
    (r"^Deduction in respect of ", "How do I claim tax relief for "),
    (r"^Deduction for ", "What is the tax benefit available for "),
    (r"^Deductions? from ", "What can be subtracted from "),
    (r"^Deductions? related to ", "What expenses are allowed for "),
    (r"^Set off of ", "How are you allowed to adjust "),
    (r"^Carry forward and set off of ", "How is future adjustment handled for "),
    (r"^Computation of ", "How do you work out "),
    (r"^Manner of computing ", "What is the method to arrive at "),
    (r"^Special provision (for|in) ", "What special rule applies to "),
    (r"^Penalty for ", "What is the consequence for "),
    (r"^Power(s)? (of|to) ", "What authority exists over "),
    (r"^Income (from|under) ", "How is income from "),
    (r"^Definitions", "the meaning of key terms used in the Act"),
    (r"^Residence in India", "how a person's residential status is decided"),
]
SYN = {"premia": "premiums", "premium": "premiums", "provident fund": "PF savings scheme",
       "insurance": "cover", "donations": "charitable giving", "loan": "borrowing",
       "residential house property": "a home you live in", "house property": "a property you own",
       "capital gains": "profit on selling assets", "salaries": "salary earnings",
       "higher education": "college studies", "health insurance": "medical cover"}


def paraphrase(title):
    q, applied = title, False
    for pat, repl in SUBS:
        if re.search(pat, q):
            q = re.sub(pat, repl, q); applied = True; break
    for k, v in SYN.items():
        q = q.replace(k, v)
    q = q.strip().rstrip(",")
    if not applied and not q.endswith("?"):
        q = f"What does the Act say about {q[0].lower() + q[1:]}?"
    elif not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


def descriptive(t):
    return len([w for w in re.findall(r"[A-Za-z]+", t) if len(w) > 2]) >= 3


must = [122, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134, 136, 137,
        108, 109, 110, 111, 112, 114, 115, 119, 120,
        6, 18, 19, 20, 22, 26, 27, 33, 67, 72, 82, 85, 86, 99]
must = [s for s in must if s in sec]
pool = [s for s in sorted(sec) if s not in must and descriptive(sec[s])]
random.shuffle(pool)
chosen = (must + pool)[:100]
queries = [(s, paraphrase(sec[s])) for s in chosen]


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(gold_sec: int, top_idx: list[int]):
    top_secs = [int(chunk_sec[i]) for i in top_idx]
    if gold_sec in top_secs:
        rank = top_secs.index(gold_sec) + 1        # 1-indexed within Top-K
        return 1.0, 1.0 / rank, 1.0 / math.log2(rank + 1)   # recall, RR, nDCG (IDCG=1)
    return 0.0, 0.0, 0.0


print(f"\nHard set: {len(queries)} queries | candidate pool N={CAND_N} | cutoff K={K}\n")
print(f"{'Arm':<26} {'Recall@5':>9} {'MRR@5':>8} {'nDCG@5':>8} {'sec/query':>10}")
print("-" * 66)
results = {}
for name, fn in ARMS.items():
    R = MRR = NDCG = 0.0
    t0 = time.time()
    for gold, q in queries:
        top = fn(q)
        r, rr, ndcg = metrics(gold, top)
        R += r; MRR += rr; NDCG += ndcg
    nq = len(queries)
    dt = (time.time() - t0) / nq
    results[name] = (R / nq, MRR / nq, NDCG / nq, dt)
    print(f"{name:<26} {R/nq:>9.3f} {MRR/nq:>8.3f} {NDCG/nq:>8.3f} {dt:>10.3f}")

print("-" * 66)
base = results["Dense (fine-tuned)"]
print("\nDelta vs Dense-only (fine-tuned first stage):")
for name, (r, mrr, ndcg, _) in results.items():
    if name == "Dense (fine-tuned)":
        continue
    print(f"  {name:<26} ΔRecall@5={r-base[0]:+.3f}  ΔMRR@5={mrr-base[1]:+.3f}  ΔnDCG@5={ndcg-base[2]:+.3f}")
print(f"\nReranker: {RERANKER}")
print("Note: downstream metrics (faithfulness, citation correctness, answer "
      "accuracy) are phase 2 — they need per-arm LLM generation over the same set.")
