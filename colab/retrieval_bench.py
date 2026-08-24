"""Retrieval benchmark on the content-based test set + a fine-tuned cross-encoder reranker.

Runs on Colab (T4). Inputs in /content: chunks.json, gen_questions.json, legacy_hard_queries.json.
Outputs in /content/out: retrieval_results.json, ce_tax/ (fine-tuned reranker).

Gold = the section a question was generated from; a hit means any passage of that section is
retrieved. Metrics are computed on sections (duplicate passages of one section count once).
The reranker is trained only on questions from TRAIN sections; test sections are disjoint.
"""
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(13); random.seed(13); np.random.seed(13)
IN, OUT = Path("/content"), Path("/content/out")
OUT.mkdir(parents=True, exist_ok=True)
chunks = json.loads((IN / "chunks.json").read_text(encoding="utf-8"))
gen = json.loads((OUT / "gen_questions.json").read_text(encoding="utf-8"))["rows"]
legacy = json.loads((IN / "legacy_hard_queries.json").read_text(encoding="utf-8"))
texts = [c["text"] for c in chunks]
csec = np.array([c["section"] for c in chunks])
test = [r for r in gen if r["split"] == "test"]
train = [r for r in gen if r["split"] == "train"]
print(f"chunks={len(chunks)} test_q={len(test)} train_q={len(train)} legacy_q={len(legacy)}")

DENSE = {
    "bge-small (base)": "BAAI/bge-small-en-v1.5",
    "bge-small FiQA-tuned (app default)": "vivekkopthsd/fiqa-retriever-bge-small",
    "bge-small tax-tuned (trained on Act titles)": "vivekkopthsd/tax-retriever-bge-small",
    "bge-base (larger reference)": "BAAI/bge-base-en-v1.5",
}
emb, qenc = {}, {}
for name, mid in DENSE.items():
    m = SentenceTransformer(mid, device="cuda")
    m.max_seq_length = 512
    emb[name] = m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    qenc[name] = m
    print("embedded", name)

tok_re = re.compile(r"[a-z0-9]+")
bm25 = BM25Okapi([tok_re.findall(t.lower()) for t in texts])


def dense_rank(name, q):
    v = qenc[name].encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    return np.argsort(-(emb[name] @ v))


def bm25_rank(q):
    return np.argsort(-bm25.get_scores(tok_re.findall(q.lower())))


def rrf(*lists, k=60, limit=50):
    s = {}
    for lst in lists:
        for r, i in enumerate(lst[:200]):
            s[int(i)] = s.get(int(i), 0) + 1 / (k + r)
    return [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])][:limit]


def sections_of(idx_list):
    seen, out = set(), []
    for i in idx_list:
        s = int(csec[i])
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def score(gold, idx_list):
    secs = sections_of(idx_list)[:10]
    if gold in secs:
        r = secs.index(gold) + 1
        return {"r1": float(r == 1), "r5": float(r <= 5), "r10": 1.0, "mrr10": 1 / r,
                "ndcg10": 1 / math.log2(r + 1)}
    return {"r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr10": 0.0, "ndcg10": 0.0}


class Reranker:
    def __init__(self, mid):
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSequenceClassification.from_pretrained(mid).cuda().eval()

    @torch.no_grad()
    def rerank(self, q, idx, limit=10):
        enc = self.tok([q] * len(idx), [texts[i] for i in idx], truncation=True, max_length=512,
                       padding=True, return_tensors="pt").to("cuda")
        s = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return [idx[j] for j in np.argsort(-s)][:limit]


def evaluate(queries, arms):
    res = {}
    for name, fn in arms.items():
        t0, per = time.time(), []
        for r in queries:
            per.append(score(r["section"], fn(r["question"])))
        res[name] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        res[name]["ms_per_query"] = 1000 * (time.time() - t0) / len(queries)
        res[name]["_hits5"] = [p["r5"] for p in per]
        print(f"  {name:<52} R@1={res[name]['r1']:.3f} R@5={res[name]['r5']:.3f} "
              f"MRR@10={res[name]['mrr10']:.3f}", flush=True)
    return res


# ---- fine-tune a cross-encoder on TRAIN-section questions with hard negatives -------------
BASE_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIRST = "bge-small FiQA-tuned (app default)"
pairs = []
for r in train:
    ranked = dense_rank(FIRST, r["question"])[:30]
    gold_chunks = [i for i in np.where(csec == r["section"])[0]]
    if not gold_chunks:
        continue
    pos = max(gold_chunks, key=lambda i: float(emb[FIRST][i] @ qenc[FIRST].encode(
        [r["question"]], normalize_embeddings=True)[0]))
    pairs.append((r["question"], texts[pos], 1.0))
    negs = [int(i) for i in ranked if csec[i] != r["section"]][:4]
    pairs += [(r["question"], texts[i], 0.0) for i in negs]
random.shuffle(pairs)
print(f"reranker training pairs: {len(pairs)}")

tok = AutoTokenizer.from_pretrained(BASE_CE)
ce = AutoModelForSequenceClassification.from_pretrained(BASE_CE).cuda().train()
opt = torch.optim.AdamW(ce.parameters(), lr=2e-5, weight_decay=0.01)
EPOCHS, BS = 2, 16
steps = EPOCHS * math.ceil(len(pairs) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
lossf = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda")
step = 0
for ep in range(EPOCHS):
    random.shuffle(pairs); tot = 0.0
    for b in range(0, len(pairs), BS):
        batch = pairs[b:b + BS]
        enc = tok([p[0] for p in batch], [p[1] for p in batch], truncation=True, max_length=384,
                  padding=True, return_tensors="pt").to("cuda")
        y = torch.tensor([p[2] for p in batch], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(ce(**enc).logits.squeeze(-1).float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); step += 1
    print(f"epoch {ep + 1}: mean loss {tot / math.ceil(len(pairs) / BS):.4f}", flush=True)
ce.save_pretrained(OUT / "ce_tax"); tok.save_pretrained(OUT / "ce_tax")

rr_base, rr_tax = Reranker(BASE_CE), Reranker(str(OUT / "ce_tax"))
CAND = 30
arms = {f"Dense: {n}": (lambda q, n=n: list(dense_rank(n, q)[:50])) for n in DENSE}
arms["BM25"] = lambda q: list(bm25_rank(q)[:50])
arms["Hybrid: FiQA dense + BM25 (RRF)"] = lambda q: rrf(dense_rank(FIRST, q), bm25_rank(q))
arms["FiQA dense -> ms-marco reranker"] = lambda q: rr_base.rerank(q, [int(i) for i in dense_rank(FIRST, q)[:CAND]])
arms["FiQA dense -> tax-tuned reranker (ours)"] = lambda q: rr_tax.rerank(q, [int(i) for i in dense_rank(FIRST, q)[:CAND]])
arms["Hybrid -> tax-tuned reranker (ours)"] = lambda q: rr_tax.rerank(q, rrf(dense_rank(FIRST, q), bm25_rank(q), limit=CAND))

print("\n== Content-based test set (held-out sections) ==")
res_test = evaluate(test, arms)
print("\n== Legacy heading-paraphrase set (100 queries) ==")
res_legacy = evaluate(legacy, arms)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired hit@5 (a vs b)."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return {"b_only": 0, "a_only": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b_only": n01, "a_only": n10, "p": p}


base_arm = f"Dense: {FIRST}"
tests = {name: mcnemar(res_test[base_arm]["_hits5"], res_test[name]["_hits5"])
         for name in res_test if name != base_arm}
for d in (res_test, res_legacy):
    for v in d.values():
        v.pop("_hits5", None)
out = {"test_questions": len(test), "test_sections": len({r['section'] for r in test}),
       "train_pairs": len(pairs), "candidates_reranked": CAND,
       "content_test": res_test, "legacy_heading_set": res_legacy,
       "mcnemar_hit5_vs_app_default": tests}
(OUT / "retrieval_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(tests, indent=1))
