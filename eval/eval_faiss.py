"""FAISS check and benchmark on the app's real corpus embeddings.

1. Equivalence: FAISS IndexFlatIP, the NumPy fallback and the previous sklearn
   cosine_similarity code must return the same top-10 passages for every question,
   for all three embedding models in the app.
2. HNSW benchmark: approximate search (M=32) at several efSearch values against exact
   search: recall@10 of the exact neighbours, section Recall@5 against gold, latency.

Questions: 277 content-based test questions (colab/gen_questions.json, split=test).
Run from the repo root:  python eval/eval_faiss.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

import vector_index  # noqa: E402
from rag import load_index  # noqa: E402

MODELS = ["vivekkopthsd/fiqa-retriever-bge-small", "vivekkopthsd/hinglish-st-33m",
          "vivekkopthsd/tax-retriever-bge-small"]
K = 10

_, chunks = load_index(str(ROOT / "out"))
sec = np.array([int(m.group(1)) if (m := re.match(r"Section (\d+)", c["title"])) else -1 for c in chunks])
qs = [r for r in json.loads((ROOT / "colab" / "gen_questions.json").read_text(encoding="utf-8"))["rows"]
      if r["split"] == "test"]
gold = np.array([r["section"] for r in qs])


def sec_recall5(ids):
    return float(np.mean([g in sec[row[:5]] for g, row in zip(gold, ids)]))


def timed(fn, Q, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(Q)
        best = min(best, time.perf_counter() - t)
    return out, 1000 * best / len(Q)


out = {"questions": len(qs), "passages": len(chunks), "faiss_version": faiss.__version__, "models": {}}
for name in MODELS:
    E = np.load(ROOT / "out" / ("emb_" + name.replace("/", "__") + ".npy"))
    m = SentenceTransformer(name)
    Q = m.encode([r["question"] for r in qs], show_progress_bar=False).astype(np.float32)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    flat = vector_index.VectorIndex(E)
    assert flat.backend == "faiss", flat.backend
    np_idx = vector_index.VectorIndex(E)
    np_idx._faiss, np_idx.backend = None, "numpy"

    faiss_ids = np.array([flat.search(q, K)[1] for q in Q])
    numpy_ids = np.array([np_idx.search(q, K)[1] for q in Q])
    old_ids = np.argsort(-cosine_similarity(Q / np.linalg.norm(Q, axis=1, keepdims=True), E), axis=1)[:, :K]
    faiss_sc = np.array([flat.search(q, K)[0] for q in Q])
    old_sc = np.sort(cosine_similarity(Qn, E), axis=1)[:, ::-1][:, :K]

    # latency, whole query set, per query
    En = flat.E
    (_, _), ms_flat = timed(lambda X: flat._faiss.search(X, K), Qn)
    _, ms_numpy = timed(lambda X: np.argsort(-(X @ En.T), axis=1)[:, :K], Qn)

    res = {"top10_identical_faiss_vs_old": float(np.mean([set(a) == set(b) for a, b in zip(faiss_ids, old_ids)])),
           "top1_identical_faiss_vs_old": float(np.mean(faiss_ids[:, 0] == old_ids[:, 0])),
           "top10_identical_faiss_vs_numpy_fallback": float(np.mean([set(a) == set(b) for a, b in zip(faiss_ids, numpy_ids)])),
           "max_abs_score_diff_vs_old": float(np.abs(faiss_sc - old_sc).max()),
           "section_recall5_exact": sec_recall5(faiss_ids),
           "ms_per_query_faiss_flat": ms_flat, "ms_per_query_numpy": ms_numpy, "hnsw": []}

    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(En.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(En)
    res["hnsw_build_s"] = time.perf_counter() - t
    for ef in (8, 16, 32, 64, 128):
        hnsw.hnsw.efSearch = ef
        (_, ids), ms = timed(lambda X: hnsw.search(X, K), Qn)
        recall = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, faiss_ids)]))
        res["hnsw"].append({"efSearch": ef, "recall10_vs_exact": recall,
                            "section_recall5": sec_recall5(ids), "ms_per_query": ms})
    out["models"][name.split("/")[1]] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != "hnsw"}, indent=1))
    for h in res["hnsw"]:
        print("  HNSW", h)

# the NumPy fallback path, as the app would hit it with faiss absent
vector_index._FORCE_NUMPY = True
fb = vector_index.VectorIndex(np.load(ROOT / "out" / ("emb_" + MODELS[0].replace("/", "__") + ".npy")))
out["forced_fallback_backend"] = fb.backend

p = ROOT / "eval" / "results" / "faiss_eval.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("forced fallback backend:", fb.backend, "| saved", p)
