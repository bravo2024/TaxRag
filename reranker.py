"""Two-stage retrieval for the app: dense + BM25 (reciprocal-rank fusion) -> cross-encoder rerank.

Chosen from the retrieval benchmark (colab/retrieval_results.json) on 277 content-based
questions from 150 held-out sections: MRR@10 0.632 (dense only) -> 0.833 (this pipeline, 30 candidates).
App defaults (15 candidates, 256 tokens) were picked from a CPU sweep: MRR@10 0.853 vs 0.738
dense-only on 80 test questions, about 2.5 s per query on a 4-core laptop CPU.

Plugs into chat_router.chat(retriever=...). The out-of-scope gate is unchanged: if the best
dense cosine similarity is below the threshold, no passages are returned and the router
falls back exactly as before. The reranker only reorders candidates.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from rag import SIMILARITY_THRESHOLD
from vector_index import VectorIndex

# Fine-tuned on content-based questions from training sections of the Act
# (https://huggingface.co/vivekkopthsd/tax-reranker-minilm). Override with TAXRAG_RERANKER,
# e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 for the untuned base reranker.
DEFAULT_CE = os.environ.get("TAXRAG_RERANKER", "vivekkopthsd/tax-reranker-minilm")
_TOK = re.compile(r"[a-z0-9]+")


class RerankRetriever:
    def __init__(self, model, embeddings: np.ndarray, chunks: List[Dict[str, Any]],
                 reranker: str = DEFAULT_CE, threshold: float = SIMILARITY_THRESHOLD,
                 max_length: int = 256):
        from rank_bm25 import BM25Okapi
        self.model, self.chunks, self.threshold, self.max_length = model, chunks, threshold, max_length
        self.index = VectorIndex(embeddings)   # FAISS, NumPy fallback
        self.bm25 = BM25Okapi([_TOK.findall(c["text"].lower()) for c in chunks])
        self.reranker_name = reranker
        self._ce = None

    def _load_ce(self):
        if self._ce is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.reranker_name)
            self._ce = AutoModelForSequenceClassification.from_pretrained(self.reranker_name).eval()
        return self._ce

    def _candidates(self, query: str, n: int) -> Tuple[List[int], float]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_scores, dense = self.index.search(q, 200)
        sparse = np.argsort(-self.bm25.get_scores(_TOK.findall(query.lower())))[:200]
        fused: Dict[int, float] = {}
        for lst in (dense, sparse):
            for r, i in enumerate(lst):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r)
        order = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:n]
        return order, float(dense_scores[0])

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True,
                 candidates: int = 15) -> Tuple[List[Dict[str, Any]], float]:
        cand, max_s = self._candidates(query, candidates)
        if max_s < self.threshold:
            return [], max_s
        scores = None
        if rerank:
            ce = self._load_ce()
            enc = self._tok([query] * len(cand), [self.chunks[i]["text"] for i in cand], truncation=True,
                            max_length=self.max_length, padding=True, return_tensors="pt")
            with self._torch.no_grad():
                scores = ce(**enc).logits.squeeze(-1).float().numpy()
            cand = [cand[j] for j in np.argsort(-scores)]
            scores = np.sort(scores)[::-1]
        out = []
        for rank, i in enumerate(cand[:top_k]):
            c = self.chunks[i]
            out.append({"score": round(float(scores[rank]) if scores is not None else 0.0, 4),
                        "text": c["text"], "title": c["title"], "source": c["source"],
                        "chunk_id": c.get("chunk_id", str(i)),
                        "retriever_source": "hybrid+rerank" if rerank else "hybrid"})
        return out, max_s
