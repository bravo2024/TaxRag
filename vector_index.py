"""Dense vector search with FAISS, falling back to NumPy.

Primary backend: FAISS IndexFlatIP (exact inner-product search over L2-normalised
vectors = cosine similarity). If faiss is not installed, fails to build, or fails at
query time, search falls back to the NumPy matrix product over the same embeddings
(the .npy / .npz files in out/), so results are identical either way.

At 1,237 passages x 384 dims exact search is ~1 ms, so the flat (exact) index is used
in the app; eval/eval_faiss.py benchmarks an approximate HNSW index against it.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)

# TAXRAG_VECTOR_BACKEND=numpy forces the fallback (used to test it).
_FORCE_NUMPY = os.environ.get("TAXRAG_VECTOR_BACKEND", "").lower() == "numpy"


class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        E = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        self.backend, self._faiss = "numpy", None
        if _FORCE_NUMPY:
            return
        try:
            import faiss
            index = faiss.IndexFlatIP(self.E.shape[1])
            index.add(self.E)
            self._faiss, self.backend = index, "faiss"
        except Exception as e:  # faiss missing or broken -> NumPy search
            log.warning("FAISS unavailable (%s); using NumPy search", e)

    def __len__(self) -> int:
        return self.E.shape[0]

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cosine scores, row ids) of the k nearest passages, best first."""
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-9)
        k = min(k, len(self))
        if self._faiss is not None:
            try:
                scores, ids = self._faiss.search(q, k)
                return scores[0], ids[0]
            except Exception as e:
                log.warning("FAISS search failed (%s); using NumPy search", e)
        sims = self.E @ q[0]
        ids = np.argsort(-sims)[:k]
        return sims[ids], ids


# One index per embedding matrix. The array itself is kept in the entry so its id()
# cannot be reused by a different array while the entry exists.
_CACHE: Dict[int, Tuple[np.ndarray, VectorIndex]] = {}


def get_index(embeddings: np.ndarray) -> VectorIndex:
    hit = _CACHE.get(id(embeddings))
    if hit is not None and hit[0] is embeddings:
        return hit[1]
    idx = VectorIndex(embeddings)
    _CACHE[id(embeddings)] = (embeddings, idx)
    return idx
