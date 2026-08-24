#!/usr/bin/env python3
"""TaxRAG multilingual hybrid retriever.

Combines BM25Okapi (rank_bm25) with dense bge-m3 retrieval over a bilingual
(English + Hindi) index, fuses the two rankings with Reciprocal Rank Fusion
(RRF), applies a language-aware boost, and optionally reranks with a
multilingual cross-encoder (BAAI/bge-reranker-v2-m3).

Parallel to the legacy `rag.retrieve` — the old code path is untouched.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi

log = logging.getLogger("taxrag.hybrid")

DEFAULT_RRF_K = 60
DEFAULT_LANG_BOOST = 1.5
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Word-run matcher: the full Devanagari block (U+0900–U+09FF, keeps combining
# vowel signs attached to their base letter) alongside Latin letters/digits, so
# mixed Devanagari + Latin text (Hindi, Hinglish, statutory Latin terms) is
# tokenized as real words instead of single characters.
WORD_RE = re.compile(r"[\u0900-\u09FFA-Za-z0-9]+")

# Minimal Hindi stopwords for BM25 (whole-word Devanagari forms).
_HI_STOP = {
    "मैं", "मेरा", "का", "की", "के", "है", "हैं", "नहीं", "में", "से", "पर",
    "और", "या",
}

# Minimum free GPU memory (GB) before we try CUDA (the GPU is often shared).
MIN_FREE_GPU_GB = 3.0

# English stopwords used for the BM25-EN tokenizer.
_EN_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have", "having",
    "he", "her", "here", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your",
}


# ---------------------------------------------------------------------------
# Tokenizers (separate per language)
# ---------------------------------------------------------------------------

def tokenize_en(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 1 and t not in _EN_STOP]


def tokenize_hi(text: str) -> List[str]:
    return [t for t in WORD_RE.findall(text.lower())
            if len(t) > 1 and t not in _HI_STOP]


def detect_language(query: str) -> str:
    """'hi' when dominated by Devanagari, 'en' when latin-only, else 'hinglish'."""
    n_deva = len(DEVANAGARI_RE.findall(query))
    n_latin = len(LATIN_RE.findall(query))
    total = n_deva + n_latin
    if total == 0:
        return "en"
    r = n_deva / total
    if r >= 0.70:
        return "hi"
    if r <= 0.10:
        return "en"
    return "hinglish"


def tokenize_query(query: str, lang: str) -> List[str]:
    if lang == "hi":
        return tokenize_hi(query)
    if lang == "en":
        return tokenize_en(query)
    return list(dict.fromkeys(tokenize_en(query) + tokenize_hi(query)))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """BM25 + dense (bge-m3) retrieval over the bilingual index with RRF.

    Expected files in `out_dir`:
      index_m3_bilingual.npz   union of EN + HI bge-m3 embeddings
        (keys: embeddings, chunk_texts, chunk_titles, chunk_sources,
               chunk_ids, languages)
      index_m3.npz            EN-only bge-m3 index (built by build_multilingual.py)
      index_m3_hi.npz         HI-only bge-m3 index (built by build_multilingual.py)
      passages_hi.json        {id, en_text, hi_text, section} translation log
    """

    def __init__(
        self,
        out_dir: str = "./out",
        model_name: str = "BAAI/bge-m3",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        rrf_k: int = DEFAULT_RRF_K,
        lang_boost: float = DEFAULT_LANG_BOOST,
        use_reranker: bool = False,
        lazy_models: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.model_name = model_name
        self.reranker_name = reranker_name
        self.device = device or _pick_device()
        self.rrf_k = rrf_k
        self.lang_boost = lang_boost
        self.use_reranker = use_reranker
        self._reranker = None
        self._model = None
        self._lock = threading.Lock()

        self._load_indices()
        self._build_bm25()

        if not lazy_models:
            self._load_dense()

    # -- model / device helpers ----------------------------------------
    def _load_dense(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            log.info("Loading dense model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def _load_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            log.info("Loading reranker %s on %s", self.reranker_name, self.device)
            self._reranker = CrossEncoder(self.reranker_name, device=self.device)

    # -- index loading --------------------------------------------------
    def _load_indices(self) -> None:
        path = os.path.join(self.out_dir, "index_m3_bilingual.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bilingual index not found: {path}. Run build_multilingual.py first."
            )
        d = np.load(path, allow_pickle=True)
        emb = d["embeddings"].astype("float32")
        self.embeddings = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.languages = np.array([str(x) for x in d["languages"]])
        self.chunks: List[Dict[str, Any]] = []
        for i in range(len(d["chunk_texts"])):
            self.chunks.append({
                "text": str(d["chunk_texts"][i]),
                "title": str(d["chunk_titles"][i]),
                "source": str(d["chunk_sources"][i]),
                "chunk_id": str(d["chunk_ids"][i]),
                "language": self.languages[i],
            })
        log.info("Bilingual index: %d vectors (%d EN, %d HI), dim %d",
                 len(self.chunks), int((self.languages == "en").sum()),
                 int((self.languages == "hi").sum()), emb.shape[1])

    # -- BM25 -----------------------------------------------------------
    def _build_bm25(self) -> None:
        en_idx = [i for i, c in enumerate(self.chunks) if c["language"] == "en"]
        hi_idx = [i for i, c in enumerate(self.chunks) if c["language"] == "hi"]
        self._en_global = en_idx
        self._hi_global = hi_idx
        self.bm25_en = BM25Okapi([tokenize_en(self.chunks[i]["text"]) for i in en_idx])
        self.bm25_hi = BM25Okapi([tokenize_hi(self.chunks[i]["text"]) for i in hi_idx])
        log.info("BM25 built: %d EN passages, %d HI passages", len(en_idx), len(hi_idx))

    # -- retrieval primitives --------------------------------------------
    def _dense(self, query: str, top: int = 200) -> List[Tuple[int, float]]:
        self._load_dense()
        qv = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qv
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order]

    def _bm25(self, query: str, lang: str, top: int = 100) -> List[Tuple[int, float]]:
        res: List[Tuple[int, float]] = []
        en_tok = tokenize_en(query)
        hi_tok = tokenize_hi(query)
        if en_tok and self.bm25_en:
            sc = self.bm25_en.get_scores(en_tok)
            res += [(self._en_global[i], float(s)) for i, s in enumerate(sc) if s > 0]
        if hi_tok and self.bm25_hi:
            sc = self.bm25_hi.get_scores(hi_tok)
            res += [(self._hi_global[i], float(s)) for i, s in enumerate(sc) if s > 0]
        res.sort(key=lambda x: -x[1])
        return res[:top]

    # -- public API ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: Optional[bool] = None,
        candidates: int = 30,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Hybrid BM25 + dense retrieval with RRF (+ optional cross-encoder).

        Returns (results, top_score). Each result carries:
          score, text, title, source, chunk_id, language, retriever_source
        When rerank=True the cross-encoder reranks the top `candidates`
        passages down to `top_k` and adds a `rerank_score` field.
        """
        rerank = self.use_reranker if rerank is None else rerank
        lang = detect_language(query)

        dense = self._dense(query, top=200)
        bm25 = self._bm25(query, lang, top=100)

        rrf: Dict[int, float] = defaultdict(float)
        src: Dict[int, set] = defaultdict(set)
        for rank, (gi, _s) in enumerate(dense):
            rrf[gi] += 1.0 / (self.rrf_k + rank + 1)
            src[gi].add("dense")
        for rank, (gi, _s) in enumerate(bm25):
            rrf[gi] += 1.0 / (self.rrf_k + rank + 1)
            src[gi].add("bm25")

        # Language-aware boost: Hindi query boosts HI passages, English boosts
        # EN passages, Hinglish weights both equally.
        for gi in list(rrf):
            if self.languages[gi] == lang and lang in ("hi", "en"):
                rrf[gi] *= self.lang_boost

        ranked = sorted(rrf.items(), key=lambda x: -x[1])
        limit = candidates if rerank else top_k
        results: List[Dict[str, Any]] = []
        for gi, score in ranked:
            c = self.chunks[gi]
            results.append({
                "score": round(score, 4),
                "text": c["text"],
                "title": c["title"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "language": c["language"],
                "retriever_source": "+".join(sorted(src[gi])),
            })
            if len(results) >= limit:
                break

        if rerank and results:
            results = self.rerank(query, results, top_k=top_k)
        top_score = results[0]["score"] if results else 0.0
        return results, top_score

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Cross-encoder reranking of candidate passages (in place)."""
        self._load_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        with self._lock:
            scores = self._reranker.predict(pairs, batch_size=16)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 4)
        out = sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]
        for c in out:
            c["score"] = c["rerank_score"]
            c["retriever_source"] = c.get("retriever_source", "dense") + "+reranker"
        return out


# ---------------------------------------------------------------------------
# NewStackRetriever — 2026 encoder (EmbeddingGemma/Qwen3-Embedding) + Qwen3
# Reranker. Sequentially loads/unloads models to cap peak RAM (~0.9 GB int8),
# which fits the 1 GB Streamlit tier. The legacy HybridRetriever above is
# untouched for A/B comparison.
# ---------------------------------------------------------------------------

GEMMA_ENCODER = os.environ.get("TAXRAG_GEMMA_ENCODER", "google/embeddinggemma-300m")
QWEN3_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_RERANKER = "Qwen/Qwen3-Reranker-0.6B"


class NewStackRetriever:
    """BM25 + dense (EmbeddingGemma-300m) retrieval with RRF + Qwen3 reranker.

    Reads `index_gemma_bilingual.npz` (or `index_qwen3_bilingual.npz` if the
    encoder fell back to Qwen3-Embedding). Model lifecycle is strict:
    encoder loaded only while its dense scores are being computed, then
    unloaded *before* the reranker is loaded, so peak RAM is the larger of the
    two (not their sum) — ~0.6–0.9 GB int8.

    Public API mirrors `HybridRetriever.retrieve` for a drop-in swap.
    """

    def __init__(
        self,
        out_dir: str = "./out",
        model_name: Optional[str] = None,
        reranker_name: str = QWEN3_RERANKER,
        device: Optional[str] = None,
        rrf_k: int = DEFAULT_RRF_K,
        lang_boost: float = DEFAULT_LANG_BOOST,
        use_reranker: bool = False,
        hold_models: bool = False,
        lazy_models: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.device = device or _pick_device()
        self.rrf_k = rrf_k
        self.lang_boost = lang_boost
        self.use_reranker = use_reranker
        # hold_models=True keeps both models resident (fast eval / GPU boxes);
        # False (Streamlit default) strictly unloads between phases.
        self.hold_models = hold_models
        self._model: Any = None
        self._reranker: Any = None
        self._lock = threading.Lock()

        self.index_path = self._pick_index()
        self._load_indices()

        encoder, ok = _load_encoder(model_name or GEMMA_ENCODER, QWEN3_ENCODER,
                                    self.device, lazy=False)
        if not ok:
            log.warning("All candidate encoders failed to load")
        self.model_name = encoder
        self.reranker_name = reranker_name

        self._build_bm25()
        if not lazy_models:
            self._load_dense()

    # -- index / model helpers ------------------------------------------
    def _pick_index(self) -> str:
        for name in ("index_gemma_bilingual.npz", "index_qwen3_bilingual.npz"):
            p = os.path.join(self.out_dir, name)
            if os.path.exists(p):
                log.info("New stack index: %s", p)
                return p
        raise FileNotFoundError(
            f"No index_gemma/index_qwen3 bilingual index in {self.out_dir}. "
            "Run build_gemma_index.py first."
        )

    def _load_indices(self) -> None:
        d = np.load(self.index_path, allow_pickle=True)
        emb = d["embeddings"].astype("float32")
        self.embeddings = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.languages = np.array([str(x) for x in d["languages"]])
        self.chunks: List[Dict[str, Any]] = []
        for i in range(len(d["chunk_texts"])):
            self.chunks.append({
                "text": str(d["chunk_texts"][i]),
                "title": str(d["chunk_titles"][i]),
                "source": str(d["chunk_sources"][i]),
                "chunk_id": str(d["chunk_ids"][i]),
                "language": self.languages[i],
            })
        log.info("New-stack bilingual index: %d vectors (%d EN, %d HI), dim %d",
                 len(self.chunks), int((self.languages == "en").sum()),
                 int((self.languages == "hi").sum()), emb.shape[1])

    def _load_dense(self) -> None:
        if self._model is None:
            log.info("Loading encoder %s on %s", self.model_name, self.device)
            self._model = _build_encoder(self.model_name, self.device)

    def _unload_dense(self) -> None:
        if self._model is not None and not self.hold_models:
            _release_model(self._model)
            self._model = None
            log.info("Unloaded encoder %s", self.model_name)

    def _load_reranker(self) -> None:
        if self._reranker is None:
            log.info("Loading reranker %s on %s", self.reranker_name, self.device)
            self._reranker = _build_reranker(self.reranker_name, self.device)

    def _unload_reranker(self) -> None:
        if self._reranker is not None and not self.hold_models:
            _release_model(self._reranker)
            self._reranker = None
            log.info("Unloaded reranker %s", self.reranker_name)

    # -- BM25 -----------------------------------------------------------
    def _build_bm25(self) -> None:
        en_idx = [i for i, c in enumerate(self.chunks) if c["language"] == "en"]
        hi_idx = [i for i, c in enumerate(self.chunks) if c["language"] == "hi"]
        self._en_global = en_idx
        self._hi_global = hi_idx
        self.bm25_en = BM25Okapi([tokenize_en(self.chunks[i]["text"]) for i in en_idx])
        self.bm25_hi = BM25Okapi([tokenize_hi(self.chunks[i]["text"]) for i in hi_idx])
        log.info("New-stack BM25 built: %d EN, %d HI", len(en_idx), len(hi_idx))

    # -- retrieval primitives --------------------------------------------
    def _dense(self, query: str, top: int = 200) -> List[Tuple[int, float]]:
        self._load_dense()
        qv = self._model.encode([query], prompt_name="query",
                                normalize_embeddings=True)[0]
        sims = self.embeddings @ qv
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order]

    def _bm25(self, query: str, lang: str, top: int = 100) -> List[Tuple[int, float]]:
        res: List[Tuple[int, float]] = []
        en_tok = tokenize_en(query)
        hi_tok = tokenize_hi(query)
        if en_tok and self.bm25_en:
            sc = self.bm25_en.get_scores(en_tok)
            res += [(self._en_global[i], float(s)) for i, s in enumerate(sc) if s > 0]
        if hi_tok and self.bm25_hi:
            sc = self.bm25_hi.get_scores(hi_tok)
            res += [(self._hi_global[i], float(s)) for i, s in enumerate(sc) if s > 0]
        res.sort(key=lambda x: -x[1])
        return res[:top]

    # -- public API ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: Optional[bool] = None,
        candidates: int = 30,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Hybrid BM25 + dense (new encoder) with RRF + optional Qwen3 rerank.

        Encoder is unloaded immediately after dense scoring; the reranker is
        loaded only afterwards (sequential RAM cap). Returns the same result
        schema as `HybridRetriever.retrieve`.
        """
        rerank = self.use_reranker if rerank is None else rerank
        lang = detect_language(query)

        dense = self._dense(query, top=200)
        bm25 = self._bm25(query, lang, top=100)
        # Free the encoder's RAM before any reranker loads.
        self._unload_dense()

        rrf: Dict[int, float] = defaultdict(float)
        src: Dict[int, set] = defaultdict(set)
        for rank, (gi, _s) in enumerate(dense):
            rrf[gi] += 1.0 / (self.rrf_k + rank + 1)
            src[gi].add("dense")
        for rank, (gi, _s) in enumerate(bm25):
            rrf[gi] += 1.0 / (self.rrf_k + rank + 1)
            src[gi].add("bm25")

        for gi in list(rrf):
            if self.languages[gi] == lang and lang in ("hi", "en"):
                rrf[gi] *= self.lang_boost

        ranked = sorted(rrf.items(), key=lambda x: -x[1])
        limit = candidates if rerank else top_k
        results: List[Dict[str, Any]] = []
        for gi, score in ranked:
            c = self.chunks[gi]
            results.append({
                "score": round(score, 4),
                "text": c["text"],
                "title": c["title"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "language": c["language"],
                "retriever_source": "+".join(sorted(src[gi])),
            })
            if len(results) >= limit:
                break

        if rerank and results:
            results = self.rerank(query, results, top_k=top_k)
        elif not rerank:
            self._unload_reranker()
        top_score = results[0]["score"] if results else 0.0
        return results, top_score

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Qwen3 cross-encoder reranking of candidate passages (in place)."""
        self._load_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        with self._lock:
            scores = self._reranker.predict(pairs, batch_size=16)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 4)
        out = sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]
        for c in out:
            c["score"] = c["rerank_score"]
            c["retriever_source"] = c.get("retriever_source", "dense") + "+reranker"
        self._unload_reranker()
        return out


def _load_encoder(primary: str, fallback: str, device: str,
                  lazy: bool = True) -> Tuple[Optional[str], bool]:
    """Try `primary`, fall back to `fallback`. Returns (name, ok)."""
    if not lazy:
        for name in (primary, fallback):
            try:
                _build_encoder(name, device)
                return name, True
            except Exception as exc:  # noqa: BLE001
                log.warning("Encoder %s failed to load (%s); trying next", name, exc)
        return None, False
    return primary, True


def _build_encoder(name: str, device: str):
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    return SentenceTransformer(name, device=device)


def _build_reranker(name: str, device: str):
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    return CrossEncoder(name, device=device)


def _release_model(model: Any) -> None:
    import gc  # noqa: PLC0415
    if hasattr(model, "to") and hasattr(model, "device"):
        try:
            model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pick_device() -> str:
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            if free_gb >= MIN_FREE_GPU_GB:
                return "cuda"
            log.warning("Only %.1fGB GPU free (shared); running on CPU", free_gb)
        except Exception as exc:  # noqa: BLE001
            log.warning("GPU mem check failed (%s); CPU", exc)
    return "cpu"


# ---------------------------------------------------------------------------
# Convenience: load the bilingual metadata (texts/titles/languages) from disk
# ---------------------------------------------------------------------------

def load_bilingual_meta(out_dir: str = "./out") -> Dict[str, Any]:
    path = os.path.join(out_dir, "index_m3_bilingual.npz")
    d = np.load(path, allow_pickle=True)
    return {
        "chunk_texts": [str(x) for x in d["chunk_texts"]],
        "chunk_titles": [str(x) for x in d["chunk_titles"]],
        "chunk_sources": [str(x) for x in d["chunk_sources"]],
        "chunk_ids": [str(x) for x in d["chunk_ids"]],
        "languages": [str(x) for x in d["languages"]],
        "num_passages": len(d["chunk_texts"]),
        "num_en": int((np.array([str(x) for x in d["languages"]]) == "en").sum()),
        "num_hi": int((np.array([str(x) for x in d["languages"]]) == "hi").sum()),
        "embedding_dim": d["embeddings"].shape[1],
    }


def load_hindi_passages(out_dir: str = "./out") -> List[Dict[str, str]]:
    path = os.path.join(out_dir, "passages_hi.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)
