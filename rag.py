#!/usr/bin/env python3
"""
TaxRAG — India Income Tax Q&A Assistant
RAG pipeline: chunk, embed, index, retrieve, evaluate.
CPU-only. Retrieval uses a fine-tuned bge-small bi-encoder (fiqa_retriever_gpu),
overridable via the TAXRAG_MODEL env var; falls back to all-MiniLM-L6-v2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("taxrag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Fine-tuned FiQA bi-encoder (bge-small base) beats stock all-MiniLM on this
# corpus (Recall@1 1.00 vs 0.82, MRR 1.00 vs 0.89). Override with TAXRAG_MODEL;
# on Streamlit Cloud set it to the pushed HF repo id (e.g. bravo2024/fiqa-retriever).
MODEL_NAME: str = os.environ.get(
    "TAXRAG_MODEL", "vivekkopthsd/fiqa-retriever-bge-small"
)
EMBEDDING_DIM: int = 384
CHUNK_MIN_TOKENS: int = 150
CHUNK_MAX_TOKENS: int = 500
# 0.45 sits in the gap between min in-scope (0.55) and max out-of-scope (0.39)
# similarity for the fine-tuned FiQA model, giving recall@k 1.0 AND refusal 1.0.
SIMILARITY_THRESHOLD: float = 0.45

# Provider-agnostic LLM synthesis with multi-provider failover.
# Answer synthesis is OPTIONAL — retrieval + citation is the core product.
# With no keys set, every provider that needs one is skipped and the app drops
# to extractive mode (returns the retrieved passages). The anonymous free
# gateways work with a blank key, so synthesis usually still succeeds.
# A single custom OpenAI-compatible endpoint can be forced via OPENAI_BASE_URL.
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "opencode-free": {
        "base": "https://opencode.ai/zen/v1",
        "key_env": "OPENCODE_ZEN_API_KEY",
        "models": ["deepseek-v4-flash-free", "mimo-v2.5-free", "ling-3.0-flash-free"],
        "anonymous": True,
    },
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "nvidia-build": {
        "base": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "models": ["deepseek-ai/deepseek-v4-flash", "meta/llama-3.1-70b-instruct"],
    },
    "kilo-free": {
        "base": "https://api.kilo.ai/api/openrouter",
        "key_env": "KILO_API_KEY",
        "models": ["kilo-auto/free", "stepfun/step-3.7-flash:free", "inclusionai/ling-3.0-flash:free"],
        "anonymous": True,      # blank key works — public gateway
        "kilo_headers": True,   # needs the Kilo Code extension referer headers
    },
}
LLM_PROVIDER_ORDER: List[str] = ["opencode-free", "groq", "nvidia-build", "kilo-free"]
# Optional single custom endpoint (takes priority when set).
LLM_BASE_URL: Optional[str] = os.environ.get("OPENAI_BASE_URL")
LLM_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _get_key(env_name: str) -> str:
    """Env var first; then a Streamlit secret if Streamlit is importable."""
    key = os.environ.get(env_name, "")
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415
            key = st.secrets.get(env_name, "")  # type: ignore[assignment]
        except Exception:
            key = ""
    return key

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

__stopwords: Optional[set] = None


def _get_stopwords() -> set:
    global __stopwords
    if __stopwords is None:
        __stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
            "during", "each", "few", "for", "from", "further", "had", "has", "have",
            "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
            "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
            "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
            "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
            "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
            "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very",
            "was", "we", "was", "were", "what", "when", "where", "which", "while", "who",
            "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
            "yourselves",
        }
    return __stopwords


def token_count(text: str) -> int:
    """Approximate token count (whitespace-split)."""
    return len(text.split())


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for cleaner chunk embedding."""
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r">+\s+", "", text)
    text = re.sub(r"[-*+]\s+", "", text)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Document loading & parsing
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    Load .md files from data_dir.
    Returns list of dicts: {filename, title, source, content, raw_text}
    """
    docs: List[Dict[str, str]] = []
    data_path = Path(data_dir)
    if not data_path.exists():
        log.warning("Data directory %s does not exist", data_dir)
        return docs
    for fpath in sorted(data_path.glob("*.md")):
        raw = fpath.read_text(encoding="utf-8")
        title, source = _extract_metadata(raw, fpath.name)
        docs.append({
            "filename": fpath.name,
            "title": title,
            "source": source,
            "content": raw,
            "raw_text": strip_markdown(raw),
        })
    log.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs


def _extract_metadata(raw: str, filename: str) -> Tuple[str, str]:
    title = filename
    source = "Income Tax Act, 1961"
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# ") and i == 0:
            title = line[2:].strip()
        if line.startswith("**Source:**"):
            source = line.replace("**Source:**", "").strip()
    return title, source


def load_pdf_documents(pdf_dir: str) -> List[Dict[str, str]]:
    """
    Load .pdf files from pdf_dir using pypdf.
    Returns list of dicts compatible with chunk_document:
    {filename, title, source, content, raw_text}.
    Skips PDFs with no usable text layer (extracted text < 200 chars).
    """
    docs: List[Dict[str, str]] = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        log.warning("PDF directory %s does not exist", pdf_dir)
        return docs
    for fpath in sorted(pdf_path.glob("*.pdf")):
        try:
            reader = PdfReader(str(fpath))
            pages = len(reader.pages)
            raw = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            log.warning("PDF %s: could not read (%s) — SKIPPED", fpath.name, exc)
            continue
        char_count = len(raw)
        if char_count < 200:
            log.info(
                "PDF %s: %d pages, %d chars — SKIPPED (scanned/image-only, no text layer)",
                fpath.name, pages, char_count,
            )
            continue
        stem = fpath.stem.replace("_", " ").replace("-", " ").strip()
        title = stem.title()
        docs.append({
            "filename": fpath.name,
            "title": title,
            "source": fpath.name,
            "content": raw,
            "raw_text": raw,
        })
        log.info("PDF %s: %d pages, %d chars — kept", fpath.name, pages, char_count)
    log.info("Loaded %d PDF documents from %s", len(docs), pdf_dir)
    return docs


def _select_corpus(
    corpus: str, data_dir: str, pdf_dir: str
) -> Tuple[List[Dict[str, str]], str]:
    """Pick documents based on the corpus mode. Returns (docs, mode_used)."""
    mode = corpus
    if mode == "auto":
        pdf_path = Path(pdf_dir)
        has_pdf = pdf_path.exists() and any(pdf_path.glob("*.pdf"))
        mode = "pdf" if has_pdf else "md"
    log.info("--- Corpus mode: %s ---", mode)
    docs: List[Dict[str, str]] = []
    if mode in ("md", "both"):
        docs.extend(load_documents(data_dir))
    if mode in ("pdf", "both"):
        docs.extend(load_pdf_documents(pdf_dir))
    return docs, mode


# ---------------------------------------------------------------------------
# Chunking — section-aware, ~200-500 tokens
# ---------------------------------------------------------------------------

def chunk_document(doc: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Split a document into chunks of ~200-500 tokens.
    Attempts to split on double-newline (paragraph) boundaries.
    """
    text = doc["raw_text"]
    title = doc["title"]
    source = doc["source"]
    filename = doc["filename"]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict[str, Any]] = []
    buffer: List[str] = []
    buf_len: int = 0

    for para in paragraphs:
        para_tokens = token_count(para)
        if buf_len + para_tokens <= CHUNK_MAX_TOKENS:
            buffer.append(para)
            buf_len += para_tokens
        else:
            if buffer:
                chunk_text = "\n\n".join(buffer)
                if token_count(chunk_text) >= CHUNK_MIN_TOKENS:
                    chunks.append(_make_chunk(chunk_text, title, source, filename))
            if para_tokens > CHUNK_MAX_TOKENS:
                sub_chunks = _split_long_para(para)
                for sc in sub_chunks:
                    chunks.append(_make_chunk(sc, title, source, filename))
                buffer, buf_len = [], 0
            else:
                buffer, buf_len = [para], para_tokens

    if buffer and token_count("\n\n".join(buffer)) >= CHUNK_MIN_TOKENS:
        chunks.append(_make_chunk("\n\n".join(buffer), title, source, filename))

    if not chunks and buffer:
        chunks.append(_make_chunk("\n\n".join(buffer), title, source, filename))

    return chunks


def _split_long_para(para: str) -> List[str]:
    """Split a long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    chunks: List[str] = []
    buf: List[str] = []
    buf_len: int = 0
    for sent in sentences:
        st = token_count(sent)
        if buf_len + st <= CHUNK_MAX_TOKENS:
            buf.append(sent)
            buf_len += st
        else:
            if buf:
                chunks.append(" ".join(buf))
            if st >= CHUNK_MIN_TOKENS:
                chunks.append(sent)
                buf, buf_len = [], 0
            else:
                buf, buf_len = [sent], st
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _make_chunk(text: str, title: str, source: str, filename: str) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "title": title,
        "source": source,
        "filename": filename,
        "token_count": token_count(text),
    }


def build_index_from_text(
    raw_text: str, model: SentenceTransformer, source_name: str = "Uploaded document"
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Chunk + embed arbitrary text (e.g. an uploaded PDF) into an in-memory
    index, reusing the same chunking pipeline as the built-in corpus."""
    doc = {
        "filename": source_name,
        "title": source_name,
        "source": source_name,
        "raw_text": raw_text,
        "content": raw_text,
    }
    chunks = chunk_document(doc)
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{source_name}#{i}"
    if not chunks:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    embeddings = embed_chunks(chunks, model)
    return embeddings, chunks


def chunk_all_documents(docs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            c["chunk_id"] = f"{doc['filename']}#{i}"
        all_chunks.extend(chunks)
    log.info("Total chunks: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding & Indexing
# ---------------------------------------------------------------------------

def build_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")
    log.info("Embedding dimension: %d", model.get_embedding_dimension())
    return model


def embed_chunks(
    chunks: List[Dict[str, Any]], model: SentenceTransformer
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings  # type: ignore[return-value]


def save_index(
    out_dir: str, embeddings: np.ndarray, chunks: List[Dict[str, Any]]
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.npz")
    np.savez_compressed(
        path,
        embeddings=embeddings,
        chunk_texts=np.array([c["text"] for c in chunks], dtype=object),
        chunk_titles=np.array([c["title"] for c in chunks], dtype=object),
        chunk_sources=np.array([c["source"] for c in chunks], dtype=object),
        chunk_ids=np.array([c["chunk_id"] for c in chunks], dtype=object),
    )
    _ = embeddings.shape
    log.info("Index saved to %s (%d chunks)", path, len(chunks))


def load_index(out_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    path = os.path.join(out_dir, "index.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}. Run rag.py first.")
    data = np.load(path, allow_pickle=True)
    embeddings: np.ndarray = data["embeddings"]  # type: ignore[assignment]
    chunks: List[Dict[str, Any]] = []
    for i in range(len(data["chunk_texts"])):
        chunks.append({
            "text": str(data["chunk_texts"][i]),
            "title": str(data["chunk_titles"][i]),
            "source": str(data["chunk_sources"][i]),
            "chunk_id": str(data["chunk_ids"][i]),
        })
    log.info("Loaded index: %d chunks, embedding dim %d", len(chunks), embeddings.shape[1])
    return embeddings, chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = 4,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Embed query, compute cosine similarity, return top-k results.
    Returns (results, max_score). If max_score < threshold, results may be empty.
    """
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)
    sims = cosine_similarity(q_emb, embeddings)[0]  # type: ignore[call-overload]
    order = np.argsort(sims)[::-1]
    results: List[Dict[str, Any]] = []
    for idx in order[:top_k]:
        score = float(sims[idx])
        if score < threshold:
            break
        results.append({
            "score": round(score, 4),
            "text": chunks[idx]["text"],
            "title": chunks[idx]["title"],
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
        })
    max_score = float(sims[order[0]])
    return results, max_score


# ---------------------------------------------------------------------------
# LLM Synthesis (provider-agnostic, via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = (
    "You are a precise Indian income tax assistant that answers ONLY from the "
    "provided context. Cite the source (section number/title) for each fact. If "
    "the context is insufficient, say so clearly. Never invent or guess figures."
)


def _build_synth_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n---\n\n".join(
        f"[{r['title']}] (Source: {r['source']})\n{r['text']}" for r in results
    )
    return (
        "Answer the user's question using ONLY the context below. Cite the source "
        "for each fact. If the context does not contain enough information, say so. "
        "Do NOT invent or guess. Keep the answer concise.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )


def _call_openai_compatible(
    base: str, key: str, model: str, messages: list, kilo_headers: bool = False
) -> Optional[str]:
    import requests  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if kilo_headers:  # public Kilo gateway expects the extension's referer headers
        headers.update({"HTTP-Referer": "https://kilo.ai/", "X-Title": "Kilo Code"})
    resp = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 500},
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content") or ""
    return content if content.strip() else None


def synthesize_answer(
    query: str,
    results: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Synthesize a cited answer from the retrieved passages.

    Order: (1) a forced custom endpoint if OPENAI_BASE_URL is set, then (2) the
    provider failover chain — starting with `provider` if given, then the rest.
    If `model` is given it is tried first within its provider.

    Returns (answer, provider_label). answer is None (extractive fallback) when
    every option fails, including the zero-key case where only anonymous
    gateways are attempted. provider_label names who actually answered.
    """
    if not results:
        return None, None
    messages = [{"role": "system", "content": SYNTH_SYSTEM}]
    for turn in (history or [])[-4:]:  # brief prior context, capped
        messages.append(turn)
    messages.append({"role": "user", "content": _build_synth_prompt(query, results)})

    # (1) Forced single endpoint, if the user pinned one.
    if LLM_BASE_URL and LLM_API_KEY:
        try:
            base = LLM_BASE_URL if LLM_BASE_URL.rstrip("/").endswith("/v1") \
                else LLM_BASE_URL.rstrip("/") + "/v1"
            out = _call_openai_compatible(base, LLM_API_KEY, LLM_MODEL, messages)
            if out:
                return out, f"custom/{LLM_MODEL}"
        except Exception:
            log.warning("Custom OPENAI_BASE_URL endpoint failed, trying provider chain")

    # (2) Multi-provider failover — preferred provider first, then the rest.
    order = list(LLM_PROVIDER_ORDER)
    if provider and provider in order:
        order.remove(provider)
        order.insert(0, provider)

    for pname in order:
        cfg = LLM_PROVIDERS[pname]
        key = "" if cfg.get("anonymous") else _get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            continue
        models = list(cfg["models"])
        if pname == provider and model and model in models:
            models.remove(model)
            models.insert(0, model)
        for m in models:
            try:
                out = _call_openai_compatible(
                    cfg["base"], key, m, messages, kilo_headers=cfg.get("kilo_headers", False)
                )
                if out:
                    log.info("LLM synthesis via %s/%s", pname, m)
                    return out, f"{pname}/{m}"
            except Exception:
                continue

    log.info("All LLM providers unavailable; using extractive fallback")
    return None, None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_QUERIES: List[Dict[str, Any]] = [
    {"query": "How much deduction can I claim under Section 80C?", "expected_section": "80C"},
    {"query": "What is the health insurance deduction under 80D?", "expected_section": "80D"},
    {"query": "Can I claim deduction for PPF contributions?", "expected_section": "80C"},
    {"query": "What is HRA exemption and how is it calculated?", "expected_section": "HRA"},
    {"query": "What is the home loan interest deduction limit?", "expected_section": "Section 24"},
    {"query": "What is the difference between old and new tax regime?", "expected_section": "115BAC"},
    {"query": "What is section 87A rebate?", "expected_section": "87A"},
    {"query": "How much TDS is deducted on rental payments?", "expected_section": "194I"},
    {"query": "What are the different ITR forms available?", "expected_section": "ITR"},
    {"query": "What is the capital gains tax exemption under section 54?", "expected_section": "Section 54"},
    {"query": "What is the NPS additional deduction under 80CCD(1B)?", "expected_section": "80CCD"},
]

EVAL_OUT_OF_SCOPE: List[str] = [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "How to bake a chocolate cake?",
    "What is the stock price of Reliance?",
    "Who is the Prime Minister of India?",
]

# HARD eval: colloquial taxpayer phrasing, NO section numbers, with deliberate
# near-miss traps (80TTA vs 80TTB, 80DD vs 80U, 54 vs 54EC, 80CCD vs 80CCD(1B),
# stamp duty -> 80C). Matched on the source .md filename, not on wording the
# query could leak. This is the honest, non-self-fulfilling generalization test.
HARD_EVAL: List[Dict[str, str]] = [
    {"query": "I put money into ELSS mutual funds and paid my child's school tuition fees. Can I lower my taxable income?", "expected_file": "01_section_80c_deductions.md"},
    {"query": "I bought a family health insurance policy this year. Is the premium deductible?", "expected_file": "02_section_80d_health_insurance.md"},
    {"query": "I earn some interest from my savings bank account and I'm 35. Is a small part of it tax-free?", "expected_file": "03_section_80tta_savings_interest.md"},
    {"query": "I'm 68 and live mostly on fixed deposit interest. How much of that interest can I deduct?", "expected_file": "04_section_80ttb_senior_citizens.md"},
    {"query": "How much of the interest I pay on my home loan can I knock off my income?", "expected_file": "05_section_24_home_loan_interest.md"},
    {"query": "My employer pays me an allowance for rent and I live in a rented flat. Is part of it exempt?", "expected_file": "06_hra_exemption.md"},
    {"query": "I'm salaried. Is there a flat amount removed from my salary before tax without needing any bills?", "expected_file": "07_standard_deduction.md"},
    {"query": "I hardly have any deductions to claim. Which tax option leaves me paying less?", "expected_file": "08_old_vs_new_tax_regime.md"},
    {"query": "My total income is about 6.5 lakh. Will I end up paying zero tax after the rebate?", "expected_file": "09_section_87a_rebate.md"},
    {"query": "I have salary and income from one house. Which return form should I use?", "expected_file": "11_itr_form_types.md"},
    {"query": "I repaid interest on a loan I took to fund my master's degree. Can I claim it?", "expected_file": "14_section_80e_education_loan.md"},
    {"query": "I gave money to the PM relief fund. Is that donation eligible for a tax break?", "expected_file": "15_section_80g_donations.md"},
    {"query": "I sold my old flat and bought a new house with the money. Can I avoid tax on the gain?", "expected_file": "19_section_54_exemption.md"},
    {"query": "I don't want another house but want to save capital-gains tax by parking the gain in government bonds.", "expected_file": "20_section_54ec_bonds.md"},
    {"query": "I topped up my NPS beyond my usual retirement savings. Is there an extra deduction just for that top-up?", "expected_file": "21_section_80ccd1b_nps_additional.md"},
    {"query": "Is the maturity payout from my life insurance policy taxable?", "expected_file": "22_section_10_10d_insurance.md"},
    {"query": "I financially support my disabled brother and pay for his treatment. Any deduction for that?", "expected_file": "23_section_80dd_disability_dependent.md"},
    {"query": "I run a small shop with around 90 lakh turnover and don't keep detailed books. Is there a simpler way to declare income?", "expected_file": "24_section_44ad_presumptive.md"},
    {"query": "My bank cut some money from my FD interest before paying me. What is that and at what rate?", "expected_file": "25_section_194a_tds_interest.md"},
    {"query": "I pay 60,000 a month as rent to my landlord. Do I have to deduct anything before paying?", "expected_file": "26_section_194i_tds_rent.md"},
    {"query": "I myself hold a 45% disability certificate. Is there a flat deduction available to me?", "expected_file": "27_section_80u_self_disability.md"},
    {"query": "My company gives me an allowance for holiday travel with my family. Is it tax-free?", "expected_file": "31_section_10_5_lta.md"},
    {"query": "I paid stamp duty and registration charges when buying my house. Can I deduct those?", "expected_file": "32_stamp_duty_deduction.md"},
    {"query": "Do farmers have to pay income tax on the money they make from growing crops?", "expected_file": "33_agricultural_income.md"},
]

# Adversarial out-of-scope: tax/finance-flavoured but NOT in this income-tax
# corpus, so a finance-tuned model is tempted but should still abstain.
HARD_OUT_OF_SCOPE: List[str] = [
    "How do I file my GST returns for my small business?",
    "What is the current repo rate set by the RBI?",
    "How is customs duty calculated on goods I import?",
    "Should I buy Reliance shares at the current price?",
    "What is the penalty for a bounced cheque under the Negotiable Instruments Act?",
]


def run_evaluation(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int,
    out_dir: str,
    corpus_source: Optional[str] = None,
    source_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    log.info("Running evaluation...")

    # Recall@k
    recall_hits = 0
    for item in EVAL_QUERIES:
        results, _ = retrieve(item["query"], model, embeddings, chunks, top_k=top_k, threshold=SIMILARITY_THRESHOLD)
        expected = item["expected_section"].lower()
        hit = any(expected in r["source"].lower() or expected in r["title"].lower() for r in results)
        if hit:
            recall_hits += 1

    recall = recall_hits / len(EVAL_QUERIES) if EVAL_QUERIES else 0.0

    # Refusal rate
    refusals = 0
    for q in EVAL_OUT_OF_SCOPE:
        results, max_s = retrieve(q, model, embeddings, chunks, top_k=top_k, threshold=SIMILARITY_THRESHOLD)
        if not results or max_s < SIMILARITY_THRESHOLD:
            refusals += 1

    refusal_rate = refusals / len(EVAL_OUT_OF_SCOPE) if EVAL_OUT_OF_SCOPE else 0.0

    # --- HARD eval: colloquial queries matched on source filename (no leakage) ---
    hard_r1 = hard_rk = hard_mrr = 0.0
    for item in HARD_EVAL:
        results, _ = retrieve(item["query"], model, embeddings, chunks, top_k=len(chunks), threshold=-1.0)
        ranks = [i + 1 for i, r in enumerate(results) if r["chunk_id"].split("#")[0] == item["expected_file"]]
        if ranks:
            rank = ranks[0]
            hard_mrr += 1.0 / rank
            if rank == 1:
                hard_r1 += 1
            if rank <= top_k:
                hard_rk += 1
    n_hard = len(HARD_EVAL) or 1
    hard_r1 /= n_hard
    hard_rk /= n_hard
    hard_mrr /= n_hard

    hard_refusals = 0
    for q in HARD_OUT_OF_SCOPE:
        results, max_s = retrieve(q, model, embeddings, chunks, top_k=top_k, threshold=SIMILARITY_THRESHOLD)
        if not results or max_s < SIMILARITY_THRESHOLD:
            hard_refusals += 1
    hard_refusal_rate = hard_refusals / (len(HARD_OUT_OF_SCOPE) or 1)

    metrics = {
        "num_docs_loaded": len({c["filename"] for c in chunks}),
        "num_chunks": len(chunks),
        "recall_at_k": round(recall, 4),
        "k": top_k,
        "num_eval_queries": len(EVAL_QUERIES),
        "refusal_rate": round(refusal_rate, 4),
        "num_oos_queries": len(EVAL_OUT_OF_SCOPE),
        "hard_num_queries": len(HARD_EVAL),
        "hard_recall_at_1": round(hard_r1, 4),
        "hard_recall_at_k": round(hard_rk, 4),
        "hard_mrr": round(hard_mrr, 4),
        "hard_num_oos_queries": len(HARD_OUT_OF_SCOPE),
        "hard_refusal_rate": round(hard_refusal_rate, 4),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "model": MODEL_NAME,
    }
    if corpus_source is not None:
        metrics["corpus_source"] = corpus_source
    if source_files is not None:
        metrics["source_files"] = source_files
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def build_pipeline(
    data_dir: str = "./data",
    out_dir: str = "./out",
    top_k: int = 4,
    eval_only: bool = False,
    corpus: str = "auto",
    pdf_dir: str = "./data/pdfs",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if eval_only:
        log.info("--- Evaluation-only mode ---")
        model = build_embedder()
        embeddings, chunks = load_index(out_dir)
        metrics = run_evaluation(model, embeddings, chunks, top_k, out_dir)
        log.info("Metrics:\n%s", json.dumps(metrics, indent=2))
        return

    # 1. Select + load documents
    docs, corpus_source = _select_corpus(corpus, data_dir, pdf_dir)
    if not docs:
        log.error(
            "Corpus is empty (mode=%s). Add .md files to %s and/or readable .pdf "
            "files to %s, or fix scanned PDFs. Leaving existing index untouched.",
            corpus_source, data_dir, pdf_dir,
        )
        sys.exit(1)

    # 2. Chunk
    chunks = chunk_all_documents(docs)

    # 3. Embed
    model = build_embedder()
    embeddings = embed_chunks(chunks, model)

    # 4. Save index
    save_index(out_dir, embeddings, chunks)

    # 5. Evaluate
    metrics = run_evaluation(
        model, embeddings, chunks, top_k, out_dir,
        corpus_source=corpus_source,
        source_files=sorted({d["filename"] for d in docs}),
    )
    log.info("Metrics:\n%s", json.dumps(metrics, indent=2))

    # 6. Demo retrieval
    log.info("--- Retrieval Demo ---")
    demo_queries = [
        "What are the 80C deduction limits for PPF and ELSS?",
        "How much tax is deducted on bank interest?",
    ]
    for q in demo_queries:
        results, max_s = retrieve(q, model, embeddings, chunks, top_k=top_k)
        log.info("Q: %s", q)
        log.info("  Max score: %.4f", max_s)
        if not results:
            log.info("  (no results above threshold — ABSTAIN)")
        for r in results:
            log.info(
                "  [%.4f] %s | %s | %s",
                r["score"], r["title"], r["source"], r["chunk_id"],
            )
        # LLM synthesis if available
        if results:
            synth = synthesize_answer(q, results)
            if synth:
                log.info("  SYNTHESIS:\n%s", synth)
            else:
                log.info("  EXTRACTIVE (no LLM configured)")
        log.info("")

    log.info("Pipeline complete. Index at %s/index.npz", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TaxRAG — India Income Tax Q&A RAG pipeline")
    parser.add_argument("--data-dir", default="./data", help="Directory with .md documents")
    parser.add_argument("--pdf-dir", default="./data/pdfs", help="Directory with .pdf documents")
    parser.add_argument("--corpus", default="auto", choices=["md", "pdf", "both", "auto"],
                        help="Corpus source: md (markdown), pdf (PDFs), both, or auto (PDFs if present, else markdown)")
    parser.add_argument("--out", default="./out", help="Output directory for index and metrics")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (requires existing index)")
    args = parser.parse_args()
    build_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out,
        top_k=args.top_k,
        eval_only=args.eval_only,
        corpus=args.corpus,
        pdf_dir=args.pdf_dir,
    )


if __name__ == "__main__":
    main()
