#!/usr/bin/env python3
"""
TaxRAG — India Income Tax Q&A Assistant (Streamlit chat app)
Cloud-optimized, CPU-only. Loads the pre-built index from out/index.npz and
answers as a chat, retrieving cited tax provisions and abstaining when unsure.
"""

import os
import sys
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import (
    load_index,
    retrieve,
    synthesize_answer,
    build_index_from_text,
    embed_chunks,
    available_models,
    LLM_PROVIDERS,
    LLM_PROVIDER_ORDER,
    SIMILARITY_THRESHOLD,
)


def _extract_pdf_text(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


st.set_page_config(page_title="TaxRAG – India Income Tax Q&A", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# Embedding models + on-the-fly corpus embedding (no per-model .npz needed)
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# CPU/free-tier friendly: lightweight bge-small retrievers (~33M params) that
# embed fast on CPU. Heavier long-context encoders (EmbeddingGemma-300m,
# Granite-97m 32K) were benchmarked at ~100 min to embed this corpus on a 4-core
# CPU — impractical here.
# Default = FiQA-tuned: on 277 content-based questions from held-out sections it
# beats the tax-tuned model (Recall@5 0.773 vs 0.675). The tax-tuned model was
# trained on section-title pairs, so it overfits title-like queries.
EMBEDDING_MODELS = {
    "Finance-tuned bge-small (FiQA, default)": "vivekkopthsd/fiqa-retriever-bge-small",
    "Hinglish ST-33m (Hindi/Hinglish queries)": "vivekkopthsd/hinglish-st-33m",
    "Tax-tuned bge-small (title-trained)": "vivekkopthsd/tax-retriever-bge-small",
}

# Passages run 200–500 tokens (median ~382, only ~0.2% exceed 512). The bge-small
# checkpoints ship with max_seq_length=256 — which would truncate a third off the
# median passage — but their BERT backbone supports 512 positions, so we pin 512:
# full passages embed without loss, and long-context models are reined in so CPU
# embedding stays fast.
MAX_SEQ_LENGTH = 512
# Some models have a lower trained ceiling that must NOT be raised (doing so uses
# position ranges the model never saw). hinglish-st-33m was trained at 192, and
# its pre-built corpus index was embedded at 192 — keep query embedding consistent.
SEQ_LEN_OVERRIDE = {"vivekkopthsd/hinglish-st-33m": 192}


@st.cache_data(show_spinner=False, ttl=600)
def _provider_models(provider: str):
    """Live-discovered models for a provider (curated first), cached 10 min."""
    return available_models(provider)


@st.cache_resource(show_spinner="Loading retrieval model (first run only)…", max_entries=1)
def load_embedder(model_name: str) -> SentenceTransformer:
    # max_entries=1: keep only the active model resident so switching models on a
    # CPU/free-tier host doesn't accumulate several models in RAM (bge-m3 is large).
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = SEQ_LEN_OVERRIDE.get(model_name, MAX_SEQ_LENGTH)
    return model


@st.cache_resource(show_spinner=False)
def load_cached_index():
    return load_index("./out")


@st.cache_resource(show_spinner="Loading reranker (first run only)…", max_entries=1)
def load_rerank_retriever(model_name: str, _model, _emb, _chunks):
    from reranker import RerankRetriever
    return RerankRetriever(_model, _emb, _chunks)


def _emb_cache_path(model_name: str) -> str:
    """Disk path for a model's pre-built corpus embeddings. Must match
    build_model_index.py so a pre-embedded file is picked up automatically."""
    return os.path.join(_APP_DIR, "out", "emb_" + model_name.replace("/", "__") + ".npy")


def get_index_for_model(model_name: str):
    """Return (model, corpus_embeddings) for the selected model.

    Loads a pre-built `out/emb_<model>.npy` instantly when present (run
    `build_model_index.py` to create one). Otherwise embeds the corpus on the
    fly the first time a model is chosen, then caches it in session_state."""
    key = f"emb_{model_name}"
    # Evict other models' cached corpus embeddings — these numpy arrays are the
    # bulk of the memory, and only the active model's are needed this turn.
    for stale in [k for k in list(st.session_state) if k.startswith("emb_") and k != key]:
        st.session_state.pop(stale, None)
    if key not in st.session_state:
        cache_path = _emb_cache_path(model_name)
        if os.path.exists(cache_path):
            st.session_state[key] = np.load(cache_path)
        else:
            _, chunks = load_cached_index()
            with st.spinner(f"Embedding corpus with {model_name.split('/')[-1]} "
                            "(first run only)…"):
                st.session_state[key] = embed_chunks(chunks, load_embedder(model_name))
    return load_embedder(model_name), st.session_state[key]


_, base_chunks = load_cached_index()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?, provider?}

# ---------------------------------------------------------------------------
# Sidebar — controls (not vanity metrics)
# ---------------------------------------------------------------------------
st.sidebar.title("TaxRAG 📋")
st.sidebar.caption("India Income Tax Q&A — retrieval-grounded, cited, abstains when unsure.")
st.sidebar.markdown("---")

st.sidebar.subheader("Answer engine (LLM)")
provider_choice = st.sidebar.selectbox(
    "Provider",
    ["Auto (failover)"] + LLM_PROVIDER_ORDER,
    help="Which LLM writes the final answer. 'Auto' tries each provider in turn. "
         "All fall back to extractive mode (showing the sources) if none are reachable.",
)
sel_provider = None if provider_choice == "Auto (failover)" else provider_choice
sel_model = None
if sel_provider:
    _opts = _provider_models(sel_provider) or LLM_PROVIDERS[sel_provider]["models"]
    sel_model = st.sidebar.selectbox(
        "Model", _opts,
        help="Live-discovered from the provider's /models endpoint (curated free "
             "models first); falls back to a built-in list if the endpoint is unreachable.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Answering mode")
use_router = st.sidebar.toggle(
    "Dual-mode routing (recommended)",
    value=True,
    help="Router classifies each query and picks the right handler: greetings and "
         "general tax questions get direct LLM answers, statute lookups stay grounded "
         "with citations, calculations compute with retrieved slab rates. When off, "
         "uses the legacy single-shot RAG pipeline.",
)

st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 4)
use_rerank = st.sidebar.toggle(
    "Rerank passages (hybrid + cross-encoder)", value=True,
    help="Dense + BM25 candidates, reordered by a cross-encoder. Measured on held-out "
         "questions: top-passage accuracy ~50% -> ~75%, at ~2-3 s extra per question on CPU.",
)
model_choice = st.sidebar.selectbox(
    "Embedding model",
    list(EMBEDDING_MODELS.keys()),
    help="Model used to embed both the corpus and your query. The corpus is "
         "embedded on the fly the first time a model is selected (~1 min); "
         "switching models re-embeds once per model.",
)
model_name = EMBEDDING_MODELS[model_choice]
threshold = st.sidebar.slider(
    "Abstention threshold", 0.0, 0.9, SIMILARITY_THRESHOLD, 0.05,
    help="Minimum similarity a passage must reach to be used. Higher = stricter "
         "(abstains more); lower = more lenient answers.",
)

# --- Optional: bring your own document ---
st.sidebar.subheader("Your document")
up = st.sidebar.file_uploader("Upload a PDF to ask about it instead", type=["pdf"])
if up is not None and st.session_state.get("uploaded_name") != up.name:
    with st.sidebar, st.spinner(f"Indexing '{up.name}'…"):
        text = _extract_pdf_text(up)
        if len(text.strip()) < 50:
            st.sidebar.error("No readable text (scanned/image-only PDF?).")
        else:
            emb, chs = build_index_from_text(text, load_embedder(model_name), source_name=up.name)
            st.session_state.uploaded_name = up.name
            st.session_state.uploaded_emb = emb
            st.session_state.uploaded_chunks = chs
if st.session_state.get("uploaded_name"):
    st.sidebar.success(f"Active: {st.session_state.uploaded_name}")
    if st.sidebar.button("✖ Use built-in tax corpus"):
        for k in ("uploaded_name", "uploaded_emb", "uploaded_chunks"):
            st.session_state.pop(k, None)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Corpus: Income-Tax Act, 2025 (as amended by FA 2026) · {len(base_chunks):,} passages")
st.sidebar.caption(f"Embedding model: `{model_name.split('/')[-1]}`")
if st.sidebar.button("🗑 Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Active corpus for this turn — embedded on the fly with the selected model
selected_model, base_embeddings = get_index_for_model(model_name)
use_uploaded = bool(st.session_state.get("uploaded_name"))
if use_uploaded:
    active_emb = st.session_state.uploaded_emb
    active_chunks = st.session_state.uploaded_chunks
    # uploaded arbitrary text scores lower, so cap the floor so uploads still answer
    active_threshold = min(threshold, 0.25)
else:
    active_emb, active_chunks = base_embeddings, base_chunks
    active_threshold = threshold


# ---------------------------------------------------------------------------
# Answer a single question -> renders assistant reply and records it
# ---------------------------------------------------------------------------
TRUST_BADGE = {
    "grounded":       ("🟢 Grounded", "Cited from Income-Tax Act 2025"),
    "llm_memory":     ("🟡 LLM knowledge", "Not statute-cited — verify with a CA"),
    "conversational": ("🔵 Chat", "Conversational reply, no citations needed"),
    "refusal":        ("🔴 Out of scope", "Politely declined"),
}


def _render_router_result(question: str, r: dict) -> None:
    badge_label, badge_help = TRUST_BADGE.get(r["trust_tier"], ("⚪ ?", ""))
    st.markdown(f"{badge_label}  · `{r['router_intent']}` · via `{r.get('provider','?')}`")
    st.caption(badge_help)
    st.markdown(r["answer"])
    if r.get("citations"):
        with st.expander(f"📖 Sources ({len(r['citations'])} passages"
                         + (f", max sim {r.get('max_sim', 0):.2f}" if r.get("max_sim") else "") + ")"):
            for i, c in enumerate(r["citations"]):
                st.markdown(f"**{i+1}. {c['title']}** — {c['source']} · score {c['score']:.3f}")
    st.session_state.messages.append({
        "role": "assistant", "content": r["answer"],
        "sources": r.get("citations", []), "provider": r.get("provider"),
        "trust_tier": r["trust_tier"], "intent": r["router_intent"],
    })


def answer_question(question: str) -> None:
    # Dual-mode router path (default): intent → handler → trust-tier badge.
    # Retrieval-backed handlers (statute lookups, calculations) use the
    # selected embedding model and its on-the-fly corpus embeddings.
    if use_router and not use_uploaded:
        from chat_router import chat as router_chat
        hist = [{"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages if m["role"] in ("user", "assistant")][:-1]
        retriever = None
        if use_rerank:
            retriever = load_rerank_retriever(model_name, selected_model, active_emb, active_chunks)
            retriever.threshold = active_threshold
        with st.spinner("Routing your question…"):
            r = router_chat(question, selected_model, active_emb, active_chunks,
                            history=hist, top_k=top_k, threshold=active_threshold,
                            retriever=retriever, rerank=use_rerank)
        _render_router_result(question, r)
        return

    results, max_score = retrieve(
        question, selected_model, active_emb, active_chunks,
        top_k=top_k, threshold=active_threshold,
    )
    if not results or max_score < active_threshold:
        msg = ("I can't answer that from the "
               f"{'uploaded document' if use_uploaded else 'tax corpus'}. "
               "Nothing was similar enough to the question, so I'm abstaining rather "
               "than guessing.")
        st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    hist = [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    with st.spinner("Retrieving provisions and writing a cited answer…"):
        answer, who = synthesize_answer(
            question, results, history=hist[:-1],
            provider=sel_provider, model=sel_model,
        )

    if answer:
        st.markdown(answer)
        caption = f"Answered by `{who}` · grounded only in the sources below."
    else:
        # Extractive fallback: no LLM reachable
        answer = "**No LLM reachable — showing the most relevant provisions directly:**\n\n"
        answer += "\n\n".join(f"- **{r['title']}** ({r['source']})" for r in results)
        st.markdown(answer)
        caption = "Extractive mode (no LLM). Set a provider key to enable written answers."
    st.caption(caption)

    with st.expander(f"📖 Sources ({len(results)} passages, top similarity {max_score:.2f})"):
        for i, r in enumerate(results):
            tag = ""
            if r.get("language"):
                tag = f" · `{r['language']}`"
            if r.get("retriever_source"):
                tag += f" · `{r['retriever_source']}`"
            st.markdown(f"**{i+1}. {r['title']}** — {r['source']}  ·  score {r['score']:.3f}{tag}")
            st.markdown(r["text"])
            st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": results, "provider": who}
    )


# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------
st.title("India Income Tax Q&A Assistant")
if use_uploaded:
    routing_note = (" Dual-mode routing is paused for uploaded documents — answers "
                    "use the direct retrieval pipeline." if use_router else "")
    st.info(f"Answering from your uploaded document **{st.session_state.uploaded_name}**. "
            "Switch back to the tax corpus from the sidebar." + routing_note)

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"📖 Sources ({len(m['sources'])} passages)"):
                for i, r in enumerate(m["sources"]):
                    st.markdown(f"**{i+1}. {r['title']}** — {r['source']}  ·  score {r['score']:.3f}")
                    if r.get("text"):  # router citations carry no passage text
                        st.markdown(r["text"])
                    st.markdown("---")

# Starter suggestions (only before the first message) — a click sets a pending
# question in session_state, which is safe (no widget key is mutated).
if not st.session_state.messages:
    st.caption("Try asking:")
    starters = [
        "What is the standard deduction from salary?",
        "How is residential status in India determined?",
        "Deduction for life insurance premium and provident fund",
        "How is agricultural income treated?",
    ]
    cols = st.columns(len(starters))
    for i, q in enumerate(starters):
        if cols[i].button(q, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending_q = q
            st.rerun()

# Input: either a starter click (pending_q) or the chat box
prompt = st.session_state.pop("pending_q", None) or st.chat_input("Ask a tax question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer_question(prompt)

# ---------------------------------------------------------------------------
# Disclaimer (always visible, below the chat)
# ---------------------------------------------------------------------------
with st.expander("ℹ️ About & disclaimer"):
    st.markdown(
        f"""
**TaxRAG** retrieves the most relevant sections of the **Income-Tax Act, 2025**,
then writes a **cited** answer grounded only in those passages. If nothing is
relevant enough, it **abstains** instead of guessing.

- **Corpus:** the full **Income-Tax Act, 2025, as amended by Finance Act 2026**
  (477 sections → {len(base_chunks):,} indexed passages). Content is sliced from the
  authoritative Act PDF; section labels are anchored to the machine-readable
  [Income-Tax-Act-2025](https://huggingface.co/datasets/ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text)
  dataset (MIT licence). This is the actual statute text, not a summary.
- **Retriever:** you pick the **embedding model** in the sidebar — a tax-tuned
  `bge-small`
  ([`vivekkopthsd/tax-retriever-bge-small`](https://huggingface.co/vivekkopthsd/tax-retriever-bge-small))
  by default (~33M params, fast on CPU), a **Hinglish** retriever
  ([`vivekkopthsd/hinglish-st-33m`](https://huggingface.co/vivekkopthsd/hinglish-st-33m))
  for Hindi/Hinglish queries (higher ranking quality on romanized-Hindi questions),
  or finance-tuned `bge-small`
  ([`vivekkopthsd/fiqa-retriever-bge-small`](https://huggingface.co/vivekkopthsd/fiqa-retriever-bge-small)).
  Pre-built embeddings load instantly when present; otherwise the corpus is
  embedded on the fly with the chosen model.
- **Answer engine:** your chosen LLM provider, or automatic failover across
  OpenCode → Groq → NVIDIA → Kilo; extractive fallback if none is reachable.
- **You can upload your own PDF** (sidebar) and ask questions about it instead.

Note: the 2025 Act renumbers sections (e.g. the old 80C deductions now sit in
Section 123), so ask by topic rather than old section numbers.
"""
    )
    st.warning(
        "Educational tool only. Monetary limits, thresholds and slab rates are "
        "assessment-year dependent and change with each Finance Act. Not legal, "
        "financial or tax advice — verify with a qualified professional."
    )
