#!/usr/bin/env python3
"""
TaxRAG — India Income Tax Q&A Assistant (Streamlit chat app)
Cloud-optimized, CPU-only. Loads the pre-built index from out/index.npz and
answers as a chat, retrieving cited tax provisions and abstaining when unsure.
"""

import os
import sys
import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import (
    load_index,
    retrieve,
    synthesize_answer,
    build_index_from_text,
    LLM_PROVIDERS,
    LLM_PROVIDER_ORDER,
    SIMILARITY_THRESHOLD,
    MODEL_NAME,
)


def _extract_pdf_text(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


st.set_page_config(page_title="TaxRAG – India Income Tax Q&A", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources — model + index loaded once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading retrieval model (first run only)…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def load_cached_index():
    return load_index("./out")


embedder = load_embedder()
base_embeddings, base_chunks = load_cached_index()

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
    sel_model = st.sidebar.selectbox("Model", LLM_PROVIDERS[sel_provider]["models"])

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 4)
threshold = st.sidebar.slider(
    "Abstention threshold", 0.0, 0.9, SIMILARITY_THRESHOLD, 0.05,
    help="Minimum similarity a passage must reach to be used. Higher = stricter "
         "(abstains more, avoids weak matches); lower = more lenient answers.",
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
            emb, chs = build_index_from_text(text, embedder, source_name=up.name)
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
st.sidebar.caption(f"Retriever: `{MODEL_NAME.split('/')[-1]}`")
if st.sidebar.button("🗑 Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Active corpus for this turn
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
def answer_question(question: str) -> None:
    results, max_score = retrieve(
        question, embedder, active_emb, active_chunks,
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
            st.markdown(f"**{i+1}. {r['title']}** — {r['source']}  ·  score {r['score']:.3f}")
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
    st.info(f"Answering from your uploaded document **{st.session_state.uploaded_name}**. "
            "Switch back to the tax corpus from the sidebar.")

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"📖 Sources ({len(m['sources'])} passages)"):
                for i, r in enumerate(m["sources"]):
                    st.markdown(f"**{i+1}. {r['title']}** — {r['source']}  ·  score {r['score']:.3f}")
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
- **Retriever:** fine-tuned `bge-small` bi-encoder ([`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME})), 384-dim, CPU.
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
