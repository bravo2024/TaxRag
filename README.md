# TaxRAG — India Income Tax Q&A Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.43%2B-FF4B4B.svg)](https://streamlit.io)
[![CPU-only](https://img.shields.io/badge/inference-CPU--only-green.svg)]()

A retrieval-augmented generation (RAG) app that answers Indian income-tax
questions with **statute-grounded, cited answers** — retrieving the relevant
provisions of the **Income-Tax Act, 2025** (as amended by Finance Act 2026) and
writing an answer only from those passages. When nothing in the corpus is
relevant enough, it **abstains instead of guessing**.

**Live app:** https://indiataxrag.streamlit.app ·
**CPU-only** · deploys free on Streamlit Community Cloud

---

## Highlights

- **Full statute corpus** — the complete **Income-Tax Act, 2025** as amended by
  **Finance Act 2026**: **477 sections** sliced into **1,237 passages**
  (200–500 tokens each), anchored to the machine-readable
  [Income-Tax-Act-2025](https://huggingface.co/datasets/ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text)
  dataset (MIT licence) and verified against the authoritative Act PDF.
  This is the actual statute text, not a summary.
- **Multiple embedding models, one click** — switch retrievers in the sidebar:
  a **tax-tuned `bge-small`** (default), a **Hinglish retriever** for
  Hindi/Hinglish questions, and a **finance-tuned `bge-small` (FiQA)**. Pre-built
  corpus embeddings for all three ship in the repo, so switching models is
  instant — no re-embedding on first use.
- **Dual-mode routing** — an intent router classifies each question and picks the
  best handler: statute lookups stay grounded with citations, calculations
  compute with the retrieved slab rates, greetings and general tax questions
  get direct LLM answers. Every reply carries a **trust-tier badge**
  (🟢 grounded · 🟡 LLM knowledge · 🔵 chat · 🔴 out of scope).
- **Abstention** — retrieval is gated by a similarity threshold; below it the
  assistant refuses rather than inventing figures. Tax accuracy matters more
  than coverage.
- **Bring your own document** — upload a PDF and ask questions about it;
  it is chunked, embedded and indexed on the fly, and the answer switches to
  that document with an automatic lower threshold.
- **Keyless-by-default LLM synthesis** — answers are written by a multi-provider
  failover chain (OpenCode → Groq → NVIDIA → Kilo) with **automatically
  discovered free models**. No API key needed (anonymous free gateways); if
  every provider is unreachable the app drops to **extractive mode** and shows
  the retrieved provisions directly.

## How it works

```
PDF (Income-Tax Act 2025 + FA 2026)
   │  section-aware chunking (200–500 tokens, section labels anchored to
   │  the machine-readable dataset)
   ▼
1,237 passages ──► embed with the selected model ──► L2-normalized vector index
                                                        (out/index.npz + out/emb_*.npy)
   ▼
your question ──► embed ──► cosine top-k ──► similarity ≥ threshold?
   │                                      │ no  → abstain ("not in the corpus")
   ▼ yes
intent router ──► GREETING / STATUTE_LOOKUP / CALCULATION / GENERAL_TAX / OFF_TOPIC
   │
   ▼
handler ──► grounded answer + citations  |  computation  |  LLM answer  |  decline
   ▼
answer + trust-tier badge 🟢🟡🔵🔴
```

Key trade-offs, made explicit in code:

- **Passage length vs. model context.** Passages run 200–500 tokens (median
  ~382; only ~0.2% exceed 512). The `bge-small` checkpoints ship with a
  `max_seq_length` of 256 — which would truncate a third off the median passage —
  so the app pins **512** (their BERT backbone supports it) to embed full
  passages. The Hinglish model was trained at **192** and must NOT be raised
  (doing so uses positions the model never saw), so it stays at 192.
- **Speed vs. reach.** Long-context encoders (EmbeddingGemma-300m,
  Granite-97m-32K) were benchmarked at ~100 min to embed this corpus on a 4-core
  CPU — impractical for a free-tier host — so the defaults are ~33M-parameter
  retrievers that embed the corpus in a couple of minutes.
- **Memory.** Only the active model's corpus embeddings are held in RAM;
  switching models evicts the previous array instead of accumulating them.

### Embedding models

| Model in sidebar | Hugging Face id | Trained for | Seq. length |
|---|---|---|---|
| **Tax-tuned bge-small (default)** | [`vivekkopthsd/tax-retriever-bge-small`](https://huggingface.co/vivekkopthsd/tax-retriever-bge-small) | Income-tax vocabulary and phrasing | 512 |
| **Hinglish ST-33m** | [`vivekkopthsd/hinglish-st-33m`](https://huggingface.co/vivekkopthsd/hinglish-st-33m) | Hindi / Hinglish (romanized) questions | 192 |
| **Finance-tuned bge-small (FiQA)** | [`vivekkopthsd/fiqa-retriever-bge-small`](https://huggingface.co/vivekkopthsd/fiqa-retriever-bge-small) | Financial-domain QA text | 512 |

Pre-built corpus embeddings for all three are committed as
`out/emb_<model>.npy` and loaded instantly. Select a model with no entry and
the app falls back to embedding the corpus on the fly once (or run
`python build_model_index.py` to pre-embed it). `TAXRAG_MODEL` overrides the
default.

## Evaluation

Retrieval quality is measured against a **100-query hard set**: real section
headings are paraphrased into natural taxpayer questions with synonym
substitution (e.g. "deduction in respect of" → "how do I claim tax relief for"),
so the retriever must match on *meaning*, not keywords. The set is deliberately
loaded with confusable clusters — "Deduction in respect of…" and the loss
set-off sections, where near-duplicate provisions compete — and scores are
matched on the gold section, never on wording the query could leak.

**1. Retrieval + abstention** — `python eval_retrieval.py` (reported figures,
finance-tuned retriever defaults, 100-query set, out-of-scope gate 0.45):

| Set | Metric | Score |
|---|---|---|
| Hard paraphrased queries (100) | Recall@1 | 0.78 |
| Hard paraphrased queries (100) | **Recall@5** | **0.96** |
| Hard paraphrased queries (100) | **MRR@5** | **0.85** |
| Out-of-scope questions (12) | **Refusals** | **11 / 12** |
| Corpus scale | passages / sections | 1,237 / 477 |

Out-of-scope questions score 0.24–0.41 similarity, cleanly under the 0.45
abstention gate — which is *why* the refusals are reliable, not luck.

**2. Cross-lingual model comparison** — `python eval_hinglish.py` reruns the
same 100 gold queries in English and Hinglish for the tax-tuned and Hinglish
retrievers over the full corpus:

| Retriever | Query language | Recall@5 | MRR@5 | nDCG@5 |
|---|---|---|---|---|
| Tax-tuned bge-small | English | 1.00 | 0.92 | 0.94 |
| Tax-tuned bge-small | Hinglish | 0.94 | 0.82 | 0.85 |
| Hinglish ST-33m | English | 0.98 | 0.93 | 0.94 |
| **Hinglish ST-33m** | **Hinglish** | **0.94** | **0.88** | **0.90** |

The Hinglish retriever leads on romanized-Hindi questions (MRR@5 0.88 vs 0.82),
while the tax-tuned retriever is the best all-round English default (Recall@5
1.00). Raw per-query numbers are in `hinglish_eval_results.json`.

**3. Two-stage retrieval ablation** — `python eval_rerank.py` tests whether a
second stage (BM25 RRF fusion and/or a cross-encoder reranker, `rank_bm25` +
`sentence-transformers`) improves on the dense first stage, reporting
Recall@5 / MRR@5 / nDCG@5 on the same hard set.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py          # launch the chat app  (http://localhost:8501)
```

The committed `out/index.npz` + `out/emb_*.npy` mean the app never re-embeds
the corpus at startup — the first page load pulls the embedding model from
Hugging Face once, then everything runs on CPU.

### Reproduce the numbers

```bash
python eval_retrieval.py      # Table 1 — retrieval + abstention (finance-tuned)
set MODEL_NAME=vivekkopthsd/tax-retriever-bge-small   # rerun with the default retriever
python eval_hinglish.py       # Table 2 — tax-tuned vs Hinglish, EN + Hinglish
python eval_rerank.py         # Table 3 — two-stage rerank ablation
```

### Embed a new model / rebuild the corpus

```bash
python build_model_index.py --model <hf-id>   # pre-embed the corpus with a new retriever
python build_from_pdf_sections.py             # rebuild index from the source PDFs
```

## Configuration

Retrieval works with **no configuration** — the anonymous free gateways make
LLM synthesis work out of the box. Environment variables (or Streamlit secrets)
to override defaults:

| Variable | Purpose |
|---|---|
| `TAXRAG_MODEL` | Default embedding model (default: `vivekkopthsd/tax-retriever-bge-small`) |
| `OPENCODE_ZEN_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` / `KILO_API_KEY` | Enable a specific provider in the failover chain |
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `LLM_MODEL` | Force a single custom OpenAI-compatible endpoint |

## Project layout

```
app.py                    Streamlit chat app (sidebar controls, router, PDF upload)
rag.py                    Chunking, embedding, indexing, retrieval, LLM synthesis, eval
chat_router.py            Intent classifier + 5 handlers + trust tiers
hybrid_retriever.py       BM25 / dense hybrid retrieval utilities
build_from_pdf_sections.py  Rebuild the corpus index from the Act PDFs
build_model_index.py      Pre-embed the corpus with a new model (out/emb_*.npy)
eval_retrieval.py         Hard-set retrieval + abstention evaluation
eval_hinglish.py          English vs Hinglish comparison across retrievers
eval_rerank.py            Two-stage retrieval ablation (BM25, cross-encoder)
out/index.npz             Pre-built corpus index (committed on purpose)
out/emb_*.npy             Pre-built embeddings per model (instant model switching)
data/pdfs/                Source PDFs: IT Act 2025 + Taxation Bill 2026
```

## Disclaimer

This is an educational, illustrative tool. All monetary limits, thresholds,
slab rates and deduction caps are **assessment-year dependent** and change with
each annual Finance Act. Nothing here is legal, financial or tax advice — verify
with a qualified tax professional for your assessment year.
