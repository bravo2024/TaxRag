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
- **Hybrid search + fine-tuned reranker** — dense (FAISS) and BM25 candidates
  are fused with reciprocal-rank fusion and reordered by a cross-encoder
  fine-tuned on tax passages. **Recall@5 0.77 → 0.91** and
  **MRR@10 0.63 → 0.83** on 277 questions from 150 held-out sections.
- **Deterministic tax calculator** — calculation questions are routed to a
  calculator tool whose slab table, rebate and standard deduction are parsed
  from the Act itself: **30/30 correct**, vs 3/25 for LLM-only answers.
- **Dual-mode routing** — an intent router classifies each question and picks the
  best handler: statute lookups stay grounded with citations, calculations go
  to the calculator, greetings and general tax questions get direct LLM
  answers. Router accuracy **0.94** with the embedding-classifier fallback.
  Every reply carries a **trust-tier badge**
  (🟢 grounded · 🟡 LLM knowledge · 🔵 chat · 🔴 out of scope).
- **Measured answer quality** — LLM-as-judge on 277 in-scope questions:
  **Qwen3-4B 98.9% faithful** vs 91.7% for Qwen3-1.7B (McNemar p = 0.001 on
  correct-and-grounded answers).
- **Multiple embedding models, one click** — a **finance-tuned `bge-small`
  (FiQA)** (default), a **Hinglish retriever** for Hindi/Hinglish questions, and
  a **tax-tuned `bge-small`**. Pre-built corpus embeddings for all three ship in
  the repo, so switching models is instant.
- **Abstention** — retrieval is gated by a similarity threshold; below it the
  assistant refuses rather than inventing figures.
- **Bring your own document** — upload a PDF and ask questions about it;
  it is chunked, embedded and indexed on the fly, and the answer switches to
  that document with an automatic lower threshold.
- **Keyless-by-default LLM synthesis** — answers are written by a multi-provider
  failover chain (OpenCode → Groq → NVIDIA → Kilo) with **automatically
  discovered free models**. No API key needed; if every provider is unreachable
  the app drops to **extractive mode** and shows the retrieved provisions.

## How it works

```
PDF (Income-Tax Act 2025 + FA 2026)
   │  section-aware chunking (200–500 tokens, section labels anchored to
   │  the machine-readable dataset)
   ▼
1,237 passages ──► embed with the selected model ──► FAISS index
                                                     (out/index.npz + out/emb_*.npy)
   ▼
your question ──► intent router ──► GREETING / STATUTE_LOOKUP / CALCULATION / GENERAL_TAX / OFF_TOPIC
   │
   ▼ (retrieval-backed handlers)
dense (FAISS) + BM25 ──► RRF fusion ──► cross-encoder rerank ──► similarity ≥ threshold?
   │                                                            │ no → abstain
   ▼ yes
grounded answer + citations  |  calculator tool  |  LLM answer  |  decline
   ▼
answer + trust-tier badge 🟢🟡🔵🔴
```

Key trade-offs, made explicit in code:

- **Passage length vs. model context.** Passages run 200–500 tokens (median
  ~382; only ~0.2% exceed 512). The `bge-small` checkpoints ship with a
  `max_seq_length` of 256, so the app pins **512** (their BERT backbone
  supports it) to embed full passages. The Hinglish model was trained at
  **192** and stays at 192.
- **Quality vs. CPU latency.** The benchmark reranks 30 candidates; the app
  reranks 15 at 256 tokens (about 2.5 s per query on a 4-core laptop CPU).
- **Speed vs. reach.** Long-context encoders (EmbeddingGemma-300m,
  Granite-97m-32K) took ~100 min to embed this corpus on a 4-core CPU, so the
  defaults are ~33M-parameter retrievers.
- **Memory.** Only the active model's corpus embeddings are held in RAM.

### Models

| Role | Hugging Face id | Notes |
|---|---|---|
| **Retriever (default)** | [`vivekkopthsd/fiqa-retriever-bge-small`](https://huggingface.co/vivekkopthsd/fiqa-retriever-bge-small) | Financial-domain QA, seq. length 512 |
| Retriever | [`vivekkopthsd/hinglish-st-33m`](https://huggingface.co/vivekkopthsd/hinglish-st-33m) | Hindi / Hinglish questions, seq. length 192 |
| Retriever | [`vivekkopthsd/tax-retriever-bge-small`](https://huggingface.co/vivekkopthsd/tax-retriever-bge-small) | Trained on Act section titles, seq. length 512 |
| **Reranker** | [`vivekkopthsd/tax-reranker-minilm`](https://huggingface.co/vivekkopthsd/tax-reranker-minilm) | Cross-encoder fine-tuned on tax passages |

The FiQA retriever is the default because on content-based held-out questions
it beats the tax-tuned model (Recall@5 0.773 vs 0.675); the tax-tuned model was
trained on section titles and overfits title-like queries. `TAXRAG_MODEL` and
`TAXRAG_RERANKER` override the defaults.

## Evaluation

All result files are committed under `colab/` and `eval/results/`.

**1. Retrieval** — `colab/retrieval_bench.py` →
`colab/retrieval_results.json`. 277 content-based questions generated from 150
held-out sections (the reranker was trained on the other sections), 30
candidates reranked:

| Pipeline | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.49 | 0.75 | 0.60 | 0.65 |
| Dense: FiQA bge-small (app default) | 0.50 | 0.77 | 0.63 | 0.69 |
| Hybrid: FiQA + BM25 (RRF) | 0.57 | 0.83 | 0.69 | 0.74 |
| FiQA → ms-marco reranker (untuned) | 0.73 | 0.90 | 0.80 | 0.83 |
| **Hybrid → tax-tuned reranker (ours)** | **0.77** | **0.91** | **0.83** | **0.86** |

**2. Calculator tool vs LLM** — `eval/eval_calc.py` →
`eval/results/calc_eval.json`. 30 new-regime tax questions:

| Method | Correct |
|---|---|
| **Calculator tool** | **30 / 30** (~1 ms) |
| LLM only (free gateway models) | 3 / 25 answered (5 unreachable), ~24 s |

Hand-audited LLM failures: outdated pre-2025 slabs, the section 156 rebate
ignored, refusals despite the slab table being retrieved, and truncated totals.

**3. Answer quality (LLM-as-judge)** — `colab/answer_eval.py` →
`eval/results/phase_d/`. Answers generated with vLLM on 277 in-scope questions,
graded by Qwen3-8B-AWQ:

| Generator | Faithful | Correct & grounded (gold retrieved) | Cites gold section | Out-of-scope abstention | s / answer |
|---|---|---|---|---|---|
| Qwen3-1.7B | 91.7% | 92.5% | 83.0% | 100% | 1.3 |
| **Qwen3-4B-Instruct** | **98.9%** | **99.1%** | **93.4%** | **100%** | 3.0 |

McNemar test on correct-and-grounded answers (small vs large): p = 0.001.
A similarity-threshold SLM/LLM router (small model when the top retrieval
score ≥ 0.66) keeps quality at 97.3% vs 98.0% for the large model alone, with
**17% less generation time**.

**4. Intent routing** — `eval/eval_router_classifier.py` →
`eval/results/router_classifier_eval.json`. 100 labelled questions:

| Router | Accuracy |
|---|---|
| Keyword heuristic | 0.47 |
| LLM router + keyword fallback | 0.74 |
| Embedding classifier alone | 0.91 |
| **LLM router + classifier fallback** | **0.94** |

**5. FAISS index** — `eval/eval_faiss.py` → `eval/results/faiss_eval.json`.
FAISS flat search returns identical top-10 results to the NumPy fallback on all
277 questions; HNSW numbers are included for comparison.

**6. Earlier sets** — `eval_retrieval.py` (100 paraphrased heading queries plus
12 out-of-scope questions: Recall@5 0.96, 11/12 refusals) and
`eval_hinglish.py` (English vs Hinglish, results in
`hinglish_eval_results.json`).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py          # launch the chat app  (http://localhost:8501)
```

The committed `out/index.npz` + `out/emb_*.npy` mean the app never re-embeds
the corpus at startup — the first page load pulls the models from
Hugging Face once, then everything runs on CPU.

### Reproduce the numbers

```bash
python colab/retrieval_bench.py           # 1. retrieval + rerank benchmark (GPU recommended)
python eval/eval_calc.py                  # 2. calculator vs LLM
python colab/answer_eval.py gen small     # 3. answer quality (vLLM, GPU)
python eval/eval_router_classifier.py     # 4. intent routing
python eval/eval_faiss.py                 # 5. FAISS vs NumPy
```

### Embed a new model / rebuild the corpus

```bash
python build_model_index.py --model <hf-id>   # pre-embed the corpus with a new retriever
python build_from_pdf_sections.py             # rebuild index from the source PDFs
```

## Configuration

Retrieval works with **no configuration**. Environment variables (or Streamlit
secrets) to override defaults:

| Variable | Purpose |
|---|---|
| `TAXRAG_MODEL` | Default embedding model (default: `vivekkopthsd/fiqa-retriever-bge-small`) |
| `TAXRAG_RERANKER` | Cross-encoder (default: `vivekkopthsd/tax-reranker-minilm`) |
| `OPENCODE_ZEN_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` / `KILO_API_KEY` | Enable a specific provider in the failover chain |
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `LLM_MODEL` | Force a single custom OpenAI-compatible endpoint |

## Project layout

```
app.py                    Streamlit chat app (sidebar controls, router, PDF upload)
rag.py                    Chunking, embedding, indexing, retrieval, LLM synthesis
chat_router.py            Intent router + 5 handlers + trust tiers
intent_classifier.py      Embedding classifier used as the router fallback
reranker.py               Hybrid (dense + BM25, RRF) retrieval + cross-encoder rerank
vector_index.py           FAISS index with NumPy fallback
tax_calculator.py         Deterministic new-regime calculator, parsed from the Act
hybrid_retriever.py       BM25 / dense hybrid retrieval utilities
build_from_pdf_sections.py  Rebuild the corpus index from the Act PDFs
build_model_index.py      Pre-embed the corpus with a new model (out/emb_*.npy)
colab/                    GPU benchmarks: retrieval/rerank, answer quality (vLLM)
eval/                     Calculator, router and FAISS evaluations + results
out/index.npz             Pre-built corpus index (committed on purpose)
out/emb_*.npy             Pre-built embeddings per model (instant model switching)
data/pdfs/                Source PDFs: IT Act 2025 + Taxation Bill 2026
```

## Disclaimer

This is an educational, illustrative tool. All monetary limits, thresholds,
slab rates and deduction caps are **assessment-year dependent** and change with
each annual Finance Act. Nothing here is legal, financial or tax advice — verify
with a qualified tax professional for your assessment year.
