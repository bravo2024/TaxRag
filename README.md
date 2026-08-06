# TaxRAG — India Income Tax Q&A Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about
Indian income tax by retrieving the relevant provisions and generating a clear,
**cited** answer. You can also **upload your own PDF** and ask questions about it.

**Live app:** https://indiataxrag.streamlit.app · **CPU-only** · deploys free on Streamlit Cloud

---

## What it does

- **Retrieval over the full Income-Tax Act, 2025** (as amended by Finance Act
  2026) — **477 sections indexed as 1,237 section-level passages**, sliced from
  the authoritative Act PDF with section labels anchored to the machine-readable
  [Income-Tax-Act-2025](https://huggingface.co/datasets/ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text)
  dataset (MIT licence). This is the actual statute text, not a summary.
- **Bring your own document** — upload a PDF and the app chunks, embeds and
  indexes it on the fly, then answers questions grounded in that document.
- **Abstention** — if nothing in the corpus is similar enough to the question,
  the assistant refuses rather than guessing. Tax accuracy matters more than
  coverage.
- **Cited answers** — every answer shows the source sections it was built from.

## How it works

1. **Embedding** — chunks are embedded with a **fine-tuned `bge-small`
   bi-encoder** ([`vivekkopthsd/fiqa-retriever-bge-small`](https://huggingface.co/vivekkopthsd/fiqa-retriever-bge-small),
   fine-tuned on financial QA), 384-dim, on CPU.
2. **Indexing** — L2-normalized embeddings are stored in a small numpy vector
   index (`out/index.npz`), built once and committed so the app never
   re-embeds the corpus at startup.
3. **Retrieval** — the query is embedded, cosine similarity ranks the chunks,
   and the top-k above a similarity threshold are returned.
4. **Synthesis (optional)** — the retrieved passages are passed to a
   multi-provider LLM chain (OpenCode → Groq → NVIDIA) with automatic failover.
   The anonymous free gateway needs no key, so synthesis works out of the box;
   if every provider is unreachable the app falls back to **extractive** mode
   (showing the retrieved passages directly).

## Evaluation

Reproduce with **`python eval_retrieval.py`** (self-contained; reads the built
`out/index.npz`). It builds a **100-query hard set** by paraphrasing real section
headings into natural taxpayer questions — with synonym substitution so the
retriever must match on *meaning*, not keywords — deliberately loaded with the
confusable "Deduction in respect of…" and loss set-off clusters where near-duplicate
sections compete. Then it measures retrieval against the gold section and
abstention on out-of-scope questions:

| Set | Metric | Score |
|---|---|---|
| Hard paraphrased queries (100) | Recall@1 | 0.78 |
| Hard paraphrased queries (100) | **Recall@5** | **0.96** |
| Hard paraphrased queries (100) | **MRR@5** | **0.85** |
| Out-of-scope questions (12) | **Refusals** | **11 / 12** |
| Corpus scale | passages / sections | 1,237 / 477 |

Out-of-scope questions score 0.24–0.41 similarity, cleanly under the 0.45
abstention gate — which is *why* the refusals are reliable, not luck. On a
separate small labelled set the fine-tuned bi-encoder also beats a stock
`all-MiniLM-L6-v2` baseline (Recall@1 1.00 vs 0.82).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python eval_retrieval.py   # reproduce the retrieval + abstention metrics
streamlit run app.py       # launch the web app
```

## Configuration

Retrieval works with **no configuration**. To force a specific LLM for answer
synthesis, set environment variables (or Streamlit secrets):

| Variable | Purpose |
|---|---|
| `TAXRAG_MODEL` | Override the retrieval model (default: the fine-tuned HF model) |
| `OPENCODE_ZEN_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` | Enable a specific provider in the failover chain |
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `LLM_MODEL` | Force a single custom OpenAI-compatible endpoint |

## Disclaimer

This is an educational, illustrative tool. All monetary limits, thresholds and
rates are assessment-year dependent and change with each annual Finance Act.
Nothing here is legal, financial or tax advice — verify with a qualified tax
professional for your assessment year.
