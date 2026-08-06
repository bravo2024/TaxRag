# TaxRAG — India Income Tax Q&A Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about
Indian income tax by retrieving the relevant provisions and generating a clear,
**cited** answer. You can also **upload your own PDF** and ask questions about it.

**Live app:** [https://indiataxrag.streamlit.app](https://indiataxrag.streamlit.app/) · **CPU-only** · deploys free on Streamlit Cloud

---

## What it does

- **Retrieval over a curated corpus** of 34 well-established provisions of the
  Income Tax Act, 1961 (80C, 80D, HRA, Section 24, old vs new regime, capital
  gains, TDS, ITR forms, and more), stored as plain markdown.
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

Measured with the eval sets in `rag.py` (`python rag.py`):

| Set | Metric | Score |
|---|---|---|
| Curated queries (11) | Recall@4 | 1.00 |
| Curated out-of-scope (5) | Refusal rate | 1.00 |
| **Hard** paraphrased queries (24) | Recall@1 / Recall@4 / MRR | 0.83 / 0.96 / 0.90 |
| **Hard** adversarial out-of-scope (5) | Refusal rate | 0.80 |

The *hard* set uses colloquial taxpayer phrasing with no section numbers and
deliberate near-miss traps (e.g. 80TTA vs 80TTB, Section 54 vs 54EC), and is the
honest generalization number. The fine-tuned bi-encoder beats a stock
`all-MiniLM-L6-v2` baseline on this set (Recall@1 0.83 vs 0.79).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py          # build the index + print evaluation
streamlit run app.py   # launch the web app
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
