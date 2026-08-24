# Deployment checklist — TaxRAG

## 1. Push to GitHub

```bash
cd company_projects/intuit/TaxRAG
git status                 # confirm out/index.npz + out/emb_*.npy ARE staged,
                           # and models/ / .hf_cache / __pycache__ are NOT
git add .
git commit -m "TaxRAG: India income-tax RAG assistant with PDF upload"
git push origin master
```

> Repo: `https://github.com/bravo2024/TaxRag` (branch `master`). The three
> pre-built `out/emb_*.npy` files (≈5 MB total) are committed on purpose so the
> app loads each model's corpus embeddings instantly — no first-run re-embed.

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io → **New app**.
2. Repo `bravo2024/TaxRag`, branch `master`, main file `app.py`.
3. **Advanced settings → Python version: 3.12** (matches `runtime.txt`).
4. Set the **custom subdomain** to `indiataxrag` so the URL is
   `https://indiataxrag.streamlit.app` (matches the CV link).
5. (Optional) Under **Secrets**, add a provider key to guarantee LLM synthesis,
   e.g. `GROQ_API_KEY = "..."` or `OPENCODE_ZEN_API_KEY = "..."`. Not required —
   the anonymous free gateways work keyless.
6. Deploy. First load pulls the embedding model from Hugging Face
   (~33 MB–130 MB per model, then cached).

## 3. Verify the live app

- [ ] Ask "What is the standard deduction from salary?" → grounded (🟢) answer with cited sections.
- [ ] Ask "How is residential status in India determined?" → router routes to the statute handler.
- [ ] Ask something in Hinglish (e.g. "home loan interest kitna deduct hota hai?") → switch the embedding model to **Hinglish ST-33m** and compare.
- [ ] Ask "Who won the cricket world cup?" → abstains / out-of-scope.
- [ ] Upload a PDF → ask a question about it → answered from the upload; "✖ Use built-in tax corpus" switches back.

## 4. Update the CV link if the subdomain differs

`res/CV.tex` uses `indiataxrag.streamlit.app` and
`github.com/bravo2024/TaxRag`. If either differs, update both there and
recompile (`pdflatex CV.tex`).

## Notes

- **`out/index.npz` + `out/emb_*.npy` must be committed** — the app loads them at
  startup instead of re-embedding the corpus. They are intentionally NOT in
  `.gitignore` (`/index.npz` — the stray root-level duplicate — IS ignored).
- The embedding models live on Hugging Face
  (`vivekkopthsd/tax-retriever-bge-small` and friends), so nothing large ships
  in the repo.
- If you ever change the corpus, re-run `python build_from_pdf_sections.py`
  and `python build_model_index.py` for each model before committing.
