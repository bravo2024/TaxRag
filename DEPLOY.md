# Deployment checklist — TaxRAG

## 1. Push to GitHub

```bash
cd company_projects/builds/TaxRAG
git init
git add .
git status                 # confirm out/index.npz + out/metrics.json ARE staged,
                           # and build.log / __pycache__ / .venv are NOT
git commit -m "TaxRAG: India income-tax RAG assistant with PDF upload"
git branch -M main
git remote add origin https://github.com/bravo2024/TaxRAG.git
git push -u origin main
```

> Create the empty `bravo2024/TaxRAG` repo on GitHub first (no README, so the
> push isn't rejected).

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io → **New app**.
2. Repo `bravo2024/TaxRAG`, branch `main`, main file `app.py`.
3. **Advanced settings → Python version: 3.12**.
4. Set the **custom subdomain** to `taxrag` so the URL is
   `https://taxrag.streamlit.app` (matches the CV link).
5. (Optional) Under **Secrets**, add a provider key to guarantee LLM synthesis,
   e.g. `GROQ_API_KEY = "..."`. Not required — the free gateway works keyless.
6. Deploy. First load pulls the HF model (~130 MB) once, then caches it.

## 3. Verify the live app

- [ ] Ask "How much deduction under Section 80C?" → returns cited provisions + answer.
- [ ] Ask "Who won the cricket world cup?" → abstains.
- [ ] Upload a PDF → ask a question about it → answers from the uploaded doc.
- [ ] Clear the uploaded doc → returns to the tax corpus.

## 4. Update the CV link if the subdomain differs

`res/CV.tex` uses `taxrag.streamlit.app` and
`github.com/bravo2024/TaxRAG`. If either differs, update both there and
recompile (`pdflatex CV.tex`).

## Notes

- **`out/index.npz` (68 KB) must be committed** — the app loads it at startup
  instead of re-embedding the corpus. It is intentionally NOT in `.gitignore`.
- The retrieval model lives on Hugging Face
  (`vivekkopthsd/fiqa-retriever-bge-small`), so nothing large ships in the repo.
- If you ever change the corpus in `data/`, re-run `python rag.py` to rebuild
  `out/index.npz` before committing.
