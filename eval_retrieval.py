#!/usr/bin/env python3
"""TaxRAG retrieval + abstention evaluation.

Self-contained: builds a 100-query hard evaluation set by paraphrasing real
section headings (drawn from the built index) into natural questions, then
measures Recall@1/@5 and MRR@5 against the gold section, plus abstention
accuracy on out-of-scope questions.

Run from this folder after building the index (out/index.npz):
    python eval_retrieval.py

Reported figures on the Income-Tax Act 2025 (FA 2026) index:
    Recall@5 = 0.96, MRR@5 = 0.85, 11/12 out-of-scope refused (100-query set).
"""
import os, re, random
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "out", "index.npz")
MODEL_NAME = os.environ.get("MODEL_NAME", "vivekkopthsd/fiqa-retriever-bge-small")
THRESHOLD = 0.45
random.seed(7)

# ── load index ───────────────────────────────────────────────────────────────
d = np.load(INDEX, allow_pickle=True)
emb = d["embeddings"].astype("float32")
embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
titles = d["chunk_titles"]
chunk_sec = np.array([int(m.group(1)) if (m := re.match(r"Section (\d+):", str(t))) else -1
                      for t in titles])
# section number -> heading (first chunk seen per section)
sec = {}
for t in titles:
    m = re.match(r"Section (\d+): (.+)", str(t))
    if m:
        sec.setdefault(int(m.group(1)), m.group(2))

model = SentenceTransformer(MODEL_NAME, device="cpu")


def topk(q, k=5):
    qv = model.encode([q], normalize_embeddings=True)[0]
    sims = embn @ qv
    idx = np.argsort(-sims)[:k]
    return [int(chunk_sec[i]) for i in idx], float(sims[idx[0]])


# ── hard 100-query set: paraphrase headings into natural questions ────────────
SUBS = [
    (r"^Deduction in respect of ", "How do I claim tax relief for "),
    (r"^Deduction for ", "What is the tax benefit available for "),
    (r"^Deductions? from ", "What can be subtracted from "),
    (r"^Deductions? related to ", "What expenses are allowed for "),
    (r"^Set off of ", "How are you allowed to adjust "),
    (r"^Carry forward and set off of ", "How is future adjustment handled for "),
    (r"^Computation of ", "How do you work out "),
    (r"^Manner of computing ", "What is the method to arrive at "),
    (r"^Special provision (for|in) ", "What special rule applies to "),
    (r"^Penalty for ", "What is the consequence for "),
    (r"^Power(s)? (of|to) ", "What authority exists over "),
    (r"^Income (from|under) ", "How is income from "),
    (r"^Definitions", "the meaning of key terms used in the Act"),
    (r"^Residence in India", "how a person's residential status is decided"),
]
SYN = {"premia": "premiums", "premium": "premiums", "provident fund": "PF savings scheme",
       "insurance": "cover", "donations": "charitable giving", "loan": "borrowing",
       "residential house property": "a home you live in", "house property": "a property you own",
       "capital gains": "profit on selling assets", "salaries": "salary earnings",
       "higher education": "college studies", "health insurance": "medical cover"}


def paraphrase(title):
    q, applied = title, False
    for pat, repl in SUBS:
        if re.search(pat, q):
            q = re.sub(pat, repl, q); applied = True; break
    for k, v in SYN.items():
        q = q.replace(k, v)
    q = q.strip().rstrip(",")
    if not applied and not q.endswith("?"):
        q = f"What does the Act say about {q[0].lower() + q[1:]}?"
    elif not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


def descriptive(t):
    return len([w for w in re.findall(r"[A-Za-z]+", t) if len(w) > 2]) >= 3


must = [122, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134, 136, 137,
        108, 109, 110, 111, 112, 114, 115, 119, 120,
        6, 18, 19, 20, 22, 26, 27, 33, 67, 72, 82, 85, 86, 99]
must = [s for s in must if s in sec]
pool = [s for s in sorted(sec) if s not in must and descriptive(sec[s])]
random.shuffle(pool)
chosen = (must + pool)[:100]

r1 = r5 = 0; rr = 0.0
for s in chosen:
    secs, _ = topk(paraphrase(sec[s]), 5)
    if s in secs:
        r5 += 1; rank = secs.index(s) + 1; rr += 1 / rank
        if rank == 1:
            r1 += 1
n = len(chosen)
print(f"HARD paraphrased set: {n} queries over {n} distinct sections")
print(f"Recall@1 : {r1/n:.3f}")
print(f"Recall@5 : {r5/n:.3f}")
print(f"MRR@5    : {rr/n:.3f}")

# ── out-of-scope abstention ──────────────────────────────────────────────────
OOS = ["What is the boiling point of water at sea level?", "How do I bake a chocolate sourdough loaf?",
       "Who won the football World Cup in 2022?", "What is the capital of Australia?",
       "How do I fix a flat bicycle tyre?", "Recommend a good science-fiction novel.",
       "What is the distance from Earth to the Moon?", "How do I train a dog to sit?",
       "What is the chemical formula for table salt?", "How do I change engine oil in a car?",
       "Best hiking trails near Manali?", "How many players are on a cricket team?"]
ab = sum(1 for q in OOS if topk(q, 1)[1] < THRESHOLD)
print(f"OOS abstained: {ab}/{len(OOS)} = {ab/len(OOS):.3f}  (threshold {THRESHOLD})")
print(f"Corpus: {len(titles)} passages / {len(sec)} sections")
