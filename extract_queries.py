#!/usr/bin/env python3
"""Reuse eval_rerank.py's exact 100-query gold-set logic (SUBS/SYN/paraphrase/
must/pool/chosen) without loading any models. Writes queries_en.json:
list of {gold, en}.

This duplicates only the pure-python query-selection code from eval_rerank.py
(no model/BM25 loading) so we can regenerate the identical gold set cheaply.
Do not modify eval_rerank.py itself.
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
random.seed(7)

d_path = os.path.join(OUT, "index.npz")
import numpy as np
d = np.load(d_path, allow_pickle=True)
titles = [str(t) for t in d["chunk_titles"]]

sec = {}
for t in titles:
    m = re.match(r"Section (\d+): (.+)", t)
    if m:
        sec.setdefault(int(m.group(1)), m.group(2))

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
queries = [(s, paraphrase(sec[s])) for s in chosen]

out = [{"gold": s, "en": q} for s, q in queries]
out_path = os.path.join(HERE, "queries_en.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(out)} queries to {out_path}")
for row in out[:5]:
    print(row)
