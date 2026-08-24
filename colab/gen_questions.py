"""Generate a realistic retrieval test set: questions written from each section's content.

Runs on Colab (T4) with vLLM. Input /content/sections.json; output /content/out/gen_questions.json.
Each question is tied to the section it was generated from (gold). Questions that mention a
section number or mostly reuse the heading's words are dropped, so the set cannot be solved
by title matching. Sections are split into disjoint train/test groups (seed 13).
"""
import json
import random
import re
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = Path("/content/out")
OUT.mkdir(parents=True, exist_ok=True)
sections = json.loads(Path("/content/sections.json").read_text(encoding="utf-8"))

PROMPT = """Below is a provision of India's Income-Tax Act, 2025.

Heading: {title}
Text:
{text}

Write 2 different questions that a taxpayer or an accountant might ask, where the answer is found in this provision.
Rules:
- Use plain everyday language, the way a person would type into a search box.
- Do NOT mention any section number, and do NOT copy the heading's wording.
- Each question must be answerable from the text above.
Return only JSON: {{"questions": ["...", "..."]}}"""

llm = LLM(model=MODEL, dtype="half", max_model_len=4096, gpu_memory_utilization=0.90, seed=13)
tok = llm.get_tokenizer()
prompts = []
for s in sections:
    text = s["text"][:6000]
    msg = [{"role": "user", "content": PROMPT.format(title=s["title"], text=text)}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    while len(tok(p).input_ids) > 3600:  # keep prompt + answer inside the context window
        text = text[: int(len(text) * 0.8)]
        msg = [{"role": "user", "content": PROMPT.format(title=s["title"], text=text)}]
        p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    prompts.append(p)

outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, seed=13))

STOP = {"the", "a", "an", "of", "in", "for", "to", "and", "or", "on", "by", "with", "under", "certain", "cases", "case"}


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


rows, dropped = [], {"parse": 0, "section_number": 0, "title_copy": 0}
for s, o in zip(sections, outs):
    raw = o.outputs[0].text
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        qs = json.loads(m.group(0))["questions"] if m else []
    except Exception:
        qs = []
    if not qs:
        dropped["parse"] += 1
        continue
    tw = words(s["title"])
    for q in qs[:2]:
        q = str(q).strip()
        if re.search(r"\bsection\s*\d|\bsec\.?\s*\d", q, re.I):
            dropped["section_number"] += 1
            continue
        if tw and len(words(q) & tw) / len(tw) >= 0.7:
            dropped["title_copy"] += 1
            continue
        rows.append({"section": s["section"], "title": s["title"], "question": q})

random.seed(13)
secs = sorted({r["section"] for r in rows})
random.shuffle(secs)
test_secs = set(secs[:150])
for r in rows:
    r["split"] = "test" if r["section"] in test_secs else "train"

summary = {"model": MODEL, "sections": len(sections), "questions": len(rows),
           "test_questions": sum(r["split"] == "test" for r in rows),
           "train_questions": sum(r["split"] == "train" for r in rows),
           "test_sections": len(test_secs), "dropped": dropped}
(OUT / "gen_questions.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
print(json.dumps(summary, indent=1))
for r in random.sample(rows, 8):
    print(f"[{r['section']}] {r['title']}  ->  {r['question']}")
