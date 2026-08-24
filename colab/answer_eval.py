"""Answer-level evaluation (Phase D) and small-vs-large model routing (Phase E). Colab T4.

    python answer_eval.py            # orchestrates everything below
    python answer_eval.py gen <key>  # one generator (separate process so vLLM frees the GPU)
    python answer_eval.py judge      # LLM-as-judge over all answers

Pipeline under test = the app's: FiQA-tuned dense retriever -> top-5 passages -> LLM with the
app's exact system prompt. Questions: content-based test questions (held-out sections) plus
out-of-scope tax questions. Judge is a different, larger model than the generators it scores.
"""
import gc
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import os
IN = Path(os.environ.get("EVAL_IN", "/content"))
OUT = Path(os.environ.get("EVAL_OUT", "/content/out"))
TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "/content/hf_token"))
OUT.mkdir(parents=True, exist_ok=True)

GENERATORS = {
    "small": {"id": "Qwen/Qwen3-1.7B", "label": "Qwen3-1.7B (SLM)", "params_b": 1.7, "think_flag": True},
    "large": {"id": "Qwen/Qwen3-4B-Instruct-2507", "label": "Qwen3-4B-Instruct", "params_b": 4.0, "think_flag": False},
}
JUDGES = ["Qwen/Qwen3-8B-AWQ", "microsoft/Phi-4-mini-instruct"]   # first that loads is used

HF_REPO = "vivekkopthsd/taxrag-eval-runs"   # private dataset: every stage output is saved here


def _hf():
    from huggingface_hub import HfApi
    tok = TOKEN_FILE.read_text().strip()
    api = HfApi(token=tok)
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    return api


def sync(path):
    """Upload one finished output so a lost VM never loses a completed stage."""
    if not TOKEN_FILE.exists():
        return
    try:
        _hf().upload_file(path_or_fileobj=str(path), path_in_repo=f"phase_d/{Path(path).name}",
                          repo_id=HF_REPO, repo_type="dataset", commit_message=f"stage output {Path(path).name}")
        print(f"[sync] saved {Path(path).name}", flush=True)
    except Exception as e:
        print(f"[sync] FAILED {Path(path).name}: {e!r}"[:300], flush=True)


def restore():
    """Pull already-finished stage outputs so a rerun resumes where the last VM stopped."""
    from huggingface_hub import hf_hub_download
    if not TOKEN_FILE.exists():
        return
    api = _hf()
    for f in api.list_repo_files(HF_REPO, repo_type="dataset"):
        if f.startswith("phase_d/") and not (OUT / Path(f).name).exists():
            hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir="/content/restore", token=api.token)
            (OUT / Path(f).name).write_bytes((Path("/content/restore") / f).read_bytes())
            print(f"[restore] {Path(f).name}", flush=True)

SYNTH_SYSTEM = (
    "You are a precise Indian income tax assistant that answers ONLY from the "
    "provided context. Cite the source (section number/title) for each fact. If "
    "the context is insufficient, say so clearly. Never invent or guess figures."
)


def build_items():
    from sentence_transformers import SentenceTransformer
    import numpy as np
    chunks = json.loads((IN / "chunks.json").read_text(encoding="utf-8"))
    gen = json.loads((OUT / "gen_questions.json").read_text(encoding="utf-8"))["rows"]
    oos = json.loads((IN / "oos_questions.json").read_text(encoding="utf-8"))
    test = [r for r in gen if r["split"] == "test"]
    m = SentenceTransformer("vivekkopthsd/fiqa-retriever-bge-small", device="cuda")
    m.max_seq_length = 512
    E = m.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True)
    items = [{"id": i, "q": r["question"], "gold": r["section"], "scope": "in"} for i, r in enumerate(test)]
    items += [{"id": len(items) + j, "q": q, "gold": None, "scope": "out"} for j, q in enumerate(oos)]
    Q = m.encode([it["q"] for it in items], batch_size=64, normalize_embeddings=True)
    S = Q @ E.T
    for it, s in zip(items, S):
        top = np.argsort(-s)[:5]
        it["passages"] = [{"title": chunks[i]["title"], "text": chunks[i]["text"][:1800],
                           "section": chunks[i]["section"]} for i in top]
        it["max_sim"] = float(s[top[0]])
        it["gold_retrieved"] = it["gold"] in {p["section"] for p in it["passages"]}
    (OUT / "answer_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    sync(OUT / "answer_items.json")
    print(f"items: {sum(i['scope']=='in' for i in items)} in-scope, {sum(i['scope']=='out' for i in items)} out-of-scope")


def user_prompt(it):
    ctx = "\n\n---\n\n".join(f"[{p['title']}] (Source: Income-Tax Act 2025)\n{p['text']}" for p in it["passages"])
    return ("Answer the user's question using ONLY the context below. Cite the source "
            "for each fact. If the context does not contain enough information, say so. "
            "Do NOT invent or guess. Keep the answer concise.\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {it['q']}\n\nANSWER:")


def generate(key):
    from vllm import LLM, SamplingParams
    cfg = GENERATORS[key]
    items = json.loads((OUT / "answer_items.json").read_text(encoding="utf-8"))
    llm = LLM(model=cfg["id"], dtype="half", max_model_len=6144, gpu_memory_utilization=0.80, max_num_seqs=48, seed=13)
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if cfg["think_flag"] else {}
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYNTH_SYSTEM},
                                        {"role": "user", "content": user_prompt(it)}],
                                       tokenize=False, add_generation_prompt=True, **kw) for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=400))
    secs = time.time() - t0
    toks = [len(o.outputs[0].token_ids) for o in outs]
    res = {"model": cfg["label"], "params_b": cfg["params_b"], "batch_seconds": secs,
           "answers": [re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).strip() for o in outs],
           "gen_tokens": toks}
    (OUT / f"answers_{key}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    sync(OUT / f"answers_{key}.json")
    print(f"{cfg['label']}: {len(outs)} answers in {secs:.0f}s, mean {sum(toks)/len(toks):.0f} tokens")


JUDGE_PROMPT = """You are grading an answer from a tax assistant that must answer ONLY from the given context.

CONTEXT:
{ctx}

QUESTION: {q}

ANSWER:
{a}

Grade strictly and return only JSON with these keys:
- "faithful": "yes" if every factual claim in the answer is supported by the context, "partial" if some are not, "no" if the main claims are not supported. An answer that only says the context is insufficient counts as "yes".
- "answers_question": "yes" if it directly answers the question, "partial" if incomplete, "no" if it does not answer (including refusals).
- "abstained": "yes" if the answer says the context does not contain the information, otherwise "no".
JSON:"""


def judge():
    from vllm import LLM, SamplingParams
    items = json.loads((OUT / "answer_items.json").read_text(encoding="utf-8"))
    answers = {k: json.loads((OUT / f"answers_{k}.json").read_text(encoding="utf-8")) for k in GENERATORS
               if (OUT / f"answers_{k}.json").exists()}
    llm, used = None, None
    for jid in JUDGES:
        try:
            llm = LLM(model=jid, dtype="half", max_model_len=8192, gpu_memory_utilization=0.80, max_num_seqs=48, seed=13)
            used = jid
            break
        except Exception as e:
            print("judge failed to load", jid, repr(e)[:300])
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "Qwen3" in used else {}
    jobs, prompts = [], []
    for k, res in answers.items():
        for it, a in zip(items, res["answers"]):
            ctx = "\n\n".join(f"[{p['title']}]\n{p['text']}" for p in it["passages"])
            prompts.append(tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(ctx=ctx, q=it["q"], a=a)}],
                                                   tokenize=False, add_generation_prompt=True, **kw))
            jobs.append((k, it["id"]))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=120))
    grades = {}
    for (k, iid), o in zip(jobs, outs):
        m = re.search(r"\{.*?\}", o.outputs[0].text, re.S)
        try:
            g = json.loads(m.group(0)) if m else {}
        except Exception:
            g = {}
        grades.setdefault(k, {})[iid] = {x: str(g.get(x, "unparsed")).lower() for x in ("faithful", "answers_question", "abstained")}
    (OUT / "judgements.json").write_text(json.dumps({"judge": used, "grades": grades}), encoding="utf-8")
    sync(OUT / "judgements.json")
    print("judge:", used, "graded", len(jobs))


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"b_only": 0, "a_only": 0, "p": 1.0}
    k = min(n01, n10)
    return {"b_only": n01, "a_only": n10, "p": min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)}


def report():
    items = json.loads((OUT / "answer_items.json").read_text(encoding="utf-8"))
    J = json.loads((OUT / "judgements.json").read_text(encoding="utf-8"))
    ans = {k: json.loads((OUT / f"answers_{k}.json").read_text(encoding="utf-8")) for k in J["grades"]}
    cite = lambda a, s: bool(re.search(rf"\bsection\s*{s}\b|\[\s*section\s*{s}\s*:", a, re.I))
    per = {}
    for k, res in ans.items():
        g = J["grades"][k]
        rows = []
        for it, a in zip(items, res["answers"]):
            gr = g.get(str(it["id"]), g.get(it["id"], {}))
            rows.append({"id": it["id"], "scope": it["scope"], "gold_retrieved": it["gold_retrieved"],
                         "max_sim": it["max_sim"],
                         "faithful": gr.get("faithful") == "yes",
                         "answers": gr.get("answers_question") == "yes",
                         "abstained": gr.get("abstained") == "yes",
                         "cites_gold": it["scope"] == "in" and cite(a, it["gold"])})
        per[k] = rows
    summary = {"judge": J["judge"], "models": {}}
    for k, rows in per.items():
        ins = [r for r in rows if r["scope"] == "in"]
        hit = [r for r in ins if r["gold_retrieved"]]
        miss = [r for r in ins if not r["gold_retrieved"]]
        oos = [r for r in rows if r["scope"] == "out"]
        good = [r["faithful"] and r["answers"] for r in hit]
        summary["models"][GENERATORS[k]["label"]] = {
            "in_scope_questions": len(ins),
            "faithfulness_rate": sum(r["faithful"] for r in ins) / len(ins),
            "hallucination_rate": 1 - sum(r["faithful"] for r in ins) / len(ins),
            "correct_and_grounded_when_gold_retrieved": sum(good) / len(hit),
            "cites_gold_section_when_retrieved": sum(r["cites_gold"] for r in hit) / len(hit),
            "abstains_when_gold_not_retrieved": sum(r["abstained"] for r in miss) / max(len(miss), 1),
            "out_of_scope_abstention": sum(r["abstained"] for r in oos) / len(oos),
            "seconds_per_answer": ans[k]["batch_seconds"] / len(rows),
            "mean_generated_tokens": sum(ans[k]["gen_tokens"]) / len(rows),
        }
    summary["retrieval_hit5_on_in_scope"] = sum(i["gold_retrieved"] for i in items if i["scope"] == "in") / sum(i["scope"] == "in" for i in items)
    if "small" in per and "large" in per:
        a = [r["faithful"] and r["answers"] for r in per["small"] if r["scope"] == "in"]
        b = [r["faithful"] and r["answers"] for r in per["large"] if r["scope"] == "in"]
        summary["mcnemar_correct_grounded_small_vs_large"] = mcnemar(a, b)

        # Phase E: route to the small model when retrieval is confident, else the large one.
        # Threshold chosen on even-id questions, reported on odd-id questions (no tuning on the report half).
        S = {r["id"]: r for r in per["small"]}; L = {r["id"]: r for r in per["large"]}
        cost = {"small": summary["models"][GENERATORS["small"]["label"]]["seconds_per_answer"],
                "large": summary["models"][GENERATORS["large"]["label"]]["seconds_per_answer"]}
        ok = lambda r: (r["abstained"] if r["scope"] == "out" else (r["faithful"] and r["answers"]))

        def policy(ids, tau):
            q = c = 0.0
            for i in ids:
                use_small = S[i]["max_sim"] >= tau
                r = S[i] if use_small else L[i]
                q += ok(r); c += cost["small"] if use_small else cost["large"]
            return q / len(ids), c / len(ids), sum(S[i]["max_sim"] >= tau for i in ids) / len(ids)

        ids = sorted(S)
        tune, hold = [i for i in ids if i % 2 == 0], [i for i in ids if i % 2 == 1]
        large_q_tune = policy(tune, 9.0)[0]
        grid = [round(0.30 + 0.02 * j, 2) for j in range(26)]
        feasible = [t for t in grid if policy(tune, t)[0] >= large_q_tune - 0.01]
        tau = min(feasible, key=lambda t: policy(tune, t)[1]) if feasible else 9.0
        qh, ch, share = policy(hold, tau)
        ql, cl, _ = policy(hold, 9.0)
        qs, cs, _ = policy(hold, -9.0)
        summary["routing"] = {"threshold_on_max_similarity": tau, "rule": "small model if top retrieval similarity >= threshold",
                              "holdout_questions": len(hold), "share_routed_to_small": share,
                              "quality_router": qh, "quality_large_only": ql, "quality_small_only": qs,
                              "seconds_router": ch, "seconds_large_only": cl, "seconds_small_only": cs,
                              "compute_saving_vs_large_only": 1 - ch / cl}
    (OUT / "answer_eval_results.json").write_text(json.dumps({"summary": summary, "per_question": per}, indent=1), encoding="utf-8")
    sync(OUT / "answer_eval_results.json")
    print("PHASE_D_DONE", flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "items":
        build_items()
    elif len(sys.argv) > 1 and sys.argv[1] == "gen":
        generate(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "judge":
        judge()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        # every GPU stage runs in its own process so memory is fully released between stages
        me = str(Path(__file__).resolve())
        restore()
        if not (OUT / "answer_items.json").exists():
            subprocess.run([sys.executable, me, "items"], check=True)
        for k in GENERATORS:
            if not (OUT / f"answers_{k}.json").exists():
                subprocess.run([sys.executable, me, "gen", k], check=True)
        if not (OUT / "judgements.json").exists():
            subprocess.run([sys.executable, me, "judge"], check=True)
        report()
