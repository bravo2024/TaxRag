"""Calculation accuracy: LLM arithmetic over retrieved slab text vs the calculator tool.

Both paths get the same 30 questions (eval/calc_questions.json, expected values
worked out by hand from the new-regime rules). The LLM answer counts as correct
when the amount on its final payable/total line matches the expected tax (±₹1).
Raw answers are saved for manual audit.

    python eval/eval_calc.py
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_router import handle_calculation, handle_calculation_llm  # noqa: E402
from rag import build_embedder, load_index  # noqa: E402

QS = json.loads((ROOT / "eval" / "calc_questions.json").read_text(encoding="utf-8"))
OUT = ROOT / "eval" / "results"
OUT.mkdir(parents=True, exist_ok=True)

FINAL = re.compile(r"\b(?:payable|total tax|net tax|final tax|tax liability|total income[- ]tax|you (?:will )?pay|tax due)\b", re.I)
ZERO = re.compile(r"\b(nil|zero|no tax|₹\s*0\b|rs\.?\s*0\b|0 tax|₹0)", re.I)


def amounts(text):
    vals = []
    for m in re.finditer(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?", text, re.I):
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        v *= 1e7 if unit.startswith("crore") else 1e5 if unit.startswith(("lakh", "lac")) else 1
        vals.append(v)
    return vals


def final_amount_correct(answer, expected):
    lines = [l for l in answer.splitlines() if FINAL.search(l)]
    if not lines:
        return False
    last = lines[-1]
    if expected == 0 and ZERO.search(last):
        return True
    return any(abs(v - expected) <= 1 for v in amounts(last))


def main():
    model = build_embedder()
    emb, chunks = load_index(str(ROOT / "out"))
    rows = []
    for i, r in enumerate(QS, 1):
        t0 = time.time()
        llm = handle_calculation_llm(r["q"], model, emb, chunks)
        t_llm = time.time() - t0
        t0 = time.time()
        tool = handle_calculation(r["q"], model, emb, chunks)
        t_tool = time.time() - t0
        tool_val = (tool.get("calculation") or {}).get("tax_payable")
        rows.append({
            "q": r["q"], "expected": r["expected"],
            "llm_correct": final_amount_correct(llm["answer"], r["expected"]),
            "llm_provider": llm.get("provider"), "llm_seconds": round(t_llm, 2), "llm_answer": llm["answer"],
            "tool_correct": tool_val is not None and abs(tool_val - r["expected"]) <= 1,
            "tool_value": tool_val, "tool_seconds": round(t_tool, 3),
        })
        print(f"[{i:2d}/{len(QS)}] llm={'OK ' if rows[-1]['llm_correct'] else 'BAD'} "
              f"tool={'OK ' if rows[-1]['tool_correct'] else 'BAD'} {r['q'][:60]}", flush=True)

    n = len(rows)
    summary = {
        "questions": n,
        "llm_accuracy": sum(r["llm_correct"] for r in rows) / n,
        "tool_accuracy": sum(r["tool_correct"] for r in rows) / n,
        "llm_providers": sorted({r["llm_provider"] or "none" for r in rows}),
        "llm_median_seconds": sorted(r["llm_seconds"] for r in rows)[n // 2],
        "tool_median_seconds": sorted(r["tool_seconds"] for r in rows)[n // 2],
    }
    (OUT / "calc_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
