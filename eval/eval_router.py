"""Router accuracy on 100 labelled questions (20 per intent).

Scores the LLM router (with its keyword fallback when no LLM answers) and the
keyword heuristic alone, with a confusion matrix for each.

    python eval/eval_router.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_router import INTENTS, classify_intent, heuristic_intent  # noqa: E402

QS = json.loads((ROOT / "eval" / "router_questions.json").read_text(encoding="utf-8"))
OUT = ROOT / "eval" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def confusion(rows, key):
    m = {t: Counter() for t in INTENTS}
    for r in rows:
        m[r["intent"]][r[key]] += 1
    return {t: dict(m[t]) for t in INTENTS}


def per_class(rows, key):
    out = {}
    for t in INTENTS:
        tp = sum(r["intent"] == t and r[key] == t for r in rows)
        pred = sum(r[key] == t for r in rows)
        gold = sum(r["intent"] == t for r in rows)
        p = tp / pred if pred else 0.0
        rc = tp / gold if gold else 0.0
        out[t] = {"precision": round(p, 3), "recall": round(rc, 3),
                  "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0}
    return out


def main():
    rows = []
    for i, r in enumerate(QS, 1):
        llm = classify_intent(r["q"])
        heur = heuristic_intent(r["q"])
        rows.append({"q": r["q"], "intent": r["intent"], "llm_pred": llm["intent"],
                     "llm_provider": llm.get("provider"), "heur_pred": heur["intent"]})
        flag = "" if llm["intent"] == r["intent"] else f"  <- predicted {llm['intent']}"
        print(f"[{i:3d}] {r['intent']:<15} {r['q'][:55]}{flag}", flush=True)

    n = len(rows)
    summary = {
        "questions": n,
        "llm_router_accuracy": sum(r["llm_pred"] == r["intent"] for r in rows) / n,
        "heuristic_accuracy": sum(r["heur_pred"] == r["intent"] for r in rows) / n,
        "llm_answered_share": sum(r["llm_provider"] != "heuristic" for r in rows) / n,
        "providers": dict(Counter(r["llm_provider"] for r in rows)),
        "llm_per_class": per_class(rows, "llm_pred"),
        "heuristic_per_class": per_class(rows, "heur_pred"),
        "llm_confusion": confusion(rows, "llm_pred"),
        "heuristic_confusion": confusion(rows, "heur_pred"),
    }
    (OUT / "router_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                     ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("llm_router_accuracy", "heuristic_accuracy",
                                              "llm_answered_share", "providers")}, indent=1))
    print("LLM confusion (rows = true intent):")
    for t in INTENTS:
        print(f"  {t:<15}", summary["llm_confusion"][t])


if __name__ == "__main__":
    main()
