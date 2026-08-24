"""Offline router: embedding classifier (trained on eval/router_train.json) scored on the
same 100 test questions as the LLM router, with the keyword heuristic for comparison.

    python eval/eval_router_classifier.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_router import INTENTS, heuristic_intent  # noqa: E402
from intent_classifier import IntentClassifier  # noqa: E402
from rag import build_embedder  # noqa: E402

test = json.loads((ROOT / "eval" / "router_questions.json").read_text(encoding="utf-8"))
train = json.loads((ROOT / "eval" / "router_train.json").read_text(encoding="utf-8"))
assert not ({r["q"].lower() for r in test} & {r["q"].lower() for r in train}), "train/test overlap"

model = build_embedder()
clf = IntentClassifier(model)
rows, t0 = [], time.time()
for r in test:
    c = clf.predict(r["q"])
    rows.append({"q": r["q"], "intent": r["intent"], "clf_pred": c["intent"], "clf_conf": c["confidence"],
                 "heur_pred": heuristic_intent(r["q"])["intent"]})
ms = 1000 * (time.time() - t0) / len(rows)

llm = {r["q"]: r for r in json.loads((ROOT / "eval" / "results" / "router_eval.json").read_text(encoding="utf-8"))["rows"]}
for r in rows:
    l = llm[r["q"]]
    r["llm_pred"] = l["llm_pred"] if l["llm_provider"] != "heuristic" else None
    # production router: LLM when it answers, otherwise the fallback
    r["router_old"] = r["llm_pred"] or r["heur_pred"]
    r["router_new"] = r["llm_pred"] or r["clf_pred"]
    # as deployed: with no LLM reachable, the fallback sends GENERAL_TAX to statute lookup
    r["router_deployed"] = r["llm_pred"] or ("STATUTE_LOOKUP" if r["clf_pred"] == "GENERAL_TAX" else r["clf_pred"])

acc = lambda k, rs=rows: sum(r[k] == r["intent"] for r in rs) / len(rs)
answered = [r for r in rows if r["llm_pred"]]
conf = {t: dict(Counter(r["clf_pred"] for r in rows if r["intent"] == t)) for t in INTENTS}
summary = {"test_questions": len(rows), "train_examples": len(train),
           "classifier_accuracy": acc("clf_pred"), "heuristic_accuracy": acc("heur_pred"),
           "classifier_ms_per_query": ms,
           "llm_accuracy_when_it_answered": acc("llm_pred", answered), "llm_answered": len(answered),
           "classifier_accuracy_on_same_questions": acc("clf_pred", answered),
           "router_with_keyword_fallback": acc("router_old"), "router_with_classifier_fallback": acc("router_new"),
           "router_as_deployed_general_to_statute": acc("router_deployed"),
           "classifier_confusion": conf}
(ROOT / "eval" / "results" / "router_classifier_eval.json").write_text(
    json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "classifier_confusion"}, indent=1))
for t in INTENTS:
    print(f"  {t:<15}", conf[t])
print("classifier errors:")
for r in rows:
    if r["clf_pred"] != r["intent"]:
        print(f"  {r['intent']:<15} -> {r['clf_pred']:<15} ({r['clf_conf']:.2f}) {r['q']}")
