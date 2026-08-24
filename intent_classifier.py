"""Offline intent classifier: logistic regression on the app's sentence embeddings.

Used when no LLM router is reachable (replaces the keyword heuristic). It needs no
network, answers in milliseconds, and reuses the embedding model the app already loads.
An extra feature flags an income amount in the text (see tax_calculator.parse_income).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from tax_calculator import parse_income

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "eval" / "router_train.json"


def _features(model, texts: List[str]) -> np.ndarray:
    E = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    amount = np.array([[1.0 if parse_income(t) else 0.0] for t in texts])
    return np.hstack([E, amount])


class IntentClassifier:
    def __init__(self, model, train_path: Path = TRAIN, C: float = 4.0):
        rows = json.loads(Path(train_path).read_text(encoding="utf-8"))
        self.model = model
        self.clf = LogisticRegression(C=C, max_iter=2000)
        self.clf.fit(_features(model, [r["q"] for r in rows]), [r["intent"] for r in rows])

    def predict(self, query: str) -> Dict:
        p = self.clf.predict_proba(_features(self.model, [query]))[0]
        k = int(np.argmax(p))
        return {"intent": str(self.clf.classes_[k]), "confidence": float(p[k]),
                "reason": "embedding classifier", "provider": "classifier"}
