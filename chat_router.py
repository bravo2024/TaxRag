"""Dual-mode routed TaxRAG: intent classifier + 5 handlers.

Fixes 3 failure modes of pure single-shot RAG:
  1. Greetings hit abstention gate (LLM never called) → route to LLM chat.
  2. Retrieval fails or context is wrong → route to LLM memory with disclaimer.
  3. Calculations return raw slab passages → compute with retrieved rates.

Trust tiers surfaced back to caller for UI badges.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag import (
    LLM_PROVIDERS, LLM_PROVIDER_ORDER, _call_openai_compatible, _get_key,
    retrieve, synthesize_answer, SIMILARITY_THRESHOLD,
)

log = logging.getLogger("taxrag.router")

INTENTS = ["GREETING", "STATUTE_LOOKUP", "CALCULATION", "GENERAL_TAX", "OFF_TOPIC"]

ROUTER_SYSTEM = (
    "You are a strict intent classifier for an Indian income-tax assistant. "
    "Classify the user's most recent message into EXACTLY ONE of these labels:\n"
    "  GREETING       — hello / thanks / bye / small talk / questions about your capabilities\n"
    "  STATUTE_LOOKUP — needs a specific Income-Tax Act 2025 fact (section text, exact rate, procedure)\n"
    "  CALCULATION    — needs a numerical answer (tax on X income, deduction cap math, regime comparison)\n"
    "  GENERAL_TAX    — general Indian tax literacy (concepts, filing basics, common terminology)\n"
    "  OFF_TOPIC      — unrelated to Indian tax\n"
    "Reply with a JSON object: {\"intent\": LABEL, \"confidence\": 0.0-1.0, \"reason\": \"short\"}. "
    "No prose outside JSON."
)


def _call_llm(messages, provider: Optional[str] = None, model: Optional[str] = None,
              max_tokens: int = 500, temperature: float = 0.1) -> Tuple[Optional[str], Optional[str]]:
    """Same failover logic as synthesize_answer but caller-controlled temperature/tokens."""
    order = list(LLM_PROVIDER_ORDER)
    if provider and provider in order:
        order.remove(provider); order.insert(0, provider)
    for pname in order:
        cfg = LLM_PROVIDERS[pname]
        key = "" if cfg.get("anonymous") else _get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            continue
        models = list(cfg["models"])
        if pname == provider and model:  # honor a discovered (non-static) pick too
            if model in models:
                models.remove(model)
            models.insert(0, model)
        for m in models:
            try:
                import requests
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                if cfg.get("kilo_headers"):
                    headers.update({"HTTP-Referer": "https://kilo.ai/", "X-Title": "Kilo Code"})
                resp = requests.post(
                    f"{cfg['base'].rstrip('/')}/chat/completions",
                    headers=headers,
                    json={"model": m, "messages": messages, "temperature": temperature,
                          "max_tokens": max_tokens},
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    return content, f"{pname}/{m}"
            except Exception as e:
                log.debug("provider %s/%s failed: %s", pname, m, e)
                continue
    return None, None


_CLASSIFIER_CACHE: Dict[int, Any] = {}


def offline_intent(query: str, model=None) -> Dict[str, Any]:
    """Fallback when no LLM answers: embedding classifier if an embedder is available,
    else the keyword heuristic."""
    if model is not None:
        try:
            if id(model) not in _CLASSIFIER_CACHE:
                from intent_classifier import IntentClassifier
                _CLASSIFIER_CACHE[id(model)] = IntentClassifier(model)
            out = dict(_CLASSIFIER_CACHE[id(model)].predict(query))
            # This path only runs when no LLM answered, so the GENERAL_TAX handler would
            # just say "can't reach any LLM". Statute lookup works offline (extractive)
            # and still escalates to GENERAL_TAX itself when retrieval finds no match.
            if out.get("intent") == "GENERAL_TAX":
                out["intent"] = "STATUTE_LOOKUP"
                out["reason"] = out.get("reason", "") + "; general->statute (no LLM reachable)"
            return out
        except Exception as e:  # never let the fallback break routing
            log.warning("intent classifier unavailable: %s", e)
    return heuristic_intent(query)


def classify_intent(query: str, history: Optional[List[Dict]] = None, model=None) -> Dict[str, Any]:
    """Ask LLM to classify. Falls back to the offline classifier (or keywords) if all LLMs are down."""
    messages = [{"role": "system", "content": ROUTER_SYSTEM}]
    for turn in (history or [])[-2:]:
        messages.append({"role": turn["role"], "content": turn["content"][:200]})
    messages.append({"role": "user", "content": query})

    raw, provider = _call_llm(messages, max_tokens=120, temperature=0.0)
    if raw:
        # extract JSON — the model may wrap in prose
        m = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if obj.get("intent") in INTENTS:
                    obj["provider"] = provider
                    return obj
            except json.JSONDecodeError:
                pass

    return offline_intent(query, model)


def heuristic_intent(query: str) -> Dict[str, Any]:
    """Keyword fallback used when no LLM is reachable."""
    q = query.lower().strip()
    if len(q) < 40 and any(w in q for w in ["hello", "hi ", "hey", "thanks", "thank you",
                                            "bye", "who are you", "what can you", "help"]):
        return {"intent": "GREETING", "confidence": 0.6, "reason": "keyword heuristic",
                "provider": "heuristic"}
    if any(w in q for w in ["tax on", "how much tax", "calculate", "compute", "kitna tax"]):
        return {"intent": "CALCULATION", "confidence": 0.6, "reason": "keyword heuristic",
                "provider": "heuristic"}
    if re.search(r"\b(weather|cricket|movie|recipe|joke|football|dog|cat)\b", q):
        return {"intent": "OFF_TOPIC", "confidence": 0.7, "reason": "keyword heuristic",
                "provider": "heuristic"}
    return {"intent": "STATUTE_LOOKUP", "confidence": 0.5, "reason": "default",
            "provider": "heuristic"}


# ── HANDLERS ────────────────────────────────────────────────────────────────

GREETING_SYS = (
    "You are TaxRAG, a friendly Indian income-tax assistant. Reply warmly to the user. "
    "If asked about capabilities, mention: (1) look up sections of the Income-Tax Act 2025, "
    "(2) calculate tax under old and new regimes, (3) explain general tax concepts, "
    "(4) answer questions from uploaded PDFs. Keep replies under 4 sentences."
)

GENERAL_SYS = (
    "You are TaxRAG. Answer this general Indian income-tax question from your knowledge. "
    "Start with this exact line: \"⚠️ General knowledge — verify with a qualified CA before acting.\" "
    "Then give a concise, helpful answer. Do NOT invent specific section numbers or exact rupee figures "
    "you are not sure about."
)

CALC_SYS = (
    "You are TaxRAG, doing tax computation using ONLY the slab rates in the context. "
    "Show your calculation step by step. Cite the source section for each rate you use. "
    "If the context does not contain the applicable slab rates, say so and offer to explain "
    "general concepts instead."
)

REFUSE_MSG = (
    "That looks off-topic for an Indian income-tax assistant. I can help with the "
    "Income-Tax Act 2025, tax calculations, and general filing questions — try one of those!"
)


def handle_greeting(query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    messages = [{"role": "system", "content": GREETING_SYS}]
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})
    answer, provider = _call_llm(messages, max_tokens=300, temperature=0.5)
    if not answer:
        answer = "Hi! I'm TaxRAG. I can look up the Income-Tax Act 2025, calculate tax, and explain concepts. What would you like to ask?"
        provider = "static"
    return {"answer": answer, "trust_tier": "conversational", "intent": "GREETING",
            "citations": [], "provider": provider}


def handle_general_tax(query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    messages = [{"role": "system", "content": GENERAL_SYS}]
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})
    answer, provider = _call_llm(messages, max_tokens=500, temperature=0.3)
    if not answer:
        answer = "⚠️ I can't reach any LLM right now. Please try again in a moment, or ask a specific statute question so I can search the Income-Tax Act 2025 directly."
        provider = "static"
    return {"answer": answer, "trust_tier": "llm_memory", "intent": "GENERAL_TAX",
            "citations": [], "provider": provider}


def handle_statute_lookup(query, model, emb, chunks, history=None,
                          retriever=None, rerank=False, top_k=5,
                          threshold=None) -> Dict[str, Any]:
    thr = SIMILARITY_THRESHOLD if threshold is None else threshold
    if retriever is not None:
        # Hybrid RRF/rerank scores are not cosine similarity, so gate on hits
        # rather than the legacy 0.45 threshold.
        results, max_s = retriever.retrieve(query, top_k=top_k, rerank=rerank, candidates=15)
        confident = len(results) > 0
    else:
        results, max_s = retrieve(query, model, emb, chunks, top_k=top_k,
                                  threshold=thr)
        confident = bool(results) and max_s >= thr
    if not confident:
        # escalate: no confident statute match, fall back to LLM memory with banner
        out = handle_general_tax(query, history)
        out["intent"] = "STATUTE_LOOKUP→GENERAL_TAX"
        out["citations"] = []
        out["max_sim"] = max_s
        return out
    ans, provider = synthesize_answer(query, results, history=history)
    if not ans:
        ans = "**No LLM reachable — top matching provisions:**\n\n" + \
              "\n\n".join(f"- **{r['title']}** ({r['source']})" for r in results)
        provider = "extractive"
    citations = [{"title": r["title"], "source": r["source"], "score": float(r["score"])}
                 for r in results]
    return {"answer": ans, "trust_tier": "grounded", "intent": "STATUTE_LOOKUP",
            "citations": citations, "provider": provider, "max_sim": float(max_s)}


_RULES_CACHE: Dict[int, Any] = {}


def _tax_rules(chunks):
    key = id(chunks)
    if key not in _RULES_CACHE:
        from tax_calculator import load_rules
        try:
            _RULES_CACHE[key] = load_rules(chunks)
        except ValueError:
            _RULES_CACHE[key] = None
    return _RULES_CACHE[key]


def handle_calculation(query, model, emb, chunks, history=None,
                       retriever=None, rerank=False, top_k=5,
                       threshold=None) -> Dict[str, Any]:
    """Tool first: new-regime questions with a parseable income go to the deterministic
    calculator (section 202 slabs, section 156 rebate, section 19 standard deduction).
    Everything else (old regime, no amount, deduction maths) uses the LLM path."""
    from tax_calculator import compute_tax, format_answer, parse_calc_query
    rules = _tax_rules(chunks)
    parsed = parse_calc_query(query)
    if rules is not None and parsed["income"] and not parsed["old_regime"]:
        res = compute_tax(parsed["income"], rules, salaried=parsed["salaried"])
        citations = [{"title": s, "source": "Income-Tax Act 2025", "score": 1.0}
                     for s in rules.sources.values()]
        return {"answer": format_answer(res, rules, parsed["salaried"]), "trust_tier": "grounded",
                "intent": "CALCULATION", "citations": citations, "provider": "tax_calculator",
                "calculation": res}
    return handle_calculation_llm(query, model, emb, chunks, history, retriever, rerank,
                                  top_k, threshold)


def handle_calculation_llm(query, model, emb, chunks, history=None,
                           retriever=None, rerank=False, top_k=5,
                           threshold=None) -> Dict[str, Any]:
    # Calculations use a softer gate so retrieved slab rates still reach the LLM;
    # never stricter than 0.35 even if the user raised the abstention threshold.
    thr = 0.35 if threshold is None else min(threshold, 0.35)
    if retriever is not None:
        results, max_s = retriever.retrieve(query, top_k=top_k, rerank=rerank, candidates=15)
        confident = len(results) > 0
    else:
        results, max_s = retrieve(query, model, emb, chunks, top_k=top_k, threshold=thr)
        confident = bool(results) and max_s >= thr
    if confident:
        # compute mode: LLM synthesizes math using retrieved slab rates
        context = "\n\n---\n\n".join(f"[{r['title']}] (Source: {r['source']})\n{r['text']}"
                                     for r in results)
        messages = [{"role": "system", "content": CALC_SYS},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}]
        ans, provider = _call_llm(messages, max_tokens=700, temperature=0.1)
        if ans:
            citations = [{"title": r["title"], "source": r["source"], "score": float(r["score"])}
                         for r in results]
            return {"answer": ans, "trust_tier": "grounded", "intent": "CALCULATION",
                    "citations": citations, "provider": provider, "max_sim": float(max_s)}
    # else: LLM memory computation with disclaimer
    out = handle_general_tax(query, history)
    out["intent"] = "CALCULATION→GENERAL_TAX"
    out["max_sim"] = max_s
    return out


def handle_off_topic(query: str) -> Dict[str, Any]:
    return {"answer": REFUSE_MSG, "trust_tier": "refusal", "intent": "OFF_TOPIC",
            "citations": [], "provider": "static"}


# ── TOP-LEVEL ROUTER ────────────────────────────────────────────────────────

def chat(query: str, model, emb, chunks, history: Optional[List[Dict]] = None,
         retriever=None, rerank: bool = False, top_k: int = 5,
         threshold: Optional[float] = None) -> Dict[str, Any]:
    """Route + dispatch. Returns {answer, trust_tier, intent, citations, provider, max_sim}.

    When `retriever` (a HybridRetriever) is given, retrieval-backed handlers use
    it instead of the legacy `rag.retrieve`, with optional cross-encoder rerank.
    """
    intent_result = classify_intent(query, history, model=model)
    intent = intent_result["intent"]
    log.info("intent=%s conf=%.2f (%s) via %s",
             intent, intent_result.get("confidence", 0), intent_result.get("reason", ""),
             intent_result.get("provider", ""))

    if intent == "GREETING":
        out = handle_greeting(query, history)
    elif intent == "OFF_TOPIC":
        out = handle_off_topic(query)
    elif intent == "CALCULATION":
        out = handle_calculation(query, model, emb, chunks, history, retriever, rerank,
                                 top_k, threshold)
    elif intent == "GENERAL_TAX":
        out = handle_general_tax(query, history)
    else:  # STATUTE_LOOKUP
        out = handle_statute_lookup(query, model, emb, chunks, history, retriever, rerank,
                                    top_k, threshold)

    out["router_intent"] = intent
    out["router_confidence"] = intent_result.get("confidence", 0)
    return out


if __name__ == "__main__":
    import sys
    from rag import build_embedder, load_index
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print("Loading model + index...")
    model = build_embedder()
    emb, chunks = load_index("./out")
    print(f"loaded {len(chunks)} chunks\n")

    queries = [
        "Hello, what can you calculate?",
        "What can you do for me?",
        "What is Section 80C deduction?",
        "what is tax on 20 lakh under new regime for individual",
        "What is the difference between old and new tax regime?",
        "What is the weather in Delhi today?",
        "20 lakh par kitna tax lagega new regime mein",
    ]
    for q in queries:
        print("=" * 78)
        print(f"Q: {q}")
        r = chat(q, model, emb, chunks)
        print(f"→ intent={r['router_intent']}  trust={r['trust_tier']}  provider={r.get('provider')}")
        print(f"→ answer (first 400 chars):\n{r['answer'][:400]}")
        print()
