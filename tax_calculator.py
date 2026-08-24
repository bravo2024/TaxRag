"""Deterministic new-regime income-tax calculator grounded in the indexed statute.

The slab table (section 202(1)), the rebate (section 156(2)) and the salaried
standard deduction (section 19, Table Sl. No. 2) are parsed from the corpus
text itself, so the numbers always match the Act the app retrieves from.

Out of scope (not in the Income-Tax Act 2025 text): old-regime slab rates,
surcharge and the 4% health & education cess, which come from each year's
Finance Act.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LAKH, CRORE = 100_000, 10_000_000


@dataclass
class Rules:
    slabs: List[tuple]                 # (lower_exclusive, upper_inclusive or None, rate)
    std_deduction_new: Optional[int]   # section 19 table, new regime
    rebate_cap: int                    # section 156(2)(a)
    rebate_limit: int                  # income limit for the full rebate
    sources: Dict[str, str] = field(default_factory=dict)


def _section_text(chunks, number: int) -> str:
    prefix = f"Section {number}:"
    return "\n".join(c["text"] for c in chunks if c["title"].startswith(prefix))


def _amount(s: str) -> int:
    return int(s.replace(",", ""))


def load_rules(chunks) -> Rules:
    """Parse the tax rules from the corpus chunks (each chunk: dict with title/text)."""
    s202 = _section_text(chunks, 202)
    rows = re.findall(r"(Upto|From|Above)\s*`?\s*([\d,]+)(?:\s*to\s*`?\s*([\d,]+))?\s+(Nil|\d+)\s*%?", s202)
    slabs = []
    for kind, a, b, rate in rows:
        r = 0.0 if rate == "Nil" else int(rate) / 100
        if kind == "Upto":
            slabs.append((0, _amount(a), r))
        elif kind == "From":
            slabs.append((_amount(a) - 1, _amount(b), r))
        else:
            slabs.append((_amount(a), None, r))
    if not slabs or slabs[-1][1] is not None:
        raise ValueError("Could not parse the section 202(1) slab table from the corpus")

    s19 = _section_text(chunks, 19)
    m = re.search(r"Standard deduction\.\s*\(a\)\s*`?\s*([\d,]+)", s19)
    std_new = _amount(m.group(1)) if m else None

    s156 = _section_text(chunks, 156)
    m = re.search(r"twelve lakh rupees,\s*100% of the income-tax\s*payable or\s*`?\s*([\d,]+)", s156)
    rebate_cap = _amount(m.group(1)) if m else 60000
    return Rules(slabs=slabs, std_deduction_new=std_new, rebate_cap=rebate_cap, rebate_limit=12 * LAKH,
                 sources={"slabs": "Section 202(1)", "rebate": "Section 156(2)",
                          "standard_deduction": "Section 19 (Table, Sl. No. 2)"})


def compute_tax(income: float, rules: Rules, salaried: bool = False) -> Dict:
    """New-regime tax before surcharge and cess, with a step-by-step breakdown."""
    gross = max(float(income), 0.0)
    std = min(rules.std_deduction_new or 0, gross) if salaried else 0
    taxable = gross - std

    steps, tax = [], 0.0
    for lower, upper, rate in rules.slabs:
        if taxable <= lower:
            break
        top = taxable if upper is None else min(taxable, upper)
        part = (top - lower) * rate
        steps.append({"from": lower, "to": top, "rate": rate, "tax": part})
        tax += part

    # Section 156(2): full rebate up to the limit; marginal relief just above it
    if taxable <= rules.rebate_limit:
        rebate = min(tax, rules.rebate_cap)
    else:
        excess = taxable - rules.rebate_limit
        rebate = tax - excess if tax > excess else 0.0
    payable = tax - rebate
    return {"gross_income": gross, "standard_deduction": std, "taxable_income": taxable,
            "slab_steps": steps, "tax_before_rebate": round(tax, 2), "rebate": round(rebate, 2),
            "tax_payable": round(payable, 2)}


_NUM = r"(\d+(?:,\d+)*(?:\.\d+)?)"


def parse_income(query: str) -> Optional[float]:
    """Pull an income amount from free text: 12 lakh, 1.2 crore, ₹15,00,000, 900k, 12L."""
    q = query.lower().replace("₹", " ").replace("rs.", " ").replace("rs ", " ").replace("inr", " ")
    m = re.search(_NUM + r"\s*(lakhs?|lacs?|lac|l\b|crores?|cr\b|k\b|thousand)", q)
    if m:
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2)
        mult = CRORE if unit.startswith(("crore", "cr")) else 1000 if unit in ("k", "thousand") else LAKH
        return val * mult
    nums = [float(n.replace(",", "")) for n in re.findall(r"\d{1,3}(?:,\d{2,3})+|\d{5,}", q)]
    return max(nums) if nums else None


def parse_calc_query(query: str) -> Dict:
    q = query.lower()
    return {
        "income": parse_income(query),
        "salaried": bool(re.search(r"salar|ctc|employ|naukri|in-hand|payslip|standard deduction", q)),
        "old_regime": bool(re.search(r"old (tax )?regime|purani|old slab", q)),
    }


def _inr(x: float) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return f"₹{s}"
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
    return f"₹{head},{tail}"


def format_answer(res: Dict, rules: Rules, salaried: bool) -> str:
    lines = [f"**New tax regime — tax payable: {_inr(res['tax_payable'])}** "
             "(before surcharge and 4% health & education cess)", ""]
    lines.append(f"- Gross income: {_inr(res['gross_income'])}")
    if salaried:
        lines.append(f"- Standard deduction ({rules.sources['standard_deduction']}): −{_inr(res['standard_deduction'])}")
    lines.append(f"- Taxable income: {_inr(res['taxable_income'])}")
    lines.append(f"- Slab computation ({rules.sources['slabs']}):")
    for s in res["slab_steps"]:
        lines.append(f"  - {_inr(s['from'])} – {_inr(s['to'])} @ {s['rate']:.0%} = {_inr(s['tax'])}")
    lines.append(f"- Tax before rebate: {_inr(res['tax_before_rebate'])}")
    lines.append(f"- Rebate ({rules.sources['rebate']}): −{_inr(res['rebate'])}")
    lines.append(f"- **Tax payable: {_inr(res['tax_payable'])}**")
    lines.append("")
    lines.append("_Computed deterministically from the Act's text. Surcharge and cess come from the "
                 "annual Finance Act and are not included; other deductions are not applied._")
    return "\n".join(lines)
