#!/usr/bin/env python3
"""Build the TaxRAG search index from the AUTHORITATIVE amended-Act PDF, with
clean per-section citations obtained by anchoring on a structured dataset.

The section titles, chapter numbers and chapter names come from the
ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text dataset (MIT),
but every chunk's CONTENT is sliced verbatim from the actual PDF text of
data/pdfs/Income_Tax_Act_2025_as_amended_by_FA_Act_2026.pdf. Sections that
cannot be located in the PDF are skipped and logged — never fabricated.

Usage: python build_from_pdf_sections.py
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download
from pypdf import PdfReader

import rag

DATASET = "ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text"
LABELS_FILE = "income_tax_act_2025.json"
PRIMARY_PDF = "data/pdfs/Income_Tax_Act_2025_as_amended_by_FA_Act_2026.pdf"
BILL_PDF = "data/pdfs/Taxations_Bill_2026.pdf"
OUT = "./out"

SEARCH_WINDOW = 60000      # chars to look ahead for the next section number
SECTION_CHUNK_LIMIT = 1800  # chars; longer section bodies get sub-chunked
TOO_SHORT = 40             # sections whose located content is this short are skipped
BILL_CHUNK_LIMIT = 1500
BILL_MIN_CHARS = 500
MIN_SECTIONS = 300         # guard: fewer located => parsing failed, abort

BILL_TITLE = "Taxation Laws (Amendment) Bill, 2026"
BILL_SOURCE = "Taxation Laws (Amendment) Bill, 2026 (as introduced in Lok Sabha)"


# ---------------------------------------------------------------------------
# Labels (anchors) — collapse the dataset to one unique record per section
# ---------------------------------------------------------------------------

def load_labels() -> list:
    path = hf_hub_download(DATASET, LABELS_FILE, repo_type="dataset")
    records = json.load(open(path, encoding="utf-8"))
    rag.log.info("Loaded %d records from %s", len(records), DATASET)

    unique: dict = {}
    for rec in records:
        try:
            n = int(str(rec.get("section", "")).strip())
        except (TypeError, ValueError):
            continue
        if n not in unique:
            unique[n] = rec

    labels = sorted(
        ({**unique[n], "section": n} for n in unique),
        key=lambda r: r["section"],
    )
    rag.log.info("Collapsed to %d unique sections (range %s..%s)",
                 len(labels), labels[0]["section"], labels[-1]["section"])
    return labels


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = len(reader.pages)
    full = "\n".join((page.extract_text() or "") for page in reader.pages)
    rag.log.info("PDF %s: %d pages, %d chars", Path(path).name, pages, len(full))
    return full


# ---------------------------------------------------------------------------
# Section detection — monotonic forward scan on the NUMBER pattern
# ---------------------------------------------------------------------------

def _body_start():
    return r"(?:\(1\)|[A-Z“*])"


def _number_pattern(n: int, glued: bool = False):
    # Section number at line start, followed by '.' then the body start.
    # Footnote markers may sit between the number and the body, either spaced
    # ("24 ["), glued to the body ("23["), or glued to the number ("5207").
    if glued:
        return r"\n\s*\d\d?%d\s*\.\s*(?:\d*\s*\[\s*)?%s" % (n, _body_start())
    return r"\n\s*%d\s*\.\s*(?:\d*\s*\[\s*)?%s" % (n, _body_start())


def locate_sections(full: str, labels: list) -> dict:
    starts: dict = {}
    missed: list = []
    pos = 0
    for lab in labels:
        n = lab["section"]
        window = full[pos:pos + SEARCH_WINDOW]
        m = re.search(_number_pattern(n), window)
        if m is None:
            m = re.search(_number_pattern(n, glued=True), window)
        if m is not None:
            starts[n] = pos + m.start()
            pos = pos + m.start() + 1
        else:
            missed.append(n)
    return starts, missed


# ---------------------------------------------------------------------------
# Content cleaning — never invent text, only collapse whitespace / drop trivia
# ---------------------------------------------------------------------------

ACT_HEADER = "INCOME-TAX ACT, 2025"


def clean_content(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if re.fullmatch(r"\d{1,4}", s):          # page-number footer
            continue
        if s.upper() == ACT_HEADER:              # running page header
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse 3+ blank lines to 2
    return text.strip()


def split_body(body: str, limit: int) -> list:
    """Split a long body into ~limit-char chunks on paragraph/sentence breaks."""
    if len(body) <= limit:
        return [body]

    units: list = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            units.append(para)
            continue
        pieces = [p.strip() for p in para.split("\n") if p.strip()]
        for piece in pieces:
            if len(piece) <= limit:
                units.append(piece)
            else:
                units.extend(_split_sentences(piece, limit))

    chunks: list = []
    cur = ""
    for unit in units:
        if cur and len(cur) + len(unit) + 2 > limit:
            chunks.append(cur)
            cur = unit
        else:
            cur = unit if not cur else cur + "\n\n" + unit
    if cur:
        chunks.append(cur)
    return chunks


def _split_sentences(text: str, limit: int) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > limit:
            out.append(buf)
            buf = sent
        else:
            buf = sent if not buf else buf + " " + sent
    if buf:
        out.append(buf)
    return out


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

def build_act_chunks(full: str, labels: list) -> tuple:
    starts, missed = locate_sections(full, labels)
    located_count = len(starts)
    rag.log.info("Located %d / %d sections in PDF; missed: %s",
                 located_count, len(labels), missed)

    labels_idx = {lab["section"]: lab for lab in labels}
    ordered = sorted(starts)
    chunks: list = []
    deleted: list = []
    too_short: list = []
    for idx, n in enumerate(ordered):
        s = starts[n]
        e = starts[ordered[idx + 1]] if idx + 1 < len(ordered) else len(full)
        content = clean_content(full[s:e])

        first_line = next(
            (l for l in content.strip("\n").split("\n") if l.strip()), ""
        )
        if "[***]" in first_line:
            deleted.append(n)
            rag.log.info("Section %d: text deleted by amendment ([***] marker) — skipped", n)
            continue
        if len(content) < TOO_SHORT:
            too_short.append(n)
            rag.log.info("Section %d: located content too short (%d chars) — skipped",
                         n, len(content))
            continue

        lab = labels_idx[n]
        title = f"Section {n}: {lab['title']}"
        source = ("Income-Tax Act, 2025 (as amended by Finance Act 2026), "
                  "Chapter %s (%s), Section %d"
                  % (lab["chapter"], lab["chapter_name"], n))
        for k, sub in enumerate(split_body(content, SECTION_CHUNK_LIMIT)):
            chunks.append({
                "text": f"{title}\n\n{sub}",
                "title": title,
                "source": source,
                "chunk_id": f"ITA2025FA26_sec{n}_{k}",
            })

    rag.log.info("Built %d act chunks from %d sections (deleted: %s, too-short: %s)",
                 len(chunks), len(ordered) - len(deleted), deleted, too_short)
    return chunks, starts, missed, deleted, too_short


def build_bill_chunks() -> list:
    full = extract_pdf_text(BILL_PDF)
    if len(full) <= BILL_MIN_CHARS:
        rag.log.info("Bill text %d chars <= %d — skipped", len(full), BILL_MIN_CHARS)
        return []

    cleaned = clean_content(full)
    parts = split_body(cleaned, BILL_CHUNK_LIMIT)
    chunks = [{
        "text": f"{BILL_TITLE}\n\n{part}",
        "title": BILL_TITLE,
        "source": BILL_SOURCE,
        "chunk_id": f"BILL2026_{k}",
    } for k, part in enumerate(parts)]
    rag.log.info("Built %d bill chunks (%d chars)", len(chunks), len(cleaned))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    labels = load_labels()

    full = extract_pdf_text(PRIMARY_PDF)
    act_chunks, starts, missed, deleted, too_short = build_act_chunks(full, labels)
    located = len(starts)

    if located < MIN_SECTIONS:
        rag.log.error(
            "PARSE FAILED: only %d / %d sections located in %s (need >= %d). "
            "Aborting WITHOUT touching the existing index.",
            located, len(labels), PRIMARY_PDF, MIN_SECTIONS,
        )
        sys.exit(1)

    bill_chunks = build_bill_chunks()
    chunks = act_chunks + bill_chunks
    rag.log.info("Total chunks: %d (%d act + %d bill)", len(chunks),
                 len(act_chunks), len(bill_chunks))

    model = rag.build_embedder()
    embeddings = rag.embed_chunks(chunks, model)
    rag.save_index(OUT, embeddings, chunks)

    section_chunked = len({c["chunk_id"].rsplit("_", 1)[0] for c in act_chunks})
    metrics = {
        "corpus_source": "pdf_section_parsed",
        "act": "Income-Tax Act, 2025 as amended by Finance Act 2026",
        "primary_pdf": Path(PRIMARY_PDF).name,
        "bill_pdf": Path(BILL_PDF).name,
        "labels_from": f"{DATASET} (MIT)",
        "total_sections_in_labels": len(labels),
        "sections_located": section_chunked,
        "sections_detected_in_pdf": located,
        "sections_missed": len(missed),
        "sections_deleted_by_amendment": len(deleted),
        "num_chunks": len(chunks),
        "model": rag.MODEL_NAME,
        "similarity_threshold": rag.SIMILARITY_THRESHOLD,
        "built": date.today().isoformat(),
    }
    Path(OUT, "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    rag.log.info("Wrote %s/metrics.json", OUT)

    print(f"located {section_chunked} / {len(labels)} sections, "
          f"missed {len(missed)}, total chunks {len(chunks)}")
    if deleted:
        print(f"(also skipped {len(deleted)} sections deleted by amendment: {deleted})")
    if too_short:
        print(f"(also skipped {len(too_short)} too-short sections: {too_short})")
    if missed:
        print("first 20 missing section numbers:", missed[:20])
    rag.log.info("Done — authoritative PDF section-parsed index built.")


if __name__ == "__main__":
    main()
