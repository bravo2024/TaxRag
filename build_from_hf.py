#!/usr/bin/env python3
"""Build the TaxRAG index from the REAL Income-Tax Act, 2025 dataset
(ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text, MIT licence).

Each of the ~676 machine-readable sections becomes one indexed chunk with a
genuine citation. This replaces the earlier AI-written corpus with actual law.
Writes out/index.npz + out/metrics.json.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

import rag

DATASET = "ThanniruVenkata/Income-Tax-Act-2025-Machine-Readable-Legal-Text"
OUT = "./out"


def main() -> None:
    path = hf_hub_download(DATASET, "income_tax_act_2025.json", repo_type="dataset")
    records = json.load(open(path, encoding="utf-8"))
    rag.log.info("Loaded %d records from %s", len(records), DATASET)

    chunks = []
    for r in records:
        section = str(r.get("section", "")).strip()
        title = str(r.get("title", "")).strip()
        chapter = str(r.get("chapter", "")).strip()
        chapter_name = str(r.get("chapter_name", "")).strip()
        content = str(r.get("content", "")).strip()
        if len(content) < 20:
            continue
        ci = r.get("chunk_index", 0)
        text = f"Section {section}: {title}\n\n{content}"
        chunks.append({
            "text": text,
            "title": f"Section {section}: {title}" if title else f"Section {section}",
            "source": f"Income-Tax Act, 2025 — Chapter {chapter} ({chapter_name}), Section {section}",
            "chunk_id": f"ITA2025_ch{chapter}_sec{section}_{ci}",
        })

    rag.log.info("Built %d chunks from real Act text", len(chunks))
    model = rag.build_embedder()
    embeddings = rag.embed_chunks(chunks, model)
    rag.save_index(OUT, embeddings, chunks)

    metrics = {
        "corpus_source": "hf_dataset",
        "dataset": DATASET,
        "dataset_license": "MIT",
        "authoritative": True,
        "act": "Income-Tax Act, 2025 (effective 2026-04-01)",
        "num_chunks": len(chunks),
        "num_sections": len({c["chunk_id"].rsplit("_", 1)[0] for c in chunks}),
        "built": date.today().isoformat(),
        "model": rag.MODEL_NAME,
        "similarity_threshold": rag.SIMILARITY_THRESHOLD,
    }
    Path(OUT, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rag.log.info("Wrote %s/metrics.json", OUT)
    rag.log.info("Done — authoritative Act 2025 corpus indexed.")


if __name__ == "__main__":
    main()
