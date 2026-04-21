"""
Phase 3 — Author deduplication.

Reads authors_raw.csv (all mentions, with duplicates) and produces:
  - authors.csv       (deduplicated registry, one row per unique normalized_name)
  - paper_authors.csv (many-to-many join table with stable author_id)

Deduplication key: normalized_name.
raw_name policy: first occurrence wins.
"""

import csv
import logging
from pathlib import Path

from .config import AUTHORS_CSV, AUTHORS_RAW_CSV, PAPER_AUTHORS_CSV

log = logging.getLogger(__name__)

Registry = dict[str, tuple[int, str]]       # normalized_name -> (author_id, canonical_raw_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate_authors() -> int:
    """
    Two-pass deduplication over authors_raw.csv.

    Pass 1: build Registry (normalized_name -> (author_id, canonical_raw_name)).
    Pass 2: resolve every mention to its author_id and write paper_authors.csv.

    Returns the number of distinct authors found.
    """
    registry: Registry = {}
    rows: list[tuple[str, int, str]] = []   # (paper_id, author_order, normalized_name)
    next_id = 1

    # --- Pass 1: build registry ---
    with open(AUTHORS_RAW_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            norm = row["normalized_name"]
            raw  = row["raw_name"]
            if norm not in registry:
                registry[norm] = (next_id, raw)   # first-seen raw_name is canonical
                next_id += 1
            rows.append((row["paper_id"], int(row["author_order"]), norm))

    # --- Write authors.csv ---
    with open(AUTHORS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_name", "normalized_name"])
        # Write in ID order (first-seen order)
        for norm, (aid, raw) in sorted(registry.items(), key=lambda x: x[1][0]):
            w.writerow([aid, raw, norm])

    # --- Pass 2: write paper_authors.csv ---
    with open(PAPER_AUTHORS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "author_id", "author_order"])
        for paper_id, order, norm in rows:
            author_id = registry[norm][0]
            w.writerow([paper_id, author_id, order])

    distinct = len(registry)
    log.info(
        "Deduplication done | distinct authors: %d | total mentions: %d",
        distinct, len(rows),
    )
    return distinct
