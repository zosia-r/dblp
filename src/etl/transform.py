"""
Phase 2 — Transformation & raw CSV writing.

Consumes Record dicts from the parser and writes:
  - papers.csv        (one row per paper)
  - authors_raw.csv   (one row per author mention, with duplicates)

Author deduplication happens in a later phase (authors.py).
"""

import csv
import logging
import re
from pathlib import Path

from .config import AUTHORS_RAW_CSV, PAPERS_CSV
from .parser import Record

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_name(raw: str) -> str:
    """
    Deterministic author name normalization:
      1. Strip leading/trailing whitespace
      2. Collapse internal whitespace to a single space
      3. Lowercase
      4. Remove dots
    """
    s = raw.strip()
    s = re.sub(r"\s+", " ", s)       # collapse internal whitespace
    s = s.lower()
    s = re.sub(r"\.", "", s)         # remove dots
    return s.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_raw_csvs(records) -> tuple[int, int]:
    """
    Stream records into papers.csv and authors_raw.csv.

    Returns (paper_count, author_mention_count).
    """
    paper_count = 0
    mention_count = 0

    with (
        open(PAPERS_CSV, "w", newline="", encoding="utf-8") as pf,
        open(AUTHORS_RAW_CSV, "w", newline="", encoding="utf-8") as af,
    ):
        pw = csv.writer(pf)
        aw = csv.writer(af)

        pw.writerow(["id", "title", "year", "venue", "type"])
        aw.writerow(["paper_id", "author_order", "raw_name", "normalized_name"])

        for record in records:
            pw.writerow([
                record["key"],
                record["title"],
                record["year"],
                record["venue"],
                record["type"],
            ])
            paper_count += 1

            for order, raw in enumerate(record["authors"], start=1):
                norm = _normalize_name(raw)
                if not norm:
                    log.warning(
                        "Empty normalized name for raw=%r in paper %s — skipping",
                        raw, record["key"],
                    )
                    continue
                aw.writerow([record["key"], order, raw, norm])
                mention_count += 1

    log.info(
        "Transform done | papers: %d | author mentions: %d",
        paper_count, mention_count,
    )
    return paper_count, mention_count
