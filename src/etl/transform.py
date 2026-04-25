"""
Phase 2 — Transformation & raw CSV writing.
 
Consumes Record dicts from the parser and writes:
  - papers.csv          (one row per paper)
  - authors_raw.csv     (one row per author mention in a paper)
  - authors.csv         (author identity registry: primary_name from <www>)
  - author_aliases.csv  (aliases per author from <www>)
 
Identity resolution uses DBLP's own <www key="homepages/..."> data:
  - first <author>      = primary (canonical) name
  - subsequent <author> = known aliases
"""

import csv
import logging
from pathlib import Path

from .config import (
    AUTHOR_ALIASES_CSV,
    AUTHORS_CSV,
    AUTHORS_RAW_CSV,
    PAPERS_CSV
)
from .parser import Record

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_raw_csvs(records) -> tuple[int, int, int]:
    """
    Stream all records into CSV files.
    Returns (paper_count, author_mention_count, author_identity_count).
    """
    paper_count = 0
    mention_count = 0
    identity_count = 0

    with (
        open(PAPERS_CSV,         "w", newline="", encoding="utf-8") as pf,
        open(AUTHORS_RAW_CSV,    "w", newline="", encoding="utf-8") as arf,
        open(AUTHORS_CSV,        "w", newline="", encoding="utf-8") as af,
        open(AUTHOR_ALIASES_CSV, "w", newline="", encoding="utf-8") as alif,
    ):
        pw   = csv.writer(pf)
        arw   = csv.writer(arf)
        aw  = csv.writer(af)
        aliw = csv.writer(alif)
 
        pw.writerow(["id", "title", "year", "venue", "type"])
        arw.writerow(["paper_id", "author_order", "name"])
        aw.writerow(["id", "primary_name"])
        aliw.writerow(["author_id", "alias"])
 
        # Author identity registry: primary_name -> int id
        identity_registry: dict[str, int] = {}
        next_author_id = 1

        for record in records:
            # --- Author identity record ---
            if record["type"] == "author":
                primary = record["primary_name"]
                if not primary or primary in identity_registry:
                    continue
 
                author_id = next_author_id
                identity_registry[primary] = author_id
                next_author_id += 1
 
                aw.writerow([author_id, primary])
                identity_count += 1
 
                for alias in record["aliases"]:
                    if alias and alias != primary:
                        aliw.writerow([author_id, alias])
 
                continue

            # --- Paper record ---
            pw.writerow([
                record["key"],
                record["title"],
                record["year"],
                record["venue"],
                record["record_type"],
            ])
            paper_count += 1
 
            for order, name in enumerate(record["authors"], start=1):
                if not name:
                    continue
                arw.writerow([record["key"], order, name])
                mention_count += 1

    log.info(
        "Transform done | papers: %d | author mentions: %d | author identities: %d",
        paper_count, mention_count, identity_count,
    )

    return paper_count, mention_count, identity_count