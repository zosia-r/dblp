"""
Phase 3 — Author resolution.

Reads:
  - authors_raw.csv     (name mentions from papers)
  - authors.csv         (primary names from <www> homepages)
  - author_aliases.csv  (aliases from <www> homepages)

Produces:
  - paper_authors.csv   (paper_id, author_id, author_order)
"""

import csv
import logging

from .config import AUTHOR_ALIASES_CSV, AUTHORS_CSV, AUTHORS_RAW_CSV, PAPER_AUTHORS_CSV

log = logging.getLogger(__name__)


def resolve_authors() -> int:
    """
    Build paper_authors.csv by resolving each name mention to an author_id.
    Returns total number of distinct authors.
    """

    # --- Load identity registry: name -> author_id ---
    name_to_id: dict[str, int] = {}

    with open(AUTHORS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name_to_id[row["primary_name"]] = int(row["id"])

    with open(AUTHOR_ALIASES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alias     = row["alias"]
            author_id = int(row["author_id"])
            if alias not in name_to_id:
                name_to_id[alias] = author_id

    log.info("Identity registry loaded | %d names (primary + aliases)", len(name_to_id))

    # --- Resolve mentions -> paper_authors.csv ---
    seen: set[tuple[str, int]] = set()   # (paper_id, author_id) already written
    total        = 0
    duplicate_count    = 0
    unresolved = 0

    with (
        open(AUTHORS_RAW_CSV,   newline="", encoding="utf-8") as inf,
        open(PAPER_AUTHORS_CSV, "w",        newline="", encoding="utf-8") as outf,
    ):
        reader = csv.DictReader(inf)
        writer = csv.writer(outf)
        writer.writerow(["paper_id", "author_id", "author_order"])

        for row in reader:
            name     = row["name"]
            paper_id = row["paper_id"]
            order    = int(row["author_order"])

            if name not in name_to_id:
                log.warning("Unresolved author %r in paper %s — skipping", name, paper_id)
                unresolved += 1
                continue

            author_id = name_to_id[name]

            key = (paper_id, author_id)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)

            writer.writerow([paper_id, author_id, order])
            total += 1

    log.info(
        "Resolution done | rows written: %d | duplicates dropped: %d | unresolved: %d",
        total, duplicate_count, unresolved
    )
    return total