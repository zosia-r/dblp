"""
Phase 4 — SQLite loader.

Bulk-loads the three CSV files into a normalized SQLite database.
Load order respects foreign key dependencies:
    papers -> authors -> paper_authors

Design decisions:
  - INSERT OR IGNORE: makes reruns idempotent
  - Indexes created AFTER bulk load for faster inserts
  - PRAGMA WAL: faster writes, allows concurrent reads
  - PRAGMA foreign_keys: enforce referential integrity
  - PRAGMA synchronous=NORMAL: faster writes
  - executemany in batches of BATCH_SIZE for memory efficiency
"""

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Generator, Iterable

from .config import (
    AUTHORS_CSV,
    BATCH_SIZE,
    DB_PATH,
    PAPER_AUTHORS_CSV,
    PAPERS_CSV,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_SQL_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers
(
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    venue TEXT,
    type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    raw_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL REFERENCES papers(id),
    author_id INTEGER NOT NULL REFERENCES authors(id),
    author_order INTEGER,
    UNIQUE (paper_id, author_id)
);
"""

_SQL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_type ON papers(type);
CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue);
CREATE INDEX IF NOT EXISTS idx_pa_paper ON paper_authors(paper_id);
CREATE INDEX IF NOT EXISTS idx_pa_author ON paper_authors(author_id);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _csv_rows(path: Path) -> Generator[list[str], None, None]:
    """Yield rows from a CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)    # skip header
        yield from reader


def _batch_insert(con: sqlite3.Connection, sql: str, rows: Iterable, label: str) -> int:
    """
    Insert rows in batches inside a single transaction.
    Returns total number of rows processed.
    """
    total = 0
    buf: list = []
    with con:
        for row in rows:
            buf.append(row)
            if len(buf) >= BATCH_SIZE:
                con.executemany(sql, buf)
                total += len(buf)
                buf.clear()
                log.info("  %s: %d rows so far...", label, total)
        if buf:
            con.executemany(sql, buf)
            total += len(buf)

    return total


# ---------------------------------------------------------------------------
# Row iterators
# ---------------------------------------------------------------------------

def _papers_rows():
    for paper_id, title, year, venue, rec_type in _csv_rows(PAPERS_CSV):
        yield (paper_id, title, int(year), venue or None, rec_type)


def _authors_rows():
    for aid, raw, norm in _csv_rows(AUTHORS_CSV):
        yield (int(aid), raw, norm)


def _paper_authors_rows():
    for paper_id, author_id, order in _csv_rows(PAPER_AUTHORS_CSV):
        yield (paper_id, int(author_id), int(order))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_into_sqlite() -> None:
    """Create schema, bulk-load all tables, then build indexes."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    con.executescript(_SQL_SCHEMA)

    n = _batch_insert(con, "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?)", _papers_rows(), "papers")
    log.info("Loaded papers:        %d rows", n)

    n = _batch_insert(con, "INSERT OR IGNORE INTO authors VALUES (?,?,?)", _authors_rows(), "authors")
    log.info("Loaded authors:       %d rows", n)

    n = _batch_insert(con, "INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)", _paper_authors_rows(), "paper_authors")
    log.info("Loaded paper_authors: %d rows", n)

    con.executescript(_SQL_INDEXES)
    log.info("Indexes created")

    con.close()
