"""
Phase 4 — SQLite loader.

Loads CSV files into SQLite in this order (respects FK dependencies):
    papers -> authors -> author_aliases -> paper_authors

Schema:
  papers          (id, title, year, venue, type)
  authors         (id INTEGER PK, primary_name TEXT UNIQUE)
  author_aliases  (author_id FK, alias TEXT)
  paper_authors   (paper_id FK, author_id FK, author_order)
"""

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Generator

from .config import (
    AUTHOR_ALIASES_CSV,
    AUTHORS_CSV,
    BATCH_SIZE,
    DB_PATH,
    PAPER_AUTHORS_CSV,
    PAPERS_CSV,
)

log = logging.getLogger(__name__)


_SQL_SCHEMA = """
CREATE TABLE papers (
    id    TEXT    PRIMARY KEY,
    title TEXT    NOT NULL,
    year  INTEGER NOT NULL,
    venue TEXT,
    type  TEXT    NOT NULL
);

CREATE TABLE authors (
    id           INTEGER PRIMARY KEY,
    primary_name TEXT    NOT NULL
);

CREATE TABLE author_aliases (
    author_id  INTEGER NOT NULL REFERENCES authors(id),
    alias      TEXT    NOT NULL
);

CREATE TABLE paper_authors (
    paper_id     TEXT    NOT NULL,
    author_id    INTEGER NOT NULL,
    author_order INTEGER
);
"""

_SQL_BULK_MODE = """
PRAGMA journal_mode = MEMORY;
PRAGMA synchronous = OFF;
PRAGMA foreign_keys = OFF;
PRAGMA cache_size = -131072;
"""

_SQL_POST_LOAD = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_authors_name ON authors(primary_name);
CREATE        INDEX IF NOT EXISTS idx_aliases_aid  ON author_aliases(author_id);
CREATE        INDEX IF NOT EXISTS idx_aliases_name ON author_aliases(alias);
CREATE UNIQUE INDEX IF NOT EXISTS uq_pa            ON paper_authors(paper_id, author_id);
CREATE        INDEX IF NOT EXISTS idx_papers_year  ON papers(year);
CREATE        INDEX IF NOT EXISTS idx_papers_type  ON papers(type);
CREATE        INDEX IF NOT EXISTS idx_papers_venue ON papers(venue);
CREATE        INDEX IF NOT EXISTS idx_pa_paper     ON paper_authors(paper_id);
CREATE        INDEX IF NOT EXISTS idx_pa_author    ON paper_authors(author_id);
"""


def _csv_rows(path: Path) -> Generator[list[str], None, None]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        yield from reader


def _batch_insert(con: sqlite3.Connection, sql: str, rows: Iterable, label: str) -> int:
    total = 0
    buf: list = []
    with con:
        for row in rows:
            buf.append(row)
            if len(buf) >= BATCH_SIZE:
                con.executemany(sql, buf)
                total += len(buf)
                buf.clear()
                if total % 1_000_000 == 0:
                    log.info("  %s: %d rows so far...", label, total)
        if buf:
            con.executemany(sql, buf)
            total += len(buf)
    return total


def _papers_rows():
    for paper_id, title, year, venue, rec_type in _csv_rows(PAPERS_CSV):
        yield (paper_id, title, int(year), venue or None, rec_type)

def _authors_rows():
    for author_id, primary_name in _csv_rows(AUTHORS_CSV):
        yield (int(author_id), primary_name)

def _aliases_rows():
    for author_id, alias in _csv_rows(AUTHOR_ALIASES_CSV):
        yield (int(author_id), alias)

def _paper_authors_rows():
    for paper_id, author_id, order in _csv_rows(PAPER_AUTHORS_CSV):
        yield (paper_id, int(author_id), int(order))


def load_into_sqlite() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        log.info("Removed existing database: %s", DB_PATH)

    con = sqlite3.connect(DB_PATH)
    con.executescript(_SQL_BULK_MODE)
    con.executescript(_SQL_SCHEMA)

    n = _batch_insert(con, "INSERT INTO papers VALUES (?,?,?,?,?)", _papers_rows(), "papers")
    log.info("Loaded papers:         %d rows", n)

    n = _batch_insert(con, "INSERT INTO authors VALUES (?,?)", _authors_rows(), "authors")
    log.info("Loaded authors:        %d rows", n)

    n = _batch_insert(con, "INSERT INTO author_aliases VALUES (?,?)", _aliases_rows(), "author_aliases")
    log.info("Loaded author_aliases: %d rows", n)

    n = _batch_insert(con, "INSERT INTO paper_authors VALUES (?,?,?)", _paper_authors_rows(), "paper_authors")
    log.info("Loaded paper_authors:  %d rows", n)

    log.info("Building indexes and constraints...")
    con.executescript(_SQL_POST_LOAD)
    log.info("Done.")
    con.close()