"""
Phase 5 — Post-load verification.

Runs a handful of test queries against the SQLite database and logs
the results. Raises AssertionError if any critical invariant is violated.
"""

import logging
import sqlite3

from .config import DB_PATH, MIN_YEAR

log = logging.getLogger(__name__)


def verify() -> None:
    """Print row counts and spot-check key invariants."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    try:
        # --- Row counts ---
        for table in ("papers", "authors", "paper_authors"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            log.info("  %-20s %d rows", table, count)
            assert count > 0, f"Table {table!r} is empty — something went wrong"

        # --- Year range ---
        cur.execute("SELECT MIN(year), MAX(year) FROM papers")
        min_year, max_year = cur.fetchone()
        log.info("  Year range: %s – %s", min_year, max_year)
        assert min_year >= MIN_YEAR, f"Found records before {MIN_YEAR}: min year = {min_year}"

        # --- No orphan paper_authors ---
        cur.execute("""
            SELECT COUNT(*) FROM paper_authors pa
            WHERE NOT EXISTS (SELECT 1 FROM papers p WHERE p.id = pa.paper_id)
        """)
        orphans = cur.fetchone()[0]
        assert orphans == 0, f"Found {orphans} orphan rows in paper_authors (no matching paper)"

        # --- No authors without papers ---
        cur.execute("""
            SELECT COUNT(*) FROM authors a
            WHERE NOT EXISTS (SELECT 1 FROM paper_authors pa WHERE pa.author_id = a.id)
        """)
        useless_authors = cur.fetchone()[0]
        log.info("  Authors without any papers: %d", useless_authors)
        assert useless_authors == 0, f"Found {useless_authors} authors with no associated papers"

        # --- Top authors ---
        cur.execute("""
            SELECT a.raw_name, COUNT(pa.paper_id) AS n
            FROM authors a
            JOIN paper_authors pa ON a.id = pa.author_id
            GROUP BY a.id
            ORDER BY n DESC
            LIMIT 5
        """)
        log.info("  Top 5 authors by paper count:")
        for name, n in cur.fetchall():
            log.info("    [%d papers] %s", n, name)

        log.info("Verification passed.")

    finally:
        con.close()
