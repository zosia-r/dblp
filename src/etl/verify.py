"""
Phase 5 — Post-load verification.
"""

import logging
import sqlite3

from .config import DB_PATH, MIN_YEAR, MAX_YEAR

log = logging.getLogger(__name__)


def verify() -> None:
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
    
        for table in ("papers", "authors", "author_aliases", "paper_authors"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            log.info("  %-22s %d rows", table, count)
            assert count > 0, f"Table {table!r} is empty"
    
        cur.execute("SELECT MIN(year), MAX(year) FROM papers")
        min_year, max_year = cur.fetchone()
        log.info("  Year range: %s – %s", min_year, max_year)
        assert min_year >= MIN_YEAR, f"Found records before {MIN_YEAR}"
    
        cur.execute("""
            SELECT p.id, p.title, COUNT(pa.author_id) AS n
            FROM papers p JOIN paper_authors pa ON p.id = pa.paper_id
            GROUP BY p.id ORDER BY n DESC LIMIT 5
        """)
        log.info("  Top 5 papers by author count:")
        for pid, title, n in cur.fetchall():
            log.info("    [%d authors] %s", n, title)
    
        cur.execute("""
            SELECT a.primary_name, COUNT(pa.paper_id) AS n
            FROM authors a JOIN paper_authors pa ON a.id = pa.author_id
            GROUP BY a.id ORDER BY n DESC LIMIT 5
        """)
        log.info("  Top 5 authors by paper count:")
        for name, n in cur.fetchall():
            log.info("    [%d papers] %s", n, name)
    
        cur.execute("SELECT COUNT(*) FROM authors a JOIN author_aliases al ON a.id = al.author_id")
        log.info("  Authors with at least one alias: %d", cur.fetchone()[0])
        log.info("Verification passed.")
    
    finally:
        con.close()
 