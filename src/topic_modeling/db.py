import sqlite3
import logging
from contextlib import contextmanager
import pandas as pd

logger = logging.getLogger(__name__)

TOPICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id   INTEGER PRIMARY KEY,
    label TEXT NOT NULL
);
"""

@contextmanager
def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def load_sample_papers(path) -> list[dict]:
    df = pd.read_parquet(path, columns=["id", "title"], engine="pyarrow")
    df = df[df["title"].notnull()]
    logger.info(f"Loaded {len(df):,} papers from sample.")
    return df['title'].tolist()

def load_papers(db_path: str) -> list[dict]:
    """Load all papers with non-null titles from the database."""
    query = "SELECT id, title FROM papers WHERE title IS NOT NULL"
    with get_connection(db_path) as conn:
        rows = conn.execute(query).fetchall()
    logger.info(f"Loaded {len(rows):,} papers from database.")
    return [dict(row) for row in rows]


def setup_topics_schema(db_path: str) -> None:
    """Create topics table and add topic_id column to papers if not exists."""
    with get_connection(db_path) as conn:
        conn.execute(TOPICS_SCHEMA)

        # Add topic_id column to papers if missing
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if "topic_id" not in existing:
            conn.execute("ALTER TABLE papers ADD COLUMN topic_id INTEGER REFERENCES topics(id)")
            logger.info("Added topic_id column to papers table.")

    logger.info("Database schema ready.")


def save_topics(db_path: str, topic_labels: dict[int, str]) -> None:
    """Insert or replace topic labels (skips topic -1 / outliers)."""
    rows = [
        (topic_id, label)
        for topic_id, label in topic_labels.items()
        if topic_id != -1
    ]
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM topics")
        conn.executemany("INSERT INTO topics (id, label) VALUES (?, ?)", rows)
    logger.info(f"Saved {len(rows)} topics to database.")


def save_paper_topics(db_path: str, paper_ids: list[str], topic_ids: list[int]) -> None:
    """Update topic_id for each paper. Sets NULL for outliers (topic_id == -1)."""
    rows = [
        (int(tid) if tid != -1 else None, pid)
        for pid, tid in zip(paper_ids, topic_ids)
    ]
    with get_connection(db_path) as conn:
        conn.executemany(
            "UPDATE papers SET topic_id = ? WHERE id = ?",
            rows,
        )
    assigned = sum(1 for _, tid in rows if tid is not None)
    logger.info(f"Updated topic_id for {assigned:,} papers ({len(rows) - assigned:,} outliers set to NULL).")