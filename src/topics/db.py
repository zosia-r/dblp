import json
import sqlite3


TOPICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id    INTEGER PRIMARY KEY,
    name  TEXT    NOT NULL,
    keywords TEXT NOT NULL  -- JSON array of top-10 words
);
"""

ADD_TOPIC_ID_COLUMN = """
ALTER TABLE papers ADD COLUMN topic_id INTEGER REFERENCES topics(id);
"""

ADD_TOPIC_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_papers_topic_id ON papers(topic_id);
"""


def apply_schema(conn: sqlite3.Connection) -> None:
    """
    Adds the topics table and topic_id column to papers (idempotent).
    """
    conn.execute(TOPICS_SCHEMA)

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    if "topic_id" not in existing_cols:
        conn.execute(ADD_TOPIC_ID_COLUMN)

    conn.execute(ADD_TOPIC_ID_INDEX)
    conn.commit()


def insert_topics(
    conn: sqlite3.Connection,
    names: dict[int, str],
    keywords: dict[int, list[str]],
) -> None:
    """
    Clears and re-inserts rows into the topics table.
    topic.id == cluster index so it can be used directly as topic_id in papers.
    """
    conn.execute("DELETE FROM topics")
    conn.executemany(
        "INSERT INTO topics (id, name, keywords) VALUES (?, ?, ?)",
        [
            (topic_id, names[topic_id], json.dumps(keywords[topic_id]))
            for topic_id in sorted(names)
        ],
    )
    conn.commit()


def update_paper_topics(
    conn: sqlite3.Connection,
    paper_ids: list[str],
    topic_ids: list[int],
) -> None:
    """
    Bulk-updates topic_id for a batch of papers.
    """
    conn.executemany(
        "UPDATE papers SET topic_id = ? WHERE id = ?",
        zip(topic_ids, paper_ids),
    )
    conn.commit()