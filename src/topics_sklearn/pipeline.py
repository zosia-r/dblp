import sqlite3

import pandas as pd

from .db import apply_schema, insert_topics, update_paper_topics
from .model import TopicModel


def run(
    db_path: str,
    n_clusters: int = 8,
    sample_size: int = 100_000,
    batch_size: int = 50_000,
    random_state: int = 42,
) -> None:
    """
    Full topic-modeling pipeline:
      1. Fit TF-IDF + KMeans on a random sample of titles.
      2. Name topics with Gemini (falls back to 'Topic N' if unavailable).
      3. Predict topic for every paper in batches and write to DB.

    Args:
        db_path:      Path to the SQLite database file.
        n_clusters:   Number of topics (k for KMeans).
        sample_size:  Number of papers used for fitting (0 = all).
        batch_size:   Papers processed per prediction + DB write batch.
        random_state: Seed for reproducibility.
    """
    conn = sqlite3.connect(db_path)

    # ── Schema migration ────────────────────────────────────────────────────
    print("Applying schema migrations...")
    apply_schema(conn)

    # ── Load sample for fitting ─────────────────────────────────────────────
    print(f"Loading sample (n={sample_size:,}) for model fitting...")
    sample_query = """
        SELECT id, title FROM papers
        WHERE title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """
    sample_df = pd.read_sql(sample_query, conn, params=(sample_size,))
    sample_df["title"] = sample_df["title"].str.lower()

    # ── Fit ─────────────────────────────────────────────────────────────────
    print(f"Fitting TF-IDF + KMeans (k={n_clusters})...")
    model = TopicModel(n_clusters=n_clusters, random_state=random_state)
    model.fit(sample_df["title"].tolist())

    for i, words in model.top_words().items():
        print(f"  Cluster {i}: {words}")

    # ── Name topics ─────────────────────────────────────────────────────────
    print("Naming topics...")
    names = model.name_topics_with_gemini()
    for i, name in names.items():
        print(f"  Topic {i}: {name}")

    # ── Persist topic definitions ────────────────────────────────────────────
    print("Writing topics table...")
    insert_topics(conn, names, model.top_words())

    # ── Predict for all papers in batches ───────────────────────────────────
    total = pd.read_sql("SELECT COUNT(*) AS n FROM papers WHERE title IS NOT NULL", conn).iloc[0]["n"]
    print(f"Predicting topics for {total:,} papers in batches of {batch_size:,}...")

    processed = 0
    offset = 0

    while True:
        batch_df = pd.read_sql(
            "SELECT id, title FROM papers WHERE title IS NOT NULL LIMIT ? OFFSET ?",
            conn,
            params=(batch_size, offset),
        )
        if batch_df.empty:
            break

        batch_df["title"] = batch_df["title"].str.lower()
        topic_ids = model.predict_batch(batch_df["title"].tolist()).tolist()
        update_paper_topics(conn, batch_df["id"].tolist(), topic_ids)

        processed += len(batch_df)
        offset += batch_size
        print(f"  {processed:,} / {total:,} papers updated")

    conn.close()
    print("Done.")