"""
topic_stats.py – statistics extracted directly from a trained BERTopic model.
All functions accept a BERTopic instance and return DataFrames or dicts
ready for Streamlit / Plotly consumption.
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from plotly import express as px


# ── Overview ──────────────────────────────────────────────────────────────────

def get_overview(topic_model: BERTopic) -> dict:
    """
    High-level model statistics.
    Returns: total_docs, num_topics, outlier_count, outlier_pct,
             avg_topic_size, largest_topic_size, smallest_topic_size.
    """
    info = topic_model.get_topic_info()
    topics_only = info[info["Topic"] != -1]
    outlier_row = info[info["Topic"] == -1]

    total_docs = int(info["Count"].sum())
    outlier_count = int(outlier_row["Count"].values[0]) if not outlier_row.empty else 0

    return {
        "total_docs": total_docs,
        "num_topics": len(topics_only),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / total_docs * 100, 2) if total_docs else 0,
        "avg_topic_size": round(float(topics_only["Count"].mean()), 1) if not topics_only.empty else 0,
        "largest_topic_size": int(topics_only["Count"].max()) if not topics_only.empty else 0,
        "smallest_topic_size": int(topics_only["Count"].min()) if not topics_only.empty else 0,
    }


# ── Topic distribution ────────────────────────────────────────────────────────

def get_topic_distribution(topic_model: BERTopic) -> pd.DataFrame:
    """
    Topic sizes sorted descending. Outliers excluded.
    Columns: topic_id, label, count, pct.
    """
    info = topic_model.get_topic_info()
    df = info[info["Topic"] != -1].copy()
    total = df["Count"].sum()

    df = df.rename(columns={"Topic": "topic_id", "Count": "count", "Name": "label"})
    df["pct"] = (df["count"] / total * 100).round(2)
    df = df[["topic_id", "label", "count", "pct"]].sort_values("count", ascending=False).reset_index(drop=True)
    
    return df


# ── Keywords ──────────────────────────────────────────────────────────────────

def get_all_topic_keywords(topic_model: BERTopic, n_words: int = 10) -> pd.DataFrame:
    """
    Top-N keywords with scores for every topic.
    Columns: topic_id, label, rank, word, score.
    """
    info = topic_model.get_topic_info()
    rows = []
    for _, row in info[info["Topic"] != -1].iterrows():
        tid = int(row["Topic"])
        for rank, (word, score) in enumerate(topic_model.get_topic(tid)[:n_words], start=1):
            rows.append({
                "topic_id": tid,
                "label": row["Name"],
                "rank": rank,
                "word": word,
                "score": round(float(score), 4),
            })
    return pd.DataFrame(rows)


def get_keywords_for_topic(topic_model: BERTopic, topic_id: int, n_words: int = 15) -> pd.DataFrame:
    """
    Keywords for a single topic.
    Columns: rank, word, score.
    """
    words = topic_model.get_topic(topic_id)[:n_words]
    return pd.DataFrame(
        [{"rank": i + 1, "word": w, "score": round(float(s), 4)} for i, (w, s) in enumerate(words)]
    )


# ── Topic similarity matrix ───────────────────────────────────────────────────

def get_topic_similarity_matrix(topic_model: BERTopic) -> pd.DataFrame:
    """
    Cosine similarity matrix between topic embeddings.
    Returns square DataFrame indexed and columned by topic labels.
    """
    info = topic_model.get_topic_info()
    topics_only = info[info["Topic"] != -1]
    tids = topics_only["Topic"].tolist()
    labels = topics_only["Name"].tolist()

    vectors = np.array([topic_model.topic_embeddings_[tid] for tid in tids])
    sim = cosine_similarity(vectors)
    return pd.DataFrame(sim, index=labels, columns=labels)


# ── Representative documents ──────────────────────────────────────────────────

def get_representative_docs(topic_model: BERTopic) -> pd.DataFrame:
    """
    Representative documents stored during fit_transform.
    Columns: topic_id, label, doc.
    """
    info = topic_model.get_topic_info()
    rows = []
    for _, row in info[info["Topic"] != -1].iterrows():
        tid = int(row["Topic"])
        rep_docs = topic_model.get_representative_docs(tid) or []
        for doc in rep_docs:
            rows.append({
                "topic_id": tid,
                "label": row["Name"],
                "doc": doc,
            })
    return pd.DataFrame(rows)


# ── Topics over time ──────────────────────────────────────────────────────────

def get_topics_over_time(
    topic_model: BERTopic,
    docs: list[str],
    timestamps: list,
    nr_bins: int = 10,
) -> pd.DataFrame:
    """
    Topic frequency over time bins.
    Requires docs and timestamps (e.g. publication years as ints).
    Columns: Topic, Words, Frequency, Timestamp.
    """
    return topic_model.topics_over_time(
        docs, timestamps, nr_bins=nr_bins, global_tuning=True
    )