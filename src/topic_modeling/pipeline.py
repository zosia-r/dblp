"""
Topic modeling pipeline for DBLP papers using BERTopic.

Usage:
    python pipeline.py              # train + transform + save
    python pipeline.py --transform  # load existing model + transform only
"""

import argparse
import logging
import os
import sys

import db
import model as m
from config import DB_PATH, EMBEDDINGS_PATH, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_train_and_transform() -> None:
    # 1. Load data
    papers = db.load_papers(DB_PATH)
    paper_ids = [p["id"] for p in papers]
    docs = db.load_sample_papers("sample_data/papers.parquet")

    # 2. Train on sample – returns plain SentenceTransformer alongside model
    topic_model, embedding_model, sample_topics = m.train(
        docs,
        model_path=MODEL_PATH,
        embeddings_path=EMBEDDINGS_PATH,
    )

    # 3. Generate labels
    logger.info("Generating topic labels ...")
    topic_labels = m.get_topic_labels(topic_model, n_keywords=5)

    # 4. Save topics table
    db.setup_topics_schema(DB_PATH)
    db.save_topics(DB_PATH, topic_labels)

    # 5. Transform ALL documents
    all_topics = m.transform_all(topic_model, docs, embedding_model)

    # 6. Save to papers table
    db.save_paper_topics(DB_PATH, paper_ids, all_topics)

    logger.info("Pipeline complete.")
    _print_summary(topic_labels, all_topics)


def run_transform_only() -> None:
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run without --transform first.")
        sys.exit(1)

    # 1. Load data
    papers = db.load_papers(DB_PATH)
    paper_ids = [p["id"] for p in papers]
    docs = db.load_sample_papers("sample_data/papers.parquet")

    # 2. Load model – returns (topic_model, plain SentenceTransformer)
    topic_model, embedding_model = m.load(MODEL_PATH)

    # 3. Generate labels
    logger.info("Generating topic labels ...")
    topic_labels = m.get_topic_labels(topic_model, n_keywords=5)

    # 4. Save topics table
    db.setup_topics_schema(DB_PATH)
    db.save_topics(DB_PATH, topic_labels)

    # 5. Transform ALL documents
    all_topics = m.transform_all(topic_model, docs, embedding_model)

    # 6. Save to papers table
    db.save_paper_topics(DB_PATH, paper_ids, all_topics)

    logger.info("Pipeline complete.")
    _print_summary(topic_labels, all_topics)


def _print_summary(topic_labels: dict, all_topics: list[int]) -> None:
    from collections import Counter

    counts = Counter(all_topics)
    total = len(all_topics)

    logger.info("─── Topic distribution (top 20) ───────────────────────")
    for tid, count in counts.most_common(20):
        label = topic_labels.get(tid, "Outliers")
        logger.info(f"  [{tid:3d}] {count:7,} docs  ({count/total*100:5.1f}%)  {label}")

    outliers = counts.get(-1, 0)
    logger.info(f"─── Outliers: {outliers:,} / {total:,} ({outliers/total*100:.1f}%) ──────────────────")


def run() -> None:
    parser = argparse.ArgumentParser(description="BERTopic pipeline for DBLP")
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Skip training; load existing model and transform all documents.",
    )
    args = parser.parse_args()

    if args.transform:
        logger.info("Running in TRANSFORM-ONLY mode.")
        run_transform_only()
    else:
        logger.info("Running in TRAIN-AND-TRANSFORM mode.")
        run_train_and_transform()

if __name__ == "__main__":
    run()