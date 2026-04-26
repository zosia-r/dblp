import logging
import os

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)

from config import CUSTOM_STOP_WORDS, RANDOM_STATE, SAMPLE_SIZE

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _build_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _build_model(embedding_model: SentenceTransformer) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=300,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
    )

    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        top_n_words=10,
        nr_topics=50,
        calculate_probabilities=False,
        verbose=True,
    )


def _build_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
    )


def _encode(
    docs: list[str],
    embedding_model: SentenceTransformer,
    embeddings_path: str,
) -> np.ndarray:
    if os.path.exists(embeddings_path):
        logger.info(f"Loading embeddings from {embeddings_path} ...")
        embeddings = np.load(embeddings_path)
        logger.info(f"Embeddings loaded: {embeddings.shape}")
        return embeddings

    logger.info(f"Encoding {len(docs):,} documents (this may take ~10 min) ...")
    embeddings = embedding_model.encode(
        docs,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(embeddings_path, embeddings)
    logger.info(f"Embeddings saved to {embeddings_path}.")
    return embeddings


def _count_outliers(topics: list[int]) -> int:
    return sum(1 for t in topics if t == -1)


def train(
    docs: list[str],
    model_path: str,
    embeddings_path: str,
) -> tuple[BERTopic, SentenceTransformer, list[int], np.ndarray]:
    """
    Train BERTopic on a sample of docs.
    Returns (topic_model, embedding_model, new_topics, sample_idx).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SAMPLE_SIZE, len(docs))
    idx = rng.choice(len(docs), size=n, replace=False)
    sample_docs = [docs[i] for i in idx]
    logger.info(f"Training on {n:,} sampled documents.")

    embedding_model = _build_embedding_model()
    topic_model = _build_model(embedding_model)
    embeddings = _encode(sample_docs, embedding_model, embeddings_path)

    logger.info("Fitting BERTopic ...")
    topics, _ = topic_model.fit_transform(sample_docs, embeddings=embeddings)
    logger.info(f"Initial topics: {len(set(topics)) - (1 if -1 in topics else 0)}, outliers: {_count_outliers(topics):,}")

    # Strategy 1: embeddings-based outlier reduction
    logger.info("Reducing outliers [strategy: embeddings, threshold=0.2] ...")
    topics = topic_model.reduce_outliers(
        sample_docs, topics,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=0.2,
    )
    logger.info(f"Outliers after embeddings strategy: {_count_outliers(topics):,}")

    # Strategy 2: c-tf-idf for remaining outliers
    logger.info("Reducing outliers [strategy: c-tf-idf] ...")
    topics = topic_model.reduce_outliers(
        sample_docs, topics,
        strategy="c-tf-idf",
        threshold=0.1,
    )
    logger.info(f"Outliers after c-tf-idf strategy: {_count_outliers(topics):,}")

    topic_model.update_topics(sample_docs, topics=topics, vectorizer_model=_build_vectorizer())

    topic_model.save(model_path)
    logger.info(f"Model saved to {model_path}.")

    return topic_model, embedding_model, topics, idx


def load(model_path: str) -> tuple[BERTopic, SentenceTransformer]:
    """
    Load a saved BERTopic model.
    Returns (topic_model, embedding_model) – embedding_model is a plain
    SentenceTransformer, safe to call .encode() on directly.
    """
    logger.info(f"Loading BERTopic model from {model_path} ...")
    embedding_model = _build_embedding_model()
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
    logger.info("Model loaded.")
    return topic_model, embedding_model


def get_topic_labels(topic_model: BERTopic, n_keywords: int = 5) -> dict[int, str]:
    """Build topic labels by joining top-N keywords."""
    info = topic_model.get_topic_info()
    labels: dict[int, str] = {}

    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            labels[-1] = "Outliers"
            continue
        keywords = [w for w, _ in topic_model.get_topic(tid)[:n_keywords]]
        labels[tid] = " / ".join(keywords)
        logger.info(f"  Topic {tid:3d} ({row['Count']:5,} docs): {labels[tid]}")

    return labels


def transform_all(
    topic_model: BERTopic,
    all_docs: list[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 50_000,
) -> list[int]:
    """
    Transform all documents in batches, reducing outliers per batch
    using two sequential strategies: embeddings → c-tf-idf.
    Returns a list of topic ids (-1 for unassigned outliers).
    """
    logger.info(f"Transforming {len(all_docs):,} documents in batches of {batch_size:,} ...")
    all_topics: list[int] = []

    for start in range(0, len(all_docs), batch_size):
        batch = all_docs[start : start + batch_size]

        batch_embeddings = embedding_model.encode(
            batch,
            batch_size=1024,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Initial transform
        topics, _ = topic_model.transform(batch, embeddings=batch_embeddings)
        topics = list(topics)

        # Strategy 1: embeddings
        topics = topic_model.reduce_outliers(
            batch, topics,
            strategy="embeddings",
            embeddings=batch_embeddings,
            threshold=0.2,
        )

        # Strategy 2: c-tf-idf for remaining outliers
        topics = topic_model.reduce_outliers(
            batch, topics,
            strategy="c-tf-idf",
            threshold=0.1,
        )

        all_topics.extend(topics)

        processed = min(start + batch_size, len(all_docs))
        batch_outliers = _count_outliers(list(topics))
        logger.info(
            f"  Processed {processed:,} / {len(all_docs):,} "
            f"| batch outliers: {batch_outliers} / {len(batch)} "
            f"({batch_outliers / len(batch) * 100:.1f}%)"
        )

    total_outliers = _count_outliers(all_topics)
    logger.info(
        f"Transform complete. Total outliers: {total_outliers:,} / {len(all_docs):,} "
        f"({total_outliers / len(all_docs) * 100:.1f}%)"
    )
    return all_topics