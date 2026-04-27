import logging
import os
from pathlib import Path
import numpy as np
import time
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import torch

try:
    from cuml.cluster import HDBSCAN as CUHDBSCAN
    from cuml.manifold import UMAP as CUUMAP

    HAS_CUML = True
    CUML_IMPORT_ERROR = None
except Exception as exc:
    CUHDBSCAN = None
    CUUMAP = None
    HAS_CUML = False
    CUML_IMPORT_ERROR = str(exc)

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

MODEL_PATH = "data/processed/bertopic_model"

from config import CUSTOM_STOP_WORDS, RANDOM_STATE

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _build_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.to(device)
    return model


def _use_gpu_backend() -> bool:
    return device == "cuda" and HAS_CUML


def _build_umap_model():
    if _use_gpu_backend():
        logger.info("Using cuML UMAP on CUDA.")
        return CUUMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_STATE,
        )

    if device == "cuda" and not HAS_CUML:
        logger.warning(
            "CUDA is available, but cuML backend is unavailable (%s). Falling back to sklearn UMAP on CPU.",
            CUML_IMPORT_ERROR,
        )
    logger.info("Using sklearn UMAP on CPU.")
    return UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )


def _build_hdbscan_model():
    if _use_gpu_backend():
        logger.info("Using cuML HDBSCAN on CUDA.")
        return CUHDBSCAN(
            min_cluster_size=300,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    if device == "cuda" and not HAS_CUML:
        logger.warning(
            "CUDA is available, but cuML backend is unavailable (%s). Falling back to hdbscan on CPU.",
            CUML_IMPORT_ERROR,
        )
    logger.info("Using hdbscan on CPU.")
    return HDBSCAN(
        min_cluster_size=300,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def _build_model(embedding_model: SentenceTransformer) -> BERTopic:
    umap_model = _build_umap_model()
    hdbscan_model = _build_hdbscan_model()

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
        try:
            embeddings = np.load(embeddings_path)
            if len(embeddings) == len(docs):
                logger.info(f"Embeddings loaded: {embeddings.shape}")
                return embeddings

            logger.warning(
                "Cached embeddings size mismatch: got %s rows, expected %s. Recomputing.",
                len(embeddings),
                len(docs),
            )
        except Exception as exc:
            logger.warning("Failed to load cached embeddings (%s). Recomputing.", exc)

    logger.info(f"Encoding {len(docs):,} documents (this may take ~10 min) ...")
    embeddings = embedding_model.encode(
        docs,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    )
    np.save(embeddings_path, embeddings)
    logger.info(f"Embeddings saved to {embeddings_path}.")
    return embeddings


def _count_outliers(topics: list[int]) -> int:
    return sum(1 for t in topics if t == -1)


def _model_variants(model_path: str | Path) -> tuple[str, str, str, str]:
    base = str(model_path)
    return base, f"{base}_gpu", f"{base}_cpu", f"{base}_safe"


def train(
    sample_docs: list[str],
    model_path: str,
    embeddings_path: str,
) -> tuple[BERTopic, SentenceTransformer, list[int], np.ndarray]:
    """
    Train BERTopic on a sample of docs.
    Returns (topic_model, embedding_model, new_topics, sample_idx).
    """
    # rng = np.random.default_rng(RANDOM_STATE)
    # n = min(SAMPLE_SIZE, len(docs))
    # idx = rng.choice(len(docs), size=n, replace=False)
    # sample_docs = [docs[i] for i in idx]
    # logger.info(f"Training on {n:,} sampled documents.")

    embedding_model = _build_embedding_model()
    topic_model = _build_model(embedding_model)
    embeddings = _encode(sample_docs, embedding_model, embeddings_path)

    logger.info("Fitting BERTopic ...")
    start = time.time()
    topics, _ = topic_model.fit_transform(sample_docs, embeddings=embeddings)
    elapsed = time.time() - start
    logger.info(f"BERTopic fit_transform completed in {elapsed:.1f}s. Initial topics: {len(set(topics)) - (1 if -1 in topics else 0)}, outliers: {_count_outliers(topics):,}")

    # Strategy 1: embeddings-based outlier reduction
    logger.info("Reducing outliers [strategy: embeddings, threshold=0.2] ...")
    start = time.time()
    topics = topic_model.reduce_outliers(
        sample_docs, topics,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=0.2,
    )
    elapsed = time.time() - start
    logger.info(f"Embeddings strategy completed in {elapsed:.1f}s. Outliers after embeddings strategy: {_count_outliers(topics):,}")

    # Strategy 2: c-tf-idf for remaining outliers
    logger.info("Reducing outliers [strategy: c-tf-idf] ...")
    start = time.time()
    topics = topic_model.reduce_outliers(
        sample_docs, topics,
        strategy="c-tf-idf",
        threshold=0.1,
    )
    elapsed = time.time() - start
    logger.info(f"c-tf-idf strategy completed in {elapsed:.1f}s. Outliers after c-tf-idf strategy: {_count_outliers(topics):,}")

    start = time.time()
    topic_model.update_topics(sample_docs, topics=topics, vectorizer_model=_build_vectorizer())
    elapsed = time.time() - start
    logger.info(f"update_topics completed in {elapsed:.1f}s.")

    base_path, gpu_path, cpu_path = _model_variants(model_path)

    # Always save a portable CPU-friendly artifact.
    start = time.time()
    try:
        topic_model.save(
            cpu_path,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=EMBEDDING_MODEL_NAME,
        )
        logger.info(f"Portable CPU artifact saved to {cpu_path}.")
    except TypeError:
        # Fallback for older BERTopic versions.
        topic_model.save(cpu_path)
        logger.info(f"Portable CPU artifact fallback-saved to {cpu_path}.")

    # Save a backend-specific artifact as well.
    if _use_gpu_backend():
        topic_model.save(gpu_path)
        logger.info(f"GPU artifact saved to {gpu_path}.")
    else:
        topic_model.save(base_path)
        logger.info(f"CPU artifact saved to {base_path}.")

    elapsed = time.time() - start
    logger.info(f"Model artifacts saved in {elapsed:.1f}s.")

    return topic_model, embedding_model, topics

def load(model_path: str) -> tuple[BERTopic, SentenceTransformer]:
    """
    Load a saved BERTopic model.
    Returns (topic_model, embedding_model) – embedding_model is a plain
    SentenceTransformer, safe to call .encode() on directly.
    Prefer GPU artifact when CUDA+cuML is available, otherwise fall back to CPU artifact.
    """
    embedding_model = _build_embedding_model()

    try:
        logger.info(f"Loading BERTopic model from {model_path} ...")
        topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
        logger.info(f"Loaded BERTopic model from {model_path}.")    
    except Exception as exc:
        logger.warning(f"Failed loading {model_path}: {exc}", exc_info=True)


    # Rebuild runtime models for current backend.
    topic_model.umap_model = _build_umap_model()
    topic_model.hdbscan_model = _build_hdbscan_model()

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
            device=device,
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

