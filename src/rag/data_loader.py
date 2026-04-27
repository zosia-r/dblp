import logging
import os

from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EmbeddingDataLoader")
from .config import *
from pathlib import Path

MODEL_REPO = os.environ.get("HF_EMBEDDING_REPO", "zofia/dblp-bertopic")
EMBEDDINGS_PATH = Path("data/models/embeddings.npy")
API_KEY = os.getenv("HF_API_KEY")
if API_KEY is None:
    log.warning("Hugging Face API key not found. Please set the HF_API_KEY environment variable.")

class DataLoader:
    def load(self):
        base = Path(PARQUET_PATH)

        papers = pd.read_parquet(base / "papers.parquet", columns=["id", "title", "venue", "year"])

        if not os.path.exists(EMBEDDINGS_PATH):
            log.info(f"Model file not found locally. Downloading from Hugging Face repo '{MODEL_REPO}'...")
            hf_hub_download(MODEL_REPO, repo_type="dataset", filename="embeddings.npy", local_dir=str(EMBEDDINGS_PATH.parent), token=API_KEY)
        embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

        assert len(papers) == len(embeddings), \
            "Mismatch papers ↔ embeddings"

        frac = max(0.0, min(1.0, float(RAG_INDEX_FRACTION)))
        if frac <= 0.0:
            raise ValueError("RAG_INDEX_FRACTION must be > 0.")

        n_total = len(papers)
        n_keep = max(1, int(n_total * frac))
        if n_keep < n_total:
            rng = np.random.default_rng(RAG_INDEX_SEED)
            keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
            papers = papers.iloc[keep_idx].reset_index(drop=True)
            embeddings = np.asarray(embeddings[keep_idx])
            log.info(
                "Indexing sample: kept %s / %s rows (%.1f%%).",
                n_keep,
                n_total,
                frac * 100,
            )
        else:
            embeddings = np.asarray(embeddings)

        papers["text"] = (
            "Title: " + papers["title"].fillna("") +
            " Venue: " + papers["venue"].fillna("") +
            " Year: " + papers["year"].astype(str)
        )

        return {
            "docs": papers["text"].tolist(),
            "ids": papers["id"].astype(str).tolist(),
            "embeddings": embeddings
        }