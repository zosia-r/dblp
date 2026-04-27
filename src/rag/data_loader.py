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

        papers = pd.read_parquet(base / "papers.parquet")

        if not os.path.exists(EMBEDDINGS_PATH):
            log.info(f"Model file not found locally. Downloading from Hugging Face repo '{MODEL_REPO}'...")
            hf_hub_download(MODEL_REPO, repo_type="dataset", filename="embeddings.npy", local_dir=str(EMBEDDINGS_PATH.parent), token=API_KEY)
        embeddings = np.load(EMBEDDINGS_PATH)

        assert len(papers) == len(embeddings), \
            "Mismatch papers ↔ embeddings"

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