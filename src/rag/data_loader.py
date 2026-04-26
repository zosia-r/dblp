import pandas as pd
import numpy as np
from .config import *
from pathlib import Path


class DataLoader:
    def load(self):
        base = Path(PARQUET_PATH)

        papers = pd.read_parquet(base / "papers.parquet")

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