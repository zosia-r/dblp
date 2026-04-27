import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import *
from .data_loader import DataLoader
from chromadb.errors import InvalidArgumentError
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

log = logging.getLogger("Retriever")

class Retriever:
    def __init__(self):
        log.info("Initializing Retriever...")

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.model_name = EMBEDDING_MODEL
        self.model_dim = self.embedding_model.get_embedding_dimension()

        log.info(f"Setting up ChromaDB client with persist directory '{CHROMA_PATH}'...")
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        self.loader = DataLoader()

    def _ensure_embedding_model_dim(self, target_dim: int) -> None:
        if self.model_dim == target_dim:
            return

        dim_to_model = {
            384: "all-MiniLM-L6-v2",
            768: "all-mpnet-base-v2",
        }
        fallback_model = dim_to_model.get(target_dim)

        if fallback_model is None:
            raise ValueError(
                f"Embedding dimension mismatch: model '{self.model_name}' outputs {self.model_dim}, "
                f"but index embeddings have dim {target_dim}."
            )

        log.warning(
            "Embedding model '%s' (dim=%s) does not match index dim=%s. Switching to '%s'.",
            self.model_name,
            self.model_dim,
            target_dim,
            fallback_model,
        )
        self.embedding_model = SentenceTransformer(fallback_model)
        self.model_name = fallback_model
        self.model_dim = self.embedding_model.get_embedding_dimension()

    def _collection_accepts_dim(self, dim: int) -> bool:
        if self.collection.count() == 0:
            return True
        try:
            self.collection.query(
                query_embeddings=[[0.0] * dim],
                n_results=1,
            )
            return True
        except InvalidArgumentError as exc:
            if "dimension" in str(exc).lower():
                return False
            raise

    def index_if_needed(self):
        log.info("Checking if indexing is needed...")
        data = self.loader.load()
        docs = data["docs"]
        ids = data["ids"]
        embeddings = data["embeddings"]
        expected_dim = int(embeddings.shape[1])

        self._ensure_embedding_model_dim(expected_dim)

        current_count = self.collection.count()
        expected_count = len(ids)
        has_valid_dim = self._collection_accepts_dim(expected_dim)

        has_valid_count = current_count == expected_count and current_count > 0
        if has_valid_count and has_valid_dim:
            return

        if current_count > 0:
            log.warning(
                "Rebuilding Chroma collection due to mismatch (count=%s expected=%s, dim_ok=%s).",
                current_count,
                expected_count,
                has_valid_dim,
            )
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        max_batch_size = getattr(self.client, "get_max_batch_size", lambda: 5000)()

        for start in range(0, len(docs), max_batch_size):
            
            end = min(start + max_batch_size, len(docs))
            self.collection.add(
                documents=docs[start:end],
                embeddings=embeddings[start:end].tolist(),
                ids=ids[start:end],
            )

    def search(self, query, k=TOP_K):
        query_embedding = self.embedding_model.encode([query])

        try:
            return self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
        except InvalidArgumentError as exc:
            # Recover once if index/model drifted after startup.
            if "dimension" not in str(exc).lower():
                raise
            log.warning("Detected embedding dimension mismatch during query, reindexing and retrying once...")
            self.index_if_needed()
            query_embedding = self.embedding_model.encode([query])
            return self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )