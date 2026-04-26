import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import *
from .data_loader import DataLoader


class Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.Client(
            settings=chromadb.config.Settings(
                persist_directory=CHROMA_PATH
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        self.loader = DataLoader()

    def index_if_needed(self):
        if self.collection.count() > 0:
            return

        data = self.loader.load()

        self.collection.add(
            documents=data["docs"],
            embeddings=data["embeddings"].tolist(),
            ids=data["ids"]
        )

    def search(self, query, k=TOP_K):
        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )

        return results