import os

MODEL_PATH = "data/models/bertopic_model"
EMBEDDINGS_PATH = "data/models/embeddings.npy"

# DATA SOURCE
USE_PARQUET = True
PARQUET_PATH = "sample_data"

SQLITE_PATH = "data/dblp.sqlite"

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "papers"

EMBEDDING_MODEL = "all-mpnet-base-v2"
RAG_INDEX_FRACTION = float(os.getenv("RAG_INDEX_FRACTION", "0.5"))
RAG_INDEX_SEED = int(os.getenv("RAG_INDEX_SEED", "42"))

# LLM
USE_GEMINI = True
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K = 5