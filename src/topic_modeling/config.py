from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pathlib import Path



SAMPLE_SIZE = 200_000
RANDOM_STATE = 42

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    "based", "using", "novel", "efficient", "approach", "method",
    "paper", "proposed", "new", "via", "toward", "towards", "use",
    "used", "study", "system", "systems",
]


DB_PATH         = Path("data/processed/dblp.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH      = Path("data/models/bertopic_model")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH = Path("data/models/embeddings.npy")
EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)