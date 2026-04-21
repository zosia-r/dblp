"""
Pipeline configuration: paths, constants, filter parameters.
All other modules import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filter settings
# ---------------------------------------------------------------------------

MIN_YEAR: int = 2011
TARGET_TAGS: frozenset[str] = frozenset({"article", "inproceedings"})

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR: Path = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR: Path = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_CSV        = INTERIM_DIR / "papers.csv"
AUTHORS_RAW_CSV   = INTERIM_DIR / "authors_raw.csv"
AUTHORS_CSV       = INTERIM_DIR / "authors.csv"
PAPER_AUTHORS_CSV = INTERIM_DIR / "paper_authors.csv"

DB_PATH = PROCESSED_DIR / "dblp.db"

# ---------------------------------------------------------------------------
# Loader settings
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 100_000

# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------
LOG_PROGRESS_EVERY: int = 500_000