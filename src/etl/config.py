"""
Pipeline configuration: paths, constants, filter parameters.
All other modules import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filter settings
# ---------------------------------------------------------------------------

MIN_YEAR: int = 2010
MAX_YEAR: int = 2025
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

RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_JSON = RESULTS_DIR / "stats.json"

PAPERS_CSV        = INTERIM_DIR / "papers.csv"
AUTHORS_RAW_CSV   = INTERIM_DIR / "authors_raw.csv"
AUTHORS_CSV       = INTERIM_DIR / "authors.csv"
AUTHOR_ALIASES_CSV = INTERIM_DIR / "author_aliases.csv"
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