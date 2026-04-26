"""
DBLP ETL pipeline — entry point.

Usage:
    python etl_pipeline.py <path/to/dblp.xml>

Phases:
    0:   Get basic information about the dataset
    1+2. Stream XML → papers.csv + authors_raw.csv
    3.   Deduplicate authors → authors.csv + paper_authors.csv
    4.   Bulk-load CSVs → dblp.db (SQLite)
    5.   Verify row counts and key invariants
"""


import logging
import sys
from pathlib import Path

from .authors import resolve_authors
from .loader import load_into_sqlite
from .parser import stream_records, get_stats
from .transform import write_raw_csvs
from .verify import verify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/dblp.xml>")
        sys.exit(1)

    xml_path = Path(sys.argv[1])
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # log.info("=== Phase 0: Get basic information about the dataset ===")
    # get_stats(xml_path)

    log.info("=== Phase 1+2: Parse XML + write raw CSVs ===")
    write_raw_csvs(stream_records(xml_path))

    # log.info("=== Phase 3: Resolve authors ===")
    # resolve_authors()

    # log.info("=== Phase 4: Load into SQLite ===")
    # load_into_sqlite()

    # log.info("=== Phase 5: Verify ===")
    # verify()

    log.info("Pipeline complete.")



