"""
sample_db.py — stratified sampling (per year) from SQLite → Parquet
Usage: python sample_db.py --db dblp.db --fraction 0.1 --out sample_data/
"""

import argparse
import sqlite3
from pathlib import Path
import logging
import sys

import pyarrow as pa
import pyarrow.parquet as pq


COMPRESSION = "zstd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def write_parquet(path: Path, rows: list[sqlite3.Row]) -> int:
    if not rows:
        log.warning(f"No data to write for {path.name}. Skipping.")
        return 0
    table = pa.Table.from_pylist([dict(r) for r in rows])
    pq.write_table(table, path, compression=COMPRESSION)
    mb = path.stat().st_size / 1024**2
    log.info(f"  {path.name:<30} {len(rows):>10,} rows   {mb:6.1f} MB")
    return len(rows)


def load_temp_table(con: sqlite3.Connection, name: str, col: str, col_type: str, values: list) -> None:
    """Wstawia listę wartości do tymczasowej tabeli — bez limitu zmiennych."""
    con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {name} ({col} {col_type} PRIMARY KEY)")
    con.execute(f"DELETE FROM {name}")
    con.executemany(f"INSERT OR IGNORE INTO {name} VALUES (?)", [(v,) for v in values])


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def sample_paper_ids(con: sqlite3.Connection, fraction: float) -> list[str]:
    """Stratified sampling — same fraction for each year."""
    years = [
        r["year"]
        for r in con.execute("SELECT DISTINCT year FROM papers ORDER BY year")
    ]

    sampled: list[str] = []
    for year in years:
        ids = [
            r["id"]
            for r in con.execute(
                "SELECT id FROM papers WHERE year = ? ORDER BY RANDOM()", (year,)
            )
        ]
        n = max(1, round(len(ids) * fraction))
        sampled.extend(ids[:n])

    return sampled


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(db_path: str, fraction: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)

    log.info(f"\n→ Connecting to DB: {db_path}")
    log.info(f"→ Fraction: {fraction:.1%}")
    log.info(f"→ Output:   {output_dir}/\n")

    # 1. papers ---------------------------------------------------------------
    log.info("Sampling papers…")
    paper_ids = sample_paper_ids(con, fraction)
    log.info(f"  Selected {len(paper_ids):,} papers total")

    load_temp_table(con, "_paper_ids", "id", "TEXT", paper_ids)

    papers_rows = con.execute("""
        SELECT p.* FROM papers p
        JOIN _paper_ids t ON p.id = t.id
        ORDER BY p.year, p.id
    """).fetchall()
    write_parquet(output_dir / "papers.parquet", papers_rows)

    # 2. paper_authors — tylko dla wybranych papers ---------------------------
    log.info("Filtering paper_authors…")
    pa_rows = con.execute("""
        SELECT pa.* FROM paper_authors pa
        JOIN _paper_ids t ON pa.paper_id = t.id
        ORDER BY pa.paper_id, pa.author_order
    """).fetchall()
    write_parquet(output_dir / "paper_authors.parquet", pa_rows)

    # 3. authors — tylko ci, którzy wystąpili w wybranych papers --------------
    log.info("Filtering authors…")
    author_ids = list({r["author_id"] for r in pa_rows})
    load_temp_table(con, "_author_ids", "id", "INTEGER", author_ids)

    authors_rows = con.execute("""
        SELECT a.* FROM authors a
        JOIN _author_ids t ON a.id = t.id
        ORDER BY a.id
    """).fetchall()
    write_parquet(output_dir / "authors.parquet", authors_rows)

    # 4. author_aliases -------------------------------------------------------
    log.info("Filtering author_aliases…")
    aliases_rows = con.execute("""
        SELECT aa.* FROM author_aliases aa
        JOIN _author_ids t ON aa.author_id = t.id
        ORDER BY aa.author_id
    """).fetchall()
    write_parquet(output_dir / "author_aliases.parquet", aliases_rows)

    # 5. topics — tylko te, do których odwołują się wybrane papers ------------
    log.info("Filtering topics…")
    has_topics = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='topics'"
    ).fetchone()

    if has_topics:
        topics_rows = con.execute("""
            SELECT DISTINCT t.* FROM topics t
            JOIN papers p ON p.topic_id = t.id
            JOIN _paper_ids s ON p.id = s.id
            ORDER BY t.id
        """).fetchall()
        write_parquet(output_dir / "topics.parquet", topics_rows)
    else:
        log.warning("  [skip] topics — table not found in database.")

    # summary -----------------------------------------------------------------
    total_mb = sum(
        f.stat().st_size for f in output_dir.glob("*.parquet")
    ) / 1024**2
    log.info(f"\n✓ Done — total {total_mb:.1f} MB in {output_dir}/")

    con.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample SQLite DB → Parquet")
    parser.add_argument("--db",       default="data/processed/dblp.db",   help="Path to SQLite DB")
    parser.add_argument("--fraction", default=0.10, type=float, help="Sampling fraction (default: 0.10)")
    parser.add_argument("--out",      default="sample_data", help="Output directory for Parquet files")
    args = parser.parse_args()

    run(args.db, args.fraction, Path(args.out))