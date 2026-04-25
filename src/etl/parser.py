"""
Phase 0 — XML stats.

Phase 1 — XML streaming parser.
 
Emits two types of records:
 
  1. Paper records  (tag: article | inproceedings)
     {"type": "paper", "key", "title", "year", "venue", "record_type", "authors"}
 
  2. Author identity records  (tag: www key="homepages/...")
     {"type": "author", "primary_name", "aliases"}
 
     First <author> = primary name, remaining = aliases.
 
Paper qualifying criteria: year >= MIN_YEAR, non-empty title, >= 1 author.
"""

import logging
from pathlib import Path
from typing import Generator
import json

from lxml import etree

from .config import (
    MIN_YEAR, MAX_YEAR, 
    TARGET_TAGS, LOG_PROGRESS_EVERY, 
    STATS_JSON
)

log = logging.getLogger(__name__)

Record = dict  # {key, title, year, venue, type, authors}



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _itertext(element) -> str:
    """
    Reconstruct full text of an element including text inside inline child
    tags such as <sub>, <sup>, <i>.

    Example: <title>Fast <i>k</i>-NN Search</title>  ->  "Fast k-NN Search"
    """
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _parse_year(text: str) -> int | None:
    """Return integer year or None if text is missing / non-numeric."""
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None

def _parse_www(elem) -> Record | None:
    """
    Parse <www> record with key="homepages/...".
    These records contain author identity information:
        - primary name (first <author>)
        - aliases (remaining <author> tags)
   """
    if not elem.get("key", "").startswith("homepages/"):
        return None
    names = [_itertext(a) for a in elem.findall("author")]
    names = [n for n in names if n]
    if not names:
        return None
    return {
        "type":         "author",
        "primary_name": names[0],
        "aliases":      names[1:],
    }
 
 
def _parse_paper(elem, stats: dict) -> Record | None:
    """
    Parse paper record.
    Returns None if record is malformed or doesn't meet criteria.
    Updates stats dict with counts of seen/skipped/accepted records.
    """
    stats["seen"] += 1

    key = elem.get("key", "").strip()
    if not key:
        stats["skipped_fields"] += 1
        return None
 
    year_el = elem.find("year")
    year    = _parse_year(year_el.text if year_el is not None else "")
    if year is None or year < MIN_YEAR or year > MAX_YEAR:
        stats["skipped_year"] += 1
        return None
 
    title_el = elem.find("title")
    title    = _itertext(title_el) if title_el is not None else ""
    if not title:
        stats["skipped_fields"] += 1
        return None
 
    authors = [_itertext(a) for a in elem.findall("author")]
    authors = [a for a in authors if a]
    if not authors:
        stats["skipped_fields"] += 1
        return None
 
    venue_tag = "journal" if elem.tag == "article" else "booktitle"
    venue_el  = elem.find(venue_tag)
    venue     = (venue_el.text or "").strip() if venue_el is not None else ""
 
    stats["accepted"] += 1
    if stats["accepted"] % LOG_PROGRESS_EVERY == 0:
        log.info("  parsing: %d accepted, %d seen so far...", stats["accepted"], stats["seen"])
 
    return {
        "type":        "paper",
        "key":         key,
        "title":       title,
        "year":        year,
        "venue":       venue,
        "record_type": elem.tag,
        "authors":     authors,
    }
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_records(xml_path: Path) -> Generator[Record, None, None]:
    """
    Stream records from the XML file, yielding one at a time.
    Yields either paper records or author identity records.
    """
    stats = {
        "seen":           0,
        "accepted":       0,
        "skipped_year":   0,
        "skipped_fields": 0,
        "authors_seen": 0,
    }

    context = etree.iterparse(
        str(xml_path),
        events=("end",),
        load_dtd=True,
        no_network=True,
        recover=True,
    )

    for _event, elem in context:
        parent = elem.getparent()
        if parent is None or parent.tag != "dblp":
            continue

        if elem.tag == "www":
            record = _parse_www(elem)
            if record is not None:
                stats["authors_seen"] += 1
                yield record
        
        elif elem.tag in TARGET_TAGS:
            record = _parse_paper(elem, stats)
            if record is not None:
                yield record
        
        
        elem.clear()

    log.info(
        "XML parse done | papers seen: %d | accepted: %d | "
        "skipped (year): %d | skipped (fields): %d | author identities: %d",
        stats["seen"], stats["accepted"],
        stats["skipped_year"], stats["skipped_fields"], stats["authors_seen"],
    )


def get_stats(xml_path: Path) -> None:
    """
    Quick pass to get overall stats about the XML file.
    Counts total number of records and distribution of record types (article, inproceedings, proceedings, book, incollection, phdthesis, mastersthesis, www, person, data).
    Checks which fields appear consistently and which are often missing.
    """

    details = {
        "count": 0,
        "author": 0,
        "editor": 0,
        "title": 0,
        "booktitle": 0,
        "pages": 0,
        "year": 0,
        "address": 0,
        "journal": 0,
        "volume": 0,
        "number": 0,
        "month": 0,
        "url": 0,
        "ee": 0,
        "cdrom": 0,
        "cite": 0,
        "publisher": 0,
        "note": 0,
        "crossref": 0,
        "isbn": 0,
        "series": 0,
        "school": 0,
        "chapter": 0,
        "publnr": 0,
        "stream": 0,
        "rel": 0,
    }

    stats = {
        "total": 0,
        "article": details.copy(),
        "inproceedings": details.copy(),
        "proceedings": details.copy(),
        "book": details.copy(),
        "incollection": details.copy(),
        "phdthesis": details.copy(),
        "mastersthesis": details.copy(),
        "www": details.copy(),
        "person": details.copy(),
        "data": details.copy()
    }

    context = etree.iterparse(
        str(xml_path),
        events=("end",),
        load_dtd=True,
        no_network=True,
        recover=True,
    )

    for _event, elem in context:
        parent = elem.getparent()
        if parent is None or parent.tag != "dblp":
            continue

        stats["total"] += 1
        stats[elem.tag]["count"] += 1

        for field in details.keys():
            if elem.find(field) is not None:
                stats[elem.tag][field] += 1


        elem.clear()

    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=4)

    log.info("Stats saved to %s", STATS_JSON)