"""
Phase 1 — XML streaming parser.

Streams the DBLP XML file using lxml iterparse,
emits one dict per qualifying record.

Qualifying criteria (all must hold):
  - tag is in TARGET_TAGS
  - year >= MIN_YEAR
  - title is non-empty
  - at least one non-empty author
"""

import logging
from pathlib import Path
from typing import Generator

from lxml import etree

from .config import MIN_YEAR, TARGET_TAGS, LOG_PROGRESS_EVERY

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_records(xml_path: Path) -> Generator[Record, None, None]:
    """
    Yield one Record dict per qualifying DBLP entry.

    Uses lxml iterparse with:
      - load_dtd=True   -> resolves DBLP HTML entities (&uuml;, &eacute; ...)
      - no_network=True -> DTD must sit next to dblp.xml
      - recover=True    -> tolerate minor XML malformations

    Each top-level record element is cleared after processing.
    Method elem.clear() is called ONLY on direct children of <dblp>.
    """
    stats = {
        "seen":           0,
        "accepted":       0,
        "skipped_year":   0,
        "skipped_fields": 0,
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

        if elem.tag not in TARGET_TAGS:
            elem.clear()
            continue

        stats["seen"] += 1

        # --- year: cheapest filter first ---
        year_el = elem.find("year")
        year    = _parse_year(year_el.text if year_el is not None else "")
        if year is None or year < MIN_YEAR:
            stats["skipped_year"] += 1
            elem.clear()
            continue

        # --- title ---
        title_el = elem.find("title")
        title    = _itertext(title_el) if title_el is not None else ""
        if not title:
            stats["skipped_fields"] += 1
            elem.clear()
            continue

        # --- authors ---
        authors = [_itertext(a) for a in elem.findall("author")]
        authors = [a for a in authors if a]
        if not authors:
            stats["skipped_fields"] += 1
            elem.clear()
            continue

        # --- venue ---
        venue_tag = "journal" if elem.tag == "article" else "booktitle"
        venue_el  = elem.find(venue_tag)
        venue     = (venue_el.text or "").strip() if venue_el is not None else ""

        # --- key ---
        key = elem.get("key", "").strip()
        if not key:
            elem.clear()
            continue

        stats["accepted"] += 1

        if stats["accepted"] % LOG_PROGRESS_EVERY == 0:
            log.info(
                "  parsing: %d accepted, %d seen so far...",
                stats["accepted"],
                stats["seen"],
            )

        yield {
            "key":     key,
            "title":   title,
            "year":    year,
            "venue":   venue,
            "type":    elem.tag,
            "authors": authors,
        }

        elem.clear()

    log.info(
        "XML parse done | seen: %d | accepted: %d | "
        "skipped (year < %d or missing): %d | skipped (missing fields): %d",
        stats["seen"],
        stats["accepted"],
        MIN_YEAR,
        stats["skipped_year"],
        stats["skipped_fields"],
    )