"""
query_filter.py — Regex-based constraint extractor for B+ tree range scans.

Exported functions:

  extract_constraints(query) -> dict
      Parse page-range and chapter references from natural language.
      Returns a dict with optional "pages" (lo, hi) and "chapter" (int) keys.

  get_chunk_allowlist(query, page_index_path, chapter_index_path) -> set | None
      Combine extracted constraints with B+ tree range scans to produce an
      allowlist of chunk_ids.  Returns None when no spatial constraints are
      found in the query.
"""

import re
from typing import Optional

from src.bptree import range_scan_chapter, range_scan_pages


# ── constraint patterns ───────────────────────────────────────────────────────

# Matches "pages 10-20", "pages 10 to 20", "pages 10 through 20", "page 10–20"
_PAGE_RANGE = re.compile(
    r"pages?\s+(\d+)\s*(?:-{1,2}|–|—|to|through)\s*(\d+)",
    re.IGNORECASE,
)
# Matches "page 42" (single page, no range)
_SINGLE_PAGE = re.compile(r"\bpages?\s+(\d+)\b", re.IGNORECASE)

# Matches "chapter 3", "chapter 3.1" (only captures the leading integer)
_CHAPTER = re.compile(r"\bchapter\s+(\d+)", re.IGNORECASE)


def extract_constraints(query: str) -> dict:
    """
    Parse spatial constraints from *query*.

    Returns a dict with zero or more of:
      "pages":   (low_page: int, high_page: int)
      "chapter": int
    """
    constraints: dict = {}

    m = _PAGE_RANGE.search(query)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        constraints["pages"] = (min(lo, hi), max(lo, hi))
    else:
        m = _SINGLE_PAGE.search(query)
        if m:
            page = int(m.group(1))
            constraints["pages"] = (page, page)

    m = _CHAPTER.search(query)
    if m:
        constraints["chapter"] = int(m.group(1))

    return constraints


def get_chunk_allowlist(
    query: str,
    page_index_path: str,
    chapter_index_path: str,
) -> Optional[set]:
    """
    Return a set of chunk_ids matching the spatial constraints in *query*, or
    None when the query contains no page/chapter references.

    When both a page range and a chapter are specified the two sets are
    *intersected* — a chunk must satisfy both constraints to be included.
    """
    import os

    constraints = extract_constraints(query)
    if not constraints:
        return None

    page_ids: Optional[set] = None
    chapter_ids: Optional[set] = None

    if "pages" in constraints:
        lo, hi = constraints["pages"]
        if os.path.exists(page_index_path):
            page_ids = range_scan_pages(page_index_path, lo, hi)

    if "chapter" in constraints:
        chapter = constraints["chapter"]
        if os.path.exists(chapter_index_path):
            chapter_ids = range_scan_chapter(chapter_index_path, chapter)

    # Combine results: intersect when both constraints present, otherwise use whichever exists.
    if page_ids is not None and chapter_ids is not None:
        allowlist = page_ids & chapter_ids
    elif page_ids is not None:
        allowlist = page_ids
    elif chapter_ids is not None:
        allowlist = chapter_ids
    else:
        return None

    return allowlist if allowlist else None
