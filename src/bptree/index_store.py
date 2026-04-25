"""
index_store.py — High-level API for building and querying the metadata B+ tree.

Primary tree (keyed by chunk_id):

  build_metadata_index(metadata_list, index_path)
  lookup_metadata(index_path, chunk_id) -> dict | None
  lookup_metadata_batch(index_path, chunk_ids) -> dict[int, dict]
  load_all_metadata(index_path) -> list[dict]

Secondary trees (keyed by page number or chapter number; values are lists of
chunk_ids):

  build_secondary_indexes(metadata_list, page_index_path, chapter_index_path)
  range_scan_pages(index_path, low_page, high_page) -> set[int]
  range_scan_chapter(index_path, chapter) -> set[int]
"""

import collections
import os
import re
from typing import Optional

from .bptree import BPlusTree


def build_metadata_index(metadata_list: list, index_path: str) -> None:
    """
    Build a disk-backed B+ tree from *metadata_list*.

    Each element must contain a ``"chunk_id"`` integer key.
    The tree file is (re)created from scratch at *index_path*.
    """
    # Remove stale index if it exists so we start with a clean file.
    if os.path.exists(index_path):
        os.remove(index_path)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Sort by chunk_id so inserts are sequential — fewer splits and a
    # shallower, more balanced tree.
    sorted_meta = sorted(metadata_list, key=lambda m: m["chunk_id"])

    with BPlusTree(index_path) as tree:
        for meta in sorted_meta:
            tree.insert(meta["chunk_id"], meta)


def lookup_metadata(index_path: str, chunk_id: int) -> Optional[dict]:
    """Return the metadata dict for *chunk_id*, or *None* if not found."""
    with BPlusTree(index_path) as tree:
        return tree.lookup(chunk_id)


def lookup_metadata_batch(index_path: str, chunk_ids: list) -> dict:
    """
    Return ``{chunk_id: metadata_dict}`` for every *chunk_id* that exists in
    the tree.  Uses a single file open/close for efficiency.
    """
    results: dict = {}
    with BPlusTree(index_path) as tree:
        for chunk_id in chunk_ids:
            meta = tree.lookup(chunk_id)
            if meta is not None:
                results[chunk_id] = meta
    return results


def load_all_metadata(index_path: str) -> list:
    """
    Load every metadata record from the B+ tree and return a positional list
    that is a drop-in replacement for ``pickle.load(meta_file)``.

    Records are returned in ascending chunk_id order, which matches the order
    they were appended to the metadata list during indexing.  FAISS uses
    positional indices (0, 1, 2, …) that map to the same positions in this
    list, so ``result[i]`` gives the metadata for FAISS index ``i``.

    Note: introduction sections are excluded from both the B+ tree and the
    pickle file, so chunk_ids may not start at 0.  The positional ordering
    is preserved in both cases.
    """
    with BPlusTree(index_path) as tree:
        pairs = tree.get_all_records()   # sorted (chunk_id, dict) pairs

    return [meta for _, meta in pairs]


# ── secondary index helpers ───────────────────────────────────────────────────

def _extract_chapter(section_path: str) -> Optional[int]:
    """Return the chapter number embedded in *section_path*, or None."""
    m = re.match(r"chapter\s+(\d+)", section_path.lower())
    return int(m.group(1)) if m else None


def build_secondary_indexes(
    metadata_list: list,
    page_index_path: str,
    chapter_index_path: str,
) -> None:
    """
    Build two secondary B+ trees from *metadata_list*:

    * ``page_index_path``    — key = page_number (int),  value = sorted list of chunk_ids
    * ``chapter_index_path`` — key = chapter_number (int), value = sorted list of chunk_ids

    Both files are recreated from scratch on every call.
    """
    page_to_chunks: dict = collections.defaultdict(set)
    chapter_to_chunks: dict = collections.defaultdict(set)

    for meta in metadata_list:
        chunk_id = meta["chunk_id"]
        for page_num in meta.get("page_numbers", []):
            page_to_chunks[page_num].add(chunk_id)
        chapter = _extract_chapter(meta.get("section_path", ""))
        if chapter is not None:
            chapter_to_chunks[chapter].add(chunk_id)

    for path, mapping in (
        (page_index_path, page_to_chunks),
        (chapter_index_path, chapter_to_chunks),
    ):
        if os.path.exists(path):
            os.remove(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with BPlusTree(path) as tree:
            for key in sorted(mapping.keys()):
                tree.insert(key, sorted(mapping[key]))


def range_scan_pages(index_path: str, low_page: int, high_page: int) -> set:
    """
    Return the set of chunk_ids whose ``page_numbers`` overlap with
    the range ``[low_page, high_page]`` (inclusive).
    """
    with BPlusTree(index_path) as tree:
        pairs = tree.range_scan(low_page, high_page)
    chunk_ids: set = set()
    for _, cids in pairs:
        chunk_ids.update(cids)
    return chunk_ids


def range_scan_chapter(index_path: str, chapter: int) -> set:
    """Return the set of chunk_ids that belong to *chapter*."""
    with BPlusTree(index_path) as tree:
        pairs = tree.range_scan(chapter, chapter)
    chunk_ids: set = set()
    for _, cids in pairs:
        chunk_ids.update(cids)
    return chunk_ids
