"""
index_store.py — High-level API for building and querying the metadata B+ tree.

The primary entry points are:

  build_metadata_index(metadata_list, index_path)
      Build a fresh B+ tree from the list of metadata dicts produced by
      index_builder.py.  Inserts are done in chunk_id order to minimise
      unnecessary page splits.

  lookup_metadata(index_path, chunk_id) -> dict | None
      Open the tree, look up a single chunk_id, and close.

  lookup_metadata_batch(index_path, chunk_ids) -> dict[int, dict]
      Look up multiple chunk_ids in a single open/close cycle.

  load_all_metadata(index_path) -> list[dict]
      Traverse the full leaf chain and return every metadata record as a list
      indexed by chunk_id (same shape as the old pickle list).
"""

import os
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
