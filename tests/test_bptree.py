"""
test_bptree.py — Unit tests for the disk-backed B+ tree.

Run with:
    pytest tests/test_bptree.py -v
"""

import os
import random
import tempfile

import pytest

from src.bptree import BPlusTree, build_metadata_index, load_all_metadata, lookup_metadata
from src.bptree.page import (
    PAGE_SIZE,
    HEADER_SIZE,
    LEAF_REC_HDR_SIZE,
    INT_MAX_SLOTS,
    encode_leaf_page,
    decode_leaf_page,
    encode_internal_page,
    decode_internal_page,
    leaf_free_bytes,
    PageType,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary B+ tree file."""
    return str(tmp_path / "test.db")


def _make_meta(chunk_id: int, extra: dict | None = None) -> dict:
    """Build a minimal metadata dict matching TokenSmith's schema."""
    base = {
        "chunk_id":     chunk_id,
        "filename":     "data/test.md",
        "section":      f"Section {chunk_id}",
        "section_path": f"Chapter 1 Section {chunk_id}",
        "char_len":     500 + chunk_id,
        "word_len":     80 + chunk_id,
        "text_preview": f"Preview of chunk {chunk_id}" * 2,
        "page_numbers": [chunk_id + 1, chunk_id + 2],
    }
    if extra:
        base.update(extra)
    return base


# ── page encoding round-trip ──────────────────────────────────────────────────

class TestPageEncoding:

    def test_leaf_page_round_trip(self):
        records = [(0, b'{"a":1}'), (5, b'{"b":2}'), (10, b'{"c":3}')]
        data = encode_leaf_page(
            is_root=True, parent_id=-1, next_id=3, prev_id=-1, records=records
        )
        assert len(data) == PAGE_SIZE
        decoded = decode_leaf_page(data)
        assert decoded["page_type"] == PageType.LEAF
        assert decoded["is_root"] is True
        assert decoded["parent_id"] == -1
        assert decoded["next_id"] == 3
        assert decoded["prev_id"] == -1
        assert decoded["num_slots"] == 3
        assert decoded["records"] == records

    def test_internal_page_round_trip(self):
        keys           = [10, 20, 30]
        right_children = [2, 3, 4]
        data = encode_internal_page(
            is_root=False, parent_id=99,
            leftmost_child=1,
            keys=keys, right_children=right_children,
        )
        assert len(data) == PAGE_SIZE
        decoded = decode_internal_page(data)
        assert decoded["page_type"] == PageType.INTERNAL
        assert decoded["is_root"] is False
        assert decoded["parent_id"] == 99
        assert decoded["leftmost_child"] == 1
        assert decoded["keys"] == keys
        assert decoded["right_children"] == right_children

    def test_leaf_free_bytes_decreases_with_records(self):
        records: list = []
        full = leaf_free_bytes(records)
        records.append((1, b"x" * 100))
        assert leaf_free_bytes(records) == full - LEAF_REC_HDR_SIZE - 100

    def test_empty_leaf_page(self):
        data    = encode_leaf_page(is_root=True, parent_id=-1, next_id=-1, prev_id=-1, records=[])
        decoded = decode_leaf_page(data)
        assert decoded["records"] == []
        assert decoded["num_slots"] == 0

    def test_internal_page_zero_slots(self):
        data    = encode_internal_page(is_root=True, parent_id=-1, leftmost_child=1, keys=[], right_children=[])
        decoded = decode_internal_page(data)
        assert decoded["keys"] == []
        assert decoded["right_children"] == []


# ── BPlusTree: single insert & lookup ────────────────────────────────────────

class TestBPlusTreeBasic:

    def test_lookup_empty_tree(self, tmp_db):
        with BPlusTree(tmp_db) as tree:
            assert tree.lookup(0) is None

    def test_single_insert_lookup(self, tmp_db):
        meta = _make_meta(42)
        with BPlusTree(tmp_db) as tree:
            tree.insert(42, meta)
            assert tree.lookup(42) == meta

    def test_lookup_missing_key(self, tmp_db):
        with BPlusTree(tmp_db) as tree:
            tree.insert(1, _make_meta(1))
            assert tree.lookup(999) is None

    def test_update_overwrites_in_place(self, tmp_db):
        with BPlusTree(tmp_db) as tree:
            tree.insert(7, _make_meta(7))
            updated = _make_meta(7, {"char_len": 9999})
            tree.insert(7, updated)
            assert tree.lookup(7)["char_len"] == 9999

    def test_persistence_across_open(self, tmp_db):
        """Data written in one session must be readable in a fresh session."""
        meta = _make_meta(55)
        with BPlusTree(tmp_db) as tree:
            tree.insert(55, meta)
        # Re-open from scratch
        with BPlusTree(tmp_db) as tree:
            assert tree.lookup(55) == meta

    def test_multiple_inserts_sequential_lookup(self, tmp_db):
        metas = [_make_meta(i) for i in range(20)]
        with BPlusTree(tmp_db) as tree:
            for m in metas:
                tree.insert(m["chunk_id"], m)
            for m in metas:
                assert tree.lookup(m["chunk_id"]) == m


# ── node splitting ────────────────────────────────────────────────────────────

class TestNodeSplitting:

    def _insert_n(self, tree, n, order="sequential"):
        """Insert *n* records, optionally in random order."""
        ids = list(range(n))
        if order == "random":
            random.seed(0)
            random.shuffle(ids)
        for i in ids:
            tree.insert(i, _make_meta(i))
        return ids

    def test_leaf_split_all_keys_accessible(self, tmp_db):
        """Insert enough records to force at least one leaf split."""
        # Each record ≈ 10 (header) + ~200 bytes (JSON), so ~210 bytes.
        # PAGE_SIZE=4096, body=4080 → ~19 records per leaf before split.
        n = 40
        with BPlusTree(tmp_db) as tree:
            self._insert_n(tree, n)
            for i in range(n):
                result = tree.lookup(i)
                assert result is not None, f"chunk_id={i} not found after leaf split"
                assert result["chunk_id"] == i

    def test_many_inserts_random_order(self, tmp_db):
        n = 200
        with BPlusTree(tmp_db) as tree:
            self._insert_n(tree, n, order="random")
            for i in range(n):
                assert tree.lookup(i) is not None, f"chunk_id={i} missing"

    def test_internal_node_split(self, tmp_db):
        """
        Insert enough records to force internal-node splits.
        INT_MAX_SLOTS ≈ 339, so we need > 339 leaf pages ≈ 339 * ~19 ≈ 6000
        records.  Use 500 here: enough for a multi-level tree.
        """
        n = 500
        with BPlusTree(tmp_db) as tree:
            self._insert_n(tree, n)
            for i in range(n):
                assert tree.lookup(i) is not None, f"chunk_id={i} missing after internal split"

    def test_tree_height_increases_with_inserts(self, tmp_db):
        """After enough inserts the root should be an internal node."""
        n = 50
        with BPlusTree(tmp_db) as tree:
            self._insert_n(tree, n)
            root_data = tree._read_page(tree._root_id)
            # With 50 records the tree must have grown beyond a single leaf.
            assert PageType(root_data[0]) == PageType.INTERNAL


# ── serialisation durability ──────────────────────────────────────────────────

class TestSerialisation:

    def test_file_has_magic_bytes(self, tmp_db):
        with BPlusTree(tmp_db) as tree:
            tree.insert(0, _make_meta(0))
        with open(tmp_db, "rb") as f:
            magic = f.read(8)
        assert magic == b"BPTREE01"

    def test_file_size_is_multiple_of_page_size(self, tmp_db):
        with BPlusTree(tmp_db) as tree:
            for i in range(30):
                tree.insert(i, _make_meta(i))
        assert os.path.getsize(tmp_db) % PAGE_SIZE == 0

    def test_reopen_after_many_inserts(self, tmp_db):
        n = 100
        metas = {i: _make_meta(i) for i in range(n)}
        with BPlusTree(tmp_db) as tree:
            for i, m in metas.items():
                tree.insert(i, m)
        # Re-open and verify every record.
        with BPlusTree(tmp_db) as tree:
            for i, expected in metas.items():
                assert tree.lookup(i) == expected, f"Mismatch for chunk_id={i}"

    def test_get_all_records_returns_sorted_pairs(self, tmp_db):
        ids = [5, 2, 8, 1, 9, 3]
        with BPlusTree(tmp_db) as tree:
            for i in ids:
                tree.insert(i, _make_meta(i))
            pairs = tree.get_all_records()
        retrieved_ids = [cid for cid, _ in pairs]
        assert retrieved_ids == sorted(ids)

    def test_get_all_records_values_correct(self, tmp_db):
        metas = {i: _make_meta(i) for i in range(25)}
        with BPlusTree(tmp_db) as tree:
            for i, m in metas.items():
                tree.insert(i, m)
            pairs = tree.get_all_records()
        for cid, meta in pairs:
            assert meta == metas[cid]


# ── high-level API (index_store) ──────────────────────────────────────────────

class TestIndexStore:

    def test_build_and_lookup(self, tmp_path):
        path   = str(tmp_path / "meta.db")
        n      = 50
        metas  = [_make_meta(i) for i in range(n)]
        build_metadata_index(metas, path)
        for m in metas:
            assert lookup_metadata(path, m["chunk_id"]) == m

    def test_build_overwrites_existing(self, tmp_path):
        path  = str(tmp_path / "meta.db")
        build_metadata_index([_make_meta(0)], path)
        # Build again with a different record set.
        build_metadata_index([_make_meta(99)], path)
        assert lookup_metadata(path, 99) is not None
        assert lookup_metadata(path, 0) is None   # old data gone

    def test_load_all_metadata_indexed_by_chunk_id(self, tmp_path):
        path  = str(tmp_path / "meta.db")
        n     = 30
        metas = [_make_meta(i) for i in range(n)]
        build_metadata_index(metas, path)
        loaded = load_all_metadata(path)
        assert len(loaded) == n
        for i in range(n):
            assert loaded[i] == metas[i]

    def test_load_all_metadata_matches_pickle_shape(self, tmp_path):
        """Verify load_all_metadata returns a list compatible with retriever usage."""
        path  = str(tmp_path / "meta.db")
        metas = [_make_meta(i) for i in range(10)]
        build_metadata_index(metas, path)
        loaded = load_all_metadata(path)
        # Spot-check page_numbers lookup as done by get_page_numbers()
        chunk_pages = loaded[3].get("page_numbers")
        assert isinstance(chunk_pages, list)
        assert 4 in chunk_pages   # _make_meta(3) sets page_numbers=[4, 5]

    def test_build_large_index(self, tmp_path):
        """Simulate the full 1744-chunk dataset size."""
        path  = str(tmp_path / "meta.db")
        n     = 1744
        metas = [_make_meta(i) for i in range(n)]
        build_metadata_index(metas, path)
        # Spot-check a sample of chunk IDs.
        for i in random.sample(range(n), 50):
            result = lookup_metadata(path, i)
            assert result is not None
            assert result["chunk_id"] == i
