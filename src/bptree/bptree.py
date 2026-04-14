"""
bptree.py — Disk-backed B+ tree keyed by integer chunk_id.

File layout
-----------
Page 0   — file header  (magic, root_id, num_pages)
Page 1…N — tree pages   (internal nodes or leaf nodes)

Each page is exactly PAGE_SIZE bytes at offset ``page_id * PAGE_SIZE``.

Supported operations
--------------------
insert(chunk_id, metadata)  — O(log N) amortized; overwrites on duplicate key
lookup(chunk_id)            — O(log N); returns dict or None
get_all_records()           — O(N); full leaf-chain scan, returns sorted list
"""

import json
import os
import struct
from typing import Optional

from .page import (
    PAGE_SIZE,
    PageType,
    INT_MAX_SLOTS,
    LEAF_REC_HDR_SIZE,
    encode_internal_page,
    decode_internal_page,
    encode_leaf_page,
    decode_leaf_page,
    leaf_free_bytes,
)

# ── file header ───────────────────────────────────────────────────────────────
# Stored in page 0: magic(8) + root_id(4) + num_pages(4) = 16 bytes,
# rest of the page is zero-padded.
_FILE_HDR = struct.Struct(">8sii")
_FILE_HDR_SIZE = _FILE_HDR.size   # 16
MAGIC = b"BPTREE01"


class BPlusTree:
    """
    Disk-backed B+ tree.

    Usage::

        with BPlusTree("path/to/index.db") as tree:
            tree.insert(42, {"chunk_id": 42, "page_numbers": [5, 6]})
            meta = tree.lookup(42)
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if os.path.exists(path):
            self._fh = open(path, "r+b")
            raw = self._fh.read(_FILE_HDR_SIZE)
            magic, root_id, num_pages = _FILE_HDR.unpack(raw)
            if magic != MAGIC:
                raise ValueError(f"Not a valid B+ tree file (bad magic): {path}")
            self._root_id   = root_id
            self._num_pages = num_pages
        else:
            self._fh = open(path, "w+b")
            # Page 0 is the file header; tree pages start at 1.
            self._num_pages = 1
            self._root_id   = -1    # empty tree
            self._write_file_header()

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # ── public API ────────────────────────────────────────────────────────────

    def insert(self, chunk_id: int, metadata: dict) -> None:
        """
        Insert *metadata* under *chunk_id*.

        If *chunk_id* already exists the record is overwritten in-place when
        the new serialised size fits in the same leaf page, otherwise the old
        record is removed and the new one is re-inserted (rare for fixed-ish
        metadata sizes).
        """
        meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        if self._root_id == -1:
            # Empty tree: allocate the first leaf page as the root.
            root_id = self._allocate_page()
            self._write_page(
                root_id,
                encode_leaf_page(
                    is_root=True, parent_id=-1, next_id=-1, prev_id=-1,
                    records=[(chunk_id, meta_bytes)],
                ),
            )
            self._root_id = root_id
            self._write_file_header()
        else:
            leaf_id = self._find_leaf(chunk_id)
            self._insert_into_leaf(leaf_id, chunk_id, meta_bytes)

        self._fh.flush()

    def lookup(self, chunk_id: int) -> Optional[dict]:
        """Return the metadata dict for *chunk_id*, or *None* if not found."""
        if self._root_id == -1:
            return None
        leaf_id = self._find_leaf(chunk_id)
        leaf    = decode_leaf_page(self._read_page(leaf_id))
        for cid, meta_bytes in leaf["records"]:
            if cid == chunk_id:
                return json.loads(meta_bytes.decode("utf-8"))
        return None

    def get_all_records(self) -> list:
        """
        Return all ``(chunk_id, metadata_dict)`` pairs in ascending key order
        by following the leaf-page chain from the leftmost leaf.

        Complexity: O(N) page reads.
        """
        if self._root_id == -1:
            return []

        # Navigate to the leftmost leaf.
        page_id = self._root_id
        while True:
            data = self._read_page(page_id)
            if PageType(data[0]) == PageType.LEAF:
                break
            node    = decode_internal_page(data)
            page_id = node["leftmost_child"]

        # Walk the leaf chain.
        results = []
        while page_id != -1:
            leaf = decode_leaf_page(self._read_page(page_id))
            for cid, meta_bytes in leaf["records"]:
                results.append((cid, json.loads(meta_bytes.decode("utf-8"))))
            page_id = leaf["next_id"]

        return results

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _read_page(self, page_id: int) -> bytes:
        self._fh.seek(page_id * PAGE_SIZE)
        return self._fh.read(PAGE_SIZE)

    def _write_page(self, page_id: int, data: bytes) -> None:
        assert len(data) == PAGE_SIZE, (
            f"Page data must be exactly {PAGE_SIZE} bytes, got {len(data)}"
        )
        self._fh.seek(page_id * PAGE_SIZE)
        self._fh.write(data)

    def _allocate_page(self) -> int:
        """Reserve a new page ID, extend the file, and return the new ID."""
        page_id = self._num_pages
        self._num_pages += 1
        # Extend the file with a zeroed page so seek-writes later are valid.
        self._fh.seek(page_id * PAGE_SIZE)
        self._fh.write(b"\x00" * PAGE_SIZE)
        self._write_file_header()
        return page_id

    def _write_file_header(self) -> None:
        self._fh.seek(0)
        raw = _FILE_HDR.pack(MAGIC, self._root_id, self._num_pages)
        # Pad the rest of page 0 with zeros.
        self._fh.write(raw + b"\x00" * (PAGE_SIZE - _FILE_HDR_SIZE))

    # ── tree traversal ────────────────────────────────────────────────────────

    def _find_leaf(self, chunk_id: int) -> int:
        """
        Descend from the root to the leaf page where *chunk_id* belongs.
        Returns the page_id of that leaf.
        """
        page_id = self._root_id
        while True:
            data = self._read_page(page_id)
            if PageType(data[0]) == PageType.LEAF:
                return page_id

            node  = decode_internal_page(data)
            child = node["leftmost_child"]
            for key, right_child in zip(node["keys"], node["right_children"]):
                if chunk_id < key:
                    break
                child = right_child
            page_id = child

    # ── leaf-level insert ─────────────────────────────────────────────────────

    def _insert_into_leaf(self, leaf_id: int, chunk_id: int, meta_bytes: bytes) -> None:
        leaf    = decode_leaf_page(self._read_page(leaf_id))
        records = list(leaf["records"])   # copy; do not mutate the decoded dict

        # Overwrite if key already exists.
        for i, (cid, _) in enumerate(records):
            if cid == chunk_id:
                records[i] = (chunk_id, meta_bytes)
                self._write_page(
                    leaf_id,
                    encode_leaf_page(
                        is_root=leaf["is_root"],
                        parent_id=leaf["parent_id"],
                        next_id=leaf["next_id"],
                        prev_id=leaf["prev_id"],
                        records=records,
                    ),
                )
                return

        # Compute free space before the new record is added.
        free     = leaf_free_bytes(records)
        required = LEAF_REC_HDR_SIZE + len(meta_bytes)

        # Insert in sorted order.
        records.append((chunk_id, meta_bytes))
        records.sort(key=lambda r: r[0])

        if required <= free:
            self._write_page(
                leaf_id,
                encode_leaf_page(
                    is_root=leaf["is_root"],
                    parent_id=leaf["parent_id"],
                    next_id=leaf["next_id"],
                    prev_id=leaf["prev_id"],
                    records=records,
                ),
            )
        else:
            self._split_leaf(leaf_id, leaf, records)

    # ── leaf split ────────────────────────────────────────────────────────────

    def _split_leaf(self, leaf_id: int, old_leaf: dict, all_records: list) -> None:
        """
        Split a full leaf into two half-full leaves and push the first key of
        the right page up to the parent.
        """
        mid          = len(all_records) // 2
        left_records = all_records[:mid]
        right_records = all_records[mid:]
        promote_key  = right_records[0][0]   # copy-up semantics

        right_id = self._allocate_page()

        if old_leaf["is_root"]:
            # The leaf was the only page; create a new internal root above it.
            new_root_id = self._allocate_page()
            parent_id   = new_root_id

            self._write_page(
                leaf_id,
                encode_leaf_page(
                    is_root=False, parent_id=parent_id,
                    next_id=right_id, prev_id=-1,
                    records=left_records,
                ),
            )
            self._write_page(
                right_id,
                encode_leaf_page(
                    is_root=False, parent_id=parent_id,
                    next_id=-1, prev_id=leaf_id,
                    records=right_records,
                ),
            )
            self._write_page(
                new_root_id,
                encode_internal_page(
                    is_root=True, parent_id=-1,
                    leftmost_child=leaf_id,
                    keys=[promote_key], right_children=[right_id],
                ),
            )
            self._root_id = new_root_id
            self._write_file_header()

        else:
            parent_id = old_leaf["parent_id"]

            self._write_page(
                leaf_id,
                encode_leaf_page(
                    is_root=False, parent_id=parent_id,
                    next_id=right_id, prev_id=old_leaf["prev_id"],
                    records=left_records,
                ),
            )
            self._write_page(
                right_id,
                encode_leaf_page(
                    is_root=False, parent_id=parent_id,
                    next_id=old_leaf["next_id"], prev_id=leaf_id,
                    records=right_records,
                ),
            )

            # Patch the prev pointer of the page that used to follow old_leaf.
            if old_leaf["next_id"] != -1:
                self._update_prev_ptr(old_leaf["next_id"], right_id)

            # Propagate the promoted key up to the parent.
            self._insert_into_internal(parent_id, promote_key, right_id)

    # ── internal-node insert ──────────────────────────────────────────────────

    def _insert_into_internal(
        self, internal_id: int, key: int, right_child: int
    ) -> None:
        """Insert *(key, right_child)* into the internal node at *internal_id*."""
        node           = decode_internal_page(self._read_page(internal_id))
        keys           = list(node["keys"])
        right_children = list(node["right_children"])

        # Find insertion position (maintain sorted order).
        i = 0
        while i < len(keys) and keys[i] < key:
            i += 1
        keys.insert(i, key)
        right_children.insert(i, right_child)

        if len(keys) <= INT_MAX_SLOTS:
            self._write_page(
                internal_id,
                encode_internal_page(
                    is_root=node["is_root"],
                    parent_id=node["parent_id"],
                    leftmost_child=node["leftmost_child"],
                    keys=keys,
                    right_children=right_children,
                ),
            )
            # Keep the new right child's parent pointer correct.
            self._update_child_parent(right_child, internal_id)
        else:
            self._split_internal(internal_id, node, keys, right_children)

    # ── internal-node split ───────────────────────────────────────────────────

    def _split_internal(
        self, node_id: int, old_node: dict, keys: list, right_children: list
    ) -> None:
        """
        Split an overfull internal node.

        The median key is *pushed up* (removed from both children, unlike leaf
        copy-up).  The left node keeps keys[:mid], the right node keeps
        keys[mid+1:], and keys[mid] is inserted into the parent.
        """
        mid         = len(keys) // 2
        promote_key = keys[mid]

        left_keys      = keys[:mid]
        left_rchildren = right_children[:mid]

        right_leftmost  = right_children[mid]
        right_keys      = keys[mid + 1:]
        right_rchildren = right_children[mid + 1:]

        right_id = self._allocate_page()

        if old_node["is_root"]:
            new_root_id = self._allocate_page()

            self._write_page(
                node_id,
                encode_internal_page(
                    is_root=False, parent_id=new_root_id,
                    leftmost_child=old_node["leftmost_child"],
                    keys=left_keys, right_children=left_rchildren,
                ),
            )
            self._write_page(
                right_id,
                encode_internal_page(
                    is_root=False, parent_id=new_root_id,
                    leftmost_child=right_leftmost,
                    keys=right_keys, right_children=right_rchildren,
                ),
            )
            self._write_page(
                new_root_id,
                encode_internal_page(
                    is_root=True, parent_id=-1,
                    leftmost_child=node_id,
                    keys=[promote_key], right_children=[right_id],
                ),
            )
            self._root_id = new_root_id
            self._write_file_header()

            # Re-parent all children of the new right internal node.
            for child_id in [right_leftmost] + right_rchildren:
                self._update_child_parent(child_id, right_id)

        else:
            self._write_page(
                node_id,
                encode_internal_page(
                    is_root=False, parent_id=old_node["parent_id"],
                    leftmost_child=old_node["leftmost_child"],
                    keys=left_keys, right_children=left_rchildren,
                ),
            )
            self._write_page(
                right_id,
                encode_internal_page(
                    is_root=False, parent_id=old_node["parent_id"],
                    leftmost_child=right_leftmost,
                    keys=right_keys, right_children=right_rchildren,
                ),
            )

            # Re-parent all children of the new right internal node.
            for child_id in [right_leftmost] + right_rchildren:
                self._update_child_parent(child_id, right_id)

            # Propagate the promoted key to the grandparent.
            self._insert_into_internal(old_node["parent_id"], promote_key, right_id)

    # ── pointer-maintenance helpers ───────────────────────────────────────────

    def _update_child_parent(self, child_id: int, parent_id: int) -> None:
        """Rewrite *child_id*'s page with an updated parent_id field."""
        data      = self._read_page(child_id)
        page_type = PageType(data[0])

        if page_type == PageType.LEAF:
            leaf     = decode_leaf_page(data)
            new_data = encode_leaf_page(
                is_root=leaf["is_root"],
                parent_id=parent_id,
                next_id=leaf["next_id"],
                prev_id=leaf["prev_id"],
                records=leaf["records"],
            )
        else:
            node     = decode_internal_page(data)
            new_data = encode_internal_page(
                is_root=node["is_root"],
                parent_id=parent_id,
                leftmost_child=node["leftmost_child"],
                keys=node["keys"],
                right_children=node["right_children"],
            )

        self._write_page(child_id, new_data)

    def _update_prev_ptr(self, leaf_id: int, new_prev: int) -> None:
        """Rewrite *leaf_id*'s page with an updated prev_id field."""
        leaf     = decode_leaf_page(self._read_page(leaf_id))
        new_data = encode_leaf_page(
            is_root=leaf["is_root"],
            parent_id=leaf["parent_id"],
            next_id=leaf["next_id"],
            prev_id=new_prev,
            records=leaf["records"],
        )
        self._write_page(leaf_id, new_data)
