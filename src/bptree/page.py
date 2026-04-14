"""
page.py — Fixed-size page layout for the disk-backed B+ tree.

Every page is exactly PAGE_SIZE (4096) bytes, stored at offset
``page_id * PAGE_SIZE`` inside the tree file.

Page header (16 bytes, big-endian):
  page_type  : uint8   — 0 = internal node, 1 = leaf node
  is_root    : uint8   — 1 if this page is the current root, else 0
  num_slots  : uint16  — number of keys (internal) or records (leaf)
  parent_id  : int32   — page_id of parent, or -1 for the root
  next_id    : int32   — next leaf page in chain (leaf only), or -1
  prev_id    : int32   — previous leaf page in chain (leaf only), or -1

Internal-node body (immediately after header):
  leftmost_child : int32   — child pointer before the first key
  Per slot       : int64 key + int32 right_child_ptr  (12 bytes each)

  Max slots = (PAGE_SIZE - HEADER_SIZE - 4) // 12  ≈ 339 for 4 KB pages.

Leaf-node body (immediately after header):
  Records packed sequentially from the start of the body:
    rec_len  : uint16  — byte count of (chunk_id + data), NOT including
                         this 2-byte length field itself
    chunk_id : int64   — primary key (= 8 bytes of rec_len)
    data     : bytes   — JSON-encoded metadata (rec_len - 8 bytes)
"""

import struct
from enum import IntEnum


PAGE_SIZE = 4096

# ── page types ────────────────────────────────────────────────────────────────

class PageType(IntEnum):
    INTERNAL = 0
    LEAF     = 1


# ── header ────────────────────────────────────────────────────────────────────
# Format: big-endian | uint8 | uint8 | uint16 | int32 | int32 | int32
_HDR = struct.Struct(">BBHiii")
HEADER_SIZE = _HDR.size   # 16 bytes


# ── internal node ─────────────────────────────────────────────────────────────
_INT_LEFTMOST = struct.Struct(">i")          # leftmost child pointer (4 bytes)
_INT_SLOT     = struct.Struct(">qi")         # key(8) + right_child(4) = 12 bytes
INT_SLOT_SIZE = _INT_SLOT.size               # 12
# Maximum number of key-pointer slots per internal page:
INT_MAX_SLOTS = (PAGE_SIZE - HEADER_SIZE - _INT_LEFTMOST.size) // INT_SLOT_SIZE  # 339


# ── leaf node ─────────────────────────────────────────────────────────────────
# Record header stored before each value: uint16 rec_len + int64 chunk_id
_LEAF_REC_HDR  = struct.Struct(">Hq")        # rec_len(2) + chunk_id(8) = 10 bytes
LEAF_REC_HDR_SIZE = _LEAF_REC_HDR.size       # 10
LEAF_BODY_SIZE    = PAGE_SIZE - HEADER_SIZE  # 4080 bytes available for records


# ── header encode / decode ────────────────────────────────────────────────────

def encode_page_header(
    page_type: int,
    is_root: bool,
    num_slots: int,
    parent_id: int,
    next_id: int,
    prev_id: int,
) -> bytes:
    return _HDR.pack(page_type, int(is_root), num_slots, parent_id, next_id, prev_id)


def decode_page_header(data: bytes) -> dict:
    pt, is_root, num_slots, parent_id, next_id, prev_id = _HDR.unpack(
        data[:HEADER_SIZE]
    )
    return {
        "page_type": PageType(pt),
        "is_root":   bool(is_root),
        "num_slots": num_slots,
        "parent_id": parent_id,
        "next_id":   next_id,
        "prev_id":   prev_id,
    }


# ── internal node encode / decode ─────────────────────────────────────────────

def encode_internal_page(
    is_root: bool,
    parent_id: int,
    leftmost_child: int,
    keys: list,
    right_children: list,
) -> bytes:
    """
    Encode an internal node into PAGE_SIZE bytes.

    ``keys`` and ``right_children`` must be the same length N.
    The full child list is [leftmost_child, right_children[0], …, right_children[N-1]].
    """
    assert len(keys) == len(right_children), (
        "keys and right_children must have the same length"
    )
    assert len(keys) <= INT_MAX_SLOTS, (
        f"Too many slots for an internal page: {len(keys)} > {INT_MAX_SLOTS}"
    )

    buf = bytearray(PAGE_SIZE)
    buf[:HEADER_SIZE] = encode_page_header(
        PageType.INTERNAL, is_root, len(keys), parent_id, -1, -1
    )

    offset = HEADER_SIZE
    buf[offset : offset + 4] = _INT_LEFTMOST.pack(leftmost_child)
    offset += 4

    for key, right_child in zip(keys, right_children):
        buf[offset : offset + INT_SLOT_SIZE] = _INT_SLOT.pack(key, right_child)
        offset += INT_SLOT_SIZE

    return bytes(buf)


def decode_internal_page(data: bytes) -> dict:
    """Return a dict with header fields plus ``leftmost_child``, ``keys``, ``right_children``."""
    node = decode_page_header(data)
    offset = HEADER_SIZE

    (leftmost_child,) = _INT_LEFTMOST.unpack(data[offset : offset + 4])
    offset += 4

    keys:           list[int] = []
    right_children: list[int] = []
    for _ in range(node["num_slots"]):
        key, right_child = _INT_SLOT.unpack(data[offset : offset + INT_SLOT_SIZE])
        keys.append(key)
        right_children.append(right_child)
        offset += INT_SLOT_SIZE

    node["leftmost_child"] = leftmost_child
    node["keys"]           = keys
    node["right_children"] = right_children
    return node


# ── leaf node encode / decode ─────────────────────────────────────────────────

def encode_leaf_page(
    is_root: bool,
    parent_id: int,
    next_id: int,
    prev_id: int,
    records: list,          # list of (chunk_id: int, meta_bytes: bytes)
) -> bytes:
    """
    Encode a leaf node into PAGE_SIZE bytes.

    ``records`` is a list of ``(chunk_id, metadata_bytes)`` pairs sorted by
    chunk_id.  Callers must ensure the records fit within LEAF_BODY_SIZE.
    """
    buf = bytearray(PAGE_SIZE)
    buf[:HEADER_SIZE] = encode_page_header(
        PageType.LEAF, is_root, len(records), parent_id, next_id, prev_id
    )

    offset = HEADER_SIZE
    for chunk_id, meta_bytes in records:
        rec_len = 8 + len(meta_bytes)          # chunk_id(8) + data
        rec_hdr = _LEAF_REC_HDR.pack(rec_len, chunk_id)
        buf[offset : offset + LEAF_REC_HDR_SIZE] = rec_hdr
        offset += LEAF_REC_HDR_SIZE
        buf[offset : offset + len(meta_bytes)] = meta_bytes
        offset += len(meta_bytes)

    return bytes(buf)


def decode_leaf_page(data: bytes) -> dict:
    """Return a dict with header fields plus ``records`` list of (chunk_id, bytes)."""
    leaf    = decode_page_header(data)
    offset  = HEADER_SIZE
    records = []

    for _ in range(leaf["num_slots"]):
        rec_len, chunk_id = _LEAF_REC_HDR.unpack(
            data[offset : offset + LEAF_REC_HDR_SIZE]
        )
        offset   += LEAF_REC_HDR_SIZE
        data_len  = rec_len - 8
        meta_bytes = data[offset : offset + data_len]
        offset   += data_len
        records.append((chunk_id, meta_bytes))

    leaf["records"] = records
    return leaf


# ── free-space helper ─────────────────────────────────────────────────────────

def leaf_free_bytes(records: list) -> int:
    """Return unused bytes in a leaf page given its current records."""
    used = HEADER_SIZE
    for _, meta_bytes in records:
        used += LEAF_REC_HDR_SIZE + len(meta_bytes)
    return PAGE_SIZE - used
