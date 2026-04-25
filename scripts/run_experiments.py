"""
run_experiments.py — Measures filter precision, retrieval latency, and chapter
coherence for the B+ tree secondary index feature.

Experiments
-----------
  Exp 1 + 2 (PAGE QUERIES): For each page-range query, report how many of the
    top-k returned chunks actually fall within the requested range (precision),
    and how long retrieval took (latency).

  Exp 5 (CHAPTER QUERIES): For each chapter query, report how many of the
    top-k returned chunks come from the target chapter (coherence), and latency.

Usage (from the project root):
    python scripts/run_experiments.py

Prerequisites:
  - Run index mode first: python -m src.main index
    (with use_bptree: true and use_bptree_filter: true in config/config.yaml)
  - Secondary index files must exist: index/bptree/idx_page.db,
    index/bptree/idx_chapter.db
"""

import re
import sys
import time
import pathlib

# Ensure project root is on sys.path when run as a plain script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import faiss  # noqa: E402 — must come before other src imports

from src.config import RAGConfig
from src.retriever import (
    FAISSRetriever,
    BM25Retriever,
    filter_retrieved_chunks,
    load_artifacts,
)
from src.ranking.ranker import EnsembleRanker
from src.query_filter import extract_constraints, get_chunk_allowlist


# ── test queries ──────────────────────────────────────────────────────────────
# Each page query is (query_string, low_page, high_page).
# Adjust these to match real page numbers in your textbook.

PAGE_QUERIES = [
    ("What is discussed on pages 10 to 15?", 10, 15),
    ("Summarize the content on pages 7 to 9", 7,  9),
    ("What is covered on page 8?",            8,  8),
]

CHAPTER_QUERIES = [
    "What are the main topics in Chapter 4?",
    "Explain the key concepts in Chapter 5",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def retrieve(query, retrievers, ranker, chunks, cfg, use_filter):
    """
    Run the retrieval pipeline for *query*.

    Returns (topk_idxs, latency_seconds, allowlist_size).
    Latency covers scoring + ranking + filtering, but NOT LLM generation.
    """
    allowlist = None
    if use_filter:
        allowlist = get_chunk_allowlist(
            query, cfg.page_index_path, cfg.chapter_index_path
        )

    pool_n = len(chunks) if allowlist else max(cfg.num_candidates, cfg.top_k + 10)

    t0 = time.perf_counter()
    raw_scores = {}
    for r in retrievers:
        raw_scores[r.name] = r.get_scores(query, pool_n, chunks)
    ordered, _ = ranker.rank(raw_scores=raw_scores)
    topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered, allowlist)
    elapsed = time.perf_counter() - t0

    return topk_idxs, elapsed, (len(allowlist) if allowlist else None)


def page_precision(topk_idxs, meta, lo, hi):
    """Return (hits, total) where hits = chunks whose page_numbers overlap [lo,hi]."""
    hits = 0
    for idx in topk_idxs:
        pages = meta[idx].get("page_numbers", [])
        if any(lo <= p <= hi for p in pages):
            hits += 1
    return hits, len(topk_idxs)


def chapter_of(chunk_meta):
    sp = chunk_meta.get("section_path", "")
    m = re.match(r"chapter\s+(\d+)", sp.lower())
    return int(m.group(1)) if m else None


def chapter_coherence(topk_idxs, meta, target_chapter):
    """Return (hits, total, list_of_chapters_seen)."""
    hits = 0
    chapters_seen = []
    for idx in topk_idxs:
        ch = chapter_of(meta[idx])
        chapters_seen.append(ch)
        if ch == target_chapter:
            hits += 1
    return hits, len(topk_idxs), chapters_seen


def separator(char="=", width=72):
    print(char * width)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = RAGConfig.from_yaml("config/config.yaml")
    artifacts_dir = cfg.get_artifacts_directory()
    index_prefix  = "textbook_index"

    print("Loading artifacts (embedding model may take a moment)...")
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
        artifacts_dir, index_prefix,
        bptree_path=cfg.bptree_index_path if cfg.use_bptree else None,
    )
    retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    print(f"Loaded {len(chunks)} chunks, top_k={cfg.top_k}.\n")

    # ── Experiments 1 + 2: page queries ──────────────────────────────────────

    separator()
    print("EXPERIMENTS 1 + 2  |  Filter Precision & Retrieval Latency  |  Page Queries")
    separator()

    for query, lo, hi in PAGE_QUERIES:
        print(f"\nQuery : {query!r}")
        print(f"Target: pages {lo}–{hi}")

        rows = []
        for label, use_filter in [("WITHOUT filter", False), ("WITH filter", True)]:
            idxs, elapsed, al_size = retrieve(
                query, retrievers, ranker, chunks, cfg, use_filter
            )
            hits, total = page_precision(idxs, meta, lo, hi)
            rows.append((label, hits, total, elapsed, al_size))
            print(f"  {label:15s} | precision = {hits}/{total} "
                  f"| latency = {elapsed:.3f}s "
                  f"| allowlist = {al_size if al_size is not None else 'N/A'}")

        # Detail view for the WITH-filter run
        _, _, _, _, _ = rows[1]
        idxs_filtered, _, _ = retrieve(
            query, retrievers, ranker, chunks, cfg, use_filter=True
        )
        print("  Returned chunks (WITH filter):")
        for rank, idx in enumerate(idxs_filtered, 1):
            pages   = meta[idx].get("page_numbers", [])
            section = meta[idx].get("section", "?")[:50]
            in_range = "✓" if any(lo <= p <= hi for p in pages) else "✗"
            print(f"    [{rank:2d}] {in_range}  pages={pages!s:<12}  section={section!r}")

    # ── Experiment 5: chapter queries ─────────────────────────────────────────

    separator()
    print("\nEXPERIMENT 5  |  Chapter-Level Coherence")
    separator()

    for query in CHAPTER_QUERIES:
        constraints    = extract_constraints(query)
        target_chapter = constraints.get("chapter")
        if target_chapter is None:
            print(f"\n[skip] Could not parse chapter from: {query!r}")
            continue

        print(f"\nQuery : {query!r}")
        print(f"Target: Chapter {target_chapter}")

        for label, use_filter in [("WITHOUT filter", False), ("WITH filter", True)]:
            idxs, elapsed, al_size = retrieve(
                query, retrievers, ranker, chunks, cfg, use_filter
            )
            hits, total, chapters_seen = chapter_coherence(idxs, meta, target_chapter)
            print(f"  {label:15s} | coherence = {hits}/{total} "
                  f"| latency = {elapsed:.3f}s "
                  f"| chapters seen = {chapters_seen}")

    separator()
    print("Done.")


if __name__ == "__main__":
    main()
