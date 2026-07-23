"""
Memory-efficient preprocessing pipeline for the Amazon Electronics reviews +
metadata JSON dumps, built on Polars lazy evaluation.

    python preprocess.py
"""

import ast
import json
import sys
import time
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

REVIEWS_PATH = BASE_DIR / "reviews_Electronics.json"
META_PATH = BASE_DIR / "meta_Electronics.json"
META_CLEAN_PATH = BASE_DIR / "_meta_Electronics.clean.jsonl"  # auto-generated, cached across runs

FULL_OUTPUT_PATH = BASE_DIR / "processed_full_dataset.parquet"
SAMPLE_OUTPUT_PATH = BASE_DIR / "sample_reviews.parquet"

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42


def convert_meta_to_ndjson(src: Path, dst: Path) -> None:
    """
    The Amazon metadata dump is NOT valid JSON/NDJSON: every line is a
    *Python* dict literal (single-quoted strings, `None`/`True`/`False`
    tokens), e.g.:

        {'asin': '0132793040', 'title': 'Some Product', ...}

    `pl.scan_ndjson` (and `json.loads`) cannot parse that. We stream through
    the file one line at a time, use `ast.literal_eval` (safe -- it only
    evaluates literals, never arbitrary code) to parse each line, keep only
    `asin`/`title`, and write it back out as a proper JSON line. Memory use
    is O(1) per line, and as a side effect this shrinks 648MB (which also
    carries descriptions, images, related-product lists, etc.) down to a
    small file containing only the two columns we need, making the later
    lazy scan both correct and cheap.
    """
    print(f"[meta] Converting Python-literal dump -> valid NDJSON: {src.name} -> {dst.name}")
    n_ok, n_skipped = 0, 0
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                n_skipped += 1
                continue
            fout.write(
                json.dumps(
                    {
                        "asin": record.get("asin"),
                        "title": record.get("title"),
                    }
                )
            )
            fout.write("\n")
            n_ok += 1
    print(f"[meta] Done: {n_ok:,} records converted, {n_skipped:,} malformed lines skipped.")


def build_reviews_lazyframe() -> pl.LazyFrame:
    """Lazily scan the reviews NDJSON file and keep only the essential columns."""
    return (
        pl.scan_ndjson(REVIEWS_PATH, infer_schema_length=10_000, low_memory=True)
        .select(["asin", "overall", "reviewText"])
        .drop_nulls(subset=["asin", "overall", "reviewText"])
    )


def build_meta_lazyframe() -> pl.LazyFrame:
    """Lazily scan the cleaned metadata NDJSON file, deduplicated on asin."""
    return (
        pl.scan_ndjson(META_CLEAN_PATH, infer_schema_length=10_000, low_memory=True)
        .select(["asin", "title"])
        .unique(subset=["asin"], keep="first")
    )


def main() -> None:
    t_start = time.perf_counter()

    if not REVIEWS_PATH.exists():
        sys.exit(f"Reviews file not found: {REVIEWS_PATH}")
    if not META_PATH.exists():
        sys.exit(f"Metadata file not found: {META_PATH}")

    # Step 1: fix up the metadata file so it can be scanned as NDJSON.
    # Cached: skipped automatically on reruns if the cleaned file already exists.
    if not META_CLEAN_PATH.exists():
        t0 = time.perf_counter()
        convert_meta_to_ndjson(META_PATH, META_CLEAN_PATH)
        print(f"[meta] Conversion took {time.perf_counter() - t0:.1f}s")
    else:
        print(f"[meta] Reusing cached cleaned file: {META_CLEAN_PATH.name}")

    # Step 2: build lazy frames -- no data is read into memory yet.
    reviews_lf = build_reviews_lazyframe()
    meta_lf = build_meta_lazyframe()

    # Step 3: left join reviews -> metadata on asin, still lazy.
    joined_lf = reviews_lf.join(meta_lf, on="asin", how="left")

    # Step 4: execute the query with the streaming engine, so Polars
    # processes the data in batches instead of materializing everything
    # in RAM at once during the scan/filter/join.
    print("[join] Executing lazy query with the streaming engine...")
    t0 = time.perf_counter()
    try:
        joined_df = joined_lf.collect(engine="streaming")
    except TypeError:
        # Older Polars versions use the `streaming=True` kwarg instead of `engine=`.
        joined_df = joined_lf.collect(streaming=True)
    print(f"[join] Collected {joined_df.height:,} rows in {time.perf_counter() - t0:.1f}s")

    # Step 5: write the full joined dataset to Parquet.
    t0 = time.perf_counter()
    joined_df.write_parquet(FULL_OUTPUT_PATH)
    size_mb = FULL_OUTPUT_PATH.stat().st_size / 1e6
    print(
        f"[write] Full dataset -> {FULL_OUTPUT_PATH.name} ({size_mb:.1f} MB) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    # Step 6: randomly sample 50,000 rows for local dev / inference testing.
    sample_size = min(SAMPLE_SIZE, joined_df.height)
    sample_df = joined_df.sample(n=sample_size, seed=RANDOM_SEED)
    sample_df.write_parquet(SAMPLE_OUTPUT_PATH)
    print(f"[write] Sample dataset ({sample_size:,} rows) -> {SAMPLE_OUTPUT_PATH.name}")

    total = time.perf_counter() - t_start
    print(f"\nTotal execution time: {total:.1f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
