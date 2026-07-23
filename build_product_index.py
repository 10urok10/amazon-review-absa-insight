"""
Builds a small product index (asin, title, review_count, avg_rating) from
processed_full_dataset.parquet, so the search UI doesn't have to scan 7.8M
review rows on every keystroke.

Run once (re-run only if processed_full_dataset.parquet changes):
    python build_product_index.py
"""

import sys
import time

import polars as pl

from config import FULL_DATASET_PATH, PRODUCT_INDEX_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    t0 = time.perf_counter()
    print(f"[build] Reading {FULL_DATASET_PATH.name}...")

    index = (
        pl.scan_parquet(FULL_DATASET_PATH)
        .group_by("asin")
        .agg(
            pl.col("title").drop_nulls().first().alias("title"),
            pl.len().alias("review_count"),
            pl.col("overall").mean().alias("avg_rating"),
        )
        .filter(pl.col("title").is_not_null())
        .sort("review_count", descending=True)
        .collect()
    )

    index.write_parquet(PRODUCT_INDEX_PATH)
    print(
        f"[build] {index.height:,} products indexed -> {PRODUCT_INDEX_PATH.name} "
        f"in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
