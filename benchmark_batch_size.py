"""
Benchmarks pyabsa ATEPC inference throughput at different batch sizes, so
BATCH_SIZE in config.py is a measured choice, not a guess. Re-run this if you
change GPUs or the pyabsa checkpoint.

Usage:
    python benchmark_batch_size.py
"""

import sys
import time

import polars as pl
import torch
from pyabsa import ATEPCCheckpointManager

from config import CHECKPOINT, DEVICE, FULL_DATASET_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Any ASIN with plenty of reviews works -- this one just happens to be one
# already used elsewhere in testing, so results are easy to cross-check.
ASIN = "B0034JWXBI"
SAMPLE_SIZE = 300
BATCH_SIZES_TO_TEST = [16, 32, 64, 128]


def main():
    print(f"[setup] Loading reviews for {ASIN}...")
    df = pl.scan_parquet(FULL_DATASET_PATH).filter(pl.col("asin") == ASIN).collect()
    texts = df["reviewText"].drop_nulls().to_list()
    texts = [t for t in texts if t.strip()][:SAMPLE_SIZE]
    print(f"[setup] {len(texts)} review texts loaded")

    print(f"[setup] Loading pyabsa ATEPC extractor (device={DEVICE})...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=CHECKPOINT, auto_device=DEVICE)
    print("[setup] Extractor ready\n")

    for batch_size in BATCH_SIZES_TO_TEST:
        if DEVICE == "cuda:0":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                extractor.predict(chunk, print_result=False, save_result=False, pred_sentiment=True)
            elapsed = time.perf_counter() - t0
            rate = len(texts) / elapsed
            if DEVICE == "cuda:0":
                peak_mem = torch.cuda.max_memory_allocated() / 1e6
                print(
                    f"[batch_size={batch_size:>3}] {elapsed:6.2f}s total, "
                    f"{rate:5.1f} reviews/sec, peak GPU mem: {peak_mem:.0f} MB"
                )
            else:
                print(f"[batch_size={batch_size:>3}] {elapsed:6.2f}s total, {rate:5.1f} reviews/sec")
        except RuntimeError as exc:
            print(f"[batch_size={batch_size:>3}] FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
