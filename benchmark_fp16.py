"""
Benchmarks pyabsa ATEPC inference speed AND prediction consistency under
torch.autocast (mixed FP16/FP32) vs. the default FP32.

Important framing: this does NOT measure "accuracy" (that would need real
ground-truth labels, which this project deliberately does not rely on --
see the memory note on the invalid 33-row spot-check). It measures
*agreement*: does FP16 change what the model outputs compared to FP32, on
the exact same reviews? A high agreement rate means FP16 is a safe speed
win; a low one means it's changing real answers and isn't worth the risk.

Usage:
    python benchmark_fp16.py
"""

import sys
import time

import polars as pl
import torch
from pyabsa import ATEPCCheckpointManager

from config import BATCH_SIZE, CHECKPOINT, DEVICE, FULL_DATASET_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASIN = "B0034JWXBI"
SAMPLE_SIZE = 200


def load_texts():
    df = pl.scan_parquet(FULL_DATASET_PATH).filter(pl.col("asin") == ASIN).collect()
    texts = df["reviewText"].drop_nulls().to_list()
    return [t for t in texts if t.strip()][:SAMPLE_SIZE]


def run_inference(extractor, texts, use_autocast: bool):
    results = [None] * len(texts)
    if DEVICE == "cuda:0":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start : start + BATCH_SIZE]
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                chunk_results = extractor.predict(
                    chunk, print_result=False, save_result=False, pred_sentiment=True
                )
        else:
            chunk_results = extractor.predict(
                chunk, print_result=False, save_result=False, pred_sentiment=True
            )
        for i, r in enumerate(chunk_results):
            results[start + i] = r
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda:0" else None
    return results, elapsed, peak_mem


def to_pairs(result):
    """{"aspect": "sentiment"} for one review's result, lowercased, or {} if none."""
    if result is None:
        return {}
    aspects = result.get("aspect") or []
    sentiments = result.get("sentiment") or []
    return dict(zip((a.lower() for a in aspects), sentiments, strict=False))


def compare(fp32_results, fp16_results):
    exact_matches = 0
    total_aspects_fp32 = 0
    total_aspects_fp16 = 0
    shared_aspect_agreements = 0
    shared_aspect_total = 0

    for r32, r16 in zip(fp32_results, fp16_results, strict=True):
        p32, p16 = to_pairs(r32), to_pairs(r16)
        total_aspects_fp32 += len(p32)
        total_aspects_fp16 += len(p16)
        if p32 == p16:
            exact_matches += 1
        shared = set(p32) & set(p16)
        shared_aspect_total += len(shared)
        shared_aspect_agreements += sum(1 for a in shared if p32[a] == p16[a])

    return {
        "exact_match_rate": exact_matches / len(fp32_results),
        "total_aspects_fp32": total_aspects_fp32,
        "total_aspects_fp16": total_aspects_fp16,
        "shared_aspect_sentiment_agreement": (
            shared_aspect_agreements / shared_aspect_total if shared_aspect_total else float("nan")
        ),
    }


def main():
    print(f"[setup] Loading {SAMPLE_SIZE} reviews for {ASIN}...")
    texts = load_texts()
    print(f"[setup] {len(texts)} review texts loaded")

    print(f"[setup] Loading pyabsa ATEPC extractor (device={DEVICE})...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=CHECKPOINT, auto_device=DEVICE)
    print("[setup] Extractor ready\n")

    print("[run] FP32 (baseline)...")
    fp32_results, fp32_time, fp32_mem = run_inference(extractor, texts, use_autocast=False)
    print(f"[run] FP32 done in {fp32_time:.2f}s ({len(texts) / fp32_time:.1f} reviews/sec)")

    print("[run] FP16 (autocast)...")
    fp16_results, fp16_time, fp16_mem = run_inference(extractor, texts, use_autocast=True)
    print(f"[run] FP16 done in {fp16_time:.2f}s ({len(texts) / fp16_time:.1f} reviews/sec)\n")

    stats = compare(fp32_results, fp16_results)

    speedup = fp32_time / fp16_time
    print("=" * 78)
    print(f"{'Metric':<40}{'FP32':>18}{'FP16 (autocast)':>18}")
    print("-" * 78)
    print(f"{'Time (s)':<40}{fp32_time:>18.2f}{fp16_time:>18.2f}")
    print(f"{'Throughput (reviews/sec)':<40}{len(texts) / fp32_time:>18.1f}{len(texts) / fp16_time:>18.1f}")
    print(f"{'Peak GPU memory (MB)':<40}{fp32_mem:>18.0f}{fp16_mem:>18.0f}")
    print(f"{'Speedup':<40}{'1.00x':>18}{f'{speedup:.2f}x':>18}")
    print("-" * 78)
    print(
        f"{'Total aspects extracted':<40}{stats['total_aspects_fp32']:>18}{stats['total_aspects_fp16']:>18}"
    )
    print(f"{'Exact per-review match rate':<58}{stats['exact_match_rate']:>20.1%}")
    print(f"{'Sentiment agreement on shared aspects':<58}{stats['shared_aspect_sentiment_agreement']:>20.1%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
