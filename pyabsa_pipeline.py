"""
Aspect-Based Sentiment Analysis (ABSA) inference pipeline using pyabsa's
ATEPC (Aspect Term Extraction and Polarity Classification) architecture.

Unlike the earlier spaCy + deberta-v3-base-absa-v1.1 two-stage pipeline,
pyabsa's ATEPC checkpoint performs aspect extraction AND sentiment
classification in a single model pass (IOB tagging for aspect spans, then
polarity classification per span), eliminating the rule-based noise of
POS-tagging-based aspect extraction.

Requirements:
    pip install pyabsa torch

Usage:
    python pyabsa_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

import polars as pl
import torch
from pyabsa import ATEPCCheckpointManager

# Windows consoles often use a non-UTF-8 codepage (e.g. cp1254 on Turkish
# Windows); reviewText can contain arbitrary Unicode, so force UTF-8 stdout
# to avoid crashing on print().
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "sample_reviews.parquet"
OUTPUT_PATH = BASE_DIR / "pyabsa_sample_results.parquet"

CHECKPOINT = "english"
N_ROWS = 100
BATCH_SIZE = 16  # reviews per pyabsa.predict() call -- lower if you hit GPU OOM

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_extractor():
    print(f"[load] Initializing pyabsa ATEPC aspect extractor (checkpoint='{CHECKPOINT}')...")
    return ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint=CHECKPOINT,
        auto_device=DEVICE,
    )


def safe_predict_chunk(extractor, texts):
    """
    Run a chunk of reviews through pyabsa in one batched call (efficient on
    GPU). If the whole chunk raises -- e.g. a pathological input triggers a
    tokenization edge case, or a transient CUDA error/OOM -- fall back to
    predicting each row individually so one bad row can't discard the other
    N-1 good results in that chunk.
    """
    try:
        return extractor.predict(texts, print_result=False, save_result=False, pred_sentiment=True)
    except Exception as exc:
        print(
            f"[warn] batch of {len(texts)} failed ({type(exc).__name__}: {exc}); retrying rows individually"
        )
        if DEVICE == "cuda:0":
            torch.cuda.empty_cache()
        results = []
        for text in texts:
            try:
                r = extractor.predict([text], print_result=False, save_result=False, pred_sentiment=True)
                results.append(r[0])
            except Exception as exc2:
                print(f"[warn] row failed individually ({type(exc2).__name__}: {exc2}), skipping")
                results.append(None)
        return results


def to_aspect_sentiment_json(result):
    """
    Turn one pyabsa result dict (with parallel 'aspect'/'sentiment' lists)
    into a {aspect: sentiment} JSON string. Returns None if inference failed
    for this row, or if no aspects were found -- per spec, rows with no
    valid aspects are ignored (left null) rather than given an empty "{}".
    """
    if result is None:
        return None
    aspects = result.get("aspect") or []
    sentiments = result.get("sentiment") or []
    if not aspects:
        return None
    # Normalize aspect keys to lowercase (e.g. all-caps reviews like "CHARGER"
    # would otherwise produce a differently-cased key per review).
    mapping = {}
    for aspect, sentiment in zip(aspects, sentiments):
        mapping[aspect.lower()] = sentiment
    return json.dumps(mapping, ensure_ascii=False)


def main():
    t_start = time.perf_counter()

    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    print(f"[load] Reading first {N_ROWS} rows from {INPUT_PATH.name}")
    df = pl.scan_parquet(INPUT_PATH).head(N_ROWS).collect()

    t0 = time.perf_counter()
    extractor = load_extractor()
    print(f"[load] Extractor ready in {time.perf_counter() - t0:.1f}s, device={DEVICE}")

    raw_texts = df["reviewText"].to_list()

    # Null/blank handling: these rows never reach the model.
    valid_indices, valid_texts = [], []
    for i, text in enumerate(raw_texts):
        if isinstance(text, str) and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)
        else:
            print(f"[warn] row {i}: null/empty reviewText, skipping")

    # Note: pyabsa's ATEPC checkpoint truncates internally to its configured
    # max_seq_len (256 tokens for the 'english' checkpoint) rather than
    # raising on long inputs, so no manual pre-truncation is required --
    # `safe_predict_chunk`'s try/except is still there as a safety net for
    # any other unexpected failure.
    results_by_row = [None] * len(raw_texts)

    print(f"[infer] Running ATEPC inference on {len(valid_texts):,} reviews (batch_size={BATCH_SIZE})...")
    t0 = time.perf_counter()
    for start in range(0, len(valid_texts), BATCH_SIZE):
        chunk_texts = valid_texts[start : start + BATCH_SIZE]
        chunk_indices = valid_indices[start : start + BATCH_SIZE]
        chunk_results = safe_predict_chunk(extractor, chunk_texts)
        for row_idx, result in zip(chunk_indices, chunk_results):
            results_by_row[row_idx] = result
    print(f"[infer] Done in {time.perf_counter() - t0:.1f}s")

    aspect_sentiments = [to_aspect_sentiment_json(r) for r in results_by_row]
    n_with_aspects = sum(1 for v in aspect_sentiments if v is not None)
    print(f"[result] {n_with_aspects}/{len(raw_texts)} rows have at least one aspect")

    df = df.with_columns(pl.Series("aspect_sentiments", aspect_sentiments))

    df.write_parquet(OUTPUT_PATH)
    print(f"[write] Results -> {OUTPUT_PATH.name}")

    with pl.Config(fmt_str_lengths=200, tbl_width_chars=200):
        print(df.select(["asin", "overall", "reviewText", "aspect_sentiments"]))

    total = time.perf_counter() - t_start
    print(f"\nTotal execution time: {total:.1f}s")


if __name__ == "__main__":
    main()
