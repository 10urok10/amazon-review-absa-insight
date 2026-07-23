"""
Faz 1: Gold-standard annotation batch generator.

Draws a stratified sample (equal count per star rating, 1-5) from
processed_full_dataset.parquet, runs the existing pyabsa ATEPC pipeline on
it, and writes an editable .xlsx workbook so a human annotator can review
each prediction and record the ground truth. See annotation_guidelines.md
for the labeling rules.

IMPORTANT: this is a raw-model evaluation batch. Aspect keys here are only
lowercased -- SYNONYM_MAP normalization is a downstream aggregation step
(used in eda_analysis.py) and must NOT be applied here, or we'd be grading
the model against its own post-processing instead of its real output.

Requirements:
    pip install polars pyabsa torch pandas openpyxl

Usage:
    python generate_annotation_batch.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import torch
from pyabsa import ATEPCCheckpointManager

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FULL_DATASET_PATH = BASE_DIR / "processed_full_dataset.parquet"
OUTPUT_XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"
OUTPUT_RAW_JSONL_PATH = BASE_DIR / "annotation_batch_raw.jsonl"

STAR_RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]
N_PER_RATING = 40  # 5 * 40 = 200 rows total
RANDOM_SEED = 42

CHECKPOINT = "english"
BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def stratified_sample() -> pl.DataFrame:
    """Equal-sized sample per star rating, so the gold set isn't skewed
    toward the mostly-positive rating distribution real Amazon reviews have."""
    print(f"[sample] Reading {FULL_DATASET_PATH.name}")
    full_df = pl.scan_parquet(FULL_DATASET_PATH).select(
        ["asin", "overall", "reviewText", "title"]
    ).collect()
    print(f"[sample] {full_df.height:,} total rows available")

    parts = []
    for star in STAR_RATINGS:
        bucket = full_df.filter(pl.col("overall") == star)
        n = min(N_PER_RATING, bucket.height)
        if n < N_PER_RATING:
            print(f"[warn] only {bucket.height} rows at overall={star}, "
                  f"using all of them instead of {N_PER_RATING}")
        parts.append(bucket.sample(n=n, seed=RANDOM_SEED, shuffle=True))

    sampled = pl.concat(parts)
    # Shuffle across star ratings so the annotator isn't unconsciously
    # anchored by seeing 40 five-star reviews in a row, etc.
    sampled = sampled.sample(fraction=1.0, seed=RANDOM_SEED, shuffle=True)
    sampled = sampled.with_row_index("review_id")
    print(f"[sample] Stratified sample: {sampled.height} rows "
          f"({N_PER_RATING} per rating x {len(STAR_RATINGS)} ratings)")
    return sampled


def load_extractor():
    print(f"[load] Initializing pyabsa ATEPC aspect extractor (checkpoint='{CHECKPOINT}')...")
    return ATEPCCheckpointManager.get_aspect_extractor(checkpoint=CHECKPOINT, auto_device=DEVICE)


def safe_predict_chunk(extractor, texts):
    """Same defensive batching used in pyabsa_pipeline.py: a whole-chunk
    failure falls back to per-row retries instead of losing the batch."""
    try:
        return extractor.predict(texts, print_result=False, save_result=False, pred_sentiment=True)
    except Exception as exc:
        print(f"[warn] batch of {len(texts)} failed ({type(exc).__name__}: {exc}); "
              f"retrying rows individually")
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


def to_prediction_json(result):
    """Lowercased {aspect: sentiment} JSON, or None. No synonym mapping here
    -- this batch grades the raw model, not the downstream aggregation."""
    if result is None:
        return None
    aspects = result.get("aspect") or []
    sentiments = result.get("sentiment") or []
    if not aspects:
        return None
    return json.dumps(dict(zip((a.lower() for a in aspects), sentiments)), ensure_ascii=False)


def main():
    t_start = time.perf_counter()

    if not FULL_DATASET_PATH.exists():
        raise SystemExit(f"Input file not found: {FULL_DATASET_PATH}")

    sampled = stratified_sample()

    t0 = time.perf_counter()
    extractor = load_extractor()
    print(f"[load] Extractor ready in {time.perf_counter() - t0:.1f}s, device={DEVICE}")

    review_ids = sampled["review_id"].to_list()
    texts = sampled["reviewText"].to_list()

    raw_results = [None] * len(texts)
    print(f"[infer] Running ATEPC inference on {len(texts):,} reviews "
          f"(batch_size={BATCH_SIZE})...")
    t0 = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        chunk_texts = texts[start:start + BATCH_SIZE]
        chunk_results = safe_predict_chunk(extractor, chunk_texts)
        for i, result in enumerate(chunk_results):
            raw_results[start + i] = result
    print(f"[infer] Done in {time.perf_counter() - t0:.1f}s")

    # Audit trail: full pyabsa output (IOB tags, positions, confidence) per
    # row, kept separately from the annotation workbook for later error
    # analysis (e.g. "does the model fail more on low-confidence predictions?").
    with OUTPUT_RAW_JSONL_PATH.open("w", encoding="utf-8") as f:
        for review_id, result in zip(review_ids, raw_results):
            record = {"review_id": int(review_id), "pyabsa_raw": result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[write] Raw predictions -> {OUTPUT_RAW_JSONL_PATH.name}")

    predicted_json = [to_prediction_json(r) for r in raw_results]
    n_with_aspects = sum(1 for v in predicted_json if v is not None)
    print(f"[result] {n_with_aspects}/{len(texts)} rows have >=1 predicted aspect")

    annotation_df = pd.DataFrame({
        "review_id": review_ids,
        "asin": sampled["asin"].to_list(),
        "overall": sampled["overall"].to_list(),
        "reviewText": texts,
        "predicted_aspects_json": predicted_json,
        "human_verdict": ["" for _ in texts],
        "corrected_aspects_json": ["" for _ in texts],
        "notes": ["" for _ in texts],
    })

    annotation_df.to_excel(OUTPUT_XLSX_PATH, index=False, engine="openpyxl")
    print(f"[write] Annotation workbook -> {OUTPUT_XLSX_PATH.name}")

    total = time.perf_counter() - t_start
    print(f"\nTotal execution time: {total:.1f}s")
    print("\nNext step: open annotation_batch.xlsx, follow annotation_guidelines.md, "
          "fill in human_verdict (+ corrected_aspects_json where needed) for all rows.")


if __name__ == "__main__":
    main()
