"""
REJECTED EXPERIMENT -- kept for the record, not used in production.

Runs the "multilingual" ATEPC checkpoint (pyabsa's only other readily
available checkpoint besides "english" for this task) over the same 100
gold-standard reviews, and compares Precision/Recall/F1 against the current
"english" checkpoint's numbers -- same-sample diagnostic, see caveats in
build_gold_standard.py.

Result: F1=0.323 (P=0.535, R=0.232) vs. the "english" baseline's F1=0.887.
Multilingual models trade per-language accuracy for cross-lingual coverage;
"english" is confirmed as pyabsa's best available checkpoint for this
language, no alternative/ensemble is worth pursuing further.

Usage:
    python compare_checkpoints.py
"""

import sys
from pathlib import Path

import pandas as pd
from pyabsa import ATEPCCheckpointManager

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"

sys.path.insert(0, str(BASE_DIR))
from build_gold_standard import aggregate, bootstrap_ci, build_gold_row, row_counts  # noqa: E402


def to_lower_dict(aspects, sentiments) -> dict:
    return dict(zip((a.lower() for a in aspects), sentiments, strict=False))


def main():
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df["human_verdict"] = df["human_verdict"].fillna("").astype(str).str.strip()
    annotated = df[df["human_verdict"] != ""]

    print("Loading multilingual ATEPC checkpoint...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="multilingual", auto_device="cuda:0")

    texts = annotated["reviewText"].tolist()
    print(f"Running inference on {len(texts)} reviews...")
    results = extractor.predict(texts, print_result=False, save_result=False, pred_sentiment=True)

    rows = []
    for (_, row), result in zip(annotated.iterrows(), results, strict=True):
        gold = build_gold_row(row)
        predicted = to_lower_dict(result.get("aspect") or [], result.get("sentiment") or [])
        rows.append(row_counts(predicted, gold))

    agg = aggregate(rows)
    ci = bootstrap_ci(rows, 3000, seed=42)

    print(f"\n{'=' * 70}")
    print("multilingual checkpoint vs. the same 100 gold rows")
    print(f"{'=' * 70}")
    print(f"Precision: {agg['precision']:.3f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]")
    print(f"Recall:    {agg['recall']:.3f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]")
    print(f"F1:        {agg['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]")
    print(f"TP/FP/FN:  {agg['tp']}/{agg['fp']}/{agg['fn']}")
    print(f"Sentiment accuracy (given correct aspect): {agg['sentiment_accuracy']:.3f}")
    print("\n(english checkpoint baseline, for reference: P=0.878 R=0.897 F1=0.887, sentiment=0.986)")


if __name__ == "__main__":
    main()
