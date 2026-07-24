"""
Diagnostic: re-runs aspect_postprocessing.filter_pyabsa_result over the same
raw pyabsa outputs used in the gold-standard annotation batch, and compares
Precision/Recall/F1 before vs. after the filter, against the same 100
already-annotated gold rows.

IMPORTANT CAVEAT: this measures the filter's effect on the SAME sample used
to design it (via analyze_absa_errors.py) -- it's a same-sample diagnostic,
not held-out validation. A real "did this help" answer needs the filter
applied to reviews that were never inspected while designing it. Treat the
numbers here as "plausible direction," not proof.

Usage:
    python compare_postprocessing_effect.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"
RAW_JSONL_PATH = BASE_DIR / "annotation_batch_raw.jsonl"

sys.path.insert(0, str(BASE_DIR))
from aspect_postprocessing import filter_pyabsa_result  # noqa: E402
from build_gold_standard import aggregate, bootstrap_ci, build_gold_row, row_counts  # noqa: E402


def to_lower_dict(aspects, sentiments) -> dict:
    return dict(zip((a.lower() for a in aspects), sentiments, strict=False))


def main():
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df["human_verdict"] = df["human_verdict"].fillna("").astype(str).str.strip()
    annotated = df[df["human_verdict"] != ""].set_index("review_id")

    raw_by_id = {}
    with RAW_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            raw_by_id[rec["review_id"]] = rec["pyabsa_raw"]

    before_rows, after_rows = [], []
    for review_id, row in annotated.iterrows():
        gold = build_gold_row(row)
        raw = raw_by_id.get(review_id)

        before_predicted = to_lower_dict((raw or {}).get("aspect") or [], (raw or {}).get("sentiment") or [])
        before_rows.append(row_counts(before_predicted, gold))

        if raw is None:
            after_rows.append(row_counts({}, gold))
            continue
        filtered = filter_pyabsa_result(raw)
        after_predicted = to_lower_dict(filtered.get("aspect") or [], filtered.get("sentiment") or [])
        after_rows.append(row_counts(after_predicted, gold))

    before_agg = aggregate(before_rows)
    after_agg = aggregate(after_rows)
    before_ci = bootstrap_ci(before_rows, 3000, seed=42)
    after_ci = bootstrap_ci(after_rows, 3000, seed=42)

    print(f"{'=' * 70}\nSAME-SAMPLE DIAGNOSTIC ONLY -- not held-out validation\n{'=' * 70}\n")
    print(f"{'Metric':<20}{'Before':<25}{'After':<25}")
    for key in ("precision", "recall", "f1"):
        b, a = before_agg[key], after_agg[key]
        bci, aci = before_ci[key], after_ci[key]
        print(f"{key:<20}{b:.3f} [{bci[0]:.3f},{bci[1]:.3f}]      {a:.3f} [{aci[0]:.3f},{aci[1]:.3f}]")
    print(f"\nTP/FP/FN before: {before_agg['tp']}/{before_agg['fp']}/{before_agg['fn']}")
    print(f"TP/FP/FN after:  {after_agg['tp']}/{after_agg['fp']}/{after_agg['fn']}")
    print(f"\nsentiment_accuracy before: {before_agg['sentiment_accuracy']:.3f}")
    print(f"sentiment_accuracy after:  {after_agg['sentiment_accuracy']:.3f}")


if __name__ == "__main__":
    main()
