"""
Faz 2: Gold-standard metrics -- turns a completed (or partially completed)
annotation_batch.xlsx into real Precision/Recall/F1 numbers for pyabsa's
aspect term extraction, plus sentiment-classification accuracy measured
*separately*, given a correctly extracted aspect.

Why these two are kept separate: a model can be bad at finding the right
words but good at polarity (or vice versa) -- collapsing both into one
"accuracy" number hides which failure mode actually dominates, and that's
exactly what determines which fix (post-processing filter vs. checkpoint
swap vs. fine-tuning) is worth pursuing.

Guardrail: this script refuses to produce a report from an incomplete
annotation batch unless --allow-partial is passed, and even then labels
the output PARTIAL/EXPLORATORY -- see the project's own history: a 33/200
partial run was previously (incorrectly) treated as a final accuracy
number, and annotate_cli.py's confidence-first row order means an
incomplete subset is not a random sample -- it's biased toward the
hardest, lowest-confidence predictions, so a partial run understates the
model's real accuracy on top of just having a wide confidence interval.

Usage:
    python build_gold_standard.py                    # requires all 200 rows done
    python build_gold_standard.py --allow-partial     # runs anyway, loudly labeled partial
    python build_gold_standard.py --bootstrap 5000    # more bootstrap resamples (default 2000)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"
REPORT_MD_PATH = BASE_DIR / "gold_standard_report.md"
REPORT_JSON_PATH = BASE_DIR / "gold_standard_report.json"

RANDOM_SEED = 42
NEEDS_CORRECTION = {"partial", "incorrect", "missed_all"}


def _parse_aspect_json(value) -> dict:
    """predicted_aspects_json / corrected_aspects_json cell -> {aspect: sentiment}.
    Empty cell (NaN, None, "") means "no aspects", not an error."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def build_gold_row(row: pd.Series) -> dict:
    """The ground-truth {aspect: sentiment} for one row, per
    annotation_guidelines.md's human_verdict semantics:
      correct    -> pyabsa's own prediction was confirmed right, gold == predicted
      no_aspect  -> confirmed no real aspect in the text, gold == {}
      partial / incorrect / missed_all -> corrected_aspects_json IS the full,
        complete gold set (not a diff) -- an empty correction here legitimately
        means "gold is {}" (e.g. pyabsa hallucinated an aspect that isn't real)."""
    verdict = str(row["human_verdict"]).strip()
    predicted = _parse_aspect_json(row["predicted_aspects_json"])
    if verdict == "correct":
        return predicted
    if verdict == "no_aspect":
        return {}
    if verdict in NEEDS_CORRECTION:
        return _parse_aspect_json(row["corrected_aspects_json"])
    raise ValueError(f"Unrecognized human_verdict {verdict!r} for review_id={row['review_id']}")


def row_counts(predicted: dict, gold: dict) -> dict:
    """Aspect Term Extraction TP/FP/FN for one row (set comparison on aspect
    keys), plus sentiment hit/total among the TP aspects (extraction-correct
    aspects only -- this is what isolates polarity accuracy from extraction
    accuracy)."""
    pred_keys = set(predicted)
    gold_keys = set(gold)
    tp_keys = pred_keys & gold_keys
    return {
        "tp": len(tp_keys),
        "fp": len(pred_keys - gold_keys),
        "fn": len(gold_keys - pred_keys),
        "sentiment_hits": sum(1 for k in tp_keys if predicted[k] == gold[k]),
        "sentiment_total": len(tp_keys),
    }


def micro_prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) and not np.isnan(precision + recall)
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    sent_hits = sum(r["sentiment_hits"] for r in rows)
    sent_total = sum(r["sentiment_total"] for r in rows)
    result = micro_prf1(tp, fp, fn)
    result["sentiment_accuracy"] = sent_hits / sent_total if sent_total else float("nan")
    result["n_rows"] = len(rows)
    result["tp"], result["fp"], result["fn"] = tp, fp, fn
    result["sentiment_hits"], result["sentiment_total"] = sent_hits, sent_total
    return result


def bootstrap_ci(rows: list[dict], n_resamples: int, seed: int) -> dict:
    """Resamples ROWS (not individual aspect matches) with replacement --
    aspects within one review aren't independent observations, so
    resampling at the review level is the statistically correct unit."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    metrics = {"precision": [], "recall": [], "f1": [], "sentiment_accuracy": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = [rows[i] for i in idx]
        agg = aggregate(sample)
        for key in metrics:
            if not np.isnan(agg[key]):
                metrics[key].append(agg[key])
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        if vals
        else (float("nan"), float("nan"))
        for key, vals in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Run even if not all rows are annotated yet. Output is loudly labeled PARTIAL/EXPLORATORY.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="Number of bootstrap resamples for 95%% CIs"
    )
    args = parser.parse_args()

    if not XLSX_PATH.exists():
        raise SystemExit(f"{XLSX_PATH.name} not found. Run generate_annotation_batch.py first.")

    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df["human_verdict"] = df["human_verdict"].fillna("").astype(str).str.strip()

    total = len(df)
    annotated = df[df["human_verdict"] != ""]
    pending = total - len(annotated)

    if pending > 0 and not args.allow_partial:
        print(
            f"[blocked] {pending}/{total} rows are not yet annotated.\n"
            f"Run annotate_cli.py to finish, or pass --allow-partial to get an "
            f"exploratory (loudly labeled, NOT final) report anyway.\n\n"
            f"Note: annotate_cli.py sorts rows lowest-model-confidence-first, so a "
            f"partial run is not a random subsample -- it's biased toward the model's "
            f"hardest cases and will understate real accuracy, on top of just having "
            f"a wider confidence interval."
        )
        sys.exit(1)

    is_partial = pending > 0
    if is_partial:
        print(
            f"{'=' * 70}\n"
            f"  PARTIAL / EXPLORATORY RUN -- {len(annotated)}/{total} rows annotated\n"
            f"  This is NOT a final accuracy number. Confidence-sorted annotation\n"
            f"  order means these rows skew toward the model's hardest cases.\n"
            f"{'=' * 70}\n"
        )

    row_metrics = []
    per_star = {}
    for _, row in annotated.iterrows():
        predicted = _parse_aspect_json(row["predicted_aspects_json"])
        gold = build_gold_row(row)
        counts = row_counts(predicted, gold)
        row_metrics.append(counts)
        per_star.setdefault(row["overall"], []).append(counts)

    overall_agg = aggregate(row_metrics)
    overall_ci = bootstrap_ci(row_metrics, args.bootstrap, RANDOM_SEED)

    verdict_counts = annotated["human_verdict"].value_counts().to_dict()

    notes = annotated["notes"].fillna("").astype(str)
    n_wrong_variant = notes.str.contains("yanlış ürün/spec", case=False, na=False).sum()
    n_generic_doa = (
        notes.str.contains("genel bozuk", case=False, na=False).sum()
        + notes.str.contains("DOA", case=False, na=False).sum()
    )

    star_lines = []
    for star in sorted(per_star):
        agg = aggregate(per_star[star])
        star_lines.append(
            f"| {star:.0f} | {agg['n_rows']} | {agg['precision']:.3f} | {agg['recall']:.3f} | "
            f"{agg['f1']:.3f} | {agg['sentiment_accuracy']:.3f} |"
        )

    label = "PARTIAL / EXPLORATORY" if is_partial else "FINAL"
    report_md = f"""# Gold-Standard ABSA Evaluation -- {label}

Sample: {len(annotated)}/{total} rows annotated (stratified, 40 per star rating 1-5).
Bootstrap resamples: {args.bootstrap} (row-level resampling, 95% CI).

{"**PARTIAL RUN -- do not cite as a final number.** Confidence-sorted annotation order means this subset skews toward the model's hardest cases." if is_partial else ""}

## Aspect Term Extraction (exact aspect-key match)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | {overall_agg["precision"]:.3f} | [{overall_ci["precision"][0]:.3f}, {overall_ci["precision"][1]:.3f}] |
| Recall | {overall_agg["recall"]:.3f} | [{overall_ci["recall"][0]:.3f}, {overall_ci["recall"][1]:.3f}] |
| F1 | {overall_agg["f1"]:.3f} | [{overall_ci["f1"][0]:.3f}, {overall_ci["f1"][1]:.3f}] |

TP={overall_agg["tp"]}, FP={overall_agg["fp"]}, FN={overall_agg["fn"]}

## Sentiment Classification (given a correctly extracted aspect)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | {overall_agg["sentiment_accuracy"]:.3f} | [{overall_ci["sentiment_accuracy"][0]:.3f}, {overall_ci["sentiment_accuracy"][1]:.3f}] |

{overall_agg["sentiment_hits"]}/{overall_agg["sentiment_total"]} correctly-extracted aspects had the right polarity.

## By star rating

| Star | n rows | Precision | Recall | F1 | Sentiment acc. |
|---|---|---|---|---|---|
{chr(10).join(star_lines)}

## human_verdict distribution

{chr(10).join(f"- {k}: {v}" for k, v in sorted(verdict_counts.items()))}

## Flagged special cases (see annotation_guidelines.md)

- Wrong variant/spec shipped: {n_wrong_variant}
- Generic broken/DOA complaint (no specific aspect named): {n_generic_doa}
"""

    REPORT_MD_PATH.write_text(report_md, encoding="utf-8")
    report_json = {
        "label": label,
        "n_annotated": len(annotated),
        "n_total": total,
        "bootstrap_resamples": args.bootstrap,
        "aspect_extraction": {
            **{k: overall_agg[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
            "ci_95": {k: overall_ci[k] for k in ("precision", "recall", "f1")},
        },
        "sentiment_classification": {
            "accuracy": overall_agg["sentiment_accuracy"],
            "hits": overall_agg["sentiment_hits"],
            "total": overall_agg["sentiment_total"],
            "ci_95": overall_ci["sentiment_accuracy"],
        },
        "verdict_counts": verdict_counts,
        "flagged": {
            "wrong_variant_or_spec": int(n_wrong_variant),
            "generic_doa_complaint": int(n_generic_doa),
        },
    }
    REPORT_JSON_PATH.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report_md)
    print(f"\n[write] {REPORT_MD_PATH.name}, {REPORT_JSON_PATH.name}")


if __name__ == "__main__":
    main()
