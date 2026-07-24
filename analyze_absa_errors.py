"""
Faz 3: Error analysis over the gold-standard annotations -- turns the
already-annotated rows in annotation_batch.xlsx into concrete lists of
false-positive aspects (pyabsa said it's an aspect, gold says it isn't) and
false-negative aspects (gold says it's an aspect, pyabsa missed it), with
review context, so any post-processing fix is designed from real error
patterns instead of guessing.

Usage:
    python analyze_absa_errors.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"

sys.path.insert(0, str(BASE_DIR))
from build_gold_standard import _parse_aspect_json, build_gold_row  # noqa: E402


def main():
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df["human_verdict"] = df["human_verdict"].fillna("").astype(str).str.strip()
    annotated = df[df["human_verdict"] != ""]

    fp_terms = Counter()
    fn_terms = Counter()
    fp_examples = {}
    fn_examples = {}

    for _, row in annotated.iterrows():
        predicted = _parse_aspect_json(row["predicted_aspects_json"])
        gold = build_gold_row(row)
        pred_keys = set(predicted)
        gold_keys = set(gold)

        for term in pred_keys - gold_keys:
            fp_terms[term] += 1
            fp_examples.setdefault(term, []).append(row["reviewText"])
        for term in gold_keys - pred_keys:
            fn_terms[term] += 1
            fn_examples.setdefault(term, []).append(row["reviewText"])

    print(f"Analyzed {len(annotated)} annotated rows\n")

    print("=" * 70)
    print(f"FALSE POSITIVES (pyabsa said aspect, gold disagreed) -- {sum(fp_terms.values())} total")
    print("=" * 70)
    for term, count in fp_terms.most_common(40):
        example = fp_examples[term][0]
        print(f"  [{count}x] {term!r}")
        print(f"          e.g.: {example[:120]}")

    print()
    print("=" * 70)
    print(f"FALSE NEGATIVES (gold has aspect, pyabsa missed it) -- {sum(fn_terms.values())} total")
    print("=" * 70)
    for term, count in fn_terms.most_common(40):
        example = fn_examples[term][0]
        print(f"  [{count}x] {term!r}")
        print(f"          e.g.: {example[:120]}")

    # Quick heuristic signals worth knowing before designing a filter:
    print()
    print("=" * 70)
    print("QUICK SIGNALS")
    print("=" * 70)
    single_char_or_stopword_fp = sum(1 for t in fp_terms if len(t) <= 3)
    multiword_fn = sum(1 for t in fn_terms if " " in t)
    print(f"FP terms with length <= 3 chars: {single_char_or_stopword_fp}/{len(fp_terms)} distinct FP terms")
    print(f"FN (missed) terms that are multi-word: {multiword_fn}/{len(fn_terms)} distinct FN terms")


if __name__ == "__main__":
    main()
