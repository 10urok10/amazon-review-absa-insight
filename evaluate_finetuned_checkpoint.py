"""
Loads the SemEval-2014-Laptop-fine-tuned ATEPC checkpoint and evaluates it
against the same 100 Amazon gold-standard rows used for the "english"
baseline (F1=0.887) and the rejected multilingual-checkpoint comparison
(F1=0.323). This is a genuinely valid, non-circular held-out comparison:
the fine-tuning data (SemEval Laptop reviews) has zero overlap with these
Amazon Electronics reviews.

Usage:
    python evaluate_finetuned_checkpoint.py <checkpoint_dir>
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from pyabsa import ATEPCCheckpointManager

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# PyTorch >=2.6 defaults torch.load(weights_only=True), which refuses to
# unpickle the chain of custom/framework classes our SAVE_FULL_MODEL
# checkpoint was saved with (LCF_ATEPC, DebertaV2Model, ...). pyabsa 2.4.3
# predates that torch default and doesn't allowlist any of it itself.
# Allowlisting one class at a time surfaces another; force weights_only=False
# instead -- acceptable here since we trust this file completely (we trained
# it ourselves, locally, moments ago).
_original_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"

sys.path.insert(0, str(BASE_DIR))
from build_gold_standard import aggregate, bootstrap_ci, build_gold_row, row_counts  # noqa: E402


def to_lower_dict(aspects, sentiments) -> dict:
    return dict(zip((a.lower() for a in aspects), sentiments, strict=False))


def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not checkpoint_dir:
        raise SystemExit("Usage: python evaluate_finetuned_checkpoint.py <checkpoint_dir>")

    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df["human_verdict"] = df["human_verdict"].fillna("").astype(str).str.strip()
    annotated = df[df["human_verdict"] != ""]

    print(f"Loading fine-tuned checkpoint from {checkpoint_dir}...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_dir, auto_device="cuda:0")

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
    print("SemEval-2014-Laptop-fine-tuned checkpoint vs. the same 100 gold rows")
    print(f"{'=' * 70}")
    print(f"Precision: {agg['precision']:.3f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]")
    print(f"Recall:    {agg['recall']:.3f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]")
    print(f"F1:        {agg['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]")
    print(f"TP/FP/FN:  {agg['tp']}/{agg['fp']}/{agg['fn']}")
    print(f"Sentiment accuracy (given correct aspect): {agg['sentiment_accuracy']:.3f}")
    print("\n(english baseline, for reference: P=0.878 R=0.897 F1=0.887, sentiment=0.986)")


if __name__ == "__main__":
    main()
