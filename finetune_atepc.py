"""
REJECTED EXPERIMENT -- kept for the record, not used in production.

Fine-tunes pyabsa's "english" ATEPC checkpoint on the SemEval-2014 Task 4
Laptop training data (see prepare_semeval_finetune_data.py) -- genuine
fine-tuning on an independent, human-labeled dataset, not self-distillation
on the model's own output (a different, legitimate approach from the
self-distillation fine-tuning already rejected earlier in this project).

Continues from the existing "english" checkpoint (from_checkpoint="english")
rather than training from scratch.

Measured against the same 100 Amazon gold-standard rows used for the
"english" baseline (F1=0.887) -- a genuinely valid, non-circular comparison
since SemEval Laptop reviews have zero overlap with these Amazon Electronics
reviews:
  Aggressive (5 epochs, lr=1e-5):  F1=0.273 (P=0.464, R=0.193)
  Light touch (1 epoch, lr=2e-6):  F1=0.326 (P=0.721, R=0.210)

Both dramatically worse than the unmodified baseline, and worse specifically
via a collapsed recall (0.19-0.21 vs 0.897) even though the light-touch
version's precision looked fine in isolation (0.721). Conclusion: fine-
tuning on this small (2744-sentence), single-domain, stylistically-narrower
dataset makes the model *more conservative* about what counts as an aspect,
at the cost of the broad recall the original checkpoint had from its much
larger, more diverse training mix (Restaurant/Laptop/Twitter/MAMS/TV/
TShirt/Yelp/MOOC/Kaggle). Reducing the fine-tuning "dose" further only
approaches a no-op; this specific dataset is not a productive fine-tuning
source for this project's use case, regardless of how gently it's applied.

This is the current config (light-touch, num_epoch=1); the aggressive
config (num_epoch=5, learning_rate=1e-5) was tried first and is described
above for the record, not left in this file.

Usage:
    python finetune_atepc.py
"""

import sys
from pathlib import Path

from pyabsa import AspectTermExtraction as ATE
from pyabsa import DeviceTypeOption, ModelSaveOption

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "semeval_finetune_data"
SAVE_DIR = BASE_DIR / "checkpoints" / "atepc_semeval_finetuned_light"


def main():
    if not DATASET_DIR.exists():
        raise SystemExit(f"{DATASET_DIR} not found -- run prepare_semeval_finetune_data.py first.")

    config = ATE.ATEPCConfigManager.get_atepc_config_english()
    # Light touch: the 5-epoch/1e-5 run caused catastrophic forgetting
    # (F1 0.887 -> 0.273 on the Amazon gold rows) -- much lower epoch count
    # and learning rate to test whether a gentler nudge avoids that while
    # still capturing some benefit.
    config.num_epoch = 1
    config.learning_rate = 2e-6
    config.batch_size = 16
    config.log_step = 20
    config.seed = 42

    ATE.ATEPCTrainer(
        config=config,
        dataset=str(DATASET_DIR),
        from_checkpoint="english",
        checkpoint_save_mode=ModelSaveOption.SAVE_FULL_MODEL,
        auto_device=DeviceTypeOption.AUTO,
        path_to_save=str(SAVE_DIR),
    )
    print(f"\nDone. Fine-tuned checkpoint saved under {SAVE_DIR}")


if __name__ == "__main__":
    main()
