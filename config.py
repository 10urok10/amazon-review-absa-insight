"""
Centralized configuration for the active pipeline (insight_engine.py,
app.py, extension_server.py, build_product_index.py, analyze_product.py).

Legacy/exploratory scripts (absa_pipeline.py, pyabsa_pipeline.py,
eda_analysis.py, generate_annotation_batch.py, annotate_cli.py -- see
CLAUDE.md) predate this module and are not wired to it; leave them as-is.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FULL_DATASET_PATH = BASE_DIR / "processed_full_dataset.parquet"
PRODUCT_INDEX_PATH = BASE_DIR / "product_index.parquet"
CACHE_DB_PATH = BASE_DIR / "product_insights_cache.sqlite3"
LOG_PATH = BASE_DIR / "app.log"

# ---------------------------------------------------------------------------
# pyabsa / model
# ---------------------------------------------------------------------------
CHECKPOINT = "english"
# Benchmarked on an RTX 5060 (8GB VRAM): batch_size=16 -> 7.5 reviews/sec,
# peak 1.86GB; batch_size=64 -> 9.4 reviews/sec (~25% faster) at the same
# peak memory (pyabsa internally caps activation memory regardless of the
# batch size passed in). 128 gave only ~5% more on top of that -- not worth
# the reduced headroom. Re-benchmark with _benchmark_batch_size.py if you
# change GPUs or the checkpoint.
BATCH_SIZE = 64
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Benchmarked with benchmark_fp16.py (torch.autocast, not a raw .half() cast --
# keeps numerically sensitive ops like softmax/layer norm in FP32): 1.40x
# throughput, *lower* peak GPU memory, 99.5% exact per-review prediction match
# vs FP32 and 100% sentiment agreement on aspects found by both, on a 200-review
# sample. Only applies on CUDA; autocast is skipped on CPU.
USE_FP16_INFERENCE = True

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Pipeline behavior
# ---------------------------------------------------------------------------
MAX_REVIEWS_PER_PRODUCT = 500  # bounds worst-case latency for very popular ASINs
RANDOM_SEED = 42

# Groups near-duplicate/related aspect terms under one canonical name before
# aggregation, so mention counts aren't split across synonyms.
SYNONYM_MAP = {
    "install": "installation",
    "cable": "charger",
    "cables": "charger",
    "ipad": "tablet",
}

# ---------------------------------------------------------------------------
# Visualization (shared by app.py; fixed status colors, never themed)
# ---------------------------------------------------------------------------
SENTIMENT_COLORS = {"Positive": "#0ca30c", "Neutral": "#898781", "Negative": "#d03b3b"}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
