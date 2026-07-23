"""
Aspect-Based Sentiment Analysis (ABSA) inference pipeline.

IMPORTANT: yangheng/deberta-v3-base-absa-v1.1 is an *aspect sentiment
classifier* (ASC) -- given a (sentence, aspect) pair it scores that aspect's
sentiment. It does NOT extract aspect terms by itself. So this script uses a
two-stage approach:

  Stage 1 (aspect extraction): a lightweight, model-free noun extractor
    (spaCy POS tagging) proposes candidate aspect terms per review, e.g.
    "battery", "screen", "price".
  Stage 2 (sentiment classification): each (review, aspect) pair is scored
    by yangheng/deberta-v3-base-absa-v1.1 as Positive / Negative / Neutral.

Requirements:
    pip install transformers torch spacy sentencepiece
    python -m spacy download en_core_web_sm

Usage:
    python absa_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

import polars as pl
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Windows consoles often use a non-UTF-8 codepage (e.g. cp1254 on Turkish
# Windows); reviewText can contain arbitrary Unicode (curly quotes, accents,
# emoji), so force stdout/stderr to UTF-8 to avoid crashing on print().
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "sample_reviews.parquet"
OUTPUT_PATH = BASE_DIR / "absa_sample_results.parquet"

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
N_ROWS = 100
MAX_ASPECTS_PER_REVIEW = 5  # cap aspects/review so one long review can't blow up the batch
BATCH_SIZE = 16  # (review, aspect) pairs per forward pass -- lower if you hit OOM
MAX_LENGTH = 512  # hard token cap; texts are truncated, never crash on overflow

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exc:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' is not installed. Run:\n    python -m spacy download en_core_web_sm"
        ) from exc


def extract_aspect_candidates(nlp, text: str, max_aspects: int = MAX_ASPECTS_PER_REVIEW):
    """Cheap, model-free aspect-term extraction: unique noun tokens, in order of appearance."""
    doc = nlp(text)
    candidates = []
    seen = set()
    for token in doc:
        if token.pos_ != "NOUN" or token.is_stop or not token.is_alpha:
            continue
        term = token.text.lower()
        if len(term) < 3 or term in seen:
            continue
        seen.add(term)
        candidates.append(term)
        if len(candidates) >= max_aspects:
            break
    return candidates


def classify_pairs(tokenizer, model, pairs, batch_size=BATCH_SIZE):
    """
    pairs: list[(review_text, aspect)]
    returns: list[str] sentiment labels, same order as `pairs`.

    Each batch is wrapped in its own try/except: a single failed batch (e.g. a
    transient CUDA OOM) is marked "Error" for those items instead of crashing
    the whole run.
    """
    labels = [None] * len(pairs)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        sentences = [p[0] for p in chunk]
        aspects = [p[1] for p in chunk]
        try:
            inputs = tokenizer(
                sentences,
                aspects,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1).tolist()
            for i, pid in enumerate(pred_ids):
                labels[start + i] = model.config.id2label[pid]

        except RuntimeError as exc:
            print(f"[warn] batch {start}-{start + len(chunk)} failed: {exc}")
            for i in range(len(chunk)):
                labels[start + i] = "Error"
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return labels


def main():
    t_start = time.perf_counter()

    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    print(f"[load] Reading first {N_ROWS} rows from {INPUT_PATH.name}")
    df = pl.scan_parquet(INPUT_PATH).head(N_ROWS).collect()

    print("[load] Loading spaCy aspect extractor (en_core_web_sm)...")
    nlp = load_spacy_model()

    print(f"[load] Loading ABSA model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    print(f"[load] Running on device: {DEVICE}")

    # Stage 1: extract aspect candidates per review.
    # Handles: null/empty reviewText, and warns (without crashing) on texts
    # that exceed the model's token limit -- they'll simply be truncated.
    texts = df["reviewText"].to_list()
    per_row_aspects = []
    for row_idx, text in enumerate(texts):
        if text is None or not isinstance(text, str) or not text.strip():
            per_row_aspects.append([])
            continue
        try:
            token_len = len(tokenizer(text, truncation=False)["input_ids"])
            if token_len > MAX_LENGTH:
                print(
                    f"[warn] row {row_idx}: reviewText has {token_len} tokens "
                    f"(> {MAX_LENGTH}), will be truncated"
                )
            per_row_aspects.append(extract_aspect_candidates(nlp, text))
        except Exception as exc:
            print(f"[warn] row {row_idx}: aspect extraction failed ({exc}), skipping")
            per_row_aspects.append([])

    # Stage 2: flatten (review, aspect) pairs across all rows into one batch job.
    flat_pairs = []  # (row_idx, aspect, text)
    for row_idx, (text, aspects) in enumerate(zip(texts, per_row_aspects)):
        for aspect in aspects:
            flat_pairs.append((row_idx, aspect, text))

    print(f"[infer] Classifying {len(flat_pairs):,} (review, aspect) pairs (batch_size={BATCH_SIZE})...")
    t0 = time.perf_counter()
    sentiments = classify_pairs(
        tokenizer,
        model,
        [(text, aspect) for _, aspect, text in flat_pairs],
    )
    print(f"[infer] Done in {time.perf_counter() - t0:.1f}s")

    # Regroup sentiments back into one {aspect: sentiment} dict per row.
    results = [dict() for _ in range(len(texts))]
    for (row_idx, aspect, _), sentiment in zip(flat_pairs, sentiments):
        results[row_idx][aspect] = sentiment

    df = df.with_columns(pl.Series("aspect_sentiments", [json.dumps(r, ensure_ascii=False) for r in results]))

    df.write_parquet(OUTPUT_PATH)
    print(f"[write] Results -> {OUTPUT_PATH.name}")

    with pl.Config(fmt_str_lengths=200, tbl_width_chars=200):
        print(df.select(["asin", "overall", "reviewText", "aspect_sentiments"]))

    total = time.perf_counter() - t_start
    print(f"\nTotal execution time: {total:.1f}s")


if __name__ == "__main__":
    main()
