"""
Shared backend for the on-demand ABSA product insight tool.

This module has no CLI or UI code -- it's imported by both analyze_product.py
(CLI) and app.py (Streamlit), so the pipeline logic (cache, ABSA, LLM call)
lives in exactly one place.

Pipeline for a single ASIN:
  1. Check the SQLite cache -- a hit returns instantly, no GPU, no LLM call.
  2. On a miss: filter that product's reviews out of
     processed_full_dataset.parquet, run pyabsa ATEPC (aspect extraction +
     sentiment), aggregate with the same synonym mapping used in
     eda_analysis.py, and ask Gemini to turn the aggregated stats into a
     short natural-language insight -- surfacing aspects where sentiment
     contradicts the overall star rating.
  3. Save the result to the cache so the next lookup for this ASIN is free.
"""

import contextlib
import json
import logging
import os
import re
import sqlite3
import time

import polars as pl
import torch
from dotenv import load_dotenv
from google import genai
from pyabsa import ATEPCCheckpointManager
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from config import (
    BATCH_SIZE,
    CACHE_DB_PATH,
    CHECKPOINT,
    DEVICE,
    FULL_DATASET_PATH,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    MAX_REVIEWS_PER_PRODUCT,
    PRODUCT_INDEX_PATH,
    RANDOM_SEED,
    SYNONYM_MAP,
    USE_FP16_INFERENCE,
)
from logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

_extractor = None  # lazy-loaded, reused across calls within one process


def _noop(_message: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def init_cache_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS product_cache (
            asin TEXT PRIMARY KEY,
            title TEXT,
            review_count INTEGER,
            avg_rating REAL,
            aspect_stats_json TEXT,
            insight_text TEXT,
            insight_text_tr TEXT,
            created_at TEXT
        )
    """)
    # Migrate DBs created before insight_text_tr existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(product_cache)")}
    if "insight_text_tr" not in existing_cols:
        logger.info("Migrating %s: adding insight_text_tr column", CACHE_DB_PATH.name)
        conn.execute("ALTER TABLE product_cache ADD COLUMN insight_text_tr TEXT")
    conn.commit()
    return conn


def get_cached(conn: sqlite3.Connection, asin: str):
    row = conn.execute(
        "SELECT asin, title, review_count, avg_rating, aspect_stats_json, insight_text, "
        "insight_text_tr, created_at FROM product_cache WHERE asin = ?",
        (asin,),
    ).fetchone()
    if row is None:
        return None
    cols = [
        "asin",
        "title",
        "review_count",
        "avg_rating",
        "aspect_stats_json",
        "insight_text",
        "insight_text_tr",
        "created_at",
    ]
    result = dict(zip(cols, row, strict=True))
    result["aspect_stats"] = json.loads(result["aspect_stats_json"])
    return result


# ---------------------------------------------------------------------------
# Product search (reads the small precomputed index, not the 7.8M-row dataset)
# ---------------------------------------------------------------------------
def search_products(keyword: str, limit: int = 20) -> list[dict]:
    keyword = keyword.strip()
    if not keyword:
        return []
    if not PRODUCT_INDEX_PATH.exists():
        raise FileNotFoundError(f"{PRODUCT_INDEX_PATH.name} not found. Run build_product_index.py first.")
    matches = (
        pl.scan_parquet(PRODUCT_INDEX_PATH)
        .filter(pl.col("title").str.to_lowercase().str.contains(keyword.lower(), literal=True))
        .sort("review_count", descending=True)
        .head(limit)
        .collect()
    )
    return matches.to_dicts()


# ---------------------------------------------------------------------------
# ABSA pipeline
# ---------------------------------------------------------------------------
def fetch_reviews_for_asin(asin: str, progress=_noop) -> pl.DataFrame:
    df = pl.scan_parquet(FULL_DATASET_PATH).filter(pl.col("asin") == asin).collect()
    if df.height > MAX_REVIEWS_PER_PRODUCT:
        progress(f"{df.height} reviews found, sampling {MAX_REVIEWS_PER_PRODUCT} to bound processing time")
        df = df.sample(n=MAX_REVIEWS_PER_PRODUCT, seed=RANDOM_SEED, shuffle=True)
    return df


def _get_extractor(progress=_noop):
    global _extractor
    if _extractor is None:
        progress(f"Loading pyabsa ATEPC extractor (device={DEVICE})...")
        t0 = time.perf_counter()
        _extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=CHECKPOINT, auto_device=DEVICE)
        progress(f"Extractor ready in {time.perf_counter() - t0:.1f}s")
    return _extractor


def _inference_context():
    """torch.autocast (mixed FP16/FP32) on CUDA when enabled -- see
    benchmark_fp16.py for the measured speed/consistency tradeoff behind
    USE_FP16_INFERENCE. No-op on CPU or when disabled."""
    if USE_FP16_INFERENCE and DEVICE == "cuda:0":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _safe_predict_chunk(extractor, texts, progress=_noop):
    try:
        with _inference_context():
            return extractor.predict(texts, print_result=False, save_result=False, pred_sentiment=True)
    except Exception as exc:
        logger.warning(
            "ATEPC batch of %d failed (%s: %s); retrying rows individually",
            len(texts),
            type(exc).__name__,
            exc,
        )
        progress(f"Batch of {len(texts)} failed ({type(exc).__name__}: {exc}); retrying rows individually")
        if DEVICE == "cuda:0":
            torch.cuda.empty_cache()
        results = []
        for text in texts:
            try:
                with _inference_context():
                    r = extractor.predict([text], print_result=False, save_result=False, pred_sentiment=True)
                results.append(r[0])
            except Exception as exc2:
                logger.warning("ATEPC row failed individually (%s: %s), skipping", type(exc2).__name__, exc2)
                progress(f"Row failed individually ({type(exc2).__name__}: {exc2}), skipping")
                results.append(None)
        return results


def run_absa(reviews_df: pl.DataFrame, progress=_noop) -> dict:
    """Returns {aspect: {"Positive": n, "Neutral": n, "Negative": n}}, synonym-mapped."""
    extractor = _get_extractor(progress)

    texts = reviews_df["reviewText"].to_list()
    valid_indices, valid_texts = [], []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)

    results_by_row = [None] * len(texts)
    progress(f"Running ATEPC inference on {len(valid_texts):,} reviews (batch_size={BATCH_SIZE})...")
    t0 = time.perf_counter()
    for start in range(0, len(valid_texts), BATCH_SIZE):
        chunk_texts = valid_texts[start : start + BATCH_SIZE]
        chunk_indices = valid_indices[start : start + BATCH_SIZE]
        chunk_results = _safe_predict_chunk(extractor, chunk_texts, progress)
        for row_idx, result in zip(chunk_indices, chunk_results, strict=True):
            results_by_row[row_idx] = result
    progress(f"ABSA inference done in {time.perf_counter() - t0:.1f}s")

    return aggregate_aspect_stats(results_by_row)


def aggregate_aspect_stats(results: list) -> dict:
    """Pure aggregation step, split out of run_absa so it's testable without a
    GPU or the pyabsa model: takes a list of pyabsa result dicts (or None for
    rows that failed/were skipped) and folds them into
    {aspect: {"Positive": n, "Neutral": n, "Negative": n}}, applying
    SYNONYM_MAP so near-duplicate aspect terms are counted once."""
    aspect_stats = {}
    for result in results:
        if result is None:
            continue
        aspects = result.get("aspect") or []
        sentiments = result.get("sentiment") or []
        # Deliberately non-strict: aspects/sentiments come from pyabsa's own
        # output, not something we construct -- tolerate a length mismatch by
        # truncating rather than crashing the whole batch on a model quirk.
        for aspect, sentiment in zip(aspects, sentiments):  # noqa: B905
            norm = aspect.lower()
            norm = SYNONYM_MAP.get(norm, norm)
            bucket = aspect_stats.setdefault(norm, {"Positive": 0, "Neutral": 0, "Negative": 0})
            if sentiment in bucket:
                bucket[sentiment] += 1
    return aspect_stats


def generate_insight_text(asin: str, title, avg_rating: float, review_count: int, aspect_stats: dict) -> dict:
    """Returns {"english": str, "turkish": str}."""
    if not aspect_stats:
        return {
            "english": (
                "Not enough specific product-feature mentions were found in "
                "these reviews to generate an aspect-level insight."
            ),
            "turkish": (
                "Bu yorumlarda aspect seviyesinde bir icgoru uretmek icin "
                "yeterli spesifik urun ozelligi bahsi bulunamadi."
            ),
        }

    ranked = sorted(aspect_stats.items(), key=lambda kv: -sum(kv[1].values()))[:15]
    summary_lines = [
        f"- {aspect}: {counts['Positive']} positive, {counts['Neutral']} neutral, "
        f"{counts['Negative']} negative (of {sum(counts.values())} mentions)"
        for aspect, counts in ranked
    ]

    prompt = f"""You are analyzing customer reviews for an Amazon product.

Product: {title or asin}
Average star rating: {avg_rating:.2f} / 5 ({review_count} reviews)

Aspect-level sentiment breakdown (from automated review analysis):
{chr(10).join(summary_lines)}

Write a concise 2-4 sentence natural-language insight for a shopper considering
this product. Specifically call out:
1. Notable POSITIVE aspects reviewers consistently praise.
2. Notable NEGATIVE aspects reviewers consistently complain about -- especially
   if they contrast with the overall star rating (e.g. a high rating but a
   specific aspect people commonly dislike).
Be specific about which aspects, not generic. Do not repeat the raw numbers
verbatim -- synthesize them into plain, direct language.

Respond in EXACTLY this format, both parts required, no other text before or after:
EN: <the insight in English>
TR: <the same insight translated into natural Turkish>"""

    return parse_bilingual_insight(_call_gemini(prompt))


@retry(
    stop=stop_after_attempt(GEMINI_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini(prompt: str) -> str:
    """Isolated so retry/backoff wraps only the network call, not prompt
    construction. Transient failures (rate limits, 5xx, network blips) are
    retried with exponential backoff; a persistent failure re-raises after
    GEMINI_MAX_RETRIES attempts rather than silently swallowing the error."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def parse_bilingual_insight(raw: str) -> dict:
    """Pure parsing step, split out of generate_insight_text so the EN:/TR:
    parsing logic is testable without calling the Gemini API. Falls back to
    treating the whole response as English if the model didn't follow the
    requested format."""
    en_match = re.search(r"EN:\s*(.+?)(?=\n\s*TR:|\Z)", raw, re.DOTALL)
    tr_match = re.search(r"TR:\s*(.+)", raw, re.DOTALL)
    english = en_match.group(1).strip() if en_match else raw
    turkish = tr_match.group(1).strip() if tr_match else ""
    return {"english": english, "turkish": turkish}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_or_compute_insight(asin: str, force_refresh: bool = False, progress=_noop) -> dict:
    """Returns the same dict shape as get_cached(): asin, title, review_count,
    avg_rating, aspect_stats, insight_text, created_at, plus a bool
    'from_cache' the caller can use to show cache-hit/miss status."""
    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {FULL_DATASET_PATH}")

    conn = init_cache_db()

    if not force_refresh:
        cached = get_cached(conn, asin)
        if cached:
            cached["from_cache"] = True
            return cached

    progress(f"Cache miss for {asin} -- running full pipeline")
    reviews_df = fetch_reviews_for_asin(asin, progress)
    if reviews_df.height == 0:
        raise ValueError(f"No reviews found for ASIN {asin}.")

    titles = reviews_df["title"].drop_nulls().to_list()
    title = titles[0] if titles else None
    avg_rating = reviews_df["overall"].mean()
    review_count = reviews_df.height

    aspect_stats = run_absa(reviews_df, progress)
    progress(f"{len(aspect_stats)} distinct aspects found")

    progress(f"Generating natural-language insight via {GEMINI_MODEL}...")
    t0 = time.perf_counter()
    insight = generate_insight_text(asin, title, avg_rating, review_count, aspect_stats)
    progress(f"LLM done in {time.perf_counter() - t0:.1f}s")

    conn.execute(
        """INSERT OR REPLACE INTO product_cache
           (asin, title, review_count, avg_rating, aspect_stats_json, insight_text, insight_text_tr, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            asin,
            title,
            review_count,
            avg_rating,
            json.dumps(aspect_stats),
            insight["english"],
            insight["turkish"],
        ),
    )
    conn.commit()

    result = get_cached(conn, asin)
    result["from_cache"] = False
    return result


def analyze_reviews_direct(
    asin: str, title, reviews: list[dict], force_refresh: bool = False, progress=_noop
) -> dict:
    """Same pipeline and cache as get_or_compute_insight, but for reviews handed
    in directly (e.g. scraped from a live product page by the browser
    extension's content script) instead of looked up from
    processed_full_dataset.parquet.

    reviews: list of {"text": str, "rating": float | None}
    """
    conn = init_cache_db()

    if not force_refresh:
        cached = get_cached(conn, asin)
        if cached:
            cached["from_cache"] = True
            return cached

    texts = [r["text"].strip() for r in reviews if r.get("text") and r["text"].strip()]
    ratings = [r["rating"] for r in reviews if r.get("rating") is not None]
    if not texts:
        raise ValueError("No review text provided.")

    review_count = len(texts)
    avg_rating = sum(ratings) / len(ratings) if ratings else None

    reviews_df = pl.DataFrame({"reviewText": texts})
    aspect_stats = run_absa(reviews_df, progress)
    progress(f"{len(aspect_stats)} distinct aspects found")

    progress(f"Generating natural-language insight via {GEMINI_MODEL}...")
    t0 = time.perf_counter()
    insight = generate_insight_text(asin, title, avg_rating or 0.0, review_count, aspect_stats)
    progress(f"LLM done in {time.perf_counter() - t0:.1f}s")

    conn.execute(
        """INSERT OR REPLACE INTO product_cache
           (asin, title, review_count, avg_rating, aspect_stats_json, insight_text, insight_text_tr, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            asin,
            title,
            review_count,
            avg_rating,
            json.dumps(aspect_stats),
            insight["english"],
            insight["turkish"],
        ),
    )
    conn.commit()

    result = get_cached(conn, asin)
    result["from_cache"] = False
    return result
