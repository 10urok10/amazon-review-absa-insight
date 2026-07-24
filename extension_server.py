"""
Local backend for the "Amazon Ürün İçgörü" browser extension.

The browser extension's content script reads reviews already rendered on a
product page the user is personally viewing (no extra requests to Amazon --
it only parses the page that page-load already fetched) and POSTs that text
here. This server runs the same pyabsa ATEPC + Gemini pipeline used by
analyze_product.py / app.py, and shares the same SQLite cache keyed by ASIN.

Run:
    python extension_server.py

Then load browser_extension/ as an unpacked extension in Chrome/Edge.
"""

import sys
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import SYNONYM_MAP
from insight_engine import analyze_reviews_direct, get_cached, init_cache_db
from logging_config import get_logger

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)  # extension origin (chrome-extension://...) needs cross-origin access


def _aspect_search_terms(aspect_stats: dict) -> dict:
    """For each canonical aspect, the raw terms folded into it via SYNONYM_MAP
    plus the canonical name itself -- lets the extension's "scroll to review"
    feature match review text even when the review used a pre-synonym term
    (e.g. "cable") rather than the canonical one ("charger")."""
    reverse: dict[str, list[str]] = {}
    for raw, canonical in SYNONYM_MAP.items():
        reverse.setdefault(canonical, []).append(raw)
    return {aspect: [aspect, *reverse.get(aspect, [])] for aspect in aspect_stats}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/cache_status/<asin>", methods=["GET"])
def cache_status(asin):
    """Cheap SQLite-only lookup (no GPU/LLM) for the content script's badge."""
    conn = init_cache_db()
    try:
        cached = get_cached(conn, asin)
    finally:
        conn.close()
    if cached is None:
        return jsonify({"cached": False})
    return jsonify(
        {
            "cached": True,
            "title": cached["title"],
            "avg_rating": cached["avg_rating"],
            "created_at": cached["created_at"],
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    asin = data.get("asin")
    title = data.get("title")
    reviews = data.get("reviews") or []
    force_refresh = bool(data.get("force_refresh", False))

    if not asin:
        return jsonify({"error": "asin is required"}), 400
    if not reviews:
        return jsonify({"error": "no reviews provided"}), 400

    logger.info(
        "Analyze request: asin=%s title=%r reviews=%d force_refresh=%s",
        asin,
        title,
        len(reviews),
        force_refresh,
    )

    t0 = time.perf_counter()
    try:
        result = analyze_reviews_direct(asin, title, reviews, force_refresh=force_refresh, progress=print)
    except ValueError as exc:
        logger.warning("Analyze request rejected for asin=%s: %s", asin, exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Analyze request failed for asin=%s", asin)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    backend_seconds = time.perf_counter() - t0

    return jsonify(
        {
            "asin": result["asin"],
            "title": result["title"],
            "review_count": result["review_count"],
            "avg_rating": result["avg_rating"],
            "insight_text": result["insight_text"],
            "insight_text_tr": result.get("insight_text_tr"),
            "aspect_stats": result["aspect_stats"],
            "aspect_search_terms": _aspect_search_terms(result["aspect_stats"]),
            "from_cache": result["from_cache"],
            "created_at": result["created_at"],
            "backend_seconds": round(backend_seconds, 2),
        }
    )


if __name__ == "__main__":
    logger.info("Starting on http://127.0.0.1:5057 ...")
    app.run(host="127.0.0.1", port=5057, debug=False)
