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

from flask import Flask, jsonify, request
from flask_cors import CORS

from insight_engine import analyze_reviews_direct

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
CORS(app)  # extension origin (chrome-extension://...) needs cross-origin access


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    asin = data.get("asin")
    title = data.get("title")
    reviews = data.get("reviews") or []

    if not asin:
        return jsonify({"error": "asin is required"}), 400
    if not reviews:
        return jsonify({"error": "no reviews provided"}), 400

    print(f"[extension] Analyze request: asin={asin} title={title!r} reviews={len(reviews)}")

    try:
        result = analyze_reviews_direct(asin, title, reviews, progress=print)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"[extension] ERROR: {type(exc).__name__}: {exc}")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "asin": result["asin"],
        "title": result["title"],
        "review_count": result["review_count"],
        "avg_rating": result["avg_rating"],
        "insight_text": result["insight_text"],
        "insight_text_tr": result.get("insight_text_tr"),
        "aspect_stats": result["aspect_stats"],
        "from_cache": result["from_cache"],
        "created_at": result["created_at"],
    })


if __name__ == "__main__":
    print("[extension_server] Starting on http://127.0.0.1:5057 ...")
    app.run(host="127.0.0.1", port=5057, debug=False)
