"""
Tests for the pure, GPU/API-free logic in insight_engine.py.

Deliberately scoped to what can run in CI with no GPU, no pyabsa model
download, and no live Gemini API call: aggregation, response parsing, the
SQLite cache, and product search against a small fixture index. Anything
that touches pyabsa inference or the Gemini API is out of scope here.
"""

import json
import sqlite3

import polars as pl
import pytest

import insight_engine as ie


# ---------------------------------------------------------------------------
# aggregate_aspect_stats
# ---------------------------------------------------------------------------
def test_aggregate_aspect_stats_basic():
    results = [
        {"aspect": ["battery", "screen"], "sentiment": ["Negative", "Positive"]},
        {"aspect": ["battery"], "sentiment": ["Positive"]},
    ]
    stats = ie.aggregate_aspect_stats(results)
    assert stats["battery"] == {"Positive": 1, "Neutral": 0, "Negative": 1}
    assert stats["screen"] == {"Positive": 1, "Neutral": 0, "Negative": 0}


def test_aggregate_aspect_stats_applies_synonym_map():
    results = [
        {
            "aspect": ["install", "cable", "cables", "ipad"],
            "sentiment": ["Positive", "Negative", "Negative", "Neutral"],
        },
    ]
    stats = ie.aggregate_aspect_stats(results)
    assert "install" not in stats
    assert "cable" not in stats and "cables" not in stats
    assert "ipad" not in stats
    assert stats["installation"] == {"Positive": 1, "Neutral": 0, "Negative": 0}
    assert stats["charger"] == {"Positive": 0, "Neutral": 0, "Negative": 2}
    assert stats["tablet"] == {"Positive": 0, "Neutral": 1, "Negative": 0}


def test_aggregate_aspect_stats_skips_none_and_empty_results():
    results = [None, {"aspect": [], "sentiment": []}, None]
    assert ie.aggregate_aspect_stats(results) == {}


def test_aggregate_aspect_stats_is_case_insensitive():
    results = [{"aspect": ["Battery", "BATTERY"], "sentiment": ["Positive", "Negative"]}]
    stats = ie.aggregate_aspect_stats(results)
    assert stats == {"battery": {"Positive": 1, "Neutral": 0, "Negative": 1}}


def test_aggregate_aspect_stats_ignores_unrecognized_sentiment_label():
    results = [{"aspect": ["battery"], "sentiment": ["Confused"]}]
    stats = ie.aggregate_aspect_stats(results)
    assert stats["battery"] == {"Positive": 0, "Neutral": 0, "Negative": 0}


# ---------------------------------------------------------------------------
# parse_bilingual_insight
# ---------------------------------------------------------------------------
def test_parse_bilingual_insight_normal_format():
    raw = "EN: Battery life is a common complaint.\nTR: Pil ömrü yaygın bir şikayet."
    result = ie.parse_bilingual_insight(raw)
    assert result["english"] == "Battery life is a common complaint."
    assert result["turkish"] == "Pil ömrü yaygın bir şikayet."


def test_parse_bilingual_insight_multiline_sections():
    raw = "EN: Line one.\nLine two.\nTR: Satır bir.\nSatır iki."
    result = ie.parse_bilingual_insight(raw)
    assert result["english"] == "Line one.\nLine two."
    assert result["turkish"] == "Satır bir.\nSatır iki."


def test_parse_bilingual_insight_missing_format_falls_back_to_raw_as_english():
    raw = "Just a plain response with no format markers."
    result = ie.parse_bilingual_insight(raw)
    assert result["english"] == raw
    assert result["turkish"] == ""


# ---------------------------------------------------------------------------
# Cache (init_cache_db / get_cached) -- isolated via a temp sqlite file
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_cache_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_cache.sqlite3"
    monkeypatch.setattr(ie, "CACHE_DB_PATH", db_path)
    return db_path


def test_init_cache_db_creates_full_schema(temp_cache_db):
    conn = ie.init_cache_db()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(product_cache)")}
    assert {
        "asin", "title", "review_count", "avg_rating", "aspect_stats_json",
        "insight_text", "insight_text_tr", "created_at",
    } <= cols
    conn.close()


def test_init_cache_db_migrates_schema_missing_insight_text_tr(temp_cache_db):
    # Simulate a DB created before the insight_text_tr column existed.
    conn = sqlite3.connect(temp_cache_db)
    conn.execute("""
        CREATE TABLE product_cache (
            asin TEXT PRIMARY KEY, title TEXT, review_count INTEGER,
            avg_rating REAL, aspect_stats_json TEXT, insight_text TEXT, created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

    conn = ie.init_cache_db()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(product_cache)")}
    assert "insight_text_tr" in cols
    conn.close()


def test_get_cached_roundtrip(temp_cache_db):
    conn = ie.init_cache_db()
    conn.execute(
        """INSERT INTO product_cache
           (asin, title, review_count, avg_rating, aspect_stats_json,
            insight_text, insight_text_tr, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            "B000TEST01", "Test Product", 5, 4.2,
            json.dumps({"battery": {"Positive": 2, "Neutral": 0, "Negative": 3}}),
            "English insight", "Türkçe içgörü",
        ),
    )
    conn.commit()

    result = ie.get_cached(conn, "B000TEST01")
    assert result["title"] == "Test Product"
    assert result["avg_rating"] == 4.2
    assert result["insight_text_tr"] == "Türkçe içgörü"
    assert result["aspect_stats"] == {"battery": {"Positive": 2, "Neutral": 0, "Negative": 3}}


def test_get_cached_returns_none_for_missing_asin(temp_cache_db):
    conn = ie.init_cache_db()
    assert ie.get_cached(conn, "NOPE") is None


# ---------------------------------------------------------------------------
# search_products -- isolated via a small temp parquet fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_product_index(tmp_path, monkeypatch):
    index_path = tmp_path / "test_product_index.parquet"
    pl.DataFrame({
        "asin": ["A1", "A2", "A3"],
        "title": ["Wireless Bluetooth Speaker", "USB Wall Charger", "Bluetooth Headphones"],
        "review_count": [10, 500, 50],
        "avg_rating": [4.5, 3.8, 4.1],
    }).write_parquet(index_path)
    monkeypatch.setattr(ie, "PRODUCT_INDEX_PATH", index_path)
    return index_path


def test_search_products_matches_case_insensitive(temp_product_index):
    results = ie.search_products("bluetooth")
    titles = {r["title"] for r in results}
    assert titles == {"Wireless Bluetooth Speaker", "Bluetooth Headphones"}


def test_search_products_sorted_by_review_count_desc(temp_product_index):
    results = ie.search_products("bluetooth")
    assert results[0]["asin"] == "A3"  # 50 reviews > 10 reviews


def test_search_products_empty_keyword_returns_empty_without_touching_index(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "PRODUCT_INDEX_PATH", tmp_path / "does_not_exist.parquet")
    assert ie.search_products("   ") == []


def test_search_products_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "PRODUCT_INDEX_PATH", tmp_path / "does_not_exist.parquet")
    with pytest.raises(FileNotFoundError):
        ie.search_products("anything")
