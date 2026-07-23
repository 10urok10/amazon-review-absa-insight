# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.10 conda environment named `amazon_absa`. All commands below assume it's active:

```
conda activate amazon_absa
```

GPU (CUDA) is required for the pyabsa model; `torch` must be installed from the CUDA wheel index
(`pip install torch --index-url https://download.pytorch.org/whl/cu128` -- not plain `pip install torch`),
otherwise pyabsa falls back to CPU and is much slower. `DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"`
in `insight_engine.py` handles the fallback automatically.

Secrets live in `.env` (loaded via `python-dotenv`): `GEMINI_API_KEY` is the one actively used
(`ANTHROPIC_API_KEY` is present from an earlier design iteration but unused by current code).

## Commands

```
# One-time data preparation (run in order, only needed once per machine / dataset refresh)
python preprocess.py              # raw JSON dumps -> processed_full_dataset.parquet (~7.8M rows)
python build_product_index.py     # processed_full_dataset.parquet -> product_index.parquet (search index)

# Day-to-day usage -- three interchangeable front ends over the same backend
python analyze_product.py --asin B0001234 [--force-refresh]   # CLI
streamlit run app.py                                            # web UI (search by product title)
python extension_server.py                                      # local backend for browser_extension/ (port 5057)

# Tests
pip install -r requirements-dev.txt
pytest -v
```

No linter or build step configured yet. `app.log` (console + file, via `logging_config.get_logger`)
holds operational diagnostics (retries, cache migrations, exceptions); it's separate from the
`progress` callback used for user-facing pipeline narration in the CLI/Streamlit/extension UIs.

## Architecture

**`insight_engine.py` is the single shared backend** -- every front end (CLI, Streamlit, browser
extension) calls into it rather than duplicating pipeline logic. When changing the ABSA/caching/LLM
pipeline, edit it there once; do not re-implement per front end. Shared constants (paths, model names,
`SYNONYM_MAP`, sentiment colors) live in `config.py`, imported by `insight_engine.py`, `app.py`, and
`build_product_index.py` rather than redefined per file. The Gemini call (`_call_gemini` in
`insight_engine.py`) is wrapped with `tenacity` retry/backoff (`GEMINI_MAX_RETRIES` in `config.py`) for
transient failures.

Pipeline for one product (`get_or_compute_insight` for a known ASIN, or `analyze_reviews_direct` for
reviews supplied directly by the browser extension):

1. Check `product_insights_cache.sqlite3` (keyed by ASIN) -- a hit returns instantly, no GPU/LLM call.
2. On a miss: get review text (either filtered out of `processed_full_dataset.parquet` by ASIN, or
   passed in directly), run pyabsa ATEPC (`run_absa`) for aspect extraction + sentiment per review.
3. Aggregate per-aspect sentiment counts, folding near-duplicate aspect terms through `SYNONYM_MAP`
   (e.g. "install"/"installation", "cable"/"cables" -> "charger", "ipad" -> "tablet"). Extend this map
   rather than adding a second aggregation mechanism.
4. `generate_insight_text` sends the aggregated stats to Gemini (`gemini-flash-latest`), asking for a
   natural-language insight in a fixed `EN:` / `TR:` format that's parsed with regex into
   `{"english": ..., "turkish": ...}`. Both languages are stored and surfaced everywhere.
5. Write-through to the cache.

The pyabsa `AspectExtractor` is expensive to load, so it's cached as a module-level global
(`_extractor` in `insight_engine.py`) and reused across calls within a process -- do not construct a
new one per request.

**Data flow for the static dataset path:** `reviews_Electronics.json` (valid JSON-lines) +
`meta_Electronics.json` (NOT valid JSON -- Python-dict-literal-per-line, requires `ast.literal_eval`;
see `preprocess.py`'s conversion step) are joined by `preprocess.py` into
`processed_full_dataset.parquet`. `build_product_index.py` then precomputes a small
asin/title/review_count/avg_rating index (`product_index.parquet`) so the Streamlit search box doesn't
scan 7.8M rows per keystroke.

**Browser extension (`browser_extension/`, Manifest V3) is the path for products *not* in the static
dataset.** Its content script (in `popup.js`) reads the Amazon product page's own already-rendered DOM
in the user's authenticated browser session -- it does not make independent requests to Amazon beyond
auto-clicking the page's own "show more reviews" button (`a[data-hook="show-more-button"]`, capped at
`MAX_SHOW_MORE_CLICKS`), which a human reading the page would also click. Scraped review text is POSTed
to `extension_server.py` (local Flask, port 5057, CORS-enabled for the extension origin), which calls
the same `insight_engine.py` functions. Two DOM-scraping details that took real iteration to get right
and will resurface if Amazon changes markup again:
- Newer Amazon review UIs (e.g. the `/portal/customer-reviews/` page) keep `data-hook="review-body"` on
  the text span but drop the classic enclosing `<div data-hook="review">` wrapper (it's an `<li>` now,
  or absent) -- so the scraper selects `span[data-hook="review-body"]` directly and walks up parents to
  find the nearby rating icon, rather than anchoring to an outer review container.
- Star rating must be read from the icon's CSS class (`a-star-5`, `a-star-4-5`, etc.), not the
  human-readable alt text -- English phrasing ("5.0 out of 5 stars") puts the value first, Turkish
  phrasing ("5 yıldız üzerinden 5,0") puts it last, so locale changes the word order of any text-based
  parse.

**Legacy/experimental scripts** (`absa_pipeline.py`, `pyabsa_pipeline.py`, `eda_analysis.py`,
`generate_annotation_batch.py`, `annotate_cli.py`) predate `insight_engine.py` and were exploratory
steps (spaCy-based ABSA before switching to pyabsa, EDA charting, a human-annotation workflow for
evaluating pyabsa). They are not part of the current pipeline and are not wired into the three front
ends above.

Windows-specific: most scripts call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` because
the default Windows console codepage (cp1254 on Turkish Windows) crashes on Unicode review text
otherwise.
