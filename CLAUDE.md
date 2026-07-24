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

# Lint / format (also enforced in CI, see .github/workflows/ci.yml)
ruff check .
ruff format .
```

`app.log` (console + file, via `logging_config.get_logger`) holds operational diagnostics (retries,
cache migrations, exceptions); it's separate from the `progress` callback used for user-facing
pipeline narration in the CLI/Streamlit/extension UIs. GitHub Actions CI runs ruff (lint + format
check) and pytest on every push/PR to master, on CPU-only torch (no GPU in CI runners).

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

`USE_FP16_INFERENCE` (`config.py`) wraps inference in `torch.autocast(device_type="cuda",
dtype=torch.float16)` (`_inference_context()` in `insight_engine.py`) rather than a raw `.half()` cast,
so numerically sensitive ops (softmax, layer norm) stay in FP32. Benchmarked at 1.40x throughput with
*lower* peak GPU memory and 99.5% exact prediction match vs FP32 (see `benchmark_fp16.py`); only
applies on CUDA. `BATCH_SIZE` (also in `config.py`) was similarly chosen via `benchmark_batch_size.py`
on real data, not guessed -- re-run either benchmark script if you change GPUs or the checkpoint.

**Data flow for the static dataset path:** `reviews_Electronics.json` (valid JSON-lines) +
`meta_Electronics.json` (NOT valid JSON -- Python-dict-literal-per-line, requires `ast.literal_eval`;
see `preprocess.py`'s conversion step) are joined by `preprocess.py` into
`processed_full_dataset.parquet`. `build_product_index.py` then precomputes a small
asin/title/review_count/avg_rating index (`product_index.parquet`) so the Streamlit search box doesn't
scan 7.8M rows per keystroke.

**Browser extension (`browser_extension/`, Manifest V3) is the path for products *not* in the static
dataset, and the primary front end this project targets.** It reads the Amazon product page's own
already-rendered DOM in the user's authenticated browser session -- it does not make independent
requests to Amazon beyond auto-clicking the page's own "show more reviews" button
(`a[data-hook="show-more-button"]`, capped at `MAX_SHOW_MORE_CLICKS`), which a human reading the page
would also click.

Files, and why they're split this way:
- **`scraper.js`** -- `getAsinFromPage()` + `scrapeAmazonProduct()`. The only file that touches the
  Amazon DOM. Shared verbatim between the toolbar popup and the in-page content script (see below) so
  scraping logic exists in exactly one place.
- **`render.js`** -- `renderInsightResult(container, data)` + the `RESULT_PANEL_CSS` string, building
  the aspect bar chart HTML. Parameterized by `container` because it renders into two different DOM
  trees: the popup's own document, and a Shadow DOM panel injected into the Amazon page (Shadow DOM
  needs its own copy of the CSS since it's style-isolated from the host page's stylesheet).
- **`popup.js`** / **`popup.html`** -- the toolbar-icon-triggered popup. Injects `scraper.js` into the
  active tab via `chrome.scripting.executeScript({files: [...]})` then invokes
  `scrapeAmazonProduct()` in a second call (the `func:` form of `executeScript` only serializes the one
  function passed to it -- it can't see other top-level functions unless they were already injected
  into the page first).
- **`content.js`** -- declared in `manifest.json`'s `content_scripts` (loaded automatically alongside
  `scraper.js`/`render.js` on any Amazon product-page URL, no toolbar click required). Injects a
  floating "Bu Ürünü Analiz Et" button into a Shadow DOM host (`#absa-insight-host`, `all: initial` +
  its own `<style>`, so Amazon's page CSS can't bleed in and vice versa). On load, checks whether the
  page's ASIN is already cached and lights up a badge if so. Clicking the button opens a results panel
  with an explicit close (X) button; clicking the button again while the panel is open toggles it
  closed instead of re-running analysis.
- **`background.js`** -- a Manifest V3 service worker that both `popup.js` and `content.js` message
  (`chrome.runtime.sendMessage`) to actually perform the `fetch()` calls to `extension_server.py`.
  Necessary specifically for `content.js`: a fetch issued directly from a content script runs inside
  Amazon's own page origin and is subject to Amazon's page CSP (`connect-src`), which can block calls to
  `127.0.0.1:5057` even though the extension's `host_permissions` allow it. The background worker runs
  in the extension's own privileged context instead, sidestepping that.

`extension_server.py` exposes `POST /analyze` (full pipeline, returns `backend_seconds` alongside the
result so the UI can show real timing) and `GET /cache_status/<asin>` (SQLite-only lookup via
`get_cached`/`init_cache_db`, no GPU/LLM call -- used for the content script's cache badge so checking
every page load stays instant).

Two DOM-scraping details in `scraper.js` that took real iteration to get right and will resurface if
Amazon changes markup again:
- Newer Amazon review UIs (e.g. the `/portal/customer-reviews/` page) keep `data-hook="review-body"` on
  the text span but drop the classic enclosing `<div data-hook="review">` wrapper (it's an `<li>` now,
  or absent) -- so the scraper selects `span[data-hook="review-body"]` directly and walks up parents to
  find the nearby rating icon, rather than anchoring to an outer review container.
- Star rating must be read from the icon's CSS class (`a-star-5`, `a-star-4-5`, etc.), not the
  human-readable alt text -- English phrasing ("5.0 out of 5 stars") puts the value first, Turkish
  phrasing ("5 yıldız üzerinden 5,0") puts it last, so locale changes the word order of any text-based
  parse.

`manifest.json`'s `content_scripts.matches` lists Amazon TLDs explicitly (`*://*.amazon.com/*`,
`*://*.amazon.com.tr/*`, etc.) rather than a single wildcard pattern -- Chrome's match-pattern syntax
only allows a single leading `*.` subdomain wildcard, not a wildcard TLD, so `*://*.amazon.*/*` is
invalid and silently matches nothing.

**Legacy/experimental scripts** (`absa_pipeline.py`, `pyabsa_pipeline.py`, `eda_analysis.py`) predate
`insight_engine.py` and were exploratory steps (spaCy-based ABSA before switching to pyabsa, EDA
charting). They are not part of the current pipeline and are not wired into the three front ends above.

**Evaluation methodology (`generate_annotation_batch.py` -> `annotate_cli.py` -> `build_gold_standard.py`)
is a separate, ongoing effort to measure pyabsa's real accuracy** -- not wired into the live pipeline,
but not legacy either; this is the answer to "how accurate is pyabsa on this data" and should stay
the only source cited for that question.
1. `generate_annotation_batch.py` draws a stratified sample (40 rows per star rating 1-5 = 200 total,
   `RANDOM_SEED`-reproducible) from `processed_full_dataset.parquet`, runs pyabsa on it, and writes
   `annotation_batch.xlsx` for manual review (raw pyabsa output, including per-aspect confidence, is
   kept separately in `annotation_batch_raw.jsonl`).
2. `annotate_cli.py` is a keyboard-driven annotator over that workbook -- see `annotation_guidelines.md`
   for the labeling rules (aspect boundary definition, sentiment-is-about-the-aspect-not-the-review
   rule, wrong-variant-shipped / generic-DOA-complaint special cases). Rows are shown **lowest-model-
   confidence-first** so early stopping still covers the most error-prone predictions for qualitative
   error analysis -- but this also means an incomplete run is *not* a random subsample of the 200.
3. `build_gold_standard.py` turns a completed (or `--allow-partial`) workbook into real Precision/
   Recall/F1 for aspect term extraction and (measured separately) sentiment-classification accuracy
   given a correctly extracted aspect, both with bootstrap 95% CIs. It refuses to run on an incomplete
   batch unless `--allow-partial` is passed, and even then labels the output PARTIAL/EXPLORATORY -- per
   the confidence-first ordering in step 2, a partial run is systematically biased toward the model's
   hardest cases, not just imprecise (an earlier now-discarded 33-row run was mistakenly treated as
   final, which is what this guardrail prevents from recurring).

**Numbers, and why they're a rough signal, not a benchmark (`gold_standard_report.md`, committed at
`657db6d`):** the original 33-row state was discarded and re-annotated from scratch; the user annotated
100/200 of the stratified sample and deliberately stopped there -- aspect extraction P~0.88, R~0.90,
F1~0.89; sentiment accuracy given a correctly-extracted aspect~0.99. **Treat these as a rough directional
estimate only, never as a citable benchmark number**, for three compounding reasons: (1) confidence-first
annotation order means this 100 is the model's systematically hardest cases, not a random half; (2) it's
a single annotator with no inter-annotator agreement check, and the user has explicitly said (twice,
unprompted) that they don't trust their own labels; (3) `annotate_cli.py` shows pyabsa's own predicted
aspect before asking for a verdict, which risks anchoring the annotator toward agreeing with it. These
numbers still replace the old 33-row/78.8% figure, which remains permanently non-citable -- but replace
it with "a rough estimate," not "the real accuracy."

**Rejected experiment (`analyze_absa_errors.py`, `aspect_postprocessing.py`,
`compare_postprocessing_effect.py`):** a rule-based filter dropping verb/adjective-only aspect spans
and brand/product-name spans, plus lemmatizing terms, was tested against the same 100 gold rows and
made F1 *worse* (~0.89 -> 0.855 at best; 0.757 with lemmatization) -- verb-form aspects like "install"/
"cost"/"fits" and named entities like "Windows 8"/"USB" are often legitimately correct in this
annotation scheme, and POS/NER tagging can't reliably tell "this product's own named part" from "a
different product mentioned for comparison." Not wired into `insight_engine.py`; spaCy is intentionally
not in `requirements.txt`/CI because of this. This is a *small* effect size on an uncertain gold set --
the shakiest of the three rejected attempts below, "no real difference" is plausible. Do not re-propose
this exact filter design -- see the scripts' docstrings for the full measured breakdown.

**Also rejected: checkpoint swap and fine-tuning (`compare_checkpoints.py`,
`prepare_semeval_finetune_data.py`, `finetune_atepc.py`, `evaluate_finetuned_checkpoint.py`).**
pyabsa ships only one other ATEPC checkpoint besides "english" -- "multilingual" -- and it's much
worse on English text (F1=0.323 vs. ~0.89). Genuine fine-tuning (not self-distillation) was tried on
SemEval-2014 Task 4 Laptop data (277 docs, 3048 sentences, real term-level span annotations,
manually downloaded from metashare.ilsp.gr -- licensed Academic/Non-Commercial/No-Redistribution,
so `Laptops_Train.xml`/`archive.zip`/`semeval_finetune_data/` are gitignored, never committed).
Both an aggressive (5 epoch, lr=1e-5) and a light-touch (1 epoch, lr=2e-6) fine-tune, continuing from
`from_checkpoint="english"`, were evaluated against the same 100 Amazon gold rows (valid, non-circular
since SemEval Laptop reviews don't overlap with Amazon Electronics reviews) -- F1=0.273 and F1=0.326
respectively, both far below baseline via collapsed recall (0.19-0.21 vs 0.897). Fine-tuning on this
small, single-domain dataset makes the model more conservative about what counts as an aspect, losing
the broad recall the original checkpoint has from its much larger, more diverse training mix. Not a
"too aggressive" problem -- both doses failed the same way. Unlike the post-processing rejection above,
this ~3x gap (e.g. TP=209 vs TP=45-54) is large enough that annotator noise/uncertainty plausibly
doesn't explain it away -- a reasonably solid negative result even given the gold set's limitations.
**All three levers (post-processing, checkpoint swap, fine-tuning) are now closed; the project keeps
using the unmodified "english" checkpoint.** A future attempt would need a genuinely different, larger,
more Amazon/Electronics-relevant labeled dataset, not more tuning on SemEval Laptop -- and ideally a
sturdier gold standard (second annotator, blind labeling) before trusting precise numbers again.

Windows-specific: most scripts call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` because
the default Windows console codepage (cp1254 on Turkish Windows) crashes on Unicode review text
otherwise.
