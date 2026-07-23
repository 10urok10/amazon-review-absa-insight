"""
Exploratory Data Analysis on pyabsa_sample_results.parquet.

Flattens the per-review {aspect: sentiment} dicts (or null, when no aspects
were found) into one row per (aspect, sentiment) pair, finds the 15 most
frequently mentioned aspects, and renders their sentiment-proportion
breakdown as a stacked bar chart.

Usage:
    python eda_analysis.py
"""

import ast
import importlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Dependency check
# ---------------------------------------------------------------------------
REQUIRED_ALWAYS = ["pandas", "matplotlib", "seaborn"]
PARQUET_ENGINES = ["pyarrow", "fastparquet"]  # need at least one


def _has(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def check_dependencies():
    missing = [p for p in REQUIRED_ALWAYS if not _has(p)]
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall with: pip install "
            + " ".join(missing)
        )
    if not any(_has(p) for p in PARQUET_ENGINES):
        raise SystemExit(
            "No parquet engine found (need pyarrow or fastparquet).\nInstall with: pip install pyarrow"
        )
    engine = next(p for p in PARQUET_ENGINES if _has(p))
    print(f"[check] pandas/matplotlib/seaborn OK, parquet engine: {engine}")


check_dependencies()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "pyabsa_sample_results.parquet"
OUTPUT_CHART_PATH = BASE_DIR / "aspect_sentiment_chart_mapped.png"

TOP_N = 15

# Groups near-duplicate/related aspect terms under one canonical name before
# the top-N ranking is computed, so mention counts aren't split across
# synonyms (e.g. "install" and "installation" counted separately).
SYNONYM_MAP = {
    "install": "installation",
    "cable": "charger",
    "cables": "charger",
    "ipad": "tablet",
}

# Fixed status colors (never themed / never reused for anything else) --
# validated for contrast on a light chart surface.
SENTIMENT_COLORS = {
    "Positive": "#0ca30c",  # status: good
    "Neutral": "#898781",  # neutral gray
    "Negative": "#d03b3b",  # status: critical
}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]  # fixed stacking order, every bar

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_cell(value):
    """aspect_sentiments cells are JSON strings (or already-parsed dicts);
    null rows are dropped upstream, so this only needs to handle the rest."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return None


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    print(f"[load] Reading {INPUT_PATH.name}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"[load] {len(df):,} rows loaded")

    col = "aspect_sentiments"
    if col not in df.columns:
        raise SystemExit(f"Expected column '{col}' not found. Columns: {list(df.columns)}")

    # Step: drop null rows (no aspects found for that review).
    non_null = df[col].dropna()
    print(
        f"[flatten] {len(non_null):,}/{len(df):,} rows have aspect data "
        f"({len(df) - len(non_null):,} null rows dropped)"
    )

    # Step: explode each {aspect: sentiment} dict into its own (aspect, sentiment)
    # row, lowercasing the aspect and folding it through SYNONYM_MAP so related
    # terms (e.g. "install"/"installation") are counted as one aspect.
    records = []
    for value in non_null:
        parsed = parse_cell(value)
        if not parsed:
            continue
        for aspect, sentiment in parsed.items():
            aspect_norm = aspect.lower()
            aspect_norm = SYNONYM_MAP.get(aspect_norm, aspect_norm)
            records.append((aspect_norm, sentiment))

    flat_df = pd.DataFrame(records, columns=["aspect", "sentiment"])
    print(
        f"[flatten] Exploded into {len(flat_df):,} aspect-sentiment pairs "
        f"(synonym-mapped: {sorted(set(SYNONYM_MAP.values()))})"
    )

    if flat_df.empty:
        raise SystemExit("No aspect-sentiment pairs found after flattening; nothing to plot.")

    # Step: top 15 aspects by total mention count.
    top_aspects = flat_df["aspect"].value_counts().head(TOP_N).index.tolist()
    print(f"[analyze] Top {len(top_aspects)} aspects: {top_aspects}")

    top_df = flat_df[flat_df["aspect"].isin(top_aspects)]
    counts = pd.crosstab(top_df["aspect"], top_df["sentiment"])

    # Guarantee every expected sentiment column exists, and surface (rather
    # than silently drop/miscolor) any label outside the three we know about.
    for sentiment in SENTIMENT_ORDER:
        if sentiment not in counts.columns:
            counts[sentiment] = 0
    unexpected = [c for c in counts.columns if c not in SENTIMENT_ORDER]
    if unexpected:
        print(f"[warn] Ignoring unexpected sentiment label(s) in data: {unexpected}")
    counts = counts[SENTIMENT_ORDER]

    # Sort ascending by total so the largest bar ends up plotted last -> at
    # the top of the horizontal chart (most-mentioned aspect on top).
    counts["__total__"] = counts.sum(axis=1)
    counts = counts.sort_values("__total__", ascending=True)
    totals = counts.pop("__total__")

    # -------------------------------------------------------------------
    # Plot: horizontal stacked bar chart
    # -------------------------------------------------------------------
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    left = pd.Series(0, index=counts.index, dtype=float)
    for sentiment in SENTIMENT_ORDER:
        values = counts[sentiment]
        ax.barh(
            counts.index,
            values,
            left=left,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor=CHART_SURFACE,  # thin surface gap between stacked segments
            linewidth=1.2,
            label=sentiment,
            height=0.68,
        )
        left = left + values

    ax.set_xlabel("Number of mentions", color=SECONDARY_INK, fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        f"Top {TOP_N} Aspects by Mention Count, Split by Sentiment (Synonym-Mapped)",
        color=PRIMARY_INK,
        fontsize=14,
        fontweight="bold",
        pad=14,
        loc="left",
    )

    ax.tick_params(colors=SECONDARY_INK, labelsize=10)
    sns.despine(ax=ax, left=True, top=True, right=True)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Direct total labels at the end of each bar.
    max_total = max(totals)
    for i, total in enumerate(totals):
        ax.text(
            total + max_total * 0.01,
            i,
            f"{int(total)}",
            va="center",
            ha="left",
            color=SECONDARY_INK,
            fontsize=9,
        )

    legend = ax.legend(
        title="Sentiment",
        loc="lower right",
        frameon=False,
        labelcolor=PRIMARY_INK,
        fontsize=10,
        title_fontsize=10,
    )
    legend.get_title().set_color(PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(OUTPUT_CHART_PATH, dpi=300, facecolor=CHART_SURFACE)
    print(f"[write] Chart saved -> {OUTPUT_CHART_PATH.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
