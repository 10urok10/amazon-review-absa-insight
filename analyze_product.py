"""
CLI wrapper around insight_engine.py -- the on-demand ABSA product insight
pipeline. See insight_engine.py for how the pipeline itself works.

Usage:
    python analyze_product.py --asin B0001234
    python analyze_product.py --asin B0001234 --force-refresh
"""

import argparse
import sys
import time

from insight_engine import get_or_compute_insight

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def print_report(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"ASIN: {result['asin']}")
    if result["title"]:
        print(f"Product: {result['title']}")
    print(f"Average rating: {result['avg_rating']:.2f} / 5  ({result['review_count']} reviews)")
    print("-" * 70)
    print("[EN]", result["insight_text"])
    print()
    print("[TR]", result.get("insight_text_tr") or "(not available -- run with --force-refresh to regenerate)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="On-demand ABSA product insight tool")
    parser.add_argument("--asin", required=True, help="Amazon product ASIN to analyze")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and recompute")
    args = parser.parse_args()

    t_start = time.perf_counter()
    result = get_or_compute_insight(args.asin, force_refresh=args.force_refresh, progress=print)

    if result["from_cache"]:
        print(f"[cache] HIT (cached at {result['created_at']}) -- no GPU, no LLM call")
    print_report(result)
    print(f"\nTotal execution time: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
