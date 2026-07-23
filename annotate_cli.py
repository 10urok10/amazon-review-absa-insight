"""
Fast keyboard-driven annotator for annotation_batch.xlsx.

Speeds up manual review three ways vs. editing the spreadsheet by hand:
  1. One keystroke per row for the common case (Enter = correct).
  2. Rows are shown lowest-model-confidence-first (using the per-aspect
     confidence in annotation_batch_raw.jsonl), so the most error-prone
     predictions get reviewed first -- if you stop early, you've already
     covered the rows most likely to matter for error analysis.
  3. Saves after every row, so you can quit (q) anytime and resume later;
     already-annotated rows are skipped on the next run.

Correction shorthand: instead of typing raw JSON, type
    aspect:Sentiment,aspect2:Sentiment2
e.g. "battery:Negative,screen:Positive" -- it's converted to JSON for you.

Usage:
    python annotate_cli.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "annotation_batch.xlsx"
RAW_JSONL_PATH = BASE_DIR / "annotation_batch_raw.jsonl"

VALID_VERDICTS = {
    "": "correct",       # Enter alone -> correct (fast path for the common case)
    "c": "correct",
    "p": "partial",
    "i": "incorrect",
    "n": "no_aspect",
    "m": "missed_all",
}
VALID_SENTIMENTS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
NEEDS_CORRECTION = {"partial", "incorrect", "missed_all"}


def load_confidence_map():
    """review_id -> min confidence across its predicted aspects (0.5 placeholder
    for rows with no predicted aspects, so they land in the middle of the queue
    rather than artificially first or last)."""
    conf_map = {}
    if not RAW_JSONL_PATH.exists():
        return conf_map
    with RAW_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            raw = rec.get("pyabsa_raw")
            confidences = (raw or {}).get("confidence") or []
            conf_map[rec["review_id"]] = min(confidences) if confidences else 0.5
    return conf_map


def parse_shorthand(text):
    """'battery:Negative,screen:Positive' -> {'battery': 'Positive', ...} or None on bad input."""
    text = text.strip()
    if not text:
        return {}
    result = {}
    for pair in text.split(","):
        if ":" not in pair:
            return None
        aspect, sentiment = pair.split(":", 1)
        aspect = aspect.strip().lower()
        sentiment_norm = VALID_SENTIMENTS.get(sentiment.strip().lower())
        if not aspect or sentiment_norm is None:
            return None
        result[aspect] = sentiment_norm
    return result


def prompt_correction():
    while True:
        raw = input("  Doğru aspect listesi (aspect:Sentiment,aspect2:Sentiment2 | boş=aspect yok): ")
        parsed = parse_shorthand(raw)
        if parsed is not None:
            return parsed
        print("  Geçersiz format. Örnek: battery:Negative,screen:Positive")


def safe_save(df):
    """Retries on PermissionError (file open in Excel) instead of crashing
    and losing the in-memory annotation progress for this session."""
    while True:
        try:
            df.drop(columns=["_confidence"]).to_excel(XLSX_PATH, index=False, engine="openpyxl")
            return
        except PermissionError:
            input(f"\n[!] {XLSX_PATH.name} başka bir programda açık görünüyor "
                  f"(muhtemelen Excel). Dosyayı kapat ve devam etmek için Enter'a bas... ")


def format_predicted(value):
    if value is None or (isinstance(value, float)):  # NaN from empty cell
        return "(pyabsa: aspect bulunamadı)"
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return str(value)
    if not parsed:
        return "(pyabsa: aspect bulunamadı)"
    return ", ".join(f"{a} -> {s}" for a, s in parsed.items())


def main():
    if not XLSX_PATH.exists():
        raise SystemExit(f"{XLSX_PATH.name} not found. Run generate_annotation_batch.py first.")

    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    for col in ("human_verdict", "corrected_aspects_json", "notes"):
        df[col] = df[col].fillna("").astype(str)

    conf_map = load_confidence_map()
    df["_confidence"] = df["review_id"].map(lambda rid: conf_map.get(rid, 0.5))

    pending_mask = df["human_verdict"].str.strip() == ""
    pending = df[pending_mask].sort_values("_confidence", ascending=True)

    total = len(df)
    already_done = total - len(pending)
    print(f"[start] {already_done}/{total} already annotated, {len(pending)} remaining "
          f"(sorted lowest-confidence-first)\n")

    if pending.empty:
        print("Her satır zaten etiketlenmiş. build_gold_standard.py çalıştırılabilir.")
        return

    done_this_session = 0
    for idx, row in pending.iterrows():
        print("=" * 80)
        print(f"[{already_done + done_this_session + 1}/{total}] review_id={row['review_id']} "
              f"asin={row['asin']} overall={row['overall']} "
              f"(model confidence: {row['_confidence']:.3f})")
        print("-" * 80)
        print(row["reviewText"])
        print("-" * 80)
        print("pyabsa tahmini:", format_predicted(row["predicted_aspects_json"]))
        print()

        while True:
            choice = input("[Enter]=doğru  p=partial  i=incorrect  n=aspect yok  "
                           "m=hepsi kaçmış  s=atla  q=kaydet&çık: ").strip().lower()
            if choice == "q":
                safe_save(df)
                print(f"\n[saved] {done_this_session} satır bu oturumda tamamlandı. "
                      f"Toplam: {already_done + done_this_session}/{total}")
                return
            if choice == "s":
                break
            if choice in VALID_VERDICTS:
                verdict = VALID_VERDICTS[choice]
                df.at[idx, "human_verdict"] = verdict
                if verdict in NEEDS_CORRECTION:
                    corrected = prompt_correction()
                    df.at[idx, "corrected_aspects_json"] = json.dumps(corrected, ensure_ascii=False) if corrected else ""
                else:
                    df.at[idx, "corrected_aspects_json"] = ""
                done_this_session += 1
                # Save after every row -- safe to interrupt anytime.
                safe_save(df)
                break
            print("Geçersiz tuş, tekrar dene.")
        print()

    print(f"\n[done] Tüm satırlar etiketlendi ({done_this_session} bu oturumda).")


if __name__ == "__main__":
    main()
