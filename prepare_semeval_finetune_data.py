"""
Converts the SemEval-2014 Task 4 Laptop training data (`Laptops_Train.xml`,
downloaded manually from metashare.ilsp.gr -- licensed Academic/Non-Commercial/
No-Redistribution, hence gitignored and never committed) into the CoNLL-style
token/IOB-tag/polarity format pyabsa's ATEPCTrainer expects
(`<dataset_name>.train.txt.atepc` / `.valid.txt.atepc`, one "token TAG polarity"
line per token, blank line between sentences -- see
pyabsa's AspectTermExtraction/dataset_utils/__lcf__/data_utils_for_training.py
readfile() for the exact parser this mirrors).

This is genuinely independent, human-labeled data (not pyabsa's own output),
so fine-tuning on it is legitimate -- unlike the self-distillation approach
already rejected for this project.

Usage:
    python prepare_semeval_finetune_data.py
"""

import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR / "Laptops_Train.xml"
OUT_DIR = BASE_DIR / "semeval_finetune_data"
RANDOM_SEED = 42
VALID_FRACTION = 0.1

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
VALID_POLARITIES = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}


def tokenize_with_spans(text: str):
    tokens, spans = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def tag_sentence(text: str, aspect_terms: list[tuple[str, str, int, int]]):
    """Returns (tokens, iob_tags, polarities) for one sentence. Drops
    "conflict"-polarity aspects (not a class pyabsa's 3-way sentiment head
    supports) and any aspect whose span doesn't land cleanly on token
    boundaries."""
    tokens, spans = tokenize_with_spans(text)
    tags = ["O"] * len(tokens)
    polarities = ["-100"] * len(tokens)

    for _term, polarity, start, end in aspect_terms:
        mapped_polarity = VALID_POLARITIES.get(polarity)
        if mapped_polarity is None:  # "conflict" or unrecognized
            continue
        overlapping = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
        if not overlapping:
            continue
        first = True
        for i in overlapping:
            if tags[i] != "O":
                continue  # already covered by an earlier aspect span, don't overwrite
            tags[i] = "B-ASP" if first else "I-ASP"
            polarities[i] = mapped_polarity
            first = False

    return tokens, tags, polarities


def write_conll(path: Path, sentences: list[tuple[list[str], list[str], list[str]]]):
    with path.open("w", encoding="utf-8") as f:
        for tokens, tags, polarities in sentences:
            for tok, tag, pol in zip(tokens, tags, polarities, strict=True):
                f.write(f"{tok} {tag} {pol}\n")
            f.write("\n")


def main():
    if not XML_PATH.exists():
        raise SystemExit(f"{XML_PATH.name} not found -- see the download instructions in CLAUDE.md.")

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    all_sentences = []
    n_conflict_dropped = 0
    n_no_aspect = 0
    for sentence_el in root.iter("sentence"):
        text_el = sentence_el.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text

        aspect_terms = []
        aspect_terms_el = sentence_el.find("aspectTerms")
        if aspect_terms_el is not None:
            for at in aspect_terms_el.findall("aspectTerm"):
                polarity = at.get("polarity")
                if polarity == "conflict":
                    n_conflict_dropped += 1
                    continue
                aspect_terms.append((at.get("term"), polarity, int(at.get("from")), int(at.get("to"))))
        else:
            n_no_aspect += 1

        tokens, tags, polarities = tag_sentence(text, aspect_terms)
        all_sentences.append((tokens, tags, polarities))

    print(f"Parsed {len(all_sentences)} sentences from {XML_PATH.name}")
    print(f"  {n_conflict_dropped} conflict-polarity aspects dropped")
    print(f"  {n_no_aspect} sentences had no aspectTerms element (no aspects)")

    rng = random.Random(RANDOM_SEED)
    shuffled = all_sentences[:]
    rng.shuffle(shuffled)
    n_valid = int(len(shuffled) * VALID_FRACTION)
    valid_set = shuffled[:n_valid]
    train_set = shuffled[n_valid:]

    OUT_DIR.mkdir(exist_ok=True)
    write_conll(OUT_DIR / "semeval_finetune_data.train.txt.atepc", train_set)
    write_conll(OUT_DIR / "semeval_finetune_data.valid.txt.atepc", valid_set)

    print(f"Wrote {len(train_set)} train / {len(valid_set)} valid sentences to {OUT_DIR}/")


if __name__ == "__main__":
    main()
