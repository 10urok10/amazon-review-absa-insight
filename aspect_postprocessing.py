"""
REJECTED EXPERIMENT -- kept for the record, not wired into insight_engine.py,
not a dependency of the live pipeline (spaCy is intentionally NOT added to
requirements.txt/CI because of this).

Hypothesis (from analyze_absa_errors.py's real FP/FN inspection): dropping
aspect spans that are verbs/adjectives (e.g. "works") or that are entirely a
brand/product/person named entity (e.g. "logitech", "coby") should improve
precision without costing much recall, and lemmatizing terms (e.g. "priced"
-> "price") should reduce fragmentation.

Measured effect on the same 100 gold-standard rows (compare_postprocessing_effect.py),
against the baseline F1=0.887:
  POS filter alone:        F1=0.873
  NER filter alone:        F1=0.870
  POS+NER combined:        F1=0.855 (best precision of the variants, 0.893, but recall drops hard)
  + lemmatization:         F1=0.757 (lemma strings don't match the gold standard's
                           original-form corrections -- breaks previously-correct matches)

All variants make F1 *worse*, not better. Why, with concrete examples (see
debug output kept in this project's history): the hypothesis was too broad.
Verb-form aspects like "install", "charging", "cost", "fits", "support" were
very often *correctly* accepted by the human annotator as real aspects (a
review discussing installation/charging/cost/fit/support quality is a
legitimate aspect even though the word is grammatically a verb here) -- "all
verbs are hallucinations" is false. Likewise "Windows 8", "USB", "NSLU2",
"Vista" are legitimate aspects when they're the reviewed product's own
component/OS/model, not a comparison to a different product -- but spaCy's
NER can't distinguish "this product's own named component" from "a different
product being compared to it," so blanket-rejecting named entities throws
out real aspects along with the handful of genuine hallucinations (e.g.
"logitech" used only as a brand comparison). That distinction needs semantic
understanding a POS/NER rule doesn't have -- likely only a better model or
fine-tuning on more labeled examples can learn it.

Conclusion: rule-based post-processing is not a productive lever here. If
extraction accuracy needs to improve further, the next legitimate options are
a different/ensembled pyabsa checkpoint, or genuine fine-tuning on an
independent labeled dataset -- not hand-written filters.
"""

import spacy
from spacy.tokens import Doc

_NLP = None  # lazy-loaded singleton, mirrors _extractor in insight_engine.py

# Entity types that indicate "this span is a brand/product/person name," not
# a product feature -- e.g. "logitech", "coby", "kindle" (as a comparison).
_NAME_ENTITY_TYPES = {"ORG", "PRODUCT", "PERSON"}


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def filter_pyabsa_result(result: dict) -> dict:
    """Returns a new result dict with `aspect`/`sentiment` filtered and each
    surviving aspect term replaced by its lemma (so e.g. "priced" and "price"
    aggregate under one key downstream). Drops a span if none of its tokens
    are NOUN/PROPN (catches verb/adjective-only spans like "works"), or if
    every token in the span is tagged as an ORG/PRODUCT/PERSON named entity
    (catches brand/product-name spans like "logitech", "kindle"). Falls back
    to returning `result` unchanged if it lacks the token/position data
    needed (e.g. rows that failed and fell back to per-row retry paths)."""
    tokens = result.get("tokens") or []
    aspects = result.get("aspect") or []
    sentiments = result.get("sentiment") or []
    positions = result.get("position") or []
    if not tokens or not aspects:
        return result

    nlp = _get_nlp()
    doc = Doc(nlp.vocab, words=tokens)
    for _, proc in nlp.pipeline:
        doc = proc(doc)

    kept_aspects, kept_sentiments = [], []
    # Non-strict: aspect/sentiment/position arrays come from pyabsa's own
    # output, not something we construct -- tolerate a length mismatch
    # rather than crashing on a model quirk (same convention as
    # aggregate_aspect_stats in insight_engine.py).
    for aspect, sentiment, pos in zip(aspects, sentiments, positions):  # noqa: B905
        span_tokens = [doc[i] for i in pos if i < len(doc)]
        if not span_tokens:
            kept_aspects.append(aspect)
            kept_sentiments.append(sentiment)
            continue

        has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in span_tokens)
        is_name_entity = all(t.ent_type_ in _NAME_ENTITY_TYPES for t in span_tokens)
        if not has_noun or is_name_entity:
            continue

        lemma = " ".join(t.lemma_.lower() for t in span_tokens)
        kept_aspects.append(lemma)
        kept_sentiments.append(sentiment)

    return {**result, "aspect": kept_aspects, "sentiment": kept_sentiments}
