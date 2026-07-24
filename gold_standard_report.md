# Gold-Standard ABSA Evaluation -- PARTIAL / EXPLORATORY

Sample: 100/200 rows annotated (stratified, 40 per star rating 1-5).
Bootstrap resamples: 3000 (row-level resampling, 95% CI).

**PARTIAL RUN -- do not cite as a final number.** Confidence-sorted annotation order means this subset skews toward the model's hardest cases.

## Aspect Term Extraction (exact aspect-key match)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | 0.878 | [0.803, 0.945] |
| Recall | 0.897 | [0.826, 0.953] |
| F1 | 0.887 | [0.829, 0.936] |

TP=209, FP=29, FN=24

## Sentiment Classification (given a correctly extracted aspect)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.986 | [0.967, 1.000] |

206/209 correctly-extracted aspects had the right polarity.

## By star rating

| Star | n rows | Precision | Recall | F1 | Sentiment acc. |
|---|---|---|---|---|---|
| 1 | 20 | 0.980 | 0.980 | 0.980 | 0.980 |
| 2 | 23 | 0.930 | 0.985 | 0.957 | 0.985 |
| 3 | 22 | 0.692 | 0.730 | 0.711 | 1.000 |
| 4 | 17 | 0.976 | 0.872 | 0.921 | 1.000 |
| 5 | 18 | 0.722 | 0.812 | 0.765 | 0.962 |

## human_verdict distribution

- correct: 76
- missed_all: 10
- no_aspect: 1
- partial: 13

## Flagged special cases (see annotation_guidelines.md)

- Wrong variant/spec shipped: 0
- Generic broken/DOA complaint (no specific aspect named): 0
