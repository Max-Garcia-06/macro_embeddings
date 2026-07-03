---
type: results-report
date: 2026-07-02
experiment_line: source-a
round: 1
purpose: section-expansion-prototype
status: closed
source_artifacts:
  - 2026-07-02--source-a--r00--insights-summary.md
---

# Source A / Round 1 / Section-Expansion Prototype / 2026-07-02

## Question

Round 0 found intro-only embeddings carry only a weak geographic signal
(Mantel r=-0.094) and no clustering structure (silhouette <=0.034). Since
Wikipedia lead sections are short and heavily templated, would embedding more
of each county's article -- additional body sections, or a specific
content-rich section -- differentiate counties better?

## Method

Two variants prototyped on the same fixed 40-county random sample (seed=42,
drawn from the existing 50-state, non-stub subset of `source_a_embeddings.parquet`),
each re-fetched fresh via the Wikimedia Enterprise API and re-embedded with
the same `BAAI/bge-m3` model, then compared against that sample's existing
intro-only embeddings:

1. **Lead + next 3 body sections** (by Parsoid section id 0-3; typically
   History/Geography/Demographics).
2. **Economy section only** (matched by `<h2>` heading text, wherever it
   falls in the article).

Comparison metrics: Mantel test (geographic distance vs. embedding
similarity) and pairwise cosine-similarity mean/std, both on L2-normalized
embeddings. Not a full re-ingestion -- prototype only, no committed dataset
changes.

## Results

**Variant 1: lead + 3 body sections** (n=40, all fetches succeeded)

| Metric | Intro-only | Lead + 3 sections |
|---|---|---|
| Mean text length | 694 chars | 4,532 chars (6.5x) |
| Pairwise similarity mean | 0.559 | 0.585 |
| Pairwise similarity std | 0.068 | 0.054 |
| Mantel r (geo <-> similarity) | -0.272, p=0.004 | -0.372, p=0.002 |

Similarity mean rose and std fell -- counties became *more* alike, not less,
despite 6.5x more text. Consistent with History/Geography/Demographics
sections being even more templated across counties than the lead
("was formed in [year]... consists of [N] square miles... population was
[N] per the [year] census...").

**Variant 2: Economy section only** (n=10/40 -- 30 of 40 sampled counties had
no `<h2>` Economy section at all, mostly smaller/rural ones)

| Metric | Intro-only (same 10 counties) | Economy-only |
|---|---|---|
| Mean text length | 949 chars | 697 chars |
| Pairwise similarity mean | 0.556 | 0.509 |
| Pairwise similarity std | 0.069 | 0.072 |
| Mantel r (geo <-> similarity) | -0.287, p=0.016 | -0.093, p=0.494 (n.s.) |

Directionally more differentiating (lower mean similarity, marginally higher
std) on the counties that have this section, but coverage collapses to 25%
of the sample -- an Economy-only corpus would drop three-quarters of
counties before any embedding step, and n=10 is too small to trust the
direction on its own.

## Conclusion

Neither variant improves on the intro-only baseline in a usable way:
expanding to more sections reduces differentiation; narrowing to Economy
only differentiates slightly better but is not available for most counties.
**Closed without further section-expansion variants** -- not pursuing a
third attempt. If more differentiating signal is wanted later, a different
approach (e.g., targeted boilerplate-stripping within body sections, or a
non-Wikipedia source) would be needed rather than another section-selection
variant.

## Reproducibility

Prototype script (`prototype_expanded_sections.py`) was deleted after this
report was written -- it was throwaway and is fully described above;
re-creating it would take the same shape (sample via
`analyze_source_a_clusters.filter_to_fifty_states` +
`analyze_source_a_similarity.drop_stub_counties`, section isolation via
BeautifulSoup on Parsoid section ids/headings, re-embed with
`ingest_source_a.BgeM3EmbeddingGenerator`, compare via
`analyze_source_a_clusters.mantel_test`).
