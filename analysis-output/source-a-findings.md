---
type: results-report
date: 2026-07-02
experiment_line: source-a
round: 2
purpose: consolidated-findings
status: active
supersedes:
  - 2026-07-02--source-a--r00--insights-summary.md
  - 2026-07-02--source-a--r01--section-expansion-prototype.md
  - 2026-07-02--source-a--r02--stability-and-coverage-followup.md
  - analysis-report.md
  - stats-appendix.md
  - figure-catalog.md
---

# Source A — Consolidated Findings / Wikipedia Intro-Text County Embeddings

> This file merges three previously separate round reports (r00 insights
> summary, r01 section-expansion prototype, r02 stability/coverage
> follow-up) plus the round-0 strict-analysis bundle (`analysis-report.md`,
> `stats-appendix.md`, `figure-catalog.md`) into one running document, to
> stop findings from fragmenting across files as the experiment line
> progresses. Those six files have been removed from `analysis-output/`;
> full history is preserved in git. `stats.json` and `figures/` are kept
> as-is (raw numbers / actual images).
>
> This repo is not bound to an Obsidian project knowledge base (no
> `.claude/project-memory/registry.yaml`), so this stays a local markdown
> artifact, not an Obsidian write-back.

## 1. Executive Summary

Source A's Wikipedia-intro-text county embeddings (`BAAI/bge-m3`, 1024-dim)
carry a real but small geographic signal. A Mantel permutation test finds a
weak, statistically significant negative correlation between geographic
distance and embedding similarity (r ≈ -0.09, p = 0.002). Neither PCA's
first component (4.9% of variance) nor K-means clustering (k=2, silhouette
≤0.035) finds strong or cleanly interpretable structure beyond that weak
geographic effect — **but** the k=2 split is highly reproducible across
random seeds (ARI≈0.99) and its ~10% geographic-coherence deviation from the
corpus mean is statistically real (permutation p≤0.001 both directions), not
sampling noise. A separate prototype found that embedding more article text
(additional sections, or an Economy-only section) does not improve
differentiation between counties. The dataset's Virginia independent-city
coverage gap (37 cities) has been backfilled; 19 unrelated ingestion
failures remain open.

## 2. Data & Setup

- **Source**: `source_a_embeddings.parquet` — 3,125 counties (after the
  Virginia backfill; was 3,088), `BAAI/bge-m3` embeddings, 1024-dim,
  L2-normalized (cosine similarity = dot product).
- **Metadata available**: FIPS code, county name, lat/lon centroid only — no
  population/economic variables are in-repo yet.
- **Pipelines**: `ingest_source_a.py` (ingestion), `visualize_source_a.py`
  (PCA), `analyze_source_a_similarity.py` (pairwise similarity/distance),
  `analyze_source_a_clusters.py` (K-means + Mantel test),
  `analyze_source_a_cluster_stability.py` (cross-seed stability +
  cluster-coherence permutation test, new).
- **Important artifact-staleness note**: only the clustering/Mantel
  pipeline (`analyze_source_a_clusters.py`) and the new stability script
  were re-run against the backfilled 3,125-row parquet. `stats.json`, the
  three saved figures, `source_a_map.html`, and
  `source_a_similarity.html`/`source_a_similarity_pairs.csv` still reflect
  the **pre-backfill** snapshot (n=3,088 / n=2,793) and were not
  regenerated — regenerating them was out of scope for this pass. Do not
  read PCA or similarity-pairs numbers as including the 37 backfilled
  Virginia cities.
- **Units of analysis**: individual county. PCA ran on n=3,088 (pre-backfill
  snapshot). Clustering/Mantel now run on n=2,830 (post-backfill: 2,793 +
  37 Virginia cities, after dropping 294 "stub" counties with <100 characters
  of real content and 1 non-50-state entry). Pairwise similarity/distance is
  treated as non-independent — inference is via Mantel permutation, not a
  naive pairwise correlation p-value.

## 3. Main Findings

1. **Geography ↔ similarity (Mantel test)**: weak negative association,
   robust to the Virginia backfill.

   | | Pre-backfill (n=2,793) | Post-backfill (n=2,830) |
   |---|---|---|
   | Mantel r | -0.0937 | -0.0898 |
   | p-value | 0.0020 | 0.0020 |

   Nearby counties are very slightly more textually similar than far-apart
   ones; geography is not a dominant driver, and this conclusion does not
   change with fuller Virginia coverage.

2. **PCA**: PC1 explains only 4.9% of total variance (n=3,088, pre-backfill
   snapshot). A manual read of the 3 highest- and 3 lowest-loading counties
   (Elliott County, KY / Wise County, VA / Kent County, TX high;
   Miami County, IN / Clay County, IN / Floyd County, GA low) found no
   shared theme — both tails read as generic formation/population
   boilerplate. **No thematic label is assigned to PC1.**

3. **Clustering (K-means, silhouette-selected k)**: k=2 selected across
   k=2..12, but every k's silhouette score is low (max ≈0.034-0.036 across
   both pre- and post-backfill runs, well under the ~0.25 threshold usually
   associated with real structure) — the "best" clustering is a weak fit,
   not a good one. Post-backfill: cluster sizes 1,295 / 1,535; cluster 0 is
   ~10% *looser* than the corpus-wide mean pairwise distance (1,583 km vs.
   1,437 km), cluster 1 is ~10% *tighter* (1,291 km) — opposite directions,
   not a coherent regional split.

4. **Notable outlier pairs**: a handful of far-apart county pairs are
   unusually similar in text, e.g. Oklahoma County, OK ↔ Spokane County, WA
   (similarity 0.776 at 2,126 km) — consistent with finding #1 (geography
   matters a little on average; individual pairs can still be highly
   similar regardless of distance).

5. **Cross-seed K-means stability** (new, closes a round-0 gap): re-running
   K-means at k=2 across 5 seeds (42, 7, 123, 2024, 99) gives silhouette
   mean=0.0339, std=0.0005, and pairwise Adjusted Rand Index across all 10
   seed pairs of mean=0.9946 (range 0.9901–0.9986). **The k=2 partition is
   highly stable and reproducible** — it is not an artifact of one random
   seed.

6. **Cluster-coherence permutation test** (new, closes a round-0 gap): for
   the k=2 clustering (999 permutations of county-to-cluster labels,
   preserving cluster sizes), both clusters' ~10% deviation from the corpus
   mean distance is more extreme than every one of the 999 random
   relabelings produced (p≤0.001, both directions). **The geographic-
   coherence split is statistically real, not chance** — but this does not
   upgrade its practical size: silhouette stays ≤0.035, so it is a real,
   reproducible, but small effect, detectable mainly because of the large
   sample size, not because the clusters are well-separated. **Do not
   describe the two clusters as "regions."**

## 4. Negative Result: Section-Expansion Prototype

**Question**: since intro-only embeddings carry only a weak geographic
signal and no clustering structure, would embedding more of each county's
Wikipedia article (additional body sections, or a specific content-rich
section) differentiate counties better?

**Method**: two variants prototyped on a fixed 40-county sample (seed=42),
re-fetched and re-embedded with the same `BAAI/bge-m3` model, compared
against the same sample's intro-only embeddings via Mantel test and
pairwise cosine-similarity mean/std.

| Metric | Intro-only | Lead + 3 body sections | Intro-only (same 10) | Economy-only |
|---|---|---|---|---|
| Mean text length | 694 chars | 4,532 chars (6.5×) | 949 chars | 697 chars |
| Pairwise similarity mean | 0.559 | 0.585 | 0.556 | 0.509 |
| Pairwise similarity std | 0.068 | 0.054 | 0.069 | 0.072 |
| Mantel r (geo↔similarity) | -0.272, p=0.004 | -0.372, p=0.002 | -0.287, p=0.016 | -0.093, p=0.494 (n.s.) |

- **Lead + 3 body sections** (n=40, all fetches succeeded): counties became
  *more* alike, not less, despite 6.5× more text — consistent with
  History/Geography/Demographics sections being even more templated across
  counties than the lead.
- **Economy section only** (n=10/40 — 30 of 40 sampled counties had no
  `<h2>` Economy section, mostly smaller/rural ones): directionally more
  differentiating on the counties that have it, but coverage collapses to
  25% of the sample and n=10 is too small to trust the direction alone.

**Conclusion**: neither variant improves on the intro-only baseline in a
usable way. **Closed without further section-expansion variants.** If more
differentiating signal is wanted later, the next lever is a different
approach entirely (targeted boilerplate-stripping within body sections, or
a non-Wikipedia source), not another section-selection variant. Prototype
script (`prototype_expanded_sections.py`) was deleted after this finding was
recorded — throwaway, fully described above.

## 5. Figure-by-Figure Interpretation

Figures live in `analysis-output/figures/` (generated pre-backfill, n=3,088
for PC1, n=2,793 for the other two — see the artifact-staleness note in
§2).

**figure-01-similarity-vs-distance.png** — Hexbin of similarity vs. distance
across all pairs, trend line, Mantel r/p annotated. The trend line's shallow
slope (≈0.56 → ≈0.51 similarity across the full distance range) visually
confirms "weak," not "strong." Rules out over-reading
`source_a_similarity_pairs.csv`'s outlier table (extreme cases by
construction) as evidence of a strong geography effect.

**figure-02-pc1-distribution.png** — Histogram of PC1 values with the 3
highest/lowest counties labeled. Unimodal, no bimodality suggesting a
natural two-group split. Pre-empts assigning a semantic label to PC1 — the
shape gives no hint of one, and a manual text check of the extremes found
none either.

**figure-03-cluster-coherence.png** — Bar chart of mean intra-cluster
distance per cluster vs. the corpus-wide baseline. The two clusters deviate
from baseline in opposite directions by similar magnitudes (~10%). Blocks
the claim that "embedding clusters correspond to US regions" — combined
with the uniformly low silhouette scores, K-means does not find strong
structure of any kind here (later confirmed statistically real but small by
the §3.6 permutation test).

## 6. Claim Candidates

- **Claim**: Geographic distance between US counties has a weak but
  statistically detectable negative association with the cosine similarity
  of their Wikipedia intro-text embeddings.
  - Evidence: Mantel test, r=-0.0898, p=0.0020, n=2,830, 499 permutations,
    seed=42 (post-backfill; original pre-backfill run: r=-0.0937, p=0.0020).
  - Allowed wording: "a weak, statistically significant negative
    association"; "geography explains a small amount of the variation in
    textual similarity."
  - Forbidden wording: "geographic distance strongly predicts / determines /
    drives embedding similarity"; any wording implying a large or dominant
    effect.
  - Status: **confirmed robust to the Virginia backfill** — the round-0
    "next check" on this claim is resolved.

- **Claim**: K-means clustering (k selected by silhouette score) finds a
  highly reproducible k=2 split whose geographic coherence is statistically
  real but small in magnitude.
  - Evidence: silhouette ≤0.0347 across 5 seeds; pairwise ARI mean=0.9946;
    permutation test p≤0.001 (both clusters, 999 permutations).
  - Allowed wording: "the two-cluster split is stable across random seeds
    and its geographic-coherence difference from chance is statistically
    significant, but the effect size (~10% deviation, silhouette ≤0.035)
    remains small — this is not evidence of clean regional structure."
  - Forbidden wording: "clusters correspond to distinct US regions";
    "embedding clusters map onto geography"; using "statistically
    significant" alone without the effect-size qualifier.
  - Status: round-0's two open gaps (seed stability, coherence
    significance) are now both resolved — see §3.5–3.6.

- **Claim**: PC1 of the Source A embeddings explains a small (4.9%) share of
  total embedding variance; no thematic label should be attached to it.
  - Evidence: `pca.explained_variance_ratio_[0]=0.0485`, n=3,088; manual
    6-county tail inspection found no shared theme.
  - Allowed wording: "PC1 explains a small share of the embedding's variance
    (4.9%)"; naming the specific extreme counties is fine.
  - Forbidden wording: any semantic label for what PC1 "represents."
  - Status: unresolved, unchanged — PC1 was not recomputed post-backfill.

- **Claim**: Expanding embedded text beyond the Wikipedia lead section does
  not improve county differentiation.
  - Evidence: §4 prototype — both tested variants performed worse or were
    coverage-limited relative to intro-only.
  - Allowed wording: "tested section-expansion variants did not improve on
    the intro-only baseline; further section-selection tuning is not
    expected to help."
  - Forbidden wording: claiming this rules out *all* possible text-source
    changes — only the two tested variants (more sections; Economy-only)
    are covered.
  - Status: closed, no further section-expansion variants planned.

## 7. Limitations / Open Items

1. **19 remaining ingestion failures** (out of the original 56; the 37
   Virginia independent cities are now fixed): 8 Alaska boroughs/census
   areas, 5 New York City boroughs (Bronx/Kings/New York/Queens/Richmond
   County), Hawaii County, De Witt County IL, Larue County KY, De Soto
   Parish LA, Nantucket County MA, Le Flore County OK — all Census-
   Gazetteer-name-vs-Wikipedia-title mismatches, not yet mapped in
   `INDEPENDENT_CITY_ARTICLE_LOOKUP`. Out of scope for this pass.
2. **Artifact staleness**: `stats.json`, the three figures, and the
   similarity/PCA HTML artifacts still reflect the pre-backfill snapshot
   (see §2). Regenerate via `generate_source_a_insights.py` if a fully
   consistent post-backfill snapshot is needed.
3. **PC1 has no established meaning** — only 4.9% of variance, no thematic
   pattern found in a small manual check.
4. **Population mismatch between analyses**: PCA runs on the pre-backfill
   3,088-county snapshot; clustering/Mantel now run on 2,830 post-backfill
   counties — not directly comparable county-for-county.
5. **No alternative clustering method tried** (e.g. hierarchical, DBSCAN) as
   a robustness check on the K-means finding.

## 8. Next Actions

1. **Do not build downstream conclusions on PC1's semantic meaning or on the
   two K-means clusters representing real regions** — both remain
   weak/unsupported.
2. **If a fully consistent post-backfill snapshot matters**, re-run
   `generate_source_a_insights.py` (regenerates `stats.json` + all three
   figures) and `visualize_source_a.py` / `analyze_source_a_similarity.py`
   against the 3,125-row parquet.
3. **The 19 remaining non-Virginia ingestion failures** are a known,
   scoped-out gap — backfill only if a specific downstream question needs
   Alaska boroughs, NYC boroughs, or the other 9 counties.
4. **Do not pursue further Wikipedia section-selection variants** (§4,
   closed) — if stronger differentiation is needed, the next lever is a
   non-Wikipedia source (Sources B–F) or targeted boilerplate-stripping.
5. **This finding set is a reasonable checkpoint** for Source B–F planning:
   the weak-but-real geography signal, the now-resolved clustering-stability
   question, and the remaining coverage gaps are the facts most likely to
   matter for that planning.

## 9. Artifact and Reproducibility Index

- Ingestion: `ingest_source_a.py` (includes `INDEPENDENT_CITY_ARTICLE_LOOKUP`
  with all 37 Virginia independent cities), `backfill_virginia_cities.py`
  (Virginia-only re-ingestion + parquet merge).
- EDA / analysis: `visualize_source_a.py` (PCA), `analyze_source_a_similarity.py`
  (pairwise similarity/distance), `analyze_source_a_clusters.py` (K-means +
  Mantel test), `analyze_source_a_cluster_stability.py` (cross-seed
  stability + cluster-coherence permutation test).
- Insights synthesis: `generate_source_a_insights.py` (produced the
  pre-backfill `stats.json` and the three saved figures; not re-run
  post-backfill — see §7.2).
- Persisted statistics: `analysis-output/stats.json` (pre-backfill
  snapshot).
- Figures: `analysis-output/figures/figure-01-similarity-vs-distance.png`,
  `figure-02-pc1-distribution.png`, `figure-03-cluster-coherence.png`,
  `figures/source-a-numeric-summary.md` (all pre-backfill).
- Companion artifacts: `source_a_map.html` (pre-backfill), `source_a_similarity.html`
  / `source_a_similarity_pairs.csv` (pre-backfill), `source_a_clusters.html`
  / `source_a_cluster_summary.csv` (**post-backfill**, refreshed by this
  round's `analyze_source_a_clusters.py` run).
- Ingestion log: `ingest_run.log` (untracked).
- Reproduction: `uv run --env-file .env backfill_virginia_cities.py`, then
  `uv run analyze_source_a_clusters.py` and
  `uv run analyze_source_a_cluster_stability.py`. All seeded
  (`RANDOM_SEED=42` throughout; stability seeds 42/7/123/2024/99).
