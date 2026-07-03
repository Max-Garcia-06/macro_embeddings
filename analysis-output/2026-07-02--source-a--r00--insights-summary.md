---
type: results-report
date: 2026-07-02
experiment_line: source-a
round: 0
purpose: insights-summary
status: active
source_artifacts:
  - analysis-output/analysis-report.md
  - analysis-output/stats-appendix.md
  - analysis-output/figure-catalog.md
  - analysis-output/stats.json
---

# Source A / Round 0 / Insights Summary / 2026-07-02

> Round is unknown/not yet tracked for this experiment line — `r00` is a placeholder and should be normalized once a real round-numbering convention exists for Source A.
>
> This repo is not bound to an Obsidian project knowledge base (no `.claude/project-memory/registry.yaml`), so this report is a local markdown artifact under `analysis-output/`, not written back to an Obsidian vault.

## 1. Executive Summary

Source A's Wikipedia-intro-text county embeddings (`BAAI/bge-m3`, 1024-dim) carry a real but small geographic signal. A Mantel permutation test finds a weak, statistically significant negative correlation between geographic distance and embedding similarity (r = -0.094, p = 0.002). Neither PCA's first component (4.9% of variance) nor silhouette-selected K-means clustering (k=2, all silhouette scores ≤0.034) finds strong or cleanly interpretable structure beyond that weak geographic effect. The dataset is missing nearly all Virginia independent cities (38 of 56 ingestion failures), which caveats every geography-related conclusion here.

## 2. Experiment Identity and Decision Context

- **Experiment line**: Source A — Wikipedia intro-text embeddings for US counties, one of a planned multi-source (A–F) macro/geo embedding dataset (see `README.md`).
- **Decision this report supports**: whether Source A's embeddings, as currently ingested, are worth building on (e.g., as a feature source for downstream macro modeling) before investing in Sources B–F or fixing ingestion gaps.
- **Trigger**: three EDA scripts (PCA map, similarity-vs-distance, clustering + Mantel test) had already been run once each, but their numeric results were never persisted or synthesized into a single evidence-based narrative.

## 3. Setup and Evaluation Protocol

- **Data**: `source_a_embeddings.parquet`, 3,088 counties, embeddings L2-normalized (cosine similarity = dot product).
- **Metadata available**: FIPS code, county name, lat/lon centroid only — no population/economic variables are in-repo yet.
- **Pipelines reused, not modified**: `visualize_source_a.py` (PCA), `analyze_source_a_similarity.py` (pairwise similarity/distance), `analyze_source_a_clusters.py` (K-means + Mantel test). A new script, `generate_source_a_insights.py`, imports these scripts' existing pure functions with their original seeds/constants to persist numbers that were previously only logged, and renders three static figures.
- **Units of analysis**: individual county for PCA (n=3,088) and for clustering/Mantel (n=2,793, after dropping 294 "stub" counties with <100 characters of real content and 1 non-50-state entry). Pairwise similarity/distance is computed over all unique county pairs (~3.9M pairs from n=2,793), and treated as non-independent — inference is via Mantel permutation, not a naive pairwise correlation p-value.

## 4. Main Findings

1. **Geography ↔ similarity**: weak negative association (Mantel r = -0.094, p = 0.002, 499 permutations, n=2,793). Nearby counties are very slightly more textually similar than far-apart ones; geography is not a dominant driver.
2. **PCA**: PC1 explains only 4.9% of total variance (n=3,088). A manual read of the 3 highest- and 3 lowest-loading counties found no shared theme — both tails read as generic formation/population boilerplate. No thematic label is assigned to PC1.
3. **Clustering**: silhouette-based model selection (k=2..12) picked k=2, but every k's silhouette score was low (max 0.034, well under the ~0.25 threshold usually associated with any real structure) — the "best" clustering is a weak fit, not a good one. The two resulting clusters move in *opposite* directions relative to the corpus-wide mean pairwise distance (1,438 km): cluster 0 (n=1,536) is ~10% tighter (1,293 km); cluster 1 (n=1,257) is ~10% looser (1,589 km) — not a coherent regional split.
4. **Notable outlier pairs**: a handful of far-apart county pairs are unusually similar in text, e.g. Oklahoma County, OK ↔ Spokane County, WA (similarity 0.776 at 2,126 km) — consistent with finding #1 (geography matters a little on average, individual pairs can still be highly similar regardless of distance).

## 5. Statistical Validation

- Mantel test is the valid inferential method for the geography/similarity question because pairwise similarity/distance values are not independent (each county appears in 2,792 pairs); a naive Pearson p-value on the raw pairwise table would be invalid. No such naive claim is made.
- K-means used a fixed seed (`random_state=42`, `n_init=10`), mitigating within-seed initialization instability, but cross-seed/cross-k stability of the k=2 solution was not checked — stated as an open gap, not resolved here.
- No significance test was run on the per-cluster geographic-coherence numbers (1,293 km vs. 1,589 km vs. 1,438 km baseline) — these are descriptive effect sizes only, not tested against a random-labeling null.
- Full statistical detail, assumptions, and the complete k-sweep table: `analysis-output/stats-appendix.md`.

## 6. Figure-by-Figure Interpretation

**figure-01-similarity-vs-distance.png** — Hexbin of similarity vs. distance across all pairs, trend line, Mantel r/p annotated directly on the plot. Key observation: the trend line's shallow slope (≈0.56 → ≈0.51 similarity across the full distance range) visually confirms "weak," not "strong," matching the Mantel r=-0.094. Decision implication: rules out over-reading the `source_a_similarity_pairs.csv` outlier table (which by construction only shows extreme cases) as evidence of a strong geography effect.

**figure-02-pc1-distribution.png** — Histogram of PC1 values (n=3,088) with the 3 highest/lowest counties labeled, 4.9% variance explained stated in the title. Key observation: unimodal, roughly bell-shaped, no bimodality suggesting a natural two-group split along this axis. Decision implication: pre-empts assigning a semantic label to PC1 — the shape gives no hint of one, and a manual text check of the extremes found none either.

**figure-03-cluster-coherence.png** — Bar chart of mean intra-cluster distance per cluster vs. the corpus-wide baseline. Key observation: the two clusters deviate from baseline in opposite directions by similar magnitudes (~10%). Decision implication: blocks the claim that "embedding clusters correspond to US regions" — combined with the uniformly low silhouette scores, the honest read is that K-means does not find strong structure of any kind here.

Full per-figure purpose/caption/caveat detail: `analysis-output/figure-catalog.md`.

## 7. Failure Cases / Negative Results / Limitations

- **Coverage gap (primary limitation)**: ingestion succeeded for 3,166 of 3,222 attempted counties; 56 failed, 38 of which are Virginia independent cities (Wikipedia article-title mismatches in `INDEPENDENT_CITY_ARTICLE_LOOKUP`, `ingest_source_a.py:147-161`, plus 2 rate-limit errors on Richmond City / Roanoke City). The committed parquet (3,088 rows) also excludes ~78 Puerto Rico municipios by design. **All of Virginia's independent cities are absent from every finding above** — its ~95 regular counties are present, but this urban-city geography is not. Fixing this is a separate ingestion task, not attempted here.
- **No strong clustering structure found**: this is itself a negative result worth stating plainly — K-means at any tested k (2–12) does not carve the embedding space into well-separated groups (silhouette ≤0.034 throughout).
- **PC1 has no established meaning**: only 4.9% of variance, no thematic pattern found in a small manual check. Do not treat PC1 as an interpretable macro/geo axis without further qualitative work.
- **Population mismatch between analyses**: PCA runs on 3,088 counties, clustering/Mantel on 2,793 — the two are not directly comparable county-for-county; inherited from the original scripts' independent filtering choices, not reconciled here.

## 8. What Changed Our Belief

Before this report, there was no single place stating whether Source A's embeddings actually encode anything geographically meaningful — only three disconnected scripts with results scattered across log lines, two small CSVs, and three interactive HTML files. This report establishes, with actual statistics: **the geographic signal is real but small**, and **neither PCA nor clustering surfaces additional strong structure on top of it**. That's a meaningfully weaker picture than "the embeddings capture geography," and it should reset expectations for what Source A alone can support before Sources B–F or richer metadata (population, economic variables) are added.

## 9. Next Actions

1. **Do not build downstream conclusions on PC1's semantic meaning or on the two K-means clusters representing real regions** — both are weak/unsupported per this report.
2. **Backfill the 38 missing Virginia independent cities** (extend `INDEPENDENT_CITY_ARTICLE_LOOKUP` in `ingest_source_a.py:147-161`, retry the 2 rate-limited cities) before treating any Virginia-specific or urban/independent-city geographic claim as reliable; re-run the Mantel test afterward as a sensitivity check.
3. **If a "does clustering find real structure" question matters going forward**, run a cross-seed stability check on K-means (multiple `random_state` values) and/or a permutation-based significance test on per-cluster geographic coherence before that claim is used elsewhere — currently unrun (stats-appendix.md).
4. **Hold off on assigning meaning to PC1** until a larger, systematic qualitative read (e.g., 20–30 extreme counties per tail) is done, if that axis is still of interest.
5. **This finding set is a reasonable checkpoint to promote into any Source A summary or subsequent Source B–F planning** — the weak-but-real geography signal and the coverage gap are the two facts most likely to matter for that planning.

## 10. Artifact and Reproducibility Index

- Computation script: `generate_source_a_insights.py` (new; imports existing EDA scripts' functions, no modification to them).
- Persisted statistics: `analysis-output/stats.json`.
- Full analysis bundle: `analysis-output/analysis-report.md`, `analysis-output/stats-appendix.md`, `analysis-output/figure-catalog.md`.
- Figures: `analysis-output/figures/figure-01-similarity-vs-distance.png`, `figure-02-pc1-distribution.png`, `figure-03-cluster-coherence.png`, `figures/source-a-numeric-summary.md`.
- Underlying EDA scripts (unmodified): `visualize_source_a.py`, `analyze_source_a_similarity.py`, `analyze_source_a_clusters.py`.
- Existing companion artifacts: `source_a_map.html`, `source_a_similarity.html`, `source_a_similarity_pairs.csv`, `source_a_clusters.html`, `source_a_cluster_summary.csv`.
- Ingestion log referenced for the coverage-gap limitation: `ingest_run.log` (untracked).
- Reproduction: `uv run generate_source_a_insights.py` — deterministic given fixed seeds (`RANDOM_SEED=42` throughout); verified to reproduce identical Mantel r/p and silhouette-selected k against a fresh run of the original `analyze_source_a_clusters.py`.
