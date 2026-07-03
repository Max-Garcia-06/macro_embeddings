# Source A Embeddings — Analysis Report

**Analysis question:** Does anything systematic show up in the Source A county embeddings (Wikipedia intro-text, `BAAI/bge-m3`, 1024-dim) — specifically, do they carry a geographic signal (via PCA, pairwise similarity, and clustering), and how strong is it?

All numbers below are pulled from `stats.json` (produced by `generate_source_a_insights.py`, which re-invokes the existing, already-validated `visualize_source_a.py` / `analyze_source_a_similarity.py` / `analyze_source_a_clusters.py` functions with their original seeds) or from `source_a_cluster_summary.csv` / `source_a_similarity_pairs.csv`. No numbers here are invented; see `stats-appendix.md` for full provenance and test details.

## Key findings

1. **Geography has a real but weak association with textual similarity.** A Mantel permutation test finds a small negative correlation between geographic distance and embedding similarity (r = -0.094, p = 0.002, 499 permutations, n = 2,793 counties) — nearby counties are very slightly more textually similar than far-apart ones, and this is unlikely to be noise, but the effect size is small. Geography is far from the dominant driver of these embeddings. See figure-01.
2. **PC1 captures only a small share of the embedding's total variance (4.9%)**, computed on all 3,088 counties. A manual read of the six most extreme counties on either tail (Elliott County, KY / Wise County, VA / Kent County, TX at the high end; Miami County, IN / Clay County, IN / Floyd County, GA at the low end) shows no obvious shared theme — both tails are dominated by generic, boilerplate-style formation/population sentences. **No thematic label should be assigned to PC1** without a much larger, systematic qualitative read of extreme-loading counties; the single axis a general-purpose PCA finds first in a 1024-dim semantic space need not correspond to a human-interpretable concept at all. See figure-02.
3. **K-means clustering finds only mild, inconsistent geographic coherence — not a clean regional split.** Silhouette-based model selection over k=2..12 picked k=2, but even the best silhouette score (0.034) is very low in absolute terms (far below the ~0.5+ that usually indicates well-separated clusters), meaning the "best" clustering here is still a weak fit to the data's actual structure. Of the two clusters: cluster 0 (n=1,536) is about 10% *more* geographically compact than the corpus-wide average pairwise distance (1,293 km vs. 1,438 km), while cluster 1 (n=1,257) is about 10% *less* compact (1,589 km) — i.e. one cluster leans mildly regional, the other leans mildly anti-regional, which is not a coherent "clustering finds regions" story. See figure-03.
4. **A handful of far-apart county pairs are unusually similar in text**, e.g. Oklahoma County, OK ↔ Spokane County, WA (similarity 0.776 at 2,126 km) — consistent with finding #1: geography matters a little on average, but individual outlier pairs can be highly similar despite being on opposite sides of the country. (`source_a_similarity_pairs.csv`, not re-derived here.)

## Coverage limitation (applies to every finding above)

Per `ingest_run.log`, the ingestion run that produced this dataset succeeded for 3,166 of 3,222 attempted counties (56 failures), and the committed `source_a_embeddings.parquet` (3,088 rows) additionally excludes ~78 Puerto Rico municipios by design. Of the 56 ingestion failures, **38 are Virginia independent cities** (e.g. Alexandria, Norfolk, Richmond, Virginia Beach — Wikipedia article-title mismatches in `INDEPENDENT_CITY_ARTICLE_LOOKUP`, plus 2 rate-limit failures), with the rest scattered across Alaska boroughs, NYC's five boroughs, a few name-prefix mismatches, Hawaii County, and Nantucket County.

This means **all of Virginia's urban independent cities are missing** from every finding above — Virginia's ~95 regular counties are present, but its distinct, more urban independent-city geography is not. Any conclusion involving Virginia specifically, or any urban-vs-rural signal that independent cities nationally would partly carry, should be read as incomplete. Fixing this is a separate ingestion task, out of scope here.

## Claim Candidates

- Claim: Geographic distance between US counties has a weak but statistically detectable negative association with the cosine similarity of their Wikipedia intro-text embeddings.
  - Source evidence: Mantel permutation test, `analyze_source_a_clusters.mantel_test`, n=2,793 counties, r=-0.0937, p=0.0020, 499 permutations, seed=42 (`stats.json`).
  - Allowed wording: "a weak, statistically significant negative association" / "geography explains a small amount of the variation in textual similarity."
  - Forbidden stronger wording: "geographic distance strongly predicts / determines / drives embedding similarity"; any wording implying a large or dominant effect.
  - Uncertainty: single Mantel run (no resampled-subset replication); Virginia's ~38 missing independent cities mean the test never observed that geography, so the true r/p over a complete corpus could differ.
  - Next check: re-run Mantel test after a future ingestion pass backfills the missing Virginia independent cities, as a sensitivity check.
  - Decision: keep, with hedged wording.

- Claim: K-means clustering (k selected by silhouette score) finds clusters that are, at best, weakly and inconsistently more geographically coherent than random grouping.
  - Source evidence: `source_a_cluster_summary.csv` / `stats.json` — selected k=2 (silhouette=0.0342, the highest across k=2..12, but low in absolute terms); cluster 0 (n=1,536) mean intra-cluster distance 1,293 km vs. corpus mean 1,438 km (~10% tighter); cluster 1 (n=1,257) mean intra-cluster distance 1,589 km (~10% looser than corpus mean).
  - Allowed wording: "one of the two clusters is modestly more geographically compact than average; the other is slightly more dispersed than average — clustering does not cleanly separate counties by region."
  - Forbidden stronger wording: "clusters correspond to distinct US regions"; "embedding clusters map onto geography"; any claim of clean or strong regional structure.
  - Uncertainty: k was chosen at a single seed (random_state=42, n_init=10) over a coarse k=2..12 sweep; silhouette was computed on a 1,000-point subsample, not the full set; all silhouette scores in the sweep are low (max 0.034), meaning even the winning k is a weak fit — cluster stability across seeds/k was never checked.
  - Next check: report the full k→silhouette curve (already in `stats.json`) so a reader can judge how close k=2 was to its runner-up (k=3, silhouette=0.0261); consider re-running KMeans with additional random seeds to check cluster-assignment stability.
  - Decision: keep, heavily hedged.

- Claim: PC1 of the Source A embeddings explains a small (4.9%) share of total embedding variance; no thematic label should be attached to it based on current evidence.
  - Source evidence: `visualize_source_a.compute_pc1`, explained_variance_ratio=0.0485 over 3,088 counties (`stats.json`); manual inspection of the 3 highest- and 3 lowest-PC1 counties' `raw_intro_text` found no shared theme distinguishing the two tails.
  - Allowed wording: "PC1 explains a small share of the embedding's variance (4.9%)"; naming the specific extreme counties is fine.
  - Forbidden stronger wording: any semantic label for what PC1 "represents" (e.g. "an urban-rural gradient," "an economic-development axis") — this has not been established and a 6-county manual read found no obvious pattern.
  - Uncertainty: PC1 was computed on the full 3,088-county set (pre-stub-filtering, pre-50-states-filtering), a different population than the 2,793 counties used for clustering/Mantel — the two analyses are not directly comparable county-for-county.
  - Next check: a larger, systematic qualitative read (e.g. top/bottom 20-30 counties) would be needed before proposing any thematic interpretation.
  - Decision: keep the quantitative statement; discard any thematic-labeling claim.

- Claim: The Virginia-independent-city coverage gap could materially bias any of the geography-related findings above.
  - Source evidence: `ingest_run.log` final summary ("Succeeded: 3166, Failed: 56"); 38 of 56 failures are Virginia independent cities.
  - Allowed wording: "this analysis excludes essentially all Virginia independent cities; geography-related findings, especially any involving Virginia or an urban/independent-city signal, should be read as incomplete."
  - Forbidden stronger wording: "Virginia is excluded" (imprecise — its regular counties are present, only independent cities are missing).
  - Uncertainty: unknown whether backfilling these 38 cities would change the Mantel r/p or cluster coherence materially — no sensitivity analysis has been run.
  - Next check: out of scope here; belongs to a future ingestion task (extend `INDEPENDENT_CITY_ARTICLE_LOOKUP`, `ingest_source_a.py:147-161`, retry the 2 rate-limited cities).
  - Decision: keep as a standing limitation on every geographic claim in this report.

## What changed in understanding

Before this analysis, the three EDA scripts' outputs existed only as disconnected artifacts (two small CSVs, three interactive HTML maps, and log lines that were never saved). This report is the first place the Mantel test result, PCA variance, and cluster coherence numbers are stated together and cross-referenced: the combined picture is that **Source A's embeddings carry a real but small geographic signal**, and that neither PCA's first axis nor K-means clustering finds a strong or cleanly interpretable geographic structure on top of it. The Virginia coverage gap is now stated explicitly as a caveat on all of this, which it previously was not.
