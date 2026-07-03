# Source A Embeddings — Statistics Appendix

## Data and unit of analysis

- Source: `source_a_embeddings.parquet` — 3,088 counties, `BAAI/bge-m3` embeddings, 1024-dim, L2-normalized (cosine similarity = dot product).
- Unit of analysis for PCA: individual county (n=3,088; only rows with a matched centroid are kept, `visualize_source_a.main()`'s `merge(..., how="left").dropna(subset=["lat","lon"])` — no rows were actually dropped at this step for the current data).
- Unit of analysis for clustering/Mantel test: individual county, but restricted to n=2,793 after two additional filters applied by `analyze_source_a_clusters.main()`: (1) `drop_stub_counties` removes 294 counties whose de-boilerplated intro text is <100 characters (mostly small-population counties with almost no narrative content beyond the templated formation/population sentence), and (2) `filter_to_fifty_states` removes 1 remaining non-50-state entry (Puerto Rico/DC — the parquet already excludes PR municipios, so only 1 row is affected here, likely DC).
- **PCA's n (3,088) and clustering/Mantel's n (2,793) are different populations** — 295 counties present in the PCA figure are absent from the clustering/Mantel figures. This is not reconciled; it mirrors exactly what the two original scripts did independently.
- Pairwise similarity/distance for the Mantel test and the similarity-vs-distance figure are computed over all C(2793,2) ≈ 3.9M unique county pairs from `build_pairwise_table`. **These pairs are not independent samples** — each county appears in 2,792 pairs — so a naive Pearson correlation p-value on the raw pairwise table would be invalid (falsely inflated significance). This is exactly why the Mantel permutation test (which permutes county labels as a block, preserving the pair structure, rather than treating pairs as i.i.d.) is the valid inferential method used here; no naive pairwise significance claim is made anywhere in this analysis.

## Test 1: PCA (descriptive only, no inferential claim)

- Method: `sklearn.decomposition.PCA(n_components=1)` on the (3088, 1024) L2-normalized embedding matrix (`visualize_source_a.compute_pc1`).
- Result: PC1 explained variance ratio = **0.04854536** (4.9%).
- No significance test applies to a single PCA component's variance ratio in this context; this is reported purely descriptively.
- Extreme counties (for reference, not thematic interpretation — see analysis-report.md):
  - Highest PC1: Elliott County, Kentucky; Wise County, Virginia; Kent County, Texas.
  - Lowest PC1: Miami County, Indiana; Clay County, Indiana; Floyd County, Georgia.
  - Manual read of all six counties' `raw_intro_text` found no shared theme distinguishing the two tails; both tails consist of generic county-formation/population boilerplate sentences.

## Test 2: K-means clustering + silhouette model selection

- Method: `sklearn.cluster.KMeans(n_clusters=k, random_state=42, n_init=10)` for k=2..12, scored by `sklearn.metrics.silhouette_score` on a 1,000-county random subsample (`random_state=42`) for tractability (`analyze_source_a_clusters.choose_k` / this script's `choose_k_with_scores`, which additionally persists every k's score).
- Full k-sweep (n=2,793 counties, embeddings as-is, no additional dimensionality reduction before clustering):

  | k | Silhouette score |
  |---|---|
  | 2 | 0.0342 (selected) |
  | 3 | 0.0261 |
  | 4 | 0.0197 |
  | 5 | 0.0149 |
  | 6 | 0.0130 |
  | 7 | 0.0098 |
  | 8 | 0.0098 |
  | 9 | 0.0092 |
  | 10 | 0.0033 |
  | 11 | 0.0041 |
  | 12 | 0.0055 |

- **All silhouette scores in this sweep are low in absolute terms** (typical thresholds: >0.5 strong, 0.25–0.5 weak-to-reasonable, <0.25 essentially no substantial structure). Every k tested here is well under 0.25, meaning the embedding space does not have well-separated cluster structure at any k in this range — k=2 is simply the *least bad* fit, not a *good* fit.
- Assumption check: K-means assumes roughly spherical, similarly-sized clusters in Euclidean space; L2-normalized embeddings make Euclidean and cosine geometry closely related, so this is a reasonable default, but no alternative clustering method (e.g. hierarchical, DBSCAN) was tried as a robustness check.
- Seed/stability: a single `random_state=42` with `n_init=10` was used (mitigates within-seed K-means initialization instability, since the best of 10 initializations is kept), but **cross-seed stability of the k=2 solution itself was not checked** — this is a stated gap, not a fabricated one.
- Cluster geographic coherence (descriptive, not an inferential test — no permutation test was run on cluster coherence specifically):
  - Corpus-wide mean pairwise distance: 1,438.1 km.
  - Cluster 0: n=1,536, mean intra-cluster distance 1,293.5 km (≈10.1% below corpus mean).
  - Cluster 1: n=1,257, mean intra-cluster distance 1,589.2 km (≈10.5% above corpus mean).
  - No confidence interval or significance test is reported for these two coherence numbers; treat them as descriptive effect sizes only. A permutation test analogous to the Mantel test (e.g., comparing observed intra-cluster distances to a null from randomly-reassigned cluster labels of the same sizes) would be needed to claim statistical significance for "cluster 0 is more compact than random" — this was not run and is a blocker for any such claim.

## Test 3: Mantel permutation test (geography vs. embedding similarity)

- Method: `analyze_source_a_clusters.mantel_test` — Pearson correlation between the upper-triangular geographic distance matrix (haversine, km) and the embedding cosine-similarity matrix, both over the same 2,793 counties; null distribution built from 499 row/column permutations of the distance matrix (seed=42), two-sided p-value.
- Result: **r = -0.0937, p = 0.0020** (499 permutations, seed=42).
- Interpretation of effect size: |r| ≈ 0.09 is a small effect by conventional (Cohen-style) thresholds for correlation coefficients (small ≈0.1, medium ≈0.3, large ≈0.5). The negative sign means greater geographic distance is associated with slightly *lower* embedding similarity.
- This is the statistically valid test for this question (see "unit of analysis" note above on why a naive pairwise correlation p-value would be invalid); it was reused unmodified from `analyze_source_a_clusters.py`, not reimplemented.
- No multiple-comparison correction was applied because only one Mantel test is reported here (not several contrasts).

## Blockers / limitations (explicit)

1. **Coverage gap**: 56 of 3,222 attempted counties failed ingestion (`ingest_run.log`), 38 of which are Virginia independent cities; the committed parquet also excludes ~78 Puerto Rico municipios by design. All findings above are conditional on this incomplete corpus. See analysis-report.md for the full breakdown.
2. **No repeated-measures / cross-seed replication** for the K-means clustering result — a single seed's k-sweep and a single seed's final fit are reported. Cluster assignment stability across seeds is unknown.
3. **No significance test for cluster geographic coherence** (only the Mantel test on the full pairwise matrix is inferentially valid here; the per-cluster coherence numbers are descriptive only).
4. **PC1 and clustering/Mantel operate on different county populations** (3,088 vs. 2,793) — not reconciled, inherited from the original scripts' independent filtering choices.
5. **No thematic interpretation of PC1** is offered — a 6-county manual read is too small a sample to support one, and none was found.

## QA gate

- [x] Primary comparison question is explicit (does geography associate with textual similarity/clustering structure).
- [x] Sample size stated for every test (n=3,088 for PCA; n=2,793 for clustering/Mantel; single seed=42 for both, n_permutations=499 for Mantel).
- [x] Inferential test (Mantel) is justified given the non-independence of pairwise data; no naive pairwise significance claim made.
- [x] Effect sizes reported (Mantel r; silhouette scores; % deviation from corpus-mean coherence).
- [x] Real figures generated from real data (figures/figure-01/02/03.png).
- [x] Each figure has an interpretation note (figure-catalog.md).
- [x] Limitations/blockers stated explicitly (this section).
- [x] Claim candidates carry evidence, uncertainty, and allowed/forbidden wording (analysis-report.md).
- [x] Over-strong wording explicitly blocked (analysis-report.md "Forbidden stronger wording" per claim).
- [x] No manuscript-style Results prose included.
