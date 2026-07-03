# Source A Embeddings — Figure Catalog

## figure-01-similarity-vs-distance.png (main figure)

- **Purpose**: Connect the descriptive similarity-vs-distance trend (previously only visible as an interactive scatter in `source_a_similarity.html`) to the one valid inferential statistic for this question, the Mantel test — these two facts previously lived in disconnected artifacts with no cross-reference.
- **Plotted variables**: x = geographic (haversine) distance between county pairs in km; y = embedding cosine similarity; color = pair density (hexbin); dashed line = linear trend fit (`np.polyfit`, degree 1) across all pairs.
- **Data source**: `analyze_source_a_similarity.build_pairwise_table` output (n=2,793 counties, ≈3.9M pairs), Mantel r/p from `stats.json`.
- **Caption requirements**: must state n counties, n pairs implied, and the annotated Mantel r/p/permutation count verbatim from `stats.json` — do not round differently than shown on the figure.
- **Key observation**: The dense core of pairs sits at short distances with similarity ~0.55; the fitted trend line slopes gently downward from ~0.56 at distance 0 to ~0.51 at ~8,500 km — a small but visible decline, consistent with Mantel r=-0.094.
- **Interpretation checklist**:
  1. Why does this figure exist? To show, visually, the same weak negative geography-similarity relationship the Mantel test quantifies, so a reader isn't left trusting a bare r/p pair without seeing the underlying data.
  2. What should the reader notice? The trend line's shallow slope relative to the y-axis range (similarity mostly stays in 0.4-0.7 regardless of distance) — the visual confirms "weak," not "strong."
  3. What does this change? It rules out over-claiming a strong geography effect from the CSV of "surprising" outlier pairs alone (`source_a_similarity_pairs.csv`), which by construction only shows extreme cases and could otherwise mislead a reader into thinking geography barely matters at all or matters a lot — this figure shows the honest middle-ground trend across all pairs.
- **Known caveats**: Excludes Virginia's independent cities and 294 stub counties (see stats-appendix.md); the density coloring can visually over-emphasize the dense short-distance core relative to the sparser long-distance tail where the "surprising" pairs live.

## figure-02-pc1-distribution.png (supporting figure)

- **Purpose**: Show how much of the embedding's variance PC1 actually captures and who sits at its extremes, since the existing map (`source_a_map.html`) shows PC1 spatially but not as a distribution, and the 4.9% explained-variance number was previously only in a log line.
- **Plotted variables**: x = per-county PC1 value; y = county count (histogram, 60 bins); text boxes list the 3 highest- and 3 lowest-PC1 counties by name; thin dotted vertical lines mark their exact PC1 values.
- **Data source**: `visualize_source_a.compute_pc1` applied to all 3,088 counties (`stats.json`'s `pc1_explained_variance_ratio` and `pc1_n_counties`).
- **Caption requirements**: must state the explained variance ratio (4.9%) and n=3,088; must not name a theme for what PC1 represents (see analysis-report.md).
- **Key observation**: The distribution is unimodal and only mildly skewed, roughly centered near PC1≈0.05-0.1, with long thin tails; there is no visible bimodality that would suggest PC1 cleanly separates two distinct groups of counties.
- **Interpretation checklist**:
  1. Why does this figure exist? To make the "4.9% of variance" number concrete and show it doesn't correspond to an obviously bimodal or otherwise structured axis.
  2. What should the reader notice? The single-peaked, roughly bell-like shape — there's no obvious "two clusters along PC1" story here, and the extreme counties on both tails read as generic in their intro text (per stats-appendix.md's manual check).
  3. What does this change? It pre-empts a natural but unsupported next question ("what does PC1 mean?") by showing the shape doesn't hint at an answer, and stats-appendix.md documents that a manual text check found no theme either — so this axis is best treated as an artifact of high-dimensional PCA on short text embeddings, not a labeled semantic dimension, until further evidence exists.
- **Known caveats**: Computed on a different (larger, unfiltered) population (n=3,088) than figures 1 and 3 (n=2,793) — see stats-appendix.md's population-mismatch note.

## figure-03-cluster-coherence.png (supporting figure)

- **Purpose**: Directly visualize whether K-means clustering (chosen by silhouette score) finds geographically coherent groups, which previously required reading two numeric columns out of `source_a_cluster_summary.csv` by hand.
- **Plotted variables**: x = cluster id (with n per cluster); y = mean intra-cluster pairwise geographic distance (km); dashed reference line = corpus-wide mean pairwise distance (1,438 km).
- **Data source**: `analyze_source_a_clusters.summarize_clusters` output, `source_a_cluster_summary.csv` (cross-checked against `stats.json`'s `cluster_mean_intra_km`/`corpus_mean_pairwise_km`).
- **Caption requirements**: must state both cluster sizes and both percentage deviations from the corpus baseline; must not claim the clusters correspond to named US regions.
- **Key observation**: Cluster 0 (n=1,536) sits below the corpus-mean reference line (~10% tighter); cluster 1 (n=1,257) sits above it (~10% looser) — the two clusters move in *opposite* directions relative to the baseline, rather than both being tighter (which would support a "clustering finds regional structure" story).
- **Interpretation checklist**:
  1. Why does this figure exist? To answer, visually, whether the "best" K-means solution (per silhouette score) actually produces geographically meaningful groups.
  2. What should the reader notice? Neither bar deviates from the reference line by more than ~10%, and they deviate in opposite directions — this is a weak and inconsistent signal, not a clean split.
  3. What does this change? It blocks the tempting but unsupported claim that "embedding clusters = US regions"; combined with the uniformly low silhouette scores in stats-appendix.md's k-sweep table, the honest conclusion is that K-means on this embedding space does not find strong structure of any kind, geographic or otherwise.
- **Known caveats**: Only descriptive percentages are shown, no significance test for whether either cluster's coherence differs from a random-labeling null (stats-appendix.md flags this as an unrun test, not a fabricated one).

## Supplementary interactive artifacts (not primary evidence, referenced for exploration only)

- `../source_a_map.html` — interactive PC1 map; useful for seeing PC1's spatial pattern county-by-county, but not a substitute for figure-02's variance/extremes summary.
- `../source_a_similarity.html` — interactive similarity-vs-distance scatter; the live counterpart of figure-01, useful for hovering over individual pairs.
- `../source_a_clusters.html` — interactive cluster map; useful for seeing exactly which counties fall into cluster 0 vs. cluster 1 geographically, complementing figure-03's aggregate coherence numbers.
