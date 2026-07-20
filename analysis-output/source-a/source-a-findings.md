---
type: results-report
date: 2026-07-10
experiment_line: source-a
round: 3
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

Source A's Wikipedia-intro-text county embeddings (`BAAI/bge-m3`, 1024-dim,
**LLM-cleaned via gemma2:9b as of 2026-07-10, see §12**) carry a real but
small-to-moderate geographic signal. A Mantel permutation test finds a
weak, statistically significant negative correlation between geographic
distance and embedding similarity (r ≈ -0.24, p = 0.002 — roughly double
the superseded regex-cleaned baseline's r ≈ -0.11). Neither PCA's first
component (3.9% of variance) nor K-means clustering (k=2, silhouette
≤0.029) finds strong or cleanly interpretable structure beyond that weak
geographic effect — **but** the k=2 split is highly reproducible across
random seeds (ARI≈0.981) and its +12.4%/-14.6% geographic-coherence
deviation from the corpus mean is statistically real (permutation p≤0.001
both directions), not sampling noise. Unlike the superseded baseline, PC1
**does** carry a real (non-economic) theme — founding/namesake narrative
presence vs. demographic-stub content, heavily Texas-concentrated; see
§3.2. A separate prototype found that embedding more article text
(additional sections, an Economy-only section, or LLM-cleaning those
additional sections) does not improve differentiation between counties
(§4, §4.1). **Coverage is complete**: the dataset covers all 3,144 US
counties/county-equivalents (the Virginia independent-city gap of 37
cities and a second batch of 19 Census-Gazetteer-vs-Wikipedia-title
mismatches have both been backfilled; see §2). Checked against
`E_macro_extendedProposal.pdf`'s justification for Source A (§10): the
proposal expects intro text to carry distinctive economic-transition
narrative; unsupervised analysis instead finds a Wikipedia
editorial-convention artifact (Texas founding-history detail vs. bare
demographic facts elsewhere), not economic-transition narrative — that
specific claim is unsupported, though correlation against real economic
variables (Source E/B) is still untested.

## 2. Data & Setup

- **Source**: `source_a_embeddings.parquet` — **3,144 counties, full
  coverage of every US county/county-equivalent** (up from 3,088 originally;
  37 Virginia independent cities backfilled first via
  `backfill_virginia_cities.py`, then the remaining 19
  Census-Gazetteer-vs-Wikipedia-title mismatches via
  `backfill_remaining_19.py` — see §7.1), `BAAI/bge-m3` embeddings,
  1024-dim, L2-normalized (cosine similarity = dot product).
- **Metadata available**: FIPS code, county name, lat/lon centroid only — no
  population/economic variables are in-repo yet.
- **Pipelines**: `ingest_source_a.py` (ingestion), `visualize_source_a.py`
  (PCA), `analyze_source_a_similarity.py` (pairwise similarity/distance),
  `analyze_source_a_clusters.py` (K-means + Mantel test),
  `analyze_source_a_cluster_stability.py` (cross-seed stability +
  cluster-coherence permutation test).
- **Artifact consistency**: all analysis pipelines and artifacts
  (`stats.json`, all three saved figures, `source_a_map.html`,
  `source_a_similarity.html`/`source_a_similarity_pairs.csv`,
  `source_a_clusters.html`/`source_a_cluster_summary.csv`,
  `source_a_key_findings.ipynb`) were re-run on 2026-07-10 against the
  adopted LLM-cleaned `source_a_embeddings.parquet` (post §12 swap) and are
  mutually consistent — no artifact reflects a stale snapshot.
- **Units of analysis**: individual county. PCA explained-variance ratio
  computed on n=3,144 (all matched to centroids). Clustering/Mantel run on
  n=2,849 (after dropping 294 "stub" counties with <100 characters of real
  content and 1 non-50-state entry). Pairwise similarity/distance is
  treated as non-independent — inference is via Mantel permutation, not a
  naive pairwise correlation p-value.

## 3. Main Findings

1. **Geography ↔ similarity (Mantel test)**: weak negative association,
   robust across every coverage backfill so far. First three columns are
   the superseded regex-cleaned baseline; the last column is the embedding
   source adopted 2026-07-10 (§12) — same 2,849-county set, LLM-cleaned
   text roughly doubles the association's magnitude.

   | | n=2,793 (original) | n=2,830 (+Virginia) | n=2,849 (regex baseline) | n=2,849 (**adopted**, LLM-cleaned) |
   |---|---|---|---|---|
   | Mantel r | -0.0937 | -0.0898 | -0.1055 | **-0.2362** |
   | p-value | 0.0020 | 0.0020 | 0.0020 | 0.0020 |

   Nearby counties are very slightly more textually similar than far-apart
   ones; geography is not a dominant driver, and this conclusion is stable
   across all three coverage snapshots (r has moved by ~0.01-0.02 each time,
   never changing sign or significance).

2. **PCA**: on the regex-cleaned baseline (superseded 2026-07-10, see §12),
   PC1 explained **4.8% of total variance** and a manual read of the 3
   highest/lowest-loading counties (Elliott County, KY / Wise County, VA /
   Kent County, TX high; Clay County, IN / Miami County, IN / Floyd County,
   GA low) found no shared theme — both tails read as generic
   formation/population boilerplate, so no thematic label was assigned.
   **This does not hold on the now-adopted LLM-cleaned embeddings** (see
   §3.2) — re-verify before reusing the "no thematic label" claim.

### 3.2 PC1 Re-Read on the Adopted LLM-Cleaned Embeddings (2026-07-10)

PC1 explains **3.9% of total variance** (n=3,144). Unlike the baseline, a
manual read of the top/bottom 10 loading counties finds a real, consistent
pattern — **PC1 is NOT thematically blank on the adopted embeddings**:

- **High PC1** (e.g. Elliott County, KY; Swisher County, TX; Jackson
  County, TX; Cottle County, TX; Eastland County, TX): cleaned text is
  dominated by "founded in [year]... named for [historical figure]"
  narrative — often a Texas Revolution soldier or 19th-century political
  figure — with specific dates, names, and events that survived LLM
  cleaning as genuinely distinctive facts.
- **Low PC1** (e.g. Otoe County, NE; Beaver County, PA; Claiborne County,
  TN; Cassia County, ID; several Iowa/North Dakota counties): cleaned text
  reduces to bare population + county-seat facts — the LLM cleaning left
  almost nothing else, i.e. these are near-stub articles with little
  distinguishing content to begin with.

Quantitative confirmation, not just an anecdotal 3-example read: **21 of
the top 30 PC1 counties are Texas counties**, against Texas's 8.1% base
rate in the corpus (254/3,144) — a 8.6x overrepresentation. Corpus-wide,
counties whose cleaned text contains "named for"/"named after" phrasing
average PC1=0.110 (n=1,281) vs. -0.076 for those that don't (n=1,863).

**Interpretation**: this is a real signal, not noise or residual
boilerplate — but it most plausibly reflects a **Wikipedia
editorial-convention artifact** (Texas county articles disproportionately
document Texas Revolution-era namesake history in enough distinctive
detail that the boilerplate-stripping LLM keeps it, while many other
states' articles have nothing beyond population/seat once stripped), not
an economic or geographic pattern relevant to the E_macro proposal's
actual claim (§10) that intro text carries distinctive economic-transition
narrative. **Do not read PC1 as evidence for or against the proposal's
economic claim** — it separates "article has rich founding/namesake prose"
from "article is a demographic stub," which is orthogonal to economic
content.

3. **Clustering (K-means, silhouette-selected k)**: k=2 selected across
   k=2..12 on the adopted LLM-cleaned embeddings, but every k's silhouette
   score is low (max ≈0.0284, well under the ~0.25 threshold usually
   associated with real structure — was ≈0.0345 on the superseded regex
   baseline) — the "best" clustering is a weak fit, not a good one. Full
   coverage: cluster sizes 1,420 / 1,429 (was 1,312 / 1,537); cluster 0 is
   ~12.4% *looser* than the corpus-wide mean pairwise distance (1,635 km vs.
   1,455 km), cluster 1 is ~14.6% *tighter* (1,243 km) — opposite
   directions, not a coherent regional split.

4. **Notable outlier pairs**: the five most similar pairs among all
   top-quartile-distance pairs (≥75th percentile; full ranked list in
   `source_a_similarity_pairs.csv`), **recomputed 2026-07-10 on the
   adopted LLM-cleaned embeddings**:

   | Rank | Pair | Similarity | Distance (km) | Why similar despite distance |
   |---|---|---|---|---|
   | 1 | Madison County, MT ↔ Madison County, VA | 0.876 | 2,866 | Shared name; both articles reduce to the same "founded/located in... population... county seat is X" skeleton once cleaned, with no eponym sentence retained for either. |
   | 2 | Jefferson County, GA ↔ Jefferson County, ID | 0.865 | 2,845 | Shared name *and* shared namesake (Thomas Jefferson) — both articles state "named for/after Thomas Jefferson" near-verbatim. |
   | 3 | Washington County, ID ↔ Washington County, WI | 0.857 | 2,279 | Shared name *and* shared namesake (George Washington) — same mechanism as #2. |
   | 4 | Washington County, KS ↔ Washington County, NY | 0.848 | 2,000 | Shared name; same skeleton pattern as #1. |
   | 5 | Lincoln County, WA ↔ Lincoln County, WV | 0.842 | 3,107 | Shared name (Abraham Lincoln counties recur across the corpus; see also KS↔OR, historically the #1 pair pre-cleaning). |

   **Mechanism shift from the superseded regex baseline**: previously two
   mechanisms produced the top-5 — (a) shared eponym and (b) generic
   boilerplate convergence (short, minimally-elaborated articles like
   Stutsman County, ND sitting close to the corpus's templated centroid,
   3 of the old top 5). **LLM cleaning eliminated mechanism (b) from the
   top ranks** — all 5 current top pairs are shared-name/shared-eponym
   (mechanism a only), consistent with §12's finding that the cleaning
   specifically collapsed generic-boilerplate-convergence similarity
   (tracked pairs involving Stutsman County dropped 0.83→0.51-0.53) while
   leaving shared-eponym similarity comparatively intact. Neither
   mechanism reflects real economic or geographic kinship.

5. **Cross-seed K-means stability**: re-running K-means at k=2 across 5
   seeds (42, 7, 123, 2024, 99) on the adopted LLM-cleaned embeddings gives
   silhouette mean=0.0281, std=0.0006, range=[0.0269, 0.0286], and pairwise
   Adjusted Rand Index across all 10 seed pairs of mean=0.9813, std=0.0100,
   range=[0.9707, 0.9972] (was mean=0.9955, range 0.9916–1.0000 on the
   superseded baseline). **The k=2 partition is still highly stable and
   reproducible** — it is not an artifact of one random seed, though
   slightly less perfectly so than the superseded baseline.

6. **Cluster-coherence permutation test**: for the k=2 clustering (999
   permutations of county-to-cluster labels, preserving cluster sizes),
   both clusters' +12.4%/-14.6% deviation from the corpus mean distance is
   more extreme than every one of the 999 random relabelings produced
   (p≤0.001,
   both directions). **The geographic-coherence split is statistically
   real, not chance** — but this does not upgrade its practical size:
   silhouette stays ≤0.029, so it is a real, reproducible, but small
   effect, detectable mainly because of the large sample size, not because
   the clusters are well-separated. **Do not describe the two clusters as
   "regions."**

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

### 4.1 Follow-Up: Does LLM Cleaning Rescue the Signal? (2026-07-09)

**Question**: §4's raw lead+3-sections variant *reduced* differentiation
because body sections are more templated than the lead. The gemma2:9b
boilerplate-cleaning pass validated for intro-only text (§12) was never
tried on expanded text — would cleaning recover the signal that raw
section-selection lost?

**Method**: same lead+3-sections expansion as §4, cleaned with the §12
gemma2:9b pass (prompt adapted for multi-section input, 6,000-char cap to
stay inside the Ollama server's context window), compared against
intro-only and raw-expanded on the same 40-county sample (50-state,
non-stub, seed=42 — a freshly-drawn sample, not bit-identical to §4's since
that prototype script was deleted; all three variants here share the same
draw for a fair within-run comparison).

| Metric | Intro-only | Raw expanded (lead+3) | LLM-cleaned expanded (lead+3) |
|---|---|---|---|
| Mean text length | 619 chars | 3,999 chars | 2,179 chars |
| Pairwise similarity mean | 0.560 | 0.559 | 0.522 |
| Pairwise similarity std | 0.062 | 0.047 | 0.056 |
| Mantel r (geo↔similarity) | -0.196, p=0.044 | -0.199, p=0.004 | **-0.106, p=0.218 (n.s.)** |

LLM cleaning did not rescue the section-expansion signal — it weakened it.
The Mantel correlation dropped from -0.199 (raw expanded, still
significant) to -0.106 and **lost significance entirely**, despite the
cleaned text having the *lowest* pairwise similarity mean of the three
(0.522) — that extra apparent "differentiation" doesn't track geography,
it's noise relative to the permutation test.

Spot-checking the cleaned text shows why: what survives the "keep
distinguishing facts" filter is inconsistent across counties. Some keep
strong geographic anchoring (Woodbury County, Iowa → "on the western edge
of Iowa... bordering the Missouri River"; Benton County, Tennessee →
"located in northwest Tennessee, bordering the... Tennessee River"), while
others lose it entirely in favor of pure naming/history trivia (Iroquois
County, Illinois → cleaned text is entirely about a 19th-century
county-seat relocation dispute, dropping "northeast part of Illinois"
altogether). Intro-only text has a uniform structure (location + population
+ metro area) that reliably anchors *some* geographic signal in every
county; body sections vary in content and ordering article to article, so
the LLM filter ends up keeping geography for some counties and dropping it
for others — adding noise rather than signal relative to geographic
distance.

**Conclusion**: closes the loop on §4 — LLM cleaning was the one lever left
untried on section-expansion, and it made the geographic signal worse, not
better. **Not pursuing section-expansion further in any form** (raw or
LLM-cleaned). Prototype script (`prototype_section_expansion_llm.py`) and
its progress/log files were deleted after this finding was recorded —
throwaway, fully described above.

## 5. Figure-by-Figure Interpretation

Figures live in `analysis-output/figures/` (all regenerated 2026-07-10
against the adopted LLM-cleaned embeddings, full 3,144-county corpus /
2,849-county clustering set — fully consistent with the numbers in §3).

**figure-01-similarity-vs-distance.png** — Hexbin of similarity vs. distance
across all pairs, trend line, Mantel r/p annotated. The trend line's shallow
slope visually confirms "weak," not "strong." Rules out over-reading
`source_a_similarity_pairs.csv`'s outlier table (extreme cases by
construction) as evidence of a strong geography effect.

**figure-02-pc1-distribution.png** — Histogram of PC1 values with the 3
highest/lowest counties labeled. Unimodal, no bimodality suggesting a
natural two-group split — still true on the adopted LLM-cleaned
embeddings. **Unlike the superseded regex baseline, a manual text check of
the extremes now finds a real theme** (founding/namesake narrative vs.
demographic-stub content, Texas-concentrated — see §3.2); the unimodal
shape rules out a clean *two-group* split, but does not mean PC1 is
thematically blank.

**figure-03-cluster-coherence.png** — Bar chart of mean intra-cluster
distance per cluster vs. the corpus-wide baseline. The two clusters deviate
from baseline in opposite directions by similar magnitudes (+12.4%/-14.6%
on the adopted LLM-cleaned embeddings, was ~11% both directions on the
superseded baseline). Blocks
the claim that "embedding clusters correspond to US regions" — combined
with the uniformly low silhouette scores, K-means does not find strong
structure of any kind here (later confirmed statistically real but small by
the §3.6 permutation test).

## 6. Claim Candidates

- **Claim**: Geographic distance between US counties has a weak but
  statistically detectable negative association with the cosine similarity
  of their Wikipedia intro-text embeddings.
  - Evidence: Mantel test, r=-0.2362, p=0.0020, n=2,849, 499 permutations,
    seed=42 (adopted LLM-cleaned embeddings, 2026-07-10 —
    `compare_llm_cleaning_full_corpus.py` / reconfirmed by
    `analyze_source_a_clusters.py`). **Superseded regex-baseline value**:
    r=-0.1055 (2026-07-03 full coverage; prior snapshots r=-0.0937 at
    n=2,793, r=-0.0898 at n=2,830) — the association roughly doubled in
    magnitude after LLM cleaning was adopted (§12).
  - Allowed wording: "a weak, statistically significant negative
    association"; "geography explains a small amount of the variation in
    textual similarity" (still "weak" by conventional |r| buckets, but
    note the magnitude if precision matters, e.g. in a table).
  - Forbidden wording: "geographic distance strongly predicts / determines /
    drives embedding similarity"; any wording implying a large or dominant
    effect.
  - Status: **confirmed robust across every coverage backfill, and
    strengthened (not weakened) by the 2026-07-10 embedding-source
    switch** — resolved.

- **Claim**: K-means clustering (k selected by silhouette score) finds a
  highly reproducible k=2 split whose geographic coherence is statistically
  real but small in magnitude.
  - Evidence (adopted LLM-cleaned embeddings, 2026-07-10): silhouette
    mean=0.0281, std=0.0006, range=[0.0269, 0.0286] across 5 seeds;
    pairwise ARI mean=0.9813, std=0.0100, range=[0.9707, 0.9972]; cluster
    0 observed=1635km vs. corpus mean 1455km (+12.4%), cluster 1
    observed=1243km (-14.6%), permutation test p=0.0010 both directions
    (999 permutations); all on the full 2,849-county set.
    **Superseded regex-baseline value**: silhouette ≤0.0353, ARI
    mean=0.9955, ~11% deviation both directions.
  - Allowed wording: "the two-cluster split is stable across random seeds
    and its geographic-coherence difference from chance is statistically
    significant, but the effect size (~12-15% deviation, silhouette
    ≤0.029) remains small — this is not evidence of clean regional
    structure."
  - Forbidden wording: "clusters correspond to distinct US regions";
    "embedding clusters map onto geography"; using "statistically
    significant" alone without the effect-size qualifier.
  - Status: resolved — reconfirmed on the full-coverage corpus, both
    before and after the 2026-07-10 embedding-source switch.

- **Claim**: PC1 of the Source A embeddings explains a small share of
  total embedding variance; on the now-adopted LLM-cleaned embeddings it
  DOES carry a real, non-economic theme (founding/namesake narrative
  presence vs. demographic-stub content), concentrated in Texas counties.
  - Evidence (adopted LLM-cleaned embeddings, 2026-07-10):
    `pca.explained_variance_ratio_[0]=0.0394`, n=3,144. Top-30 PC1 counties
    are 70% Texas (21/30) against Texas's 8.1% corpus base rate; corpus-wide,
    counties whose cleaned text contains "named for"/"named after" average
    PC1=0.110 (n=1,281) vs. -0.076 for those that don't (n=1,863). See §3.2
    for the full manual-read writeup. **Superseded regex-baseline value**:
    `explained_variance_ratio_[0]=0.0478`, manual 6-county tail read found
    no shared theme (Elliott County KY, Wise County VA, Kent County TX
    high; Clay County IN, Miami County IN, Floyd County GA low) — this
    finding does NOT carry over to the adopted embeddings.
  - Allowed wording: "PC1 explains a small share of the embedding's variance
    (~3.9%) and separates counties with rich founding/namesake narrative
    (heavily Texas-concentrated) from those reduced to bare demographic
    facts after cleaning — not an economically meaningful axis."
  - Forbidden wording: any claim that PC1 has no theme (true for the
    superseded baseline only, not the adopted embeddings); any claim that
    PC1's theme is economic or geographic rather than a Wikipedia
    content-richness/editorial-convention artifact.
  - Status: **reversed on 2026-07-10** — re-verify before reusing either
    version of this claim; state which embedding source it refers to.

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

1. **Coverage gap — resolved.** All 56 originally-failed counties are now
   ingested: 37 Virginia independent cities (`backfill_virginia_cities.py`)
   plus the remaining 19 Census-Gazetteer-name-vs-Wikipedia-title mismatches
   (8 Alaska boroughs/census areas, 5 New York City boroughs, Hawaii County,
   De Witt County IL, Larue County KY, De Soto Parish LA, Nantucket County
   MA, Le Flore County OK — via `backfill_remaining_19.py`, both now mapped
   in `INDEPENDENT_CITY_ARTICLE_LOOKUP`). The dataset covers all 3,144 US
   counties/county-equivalents; 0 known ingestion gaps remain.
2. **Artifact staleness — resolved.** `stats.json`, all three figures,
   `source_a_map.html`, and the similarity/clusters HTML artifacts were
   regenerated on 2026-07-10 against the adopted LLM-cleaned embeddings
   (`source_a_embeddings.parquet`, post §12 swap) and are mutually
   consistent (see §2, §9).
3. **PC1 has a real, non-economic meaning on the adopted embeddings** —
   3.9% of variance; a manual check of the extremes finds a
   founding/namesake-narrative-vs-demographic-stub axis, heavily
   Texas-concentrated (§3.2). This reverses the superseded regex-baseline
   finding of "no established meaning" — do not cite the old finding
   without checking which embedding source it refers to.
4. **No alternative clustering method tried** (e.g. hierarchical, DBSCAN) as
   a robustness check on the K-means finding.

## 8. Next Actions

1. **Do not build economic or geographic downstream conclusions on PC1**
   — it does carry a real theme on the adopted embeddings (§3.2), but
   that theme is a Wikipedia editorial-convention artifact (Texas-heavy
   founding/namesake narrative vs. demographic stubs), not economic or
   geographic content. **Do not treat the two K-means clusters as real
   regions** — that remains weak/unsupported.
2. **Coverage and artifact consistency are both done** as of the
   2026-07-10 embedding-source switch — no further backfill or
   regeneration work is needed for Source A's current scope.
3. **Do not pursue further Wikipedia section-selection variants** (§4 and
   §4.1, both closed, including the LLM-cleaned variant) — if stronger
   differentiation is needed, the next lever is a non-Wikipedia source
   (Sources B–F).
4. **This finding set is a reasonable checkpoint** for Source B–F planning:
   the weak-but-real geography signal, the resolved clustering-stability
   question, and full coverage are the facts most likely to matter for that
   planning.
5. **Before using Source A in E_macro as the proposal describes**, test
   embedding distance against real economic variables (Source E's
   capital-gains/W-2 ratio, Source B's industry location quotients) once
   those sources are ingested — see §10. That correlation, not geographic
   correlation, is the proposal's actual claim, and it is still untested.

## 9. Artifact and Reproducibility Index

- Ingestion: `ingest_source_a.py` (`INDEPENDENT_CITY_ARTICLE_LOOKUP` now
  covers all 37 Virginia independent cities plus the 19 remaining
  Gazetteer-vs-Wikipedia mismatches, 56 entries total),
  `backfill_virginia_cities.py` (Virginia-only re-ingestion + parquet
  merge), `backfill_remaining_19.py` (final 19 counties, re-ingestion +
  parquet merge — brings coverage to all 3,144 US counties).
- LLM cleaning / adoption (2026-07-10, §12): `reembed_source_a_llm.py`
  (gemma2:9b boilerplate-cleaning pass, full 3,144-county run, wrote
  `source_a_embeddings_llm.parquet`, resumable via
  `source_a_llm_cleaning_progress.jsonl`), `compare_llm_cleaning_full_corpus.py`
  (baseline-vs-LLM-cleaned gate check: pairwise similarity, Mantel r,
  tracked-pair mean, on the full 2,849-county set). After the gate passed,
  `source_a_embeddings_llm.parquet` was copied over `source_a_embeddings.parquet`
  (the superseded regex-cleaned version preserved at
  `source_a_embeddings_regex_baseline.parquet`).
- EDA / analysis: `visualize_source_a.py` (PCA), `analyze_source_a_similarity.py`
  (pairwise similarity/distance), `analyze_source_a_clusters.py` (K-means +
  Mantel test), `analyze_source_a_cluster_stability.py` (cross-seed
  stability + cluster-coherence permutation test). All rerun 2026-07-10
  against the adopted LLM-cleaned `source_a_embeddings.parquet`.
- Insights synthesis: `generate_source_a_insights.py` (produced the current
  `stats.json` and the three saved figures, rerun 2026-07-10).
- Persisted statistics: `analysis-output/stats.json` (adopted-embeddings
  snapshot, n=3,144 / n=2,849, 2026-07-10).
- Figures: `analysis-output/figures/figure-01-similarity-vs-distance.png`,
  `figure-02-pc1-distribution.png`, `figure-03-cluster-coherence.png`,
  `figures/source-a-numeric-summary.md` (all regenerated 2026-07-10 against
  the adopted embeddings).
- Companion artifacts: `source_a_map.html`, `source_a_similarity.html` /
  `source_a_similarity_pairs.csv`, `source_a_clusters.html` /
  `source_a_cluster_summary.csv` — all refreshed 2026-07-10, mutually
  consistent with `stats.json`.
- Notebook: `analysis-output/source_a_key_findings.ipynb` — presentation
  notebook covering the findings in §3-§6; loads the artifacts above rather
  than recomputing them, so re-run it after any future backfill or
  embedding-source change to keep it in sync. **Caveat**: several cells'
  markdown prose states numbers as hardcoded text rather than computing
  them — re-execution alone does not update prose, it must be edited
  separately (done 2026-07-10 for the adoption; if this recurs, grep the
  `.ipynb`'s markdown cells for numeric literals before trusting them
  post-rerun).
- Ingestion log: `ingest_run.log` (untracked).
- Reproduction (coverage backfill): `uv run --env-file .env backfill_virginia_cities.py`,
  then `uv run --env-file .env backfill_remaining_19.py`.
- Reproduction (LLM adoption): `uv run --env-file .env reembed_source_a_llm.py`,
  then `uv run compare_llm_cleaning_full_corpus.py` to re-check the gate,
  then re-run the EDA/insights scripts below.
- Reproduction (EDA/insights, run after either of the above):
  `uv run generate_source_a_insights.py`, `uv run visualize_source_a.py`,
  `uv run analyze_source_a_similarity.py`,
  `uv run analyze_source_a_clusters.py`,
  `uv run analyze_source_a_cluster_stability.py`, then
  `uv run jupyter nbconvert --to notebook --execute --inplace analysis-output/source_a_key_findings.ipynb`
  (plus a manual pass over its markdown cells, see caveat above). All
  seeded (`RANDOM_SEED=42` throughout; stability seeds 42/7/123/2024/99).

## 10. Proposal Alignment Assessment (E_macro Extended Proposal, 2026-07-03)

**Question**: does Source A, as specified in `E_macro_extendedProposal.pdf`
(Wikipedia intro-text embeddings via `bge-m3`, providing "narrative
identity" / qualitative context for the Capital Flow pillar alongside
Source E), deliver on that role?

**Proposal's claim**: intro text contains distinctive economic-transition
language (e.g. "former rust-belt manufacturing center currently
transitioning to a health-tech hub") that gives immediate semantic
separation between counties, breaking symmetries invisible to numeric or
visual data.

**Assessment against §3–4 findings**:
- If economic-narrative language of the kind the proposal describes were
  common and distinguishing, it should surface as structure in PCA or
  clustering. It doesn't: PC1 explains only 3.9% of variance (adopted
  LLM-cleaned embeddings), and while it does carry a real theme (§3.2),
  that theme is founding/namesake narrative presence vs. demographic-stub
  content — not economic-transition narrative. K-means finds no cleanly
  separated structure (silhouette ≤0.029, §3.3) beyond the weak geographic
  effect.
- The manual tail-read of PC1's extreme counties (§3.2) found
  Texas-concentrated founding/namesake history at the high end and bare
  population/seat facts at the low end — genuinely distinctive content on
  the high end, but not economic-transition narrative of the kind the
  proposal describes.
- §4/§4.1's negative results reinforce this: embedding more article text made
  counties *more* similar, consistent with Wikipedia county articles being
  dominated by templated incorporation/geography/demographics prose rather
  than distinctive economic narrative.
- **Not tested**: correlation between Source A embedding distance and
  actual economic variables (Source E's capital-gains/W-2 ratio, Source B's
  industry location quotients). No economic ground truth is in-repo yet
  (§2), so this is an open gap, not a resolved negative.

**Verdict**: Source A's specific mechanism in the proposal — that intro
text carries salient economic-transition language usable as narrative
context for Capital Flow — is not supported by the unsupervised analysis.
The corpus instead reads as generic boilerplate with a very small
geographic echo. This does not mean Source A should be scrapped (the
geographic signal is real, and correlation with actual economic labels is
untested), but it should not be treated as validated for the proposal's
stated role. Before using Source A in E_macro:
1. Test embedding distance against real economic variables (Source E
   ratio, Source B location quotients) once those sources are ingested —
   the proposal's actual claim is economic correlation, not geographic
   correlation, and that's still untested.
2. If that correlation is also weak, consider targeted extraction
   (sentences containing transition/industry keywords) rather than
   whole-intro embedding, since naive full-text expansion already failed
   (§4).

**Claim candidate**:
- **Claim**: Source A's intro-text embeddings do not show the
  distinctive economic-transition signal the E_macro proposal's Source A
  justification assumes; the strongest structure found (PC1) is a
  Wikipedia editorial-convention artifact, not economic content.
  - Evidence (adopted LLM-cleaned embeddings, 2026-07-10): PC1 3.9%
    variance, real but non-economic theme — founding/namesake narrative
    vs. demographic-stub content, Texas-concentrated (§3.2); K-means
    silhouette ≤0.029 (§3.3); section-expansion negative result including
    the LLM-cleaned variant (§4, §4.1).
  - Allowed wording: "unsupervised analysis finds no evidence of the
    distinctive economic-narrative signal the proposal's Source A
    justification describes; the strongest structure found is a
    Wikipedia editorial-convention artifact (Texas founding-history
    detail vs. bare demographic facts elsewhere), plus a weak geographic
    echo."
  - Forbidden wording: "Source A has no economic signal" — untested
    against real economic variables; only claims about *self-organizing /
    unsupervised* structure are supported here.
  - Status: open — correlation against Source E/B ground truth not yet
    run.

## 11. Model Choice Rationale: `bge-m3` vs. a Frozen LLM Encoder (e.g., Gemma)

**Context**: `E_macro_extendedProposal.pdf` (§2, Source A) and
`macro_pre_scoping_spec.pdf` (§Source A) both specify `bge-m3` as the
embedding backbone but do not argue for it over alternatives — the choice
was never compared against using a frozen decoder-only LLM (e.g. Gemma) as
an embedding source. That comparison has not been run empirically in this
project; the reasoning below is a design rationale, not a tested ablation.

**Why `bge-m3` was used**:

1. **Training objective match.** `bge-m3` is trained with a contrastive/
   retrieval objective specifically to place semantically similar texts
   close together under cosine similarity. A frozen decoder-only LLM like
   Gemma is trained for next-token prediction; its hidden states are not
   calibrated for similarity search out of the box. Pooling a frozen LLM's
   hidden states (mean- or last-token-pooling) is a known workaround, not
   what the model was optimized to produce, and typically needs
   contrastive fine-tuning (as in dedicated embedding variants) before it
   is competitive with a purpose-built encoder.
2. **Benchmark performance for this task type.** Purpose-built embedding
   models in `bge-m3`'s class lead retrieval/STS benchmarks (e.g. MTEB) at
   their parameter size; frozen general-purpose LLMs underperform them on
   semantic-similarity tasks unless separately fine-tuned into an
   embedding checkpoint — at which point they are no longer "frozen."
3. **Context length fits the ingestion design.** `bge-m3` natively supports
   up to 8,192 tokens, which is what `ingest_source_a.py` and the
   pre-scoping spec's chunking/mean-pooling fallback (`pre_scoping.txt`
   §Source A) are built around. A different backbone with a shorter native
   context would force chunking on far more articles than currently
   needed.
4. **Compute footprint.** `bge-m3` (568M params) runs on CPU at usable
   throughput for ~3,144 short texts (see `README.md`'s note that
   MPS/GPU auto-selection was *slower* than CPU for this model on short
   inputs). A multi-billion-parameter decoder LLM would cost substantially
   more compute for the same end product — a single fixed vector per
   county — with no generation capability actually used.
5. **Single deterministic forward pass.** Producing an embedding is one
   forward pass with pooling, not autoregressive sampling, which fits this
   project's reproducibility requirements (fixed seeds, deterministic
   outputs elsewhere in the pipeline) more naturally than adapting a
   generation-oriented model into an embedding role.

**Caveat tying back to §3–4's findings**: switching to a frozen LLM encoder
would not obviously fix the weak differentiation this report documents
(PC1 at ~3.9% variance, silhouette ≤0.029, section-expansion making
counties more similar rather than less, in both its raw and LLM-cleaned
forms — §4.1). Frozen decoder-LLM hidden states
are themselves known to concentrate variance in a small number of
high-magnitude dimensions, which tends to *hurt* naive similarity
comparisons rather than help them. The weak signal found in §3–4 looks more
like a property of the source text (generic Wikipedia county boilerplate)
than something a different backbone would resolve on its own — but since
no frozen-LLM-encoder ablation has been run against this corpus, that
remains an inference, not a measured result.

- **Status**: design rationale only — no in-repo ablation compares `bge-m3`
  against a frozen-LLM-encoder baseline on Source A's data. If that
  comparison becomes worth running, it would need to reuse the same
  40-county sample and Mantel/pairwise-similarity protocol from §4 to be
  comparable to existing results.

## 12. De-Boilerplating via LLM Rewrite (`gemma2:9b`) — Adopted

**Context**: the regex/corpus-frequency approach outlined in a since-removed
`PLAN.md` (targeted patterns for eponym clauses, metro-area sentences, etc.,
plus a corpus-frequency sentence filter) was judged to be an escalating,
hard-to-explain fix — every new boilerplate phrasing found would need
another regex. Instead: a single local LLM call per county rewrites
`raw_intro_text`, instructed to keep only county-specific facts and drop
generic templated phrasing, with an explicit no-fabrication constraint.
Output is re-embedded with the existing `bge-m3` model — no change to the
embedding backbone itself, only to what text is fed into it.

**Method**: `gemma2:9b` via a local Ollama server, temperature 0, one prompt
per county (see `reembed_source_a_llm.py`). An initial prompt version
produced a fabricated-content hallucination for 1 of 8 tracked-pair
counties (invented geography/economy detail not present in the source
text); tightening the prompt to explicitly forbid adding any fact not
literally stated in the input eliminated it, confirmed across all
subsequent runs.

**Results** (baseline = current `source_a_embeddings.parquet`):

| Sample | Metric | Baseline | Cleaned | Δ |
|---|---|---|---|---|
| 8 tracked-pair counties (§3.4) | `tracked_pair_mean` | 0.832 | 0.639 | **−0.193** |
| 128 counties (8 tracked + 120 random, seed 42) | pairwise similarity mean | 0.545 | 0.501 | −0.044 |
| 128 counties | pairwise similarity std | 0.071 | 0.060 | **−0.011** |
| 128 counties | hallucination flags (cleaned text longer than raw, manually spot-checked) | — | — | 0 / 128 |

The `tracked_pair_mean` drop is roughly 6x the PLAN.md decision gate's
minimum threshold (−0.03), and driven mostly by the three
generic-boilerplate-convergence pairs (Montgomery↔Stutsman, Stutsman↔
Williamsburg, Stutsman↔Providence: −0.25 to −0.32 each) rather than the
shared-eponym pairs, consistent with the LLM correctly identifying and
removing template-only content rather than just deleting names.

**Known tradeoff**: pairwise similarity std shrank by ~15% relative
(reproduced at both n=38 and n=128, not sampling noise) — the corpus
became slightly *more* uniform in similarity spread, likely because the
LLM tends to normalize every article into the same terse factual register
regardless of the original prose's length or style. Accepted as a
reasonable tradeoff for the large gain on the actual boilerplate-similarity
problem and the simplicity of a single-mechanism approach (one prompt,
no regex catalog, no frequency-threshold tuning).

**Full-corpus validation (2026-07-10)**: `reembed_source_a_llm.py` completed
all 3,144 counties, written to `source_a_embeddings_llm.parquet`
(`source_a_embeddings.parquet` untouched). `compare_llm_cleaning_full_corpus.py`
reran the same comparison on the full 2,849-county 50-state/non-stub set
(`filter_to_fifty_states` + `drop_stub_counties`, matching §3-4's
methodology) plus a fresh Mantel test not run on the subset:

| Metric | Baseline | LLM-cleaned | Δ |
|---|---|---|---|
| Pairwise similarity mean (n=2,849) | 0.548 | 0.495 | -0.053 |
| Pairwise similarity std | 0.063 | 0.058 | -0.005 |
| Mantel r (geo↔similarity) | -0.106, p=0.002 | **-0.236, p=0.002** | -0.130 |
| `tracked_pair_mean` (§3.4, n=5 pairs) | 0.832 | 0.639 | -0.194 |

Both the similarity-spread and `tracked_pair_mean` results reproduce the
128-county subset almost exactly (Δ-0.053 vs. the subset's Δ-0.044;
`tracked_pair_mean` Δ-0.194 vs. the subset's Δ-0.193), still ~6.5× past the
PLAN.md decision gate. The per-pair breakdown reproduces the same
mechanism: generic-boilerplate-convergence pairs collapsed hard (Stutsman↔
Williamsburg 0.833→0.513, Stutsman↔Providence 0.826→0.526), shared-eponym
pairs barely moved (Lincoln KS↔OR 0.845→0.825).

New at full scale: the geographic signal **more than doubled** (Mantel r
-0.106→-0.236, both significant at the 499-permutation floor, p=0.002) —
cleaning doesn't just spread counties apart, it spreads them apart in a way
that tracks geography better than the currently-adopted baseline. This
wasn't measured on the subset.

- **Status**: **adopted and fully regenerated (2026-07-10)**.
  `source_a_embeddings.parquet` now contains the LLM-cleaned embeddings (the
  former regex-cleaned baseline is preserved at
  `source_a_embeddings_regex_baseline.parquet`, and unaltered in git
  history). Every script that reads `EMBEDDINGS_PARQUET_PATH` /
  `source_a_embeddings.parquet` (`visualize_source_a.py`,
  `analyze_source_a_clusters.py`, `analyze_source_a_similarity.py`,
  `analyze_source_a_cluster_stability.py`, `generate_source_a_insights.py`,
  the key-findings notebook) reads the adopted embeddings without code
  changes, and all of their output artifacts (`source_a_map.html`,
  `source_a_clusters.html`, `source_a_similarity.html`,
  `source_a_similarity_pairs.csv`, `source_a_cluster_summary.csv`,
  `stats.json`, `figures/*.png`, `figures/source-a-numeric-summary.md`,
  `source_a_key_findings.ipynb`) have been regenerated against it and are
  mutually consistent — see §9 for the updated index. The notebook's
  markdown prose (hardcoded numbers, not cell-computed) was patched
  separately since re-execution alone does not update prose text; see §3.2
  for the one substantive change re-execution surfaced (PC1 now has a real
  theme, reversing the superseded baseline's finding).
