---
type: results-report
date: 2026-08-03
experiment_line: source-a
round: 4
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

Figures live in `analysis-output/source-a/figures/` (all regenerated 2026-07-10
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
- **Five of those scripts were deleted 2026-08-03** as the last artifacts of
  the cut embedding, on the same grounds as `reembed_source_a_llm.py` below:
  `analyze_source_a_clusters.py`, `analyze_source_a_cluster_stability.py`,
  `generate_source_a_insights.py`, `analyze_source_a_source_c_correlation.py`,
  `analyze_source_a_source_f_correlation.py`. Nothing outside that set imported
  them. Every reference to them in §3–§6, §10 and the reproduction block below
  is a record of what was run, not a live path — recover from git history
  (`git log -- scripts/analyze_source_a_clusters.py`) if the cut is ever
  reversed. `visualize_source_a.py` and `analyze_source_a_similarity.py`
  **stay**: sources B–F import `fetch_county_centroids`,
  `CENTROIDS_CACHE_PATH` and `haversine_distance_matrix` from them, so they are
  shared geospatial utilities rather than Source A EDA. Their numeric outputs
  survive in `stats.json`, `figures/`, `outputs/source_a_similarity_pairs.csv`,
  `outputs/source_a_cluster_summary.csv`,
  `outputs/source_a_source_c_correlation_pairs.csv` and
  `outputs/source_a_source_f_crossvalidation.csv`; only the five multi-megabyte
  Plotly `.html` renders were removed with them.
- **Typed-extraction era (§13–§17)**: `extract_source_a_features.py` and
  `extract_source_a_section_features.py` (the 29 columns),
  `analyze_source_a_tiers.py`, `analyze_source_a_representation.py` (thin
  baseline, §14), `analyze_source_a_marginal.py` (crowded baseline, §17), and
  `paired_power.py` (effect size, achieved power, and pillar-clustering
  diagnostics for every paired test in §14 and §17 — no point estimate depends on
  it, it only says how much weight each p-value carries). Outputs:
  `outputs/source_a_representation.csv`, `..._by_tier.csv`, `..._by_pillar.csv`,
  `outputs/source_a_marginal.csv`, `..._by_pillar.csv`, and the matching
  `source_a_representation_stats.json` / `source_a_marginal_stats.json`. **The
  two `_by_pillar.csv` files are the primary reportable results** (§14.2c,
  §17.2a); the aggregates in the headline tables are secondary and are weighted
  71% toward QCEW.
- Persisted statistics: `analysis-output/source-a/stats.json` (adopted-embeddings
  snapshot, n=3,144 / n=2,849, 2026-07-10).
- Figures: `analysis-output/source-a/figures/figure-01-similarity-vs-distance.png`,
  `figure-02-pc1-distribution.png`, `figure-03-cluster-coherence.png`,
  `figures/source-a-numeric-summary.md` (all regenerated 2026-07-10 against
  the adopted embeddings).
- Companion artifacts: `outputs/source_a_similarity_pairs.csv`,
  `outputs/source_a_cluster_summary.csv` — refreshed 2026-07-10, mutually
  consistent with `stats.json`. Their Plotly companions
  (`source_a_map.html`, `source_a_similarity.html`, `source_a_clusters.html`)
  were deleted 2026-08-03; the CSVs carry the numbers, the renders did not.
- Notebook: `analysis-output/source-a/source_a_key_findings.ipynb` — presentation
  notebook covering the findings in §3-§6; loads the artifacts above rather
  than recomputing them, so re-run it after any future backfill or
  embedding-source change to keep it in sync. **Caveat**: several cells'
  markdown prose states numbers as hardcoded text rather than computing
  them — re-execution alone does not update prose, it must be edited
  separately (done 2026-07-10 for the adoption; if this recurs, grep the
  `.ipynb`'s markdown cells for numeric literals before trusting them
  post-rerun).
- Ingestion log: `ingest_run.log` (untracked).
- Reproduction (coverage backfill): `uv run --env-file .env python scripts/backfill_virginia_cities.py`,
  then `uv run --env-file .env python scripts/backfill_remaining_19.py`.
- Reproduction (LLM adoption): **no longer runnable as written (2026-07-27).**
  `reembed_source_a_llm.py` and `compare_llm_cleaning_full_corpus.py` were
  deleted from `scripts/` once the embedding step itself was retired (see
  `analysis-output/E_macro_key_findings.ipynb` §2); their intermediate,
  `data/source_a_embeddings_llm.parquet`, was never kept either. Every
  reference to those two scripts below and in §12 is a record of what was
  run at the time, not a live path. Recover them from git history
  (`git log -- scripts/reembed_source_a_llm.py`) if this needs re-running.
  The adopted output survives as `data/source_a_embeddings.parquet`, with
  the superseded regex version at `data/source_a_embeddings_regex_baseline.parquet`.
- Reproduction (EDA/insights, run after either of the above):
  **partly unrunnable as written (2026-08-03)** — three of the five scripts
  below were deleted with the embedding, see the EDA bullet above.
  `uv run python scripts/generate_source_a_insights.py` *(deleted)*,
  `uv run python scripts/visualize_source_a.py`,
  `uv run python scripts/analyze_source_a_similarity.py`,
  `uv run python scripts/analyze_source_a_clusters.py` *(deleted)*,
  `uv run python scripts/analyze_source_a_cluster_stability.py` *(deleted)*, then
  `uv run jupyter nbconvert --to notebook --execute --inplace analysis-output/source-a/source_a_key_findings.ipynb`
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

## 13. Typed Feature Extraction — Beating `content_length` Without the Embedding (2026-08-03)

**Context**: the `bge-m3` embedding step was cut from the pipeline (see
`E_macro_key_findings.ipynb` §2). The cut rested on cost, not on absence of
signal — the embedding beat `content_length` on 23 of 28 cross-pillar targets
(Wilcoxon p = 4.2e-5) but by +0.0030 mean R² lift against +0.0010, for a 2.2GB
model and CPU inference over 3,144 articles. That left Source A shipping one
scalar. This round asks whether cheap typed extraction can do better.

### 13.1 The corpus is extremely uneven, and the unevenness is economic

Splitting all 3,144 counties into content tiers on `content_length`
(`analyze_source_a_tiers.py`; stub <100 chars, thin 100–283, mid 284–461,
rich ≥462):

| tier | n | any named industry | mean distinct proper nouns | founding year present |
|---|---|---|---|---|
| stub | 294 | 1.0% | 2.0 | 5.8% |
| thin | 1,281 | 1.1% | 5.2 | 32.9% |
| mid | 784 | 5.5% | 8.8 | 52.4% |
| rich | 785 | **25.2%** | 17.8 | 46.5% |

Named industry content is **23× more common in the rich tier than the thin
tier**, but only 6.5% of the corpus carries it at all. Two consequences:

1. A dense representation averaged over all 3,144 articles is dominated by
   counties with no economic content. This is the quantitative form of the
   heterogeneity that §4.1 identified as a failure *mechanism* (LLM cleaning kept
   geographic anchoring for some counties and dropped it for others).
2. The founding/namesake axis is flat across the thin/mid/rich tiers
   (42.5 / 53.7 / 51.7%) — which is exactly why PC1 (§3.2) was a dead end. It has
   mass everywhere and therefore separates nothing.

**Caution on an earlier draft of this table**: a first pass reported the industry
gradient as 9.4% → 43.1% and the corpus rate as 19.7%. Those numbers were wrong
in both directions — the patterns were case-sensitive (missing "Metropolitan
Statistical Area") and lacked word boundaries (`port` matched "important",
"airport", "transport"). The corrected figures above are steeper in gradient and
much lower in absolute rate. Any reuse should take these, not the earlier ones.

### 13.2 Method: one uniform schema, sparsity encodes the tier

`extract_source_a_features.py` writes 20 typed columns for every county from
`raw_intro_text` — industry family flags, institution flags (university,
military base, protected land, tribal land), transport flags, metro attachment,
namesake, founding year, and a distinct-proper-noun count. Design points:

- **Absence is `False`, not null.** A stub county returns False across the board
  and that sparsity *is* its tier. One schema for all 3,144 counties, so the
  feature-store handoff has no "not applicable" null category.
- **Extraction reads `raw_intro_text`, never `embedding_text`.** The corpus
  stripper that produces the latter removes the county name, the state name, and
  "U.S. state of" along with boilerplate, leaving damaged input.
- **Tiers are not shipped.** Tier membership tracks county size, so it is used to
  route work and break out results, never as a feature.
- **Precision was checked on sampled matches, and two flags failed it.**
  `has_military_base` originally matched `\bFort [A-Z]` and `\bArmy\b`: five of
  six sampled hits were false ("Fort Wayne" and "Fort Yates" are city names,
  "Fort Lemhi" an 1855 Mormon settlement). `has_tribal_land` matched bare
  `Indian`, catching "American Indian Wars" and reservations dissolved in the
  1830s. Both were tightened to require installation/present-tense-land terms,
  cutting them from 163→21 and 157→79 counties respectively. This step is what
  separates lexicon extraction from plausible-looking noise.

### 13.3 Result: 2.6× the incumbent, 94% of the embedding, at 1/50th the width

`analyze_source_a_representation.py`, same protocol as the 2026-07-27 run — 28
targets in pillars B–F, unpenalized size-plus-state baseline, each representation
fitted to its residuals, ridge penalty by nested crossvalidation, seed 42, 5 folds.

| variant | columns | mean R² lift | raw R² alone | beats `length` | Wilcoxon p |
|---|---|---|---|---|---|
| `content_length` (incumbent) | 1 | +0.00098 | 0.019 | — | — |
| `extracted_min` | 4 | +0.00223 | 0.041 | 14/28 | 0.493 |
| `extracted_mid` | 8 | +0.00231 | 0.041 | 17/28 | 0.053 |
| `extracted_full` | 20 | **+0.00257** | 0.044 | 16/28 | 0.274 |
| `bge-m3` PCA-50 | 50 | +0.00171 | 0.085 | 13/28 | 0.522 |
| `bge-m3` full | 1024 | +0.00273 | 0.112 | 21/28 | 0.0008 |

**The mean-lift criterion passes decisively; the paired-consistency criterion does
not.** Extraction delivers 2.6× the incumbent's mean lift and 94% of the 1024-dim
embedding's, for 20 regex columns and no model download. But it gets there by
winning large on a handful of targets and tying on the rest — mean difference
against the incumbent is +0.00159 while the *median* is +0.00021 — so the
rank-based Wilcoxon does not reach significance. The embedding shows the opposite
profile: smaller per-target gains, but on 21 of 28 targets, which is what makes
its test significant.

Stated plainly: **extraction is the better representation on average and the
cheaper one by far, but it is not uniformly better target by target.**

### 13.4 The wins are semantically coherent, not curve-fitting

The largest single gain is Accommodation & Food Services LQ (`lq_emp_72`,
+0.0189 against the incumbent's −0.0000), and it has an obvious mechanism:

| | mean `lq_emp_72` | n |
|---|---|---|
| intro mentions tourism | 1.407 | 78 |
| intro does not | 1.010 | 2,291 |

Wikipedia saying "resort", "ski", or "casino" predicts measured tourism
employment (r = 0.157). Other top gains follow the same pattern — Information LQ
(+0.0086 over incumbent), Finance & Insurance LQ (+0.0054), and Source E's
capital-to-wage ratio (+0.0049), where extraction beats even the full embedding
(+0.0058 vs +0.0026). Source E is Capital Flow, the pillar the E_macro proposal
originally assigned Source A to support (§10).

### 13.5 Gains concentrate where the content is — the tier hypothesis holds

Mean lift by content tier, re-scored from the same out-of-fold predictions
(`outputs/source_a_representation_by_tier.csv`, tiers with <150 rows suppressed):

| variant | stub | thin | mid | rich |
|---|---|---|---|---|
| `content_length` | +0.00060 | +0.00039 | +0.00008 | +0.00331 |
| `extracted_full` | −0.00007 | +0.00243 | +0.00104 | **+0.00501** |
| `bge-m3` full | +0.00103 | +0.00236 | +0.00054 | +0.00545 |

Extraction contributes essentially nothing on stub counties — correctly, since
there is nothing to extract — and most in the rich tier, where it nearly matches
the 1024-dim embedding (+0.00501 vs +0.00545). **The heterogeneity hypothesis is
confirmed: the pillar's value is concentrated in the counties whose articles have
something to say.**

### 13.6 The win is not an artifact of Source F circularity

Some intros restate USDA's own county classification verbatim (Marquette County
WI: "considered a high-recreation retirement destination by the U.S. Department
of Agriculture"), and Source F's `distress_count` is built from those
classifications. `has_usda_echo` flags the 16 counties affected and is excluded
from every scored variant. As a stronger check, dropping Source F's target
entirely: `extracted_full` +0.00190 vs incumbent +0.00073 across the remaining 27
targets — still 2.6×. The advantage does not depend on Source F.

### 13.7 Dimensionality: reduce by column selection, never by PCA

The nested widths are close (+0.00223 / +0.00231 / +0.00257 for 4 / 8 / 20
columns), so most of the value sits in the four-column core — `content_length`,
`n_industry_mentions`, `has_metro_attachment`, `n_distinct_proper_nouns` — and the
remaining sixteen add roughly 15% more. Either is defensible; the 4-column block
is the better feature-store citizen if width matters at fusion.

PCA is separately ruled out. On the embedding, compressing 1024 → 50 retained only
42% of its advantage over `content_length` (+0.00073 vs +0.00175), losing on 23 of
28 targets (p = 0.0009). The mechanism is §3.2: this corpus's highest-variance
direction is the Texas founding-narrative artifact, so a variance criterion
selects against economic content. Ridge over the full width is itself a soft,
target-aware reduction and does the job better. **If the embedding is ever
reinstated, reduce it supervised (PLS) or not at all.**

### 13.8 Status and open items

- **Status**: `extracted_full`'s 20 columns are written into
  `data/source_a_text_features.parquet` and flow into `pillar_matrix.build_matrix`
  as Source A's block (A = 21 columns) with no change to that module.
- **Allowed wording**: "typed extraction from Wikipedia intros delivers 2.6× the
  mean cross-pillar lift of article length, and 94% of the cut embedding's, at 20
  columns and no model download — but its per-target advantage is concentrated
  rather than uniform, and does not reach paired significance."
- **Forbidden wording**: "extraction beats the embedding" (it does not, on
  consistency: 16/28 vs 21/28 targets); "the improvement is statistically
  significant" (the paired Wilcoxon is p = 0.274).
- **Body sections — now tested, see §14 (resolved).** Originally open: Extraction runs on intro text only. The
  feature family that produces the largest wins (named industry) fires on just
  6.5% of counties, while §4 found ~25% of counties have a dedicated Economy
  section that was never parsed. Fetch cost is flat — `extract_article_html`
  already returns the full article body and `isolate_lead_section` discards
  everything after the lead — so this is a parsing question, not an acquisition
  cost question. §4/§4.1 closed section expansion, but did so on Mantel-r against
  *geographic* distance, the yardstick this project has since rejected for exactly
  this pillar. Re-testing against economic targets is the open increment.
- **Reproduction**: `uv run python scripts/extract_source_a_features.py`, then
  `uv run python scripts/analyze_source_a_tiers.py`, then
  `uv run python scripts/analyze_source_a_representation.py`. Seed 42 throughout.

## 14. Body Sections Reopened — And They Pay (2026-08-03)

**Context**: §13.8 left one open item — extraction ran on intro text only, while
the feature family producing its largest wins (named industry) reached just 8.2%
of counties. §4/§4.1 had closed section expansion, but on Mantel-r against
geographic distance, the yardstick this project rejected for this pillar. This
section re-opens it against economic targets.

Two things made the re-test cheap and the old objection weak:

- **Fetch cost was always flat.** `extract_article_html` has always returned the
  full article body and `isolate_lead_section` has always discarded everything
  after the lead. The re-ingest cost the same 3,144 requests as the original.
  `ingest_source_a.py` now persists every body section to
  `data/source_a_sections.parquet` (64,588 rows, 20.5 sections per county, 3,144
  counties, 0 failures), so no section question needs another fetch.
- **§4's finding was that body sections are *more templated* than the lead.** That
  is fatal for a dense embedding, which absorbs boilerplate into every dimension.
  Targeted extraction reads named facts and ignores prose, so templating costs it
  little. §4.1's diagnosed failure mode — an LLM cleaner keeping geography for
  some counties and dropping it for others — cannot occur with a fixed lexicon.

`extract_source_a_section_features.py` therefore applies only the industry
lexicon, and only to sections whose title marks them economic. This is targeted
extraction from one named section, not section expansion.

### 14.1 Yield: sections help most where the intro says least

| tier | has economy section | industry in intro | **industry added by sections** |
|---|---|---|---|
| stub | 10.5% | 0.7% | +5.4% |
| thin | 14.2% | 1.1% | **+8.6%** |
| mid | 21.2% | 5.5% | +12.6% |
| rich | 35.7% | 25.3% | +13.7% |

Corpus-wide, industry coverage rises from **8.2% to 18.8% (+332 counties)**. The
marginal yield is largest in absolute terms for the rich tier, but the thin tier
gains roughly 8× relative to its own near-zero base — these are counties whose
lead section says nothing and whose Economy section says something. That is the
opposite of the intuition that rich counties justify deeper reading, and it is
what §4's intro-only framing could not see.

### 14.2 Result: 29 typed columns beat the 1024-dim embedding

Full re-run on the refetched corpus, same protocol throughout (28 targets, seed
42, 5 folds, unpenalized size-plus-state baseline, nested-CV ridge penalty):

| variant | columns | mean R² lift | raw R² alone | beats `length` | Wilcoxon p |
|---|---|---|---|---|---|
| `content_length` (incumbent) | 1 | +0.00117 | 0.020 | — | — |
| `extracted_min` | 4 | +0.00254 | 0.042 | 13/28 | 0.493 |
| `extracted_mid` | 8 | +0.00243 | 0.042 | 19/28 | 0.066 |
| `extracted_full` (intro only) | 20 | +0.00263 | 0.044 | 16/28 | 0.339 |
| **`extracted_sections`** | 29 | **+0.00320** | 0.048 | 19/28 | 0.082 |
| `bge-m3` PCA-50 | 50 | +0.00171 | 0.085 | 13/28 | 0.678 |
| `bge-m3` full | 1024 | +0.00273 | 0.112 | 19/28 | 0.014 |

**29 interpretable regex columns now exceed the 1024-dim embedding's mean lift
(+0.00320 against +0.00273), at 2.7× the incumbent scalar, with no model
download.** Adding the economy section is worth +0.00057 over intro-only
extraction — about 22% more lift for 9 more columns.

The consistency caveat from §13.3 still stands: p = 0.082, short of 0.05. The
extraction variants win large on a handful of targets and tie elsewhere, while
the embedding wins smaller but on more of them. **Extraction is the better and
far cheaper representation on average; it is not uniformly better target by
target, and that should not be claimed.** §14.2a gives that p-value the sample
size it needed, which changes how it should be read.

### 14.2a That p = 0.082 is underpowered, not close

`p = 0.082` has been reported four times in this file without the sample size it
would have needed. Paired difference, `extracted_sections` minus
`content_length`, across the 28 targets:

| statistic | value |
|---|---|
| mean | +0.00203 |
| median | +0.00061 |
| sd | 0.00605 |
| Cohen dz | 0.335 |
| **power at n = 28, α = 0.05 one-sided** | **0.53** |
| targets needed for 80% power | 57 |
| targets needed for 90% power | 78 |

**Power of 0.53 means that if the observed effect is the true effect, this test
detects it about half the time.** `p = 0.082` is the ordinary output of a real
effect measured with roughly half the sample its effect size requires. It is not
evidence against the effect, and it must not be written up as though the question
were close or as though the result had failed a fair test.

**It is not a test-choice problem.** The suspicion that Wilcoxon misses the
result by discarding magnitude was checked directly: Wilcoxon signed-rank
p = 0.0815, paired t p = 0.0873. Switching to a magnitude-weighted test changes
the third decimal. Any proposal to "use a better test" is proposing a rounding
error.

**The mean is carried by five targets.** Mean is 3.3× the median because the
distribution is concentrated: Accommodation & Food Services LQ (+0.02619),
demographic distress count (+0.01356), Information LQ (+0.00783),
capital-to-wage ratio (+0.00590), Transportation & Warehousing LQ (+0.00428).
Losses concentrate too — Professional Services −0.00653, Retail Trade −0.00294,
Educational Services −0.00284. Dropping Accommodation alone roughly halves the
mean. Its mechanism is documented and independently verified (§13.4: counties
whose articles mention tourism average 1.407 Accommodation LQ against 1.010 for
those that do not, r = 0.157), so the concentration is explicable rather than
suspicious — but the headline rests heavily on one target.

**Against the embedding the two are a statistical dead heat.** Head to head,
`extracted_sections` minus `bge-m3` full: mean difference **+0.00047, 13 of 28
targets, Wilcoxon p = 0.76**, dz = 0.089. §14.5 already forbids "significantly
beats the embedding," which is correct, but the true relationship is weaker than
that phrasing implies. Typed extraction is *statistically indistinguishable* from
the 1024-dim embedding and wins on cost, interpretability, and the absence of a
2.2GB model download. **Those are the defensible arguments.** The +0.00320
against +0.00273 gap is noise and must not be leaned on.

One asymmetry cuts the other way and belongs on the record: against
`content_length`, the embedding's advantage is both larger and more consistent
(dz = 0.435, power 0.72, Wilcoxon p = 0.014, and it survives the pillar-blocked
test at p = 0.022) where the typed block's does not (pillar-blocked p = 0.132).
That is the same fact §14.2 states in prose — the embedding wins smaller but on
more targets — now measured. It does not change the shipping decision, because
the two tie head to head and only one of them costs a model download.

### 14.2b The 28 targets are 5 pillars, and 20 of them are one table

| target pillar | count | share |
|---|---|---|
| **B (QCEW location quotients)** | **20** | **71%** |
| C (velocity series) | 3 | 11% |
| D (freight) | 3 | 11% |
| E (capital-to-wage) | 1 | 4% |
| F (typology) | 1 | 4% |

Seventy-one percent of the basket is a single table, and QCEW location quotients
are compositional — each is a share against a national base, so they are
mechanically coupled. **Every claim to 28-target breadth in this file overstates
the evidence**; the honest phrasing is "28 targets drawn from five pillars, 20 of
them QCEW sectors."

The natural follow-on worry — that the significance test is therefore inflated by
treating coupled targets as independent draws — was tested and **does not hold
for the shipping variant.** `analyze_source_a_representation.py` now reports the
intraclass correlation of the paired differences within target pillar, the Kish
design effect it implies, and the effective n:

| variant | ICC of differences | effective n | pillar-blocked p |
|---|---|---|---|
| `extracted_min` | 0.360 | 15.5 | 0.212 |
| `extracted_mid` | 0.351 | 15.6 | 0.158 |
| `extracted_full` | 0.158 | 20.6 | 0.158 |
| **`extracted_sections`** | **0.031** | **26.2** | 0.132 |
| `bge-m3` full | 0.000 | 28.0 | 0.022 |

The targets are coupled; the *paired differences between two variants scored on
them* largely are not, because the coupling affects both variants and cancels in
the difference. For `extracted_sections` the design effect is 1.07, so the
nominal 28 is very nearly the effective 28 and `p = 0.082` needs no clustering
discount. The narrower variants do carry real within-pillar dependence and their
p-values should be read against an effective n near 15.

The pillar-blocked column is a sensitivity check, not a replacement: it gives
each pillar one vote regardless of how many targets it supplied, which is the
right correction for a breadth claim but leaves a five-observation test with very
little power. It is reported so that no reader has to take the aggregate on
trust.

### 14.2c The aggregate is a property of the basket

| target pillar | `content_length` | `extracted_sections` |
|---|---|---|
| B (20 targets) | +0.00099 | +0.00258 |
| C (3 targets) | +0.00062 | +0.00166 |
| D (3 targets) | +0.00075 | +0.00156 |
| E (1 target) | +0.00090 | +0.00680 |
| F (1 target) | +0.00803 | +0.02158 |

The typed block leads in every pillar, which is the useful robustness statement
and a stronger one than the aggregate. But the aggregate itself is weighted 71%
toward B, where the margin is smallest in absolute terms. Full table in
`outputs/source_a_representation_by_pillar.csv`; **this breakout is the primary
result and the single mean is secondary.**

*Numbers differ slightly from §13 because the refetch pulled live Wikipedia text
three weeks newer — `content_length` mean moved 388.1 → 390.0 and the incumbent's
mean lift 0.00098 → 0.00117. §14's table supersedes §13.3's. The embedding's own
lift is unchanged at +0.00273 (it is scored from the frozen July parquet), so it
remains a valid reference point, though it is now measured against marginally
different text than the extraction variants.*

### 14.3 The gain is content, not another size proxy

`n_body_sections` correlates with county size at r = 0.550 against log tax
returns — higher than `content_length`'s 0.359 and the most size-dependent column
in Source A. An ablation isolates its contribution:

| feature set | columns | mean lift |
|---|---|---|
| intro only | 20 | +0.00263 |
| + sections, including `n_body_sections` | 30 | +0.00328 |
| + sections, `n_body_sections` dropped | 29 | +0.00320 |
| + sections, both structural columns dropped | 28 | +0.00320 |

**97.6% of the section gain survives removing the size proxy.** The signal is
`sec_n_industry_mentions` (r = 0.108 with size — effectively size-independent),
not section count. `n_body_sections` is therefore written to the parquet as a
diagnostic and excluded from the scored feature set, on the same footing as
`has_usda_echo`: 2.4% of the gain does not justify that much size dependence in a
feature set whose central open question is whether size is a control or a target.

### 14.4 Per-tier: the section variant wins in every tier

| variant | stub | thin | mid | rich |
|---|---|---|---|---|
| `content_length` | +0.00061 | +0.00037 | +0.00005 | +0.00390 |
| `extracted_full` | −0.00008 | +0.00242 | +0.00098 | +0.00542 |
| `extracted_sections` | **+0.00051** | +0.00278 | **+0.00208** | **+0.00634** |
| `bge-m3` full | +0.00085 | +0.00235 | +0.00065 | +0.00539 |

Adding sections turns the stub tier from slightly negative to slightly positive
and roughly doubles the mid tier, while still gaining most in the rich tier. It
beats the 1024-dim embedding in all three non-stub tiers.

### 14.5 Status

- **Recommended shipping configuration**: `extracted_sections`, 29 columns, all
  interpretable, uniform schema across all 3,144 counties, absence encoded as
  `False`. Beats both the incumbent scalar and the cut embedding on mean lift.
- **Allowed wording**: "targeted extraction from Wikipedia leads plus economy
  sections yields 2.7× the incumbent's mean cross-pillar lift and matches the cut
  1024-dim embedding's, at 29 interpretable columns and no model download —
  though its per-target advantage is concentrated rather than uniform, and the
  paired comparison against the incumbent is underpowered (p = 0.082 at power
  0.53; 57 targets would be needed for 80%)."
- **Forbidden wording**: "significantly beats the embedding" (p = 0.082, and the
  embedding wins on comparable target counts); "beats the embedding" without
  qualification — head to head the two are a rank tie at 13/28, p = 0.76, and the
  case for the typed block is cost and interpretability, not lift; "the typed
  block failed to reach significance" — the test was powered at 0.53, so it was
  never in a position to reach it; "across 28 targets" as a breadth claim — 20 of
  the 28 are QCEW sectors from five pillars total; "section expansion works" —
  this tests *targeted lexicon extraction from one named section*, not the
  concatenation §4 and §4.1 ruled out, which remains closed.
- **Reporting rule**: the per-pillar breakout in
  `outputs/source_a_representation_by_pillar.csv` is the primary result. Any
  handoff of the single aggregate must carry the basket composition (§14.2b)
  with it.
- **Reproduction**: `uv run --env-file .env python scripts/ingest_source_a.py`
  (refetch, ~16 min, also rewrites the text-features parquet so the extraction
  steps must follow), then `extract_source_a_features.py`,
  `extract_source_a_section_features.py`, `analyze_source_a_tiers.py`,
  `analyze_source_a_representation.py`. Seed 42 throughout. Power, effective n,
  and the pillar-blocked p-values come from `scripts/paired_power.py` and land in
  `source_a_representation_stats.json` under each variant's `effect` and
  `clustering` keys.

## 15. Should the Tiers Branch the Model? No — Tested Both Ways (2026-08-03)

**Context**: the tiers in §13.1 were used to decide *what to extract* and to break
out results, but every county runs through one schema and one model. The original
framing behind this experiment line was stronger: split counties into groups and
handle each group differently. That stronger version was never tested, so the
question stayed open — does letting the tiers change the model buy anything?

Two forms, both scored on the same 28 targets, same folds, same seed, against the
flat 29-column `extracted_sections` block:

| approach | width | mean R² lift |
|---|---|---|
| one model, one global coefficient per feature | 29 | **+0.00320** |
| one model, coefficients free to vary by tier | 120 | +0.00265 |
| four independent models, one per tier | 29 × 4 fits | **−0.01595** |

**Both branching forms lose, and the loss scales with how much branching there
is.** Tier-specific slopes cost 17% of the lift. Fully separate per-tier models
go negative — worse than dropping Source A entirely — because each trains on
roughly a quarter of the rows and overfits, with no shared penalty to restrain it.

The mechanism is ordinary bias-variance. Crossing 29 features with 4 tiers puts
120 columns against targets whose smallest sample is n ≈ 1,026, and the ridge
penalty large enough to control that width also over-shrinks the coefficients
that were doing the work in the flat model. Splitting the fit entirely removes
even the shared penalty's protection.

**Interpretation, stated carefully because it is easy to get backwards**: the tier
structure is real and it mattered — §13.1's 23× industry gradient is what
identified industry as the feature family worth building, and §14.1's per-tier
yield is what justified the section refetch. The tiers were the right
*diagnostic*. They are not the right *architecture*. Heterogeneity in the corpus
is best handled by features that are simply absent when a county has nothing to
say, not by fitting each group separately — sparsity already encodes the tier, and
the model has more data to learn from when it is not partitioned.

- **Allowed wording**: "content tiers guided which features to build and where to
  look for yield, but branching the model on them — whether by tier-specific
  coefficients or by separate per-tier fits — reduced out-of-fold lift, so the
  shipped configuration uses one uniform schema and one model."
- **Forbidden wording**: "the tiers do not matter" — they determined the feature
  set and the refetch decision; only the modeling split failed.
- **Status**: resolved. `sections_x_tier` is retained as a scored variant in
  `analyze_source_a_representation.py` so the negative result stays reproducible
  rather than becoming folklore.

## 16. Repairing the Cross-Pillar Sweeps After Source A's Expansion (2026-08-03)

Source A's block went from 1 column to 31 in §13–§14. Two cross-pillar analyses
consume that block, and both needed attention — one for a defect the expansion
introduced, one for a convention it made stale.

### 16.1 Defect: diagnostics were leaking into the matrix as predictors

`pillar_matrix._derive_pillar_columns` forwards every Source A column except the
two raw-text ones. That was correct when Source A shipped one scalar. After
§13–§14 it silently promoted two **diagnostics** into the predictor set:

- **`has_usda_echo`** exists to *detect* a contamination — it marks counties
  whose Wikipedia intro quotes USDA's own classification back ("considered a
  high-recreation retirement destination by the U.S. Department of
  Agriculture"). Source F's `distress_count` is built from those
  classifications. Letting the detector predict the thing it detects would have
  credited Source A for reciting a label it copied. **A detector for circularity
  must not itself be a predictor.**
- **`n_body_sections`** is a size proxy — r = 0.550 against log tax returns,
  above `content_length`'s 0.359 — carrying 2.4% of its block's lift (§14.3).

Both are now excluded in `pillar_matrix` via `SOURCE_A_DIAGNOSTIC_COLUMNS`, so
every consumer of the matrix gets the exclusion rather than each re-deriving it.
Source A's block is 29 columns. Source A's own representation harness is
unaffected — it already excluded both from scored variants, and re-running it
reproduces §14.2 exactly, which confirms the leak was confined to the sweeps.

### 16.2 Restatement: `has_metro_attachment` duplicates Source F

`analyze_pillar_matrix_signal.py` already ablates `F_INDUSTRY_COLUMNS` on the
grounds that USDA derives its typology from the same industry shares QCEW
measures, so agreement between them is bookkeeping rather than corroboration.
Source A's `has_metro_attachment` is the same phenomenon across a different
pair: it fires when an intro states the county belongs to a metropolitan or
micropolitan statistical area, and Source F's `metro_2023` is that same OMB
delineation. The ablation is now `RESTATEMENT_COLUMNS`, covering both.

### 16.3 What the repaired leave-one-pillar-out sweep shows

| | before (A = 1 column) | after (A = 29 columns) |
|---|---|---|
| mean lift | +0.0772 | +0.0739 |
| mean ablated lift | +0.0296 | +0.0234 |
| targets carrying signal | — | 22 of 29 |

**Source A's typed features do not help this sweep, and slightly dilute it.**
That is not a contradiction of §14, it is a different question. The
representation harness asks whether Source A adds anything over county size and
state; this sweep asks whether the *other five pillars together* know about a
target, and A's contribution sits inside a pool that already carries far more.
Adding 28 mostly-sparse columns to a 44-column design costs a little out-of-fold
efficiency and buys little.

Checked and ruled out: the ridge penalty grid. This sweep caps at 1000 while
Source A's own harness needed 1e6 for its 1024-dim variant, so the obvious
suspect was an alpha ceiling. Re-running three targets with the grid extended to
1e6 returned **identical** results to four decimal places — `RidgeCV` was already
selecting well below the cap. `RIDGE_ALPHAS` is therefore left alone rather than
changed on a hunch, which would have shifted every previously published number in
this sweep for no reason.

### 16.4 The pair sweep gets a second Source A feature

`analyze_pillar_pair_crossvalidation.py` represented Source A by
`content_length` alone. Its own stated convention gives Source B two features
"so the sweep can distinguish 'the pillar is weak' from 'the feature is weak'" —
and Source A now has exactly that ambiguity. `sec_n_industry_mentions` was added
as the feature family that carried §14's multivariate result.

**The result is worth stating plainly because it cuts against the new features.**
Bivariately, `content_length` remains Source A's strongest scalar:

| candidate | mean abs. r across sweep targets |
|---|---|
| `content_length` | **0.132** |
| `n_distinct_proper_nouns` | 0.116 |
| `founding_year` | 0.115 |
| `n_industry_mentions` | 0.047 |
| `sec_n_industry_mentions` | 0.038 |

The typed features win in the multivariate harness, where they combine and where
particular features match particular targets (tourism → Accommodation & Food
Services LQ). **One at a time, against one target at a time, they are weak.** Both
results are correct; they answer different questions, and this entry is what makes
that visible rather than leaving it to be discovered later.

A further check, run because it seemed likely and turned out not to hold: the
typed feature is **not** systematically more robust to the size control. Among
significant pairs it retains 51.0% of its raw |r| against `content_length`'s
49.9% — indistinguishable, and the median favours `content_length` (0.386 against
0.537). Its one distinctive pair is `gdp_velocity_pct`, where it keeps 84%
(r = 0.063 raw, 0.052 controlled, q = 0.004). That single pair should not be
generalized into a claim about the feature family.

Sweep totals moved from 41 feature pairs / 28 significant to **50 / 33**. The
Benjamini-Hochberg correction is applied across the whole sweep, so adding tests
tightens it for everyone — but **no pair changed its significance verdict**, which
is the robustness check that matters. `docs/PROJECT_GOAL.md` has been updated
(50 pairs, 17 of 33 significant correlations losing more than half their effect to
the size control).

### 16.5 Status

- **Allowed wording**: "Source A's typed features earn their place against county
  size and state, but add little to the leave-one-pillar-out sweep and are weak
  bivariately; `content_length` remains the pillar's strongest single scalar."
- **Forbidden wording**: "the typed features improve cross-pillar prediction" —
  they do not improve this sweep; "the typed features are more size-robust" —
  measured and not supported.
- **Reproduction**: `uv run python scripts/analyze_pillar_matrix_signal.py` and
  `uv run python scripts/analyze_pillar_pair_crossvalidation.py`, seed 42.

## 17. Marginal Value Against a Crowded Baseline — The Fusion-Relevant Test (2026-08-03)

**Context**: §14 and §16 disagreed about Source A's typed features. The
representation harness scored them at +0.0032 mean lift; the leave-one-pillar-out
sweep found they added nothing and slightly diluted it. Neither settles the
fusion decision, because neither matches what a downstream consumer sees:

- The harness baseline is **size + state only**. A model at Comcast will have
  every pillar E_macro publishes, not just county size and geography.
- The sweep never isolates Source A — it measures the other five pillars as a
  bloc, predicting a pillar's own features, where the target's sibling columns
  dominate by construction.

`analyze_source_a_marginal.py` runs the missing configuration:

    baseline = size + state + every pillar except Source A and the target's own
    variant  = baseline + Source A's block

Same 28 targets, seed 42, 5 folds, baseline and block fitted separately so no
shared penalty artifact. The target's own pillar is excluded for the same reason
the sweep excludes it: predicting one QCEW sector from nineteen others is a
within-pillar task no outside source can contribute to.

### 17.1 Result: Source A survives, at roughly a quarter of its headline value

Adding B–F to the baseline raises mean R² from 0.255 to 0.327 across 106 columns.

| variant | lift, thin baseline | lift, crowded baseline | retained | positive | Wilcoxon p |
|---|---|---|---|---|---|
| `content_length` | +0.00143 | +0.00052 | 37% | 20/28 | **0.014** |
| `extracted_sections` | +0.00373 | **+0.00104** | 28% | 19/28 | **0.013** |

**Three findings, in order of how much they should change behaviour:**

1. **Roughly 70% of Source A's measured value was information other pillars
   already carry.** The harness overstates the pillar's contribution by about
   3.5×. Any planning that used +0.0032 as Source A's expected contribution to a
   fused model should use **+0.0010**.
2. **What remains is statistically real.** Both variants stay significantly
   positive against a baseline that already contains every other pillar
   (p = 0.013, 0.014). The sweep's reading — that Source A's expansion adds
   nothing — was too strong. It measured the wrong thing.
3. **The typed block keeps roughly twice the marginal value of the scalar**
   (+0.00104 against +0.00052). That difference is not individually significant
   (17/28 targets, p = 0.295), so it supports "ship the typed block" on point
   estimate and cost, not on a demonstrated gap.

**How much power each of those three tests had**, which decides how much weight
each can carry:

| test | dz | power at n = 28 | targets for 80% | p |
|---|---|---|---|---|
| typed block vs zero, crowded baseline | 0.549 | **0.88** | 22 | **0.013** |
| `content_length` vs zero, crowded baseline | 0.339 | 0.54 | 56 | **0.014** |
| typed vs scalar, crowded baseline | 0.264 | 0.39 | 91 | 0.295 |

Two consequences that should govern how this section is cited:

- **"Source A carries marginal value over all five other pillars combined" is a
  well-powered, significant result** — power 0.88, p = 0.013, and the effective n
  equals the nominal n (ICC of the paired lifts within pillar is 0.000, design
  effect 1.00). This is the load-bearing finding of the whole experiment line and
  it is solid.
- **"The typed block beats the scalar" is powered at 0.39 and would need 91
  targets.** It will not be settled in this repo by adding targets, and it is not
  what the shipping decision rests on. The typed block ships on cost and
  interpretability.

The pillar-blocked sensitivity check — one vote per pillar, five observations —
gives p = 0.092 for the typed block and p = 0.166 for the scalar. Neither clears
0.05 there, but a five-observation test is underpowered by construction and this
is reported as a caution about breadth claims, not as a competing verdict.

### 17.2 Where it survives, and why that is the coherence check

Source A's surviving contribution is concentrated exactly where its features name
something no federal statistic encodes:

| target | crowded lift | plausible mechanism |
|---|---|---|
| Accommodation & Food Services LQ | +0.0058 | tourism / resort / ski / casino mentions |
| Educational Services LQ | +0.0051 | `has_university` |
| Transportation & Warehousing LQ | +0.0042 | river / interstate / port |
| demographic distress count | +0.0040 | mixed |
| outbound partner concentration | +0.0035 | transport and logistics mentions |
| GDP velocity (normalized) | +0.0025 | mixed |

And it is fully absorbed where another pillar measures the same quantity
directly — Source E's capital-to-wage ratio collapses from +0.0079 to +0.0000,
Information LQ turns negative, Manufacturing LQ retains 11%.

**This is the strongest evidence in the whole experiment line that the extraction
is doing something real.** A feature set that survived arbitrarily would look like
noise; this one survives on the three targets whose semantics match its
lexicons and dies on the ones another agency already measures. The largest
absorbed loss (capital-to-wage) and the largest surviving gain (accommodation)
are both predicted by what the columns actually contain.

Per-pillar, the typed block's advantage over `content_length` under the crowded
baseline concentrates in F (+0.0032), C (+0.0013) and D (+0.0012); it is
negligible for B (+0.0002) and E (+0.00001).

### 17.2a Retention varies by two orders of magnitude, so +0.0010 is a property of the basket

| target pillar | targets | thin lift | crowded lift | retained |
|---|---|---|---|---|
| C (velocities) | 3 | +0.00170 | +0.00118 | **69%** |
| D (freight) | 3 | +0.00212 | +0.00133 | 63% |
| B (QCEW) | 20 | +0.00317 | +0.00087 | 28% |
| F (typology) | 1 | +0.02172 | +0.00401 | 18% |
| E (capital-to-wage) | 1 | +0.00786 | +0.00002 | **0.2%** |

Retention is highest where the other pillars know least — Source C's velocity
series are near-orthogonal to county size and to the rest of the matrix — and
collapses where a federal agency measures the same construct directly. That
gradient is §17.2's coherence argument stated quantitatively.

It also means the headline is basket-dependent. **The basket is 71% QCEW, the
worst-retaining large block**, so the single published number is as much a
property of the target mix as of Source A. A downstream model that cares about
dynamics and velocity should expect considerably more than +0.0010; one that
cares about industry composition, considerably less. Full table in
`outputs/source_a_marginal_by_pillar.csv`; **it is the primary result and the
aggregate must not travel without it.**

### 17.3 What this does not settle

Every target here is still another pillar's feature. Downstream targets at
Comcast are external — churn, propensity, whatever the consuming model predicts.
A source can be redundant for predicting `lq_emp_72` and still carry independent
information about an external outcome, or vice versa. **This is the closest proxy
available in-repo, not the answer.** The only thing that settles it is a
downstream label, which this project does not have and is not scoped to obtain.

There is a structural consequence of that which is easy to miss. Because every
target is a pillar feature, **a source is penalized precisely for agreeing with
the pillars it will ship alongside.** For assembling a non-redundant feature
store that is arguably the correct penalty. For predicting an external outcome it
is not: Source A and Source F can be redundant with each other and both
predictive of churn. Relatedly, `has_metro_attachment` is ablated in the sweeps
as a restatement of Source F's `metro_2023` (§16.2) — justified for
pillar-versus-pillar work, unjustified against an external target, where it would
be a legitimate free feature. Any future external-target harness must revisit
both decisions rather than inherit them.

- **Allowed wording**: "against a baseline that already contains every other
  pillar, Source A's typed block retains a small but statistically real
  contribution (+0.0010 mean R² lift, p = 0.013, power 0.88), roughly twice the
  shipped scalar's, concentrated on targets whose semantics its lexicons match —
  with retention ranging from 69% on Source C's velocities to 0.2% on Source E's
  capital ratio."
- **Forbidden wording**: "Source A contributes +0.0032 to a fused model" — that
  is the thin-baseline figure and it overstates by ~3.5×; "the typed block
  significantly beats `content_length` in fusion" — p = 0.295 at power 0.39, a
  test that would need 91 targets; quoting +0.0010 as Source A's expected
  contribution without stating that the basket is 71% QCEW.
- **Verdict for the fusion step**: **ship the typed block.** It is positive
  against the hardest in-repo baseline on the best-powered test in this file, its
  surviving contribution is semantically coherent rather than diffuse, and it
  costs one regex pass. Plan around +0.0010, not +0.0032, and adjust for which
  pillars the consuming model's targets resemble.
- **Reproduction**: `uv run python scripts/analyze_source_a_marginal.py`, seed 42.
  Power and clustering diagnostics come from `scripts/paired_power.py` and land
  in `source_a_marginal_stats.json`; the per-pillar breakout is written to
  `outputs/source_a_marginal_by_pillar.csv`.
