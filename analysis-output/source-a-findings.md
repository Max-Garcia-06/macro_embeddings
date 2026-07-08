---
type: results-report
date: 2026-07-03
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

Source A's Wikipedia-intro-text county embeddings (`BAAI/bge-m3`, 1024-dim)
carry a real but small geographic signal. A Mantel permutation test finds a
weak, statistically significant negative correlation between geographic
distance and embedding similarity (r ≈ -0.11, p = 0.002). Neither PCA's
first component (4.8% of variance) nor K-means clustering (k=2, silhouette
≤0.035) finds strong or cleanly interpretable structure beyond that weak
geographic effect — **but** the k=2 split is highly reproducible across
random seeds (ARI≈0.996) and its ~11% geographic-coherence deviation from the
corpus mean is statistically real (permutation p≤0.001 both directions), not
sampling noise. A separate prototype found that embedding more article text
(additional sections, or an Economy-only section) does not improve
differentiation between counties. **Coverage is now complete**: the dataset
covers all 3,144 US counties/county-equivalents (the Virginia
independent-city gap of 37 cities and a second batch of 19 Census-Gazetteer-
vs-Wikipedia-title mismatches have both been backfilled; see §2). Checked
against `E_macro_extendedProposal.pdf`'s justification for Source A (§10):
the proposal expects intro text to carry distinctive economic-transition
narrative; unsupervised analysis instead finds generic formation/demographic
boilerplate with only the weak geographic echo above — that specific claim
is unsupported, though correlation against real economic variables
(Source E/B) is still untested.

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
  `source_a_clusters.html`/`source_a_cluster_summary.csv`) were re-run on
  2026-07-03 against the full 3,144-row parquet and are now mutually
  consistent — no artifact reflects a stale snapshot.
- **Units of analysis**: individual county. PCA explained-variance ratio
  computed on n=3,144 (all matched to centroids). Clustering/Mantel run on
  n=2,849 (after dropping 294 "stub" counties with <100 characters of real
  content and 1 non-50-state entry). Pairwise similarity/distance is
  treated as non-independent — inference is via Mantel permutation, not a
  naive pairwise correlation p-value.

## 3. Main Findings

1. **Geography ↔ similarity (Mantel test)**: weak negative association,
   robust across every coverage backfill so far.

   | | n=2,793 (original) | n=2,830 (+Virginia) | n=2,849 (full coverage) |
   |---|---|---|---|
   | Mantel r | -0.0937 | -0.0898 | -0.1055 |
   | p-value | 0.0020 | 0.0020 | 0.0020 |

   Nearby counties are very slightly more textually similar than far-apart
   ones; geography is not a dominant driver, and this conclusion is stable
   across all three coverage snapshots (r has moved by ~0.01-0.02 each time,
   never changing sign or significance).

2. **PCA**: PC1 explains **4.8% of total variance** on the full 3,144-county
   corpus (`visualize_source_a.py`, rerun 2026-07-03; essentially unchanged
   from 4.9% at n=3,088 and 4.8% at n=3,125). A manual read of the 3
   highest- and 3 lowest-loading counties — re-verified on the full,
   final corpus and **identical to every prior snapshot**
   (Elliott County, KY / Wise County, VA / Kent County, TX high;
   Clay County, IN / Miami County, IN / Floyd County, GA low) — found no
   shared theme; both tails read as generic formation/population
   boilerplate. **No thematic label is assigned to PC1.**

3. **Clustering (K-means, silhouette-selected k)**: k=2 selected across
   k=2..12, but every k's silhouette score is low (max ≈0.0345 on the full
   corpus, consistent with 0.034-0.036 across every earlier snapshot, well
   under the ~0.25 threshold usually associated with real structure) — the
   "best" clustering is a weak fit, not a good one. Full coverage: cluster
   sizes 1,312 / 1,537; cluster 0 is ~11% *looser* than the corpus-wide mean
   pairwise distance (1,618 km vs. 1,455 km), cluster 1 is ~11% *tighter*
   (1,292 km) — opposite directions, not a coherent regional split.

4. **Notable outlier pairs**: the five most similar pairs among all
   top-quartile-distance pairs (≥75th percentile, ≥~1,940 km apart; full
   ranked list in `source_a_similarity_pairs.csv`):

   | Rank | Pair | Similarity | Distance (km) | Why similar despite distance |
   |---|---|---|---|---|
   | 1 | Lincoln County, KS ↔ Lincoln County, OR | 0.845 | 2,207 | Shared name *and* shared namesake (Abraham Lincoln) — both articles use near-identical templated phrasing ("named for/after Abraham Lincoln, 16th president of the United States"). |
   | 2 | Montgomery County, AL ↔ Stutsman County, ND | 0.833 | 1,967 | No name in common; both follow the same "county seat is X ... county comprises/included in the X metropolitan/micropolitan area" infobox-style sentence almost verbatim. |
   | 3 | Stutsman County, ND ↔ Williamsburg County, SC | 0.833 | 2,194 | Both are short, minimally-elaborated articles (population + county seat + founding year, no distinguishing detail) — they converge because there is little content to differentiate them, not because of shared subject matter. |
   | 4 | Franklin County, ME ↔ Franklin County, NE | 0.826 | 2,384 | Same mechanism as #1: shared name and shared namesake (Benjamin Franklin) reproduces the same "named for Benjamin Franklin" sentence in both articles. |
   | 5 | Stutsman County, ND ↔ Providence County, RI | 0.826 | 2,234 | Same mechanism as #3 — Stutsman's terse, template-only intro sits close to the corpus's generic-boilerplate centroid, so it reads as "similar" to several otherwise-unrelated counties. |

   Two distinct, both surface-level, mechanisms produce these matches:
   (a) **shared eponym** — two counties named after the same historical
   figure reuse the same "named for X" sentence almost verbatim (#1, #4);
   (b) **generic-boilerplate convergence** — a short, unelaborated article
   (Stutsman County, ND appears in 3 of the top 5) sits close to the
   corpus's templated-boilerplate centroid and reads as similar to any
   other equally generic article, regardless of subject (#2, #3, #5).
   Neither mechanism reflects real economic or geographic kinship — both
   are consistent with finding #1 (geography matters a little on average;
   individual pairs can still be highly similar for reasons unrelated to
   distance) and reinforce the report's broader theme that the corpus is
   dominated by generic formation/demographic boilerplate rather than
   distinctive content (§10).

5. **Cross-seed K-means stability**: re-running K-means at k=2 across 5
   seeds (42, 7, 123, 2024, 99) on the full corpus gives silhouette
   mean=0.0343, std=0.0008, and pairwise Adjusted Rand Index across all 10
   seed pairs of mean=0.9955 (range 0.9916–1.0000). **The k=2 partition is
   highly stable and reproducible** — it is not an artifact of one random
   seed.

6. **Cluster-coherence permutation test**: for the k=2 clustering (999
   permutations of county-to-cluster labels, preserving cluster sizes),
   both clusters' ~11% deviation from the corpus mean distance is more
   extreme than every one of the 999 random relabelings produced (p≤0.001,
   both directions). **The geographic-coherence split is statistically
   real, not chance** — but this does not upgrade its practical size:
   silhouette stays ≤0.035, so it is a real, reproducible, but small
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

## 5. Figure-by-Figure Interpretation

Figures live in `analysis-output/figures/` (all regenerated 2026-07-03
against the full 3,144-county corpus / 2,849-county clustering set — fully
consistent with the numbers in §3).

**figure-01-similarity-vs-distance.png** — Hexbin of similarity vs. distance
across all pairs, trend line, Mantel r/p annotated. The trend line's shallow
slope visually confirms "weak," not "strong." Rules out over-reading
`source_a_similarity_pairs.csv`'s outlier table (extreme cases by
construction) as evidence of a strong geography effect.

**figure-02-pc1-distribution.png** — Histogram of PC1 values with the 3
highest/lowest counties labeled. Unimodal, no bimodality suggesting a
natural two-group split. Pre-empts assigning a semantic label to PC1 — the
shape gives no hint of one, and a manual text check of the extremes found
none either.

**figure-03-cluster-coherence.png** — Bar chart of mean intra-cluster
distance per cluster vs. the corpus-wide baseline. The two clusters deviate
from baseline in opposite directions by similar magnitudes (~11%). Blocks
the claim that "embedding clusters correspond to US regions" — combined
with the uniformly low silhouette scores, K-means does not find strong
structure of any kind here (later confirmed statistically real but small by
the §3.6 permutation test).

## 6. Claim Candidates

- **Claim**: Geographic distance between US counties has a weak but
  statistically detectable negative association with the cosine similarity
  of their Wikipedia intro-text embeddings.
  - Evidence: Mantel test, r=-0.1055, p=0.0020, n=2,849, 499 permutations,
    seed=42 (full coverage, 2026-07-03; prior snapshots: r=-0.0937 at
    n=2,793, r=-0.0898 at n=2,830, both p=0.0020).
  - Allowed wording: "a weak, statistically significant negative
    association"; "geography explains a small amount of the variation in
    textual similarity."
  - Forbidden wording: "geographic distance strongly predicts / determines /
    drives embedding similarity"; any wording implying a large or dominant
    effect.
  - Status: **confirmed robust across every coverage backfill, including
    full coverage** — resolved.

- **Claim**: K-means clustering (k selected by silhouette score) finds a
  highly reproducible k=2 split whose geographic coherence is statistically
  real but small in magnitude.
  - Evidence: silhouette ≤0.0353 across 5 seeds; pairwise ARI mean=0.9955;
    permutation test p≤0.001 (both clusters, 999 permutations); all on the
    full 2,849-county set.
  - Allowed wording: "the two-cluster split is stable across random seeds
    and its geographic-coherence difference from chance is statistically
    significant, but the effect size (~11% deviation, silhouette ≤0.035)
    remains small — this is not evidence of clean regional structure."
  - Forbidden wording: "clusters correspond to distinct US regions";
    "embedding clusters map onto geography"; using "statistically
    significant" alone without the effect-size qualifier.
  - Status: resolved — reconfirmed on the full-coverage corpus.

- **Claim**: PC1 of the Source A embeddings explains a small share of
  total embedding variance (~4.8%); no thematic label should be attached
  to it.
  - Evidence: `pca.explained_variance_ratio_[0]=0.0478`, n=3,144, full
    coverage, all matched to centroids (2026-07-03 rerun of
    `visualize_source_a.py`). Manual 6-county tail inspection re-verified on
    the full-coverage extremes — identical county set to every prior
    snapshot (Elliott County KY, Wise County VA, Kent County TX high;
    Clay County IN, Miami County IN, Floyd County GA low) — found no shared
    theme.
  - Allowed wording: "PC1 explains a small share of the embedding's variance
    (~4.8%, stable from n=3,088 through full n=3,144 coverage)"; naming the
    specific extreme counties is fine.
  - Forbidden wording: any semantic label for what PC1 "represents."
  - Status: resolved — both the variance-ratio number and the manual
    thematic tail-check are confirmed on the final, full-coverage corpus.

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
   `source_a_map.html`, and the similarity/clusters HTML artifacts were all
   regenerated on 2026-07-03 against the full 3,144-row parquet and are
   mutually consistent (see §2, §9).
3. **PC1 has no established meaning** — 4.8% of variance, stable from
   n=3,088 through the final n=3,144 (full coverage); no thematic pattern
   found in a manual check of the extremes, now re-verified on the final
   corpus (§3.2).
4. **No alternative clustering method tried** (e.g. hierarchical, DBSCAN) as
   a robustness check on the K-means finding.

## 8. Next Actions

1. **Do not build downstream conclusions on PC1's semantic meaning or on the
   two K-means clusters representing real regions** — both remain
   weak/unsupported.
2. **Coverage and artifact consistency are both done** — no further
   backfill or regeneration work is needed for Source A's current scope.
3. **Do not pursue further Wikipedia section-selection variants** (§4,
   closed) — if stronger differentiation is needed, the next lever is a
   non-Wikipedia source (Sources B–F) or targeted boilerplate-stripping.
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
- EDA / analysis: `visualize_source_a.py` (PCA), `analyze_source_a_similarity.py`
  (pairwise similarity/distance), `analyze_source_a_clusters.py` (K-means +
  Mantel test), `analyze_source_a_cluster_stability.py` (cross-seed
  stability + cluster-coherence permutation test). All rerun 2026-07-03
  against the full 3,144-row parquet.
- Insights synthesis: `generate_source_a_insights.py` (produced the current
  `stats.json` and the three saved figures, rerun 2026-07-03).
- Persisted statistics: `analysis-output/stats.json` (full-coverage
  snapshot, n=3,144 / n=2,849).
- Figures: `analysis-output/figures/figure-01-similarity-vs-distance.png`,
  `figure-02-pc1-distribution.png`, `figure-03-cluster-coherence.png`,
  `figures/source-a-numeric-summary.md` (all full-coverage).
- Companion artifacts: `source_a_map.html`, `source_a_similarity.html` /
  `source_a_similarity_pairs.csv`, `source_a_clusters.html` /
  `source_a_cluster_summary.csv` — all refreshed 2026-07-03, mutually
  consistent with `stats.json`.
- Notebook: `analysis-output/source_a_key_findings.ipynb` — presentation
  notebook covering the findings in §3-§6; loads the artifacts above rather
  than recomputing them, so re-run it after any future backfill to keep it
  in sync.
- Ingestion log: `ingest_run.log` (untracked).
- Reproduction: `uv run --env-file .env backfill_virginia_cities.py`, then
  `uv run --env-file .env backfill_remaining_19.py`, then
  `uv run generate_source_a_insights.py`, `uv run visualize_source_a.py`,
  `uv run analyze_source_a_similarity.py`,
  `uv run analyze_source_a_clusters.py`, and
  `uv run analyze_source_a_cluster_stability.py`. All seeded
  (`RANDOM_SEED=42` throughout; stability seeds 42/7/123/2024/99).

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
  clustering. It doesn't: PC1 explains only 4.8% of variance with no
  thematic label (§3.2), and K-means finds no cleanly separated structure
  (silhouette ≤0.035, §3.3) beyond the weak geographic effect.
- The manual tail-read of PC1's extreme counties (§3.2) found "generic
  formation/population boilerplate" — not economic-transition narrative —
  on both ends.
- §4's negative result reinforces this: embedding more article text made
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
- **Claim**: Source A's raw intro-text embeddings do not show the
  distinctive economic-transition signal the E_macro proposal's Source A
  justification assumes; the corpus is dominated by generic
  county-formation boilerplate.
  - Evidence: PC1 4.8% variance, no theme (§3.2); K-means silhouette
    ≤0.035 (§3.3); section-expansion negative result (§4).
  - Allowed wording: "unsupervised analysis finds no evidence of the
    distinctive economic-narrative signal the proposal's Source A
    justification describes; the corpus reads as generic boilerplate with
    a weak geographic echo."
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
(PC1 at ~4.8% variance, silhouette ≤0.035, section-expansion making
counties more similar rather than less). Frozen decoder-LLM hidden states
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

## 12. Round 4 — Targeted Boilerplate Stripping (2026-07-07)

**Question**: §10-11 diagnosed the corpus as dominated by generic
formation/demographic boilerplate rather than distinctive
economic-transition narrative, and §3.4 identified two concrete
surface-level mechanisms driving false high-similarity pairs (shared-eponym
templated sentences; short, template-only articles converging on a generic
centroid). Could targeted stripping of that boilerplate measurably reduce
those mechanisms without destroying the corpus's one confirmed real
signal (the weak negative Mantel correlation)?

**Method**: two candidate cleaning variants, both layered on top of the
existing Task-1/2 `strip_self_reference` → `strip_boilerplate_phrasing`
pipeline, then re-embedded with the same `BAAI/bge-m3` model from the
stored `raw_intro_text` (no Wikimedia API calls):
- **v2**: three targeted regex families for the exact mechanisms behind
  §3.4's top-5 far-but-similar pairs — eponym clauses ("named for/after
  X"), metro/micropolitan-area sentences, and formation connectives.
- **v3**: v2's regex families plus a corpus-frequency sentence filter that
  drops any sentence whose number/proper-noun-masked "template" appears in
  ≥5% of counties.

**Pre-registered Decision Gate** (plan top-level, applied without
modification): a candidate variant passes iff, on the fixed analysis
county set:
1. `tracked_pair_mean` (mean cosine similarity of the 5 tracked boilerplate
   pairs from §3.4) drops by **≥ 0.03** vs. baseline;
2. `pairwise_similarity_std` is **≥ baseline**;
3. Mantel `r < 0` with `p < 0.05`.

If both variants pass, adopt the one with the lower `tracked_pair_mean`
(tiebreak: higher `pairwise_similarity_std`). If neither passes, keep the
baseline and record the negative result.

**Evaluation harness**: `evaluate_source_a_variants.py`, run once over all
three parquets (baseline, v2, v3) against a single fixed analysis set —
**2,275 counties**, after the harness's own `drop_stub_counties` filter
(now running against the *stronger* post-Task-1/2/3-stripped text, hence
lower than the 2,849 cited in §2/§3 — an expected consequence of more
aggressive de-boilerplating creating more near-empty stubs, not a
regression) plus the 50-states filter and centroid match. Full output:
`analysis-output/variant-eval.json`.

**Results** (all seven `REPORT_METRICS`, n=2,275 for every row):

| Metric | Baseline | v2 | v3 |
|---|---|---|---|
| tracked_pair_mean | 0.82926 | 0.79366 | 0.75410 |
| pairwise_similarity_mean | 0.55147 | 0.55236 | 0.51554 |
| pairwise_similarity_std | 0.06251 | 0.06684 | 0.08830 |
| mantel_r | -0.12171 | -0.09215 | -0.04999 |
| mantel_p | 0.002 | 0.002 | 0.002 |
| silhouette_k2 | 0.02889 | 0.02827 | 0.03797 |
| n_counties | 2,275 | 2,275 | 2,275 |

The Decision Gate's Mantel criterion only requires `r < 0` with `p < 0.05` —
it does not require preserving the correlation's magnitude. v3's `mantel_r`
(−0.04999) is about 41% of baseline's magnitude (−0.12171), a roughly 59%
reduction in the strength of the corpus's one confirmed geography↔similarity
signal. Both variants clear the gate's floor, but this magnitude loss is a
real cost of adoption, not merely "preservation" of the signal.

**Gate check**:

| Candidate | tracked_pair_mean drop | ≥0.03? | std ≥ baseline? | mantel r<0, p<0.05? | Passes? |
|---|---|---|---|---|---|
| v2 | 0.82926 − 0.79366 = 0.03560 | yes | yes (0.06684 ≥ 0.06251) | yes (−0.09215, p=0.002) | **yes** |
| v3 | 0.82926 − 0.75410 = 0.07516 | yes | yes (0.08830 ≥ 0.06251) | yes (−0.04999, p=0.002) | **yes** |

Both variants pass. Tiebreak (lower `tracked_pair_mean` wins): v3
(0.75410) < v2 (0.79366) → **v3 adopted**.

**v3's aggregate std improvement masks a worse single worst-case outlier.**
Per each variant's `top_far_similar_pairs` in `variant-eval.json`,
baseline's most extreme far-apart-but-similar pair is Montgomery County,
Alabama | Stutsman County, North Dakota at 0.8327 (~1967 km apart) — the
same pair already tracked above. Under v3, the single worst far-similar
pair is Allamakee County, Iowa | Clatsop County, Oregon at **0.9607**
(~2557 km apart), exceeding baseline's worst case by a wide margin even
though the aggregate `pairwise_similarity_std` improved (0.0625→0.0883).
This is the expected flip side of aggressive stripping: very short
residual text after stripping converges harder toward a generic centroid,
producing occasional near-duplicate pairs that are both farther apart
geographically and more similar in embedding space than anything in the
baseline.

**Tracked-pair count is 4, not 5.** `TRACKED_BOILERPLATE_PAIRS` in
`evaluate_source_a_variants.py` enumerates all 5 pairs from §3.4's outlier
table, including the #1-ranked pair, "Lincoln County, Kansas | Lincoln
County, Oregon". That pair is absent from `tracked_pair_similarity` in
**all three** parquet results, baseline included — one of the two counties
fell out of the fixed 2,275-county analysis set because it failed the
(now-stronger) stub-content filter, not because of an evaluation-harness
bug. The remaining 4 tracked pairs, baseline → v3:

| Pair | Baseline | v3 | Δ |
|---|---|---|---|
| Montgomery County, AL ↔ Stutsman County, ND | 0.8327 | 0.8308 | −0.0019 |
| Stutsman County, ND ↔ Williamsburg County, SC | 0.8325 | 0.7500 | −0.0826 |
| Franklin County, ME ↔ Franklin County, NE | 0.8263 | 0.6029 | −0.2233 |
| Stutsman County, ND ↔ Providence County, RI | 0.8255 | 0.8327 | +0.0071 |

Three of the four mechanisms tracked in §3.4 show sizable drops under v3 —
most notably the shared-eponym "Franklin/Franklin" pair (−0.22) and the
generic-boilerplate "Stutsman/Williamsburg" pair (−0.08), both directly
targeted by the v2/v3 regex families and frequency filter. The
Montgomery/Stutsman pair barely moved (−0.002), and Stutsman/Providence
*increased* slightly (+0.007) — consistent with Stutsman County's article
being short enough that stripping its templated sentences removes a large
fraction of its remaining content, pushing it toward the corpus's
post-stripping generic centroid from a different direction. This is the
same "short, minimally-elaborated article" failure mode described in
§3.4(b); it is not resolved by this round's variants and should be flagged
as an open item rather than absorbed into the aggregate `tracked_pair_mean`
drop.

**Decision**: **v3 adopted** as the new baseline
(`source_a_embeddings.parquet` overwritten with the former
`source_a_embeddings_v3.parquet`; git history retains the prior baseline).
All §9 artifacts were regenerated against the new baseline:

| Script | Regenerated artifact(s) | Analysis n it reports | Exit |
|---|---|---|---|
| `visualize_source_a.py` | `source_a_map.html`, PC1 stats | 3,144 matched to centroids; PC1 explains **7.0%** of variance (up from 4.8% pre-stripping) | 0 |
| `analyze_source_a_similarity.py` | `source_a_similarity.html`, `source_a_similarity_pairs.csv` | drops 868 stub counties (own pipeline, no 50-states filter applied) → **2,276** | 0 |
| `analyze_source_a_clusters.py` | `source_a_clusters.html`, `source_a_cluster_summary.csv` | drops 868 stub + 1 non-50-state → **2,275** (matches the harness); silhouette-selected **k=3** (previously k=2), silhouette 0.0408; Mantel r=−0.0500, p=0.002 | 0 |
| `analyze_source_a_cluster_stability.py` | stdout only (no persisted artifact) | 2,275; k=3 cross-seed silhouette mean=0.0387, std=0.0023; pairwise ARI mean=0.9924 (range 0.9870–0.9987, 5 seeds) | 0 |
| `generate_source_a_insights.py` | `analysis-output/stats.json`, `analysis-output/figures/*` | 2,275 for clustering/Mantel; 3,144 for PC1 | 0 |
| `nbconvert --execute` on `source_a_key_findings.ipynb` | `analysis-output/source_a_key_findings.ipynb` | re-executed against the new baseline; no separate n computed | 0 |

Two things worth calling out explicitly rather than silently absorbing:
- **The 2,275 vs. 2,276 counts are not the same n.**
  `analyze_source_a_similarity.py` never applies the 50-states filter, so
  it retains one extra county (2,276) relative to
  `analyze_source_a_clusters.py`, `analyze_source_a_cluster_stability.py`,
  and `generate_source_a_insights.py`, which all apply it and land on
  2,275 — matching the harness's independently-implemented
  `build_analysis_frame`. That the two separately-implemented pipelines
  (harness vs. repo analysis scripts) agree on 2,275 once both filters are
  applied was verified against actual script output, not assumed.
- **K-means now selects k=3, not k=2.** With v3 embeddings, silhouette
  peaks at k=3 (0.0408) rather than k=2 (0.0405 — close, but k=3 now
  wins). This changes §3.3/§3.5/§3.6's k=2-specific narrative (cluster
  sizes, ARI, permutation-test structure) for the new baseline; those
  sections are not rewritten here (out of scope for this round), but any
  future work citing "the k=2 split" should re-check against the current
  baseline's k=3 selection.

**Claim candidates**:
- **Claim**: targeted boilerplate stripping (v3: eponym/metro-area/
  formation regex families + ≥5%-frequency sentence-template filter)
  measurably reduced the surface-level false-similarity mechanisms
  documented in §3.4, while preserving the corpus's negative
  geography↔similarity signal.
  - Evidence: `tracked_pair_mean` drop of 0.075 (0.82926→0.75410, gate
    threshold 0.03), `pairwise_similarity_std` increase (0.0625→0.0883),
    Mantel r remains negative and significant (−0.050, p=0.002, n=2,275)
    — all measured against the pre-registered gate in
    `analysis-output/variant-eval.json`.
  - Allowed wording: "de-boilerplating (v3) reduced tracked
    false-similarity-pair similarity by ~0.075 and increased pairwise
    similarity dispersion, while the geography↔similarity Mantel signal
    remained negative and significant"; "3 of the 4 evaluable tracked
    pairs (the eponym and generic-boilerplate mechanisms) showed the
    expected reduction."
  - Forbidden wording: any claim that de-boilerplating validates Source A
    for the E_macro proposal's economic-narrative role (§10) — that still
    requires the untested Source E/B correlation check; also forbidden:
    describing this as fixing "all" false-similarity mechanisms — the
    Stutsman/Providence pair moved in the wrong direction, and no claim
    should paper over that.
  - Status: resolved for this round's specific gate criteria; the
    proposal-alignment question from §10 remains open.

## 13. Round 5 — Rural-County Boilerplate-Filter Protection (2026-07-08)

**Question**: §12 documented that v3's corpus-frequency filter, while
passing its adoption gate, introduced a worse-than-baseline worst-case
outlier (0.9607 vs. baseline's 0.8327) by over-stripping short/rural
articles — most visibly Stutsman County, ND's tracked pairs, which barely
moved or moved the wrong direction. Could an outcome-gated version of the
same filter fix this without giving up v3's tracked-pair and dispersion
gains?

**Method**: a `v4` variant, implemented in `reembed_source_a.py`'s
`build_embedding_texts`. For each county, v3's regex + corpus-frequency
sentence filter is applied as before, but the filtered result is only
used if it still clears the existing `MIN_CONTENT_LENGTH` (100-char) stub-
content bar (imported from `analyze_source_a_similarity.py`); otherwise
that county keeps its unfiltered v2 (regex-only) text. Evaluated via
`evaluate_source_a_variants.py` against a freshly reconstructed `raw`
(no-cleaning) parquet and the current v3 baseline, n=2,275 counties for
every row.

**Methodology caveat — the `raw` parquet here is not this experiment
line's historical baseline, and should not be treated as one.** The `raw`
parquet built for this round embeds fully unprocessed `raw_intro_text`:
literal, unstripped Wikipedia intro text, including each county's own name
and state name spelled out, plus leading breadcrumb/hatnote content. Its
much stronger Mantel correlation (r=−0.2892, roughly 5.8× the magnitude of
v3's −0.0500) is very likely an artifact of literal state-name token
overlap between same-state counties — `bge-m3` will score two texts that
both contain the literal token "Texas" as more similar on that basis alone
— not evidence of a stronger genuine narrative signal. This is exactly the
failure mode `strip_self_reference` (in `text_cleaning.py`) was built to
prevent in the first place (§1). The actual historical baseline used by
every prior gate in this experiment line (§4, §12) already had self-
reference and boilerplate-clause stripping applied at ingestion time; it
was never literal raw HTML/text. Any future reuse of a zero-cleaning
reconstruction like this round's `raw` parquet as a stand-in for "the
original baseline" would silently launder this token-overlap artifact into
whatever comparison uses it — this section exists to flag that risk before
it happens, not after.

**Results** (all rows n=2,275):

| Metric | raw (zero-cleaning) | v3 (baseline) | v4 (candidate) |
|---|---|---|---|
| tracked_pair_mean | 0.6067 | 0.7541 | 0.7541 (identical to v3) |
| pairwise_similarity_std | 0.0599 | 0.0883 | 0.0834 |
| mantel_r | −0.2892 | −0.0500 | −0.0654 |
| mantel_p | 0.002 | 0.002 | 0.002 |
| worst top_far_similar_pairs entry | 0.7968 (Grant County, OK ↔ Grant County, OR) | 0.9607 (Allamakee, IA ↔ Clatsop, OR) | 0.8727 (Hendry, FL ↔ Finney, KS) |
| Stutsman ND ↔ Providence RI tracked pair | 0.5283 | 0.8327 | 0.8327 (identical to v3) |

Per the methodology caveat above, `raw` is excluded from the gate check
below — it is a useful sanity check on the token-overlap risk, but it is
not the baseline the pre-registered gate was written against. The gate
check instead compares v4 to the real historical baseline already
published in §12 (`tracked_pair_mean`=0.82926, `pairwise_similarity_std`=
0.06251, `mantel_r`=−0.12171, Stutsman/Providence=0.8255, worst-case pair
Montgomery AL ↔ Stutsman ND=0.8327) — none of those figures are
recomputed here.

**Gate check** (v4 vs. the real historical baseline from §12):

| # | Criterion | v4 value | Real baseline reference | Result |
|---|---|---|---|---|
| 1 | tracked_pair_mean drop ≥0.03 vs. real baseline | 0.7541 | 0.82926 − 0.03 = 0.79926 | PASS (0.7541 ≤ 0.79926; drop = 0.07516) |
| 2 | pairwise_similarity_std ≥ real baseline | 0.0834 | 0.06251 | PASS |
| 3 | mantel r<0, p<0.05 | −0.0654, p=0.002 | — | PASS |
| 4 | v4's worst top-far-similar pair ≤ v3's worst (0.9607) | 0.8727 | 0.9607 | PASS — this was the round's primary goal, and it worked: the dramatic tail-outlier regression from §12 is fixed |
| 5 | Stutsman/Providence must not increase vs. real baseline (0.8255) | 0.8327 | 0.8255 | FAIL — identical to v3, no improvement; v4's outcome gate (keyed on `MIN_CONTENT_LENGTH`=100) never triggers for this specific pair because neither county's filtered text drops below that floor |

4 of 5 criteria pass; criterion 5 fails. Per this experiment line's
pre-registered rule ("if any criterion fails → reject"), the gate is not
satisfied.

**Decision**: **v4 rejected** — baseline remains v3
(`source_a_embeddings.parquet` unchanged). The round's primary motivating
goal, the worst-case tail outlier flagged in §12, was fixed
(0.9607→0.8727), and that gain is real. But v4 does not clear the strict
pre-registered gate: the specific Stutsman/Providence pair that motivated
this round in the first place did not move, because that pair's filtered
text never drops below the 100-char outcome-gate floor in the first place
— the mechanism simply doesn't engage for it. Per gate discipline, a
partial fix is not an adopted fix; §12's rural/short-county over-stripping
regression remains an open, undismissed limitation of the current (v3)
baseline.

**Next Actions**:
1. The outcome gate's `MIN_CONTENT_LENGTH`=100 floor was reused from the
   pre-existing stub-content filter rather than derived for this purpose.
   A future attempt should raise the floor to a value specifically chosen
   to catch the Stutsman/Providence pair (and verify it against the other
   three tracked pairs so it doesn't just trade one over-stripping failure
   for another).
2. If a zero-cleaning `raw`-style reconstruction is ever rebuilt for a
   future round, it must carry the caveat above — it is not equivalent to
   "the original baseline" and its Mantel/similarity numbers should not be
   quoted without noting the literal state-name token-overlap confound.

**Claim candidates**:
- **Claim**: an outcome-gated version of v3's frequency filter (v4) fixes
  v3's worst-case tail-outlier regression documented in §12 without
  regressing v3's tracked-pair or dispersion gains, but does not clear this
  experiment line's pre-registered adoption gate because it leaves one of
  the four tracked boilerplate pairs (Stutsman/Providence) unchanged.
  - Evidence: worst `top_far_similar_pairs` entry improves from 0.9607
    (v3) to 0.8727 (v4), while `tracked_pair_mean` (0.7541) and
    `pairwise_similarity_std` (0.0834) both still clear the gate's
    thresholds against the real §12 baseline; Stutsman/Providence
    similarity is 0.8327 under both v3 and v4, unchanged, versus 0.8255 at
    the real baseline — all measured in
    `analysis-output/variant-eval.json`.
  - Allowed wording: "an outcome-gated variant (v4) fixed the single
    worst-case far-but-similar pair identified in the prior round, but was
    rejected under the pre-registered gate because it left one tracked
    boilerplate pair unimproved"; "the current baseline (v3) still carries
    an unresolved short/rural-county over-stripping risk for at least one
    tracked pair."
  - Forbidden wording: any claim that v4 was adopted, partially adopted,
    or represents the new baseline — it was rejected in full, and
    `source_a_embeddings.parquet` is unchanged; also forbidden: citing this
    round's `raw` parquet's Mantel r (−0.2892) as a "true baseline" number
    without the token-overlap caveat above.
  - Status: resolved for this round's specific gate criteria; a follow-up
    attempt with a purpose-tuned outcome-gate floor (Next Actions #1)
    remains open.
