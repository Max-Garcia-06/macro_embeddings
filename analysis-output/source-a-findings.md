---
type: results-report
date: 2026-07-03
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

4. **Notable outlier pairs**: a handful of far-apart county pairs are
   unusually similar in text, e.g. Lincoln County, KS ↔ Lincoln County, OR
   (similarity 0.845 at 2,207 km — likely driven by sharing a county name,
   which produces near-identical opening-sentence boilerplate) — consistent
   with finding #1 (geography matters a little on average; individual pairs
   can still be highly similar regardless of distance).

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
