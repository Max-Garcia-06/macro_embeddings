---
type: results-report
date: 2026-07-13
experiment_line: source-a-source-c-correlation
round: 1
purpose: joint-correlation-findings
status: active
---

# Source A ↔ Source C — Embedding Similarity vs. Economic Momentum (Joint Findings)

> **Not reproducible as written (2026-08-03).** Everything here tests the
> `bge-m3` embedding, which was cut. `analyze_source_a_source_c_correlation.py`
> and `analyze_source_a_clusters.py` were deleted along with it, as was
> `outputs/source_a_source_c_correlation.html`. Script paths and the HTML below
> are a record of what was run, not live paths — recover from git history
> (`git log -- scripts/analyze_source_a_source_c_correlation.py`) if the cut is
> reversed. The numbers survive in this document and in
> `outputs/source_a_source_c_correlation_pairs.csv`; the input
> `data/source_a_embeddings.parquet` is retained. Source A's current position
> against the other pillars is `source-a-findings.md` §13–§17, on typed
> features rather than on the embedding.

> This repo is not bound to an Obsidian project knowledge base (no
> `.claude/project-memory/registry.yaml`), so this stays a local markdown
> artifact, not an Obsidian write-back.

## 1. Executive Summary

Source A's findings report (`source-a-findings.md` §8 item 5, §10) flags one
concrete open gap: everything tested there was Source A embedding similarity
against *geography* — never against a real economic variable, which is the
`E_macro_extendedProposal.pdf` claim that actually matters (intro text
carries distinctive economic-transition narrative). Source C now supplies
that variable, so this round runs the same Mantel-test methodology
(`analyze_source_a_clusters.mantel_test`) against economic distance instead.

**Result**: Mantel r = -0.0406, p = 0.0020 (499 permutations, n = 2,786
counties). Statistically significant, but the association is roughly 6x
weaker than Source A's own geography correlation (r = -0.2362) and explains a
negligible share of variance — detectable only because of the large number
of county pairs, not because the relationship is meaningfully strong. The
negative sign means higher textual similarity weakly associates with *lower*
economic distance (a faint positive link, not a repellent one), but the
magnitude does not support the proposal's claim that intro text carries a
usable economic-transition signal.

A secondary finding: the ranked "textually similar but economically
diverging" pair list is dominated by a small handful of counties sitting at
the extreme tails of the whole corpus's GDP-velocity distribution (§3.2) —
the same "one outlier county repeats across the whole top-N list" pattern
Source C's own report already documented for King County, WA (§3.5 of
`source-c-findings.md`). This is a real methodological limitation of the
ranking, not a data error, and is flagged rather than silently patched.

## 2. Data & Setup

- **Source A**: `source_a_embeddings.parquet`, filtered exactly as in
  `analyze_source_a_clusters.py` — stub counties dropped (<100 chars of
  de-boilerplated content, `drop_stub_counties`) and non-50-state entries
  dropped (`filter_to_fifty_states`).
- **Source C**: `source_c_fred.parquet` — `unemployment_velocity`,
  `gdp_velocity`, `gdp_latest` columns.
- **Join**: inner join on `fips_code`, dropping any county missing an
  embedding or either velocity value. **n = 2,786** counties (down from
  Source A's 2,849 after the 50-state/non-stub filter, and Source C's 3,080
  fully-covered counties — the intersection after both filters).
- **Economic distance metric**: z-scored Euclidean distance over
  `(unemployment_velocity, gdp_velocity_pct)`, where
  `gdp_velocity_pct = gdp_velocity / gdp_latest` is computed locally in this
  script (not written back to `source_c_fred.parquet`). Raw `gdp_velocity` is
  dollar-denominated and confounded with economy size, per
  `source-c-findings.md` §5's own finding that the largest county economies
  dominate any distance-based ranking on the raw column — using the
  percentage-growth form avoids importing that confound into this
  correlation test.
- **Test**: `mantel_test` from `analyze_source_a_clusters.py`, reused
  unchanged (it operates on any two same-shape square matrices, not
  geography-specific) — 499 permutations, seed=42, same protocol as Source
  A's geography test for direct comparability.

## 3. Main Findings

1. **Mantel test (embedding similarity vs. economic distance)**:

   | | Source A vs. geography (for comparison) | Source A vs. Source C economic distance |
   |---|---|---|
   | Mantel r | -0.2362 | **-0.0406** |
   | p-value | 0.0020 | 0.0020 |
   | n | 2,849 | 2,786 |

   Both are significant at the 499-permutation floor (p=0.002), but the
   economic-distance association is roughly 6x smaller in magnitude than the
   already-"weak" geography association. By the same effect-size convention
   Source A's report uses for its own geography claim, this is weaker than
   weak — real, but not practically usable as an economic signal.

2. **Outlier domination in the ranked pair list**: the top-20
   "textually-similar-but-economically-diverging" pairs
   (`source_a_source_c_correlation_pairs.csv`) repeatedly feature the same
   few counties:

   | County | `gdp_velocity_pct` | Corpus percentile |
   |---|---|---|
   | Kittson County, Minnesota | -0.2372 | ~min (corpus min = -0.2372) |
   | Massac County, Illinois | -0.2353 | 2nd-lowest |
   | Campbell County, South Dakota | -0.2054 | 3rd-lowest |
   | Borden County, Texas | +0.2060 | ~max (corpus max = +0.2060) |

   Massac County, IL appears in 6 of the top 20 pairs; Kittson County, MN in
   4; Campbell County, SD and Borden County, TX in 3 each. These four
   counties are literally the most extreme `gdp_velocity_pct` values in the
   entire 3,080-county corpus, so any pairing of them with a county that
   happens to share boilerplate-level embedding similarity (small rural
   counties with near-identical "founded... population... county seat" text,
   per Source A §3.4's shared-skeleton mechanism) ranks near the top by
   construction. **This mirrors Source C §3.5's King County, WA finding
   almost exactly** — a single-axis outlier repeating across a distance-based
   ranking, not a set of genuinely informative "close pairs."

## 4. Figure-by-Figure Interpretation

**`outputs/source_a_source_c_correlation.html`** — density heatmap of
embedding similarity (x) vs. economic distance (y) across all 3.9M county
pairs, with a fitted trend line and the top-20 outlier pairs highlighted.
The trend line's slope is visually shallow, consistent with the small
Mantel r — this rules out reading the highlighted outlier pairs (extreme
cases by construction, see §3.2) as evidence of a meaningful relationship
between text and economic momentum.

## 5. Claim Candidates

- **Claim**: Source A embedding similarity has a statistically detectable
  but practically negligible association with Source C economic-momentum
  distance, roughly 6x weaker than Source A's own geography association.
  - Evidence: Mantel r=-0.0406, p=0.0020, n=2,786, 499 permutations, seed=42
    (`analyze_source_a_source_c_correlation.py`); compare Source A's
    geography result r=-0.2362 (`source-a-findings.md` §3, claim 1).
  - Allowed wording: "embedding similarity and economic-momentum distance
    are weakly associated at a statistically significant level, but the
    effect is smaller than Source A's already-weak geographic signal and
    should not be read as evidence that intro text carries usable economic
    information."
  - Forbidden wording: "Source A embeddings track/predict economic
    momentum"; any wording implying a moderate or usable-strength effect.
  - Status: **resolves the open gap in `source-a-findings.md` §8 item 5 /
    §10** — proposal's specific economic-transition-narrative claim remains
    unsupported, now against a real economic variable rather than only
    geography.

## 6. Limitations / Open Items

1. **Outlier domination (§3.2)**: the ranked "diverging pairs" list is not a
   useful source of individually interesting pairs as-is — it mostly
   re-surfaces the same 3-4 corpus-extreme counties. A percentile-rank or
   log-scaled economic distance (rather than raw z-scored Euclidean
   distance) would likely surface more genuinely distinct pairs, matching
   `source-c-findings.md` §6's own open item about its King County, WA
   domination.
2. **Only whole-embedding similarity was tested**, not PC1 or K-means
   cluster membership specifically against Source C's quadrants. Source A's
   report (§3.2) already establishes PC1 as a non-economic
   Wikipedia-editorial-convention artifact (Texas founding/namesake
   narrative), so a null result there would be unsurprising and was not run
   this round to keep scope to the whole-embedding test the open gap
   actually named.
3. **Source B not yet built** — the proposal's stated three-way synergy
   (Source B's industry mix "explaining" Source C's momentum, with Source A
   as narrative context) still can't be tested end-to-end.

## 7. Next Actions

1. This closes the specific open item in `source-a-findings.md` §8 item 5 /
   §10 ("test embedding distance against real economic variables") — worth a
   short cross-reference edit in that file if it's kept as the canonical
   Source A doc.
2. If the "close-in-text but diverging-in-economics" framing is still wanted
   for its own sake (independent of the correlation test), rebuild the
   ranking on a percentile-rank or log-scaled economic distance to avoid the
   §3.2 outlier-domination effect before trusting the pair list.
3. No action needed on the correlation result itself — it's a resolved
   negative, consistent with Source A's existing "generic boilerplate, weak
   geographic echo, no proposal-described economic-narrative signal"
   picture.

## 8. Artifact and Reproducibility Index

- Analysis script: `scripts/analyze_source_a_source_c_correlation.py`
  (reuses `analyze_source_a_clusters.filter_to_fifty_states` /
  `mantel_test`, `analyze_source_a_similarity.drop_stub_counties`,
  `visualize_source_a.load_embeddings`, `visualize_source_c.load_source_c` —
  no changes to any existing script or dataset).
- Outputs: `outputs/source_a_source_c_correlation.html` (scatter),
  `outputs/source_a_source_c_correlation_pairs.csv` (top 20 ranked pairs).
- Reproduction: `uv run scripts/analyze_source_a_source_c_correlation.py`.
  Seeded (`RANDOM_SEED=42`, `N_PERMUTATIONS=499`, matching Source A's
  geography Mantel test for comparability).

## 9. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf`)

Source A's own proposal-alignment assessment (§10 of `source-a-findings.md`)
concluded the unsupervised structure found in the embeddings didn't look
like the proposal's described economic-transition narrative, but left one
door open: "correlation against real economic variables (Source E/B) is
still untested." This round closes that door for Source C specifically
(Source E/B remain untested): the correlation exists and is statistically
real, but at r=-0.04 it is an order of magnitude too small to support the
proposal's claim that intro text gives "immediate semantic separation"
between counties on economic grounds. Combined with §3.2's outlier-driven
pair list, the practical takeaway is the same as Source A's standalone
verdict — Source A's mechanism as specified in the proposal is not
delivering the described role, now confirmed against a real economic axis
rather than only geography.
