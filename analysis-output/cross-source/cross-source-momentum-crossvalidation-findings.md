---
type: results-report
date: 2026-07-14
experiment_line: cross-source-momentum-crossvalidation
round: 2
purpose: three-pillar-synthesis
status: active
---

# Source A / D / F ↔ Source C — Three-Pillar Momentum Crossvalidation (Synthesis)

> **The Source A arms are not reproducible as written (2026-08-03).** Both route
> through the cut `bge-m3` embedding, and their scripts
> (`analyze_source_a_source_c_correlation.py`,
> `analyze_source_a_source_f_correlation.py`, `analyze_source_a_clusters.py`)
> plus the two `.html` renders were deleted with it. The D and F arms are
> unaffected and still run. Recover the deleted scripts from git history if the
> cut is reversed; the Source A numbers survive here and in
> `outputs/source_a_source_c_correlation_pairs.csv` /
> `outputs/source_a_source_f_crossvalidation.csv`.

> This repo is not bound to an Obsidian project knowledge base (no
> `.claude/project-memory/registry.yaml`), so this stays a local markdown
> artifact, not an Obsidian write-back.

> **Round 2 update**: Round 1 synthesized the three existing pillar-vs-Source-C
> results and flagged that Source D's and Source F's crossvalidation scripts
> reported bare Pearson r with no significance test, and that no pillar had
> been tested directly against another (every crossvalidation routed through
> Source C). This round closes both gaps: permutation-test p-values were
> added to the Source D↔C and Source F↔C scripts, and two new direct
> pillar-to-pillar crossvalidations were run (Source D↔F, Source A↔F). The
> headline changed as a result — see SS1.

## 1. Executive Summary

`E_macro_extendedProposal.pdf` frames Source C's rolling 3-year unemployment/GDP
velocity as the hub that the other pillars should each explain some slice of,
while also describing direct pillar-to-pillar synergies (Trade Logistics:
Source D × Source F; Capital Flow: Source A × Source F). Round 1 of this note
synthesized three pillar-vs-Source-C results in isolation. Round 2 adds
significance testing to two of those and runs the two direct pillar-pair
tests the proposal itself describes.

**Result, revised**: the pillar-vs-Source-C-momentum links remain uniformly
weak (|r| ≤ 0.08 in every case), now confirmed by permutation tests rather
than bare point estimates — Source D's tonnage/HHI signals are the only ones
of the three that are consistently significant (p ≤ 0.006 in three of four
tests), Source A's embedding-similarity Mantel test remains significant
(p=0.0020) at a smaller magnitude, and Source F's demographic-distress
correlations are confirmed as noise (p=0.72, p=0.30). **But the two new
direct pillar-to-pillar tests are the strongest, most significant results
found in this project so far**: Source F's demographic distress count
correlates with Source D's log-tonnage at r = -0.2171 (p = 0.0020) — more
than 2.5x the size of any pillar-vs-Source-C link — and Source F's metro
status correlates with Source A's intro-text length at r = 0.2474 (p =
0.0020), the single largest effect size in the whole `E_macro` crossvalidation
program to date. Both proposal-described synergies (Trade Logistics, Capital
Flow) hold up at this coarse level. A finer-grained version of the Capital
Flow claim does not: Source F's graded distress count (as opposed to its
binary metro flag) does not track Source A's text length (r=0.0358, p=0.052,
and the wrong sign for the "more distress → less text" reading).

A separate, unrelated finding from Round 1 stands: Source C's missing-GDP
Connecticut counties and Source F's not-classified sentinel land on the exact
same 9 Planning Regions — a genuine cross-source confirmation of a shared
upstream data-lag artifact, not a momentum-signal result.

## 2. Data & Setup

| Pillars | Test | Target variable(s) | n | Significance test |
|---|---|---|---|---|
| A ↔ C | Mantel test, embedding-similarity matrix vs. economic-distance matrix | z-scored `(unemployment_velocity, gdp_velocity_pct)` distance | 2,786 | 499 permutations, p-value |
| D ↔ C | Pearson r, scalar signals vs. velocity | `unemployment_velocity`, `gdp_velocity_pct` | 3,143 / 3,080 | **added this round**: 499 permutations, p-value |
| F ↔ C | Pearson r, scalar signal vs. velocity | `unemployment_velocity`, `gdp_velocity_pct` | 3,143 / 3,080 | **added this round**: 499 permutations, p-value |
| D ↔ F | Pearson r, trade-flow signals vs. typology signals | `distress_count`, industry-dependence group means | 3,144 | **new**: 499 permutations, p-value |
| A ↔ F | Pearson r, text length vs. typology signals | `distress_count`, `metro_2023` (point-biserial) | 3,143 | **new**: 499 permutations, p-value |

`gdp_velocity_pct = gdp_velocity / gdp_latest` remains the shared
size-normalization fix (`source-c-findings.md` SS5) used by every test that
touches Source C's GDP column; it plays no role in the two new D↔F/A↔F tests,
which don't involve Source C at all.

Sample sizes are still **not directly comparable across pillar pairs**:
Source A's n=2,786/2,786-ish reflects its own stub-county and non-50-state
filters; Source D's and Source F's crosswalk coverage is complete (3,144/
3,144), so D↔F needed no filtering and A↔F only dropped the 1 non-50-state
entry Source A's own convention drops (`filter_to_fifty_states`) — notably,
the A↔F join did **not** drop stub counties, since thin/stub text length is
exactly the signal under test here, unlike Source A's other EDA scripts.

## 3. Main Findings

1. **Effect-size comparison across all five pillar-pair tests, ranked by
   |r|**:

   | Pillars | Comparison | r | p-value |
   |---|---|---|---|
   | A ↔ F | metro status vs. intro-text length (point-biserial) | **0.2474** | 0.0020 |
   | D ↔ F | distress count vs. log(tonnage) | **-0.2171** | 0.0020 |
   | D ↔ C | log(tonnage) vs. size-normalized GDP velocity | 0.0765 | 0.0020 |
   | D ↔ C | partner HHI vs. size-normalized GDP velocity | 0.0688 | 0.0020 |
   | D ↔ F | distress count vs. partner HHI | -0.0525 | 0.0040 |
   | D ↔ C | log(tonnage) vs. unemployment velocity | 0.0498 | 0.0060 |
   | A ↔ C | embedding similarity vs. economic distance (Mantel) | -0.0406 | 0.0020 |
   | A ↔ F | distress count vs. intro-text length | 0.0358 | 0.0520 (n.s.) |
   | D ↔ C | partner HHI vs. unemployment velocity | -0.0261 | 0.1640 (n.s.) |
   | F ↔ C | distress count vs. size-normalized GDP velocity | -0.0189 | 0.2960 (n.s.) |
   | F ↔ C | distress count vs. unemployment velocity | 0.0065 | 0.7180 (n.s.) |

   The two largest and most significant effects in the entire table are the
   two direct pillar-to-pillar tests added this round — both roughly 3-6x
   the magnitude of every pillar-vs-Source-C link. Every test against Source
   C's momentum measure tops out at |r|=0.08; the two structural-pillar-to-
   structural-pillar tests reach |r|=0.22-0.25. This is a real, substantive
   pattern, not an artifact of one noisy test: momentum (a short-run
   derivative) appears to be a fundamentally harder target for any of these
   pillars to explain than another pillar's own structural character is.

2. **Source D ↔ Source F confirms the proposal's Trade Logistics synergy at
   moderate strength**: counties with higher demographic distress ship
   substantially less freight (r=-0.2171, p=0.0020) and funnel it through
   less concentrated partner networks (r=-0.0525, p=0.0040). By industry
   dependence category (excluding the 9-county "Not classified" Connecticut
   sentinel, which sits at an outlying mean log(tons)=4.33 driven entirely by
   small-sample noise — see SS6 item 1), "None (nonspecialized)" and
   Manufacturing counties carry the highest mean tonnage (log10 tons = 3.80,
   3.73), Farming the lowest (3.21), with Government/Mining/Recreation in
   between (3.66/3.60/3.47) — a sensible ordering (manufacturing and
   diversified counties generate/attract more freight than farming-dependent
   ones) that lends the correlation face validity beyond the bare r value.

3. **Source A ↔ Source F confirms the coarse version of the proposal's
   Capital Flow framing, but not the fine-grained version**: metro counties'
   Wikipedia intro text is on average 63% longer than nonmetro counties'
   (509 vs. 313 characters, r=0.2474, p=0.0020) — the single strongest,
   most significant result in the whole `E_macro` crossvalidation program to
   date, and a direct confirmation that "for low-population or hyper-rural
   counties... online text data might be sparse" (`E_macro_extendedProposal.pdf`).
   But Source F's more granular 0-6 demographic distress count does **not**
   track text length beyond what the binary metro split already captures
   (r=0.0358, p=0.0520 — just short of conventional significance, and
   positive-signed, i.e. *more* distress associates with slightly *longer*
   text if anything, the opposite of the proposal's implied direction). Since
   nonmetro counties independently have higher mean distress
   (`source-f-findings.md` SS3.3), this raw distress-count correlation may
   simply be picking up residual metro/nonmetro composition rather than an
   independent distress effect — a partial-correlation check controlling for
   metro status was not run this round (SS6 item 3).

4. **Statistical rigor is now uniform across every pillar-vs-Source-C test**
   (resolves Round 1's SS3.2 gap): Source D's and Source F's
   crossvalidation scripts now report the same permutation-test p-value
   Source A's Mantel test always has, seeded and run identically
   (499 permutations, seed=42, `scripts/stats_utils.permutation_test_corr`).
   Confirmed: three of Source D's four correlations are genuinely
   significant (p ≤ 0.006); its fourth (partner HHI vs. unemployment
   velocity, p=0.164) is not. Both of Source F's correlations against Source
   C are confirmed as indistinguishable from noise (p=0.72, p=0.30) — this
   was asserted from the point estimate alone in Round 1 and is now formally
   supported.

5. **Connecticut Planning Region gap independently confirmed by two
   pillars** (carried over from Round 1, unrelated to momentum-signal
   strength): Source C's 9 GDP-missing Connecticut geographies
   (`source-c-findings.md` SS3.2) are the exact same 9 counties as Source
   F's 9 "not classified" industry-dependence sentinel counties
   (`source-f-findings.md` SS3.2). This also explains the "Not classified"
   outlier in finding 2 above — with only 9 counties and no real industry
   classification, its group means are not informative of anything about
   industry dependence itself.

## 4. Figure-by-Figure Interpretation

- `outputs/source_a_source_c_correlation.html` — density heatmap of
  embedding similarity vs. economic distance; shallow trend line, consistent
  with the small Mantel r.
- `outputs/source_d_source_c_crossvalidation.html` — bar chart of mean
  size-normalized GDP velocity by tonnage quartile; clean monotonic step
  (Q4 roughly 1.5x Q1), now backed by permutation-confirmed significance
  for the underlying correlations.
- `outputs/source_f_source_c_crossvalidation.html` — bar chart of mean
  size-normalized GDP velocity by demographic-distress count; flat across
  0-4 with a noisy 17-county dip at 5 — now formally confirmed as
  statistically indistinguishable from zero (p=0.30).
- `outputs/source_d_source_f_crossvalidation.html` — bar chart of mean
  log-tonnage by dominant industry dependence, colored to match Source F's
  own dependence-map palette. Farming sits visibly lowest, None/Manufacturing
  visibly highest (excluding the noisy 9-county "Not classified" bar) — the
  clearest categorical separation of any chart in this synthesis.
- `outputs/source_a_source_f_crossvalidation.html` — bar chart of mean
  intro-text length by demographic distress count. Visually near-flat across
  0-5, consistent with the borderline/null distress-count correlation in
  finding 3 — the metro/nonmetro effect (not shown in this particular chart)
  is the one that actually carries the signal.

## 5. Claim Candidates

- **Claim**: Direct pillar-to-pillar structural relationships (Source D ↔
  Source F, Source A ↔ Source F) are substantially stronger and more
  significant than any pillar's link to Source C's short-run momentum
  measure.
  - Evidence: |r|=0.22-0.25, p=0.0020 for both new tests, vs. |r|≤0.08 for
    every existing pillar-vs-C test (SS3.1 table).
  - Allowed wording: "structural pillars explain each other's static
    character substantially better than any of them explains Source C's
    short-run economic momentum."
  - Forbidden wording: implying these pillar-pair correlations establish
    causation, or that they generalize to momentum prediction — they don't
    involve Source C at all.
  - Status: **new finding this round**, the headline revision to Round 1's
    synthesis.

- **Claim**: The proposal's Trade Logistics synergy (Source D × Source F)
  is real at moderate strength: higher demographic distress predicts lower
  freight tonnage and lower partner concentration.
  - Evidence: r=-0.2171 (p=0.0020) tonnage, r=-0.0525 (p=0.0040) HHI, n=3,144
    (`analyze_source_d_source_f_correlation.py`).
  - Allowed wording: "counties with higher demographic distress ship
    measurably less freight and through less concentrated partner networks,
    a moderate but real confirmation of the proposal's Trade Logistics
    framing."
  - Forbidden wording: "distress causes low freight volume" (directionality
    untested); ignoring that the "Not classified" 9-county Connecticut group
    is noise, not a real industry-dependence signal.
  - Status: **resolves** `source-d-findings.md` SS6 item 1 /
    `source-f-findings.md` SS6 item 1.

- **Claim**: The proposal's Capital Flow synergy (Source A × Source F) holds
  at the coarse metro/nonmetro level but not at Source F's finer distress-count
  granularity.
  - Evidence: metro r=0.2474 (p=0.0020, n=3,143); distress-count r=0.0358
    (p=0.0520, n.s.) (`analyze_source_a_source_f_correlation.py`).
  - Allowed wording: "nonmetro counties have systematically thinner
    Wikipedia intro text than metro counties, confirming the proposal's
    rural-data-sparsity framing at the metro/nonmetro level; a finer
    distress-based gradient was not confirmed and may be confounded with
    metro status rather than independent."
  - Forbidden wording: "distress predicts text sparsity" (not significant,
    wrong-signed); treating the metro finding as validating any specific
    distress mechanism.
  - Status: **partially resolves** `source-f-findings.md` SS6 item 2 —
    metro confirmed, distress-count component open (SS6 item 3, this note).

- **Claim** (carried from Round 1, revised): across all pillar-vs-Source-C
  tests, none shows a practically strong momentum association; Source D's is
  the largest and most consistently significant, Source A's is smaller but
  significant, and Source F's is confirmed noise.
  - Evidence: full table in SS3.1; permutation p-values now computed for
    every test (SS3.4).
  - Status: **confirmed and strengthened** — Round 1 asserted this from
    point estimates alone; Round 2 formally verifies it.

## 6. Limitations / Open Items

1. **The 9-county "Not classified" Connecticut sentinel skews the D↔F
   dependence-category comparison (SS3.2)**: its outlying mean tonnage/HHI
   reflects small-sample noise from the same Connecticut Planning Region gap
   documented in SS3.5, not a real industry-dependence signal. Excluding it
   changes none of the reported correlations (which use the continuous
   `distress_count`, not the dependence label) but should be excluded from
   any future dependence-category ranking.
2. **The A↔F distress-count correlation may be confounded with metro status
   (SS3.3)**: nonmetro counties have higher mean distress
   (`source-f-findings.md` SS3.3), so the raw r=0.0358 between distress and
   text length could be residual metro/nonmetro composition rather than an
   independent effect. A partial correlation controlling for metro status,
   or restricting the test to nonmetro counties only, was not run this
   round.
3. **Source A ↔ Source D remains untested** — the one pillar pair not yet
   directly crossvalidated. No specific proposal claim points to this pair
   the way Trade Logistics (D×F) and Capital Flow (A×F) do, so it wasn't
   prioritized this round.
4. **Sample sizes still differ across pillar pairs (SS2)** — the D↔F/A↔F
   comparisons use the (near-)complete 3,144-county crosswalk, while A↔C
   uses Source A's own filtered n=2,786. Cross-pair effect-size comparisons
   in SS3.1 remain only approximately apples-to-apples for that reason.
5. **Source B (BLS QCEW) still doesn't exist** — the proposal's full
   three-way industry-mix synergy remains untestable end-to-end, as already
   noted in `source-a-source-c-correlation-findings.md` SS6 item 3 and
   `source-d-findings.md` SS6 item 2.

## 7. Next Actions

1. Run a partial correlation (or nonmetro-only subset) of Source F's
   distress count against Source A's content length, controlling for metro
   status, to determine whether SS3.3's borderline result is a real
   independent effect or purely metro composition.
2. Run the one remaining untested pillar pair, Source A ↔ Source D, for
   completeness, even though no specific proposal claim currently motivates
   it.
3. No further action needed on statistical rigor (SS3.4) — all five
   pillar-pair tests now report permutation p-values on the same protocol.
4. No action needed on the core substantive conclusion: structural
   pillar-to-pillar relationships (D×F, A×F-metro) are real and moderate;
   momentum-vs-any-pillar relationships remain uniformly weak. Both
   proposal-described synergies (Trade Logistics, Capital Flow) are
   confirmed at the level the evidence actually supports — moderate and
   coarse-grained, not the stronger or more granular claims a looser reading
   of the proposal might suggest.

## 8. Artifact and Reproducibility Index

- New shared utility: `scripts/stats_utils.py` (`permutation_test_corr`) —
  used by all four scripts below plus `analyze_source_a_clusters.py`'s
  existing `mantel_test` remains separate (matrix-pair case, unchanged).
- Source A ↔ C: `scripts/analyze_source_a_source_c_correlation.py` →
  `outputs/source_a_source_c_correlation.html`,
  `outputs/source_a_source_c_correlation_pairs.csv` (unchanged this round).
- Source D ↔ C: `scripts/analyze_source_d_source_c_correlation.py` →
  `outputs/source_d_source_c_crossvalidation.csv`,
  `outputs/source_d_source_c_crossvalidation.html` (permutation p-values
  added this round; regenerated via `scripts/generate_source_d_insights.py`
  → `analysis-output/source-d/source_d_stats.json`).
- Source F ↔ C: `scripts/analyze_source_f_source_c_crossvalidation.py` →
  `outputs/source_f_source_c_crossvalidation.csv`,
  `outputs/source_f_source_c_crossvalidation.html` (permutation p-values
  added this round; regenerated via `scripts/generate_source_f_insights.py`
  → `analysis-output/source-f/source_f_stats.json`).
- Source D ↔ F (new): `scripts/analyze_source_d_source_f_correlation.py` →
  `outputs/source_d_source_f_crossvalidation.csv`,
  `outputs/source_d_source_f_crossvalidation.html`.
- Source A ↔ F (new): `scripts/analyze_source_a_source_f_correlation.py` →
  `outputs/source_a_source_f_crossvalidation.csv`,
  `outputs/source_a_source_f_crossvalidation.html`.
- Reproduction: `uv run scripts/<script_name>.py` for each script above;
  all use `RANDOM_SEED=42`, `N_PERMUTATIONS=499`, matching Source A's
  original Mantel-test protocol for comparability across every test in
  SS3.1's table.

## 9. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf`)

The proposal frames `E_macro`'s pillars as mutually reinforcing along two
specific stated synergies — Trade Logistics (Source D's freight character
explained by Source F's structural typology) and Capital Flow (Source F
anchoring counties where Source A's text is sparse) — on top of the general
expectation that each pillar adds independent context around Source C's
momentum core. This round's evidence supports a more precise version of that
framing than either individual pillar's report or Round 1's synthesis could
establish alone:

- **Momentum links (pillar vs. Source C) remain uniformly weak**, now with
  full statistical confirmation rather than point estimates — consistent
  with every individual pillar's own conclusion that it measures something
  structurally complementary to, not predictive of, Source C's short-run
  velocity.
- **The proposal's specific pillar-to-pillar synergy claims are the ones
  that actually hold up with real effect sizes**: Trade Logistics (D×F,
  r=-0.22) and Capital Flow (A×F metro, r=0.25) are both confirmed, and are
  the two strongest results in the entire `E_macro` crossvalidation program
  to date — stronger evidence for the proposal's synergy narrative than any
  of the momentum tests provide.
- **The Capital Flow claim's finer-grained version is not supported**: the
  proposal's framing implies a graded relationship between rural/sparse-data
  severity and text thinness, but only the binary metro/nonmetro split shows
  a real effect; Source F's graded distress count does not add signal beyond
  that split (SS3.3, SS6 item 2).

Net assessment: `E_macro`'s pillars are best read as a mosaic of genuinely
complementary structural signals that explain each other's static character
better than any of them explains short-run economic momentum — a more
precise and more positive reading of the proposal's synergy claims than
Round 1 could support, now that the two pillar-to-pillar tests the proposal
actually describes have been run directly rather than only inferred through
Source C.
