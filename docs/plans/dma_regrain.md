# DMA Re-grain — Plan

## Status

**Nothing built. This is a plan, written 2026-08-05.**

The downstream consumer joins county features on **DMA**, not on `fips_code`
(Comcast FreeWheel Revenue Science; asserted by Max, pending written
confirmation, same provenance as the rate answer in
`docs/downstream_target.md` Part 1).

`E_macro` is keyed on `fips_code` at n = 3,143. Nielsen defines 210 DMAs as sets
of whole counties, so the mapping is clean and many-to-one — roughly 15 counties
per DMA. Nothing in this repo is DMA-aware.

This document states what that costs, what it does not cost, and what would have
to be built. It does not re-derive `docs/downstream_target.md`; the rate answer
is unaffected, since row grain and geo key are separate properties.

---

## The three problems, ranked

### 1. DMA fixed effects make `E_macro` free — the existential one

With 210 DMAs and millions of impressions each, the consuming model can estimate
a 210-level DMA categorical precisely from its own data at no cost.

Any static DMA-keyed feature is a deterministic function of DMA identity.
`E_macro` is therefore **exactly collinear with a DMA fixed effect** — not
approximately, by construction. Cross-sectionally and in-sample, it adds
mathematically zero over `C(dma)`.

At county grain this argument was much weaker: 3,143 units, thinner data per
unit, fixed effects noisy and expensive to estimate. At 210 units the fixed
effect is cheap and precise, and it dominates.

**Every test in this repo measures cross-sectional association.** That is
precisely the quantity DMA dummies supply for nothing. The validation line is
not wrong, but it no longer targets the question that decides whether the
pillars are worth shipping.

Where a fixed effect genuinely fails, and `E_macro` can win:

| Value proposition | Why the fixed effect fails there |
|---|---|
| **Cold start** | A new or unmeasured DMA has no estimated effect. |
| **Partial pooling** | Shrink thin DMAs toward economically *similar* DMAs rather than toward the grand mean. `E_macro` supplies the similarity metric. |
| **Temporal transfer** | Effects fit on 2024 data go stale; economic features refresh on their own cadence. |
| **Interactions** | `E_macro × campaign vertical` may generalize where 210 × K dummies overfit. |
| **Interpretability** | Explains *why* a DMA prices the way it does. A dummy is a number. |

All five are defensible. **None has been tested in this repo**, and the harness
that would test them does not exist. This is the single largest gap the DMA
answer opens.

### 2. Feature count against row count

| | county | DMA |
|---|---|---|
| rows | 3,143 | 210 |
| pillar features (current) | 118 | 118 |
| critical \|r\|, α = 0.05 two-sided | 0.035 | **0.136** |
| standard-error inflation | — | **3.9×** |
| 5-fold grouped CV, test units per fold | ~629 | 42 |

Near p ≈ n. The project's posture inverts: it has spent its life deciding which
features earn a slot on evidence, and the binding constraint becomes that only
about **15–25 features** can be carried at all.

Partly offsetting, and genuinely unknown until measured: aggregating ~15 counties
averages measurement noise away, and aggregated correlations typically come out
**higher** than their county-level counterparts. Power falls, effect sizes
probably rise. The net direction is an empirical question Phase 2 answers
directly, and it should not be asserted in either direction before then.

### 3. Aggregation is re-derivation, not averaging

The tractable problem, but the one most likely to be silently botched. Most
pillars must be rebuilt from underlying quantities:

| Pillar | Rule | Why |
|---|---|---|
| **B** — QCEW LQs | **Re-derive.** Sum employment by sector across the DMA, recompute the quotient against the national base. | An LQ is a ratio against a national denominator. Averaging LQs across counties is simply wrong. |
| **E** — capital/wage | **Re-derive.** Sum numerator and denominator across member counties, then divide. | Same ratio problem. Also fixes the thin-county flags — see "What improves". |
| **D** — commodity shares | **Re-derive.** Sum tonnage by commodity group, then re-share. HHI recomputed on pooled partner flows, not averaged. | Shares and concentration indices do not average. |
| **C** — velocities | Recompute slopes from summed GDP and pooled unemployment where possible; population-weighted mean as fallback. | A slope of a sum ≠ mean of slopes, but the gap is small here. |
| **F** — typology flags | **Ambiguous — pick and document.** Fraction of member counties, population-weighted fraction, or dominant county. | Booleans have no canonical aggregate. This is a judgment call and must be recorded as one. |
| **A** — typed columns | Fraction of member counties carrying each flag; sum the count columns. | **This one improves.** Sparse booleans become continuous shares, which carry strictly more information. |

Governing rule, worth stating once and enforcing: **aggregate the inputs, not the
outputs.** Every derived feature in this repo is a ratio, share, quotient, slope
or index, and none of them survive a county-level mean.

---

## What improves

Not everything gets worse, and the plan should not pretend otherwise.

- **BLS suppression stops mattering much.** ~30% of county × sector LQ cells are
  suppressed, running to 67.4% on `lq_emp_21`. Suppressed cells are small-employer
  counties, so their share of a DMA-level employment total is small. DMA LQs are
  materially more reliable than county LQs. A per-DMA coverage flag replaces the
  per-county `disclosure_*` flags.
- **Source E's thin-county flags mostly dissolve.** `thin_claimer_flag` covers 37
  counties and `low_return_flag` 325; almost none survive aggregation into a
  15-county market.
- **Source A gets richer, not poorer.** 22 of its 29 columns are booleans on a
  single article. As DMA-level fractions they become continuous and better
  behaved.
- **Serving cost collapses.** 210 rows is a trivial feature table.
- **Political advertising fits exactly.** DMA is *the* unit of political ad
  buying, and 2026 is a midterm year. If that inventory is in scope, DMA-grain
  economic features are natively the right shape rather than a compromise.

---

## What the DMA answer does *not* invalidate

Stated so the re-grain does not get treated as a reset:

- **All six ingestion pipelines.** Unchanged; county parquets remain the source
  of truth and the aggregation is a layer on top.
- **Null semantics, vintage stamping, `as_of_date`, the schema docs.** Transfer
  as-is.
- **The rate answer.** Row grain and geo key are independent. Size stays a
  control.
- **Source A's typed-versus-scalar resolution.** It rests on construction — 22 of
  29 columns below |r| = 0.15 with size — not on a p-value, so it does not need
  the power that DMA grain removes.
- **The size-dependence *framework*.** The scan must be recomputed against
  *DMA* population; the method stands.

What does **not** transfer: every measured correlation, every power figure, and
every effect size in `analysis-output/`. All were computed at county grain.

---

## Phase 0 — Ask before building anything

**Cost: zero. Blocking.**

### 0.1 Does the impression row carry sub-DMA geo?

"Joins on DMA" has two readings and they imply very different work:

- **Case 1 — the feature store is keyed on DMA.** Everything in this document
  applies.
- **Case 2 — DMA is the *targeting and reporting* grain, but the impression row
  carries finer geo** (IP → ZIP or lat/long is standard in ad tech, and county is
  derivable from either). Then `E_macro` ships at county grain unchanged, and
  only the reporting rolls up.

Case 2 is far better and is common enough that it must be ruled out before a day
is spent. **Ask: does the impression row carry ZIP, postal code, or lat/long, and
can county-level features be joined even though targeting is DMA-level?**

### 0.2 Request their DMA crosswalk

Nielsen DMA definitions are proprietary. Public county→DMA mappings circulate
widely but with murky licensing.

**Ask FreeWheel for their own mapping.** Theirs is the definition that governs
the join, it removes the licensing question entirely, and a mismatch between
their DMA boundaries and a scraped one would silently corrupt every aggregate.
Also confirm the DMA count — 210 is the standard Nielsen figure, but vendor
mappings differ at the margin and some carry unassigned counties.

### 0.3 Confirm what baseline `E_macro` is being scored against

**Ask: does your current model already include a DMA-level effect, dummy, or
learned embedding?** If yes, that is the true baseline and problem 1 is live
today. If no, ask why not — the answer reveals whether they are volume-limited
per DMA, which is exactly the cold-start regime where `E_macro` wins.

---

## Phase 1 — Aggregation layer

**Gate: Phase 0 returns Case 1. Cost: 1–2 days.**

- `scripts/dma_crosswalk.py` — county→DMA mapping, cached parquet, coverage
  audit against all 3,143 counties, explicit handling of unassigned counties.
- `scripts/dma_aggregate.py` — the re-derivation rules in the table above, one
  function per pillar, each documenting its aggregation choice inline. Reads the
  six county parquets; writes `data/e_macro_dma.parquet`.
- Source F's flag rule is a judgment call. Implement the population-weighted
  fraction, and record the alternatives considered.
- **Ship both grains.** Once the layer exists, emitting county *and* DMA parquets
  from one derivation is nearly free, and it hedges Case 1 against Case 2 and
  against any future consumer at a different grain. This is the recommended
  posture regardless of which case Phase 0 returns.

Verification: every DMA aggregate reproduces its county inputs on a
single-county DMA; ratio columns re-derive rather than average, checked against
a hand-computed example per pillar.

---

## Phase 2 — Re-measure at DMA grain

**Gate: Phase 1. Cost: 0.5–1 day, mostly re-runs.**

Re-run at n = 210, and report against the county figures rather than replacing
them:

- `analyze_feature_size_dependence.py` against **DMA** population. Tier
  membership will move; a feature size-free at county grain need not be at DMA
  grain, and the reverse.
- `analyze_pillar_pair_crossvalidation.py` — the 50-pair sweep. Expect fewer
  survivors at the higher critical |r|, and larger surviving effects.
- `analyze_pillar_matrix_signal.py` — the 29-target matrix sweep.
- The B ↔ E link (r = 0.382 size-controlled at county grain) is the headline
  finding and must be re-measured explicitly.

Output: a paired county-versus-DMA table for every published figure. This is
also the deliverable that answers "does aggregation help or hurt", which is
currently unknown in both directions.

---

## Phase 3 — The fixed-effect benchmark

**The phase that decides whether the project has a value proposition.
Cost: 2–4 days. Cannot be fully run in-repo.**

The bar is no longer "does `E_macro` correlate with pillar features." It is:

> Does `E_macro` beat `target ~ C(dma)` **on held-out DMAs**?

Design:

- Grouped CV **by DMA**, 5 folds, 42 test DMAs per fold. Held-out-DMA evaluation
  is the primary test, not a robustness check — it is the only setting where a
  fixed effect cannot compete, because the test DMA's effect is unestimable.
- Compare three models: DMA dummies; `E_macro` features; both. The interesting
  quantity is `E_macro`'s lift on *unseen* DMAs, plus whether the combined model
  beats dummies alone on seen ones.
- Feature count must come down to roughly 15–25 first, or the comparison is a
  regularization contest rather than an information one. See Phase 4.
- Test partial pooling explicitly: `E_macro`-derived DMA similarity as a
  shrinkage prior on the fixed effects, scored against shrinkage toward the grand
  mean.

**This needs a target.** Options, in order of preference: a real label from the
consuming team even at 210 rows and one period; a public DMA-level proxy
(broadband adoption aggregates cleanly to DMA); or nothing, in which case
Phase 3 cannot run and that fact should be reported rather than papered over.

---

## Phase 4 — Feature reduction

**Gate: Phase 2. Cost: 1 day.**

118 features against 210 rows will not survive. Reduce before Phase 3, not after:

- Drop what Phase 2 shows is size-in-disguise at *DMA* grain.
- Collapse Source B's 20 LQs — plausibly to a handful of components or to the
  subset that survived the DMA sweep.
- Source A's 29 columns aggregate to fractions; keep the economic-fact columns
  (`sec_n_industry_mentions` carries 97.6% of the section gain) and drop the
  structural ones.
- Target 15–25 columns, chosen on Phase 2 evidence, with the cut list recorded.

---

## Decision points

1. **Case 1 or Case 2** (Phase 0.1). Case 2 makes Phases 1–2 unnecessary and
   leaves only Phase 3's benchmark question, which applies at any grain.
2. **Does a DMA-level effect already exist in their model** (Phase 0.3). If yes,
   Phase 3 is the whole project and Phases 1–2 are prerequisites to it. If no,
   `E_macro` has a much easier road and cold-start is the pitch.
3. **Is a target obtainable** — unchanged from `source_a_next_steps.md`
   question 2, but now decisive rather than merely valuable. Without one, Phase 3
   is unrunnable and the fixed-effect objection stands unanswered.
4. **Ship both grains, or DMA only.** Recommend both; the marginal cost after
   Phase 1 is near zero.

---

## Forbidden wording

Added on the same basis as the reporting rules in `source-a-findings.md` §14.5.

- **Do not quote a county-grain correlation as a DMA-grain result, or the
  reverse.** Aggregation changes correlations, usually upward. The repo's
  existing figures are county figures and must be labelled as such once a DMA
  table exists alongside them.
- **Do not describe `E_macro` as adding predictive value at DMA grain** until the
  Phase 3 held-out-DMA comparison has been run. Cross-sectional association is
  not evidence against a fixed effect; it is the thing the fixed effect already
  captures.
- **Do not describe the DMA aggregate as "county features rolled up"** where the
  aggregation was a re-derivation. The distinction is the whole correctness
  argument.
- **Do not claim the 210-DMA table has the statistical properties of the
  3,143-county table.** Critical |r| moves 0.035 → 0.136.

---

## Honest note on the project's premise

`docs/PROJECT_GOAL.md` states `E_macro`'s job as distinguishing physically
identical places in different economic climates — "a suburb outside New York
versus one outside Cleveland." That example survives: those are different DMAs.

What does not survive is discrimination *within* a market. At DMA grain,
Manhattan and rural Sullivan County share one vector. If the consumer's decisions
are made at DMA grain, that is not a loss — the features are exactly as fine as
the decisions they inform. If the consumer ever wants sub-DMA targeting, the
county parquets are still there, which is the strongest practical argument for
shipping both grains.

---

## Related

- `docs/downstream_target.md` — Part 1 records the geo-key question this document
  answers, and the rate answer that is unaffected by it.
- `docs/PROJECT_GOAL.md` — open decision #1 (closed), and the fusion step this
  re-grain sits in front of.
- `docs/plans/source_a_next_steps.md` — question 2 (is a label obtainable) is
  Phase 3's blocker.
- `scripts/pillar_matrix.py` — `SIZE_COLUMNS` and `SIZE_FEATURES`; the size
  control that must be recomputed against DMA population in Phase 2.
- `analysis-output/cross-source/` — every figure that Phase 2 re-measures.
