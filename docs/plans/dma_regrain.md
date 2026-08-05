# DMA Re-grain — Plan

## Status

**Nothing built. This is a plan, written 2026-08-05, revised the same day.**

The downstream consumer joins county features on **DMA**, not on `fips_code`
(Comcast FreeWheel Revenue Science; asserted by Max, pending written
confirmation, same provenance as the rate answer in
`docs/downstream_target.md` Part 1).

`E_macro` is keyed on `fips_code` at n = 3,143. Nielsen defines 210 DMAs as sets
of whole counties, so the mapping is clean and many-to-one — roughly 15 counties
per DMA. Nothing in this repo is DMA-aware.

**Revision: the impression row almost certainly carries ZIP** (Max, high
confidence; lat/long possible but unconfirmed). That is Phase 0.1 answered in the
favourable direction, and it changes what this plan recommends. Two facts were
being conflated:

- *They join on DMA* — a **choice** about their feature store.
- *The row carries ZIP* — a **capability**, which means county is derivable and
  the choice is reversible.

So the primary path is no longer "aggregate `E_macro` to DMA." It is **make the
case for a county-grain join**, with DMA aggregation retained as the fallback if
that case loses. The argument for it is in the next section and it is a
substantive one, not a preference.

This document states what the DMA grain costs, what it does not cost, and what
would have to be built under either outcome. It does not re-derive
`docs/downstream_target.md`; the rate answer is unaffected, since row grain and
geo key are separate properties.

---

## The argument for a county-grain join

> **[weakened 2026-08-05 by measurement]** The reasoning below is intact but its
> conclusion is no longer supported as stated.
> `analysis-output/cross-source/external-target-findings.md` §12 ran the three-arm
> comparison this document scoped for Phase 1B: row count costs −0.122 mean lift
> and **aggregation gains +0.106**, so the two roughly cancel. On three of five
> public targets the aggregated 208-market arm matches or beats full county
> grain. County grain is better on `broadband_rate` and `mean_commute_minutes`
> only — the first of which matters disproportionately, being closest to the
> consumer's domain and the one where the aggregate arm goes negative.
>
> **The defensible position is that the grain question is open and county is not
> established as better.** Do not carry the argument below into a conversation as
> though it were settled.

The fixed-effect objection in problem 1 below is not a fixed property of the
project. It is a function of **units per parameter**, and it swings hard with
grain:

| Grain | Units | Impressions per unit | A unit fixed effect is |
|---|---|---|---|
| **DMA** | 210 | millions each | Cheap and precise. `E_macro` is exactly collinear with it and adds zero. |
| **County** | 3,143 | fat head, **long thin tail** | Precise for large counties, poorly estimated for most of them. |

The median US county holds roughly 26,000 people, and its impression volume is
correspondingly small. A 3,143-level fixed effect is well estimated for Los
Angeles County and badly estimated across the several hundred counties carrying
thin traffic — which is precisely the cold-start and partial-pooling regime where
economic features are *not* redundant with the consumer's own history.

The case to put to the team:

> At DMA grain your own data already tells you everything these features could —
> a DMA dummy captures it and costs nothing. At county grain it does not, because
> most counties are thin, and that is where economic features substitute for
> history you do not have.

This is the strongest argument the project has, and it only exists if the join
can happen at county grain. **Establishing that it can is worth more than any
analysis in Phase 2.**

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

> **Revised 2026-08-05 for the access constraints.** There is no channel to the
> downstream team — communication is in person with one person, four days a week,
> with no async path (`docs/PROJECT_GOAL.md`, "Operating constraints"). So none
> of the questions below are emails. They are items on a **written one-pager**
> carried into a scheduled conversation, and they should be batched into a single
> pass rather than asked one at a time.
>
> Two consequences. **A data *extract* is a policy ask; a schema *look* is not.**
> What Phase 0 actually needs is column names, not values — one row's values
> answer nothing, and its schema answers most of 0.1 and 0.2. Being shown a row
> on a screen in a scheduled conversation is expected to be available, and is a
> far smaller request than a copy of the data. Record column names, not values.
>
> And **none of Phase 0 blocks Phase 1A or Phase 3**, both of which run entirely
> on public data. Nothing here should stall waiting for an answer.

### 0.0 Schema-inspection checklist

For the in-person look. Priority order, most decisive first.

| Look for | What it settles |
|---|---|
| A fine-geo column **alongside** a DMA column (`zip`, `postal_code`, `lat`/`lon`, `fips`, next to `dma`/`metro_code`) | The most valuable observation available. If both exist, DMA is a **proven choice** rather than a constraint, and the county-grain argument stops being an inference. |
| What one row is — impression, bid request, auction, pre-bid | Independently confirms the rate answer, and pre-bid versus post-bid rows carry different geo. |
| Target-shaped columns (`price`, `clearing_price`, `revenue`, `cpm`, `won`, `filled`) | Shows what is actually predictable, rather than reasoning about the target from first principles. |
| A geo-source or precision column (`geo_type`, `geo_source`, `location_precision`, an IAB-style accuracy code) | The only way to tell genuine lat/long from an IP-derived centroid. Otherwise unknowable. |
| Timestamp or date-partition columns | Whether models train on rolling windows, which decides if the vintage spread is a leakage defect. |
| Any volume or count column | Whether impressions-per-geo-unit is derivable, which is what Phase 3's thin-tail stratification needs at their grain rather than proxied by population. |

Three questions a single row cannot answer, so ask them aloud during the look:

1. **Is ZIP usually populated, or often null?** One row proves the column exists,
   not that it is filled. Coverage is the real question.
2. **Is this the serving-time payload, or an enriched table downstream?** Only
   serving-time geo supports a county-grain join.
3. **Does the model read these geo columns, or only DMA?** This is 0.3 — whether
   `E_macro` is arguing against an incumbent geo effect or filling a gap.

### 0.1 Does the impression row carry sub-DMA geo? — **probably yes: ZIP**

"Joins on DMA" has two readings and they imply very different work:

- **Case 1 — the feature store is keyed on DMA**, and nothing finer is available.
  Everything in this document applies.
- **Case 2 — DMA is the *targeting and reporting* grain, but the impression row
  carries finer geo.** IP → ZIP is standard in ad tech and county is derivable
  from it. `E_macro` then ships at county grain unchanged and only the reporting
  rolls up.

**Max is highly confident the row carries ZIP; lat/long is possible but
unconfirmed.** That puts this in Case 2, so the remaining question is not
capability but willingness — whether the consuming team will join at county grain
when their targeting is DMA-level. The argument for it is at the top of this
document.

Still to confirm, and all cheap:

- **Is ZIP on the row at serving time, or only in post-hoc reporting?** Only the
  former supports a county-grain feature join.
- **Does lat/long exist, and how is it sourced?** Point-in-polygon against TIGER
  county shapefiles is exact and removes the crosswalk entirely. But IP-derived
  lat/long is frequently a ZIP or city centroid, in which case it carries no real
  precision over the ZIP and should not be treated as better. Ask how it is
  derived before relying on it.
- **Is there an appetite to join at a finer grain than targeting?** This is the
  actual decision, and it is theirs.

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

## Phase 1A — ZIP→county join specification (the primary path)

**Gate: Phase 0 confirms ZIP is available at serving time. Cost: 0.5 day.**

This is not an aggregation. `E_macro` stays keyed on `fips_code` and unchanged;
what gets built is the mapping the consumer needs to attach it to a ZIP-keyed
row.

- **Crosswalk source: HUD-USPS ZIP Crosswalk Files.** Free, refreshed quarterly,
  gives ZIP → county FIPS with allocation ratios (`res_ratio`, `bus_ratio`,
  `oth_ratio`, `tot_ratio`) derived from address counts. Registration is needed
  for the API; a plain file download is also published. Fallback with no
  registration: the Census ZCTA-to-county relationship file.
- **ZIPs are not areas.** They are USPS delivery routes; ZCTAs are the areal
  approximation. Any document describing the join must say which is in use.
- **Assignment rule: dominant county by `res_ratio`,** carrying a
  `zip_county_confidence` column equal to that ratio so the consumer can see
  which ZIPs are ambiguous. Note what this is *not*: no feature is split,
  averaged or re-derived, because county features are being **attached to rows**
  rather than rebuilt from parts. The whole class of aggregation error in
  Phase 1B does not arise here.
- **Measure the ambiguity, do not quote it.** The share of ZIPs spanning more
  than one county — and the share of *addresses* affected, which is much smaller
  — is computable directly from the crosswalk. Report both.
- **If lat/long turns out to be genuine** (not an IP-derived centroid),
  point-in-polygon against TIGER county boundaries supersedes all of the above
  and is exact.

Deliverable: `data/zip_county_crosswalk.parquet` plus a short join note for the
handoff. No change to any pillar.

---

## Phase 1B — DMA aggregation layer (the fallback)

**Gate: Phase 0 returns Case 1, or the county-grain case is rejected.
Cost: 1–2 days — revised to 2–3, see below.**

> **[partly built 2026-08-05]** `scripts/geo_aggregate.py` implements the
> re-derivation rules against a caller-supplied grouping, and
> `scripts/analyze_grain_effect.py` runs the three-arm comparison. Both work on
> k-means clusters of county centroids at Nielsen cardinality, which stands in
> for DMAs without the proprietary delineation. Swapping in a real crosswalk is a
> one-argument change.
>
> **Cost correction: Source B cannot be re-derived from its shipped parquet.**
> `source_b_qcew.parquet` carries location quotients and disclosure flags and no
> employment counts, so the widest block in the matrix — 40 columns — can only be
> population-weighted, which is the operation §3 forbids. 62 of 118 columns are
> approximated for this and related reasons. Fixing it means changing
> `ingest_source_b.py` to carry `emp` alongside `lq_emp` and re-downloading the
> ~2.2GB QCEW singlefile. Add about a day. The 1–2 day estimate assumed every
> pillar could be rebuilt from its parquet and that assumption was wrong.

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

**Gate: Phase 1B. Skipped entirely if the county-grain join is accepted — every
existing figure already holds at county grain. Cost: 0.5–1 day, mostly re-runs.**

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

**This phase survives at either grain**, which is why it is the one that matters
regardless of how the join question resolves. The bar is no longer "does
`E_macro` correlate with pillar features." It is:

> Does `E_macro` beat `target ~ C(geo_unit)` **on held-out geo units**?

Design:

- Grouped CV **by geo unit**, 5 folds. At DMA grain that is 42 test DMAs per
  fold; at county grain ~629 test counties. Held-out-unit evaluation is the
  primary test, not a robustness check — it is the only setting where a fixed
  effect cannot compete, because the test unit's effect is unestimable.
- Compare three models: unit dummies; `E_macro` features; both. The interesting
  quantity is `E_macro`'s lift on *unseen* units, plus whether the combined model
  beats dummies alone on seen ones.
- **Stratify the county-grain version by volume.** The whole argument for county
  grain is that thin units are where fixed effects fail, so the result must be
  reported by impression-volume decile rather than pooled. A flat average would
  hide exactly the effect being claimed.
- Test partial pooling explicitly: `E_macro`-derived similarity as a shrinkage
  prior on the unit effects, scored against shrinkage toward the grand mean.
- At DMA grain, feature count must come down to roughly 15–25 first or the
  comparison is a regularization contest rather than an information one
  (Phase 4). At county grain 118 features against 3,143 units is workable and
  the reduction is optional.

**This needs a target, and it will be a public one.** A real label from the
consuming team is unobtainable by design, not merely unavailable
(`docs/PROJECT_GOAL.md`, "Operating constraints"), so the choice is not between a
label and a proxy. It is a public proxy or nothing.

Recommended: **ACS table B28002**, household presence and type of internet
subscription — county-level, free via the Census API, and the closest public
analogue to a FreeWheel-adjacent outcome. It aggregates cleanly to either grain.

**Disclose the limitation in the go/no-go rather than burying it.** The
fixed-effect comparison this phase runs is against a public proxy, so it answers
the objection *by analogy*, not directly. A confident answer to a slightly wrong
question is the specific failure mode here, and it should be named in the writeup
the same way `source-a-findings.md` §14.5 names its forbidden phrasings.

---

## Phase 4 — Feature reduction

**Gate: Phase 2. Cost: 1 day. Required only at DMA grain.**

118 features against 210 rows will not survive. Against 3,143 counties it is
fine, so this phase is a consequence of losing the grain argument rather than a
standing item. Reduce before Phase 3, not after:

- Drop what Phase 2 shows is size-in-disguise at *DMA* grain.
- Collapse Source B's 20 LQs — plausibly to a handful of components or to the
  subset that survived the DMA sweep.
- Source A's 29 columns aggregate to fractions; keep the economic-fact columns
  (`sec_n_industry_mentions` carries 97.6% of the section gain) and drop the
  structural ones.
- Target 15–25 columns, chosen on Phase 2 evidence, with the cut list recorded.

---

## Decision points

1. ~~**Case 1 or Case 2**~~ **(Phase 0.1) — probably Case 2.** The row carries
   ZIP, so county is derivable and the DMA join is a choice rather than a
   constraint. What remains is whether the consuming team will *accept* a
   county-grain join, which is a conversation, not an analysis.
2. **Will they join at county grain?** The live decision, and the highest-value
   one in the project. Winning it skips Phases 1B, 2 and 4 entirely and preserves
   every measured figure in `analysis-output/`. Losing it costs 3–4 days of
   re-grain work and materially weakens the pitch. The argument to make is at the
   top of this document.
3. **Does a unit-level effect already exist in their model** (Phase 0.3). If yes,
   Phase 3 is the whole project. If no, `E_macro` has a much easier road and
   cold-start is the pitch.
4. **Is a target obtainable** — unchanged from `source_a_next_steps.md`
   question 2, but now decisive rather than merely valuable. Without one, Phase 3
   is unrunnable and the fixed-effect objection stands unanswered at either
   grain.
5. **Ship both grains, or one.** Recommend both if Phase 1B is ever built; the
   marginal cost once the aggregation layer exists is near zero, and it hedges
   against a future consumer at a different grain.

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
- **Do not state that the consumer *cannot* join at county grain.** The row
  carries ZIP, so they can. What is true is that they currently join at DMA, and
  that is a choice they have not yet been asked to revisit. The two must not be
  reported as the same thing.
- **Do not report a pooled county-grain Phase 3 result.** The claim being tested
  is that fixed effects fail on thin units, so the result must be stratified by
  impression volume. A pooled average hides the effect in either direction.

---

## Honest note on the project's premise

`docs/PROJECT_GOAL.md` states `E_macro`'s job as distinguishing physically
identical places in different economic climates — "a suburb outside New York
versus one outside Cleveland." That example survives: those are different DMAs.

What does not survive is discrimination *within* a market. At DMA grain,
Manhattan and rural Sullivan County share one vector — both sit in the New York
DMA, which spans roughly 30 counties across four states. If the consumer's
decisions are made at DMA grain, that is not a loss: the features are exactly as
fine as the decisions they inform.

But this is also the second argument for the county-grain join, alongside the
fixed-effect one. A feature layer built to distinguish physically identical
places in different economic climates should not be delivered at a grain that
cannot separate Manhattan from Sullivan County. The premise and the delivery
grain should match, and at county grain they do.

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
