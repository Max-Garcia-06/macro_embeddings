# Downstream Target — The Answered Question, and What It Settles

> **STATUS: TARGET SHAPE IS A RATE. Asserted by Max 2026-08-05, pending written
> confirmation from the consuming team.**
>
> The consumer is the **Comcast FreeWheel Revenue Science** team. One row in
> their training data is an impression, ad request, auction, household, or
> device — not a county. Every such row carries a per-row target, so county
> population is not on the left-hand side under any of them. **County size is a
> control, not a feature.**
>
> No target *variable* has been supplied, and none is needed for this decision:
> the row grain settles it. Part 1 records the question, why row grain answers
> it, and what confirmation would still add. Part 2 was written as a conditional
> placeholder and is now **operative** — its predictions were made in advance and
> can be scored.
>
> **Confidence and its limit.** The assertion is Max's knowledge of the consuming
> team, not a written statement from them. It is strong enough to build on and
> is recorded as the repo's operating assumption. If written confirmation comes
> back contradicting it, the invalidation conditions at the end of Part 2 are the
> rollback path and every one of them still applies.
>
> Merged 2026-08-04 from `downstream_target_assumptions.md` and
> `plans/downstream_target_question.md`. Content refreshed against
> `source-a-findings.md` §13–§17, which reinstated Source A as 29 typed columns
> and changed its refeaturization row from "stays cut" to "ship". Answered
> 2026-08-05.

---

# Part 1 — The question, and why the row grain answers it

## The answer

**Rate.** County size is a control. `r_size_controlled` is the operative
scorecard everywhere in this repo, and the raw `r` column must not be quoted
without its size-controlled partner.

## How it was settled — row grain, not the metric

The question below was written to be put to the downstream team. It did not need
to be asked in that form, because a weaker and more certain fact answers it:

> **What is one row in your training data?**

For FreeWheel Revenue Science the row is an impression, ad request, auction,
household, or device. The answer is the same across all five:

| Row | Target example | County population on the left-hand side? |
|---|---|---|
| impression / request / auction | clearing price, eCPM, fill, win rate | no — population sets the row *count*, not the row *value* |
| household | ARPU, lifetime value, churn | no — household spend does not scale with county size |
| device | completion rate, engagement | no |

There is no per-row target for which county population is part of the outcome.
Population determines how many rows a county contributes, not what any one of
them is worth. That is what makes size a control here, and it is why the answer
is robust to not yet knowing the exact metric.

**Row grain is the better question to ask in general.** It is factual, teams
always know it, and it answers rate-versus-count as a byproduct. A team that
cannot name its metric can always name its row.

## What written confirmation would still add

Not the rate/count answer — that follows from the row grain. Confirmation is
worth having for three things it would settle that this reasoning does not:

1. **The geo join key** — county, ZIP, or DMA. Now the highest-value open
   question in the project; see "What is still open" below.
2. **Whether they hold a size feature already.** An ad-tech model holds request
   volume per geography, which is a better population proxy than anything this
   repo would ship. If so, `E_macro` should ship no size column at all.
3. **Whether the outcome is time-varying with a backtest window**, which would
   move the vintage spread from a documentation gap to a blocking defect.

## The geo key — answered 2026-08-05: DMA

Same provenance as the rate answer, same pending-confirmation caveat. It is the
worst of the three possible answers.

- ~~**County**~~ — would have shipped as-is.
- **DMA** ← **this one.** 3,143 counties collapse to 210 markets. DMAs are
  defined as sets of whole counties so the mapping is clean, but effective sample
  size drops ~15×, critical |r| moves 0.035 → 0.136, and every power figure in
  this repo needs restating. The larger problem is not resolution: a DMA fixed
  effect is cheap and precise at 210 units with millions of rows each, and any
  static DMA-keyed feature is **exactly collinear with it by construction**.
- ~~**ZIP**~~ — would have needed a population-weighted many-to-many crosswalk.

**Full consequences and the staged plan: `docs/plans/dma_regrain.md`.** Nothing
is built. Its Phase 0 is three zero-cost questions that come before any code, the
first of which — does the impression row carry sub-DMA geo? — could make most of
the work unnecessary.

## The grain mismatch is now a live defect, not a hypothetical

This was trap 1 in Part 2 and was written while the target was unknown. With the
row confirmed as impression-level it is the single most important thing to hand
to the consuming team — **more important than which columns ship.**

Millions of training rows carry only **3,143 distinct feature values**. Effective
sample size is the county count, not the row count. A model evaluated with random
k-fold over impression rows puts every county in both train and test with
identical feature values; measured performance for `E_macro` will be inflated,
confidence intervals will be far too tight, and the failure will surface in
production rather than in evaluation. This is the same error as the naive
correlation test over 3.9M county *pairs* caught in the Source A analysis, one
level up.

Both mitigations are required, not recommended:

- Cluster standard errors by `fips_code`.
- Grouped, **spatially blocked** CV folds. Never random k-fold — counties are
  spatially autocorrelated, so random folds put a neighbour of every test county
  into train.

## The question as it was written, retained for reuse

The framing below is kept because the same question has to be asked of any future
consumer, and because it records what each answer would have changed.

## The question

> When your model consumes county features, is the thing you're predicting a
> **rate** or a **count**?
>
> Concretely: is the target something like *revenue per subscriber*, *churn
> probability*, *take rate* — already normalized, so a big county and a small
> county can score the same? Or is it something like *total revenue*, *subscriber
> count*, *number of service calls* — where a big county scores higher because it
> is big?
>
> I'm not asking what the target is. Just whether population is already divided
> out of it.

That framing matters. Teams often cannot name their metric yet but always know
whether it is per-customer or a county total.

## Why this one question

It decides whether county size is a **control** or a **feature**. That single
choice reverses conclusions across the whole repo, not just one pillar.

| | rate target | count target |
|---|---|---|
| county size | **control** — regress it out before fusion | **feature** — keep it |
| operative sweep column | `r_size_controlled` | raw `r` |
| Source D freight `log_total_tons` | dead — r = +0.865 with size | strong |
| Source F `metro_2023` | demote — r = +0.592 | strong |
| Source A `content_length` | compromised — r = +0.355 | defensible, cheap size proxy |
| Source A's 29 typed columns | **win** — 22 of 29 below \|r\| = 0.15 | interpretability only |
| Source C velocities, Source E ratio | cleanest pillars — r ≈ 0.02–0.10 | unremarkable |

Source: `outputs/feature_size_dependence.csv`, 120 features scanned against
`log10(Census population)` — full tiering in Part 2. The proxy was the
tax-return count until 2026-08-04; swapping it changed no tier membership above
\|r\| = 0.30 and moved `has_university` across the 0.15 line
(`source-a-findings.md` §18). The two figures in this table were still quoted
against that retired proxy (+0.871 and +0.596) until the 2026-08-12 rescan;
they now read against Census population like every other number here.

### It settled something no amount of further testing could — and did

Whether Source A's 29 typed columns beat the `content_length` scalar they
replaced is powered at **0.34** and would need **110 targets** to resolve
statistically (`source-a-findings.md` §17.3). That comparison was never going to
be settled in this repo by adding targets.

Under the rate answer it does not need to be. `content_length` sits at +0.355
with county size; `sec_n_industry_mentions` — the single column carrying 97.6% of
the section gain — sits at +0.110. **The typed block wins on construction rather
than on a p-value.** The comparison is closed, and Plan 3 in
`docs/plans/source_a_next_steps.md` — expand the in-repo target set — is dead,
because the only comparison it could have won is this one.

## What the answer changes

### Rate — the live branch

- County size is a control.
- `r_size_controlled` is the operative scorecard; the raw `r` column is actively
  misleading and must not be quoted without its partner.
- **Source D's ten raw per-commodity tonnages move into `SIZE_COLUMNS`**
  (done 2026-08-05, `pillar_matrix.py`). This was pre-registered in
  `source-d-findings.md` §11 as the action a rate answer would trigger: the raw
  tonnages are levels wearing a commodity label and were retained only because a
  count target would have made them legitimate. The ten `share_*` composition
  columns stay. Cost of the removal, measured: mean lift across the 29-target
  matrix sweep moved +0.0847 → +0.0851 and the definitional share of that lift
  fell 0.522 → 0.514. One target (`lq_emp_22`, Utilities LQ) dropped below the
  signal bar from an ablated lift of +0.0009, which is noise-level. **Removing
  ten size-in-disguise columns cost nothing.**
- Demote Source F's `metro_2023`, `population_loss`, `housing_stress`.
- Source A ships the typed block, and the case for it is structural rather than
  statistical.
- Add pillars in cleanliness order: C, E and Source A's typed block first, then
  the B LQ vector, then D's shares and HHIs, then F's distress flags.

### "Control" does not mean "delete" — a correction this document needed

An earlier version of this line read *"regress it out of every pillar before
fusion."* That is wrong for this consumer, and the error is worth recording
because it would have destroyed real signal.

Advertiser demand is genuinely denser in metro markets. A size-correlated column
such as `metro_2023` may be predicting ad price **through** urbanicity, which is
a causal driver, not through a population artifact. Deleting every size-loaded
column would delete that.

The correct operation is the one Part 2's validation plan already specifies:
fit `target ~ log_population + density` as the floor, then add each pillar and
take **marginal lift over that baseline**. A size-loaded column is kept if and
only if it beats size alone. That is an empirical test run per column, not a
policy applied to a whole pillar — and it is the same procedure under either
answer, which is why the fusion build was never truly blocked on this question.

What the answer does decide: `E_macro` **ships no size column**. The consuming
team holds request volume per geography, which is a better population proxy than
anything in these six pillars.

### Count — the branch not taken

- Size is a legitimate feature and the raw correlations stand.
- D and F recover most of their apparent value.
- `content_length` is defensible again as a cheap size proxy, and the typed
  block's case rests on interpretability alone.
- Flag honestly that the project is then partly a population model, and that this
  is a choice rather than an accident.

### "Already size-normalized upstream" — still worth confirming

A different failure mode, and the one branch the row-grain argument does *not*
close. If the consuming team supplies per-capita features themselves,
double-normalization removes real signal. Ask whether they normalize before or
after joining county features. Carry this into the confirmation conversation.

## If they push back — retained for the confirmation conversation

**"Why does it matter — just give me all the features."**
You can, and the features ship either way. But 18 of the 50 cross-pillar
correlations lose more than half their effect once county size is controlled
(`outputs/pillar_pair_crossvalidation.csv`). If the target is a rate and the raw
numbers get taken at face value, the model gets built on structure that is mostly
population wearing a costume. The largest raw effect in the whole sweep — freight
tonnage against metro status, r = 0.495 — collapses to −0.036 under the control.

**"We haven't picked the target yet."**
The weaker version is still decisive: *is your outcome per-customer/per-request,
or a county total?* That is usually known long before the exact metric is.

**"Can't you just test both?"**
Both are already reported. The problem is not producing two numbers, it is that
the two point at opposite feature sets, and shipping a feature store means
committing to one. Publishing both without a decision pushes the decision onto
whoever consumes it, with less context than we have.

## Still to ask — these did not get answered by the row grain

### 0. ~~What geo key do the models join on?~~ Answered: DMA

Superseded by three new questions, all zero-cost and all blocking:
does the impression row carry sub-DMA geo (ZIP or lat/long)? Can we have their
DMA crosswalk? Does their model already carry a DMA-level effect?
`docs/plans/dma_regrain.md` Phase 0.

### 1. Is there any path to a real label — at any horizon?

Not "do you have one now" — *ever*. Even 500 counties, even a stale extract.

- **Yes, with meaningful probability** → every in-repo test is provisional.
  Minimize further validation spend and hold budget for the real thing.
- **Hard no, never** → that is the argument for spending 2–4 days ingesting an
  external public county-level target. Broadband or household internet adoption
  is the closest analogue to a Comcast downstream model, and would be the only
  non-circular evidence this project ever produces.

The reason this matters: every target in the repo's validation is another
pillar's feature, which penalizes a source precisely for agreeing with the
pillars it will ship alongside. That is the right penalty for assembling a
non-redundant feature store and the wrong one for predicting an external outcome.

### 2. Does your model care about dynamics, or about industry composition?

Source A's marginal value against a baseline holding every other pillar swings by
two orders of magnitude depending on the target:

| target pillar | retained |
|---|---|
| C (velocity series) | 68% |
| D (freight) | 61% |
| B (QCEW location quotients) | 30% |
| F (typology) | 14% |
| E (capital-to-wage) | ≈0% |

Source A survives where no federal agency measures the same construct and is
absorbed where one does. The published headline of **+0.0010** is an average over
a basket that is **71% a single BLS table**, so it should never travel without
that composition attached.

## What not to ask

**Do not ask them to bless the feature set, or to confirm a p-value.** They
cannot, and it invites a decision they are not positioned to make. Source A's
typed block ships either way — it costs one regex pass and no model download.

**The rate-versus-count question was the only one whose answer changed what gets
built.** It is answered. The geo-key question is what now sits in that position.

---

# Part 2 — The analysis, now operative

> Written 2026-08-04 as a conditional placeholder assuming a rate-shaped target.
> That assumption was confirmed by row grain on 2026-08-05, so **everything below
> is the repo's operative analysis rather than a hypothesis** — with the one
> correction recorded in Part 1: "control" means scored over a size baseline, not
> regressed out and discarded. Its predictions were stated before the answer
> arrived and can be scored on the record.

## Why a placeholder was needed

All validation in this repo to date is **pillar-versus-pillar**: 50 feature-pair
tests across 15 pillar pairs (`outputs/pillar_pair_crossvalidation.csv`), plus
the Source A marginal harness (`outputs/source_a_marginal.csv`), which scores one
pillar against a baseline holding all five others. Both measure coherence and
redundancy between sources. Neither can measure predictive usefulness, because
usefulness is defined relative to a target that does not exist here — and every
target in both is another pillar's feature, which penalizes a source precisely
for agreeing with the pillars it will ship alongside.

Without a target, `docs/PROJECT_GOAL.md` open decision #1 — *is county size a
control or a feature?* — was unanswerable, and it blocked the fusion step. The
placeholder existed to unblock that reasoning, not to pre-empt the downstream
team's actual choice. **Part 1 has since closed it on row grain**, and the
placeholder's shape turned out to be the right one.

## The target shape, and what it makes county size

| Target shape | Example | What county size is |
|---|---|---|
| **Count / total** | total revenue, total subscribers, request volume | A **feature**. The target scales mechanically with population; a model without size is badly specified. |
| **Rate / ratio** ← **this one** | revenue **per request**, eCPM, ARPU, margin % | A **control**. The denominator already normalizes for volume. Size enters only through correlates such as urbanicity and pricing tier. |

Revenue per request was adopted as the placeholder metric and is a rate. The row
grain confirms the shape independently of the metric: **county size is a control,
not a feature** — so `r_size_controlled` is the operative scorecard and the raw
`r` column should not be quoted without its size-controlled partner.

Note the second row's final clause, which the earlier "regress it out" framing
contradicted: size enters legitimately through urbanicity and pricing tier. That
is why the operation is marginal lift over a size baseline rather than deletion.

## Evidence: which features are size in disguise

Every pillar feature correlated against `log10(Census population)`. This is the
diagnostic that determines whether a feature transfers to a rate target. The
proxy was Source E's `num_returns` until 2026-08-04; each row's correlation
against the retired proxy is retained in `r_with_log_returns` so the swap stays
checkable (`source-a-findings.md` §18).

**120 features scanned; 44 exceed |r| = 0.30 with county size.** Full table in
`outputs/feature_size_dependence.csv`.

> **Rescanned 2026-08-12.** The counts above were 62 and 15. The scan's feature
> list is now read from `pillar_matrix.build_matrix()` rather than from a
> hand-kept list inside `analyze_feature_size_dependence.py`, so it covers every
> shipping column by construction. What it had been missing: Source D's ten
> commodity shares and its three log tonnage columns, Source B's 20
> `disclosure_*` flags, Source C's `unemployment_rate_latest` and
> `gdp_velocity`, and Source E's whole re-featurized block. **Every correlation
> that existed before is unchanged to the last decimal** — nothing was
> re-estimated, the panel was widened. One rename: `log_tons` is now
> `log_total_tons`, the name the matrix uses for the same quantity.

**Tier 1 — size in disguise** (|r| ≥ 0.30). Contribute little to a rate target
beyond what population already supplies.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `log_inbound_tons` | D | **+0.904** | 82% | 3,143 |
| `log_total_tons` | D | **+0.865** | 75% | 3,143 |
| `log_outbound_tons` | D | +0.802 | 64% | 3,143 |
| `disclosure_61` (Education) | B | −0.614 | 38% | 2,669 |
| `metro_2023` | F | +0.592 | 35% | 3,143 |
| `disclosure_55` (Management) | B | −0.592 | 35% | 2,502 |
| `wage_per_return_thousands` | E | +0.568 | 32% | 3,143 |
| `low_return_flag` | E | −0.560 | 31% | 3,143 |
| `n_body_sections` | A | +0.547 | 30% | 3,143 |
| `has_metro_attachment` | A | +0.541 | 29% | 3,143 |
| `disclosure_71` (Arts/Rec) | B | −0.538 | 29% | 2,948 |
| `disclosure_72` (Accom/Food) | B | −0.514 | 26% | 3,126 |
| `share_out_sctg3499` | D | +0.514 | 26% | 3,143 |
| `lq_emp_11` (Agriculture) | B | −0.506 | 26% | 1,457 |
| `high_farming` | F | −0.497 | 25% | 3,134 |
| `disclosure_52` (Finance) | B | −0.478 | 23% | 3,125 |
| `share_in_sctg0109` | D | −0.473 | 22% | 3,143 |
| `disclosure_56` (Admin/Waste) | B | −0.470 | 22% | 3,112 |
| `disclosure_53` (Real Estate) | B | −0.467 | 22% | 3,019 |
| `disclosure_62` (Health Care) | B | −0.465 | 22% | 3,128 |
| `industry_dependence_farming` | F | −0.456 | 21% | 3,134 |
| `lq_emp_54` (Professional svcs) | B | +0.453 | 21% | 2,268 |
| `share_out_sctg0109` | D | −0.426 | 18% | 3,143 |
| `disclosure_22` (Utilities) | B | −0.426 | 18% | 2,852 |
| `gdp_velocity` (dollar-denominated) | C | +0.420 | 18% | 3,080 |
| `lq_emp_21` (Mining) | B | −0.416 | 17% | 1,026 |
| `lq_emp_56` (Admin/Waste) | B | +0.412 | 17% | 2,212 |
| `population_loss` | F | −0.400 | 16% | 3,126 |
| `founding_year` | A | −0.395 | 16% | 1,214 |
| `share_in_sctg3499` | D | +0.390 | 15% | 3,143 |
| `disclosure_81` (Other svcs) | B | −0.384 | 15% | 3,128 |
| `disclosure_51` (Information) | B | −0.383 | 15% | 3,028 |
| `disclosure_31-33` (Manufacturing) | B | −0.381 | 14% | 3,083 |
| `disclosure_48-49` (Transport) | B | −0.380 | 14% | 3,116 |
| `share_in_sctg1014` | D | +0.364 | 13% | 3,143 |
| `content_length` | A | +0.355 | 13% | 3,143 |
| `disclosure_54` (Professional svcs) | B | −0.352 | 12% | 3,132 |
| `housing_stress` | F | +0.350 | 12% | 3,143 |
| `nonspecialized` | F | +0.337 | 11% | 3,134 |
| `in_partner_hhi` | D | +0.329 | 11% | 3,143 |
| `n_distinct_proper_nouns` | A | +0.328 | 11% | 3,143 |
| `out_partner_hhi` | D | +0.326 | 11% | 3,143 |
| `industry_dependence_none` | F | +0.320 | 10% | 3,134 |
| `lq_emp_53` (Real Estate) | B | +0.314 | 10% | 2,453 |

Three groups joined tier 1 on the wider scan, and each says something:

- **Source B's `disclosure_*` flags, 13 of 20 of them.** Suppression tracks
  county size almost as strongly as any feature here does, which is the same
  fact `source-b-findings.md` reports as suppressed cells having a median of 5
  establishments against 40 for disclosed ones. They are shipped features, not
  metadata, so their size loading belongs on this list.
- **Source D's commodity shares, 5 of 10.** Expected and already documented in
  `source-d-findings.md` §11: the shares are far cleaner than the raw tonnages
  they replaced, not clean outright.
- **`gdp_velocity`.** The dollar-denominated velocity Source C already
  recommends against, at +0.420 while its normalized counterpart sits at +0.101.
  It is still inside the matrix's Source C block; see
  `docs/source_c_feature_schema.md`.

**Tier 2 — partly size** (0.15 ≤ |r| < 0.30), 28 features. Usable; know what
they carry. `lq_emp_22` (−0.292), `lq_emp_55` (+0.292), `disclosure_23`
(−0.287), `disclosure_21` (−0.287), `disclosure_44-45` (−0.286), `disclosure_99`
(−0.285), `has_economy_section` (+0.268), `share_out_sctg1014` (+0.259),
`lq_emp_61` (+0.251), `lq_emp_62` (+0.245), `lq_emp_42` (−0.243),
`thin_claimer_flag` (−0.228), `lq_emp_99` (−0.228),
`gain_per_claimer_thousands` (+0.215), `share_out_sctg2033` (+0.214),
`share_in_sctg2033` (+0.211), `disclosure_42` (−0.209), `low_postsecondary_ed`
(−0.191), `lq_emp_81` (+0.189), `high_mining` (−0.175), `distress_count`
(−0.172), `disclosure_11` (−0.172), `dividend_participation_rate` (+0.171),
`concentrated_gain_flag` (+0.169), `low_employment` (−0.168),
`sec_has_manufacturing` (+0.163), `capital_to_wage_ratio_normalized_std`
(−0.159), `has_university` (+0.150 — 0.145 under the retired proxy, so it sits
on the tier boundary rather than having changed character).

**Tier 3 — effectively size-free** (|r| < 0.15). These transfer cleanly to a
rate target.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `has_tribal_land` | A | −0.002 | 0.0% | 3,143 |
| `lq_emp_71` (Arts/Rec) | B | +0.010 | 0.0% | 2,243 |
| `has_oil_gas` | A | +0.011 | 0.0% | 3,143 |
| `has_timber` | A | −0.016 | 0.0% | 3,143 |
| `has_mining` | A | −0.018 | 0.0% | 3,143 |
| `has_protected_land` | A | +0.019 | 0.0% | 3,143 |
| `capital_to_wage_ratio` | E | **+0.019** | 0.0% | 3,143 |
| `lq_emp_23` (Construction) | B | −0.030 | 0.1% | 2,469 |
| `has_interstate` | A | +0.035 | 0.1% | 3,143 |
| `has_agriculture` | A | +0.049 | 0.2% | 3,143 |
| `retirement_destination` | F | +0.050 | 0.3% | 3,134 |
| `has_manufacturing` | A | +0.064 | 0.4% | 3,143 |
| `has_logistics` | A | +0.075 | 0.6% | 3,143 |
| `unemployment_velocity` | C | **+0.075** | 0.6% | 3,143 |
| `n_industry_mentions` | A | +0.079 | 0.6% | 3,143 |
| `has_tourism` | A | +0.086 | 0.7% | 3,143 |
| `lq_emp_52` (Finance) | B | +0.086 | 0.7% | 2,638 |
| `has_namesake` | A | −0.090 | 0.8% | 3,143 |
| `has_river` | A | +0.094 | 0.9% | 3,143 |
| `gdp_velocity_pct` | C | **+0.101** | 1.0% | 3,080 |
| `has_port` | A | +0.101 | 1.0% | 3,143 |
| `has_military_base` | A | +0.105 | 1.1% | 3,143 |
| `sec_n_industry_mentions` | A | **+0.110** | 1.2% | 3,143 |
| `lq_emp_48-49` (Transport) | B | +0.113 | 1.3% | 2,160 |
| `persistent_poverty` | F | −0.114 | 1.3% | 3,111 |
| `lq_emp_44-45` (Retail) | B | +0.118 | 1.4% | 3,050 |
| `lq_emp_51` (Information) | B | +0.131 | 1.7% | 1,914 |
| `lq_emp_72` (Accommodation/Food) | B | +0.135 | 1.8% | 2,369 |
| `lq_emp_31-33` (Manufacturing) | B | −0.140 | 2.0% | 2,570 |

Source A's remaining section columns — `sec_has_tourism` (+0.094),
`sec_has_logistics` (+0.087), `sec_has_agriculture` (+0.059), `sec_has_mining`
(+0.033), `sec_has_oil_gas` (−0.029), `sec_has_timber` (+0.009) — also sit in
Tier 3.

The 2026-08-12 rescan adds 13 more, for **48 in Tier 3 of 120 scanned**:
`high_recreation` (F, +0.005), `industry_dependence_manufacturing` (F, +0.018),
`has_usda_echo` (A, −0.020), `share_out_sctg1519` (D, −0.024),
`high_manufacturing` (F, −0.034), `high_government` (F, +0.034),
`industry_dependence_recreation` (F, −0.039), `capgain_participation_rate`
(E, −0.039), `industry_dependence_government` (F, +0.048),
`unemployment_rate_latest` (C, +0.063), `capital_to_wage_ratio_normalized_mean`
(E, +0.068), `share_in_sctg1519` (D, +0.102), `industry_dependence_mining`
(F, −0.134).

Worth noting which pillar dominates that list. **Seven of the 13 are Source F**,
and Source F's typology columns split cleanly: the industry-dependence and
high-concentration flags for manufacturing, government, recreation and mining
are size-free, while `metro_2023`, `high_farming`, `nonspecialized`,
`industry_dependence_farming` and `population_loss` are tier 1. The pillar is not
uniformly a size proxy, which matters for how its slot is argued — see
`analysis-output/cross-source/pillar-marginal-findings.md`.

**Source A's typed block is the cleanest large block in this table under a rate
target.** Of its 29 columns, **22 fall in Tier 3, three in Tier 2, and four in
Tier 1.** Of those four, `has_metro_attachment` is already ablated from the
cross-pillar sweeps as a restatement of Source F's `metro_2023`
(`source-a-findings.md` §16.2), `founding_year` fires on only 1,214 counties, and
`content_length` is the retired incumbent scalar, which the block retains as one
column among 29 rather than as the pillar's whole representation.

The distinction that matters is *within* the block. Its size-loaded columns are
the structural ones — how long the article is, how many proper nouns it contains,
whether it has an economy section — and its size-free columns are the ones that
name economic facts. `sec_n_industry_mentions`, at +0.110, is the single column
responsible for 97.6% of the section gain (§14.3). **Under a rate target the part
of Source A that carries the signal is also the part that survives the size
control**, which is the strongest evidence here that the rate-versus-count answer
moves Source A rather than leaving it where it was.

Reproduce with:

```bash
uv run python scripts/analyze_feature_size_dependence.py
```

The same analysis is presented for review in the notebook section "What a
downstream target would settle."

**The inversion.** The pillars with the largest raw cross-pillar effects (D, F)
are largely population wearing a costume. The pillars that looked weakest in the
raw sweep (C, E) are the ones that transfer cleanly to a rate target. D's
`log_tons` at r = 0.871 with size is why D↔F collapsed from 0.495 to −0.057 —
nothing subtle was happening.

**The size proxy no longer belongs to a pillar.** This table used
`num_returns`, a Source E column, until 2026-08-04, which made Source E's
apparent independence partly self-referential. It is now Census PEP population
(`scripts/county_population.py`). The two agree at r = 0.998 in logs and the
tiering is unchanged — but the self-reference was real, and removing it moved
`capital_to_wage_ratio` from +0.040 to **+0.019**, making Source E cleaner than
the old table showed rather than dirtier. Full accounting in
`source-a-findings.md` §18.

## Refeaturization the rate target implies

Every row below is now an instruction rather than a conditional. A, D and E have
landed; B, C and F are annotation and demotion work carried into the fusion step.

| Pillar | Action | Rationale |
|---|---|---|
| **A** | **Ship the 29 typed columns, not the `content_length` scalar alone** | Reverses this document's original row, which predated `source-a-findings.md` §13–§17. The embedding stays cut; the text source does not. 22 of the 29 columns are Tier 3, and the signal-carrying `sec_n_industry_mentions` sits at +0.110 against `content_length`'s +0.355 — so a rate target *strengthens* Source A's case rather than confirming a cut. Exclude `has_metro_attachment` (+0.541, duplicates F's `metro_2023`), treat `founding_year` (−0.395, n = 1,214) as low-coverage, and annotate `content_length` and `n_distinct_proper_nouns` as the block's size-loaded columns. |
| **B** | Ship the 20-dim LQ vector; annotate per-column size loading | LQs are **not** uniformly size-neutral. `lq_emp_11`/`lq_emp_54` are urbanicity axes; `lq_emp_23`/`lq_emp_71` are clean. Both usable — the point is knowing which is which. |
| **C** | Ship as-is | Both velocities are near-orthogonal to size. |
| **D** | **[corrected 2026-08-05]** Ship the ten commodity *shares* and the two HHIs; the ten raw per-commodity tonnages moved to `SIZE_COLUMNS` | This row originally read "normalize to `tons_per_return` / `tons_per_capita`, the highest-value single change implied here." **That was wrong.** `log10(tons / population) == log_total_tons − log_population` to 8.9e-16, and both terms already sit in the design — so the per-capita column is an exact linear combination of columns every baseline holds, and D-vs-F is −0.036 size-controlled whichever input is used (`source-d-findings.md` §10). No `tons_per_capita` column ships. What did help was the transformation this document never proposed: composition rather than volume. Five of the ten shares clear the size-free bar that none of the raw tonnages cleared (§11). `out_partner_hhi` was rechecked as this row asked — +0.275 raw to +0.115 size-controlled, the least size-dependent thing D has. |
| **E** | Ship as-is | Cleanest pillar under this target. |
| **F** | Demote `metro_2023`, `population_loss`, `housing_stress`; keep the rest | `metro_2023` (0.596) duplicates population, which the downstream model very likely already holds. `population_loss` (−0.393) and `housing_stress` (+0.349) are also size-loaded — the "distress flags are size-light" reading only holds for `persistent_poverty` (−0.139), `low_employment` (−0.197), and `retirement_destination` (+0.052). |

## Modeling traps specific to this target

### 1. Grain mismatch — the Mantel problem one level up

**Promoted 2026-08-05 from hypothetical to live.** The row grain is confirmed as
impression / request / auction / household / device, so this trap is not a risk
the consuming team might run into — it is one they are exposed to by default, and
it is the most important item in this document to hand them.

Features are county-level (n = 3,143). The target is per-row. Joining county
features onto request rows produces millions of rows carrying only 3,143
distinct feature values.

**Effective sample size is the county count, not the request count.** A model
reporting significance on request-level rows will report absurdly tight
confidence intervals, for exactly the reason a naive correlation test on 3.9M
county *pairs* did in the Source A analysis. Under random k-fold, every county
appears in both train and test carrying identical feature values, so `E_macro`
will measure well in evaluation and do nothing in production.

Mitigations, both required:
- Cluster standard errors by `fips_code`.
- Cross-validate with **grouped, spatially blocked folds** — never random k-fold.
  Counties are spatially autocorrelated; random folds put a neighbor of every
  test county in train and inflate measured performance.

### 2. Ratio denominator noise

Revenue per request in a county with 40 requests is mostly variance. Decide the
policy **before** fitting, not after seeing which counties look like outliers:
weight by request count, model numerator and denominator separately, or impose a
minimum-volume floor. Source E has a precedent worth copying — `low_return_flag`
marks the 325 counties where a ratio is unreliable for the same reason.

### 3. Null semantics under a rate target

Unchanged by the target choice, but still unresolved. BLS suppression runs far
above the 30% headline on individual columns (`lq_emp_21` 67.4%, `lq_emp_99`
65.8%, `lq_emp_55` 58.8%, `lq_emp_11` 53.6%). Tree ensembles consume
null-plus-`disclosure_*`-flag natively; linear models and neural networks need a
stated imputation policy that does not exist yet.

### 4. Vintage spread and leakage

| Pillar | Vintage | `as_of_date` |
|---|---|---|
| A | continuous (scrape date) | 2026-08-03 |
| B | QCEW 2025 Q4 | 2025-12-31 |
| C | unemployment 2025, GDP 2024 | 2025-12-31 |
| D | FAF `tons_2022` | 2022-12-31 |
| E | IRS TY 2022 | 2022-12-31 |
| F | USDA typology, **2025 edition** (2023 OMB metro delineation) | 2025-12-31 |

Three years of spread presented as one county snapshot. Acceptable for a static
feature layer; a leakage vector if a downstream model backtests against pre-2025
outcomes, since B, C and F would carry future information.

**The `as_of_date` column now exists** (`scripts/pillar_vintage.py`, 2026-08-04)
on all six parquets, with the full table in `outputs/pillar_vintages.csv`, so the
consuming team can detect this without reading these docs. It is excluded from
the feature matrix, so it travels with the data without entering any model. Note
the correction in that row: Source F is the **2025 edition** — `metro_2023` is
named for the OMB delineation it uses, not for the release year, and this table
previously said "USDA 2023 typology".

## Validation plan

1. ~~**Confirm grain with the downstream team**~~ **Done 2026-08-05** — rows are
   per impression / request / auction / household / device. This settled the
   target shape (Part 1) and promoted trap 1 to a live defect. What it did *not*
   settle is the **geo join key** (county / ZIP / DMA), which is now the open
   item in its place.
2. ~~**Census population swap.**~~ **Done 2026-08-04** —
   `scripts/county_population.py`.
3. ~~**Refeaturize D as a rate.**~~ **Superseded 2026-08-04, closed
   2026-08-05.** The rate normalization was a re-expression, not a fix; the ten
   commodity shares were the transformation that helped, and the ten raw
   tonnages have now moved to `SIZE_COLUMNS`. See the corrected D row above.
4. **Baseline: target ~ county size + population density only.** This is the bar
   every pillar must clear, and it is the operation "size is a control" actually
   means — marginal lift over this floor, scored per column, rather than
   regressing size out of the pillars.
5. **Add pillars in cleanliness order** — C, E and the Source A typed block
   first, then the B LQ vector, then D's shares and HHIs, then F distress flags.
   Permutation importance at each step, grouped CV throughout.
6. **Cut on the same standard applied to the Source A embedding.** Any pillar
   that fails to beat the size-only baseline goes, regardless of its cross-pillar
   r. Source A's embedding step was cut on exactly this standard and its text
   source survived the same test in typed form — the standard is not a proxy for
   a verdict about a source, only about a representation of it.

Steps 4–6 are the fusion build, and they are no longer blocked. They were never
as blocked as this document assumed: the procedure is identical under either
target shape, and the answer changes which pillars survive it rather than what
has to be written.

Expected outcome, stated in advance so it can be wrong on the record: C, E, the B
vector and Source A's typed block clear the bar; D clears it through its shares
and HHIs rather than its tonnages; F contributes through `persistent_poverty` and
`low_employment` rather than metro status.

## Invalidation conditions

**These are now the rollback path**, not a hypothetical. The rate answer rests on
Max's assertion of the consuming team's row grain, pending written confirmation.
Re-derive Part 2 if the real target turns out to be:

- **A count or total** (total revenue, subscriber counts, request volume). Size
  becomes a feature, the raw `r` column becomes the operative one, D and F
  recover most of their apparent value, and Source A's `content_length` scalar
  becomes defensible again as a cheap size proxy.
- **Already size-normalized upstream** by the downstream team (e.g. they supply
  per-capita features themselves). Double-normalization would remove real signal.
- **Time-varying with a backtest window.** The vintage table above moves from a
  documentation gap to a blocking defect.
- ~~**Defined at a grain finer than county**~~ **This one has already fired, and
  it does not invalidate anything.** The target *is* finer than county —
  impression, household, device. That makes `E_macro`'s columns group-level
  covariates and makes the trap-1 clustering requirements non-negotiable, which
  is now recorded there. It does not touch the rate conclusion: a finer grain is
  what *produces* a per-row target, so it corroborates Part 1 rather than
  contradicting it. Listed here originally as a risk to the analysis; in the
  event it is a risk to the *consumer's evaluation setup*.

---

## Related

- `docs/PROJECT_GOAL.md` — open decision #1 was Part 1's question, stated from the
  repo's side; both are now closed.
- `analysis-output/E_macro_key_findings.ipynb` — section "What a downstream
  target would settle" presents this analysis for review.
- `outputs/pillar_pair_crossvalidation.csv` — the 50-test sweep this reinterprets.
- `outputs/feature_size_dependence.csv` — every r-with-size figure quoted here.
- `analysis-output/source-a/source-a-findings.md` §13–§17 — the typed-extraction
  round that reinstated Source A. §14.2a, §17.2a and §17.3 carry the power figures
  and basket composition behind the claim that the rate-versus-count answer
  settles more than another sweep could.
- `docs/plans/source_a_next_steps.md` — the five plans for the two items §17.3
  leaves open; Part 1's question is question 4 there.
