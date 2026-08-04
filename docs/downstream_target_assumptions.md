# Downstream Target Assumptions

> **STATUS: PLACEHOLDER — NOT A REQUIREMENT FROM THE DOWNSTREAM TEAM.**
>
> No target variable has been supplied to this repo. This document adopts
> **revenue per request** as a working assumption so that pillar-selection
> decisions can be reasoned about instead of deferred. Every conclusion here is
> conditional on that assumption and must be re-derived if the real target
> differs in the one property that matters: whether it is a **rate** or a
> **count**. See [Invalidation conditions](#invalidation-conditions).

> **Refreshed 2026-08-04** against `source-a-findings.md` §13–§17, which
> reinstated Source A as 29 typed columns and changed its row in the
> refeaturization table below from "stays cut" to "ship".

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
control or a feature?* — is unanswerable, and it blocks the fusion step. The
placeholder exists to unblock that reasoning, not to pre-empt the downstream
team's actual choice.

## The property that decides everything: rate vs count

| Target shape | Example | What county size is |
|---|---|---|
| **Count / total** | total revenue, total subscribers, request volume | A **feature**. The target scales mechanically with population; a model without size is badly specified. |
| **Rate / ratio** | revenue **per request**, ARPU, margin % | A **control**. The denominator already normalizes for volume. Size enters only through correlates such as urbanicity and pricing tier. |

Revenue per request is a rate. Under this placeholder, **county size is a
control, not a feature.**

Three consequences for how existing results are read:

1. The `r_size_controlled` column in the sweep is the operative scorecard.
2. The raw `r` column becomes actively misleading and should not be quoted
   without its size-controlled partner.
3. **It settles a Source A comparison that no amount of in-repo testing can.**
   Whether the 29 typed columns beat the `content_length` scalar is powered at
   0.39 and would need 91 targets (`source-a-findings.md` §17.3) — it will not be
   resolved statistically here. But under a rate target the question changes
   shape: `content_length` at +0.359 with size is compromised, while
   `sec_n_industry_mentions` at +0.108 is not, so the typed block wins on
   construction rather than on a p-value. Under a count target the scalar is
   defensible and the typed block's case rests on interpretability alone. **One
   answer from the downstream team settles what 91 targets could not.**

## Evidence: which features are size in disguise

Every pillar feature correlated against `log10(num_returns)`. This is the
diagnostic that determines whether a feature transfers to a rate target.

62 features scanned; **15 exceed |r| = 0.30 with county size.** Full table in
`outputs/feature_size_dependence.csv`. The scan now covers Source A's 29 shipping
typed columns rather than the `content_length` scalar alone, which is why the
count rose from the 34 features this document originally reported.

**Tier 1 — size in disguise** (|r| ≥ 0.30). Contribute little to a rate target
beyond what population already supplies.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `log_tons` | D | **+0.871** | 76% | 3,143 |
| `metro_2023` | F | +0.596 | 36% | 3,143 |
| `has_metro_attachment` | A | +0.541 | 29% | 3,143 |
| `lq_emp_11` (Agriculture) | B | −0.501 | 25% | 1,457 |
| `lq_emp_54` (Professional svcs) | B | +0.461 | 21% | 2,268 |
| `lq_emp_21` (Mining) | B | −0.419 | 18% | 1,026 |
| `lq_emp_56` (Admin/Waste) | B | +0.408 | 17% | 2,212 |
| `founding_year` | A | −0.398 | 16% | 1,214 |
| `population_loss` | F | −0.393 | 15% | 3,126 |
| `content_length` | A | +0.359 | 13% | 3,143 |
| `housing_stress` | F | +0.349 | 12% | 3,143 |
| `in_partner_hhi` | D | +0.333 | 11% | 3,143 |
| `out_partner_hhi` | D | +0.329 | 11% | 3,143 |
| `n_distinct_proper_nouns` | A | +0.329 | 11% | 3,143 |
| `lq_emp_53` (Real Estate) | B | +0.319 | 10% | 2,453 |

**Tier 2 — partly size** (0.15 ≤ |r| < 0.30). Usable; know what they carry.
`lq_emp_55` (+0.299), `lq_emp_22` (−0.291), `has_economy_section` (+0.267),
`lq_emp_61` (+0.255), `lq_emp_62` (+0.244), `lq_emp_42` (−0.230), `lq_emp_99`
(−0.222), `low_postsecondary_ed` (−0.216), `lq_emp_81` (+0.207),
`low_employment` (−0.197), `sec_has_manufacturing` (+0.163).

**Tier 3 — effectively size-free** (|r| < 0.15). These transfer cleanly to a
rate target.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `has_tribal_land` | A | −0.005 | 0.0% | 3,143 |
| `has_oil_gas` | A | +0.009 | 0.0% | 3,143 |
| `has_mining` | A | −0.018 | 0.0% | 3,143 |
| `has_timber` | A | −0.019 | 0.0% | 3,143 |
| `has_protected_land` | A | +0.020 | 0.0% | 3,143 |
| `lq_emp_23` (Construction) | B | −0.024 | 0.1% | 2,469 |
| `lq_emp_71` (Arts/Rec) | B | +0.029 | 0.1% | 2,243 |
| `has_interstate` | A | +0.034 | 0.1% | 3,143 |
| `capital_to_wage_ratio` | E | **+0.040** | 0.2% | 3,143 |
| `has_agriculture` | A | +0.047 | 0.2% | 3,143 |
| `retirement_destination` | F | +0.052 | 0.3% | 3,134 |
| `has_manufacturing` | A | +0.066 | 0.4% | 3,143 |
| `unemployment_velocity` | C | **+0.075** | 0.6% | 3,143 |
| `has_logistics` | A | +0.076 | 0.6% | 3,143 |
| `n_industry_mentions` | A | +0.079 | 0.6% | 3,143 |
| `has_tourism` | A | +0.091 | 0.8% | 3,143 |
| `has_river` | A | +0.092 | 0.9% | 3,143 |
| `lq_emp_52` (Finance) | B | +0.094 | 0.9% | 2,638 |
| `has_namesake` | A | −0.095 | 0.9% | 3,143 |
| `gdp_velocity_pct` | C | **+0.095** | 0.9% | 3,080 |
| `sec_has_tourism` | A | +0.098 | 1.0% | 3,143 |
| `has_military_base` | A | +0.104 | 1.1% | 3,143 |
| `has_port` | A | +0.105 | 1.1% | 3,143 |
| `sec_n_industry_mentions` | A | **+0.108** | 1.2% | 3,143 |
| `lq_emp_48-49` (Transport) | B | +0.116 | 1.3% | 2,160 |
| `lq_emp_44-45` (Retail) | B | +0.117 | 1.4% | 3,050 |
| `lq_emp_51` (Information) | B | +0.137 | 1.9% | 1,914 |
| `persistent_poverty` | F | −0.139 | 1.9% | 3,111 |
| `lq_emp_31-33` (Manufacturing) | B | −0.139 | 1.9% | 2,570 |
| `lq_emp_72` (Accommodation/Food) | B | +0.145 | 2.1% | 2,369 |
| `has_university` | A | +0.145 | 2.1% | 3,143 |

Source A's remaining columns — `sec_has_logistics` (+0.087), `sec_has_agriculture`
(+0.056), `sec_has_oil_gas` (−0.032), `sec_has_mining` (+0.028),
`sec_has_timber` (+0.008) — also sit in Tier 3.

**Source A's typed block is the cleanest large block in this table under a rate
target.** Of its 29 columns, **23 fall in Tier 3, two in Tier 2, and four in Tier
1.** Of those four, `has_metro_attachment` is already ablated from the
cross-pillar sweeps as a restatement of Source F's `metro_2023`
(`source-a-findings.md` §16.2), `founding_year` fires on only 1,214 counties, and
`content_length` is the retired incumbent scalar, which the block retains as one
column among 29 rather than as the pillar's whole representation.

The distinction that matters is *within* the block. Its size-loaded columns are
the structural ones — how long the article is, how many proper nouns it contains,
whether it has an economy section — and its size-free columns are the ones that
name economic facts. `sec_n_industry_mentions`, at +0.108, is the single column
responsible for 97.6% of the section gain (§14.3). **Under a rate target the
part of Source A that carries the signal is also the part that survives the size
control**, which is the strongest evidence in this document that the
rate-versus-count answer moves Source A rather than leaving it where it was.

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

**Caveat on the size proxy.** `num_returns` is a Source E column, so E's
apparent independence (0.040) is partly self-referential — though
`capital_to_wage_ratio` is a dollar ratio and `num_returns` is a filer count, so
they are different constructs. Swapping in Census population
(`PROJECT_GOAL.md` next-work item 2) is a prerequisite for trusting this table
in full.

## Refeaturization implied by a rate target

| Pillar | Action | Rationale |
|---|---|---|
| **A** | **Ship the 29 typed columns, not the `content_length` scalar alone** | Reverses this document's original row, which predated `source-a-findings.md` §13–§17. The embedding stays cut; the text source does not. 23 of the 29 columns are Tier 3, and the signal-carrying `sec_n_industry_mentions` sits at +0.108 against `content_length`'s +0.359 — so a rate target *strengthens* Source A's case rather than confirming a cut. Exclude `has_metro_attachment` (+0.541, duplicates F's `metro_2023`), treat `founding_year` (−0.398, n = 1,214) as low-coverage, and annotate `content_length` and `n_distinct_proper_nouns` as the block's size-loaded columns. |
| **B** | Ship the 20-dim LQ vector; annotate per-column size loading | LQs are **not** uniformly size-neutral. `lq_emp_11`/`lq_emp_54` are urbanicity axes; `lq_emp_23`/`lq_emp_71` are clean. Both usable — the point is knowing which is which. |
| **C** | Ship as-is | Both velocities are near-orthogonal to size. |
| **D** | **Normalize to a rate** — `tons_per_return`, `tons_per_capita` | Highest-value single change in this document. A raw total cannot work against a per-request target. Recheck `out_partner_hhi` (0.33) as well; a concentration index should not carry that much size. |
| **E** | Ship as-is | Cleanest pillar under this target. |
| **F** | Demote `metro_2023`, `population_loss`, `housing_stress`; keep the rest | `metro_2023` (0.596) duplicates population, which the downstream model very likely already holds. `population_loss` (−0.393) and `housing_stress` (+0.349) are also size-loaded — the "distress flags are size-light" reading only holds for `persistent_poverty` (−0.139), `low_employment` (−0.197), and `retirement_destination` (+0.052). |

## Modeling traps specific to this target

### 1. Grain mismatch — the Mantel problem one level up

Features are county-level (n = 3,143). The target is per-request. Joining county
features onto request rows produces millions of rows carrying only 3,143
distinct feature values.

**Effective sample size is the county count, not the request count.** A model
reporting significance on request-level rows will report absurdly tight
confidence intervals, for exactly the reason a naive correlation test on 3.9M
county *pairs* did in the Source A analysis (see
`analyze_source_a_clusters.mantel_test`).

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

| Pillar | Vintage |
|---|---|
| A | continuous (scrape date) |
| B | QCEW 2025 |
| C | unemployment 2025, GDP 2024 |
| D | FAF `tons_2022` |
| E | IRS TY 2022 |
| F | USDA 2023 typology |

Three years of spread presented as one county snapshot, with no `as_of_date`
column anywhere. Acceptable for a static feature layer; a leakage vector if a
downstream model backtests against pre-2025 outcomes, since B and C would carry
future information. The consuming team cannot detect this without the column.

## Validation plan under this placeholder

1. **Confirm grain with the downstream team** — per-request rows, or
   county-aggregated revenue-per-request? Changes the modeling setup, not the
   feature work.
2. **Census population swap.** Unblocks trust in the size table above.
3. **Refeaturize D as a rate.** Turns a dead pillar into a possibly live one.
4. **Baseline: target ~ county size + population density only.** This is the bar
   every pillar must clear.
5. **Add pillars in cleanliness order** — C, E and the Source A typed block
   first, then the B LQ vector, then normalized D, then F distress flags.
   Permutation importance at each step, grouped CV throughout.
6. **Cut on the same standard applied to the Source A embedding.** Any pillar
   that fails to beat the size-only baseline goes, regardless of its cross-pillar
   r. Source A's embedding step was cut on exactly this standard and its text
   source survived the same test in typed form — the standard is not a proxy for
   a verdict about a source, only about a representation of it.

Expected outcome if the placeholder is roughly right: C, E, the B vector and
Source A's typed block clear the bar; D clears it only once normalized; F
contributes through `persistent_poverty` and `low_employment` rather than metro
status. Stated in advance so it can be wrong on the record.

## Invalidation conditions

Re-derive this document if the real target is:

- **A count or total** (total revenue, subscriber counts, request volume). Size
  becomes a feature, the raw `r` column becomes the operative one, D and F
  recover most of their apparent value, and Source A's `content_length` scalar
  becomes defensible again as a cheap size proxy.
- **Already size-normalized upstream** by the downstream team (e.g. they supply
  per-capita features themselves). Double-normalization would remove real signal.
- **Time-varying with a backtest window.** The vintage table above moves from a
  documentation gap to a blocking defect.
- **Defined at a grain finer than county** (household, address, H3 cell).
  `E_macro` is keyed on `fips_code`; a finer target makes these features
  group-level covariates and the clustering requirements in trap 1 become
  non-negotiable.

## Related

- `docs/PROJECT_GOAL.md` — open decision #1, which this document conditionally
  resolves.
- `analysis-output/E_macro_key_findings.ipynb` — section "What a downstream
  target would settle" presents this analysis for review.
- `outputs/pillar_pair_crossvalidation.csv` — the 50-test sweep this reinterprets.
- `analysis-output/source-a/source-a-findings.md` §13–§17 — the typed-extraction
  round that reinstated Source A. §14.2a and §17.3 carry the power figures behind
  the claim that the rate-versus-count answer settles more than another sweep
  could.
- `docs/plans/source_a_next_steps.md` — the five plans for the two items §17.3
  leaves open, and the questions that select among them.
- `docs/plans/downstream_target_question.md` — **the rate-versus-count question
  written out for the downstream team**, with what each answer changes across all
  six pillars. This document is the placeholder; that one is how to retire it.
