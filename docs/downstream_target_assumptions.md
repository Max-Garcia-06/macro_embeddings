# Downstream Target Assumptions

> **STATUS: PLACEHOLDER — NOT A REQUIREMENT FROM THE DOWNSTREAM TEAM.**
>
> No target variable has been supplied to this repo. This document adopts
> **revenue per request** as a working assumption so that pillar-selection
> decisions can be reasoned about instead of deferred. Every conclusion here is
> conditional on that assumption and must be re-derived if the real target
> differs in the one property that matters: whether it is a **rate** or a
> **count**. See [Invalidation conditions](#invalidation-conditions).

## Why a placeholder was needed

All validation in this repo to date is **pillar-versus-pillar**: 41 feature-pair
tests across 15 pillar pairs (`outputs/pillar_pair_crossvalidation.csv`). That
measures coherence and redundancy between sources. It cannot measure predictive
usefulness, because usefulness is defined relative to a target that does not
exist here.

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

Two consequences for how existing results are read:

1. The `r_size_controlled` column in the sweep is the operative scorecard.
2. The raw `r` column becomes actively misleading and should not be quoted
   without its size-controlled partner.

## Evidence: which features are size in disguise

Every pillar feature correlated against `log10(num_returns)`. This is the
diagnostic that determines whether a feature transfers to a rate target.

34 features scanned; **12 exceed |r| = 0.30 with county size.** Full table in
`outputs/feature_size_dependence.csv`.

**Tier 1 — size in disguise** (|r| ≥ 0.30). Contribute little to a rate target
beyond what population already supplies.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `log_tons` | D | **+0.871** | 76% | 3,143 |
| `metro_2023` | F | +0.596 | 36% | 3,143 |
| `lq_emp_11` (Agriculture) | B | −0.501 | 25% | 1,457 |
| `lq_emp_54` (Professional svcs) | B | +0.461 | 21% | 2,268 |
| `lq_emp_21` (Mining) | B | −0.419 | 18% | 1,026 |
| `lq_emp_56` (Admin/Waste) | B | +0.408 | 17% | 2,212 |
| `population_loss` | F | −0.393 | 15% | 3,126 |
| `content_length` | A | +0.356 | 13% | 3,143 |
| `housing_stress` | F | +0.349 | 12% | 3,143 |
| `in_partner_hhi` | D | +0.333 | 11% | 3,143 |
| `out_partner_hhi` | D | +0.329 | 11% | 3,143 |
| `lq_emp_53` (Real Estate) | B | +0.319 | 10% | 2,453 |

**Tier 2 — partly size** (0.15 ≤ |r| < 0.30). Usable; know what they carry.
`lq_emp_55` (+0.299), `lq_emp_22` (−0.291), `lq_emp_61` (+0.255), `lq_emp_62`
(+0.244), `lq_emp_42` (−0.230), `lq_emp_99` (−0.222), `low_postsecondary_ed`
(−0.216), `lq_emp_81` (+0.207), `low_employment` (−0.197).

**Tier 3 — effectively size-free** (|r| < 0.15). These transfer cleanly to a
rate target.

| Feature | Pillar | r with size | Shared variance | n |
|---|---|---|---|---|
| `lq_emp_23` (Construction) | B | −0.024 | 0.1% | 2,469 |
| `lq_emp_71` (Arts/Rec) | B | +0.029 | 0.1% | 2,243 |
| `capital_to_wage_ratio` | E | **+0.040** | 0.2% | 3,143 |
| `retirement_destination` | F | +0.052 | 0.3% | 3,134 |
| `unemployment_velocity` | C | **+0.075** | 0.6% | 3,143 |
| `lq_emp_52` (Finance) | B | +0.094 | 0.9% | 2,638 |
| `gdp_velocity_pct` | C | **+0.095** | 0.9% | 3,080 |
| `lq_emp_48-49` (Transport) | B | +0.116 | 1.3% | 2,160 |
| `lq_emp_44-45` (Retail) | B | +0.117 | 1.4% | 3,050 |
| `lq_emp_51` (Information) | B | +0.137 | 1.9% | 1,914 |
| `persistent_poverty` | F | −0.139 | 1.9% | 3,111 |
| `lq_emp_31-33` (Manufacturing) | B | −0.139 | 1.9% | 2,570 |
| `lq_emp_72` (Accommodation/Food) | B | +0.145 | 2.1% | 2,369 |

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
| **A** | No change (stays cut) | `content_length` at 0.356 with size confirms the cut on independent grounds. |
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
5. **Add pillars in cleanliness order** — C and E first, then the B LQ vector,
   then normalized D, then F distress flags. Permutation importance at each
   step, grouped CV throughout.
6. **Cut on the same standard applied to Source A.** Any pillar that fails to
   beat the size-only baseline goes, regardless of its cross-pillar r.

Expected outcome if the placeholder is roughly right: C, E, and the B vector
clear the bar; D clears it only once normalized; F contributes through
`persistent_poverty` and `low_employment` rather than metro status. Stated in
advance so it can be wrong on the record.

## Invalidation conditions

Re-derive this document if the real target is:

- **A count or total** (total revenue, subscriber counts, request volume). Size
  becomes a feature, the raw `r` column becomes the operative one, and D and F
  recover most of their apparent value.
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
- `outputs/pillar_pair_crossvalidation.csv` — the 41-test sweep this reinterprets.
