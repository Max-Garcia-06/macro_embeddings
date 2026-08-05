---
type: results-report
date: 2026-08-05
experiment_line: cross-source
round: 1
purpose: first non-circular validation, and the go/no-go evidence
status: active
---

# External Targets — Does `E_macro` Predict Anything Outside Itself?

## 1. Why this exists

Every validation in this repo before today was **pillar-versus-pillar**. The
50-pair crossvalidation sweep, the 29-target matrix sweep, and the Source A
marginal harness all predict one pillar's feature from the other five. That
design measures whether six federal sources agree with each other. It cannot
measure whether any of them is useful, for two reasons stated in
`docs/downstream_target.md` Part 2 and never resolved until now:

1. **Usefulness is relative to a target, and there was none.** Every target in
   those sweeps is another pillar's own feature.
2. **The circularity has a direction.** Scoring a source against its
   co-shipping pillars penalizes it precisely for agreeing with them. That is
   the correct penalty when assembling a non-redundant feature store and the
   wrong one when predicting an external outcome — Source A and Source F can be
   redundant with each other and both predictive of churn.

Three further pressures made this the next thing to build rather than a
someday item:

- **The stage ends in a go/no-go** (`docs/PROJECT_GOAL.md`). The deliverable is
  a validated claim about whether `E_macro` is worth continuing, and no
  pillar-versus-pillar number can support one.
- **A real downstream label is unobtainable**, structurally: the project has no
  access to company data and was scoped to public sources for that reason. That
  closes `docs/plans/source_a_next_steps.md` question 2 with "no", which per that
  document makes an external public target **mandatory rather than optional**.
- **The fixed-effect objection was unanswered.** The consumer joins on DMA and
  holds millions of impressions per market, so it can estimate a 210-level
  geographic fixed effect precisely and for free. Any static geo-keyed feature is
  exactly collinear with that effect by construction, which means no
  cross-sectional correlation anywhere in `analysis-output/` is evidence against
  it (`docs/plans/dma_regrain.md`, problem 1).

A fixed effect has exactly one weakness: **it has no parameter for a unit it has
never seen.** That is the seam this experiment is built around.

## 2. Design

`scripts/ingest_external_targets.py` pulls three county-level outcomes from the
ACS 2023 5-year summary file — keyless, since the Census data API now requires a
key and the `www2.census.gov` table-based files do not.

| target | ACS table | why this one |
|---|---|---|
| `broadband_rate` | B28002 | Closest public analogue to a FreeWheel-adjacent outcome. |
| `median_household_income` | B19013 | Economic level; the hardest case for beating a size baseline. |
| `median_age` | B01002 | Demographic texture, near-orthogonal to size. |

**Three rather than one, deliberately.** A single target would repeat a mistake
this repo already caught — the Source A headline of +0.0010 rests on a basket
that is 71% one BLS table, and §17.3 forbids publishing it without that
composition attached. One external target is a basket of one.

`scripts/analyze_external_target.py` scores four models as pooled out-of-fold R²
under **`GroupKFold` on `state_fips`**:

| model | predictors |
|---|---|
| `grand_mean` | intercept only — and exactly what a geo fixed effect predicts for a held-out unit |
| `size` | `log_population`, `log_agi`, `log_gdp_latest` |
| `emacro` | all 118 pillar features |
| `size_emacro` | both |

Two design choices, both deliberate departures from the repo's other sweeps:

- **State-blocked folds, not random `KFold`.** Every scored county sits in a
  state the model never trained on. Counties are spatially autocorrelated;
  random folds put a neighbour of every test county into train and inflate
  everything. Random `KFold` is defensible when the question is association and
  wrong when the question is extrapolation.
- **No state dummies in any design.** Under state-blocked folds a state fixed
  effect degenerates to the grand mean for held-out states — which is the whole
  argument, made concrete rather than asserted.

## 3. Headline result

**`E_macro` predicts all three external targets well above a size-only baseline,
on counties in states the model never saw.**

| target | `grand_mean` | `size` | `size+emacro` | lift | **lift, ablated** |
|---|---|---|---|---|---|
| `broadband_rate` | −0.003 | +0.421 | +0.511 | +0.091 | **+0.091** |
| `median_household_income` | −0.008 | +0.579 | +0.826 | +0.247 | **+0.154** |
| `median_age` | −0.005 | +0.227 | +0.483 | +0.256 | **+0.239** |
| **mean** | | | | +0.198 | **+0.161** |

`grand_mean` sitting at ≈0 is the sanity check passing: with whole states held
out, an intercept explains nothing, which is the fixed-effect model's position on
unseen geography.

**Read the ablated column, not the raw one.** Two pillar columns restate a target
by construction rather than predicting it, and are removed on the same principle
as `RESTATEMENT_COLUMNS` in `analyze_pillar_matrix_signal.py`:

- `wage_per_return_thousands` (Source E) is average wage income per tax return,
  which is close to a *definition* of median household income. Removing it drops
  that target's lift from +0.247 to **+0.154** — one column carried 38% of the
  apparent result.
- `retirement_destination` (Source F) is USDA's code for high net in-migration of
  people aged 60 and over, so it restates age structure. Removing it moves median
  age +0.256 → +0.239, a much smaller effect.

The defensible headline is **+0.161 mean lift over size, across three
independent public outcomes, extrapolating to unseen states.**

## 4. The row-count decomposition — the grain finding

The DMA penalty has two separable halves: **fewer rows** (210 versus 3,143) and
**aggregation** blurring within-market variation. Only the first is measurable
without the re-derivation layer in `docs/plans/dma_regrain.md` Phase 1B. It is
measured here by retraining on random county subsets, 10 reps each.

Mean lift over the size baseline by training-set size:

| n units | `broadband_rate` | `median_household_income` | `median_age` |
|---|---|---|---|
| **210** (DMA count) | **−0.058** (sd 0.073) | +0.048 (sd 0.215) | +0.218 (sd 0.089) |
| 400 | +0.015 | +0.178 | +0.206 |
| 800 | +0.050 | +0.205 | +0.230 |
| 1,600 | +0.074 | +0.226 | +0.227 |
| 3,000 | +0.096 | +0.240 | +0.235 |

Three things fall out:

1. **At the DMA row count, `E_macro` is worth roughly nothing on two of three
   targets** — and is actively negative for broadband adoption, the target
   closest to the consumer's domain.
2. **It is also unstable there.** Standard deviations at n = 210 run 0.07 to
   0.22 across reps; at n = 3,000 they fall to 0.01–0.04. At DMA cardinality the
   answer depends heavily on which units you happen to have.
3. **`median_age` is the exception and is essentially flat with n.** Whatever
   drives it is low-dimensional enough to be learned from 210 units.

**This supports joining at county grain, and it does so on evidence.** It does
not settle the question, because the aggregation half is unmeasured and could cut
either way — averaging 15 counties per market removes noise as well as signal.

## 5. Pre-registration scorecard

Three predictions were recorded in conversation before this ran. Scored honestly,
because a prediction that is only reported when it lands is not a prediction.

| prediction | outcome |
|---|---|
| Lift over the size baseline of roughly +0.03 to +0.08 | **Wrong, understated.** Actual ablated mean +0.161, roughly 2–5× the predicted range. |
| County grain shows larger lift than DMA | **Right, but for a different reason than given.** The mechanism claimed was thin units; the mechanism measured is row count. |
| The advantage concentrates in the low population deciles | **Wrong.** See below. |

### The thin-unit hypothesis is not supported

The county-grain argument as originally framed was that fixed effects fail on
thin units, so `E_macro` should help most in small counties. Proportional RMSE
reduction from `size` to `size+emacro`, by population decile:

| decile | median population | `broadband_rate` | `median_household_income` | `median_age` |
|---|---|---|---|---|
| 1 (smallest) | 2,631 | 0.112 | 0.438 | **0.016** |
| 5 | 21,833 | 0.094 | 0.249 | 0.336 |
| 10 (largest) | 468,058 | **0.232** | 0.405 | 0.208 |

No target shows the predicted pattern. Broadband is strongest in the *largest*
decile, income is U-shaped, and median age is *weakest* in the two smallest
deciles — the opposite of the hypothesis.

**One confound could explain part of this, and is not yet measured.** ACS 5-year
estimates carry large margins of error in small counties, so a share of the
target variance in decile 1 is sampling noise that nothing can predict. That caps
measurable performance in exactly the units the hypothesis is about. The
summary-file margin-of-error columns are available and this is quantifiable; it
has not been done, and until it is, **neither the hypothesis nor its rejection
should be reported as settled.**

## 6. What this settles, and what it does not

**Settles:**

- `E_macro` carries information about outcomes outside all six pillars, beyond
  county size, extrapolating to unseen states. First non-circular evidence in the
  project's life.
- The pillars are not "six independent noise sources plus a population
  variable" — a possibility the matrix-signal test named and could not rule out
  from inside the matrix.
- On the go/no-go: this is a **go** signal on the evidence available.

**Does not settle:**

- **The fixed-effect objection is answered by analogy, not directly.** The
  targets are public proxies, not the consumer's label, which is unobtainable.
  A confident answer to a slightly wrong question is the specific failure mode
  here.
- **Temporal transfer is untested.** All three targets are cross-sectional and
  single-period, and staleness resistance is one of the five things a fixed
  effect genuinely fails at.
- **The aggregation half of the DMA penalty is unmeasured** (Phase 1B).
- **Nothing here is measured at DMA grain**, only at DMA *cardinality* via
  subsampling. Those are different quantities.

## 7. Forbidden wording

On the same basis as `source-a-findings.md` §14.5.

- **Do not quote the +0.198 raw mean.** The ablated +0.161 is the defensible
  figure; one Source E column carried 38% of the income result.
- **Do not describe this as beating a DMA fixed effect.** It beats an intercept
  on held-out states using public proxies. Those are not the same claim.
- **Do not say `E_macro` helps most in small counties.** Measured, it does not,
  and the ACS margin-of-error confound means the reverse should not be asserted
  either.
- **Do not report the n = 210 subsample as "the DMA result."** It is the row
  count without the aggregation.

## 8. Next

1. **Quantify the ACS margin-of-error floor per population decile.** Decides
   whether §5's decile finding is real or an artifact, and it is the cheapest
   open item here.
2. Phase 1B, then rerun at true DMA grain — separates aggregation from row count.
3. A fourth and fifth target with different texture, once the harness exists the
   marginal cost is one constant.

## 9. Artifact index

- Ingest: `scripts/ingest_external_targets.py` → `data/external_targets.parquet`
- Harness: `scripts/analyze_external_target.py`
- `outputs/external_target_scores.csv` — per target, per model
- `outputs/external_target_by_decile.csv` — §5's decile breakdown
- `analysis-output/cross-source/external_target_stats.json` — summary plus the
  row-count sensitivity

```bash
uv run python scripts/ingest_external_targets.py
uv run python scripts/analyze_external_target.py
```
