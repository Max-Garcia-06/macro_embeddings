---
type: results-report
date: 2026-08-05
experiment_line: cross-source
round: 2
purpose: first non-circular validation, and the go/no-go evidence
status: active
---

> **Round 2 (2026-08-05).** §1–§9 are round 1 as written. §10–§16 add two
> targets, quantify the ACS sampling-noise floor §5 left open, and measure the
> aggregation half of the grain penalty §4 left unmeasured. **§12 reverses round
> 1's grain conclusion** — aggregation helps, and roughly cancels the row-count
> loss. Read §14 before quoting anything from §4.

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

---

# Round 2 — Noise floor, five targets, and the aggregation arm (2026-08-05)

Three items closed, and **one of them reverses a conclusion from round 1.**

## 10. The five-target headline

Two targets added, on the same argument that motivated three rather than one:
`median_home_value` (B25077, asset rather than income flow) and
`mean_commute_minutes` (B08013 over B08012, settlement geometry). Neither has a
pillar column measuring its construct, so neither carries an ablation.

| target | `size` | `size+emacro` | **lift, ablated** |
|---|---|---|---|
| `broadband_rate` | +0.421 | +0.511 | **+0.091** |
| `median_household_income` | +0.579 | +0.826 | **+0.154** |
| `median_age` | +0.227 | +0.483 | **+0.239** |
| `median_home_value` | +0.488 | +0.721 | **+0.234** |
| `mean_commute_minutes` | +0.149 | +0.382 | **+0.232** |
| **mean** | | | **+0.190** |

Five of five positive. The headline moves +0.161 → **+0.190** with the basket
widened, which is the direction that matters: broadening a basket usually dilutes
a result, and this one strengthened.

## 11. The sampling-noise floor, and what it does to §5

ACS publishes a 90% margin of error per estimate, and these are now ingested
alongside the values (`{column}_se`, from the summary file's `_M` columns, using
the General Handbook's proportion and ratio formulas). A share of each target's
variance is sampling error no model can explain, and it is concentrated exactly
where §5's open question sat.

Mean across the five targets, by county population decile:

| decile | median pop | noise share | R² ceiling | R² `size` | R² `size+emacro` | share of explainable |
|---|---|---|---|---|---|---|
| 1 (smallest) | 2,631 | **29.7%** | 0.703 | **−0.474** | +0.140 | 0.213 |
| 2 | 6,762 | 10.3% | 0.897 | +0.228 | +0.421 | 0.467 |
| 5 | 21,833 | 6.0% | 0.940 | +0.378 | +0.614 | 0.648 |
| 9 | 144,842 | 1.5% | 0.985 | +0.252 | +0.567 | 0.573 |
| 10 (largest) | 468,058 | 0.6% | 0.994 | +0.137 | +0.517 | 0.520 |

**The confound was real and it does not rescue the hypothesis.** Nearly 30% of
the smallest decile's variance is sampling noise against 0.6% in the largest, so
small counties genuinely are harder to predict for reasons having nothing to do
with `E_macro`. But correcting for it — the `share of explainable` column — leaves
the pattern intact: 0.213 in the smallest decile against 0.520 in the largest,
peaking mid-distribution. **`E_macro` does not capture more of the available
signal in thin counties.**

One finding here is worth more than the hypothesis it failed to save. **A
size-only model is actively harmful on the smallest counties: R² = −0.474**,
worse than predicting the decile's own mean. A global size relationship
extrapolates badly to the bottom of the distribution, and `E_macro` pulls it back
to +0.140. Whatever a consumer holding only population would do, it fails hardest
exactly where a feature layer is supposed to help.

**Two metrics disagree here and both are in the artifacts.** Proportional RMSE
reduction is flat-to-rising across deciles; R² lift over size is largest in
decile 1 (+0.614 mean). They diverge because decile 1's variance is enormous, so
a small proportional error reduction is a large variance-explained gain. The R²
lift is also carried by two targets — `median_household_income` (+1.316) and
`median_home_value` (+1.416) — while the other three sit between +0.034 and
+0.212. That is the same concentration problem `source-a-findings.md` §14.5
forbids publishing without composition. **Neither metric supports a general
thin-unit claim.**

Both dollar-denominated targets are the ones favouring small counties, and the
rate, the demographic median and the physical quantity all favour large ones.
Recorded as a pattern worth a hypothesis, not as a finding.

## 12. Aggregation helps — a correction to round 1

§4 measured the row-count half of the DMA penalty and left the aggregation half
explicitly unmeasured, with the caveat that it "could cut either way." It cuts,
and it cuts the other way.

`scripts/geo_aggregate.py` re-derives the matrix at a coarser geography under the
rule in `dma_regrain.md` §3 — aggregate the inputs, not the outputs. Groups are
k-means clusters of county centroids at Nielsen cardinality: 208 usable groups,
median 16 counties each, matching a DMA in count and character. **They are not
DMAs**; that delineation is proprietary. Three arms, same five targets:

| target | `county_full` | `county_subsample` (n=208) | `market_aggregate` (n=208) | aggregation effect |
|---|---|---|---|---|
| `broadband_rate` | +0.091 | −0.051 | **−0.110** | −0.059 |
| `mean_commute_minutes` | +0.232 | +0.144 | +0.058 | −0.087 |
| `median_age` | +0.256 | +0.206 | **+0.358** | +0.153 |
| `median_household_income` | +0.247 | +0.030 | **+0.271** | +0.242 |
| `median_home_value` | +0.234 | +0.124 | **+0.404** | +0.281 |
| **mean** | +0.212 | +0.091 | +0.196 | **+0.106** |

- **Row-count effect: −0.122.** Fewer rows hurt, as §4 found.
- **Aggregation effect: +0.106.** Aggregation *helps*, and nearly cancels it.

**On three of five targets the aggregated 208-market arm matches or beats full
county grain.** Median home value goes +0.234 → +0.404. The mechanism is
plausible rather than artefactual: population-weighted aggregation converts
sparse, noisy county columns — suppressed LQ cells, single-article Wikipedia
booleans — into stable continuous shares, and does the same to the target.

### What this does to the county-grain argument

It weakens it substantially, and that has to be said plainly rather than
qualified into survival.

The argument as recorded in `dma_regrain.md` and `PROJECT_GOAL.md` is that DMA
grain is the worse answer. On this evidence it is **worse only on two of five
targets** — and the two are `broadband_rate` and `mean_commute_minutes`. That
`broadband_rate` is among them matters more than a 2-of-5 tally suggests, since
it is the target closest to the consumer's domain and it is the one where the
market arm goes outright negative. But that is one target, and one target is a
basket of one.

**The defensible position is now: the grain question is open, and county is not
established as better.**

### Two biases, both favouring the aggregate arm

Neither is corrected, so the +0.106 should be read as an upper bound:

1. **The aggregated target is cleaner.** Averaging five ACS estimates over ~16
   counties removes much of the sampling error §11 quantifies. The market arm
   predicts a less noisy outcome than the county arms do.
2. **k-means markets are spatially compact by construction**, so within-market
   homogeneity is near the maximum for a county grouping of that size. Real DMAs
   are drawn on media-market boundaries, not economic ones.

### And one bias against it

**62 of the 118 pillar columns are approximated rather than re-derived** at group
grain — Source B's 40 location quotients chief among them. The aggregate arm is a
lower bound in that respect. Which brings the cost finding:

## 13. Source B cannot be aggregated from its shipped parquet

`data/source_b_qcew.parquet` carries `lq_emp_{naics2}` and
`disclosure_{naics2}` and **no employment counts**. A group-level location
quotient is summed employment by sector over summed total employment against the
national base; without `emp` there is nothing to sum, so the 40-column widest
block in the matrix can only be population-weighted, which is the operation §3 of
the plan forbids.

Source F is approximated for a different reason — its typology flags are
categorical classifications with no underlying quantity — and Source D's two HHIs
need the partner-flow table.

`dma_regrain.md` Phase 1B estimated 1–2 days assuming every pillar could be
rebuilt from its parquet. **Source B cannot, and fixing it is a change to
`ingest_source_b.py` to carry `emp` alongside `lq_emp`, plus a re-download of the
~2.2GB QCEW singlefile.** Add roughly a day, and note the estimate was wrong
rather than quietly absorbing it.

## 14. Revised scorecard

| round-1 prediction | round-2 verdict |
|---|---|
| Lift of +0.03 to +0.08 | Still wrong, still understated. Five-target ablated mean **+0.190**. |
| County grain beats DMA | **Now doubtful.** Row count −0.122, aggregation +0.106; they roughly cancel, and 3 of 5 targets prefer the aggregate. |
| Advantage concentrates in small counties | **Still wrong**, and the noise floor does not rescue it. Share of explainable variance is 0.213 in the smallest decile against 0.520 in the largest. |

## 15. Forbidden wording, round 2

Additions to §7, which stands.

- **Do not say the aggregation half of the grain penalty is unmeasured.** It is
  measured, at +0.106, and it points the opposite way from the row-count half.
- **Do not quote the aggregate arm as a DMA result.** k-means clusters at
  Nielsen cardinality are a mechanism stand-in, not the delineation.
- **Do not claim county grain is established as better.** On five public targets
  it is better on two.
- **Do not report the R² lift by decile without its composition.** Two
  dollar-denominated targets carry the decile-1 mean.

## 16. Round 2 artifact index

- `scripts/geo_aggregate.py` — re-derivation rules and the provenance map
- `scripts/analyze_grain_effect.py` — the three-arm comparison
- `outputs/grain_effect.csv`, `analysis-output/cross-source/grain_effect_stats.json`
- `outputs/external_target_by_decile.csv` — now carries `noise_share`,
  `r2_ceiling`, `share_of_explainable` and `r2_lift_over_size`

```bash
uv run python scripts/ingest_external_targets.py
uv run python scripts/analyze_external_target.py
uv run python scripts/analyze_grain_effect.py
```
