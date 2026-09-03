# Three experiments the notebook read-through asked for

Written 24 August 2026, after a read of `analysis-output/E_macro_pillar_worth_2026-08-13.ipynb`.

The notebook is internally consistent and unusually honest about its own limits.
The three items below are the ones a reader who believes it would still want
answered, in the order they should be run. All three are now implemented and
run; **Result** sections carry what came back.

## The asymmetry that motivated all three

The representation section applies two instruments to Source A that no other
part of the project applies to anything:

1. **a geography control** — re-scoring every arm against a baseline that
   already holds two columns of county-centroid latitude and longitude;
2. **bootstrap intervals** — on the basket mean, both naive and clustered by
   ACS table.

Both changed what that section concludes. The geography control moved Source A's
best encoder arm from +0.0164 to +0.0006, and the interval on the residual
covers zero. Neither instrument had ever been pointed at the **pillar-worth**
figure, which is where the +0.190 headline and all six pillar verdicts live.

That is one project applying two evidence standards, and the weaker one is
carrying the go/no-go decision.

---

## A stale artifact found on the way in

`ingest_external_targets.EXTERNAL_TARGETS` has carried **42** targets since
commit `6f145c0`. `analyze_external_target.py` loops over all of them, but its
committed artifact was produced before that widening and scores **5**. The
notebook's pillar-worth section is therefore running on a five-target basket
while a forty-two-target one sits unused in the same repo.

Re-running the script picks the wider basket up automatically, which would have
silently moved the headline. It does not, because:

- every summary statistic is now computed on the full basket **and** on the
  original five, the latter under `headline_basket`;
- the two are never read across, exactly as the notebook's evidence-basket table
  requires.

---

## 1. The geography control, on all six pillars

**Status: DONE.**

### Why

If two float columns reproduce 96% of Source A's measured encoder gain, the
question is not settled for Source A. It is *open for every pillar*, and the
five ACS proxies are all strongly spatial. Source B is location quotients,
Source D is freight, Source F is county typology — every one of them plausibly
carries position on the map as part of what it "knows".

The consuming team's position makes this the decision-relevant form of the
question. A DMA fixed effect encodes location by construction, so the number
that matters to them is not what a pillar adds to a size baseline; it is what a
pillar adds to somebody who **already knows where the county is**.

### What was done

A parallel `geo` family in `analyze_external_target.py`, carrying two centroid
columns in **both** the full and the reduced design:

    contribution_geo(P) = R2(size + lat/lon + all pillars)
                        - R2(size + lat/lon + all pillars except P)

Geography on both sides is deliberate and is a different construction from the
representation script's `contribution_geo`, which puts lat/lon in the reduced
model only in order to ask whether one arm beats coordinates outright. The
question here is not whether a pillar beats geography — it is what survives it.

`size_geo` is carried too, so the whole matrix's lift can be restated against a
geography-aware baseline rather than only each pillar's slice of it.

### Where

- `scripts/analyze_external_target.py` — `GEO_MODELS`, `GEO_FEATURES`,
  `ModelSpec.uses_geo`, `geo_control_summary`.
- `tests/test_drop_one_geo_control.py`.

### Result

See **Results** at the end of this document.

---

## 2. Intervals on the pillar-worth figure

**Status: DONE.**

### Why

The figure reports six means to four decimal places with nothing attached, on a
basket of five targets. Appendix A4 already concedes the consequence — "B's and
C's ordering (+0.0067, +0.0054) should not be quoted as settled" — which is a
statement about an interval, made without one. `-0.0000` for Source A is
carrying a shipping decision as a point estimate.

### What was done

`bootstrap_drop_one` resamples **targets**, the unit the reported mean is taken
over, paired across pillars within a replicate so that:

- `pairwise` gives an interval on each pillar's lead over each other pillar —
  the quantity A4 declines to quote — with shared target-level variance
  cancelling;
- `geo_minus_plain` gives an interval on how much of a pillar's contribution was
  geography, differenced on one draw rather than between two independent means.

Two schemes, matching the representation script: naive, and clustered on the ACS
table each target is drawn from. No model is re-fitted.

### Where

- `scripts/analyze_external_target.py` — `bootstrap_drop_one`,
  `_draw_target_positions`, `_interval`.
- `tests/test_drop_one_geo_control.py`.

### Result

See **Results** at the end of this document.

---

## 3. Temporal transfer

**Status: DONE.**

### Why

The limits section lists it as "Not started" and calls it "the one argument a
fixed effect cannot answer". Both are true, and together they make it the
highest-upside item on the list: if it works, the DMA objection stops being the
thing the project is blocked on.

The cross-sectional arm answers the fixed-effect objection through the *unseen
unit* seam — hold out whole states, and a fixed effect has no parameter for a
place it has never seen. That seam is real and it is the weaker one, because the
consumer sees almost every market almost all the time.

The second seam does not need the unit to be unseen. A fixed effect estimated on
history predicts, for a unit it knows well, that unit's own past level. **It has
no parameter for movement.**

### What was done

`scripts/analyze_temporal_transfer.py`, scoring the **change** in each ACS
target between two vintages:

    target      y(late) - y(early)
    lagged      y(early) -- the fixed effect's own prediction, and the control
                that absorbs mean reversion
    headline    R2(lagged + size + emacro) - R2(lagged + size)

`lagged` sits in every design including the baselines. Change is mechanically
anti-correlated with the level whenever the level is measured with error, and
ACS county estimates in small counties are measured with a lot of it; without
`y(early)` in the baseline, any feature correlated with the level would score as
a predictor of change through regression to the mean alone.

The geography control from item 1 is carried through: the same lift is also
reported against a baseline that already holds lat/lon.

**Data.** `ingest_external_targets` gained a vintage parameter. The table-based
ACS summary file exists for 2021–2024 only — earlier years publish the
sequence-based format, which is a different parse — so the widest available pair
is **ACS 2021 (2017–2021)** against **ACS 2024 (2020–2024)**. Both were fetched
and cached; each carries all 42 targets.

### Three things this deliberately does not claim

**It is not an out-of-time forecast.** The pillar features are one current
vintage and sit between the two target vintages. A real forecast needs feature
vintages predating the early target, and only Source E has a panel deep enough
(`source_e_irs_soi_panel.parquet`, TY2018–TY2022). That is an ingest problem,
not an analysis one, and it is the obvious next step rather than something this
script approximates quietly. Reporting change-prediction as forecasting would be
the same class of error the geography control caught in the Source A work: a
number measured correctly, answering a different question from the one its label
implies.

**The two vintages share sample.** Five-year windows three years apart overlap by
two years, so the measured change is attenuated. That biases *against* the
hypothesis, which means a null here is weaker evidence of "no temporal signal"
than a positive result is of "there is one" — and the writeup has to say so in
that direction and not the other.

**Differencing amplifies noise.** The sampling noise in a difference is
`se_early^2 + se_late^2`, against a change variance far smaller than either
level's. `noise_share_of_change` is reported per target, and targets above
`MAX_NOISE_SHARE = 0.75` are excluded from the headline mean and listed, so a
low R2 can be read as "unpredictable by anyone" rather than "E_macro failed".

### Where

- `scripts/ingest_external_targets.py` — `ACS_YEAR`, `SUPPORTED_ACS_YEARS`,
  `vintage_cache_path`, a `year` parameter threaded through `_download_table`,
  `_fetch_table_uncached`, `_derive` and `fetch_external_targets`.
- `scripts/analyze_temporal_transfer.py`.
- `tests/test_temporal_transfer.py`, `tests/test_external_targets.py`.

### Result

See **Results** at the end of this document.

---

## One bug caught during implementation

`contribution` is defined throughout `analyze_external_target.py` as *reference
minus self*, which states a withheld block's worth because `self` is the
**reduced** model. Both new "add the pillars" arms — `size_geo_emacro` and the
temporal script's two pillar arms — are the **fuller** model, so the same
subtraction reports the negative of their lift.

Fixed by giving those arms no `reference` and differencing them in the correct
direction in their own column.
`test_the_two_pillar_arms_state_a_lift_rather_than_a_contribution` and
`test_geo_control_summary_reports_share_retained` hold the convention in place.

---

## A runtime note, so the next person does not lose an hour to it

The first full run of `analyze_external_target.py` appeared to take ~20 minutes
per target and was on course for roughly 14 hours. It is not slow: a stale
process from an earlier attempt was still holding the cores, and the two runs
were oversubscribing BLAS threads against each other. Cleanly, one target costs
about 18 seconds — 3s for the models, 11s for the placebo, 4s for the
training-size sweep — and the full 42-target sweep runs in about 15 minutes.

**Check `ps aux | grep analyze_` before concluding anything about this repo's
runtime.**

---

## Results

**Nothing in the notebook moved.** The headline basket reproduces exactly what
the pillar-worth section already states: mean lift **+0.1897** over size, A at
**−0.0000**, E **+0.0582**, F **+0.0413**, D **+0.0191**, B **+0.0067**, C
**+0.0054**. That was the point of splitting the artifact, and it is also the
check that the widening did not disturb the existing code path.

### 1. The geography control — the matrix survives it; two pillars do not, quite

On the wide basket (41 of 42 scored, 28 ACS tables):

| | value |
|---|---|
| matrix lift over size | **+0.1834** |
| matrix lift over size + lat/lon | **+0.1619** |
| lat/lon alone, over size | **+0.0361** |

**This is the good news and it is worth stating first.** The Source A result did
*not* generalise. Two coordinate columns reproduce 96% of Source A's best
encoder arm; they reproduce roughly a fifth of the whole matrix's lift, and the
matrix keeps **88%** of its advantage once geography is already in the baseline.
`E_macro` as a whole is not a geography proxy.

Per pillar, contribution and the same contribution net of latitude and
longitude, with 95% table-clustered intervals:

| Pillar | contribution | net of lat/lon | kept |
|---|---|---|---|
| F · Structural resilience | +0.0437 [+0.0305, +0.0595] | +0.0394 [+0.0284, +0.0535] | 90% |
| E · Capital flow | +0.0416 [+0.0314, +0.0529] | +0.0386 [+0.0275, +0.0517] | 93% |
| D · Trade logistics | +0.0139 [+0.0083, +0.0209] | +0.0113 [+0.0049, +0.0208] | 81% |
| B · Industrial core | +0.0109 [−0.0007, +0.0230] | +0.0066 [−0.0030, +0.0161] | 60% |
| C · Economic velocity | +0.0033 [+0.0010, +0.0061] | +0.0028 [−0.0011, +0.0077] | 85% |
| A · Place identity | +0.0031 [−0.0006, +0.0078] | +0.0020 [−0.0019, +0.0066] | 64% |

**E and F carry something other than position on the map**, keeping 90%+ of
their contribution under the control, and both clear zero comfortably on both
sides of it. **B loses 40%** and its interval covers zero before the control as
well as after, which is the one result here that should change how B is
described. C's contribution survives in point terms but stops clearing zero once
geography is held.

### 2. The intervals — the ordering is three tiers, not six ranks

Fifteen pairwise differences, wide basket, table-clustered. Ten separate; five
do not:

- **not separated:** E−F (−0.0021 [−0.0241, +0.0183]), B−D, B−C, A−B, A−C
- **separated:** every pair spanning the {E, F} / {B, D} / {A, C} tiers

So the honest reading of the figure is **three tiers**: E and F far ahead and
indistinguishable from each other; D and B in the middle; A and C at the bottom
and indistinguishable from each other. The six-way rank the figure invites is
not supported.

Three specific consequences:

**A4's caution was right and understated.** On the headline basket B−C is
**+0.0013 [−0.0102, +0.0129]** — the appendix says the ordering "should not be
quoted as settled", and the interval is roughly ten times the gap.

**"Source F, second of the six" is not a claim the basket supports.** E−F on the
headline basket is +0.0169 [−0.0333, +0.0632], and on the wide basket F actually
edges E. The rank is noise; what is real is that both are far ahead of the other
four. A5's wording needs to say "with E, far ahead of the rest" rather than
"second".

**Source A's −0.0000 now has an interval, and it covers zero:** [−0.0028,
+0.0039] on the headline basket, [−0.0006, +0.0078] on the wide one. That
supports the section's existing reading — "redundant, not broken" — and upgrades
it from a point estimate to a bounded one. Note the wide-basket point estimate is
mildly positive; it still does not clear zero.

### 3. Temporal transfer — inconclusive, and the reason is measurable

**The instrument is not sensitive enough, and that is the finding.** Of 42
targets, **36 were dropped because their differenced sampling noise exceeds 75%
of their change variance** — many by a wide margin, several above 100%, one at
458%. Two ACS 5-year vintages three years apart share two years of sample, and
differencing adds their variances; for most county-level ACS constructs there is
simply no measurable change left after that.

Six targets survived: `broadband_rate`, `median_contract_rent`,
`median_gross_rent`, `median_home_value`, `median_monthly_housing_cost`,
`work_from_home_share`. On those:

| model | R2 on the change |
|---|---|
| a geographic fixed effect | 0.0000, by construction |
| the county's own earlier level | +0.3778 |
| lagged + size | +0.4318 |
| lagged + size + E_macro | +0.4342 |

**Headline lift +0.0023 [−0.0147, +0.0198]**, positive on 3 of 6 targets; net of
lat/lon +0.0003. The interval covers zero comfortably. **This does not show that
`E_macro` fails at temporal transfer, and it must not be written up as though it
does** — a six-target basket of mostly housing-cost measures, attenuated by
two years of shared sample, cannot resolve a lift of this size in either
direction.

**One result inside it is worth keeping.** The aggregate is flat because the
pillars disagree, not because none of them moves:

| Pillar | contribution to the change | positive on | shuffled max |
|---|---|---|---|
| E · Capital flow | **+0.0086** [+0.0030, +0.0152] | 5 of 6 | +0.0018 |
| F · Structural resilience | **+0.0060** [+0.0029, +0.0084] | 5 of 6 | +0.0041 |
| D · Trade logistics | **+0.0052** [+0.0013, +0.0107] | 5 of 6 | +0.0011 |
| C · Economic velocity | −0.0057 [−0.0078, −0.0036] | 0 of 6 | +0.0021 |
| A · Place identity | −0.0038 [−0.0050, −0.0024] | 0 of 6 | +0.0021 |
| B · Industrial core | −0.0132 [−0.0190, −0.0091] | 0 of 6 | +0.0056 |

D, E and F each clear their own shuffled noise floor and are positive on 5 of 6
targets; A, B and C are negative on 6 of 6 and drag the total to zero. **The same
three pillars that survive the geography control are the three that predict
change.** Two independent tests, run for different reasons, agreeing on the same
partition of the matrix is worth more than either on its own — and it is a
hypothesis for the next round, not a result, because both run on baskets this
small.

### What would make item 3 conclusive

Not another pass at this design. Two options, in order of cost:

1. **A longer baseline.** ACS 2021 against ACS 2024 is the widest pair the
   table-based summary file supports. The 2019 and earlier vintages exist only
   in the sequence-based format, which is a different parse — a day of ingest
   work, and it buys five clear years with no shared sample.
2. **A real out-of-time forecast.** Feature vintages predating the early target.
   Source E's panel already supports it (TY2018–TY2022); the other five are
   single snapshots, so this is an ingest question for B, C, D and F.

Option 1 is the one that would settle whether the D/E/F pattern above is real.
