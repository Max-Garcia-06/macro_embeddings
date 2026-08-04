# The One Question to Ask the Downstream Team

## Why this document exists

`E_macro` has six ingested pillars, a full cross-pillar validation sweep, and a
shipping decision on every source. What it does not have is a target variable.
`docs/downstream_target_assumptions.md` adopts one as a placeholder so that
pillar-selection can be reasoned about at all, and states plainly that every
conclusion in it is conditional.

One property of the real target decides more than everything else combined, and
it can be answered in a sentence by someone who has not yet chosen the metric.
This document is what to ask, why it matters that much, and what to do with each
answer.

---

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

---

## Why this one question

It decides whether county size is a **control** or a **feature**. That single
choice reverses conclusions across the whole repo, not just one pillar.

| | rate target | count target |
|---|---|---|
| county size | **control** — regress it out before fusion | **feature** — keep it |
| operative sweep column | `r_size_controlled` | raw `r` |
| Source D freight `log_tons` | dead — r = +0.871 with size | strong |
| Source F `metro_2023` | demote — r = +0.596 | strong |
| Source A `content_length` | compromised — r = +0.359 | defensible, cheap size proxy |
| Source A's 29 typed columns | **win** — 23 of 29 below \|r\| = 0.15 | interpretability only |
| Source C velocities, Source E ratio | cleanest pillars — r ≈ 0.04–0.10 | unremarkable |

Source: `outputs/feature_size_dependence.csv`, 62 features scanned against
`log10(num_returns)`. Reproduce with
`uv run python scripts/analyze_feature_size_dependence.py`.

### It also settles something no amount of further testing can

Whether Source A's 29 typed columns beat the `content_length` scalar they
replaced is powered at **0.39** and would need **91 targets** to resolve
statistically (`source-a-findings.md` §17.3). That comparison will not be settled
in this repo by adding targets.

Under a rate target it does not need to be. `content_length` sits at +0.359 with
county size; `sec_n_industry_mentions` — the single column carrying 97.6% of the
section gain — sits at +0.108. The typed block wins on construction rather than
on a p-value.

**One answer here resolves what 91 targets could not.** That is the strongest
argument for asking rather than for running another sweep.

---

## What each answer changes

### If the answer is "rate"

- County size becomes a control. Regress it out of every pillar before fusion.
- `r_size_controlled` becomes the operative scorecard; the raw `r` column becomes
  actively misleading and should not be quoted without its partner.
- **Source D needs refeaturization** — normalize to `tons_per_return` or
  `tons_per_capita`. This is the highest-value single change implied by a rate
  target: a raw total cannot work against a per-unit outcome.
- Demote Source F's `metro_2023`, `population_loss`, `housing_stress`.
- Source A ships the typed block, and the case for it is now structural rather
  than statistical.
- Add pillars in cleanliness order: C, E and Source A's typed block first, then
  the B LQ vector, then normalized D, then F's distress flags.

### If the answer is "count"

- Size is a legitimate feature and the raw correlations stand.
- D and F recover most of their apparent value.
- `content_length` is defensible again as a cheap size proxy, and the typed
  block's case rests on interpretability alone.
- Flag honestly that the project is then partly a population model, and that this
  is a choice rather than an accident.

### If the answer is "already size-normalized upstream"

Different failure mode, worth catching explicitly. If they supply per-capita
features themselves, double-normalization removes real signal. Ask whether
they normalize before or after joining county features.

---

## If they push back

**"Why does it matter — just give me all the features."**
You can, and the features ship either way. But 19 of the 50 cross-pillar
correlations lose more than half their effect once county size is controlled
(`outputs/pillar_pair_crossvalidation.csv`). If the target is a rate and the raw
numbers get taken at face value, the model gets built on structure that is mostly
population wearing a costume. The largest raw effect in the whole sweep — freight
tonnage against metro status, r = 0.495 — collapses to −0.057 under the control.

**"We haven't picked the target yet."**
The weaker version is still decisive: *is your outcome per-customer/per-request,
or a county total?* That is usually known long before the exact metric is.

**"Can't you just test both?"**
Both are already reported. The problem is not producing two numbers, it is that
the two point at opposite feature sets, and shipping a feature store means
committing to one. Publishing both without a decision pushes the decision onto
whoever consumes it, with less context than we have.

---

## Two more, if the conversation is going well

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
| C (velocity series) | 69% |
| D (freight) | 63% |
| B (QCEW location quotients) | 28% |
| F (typology) | 18% |
| E (capital-to-wage) | 0.2% |

Source A survives where no federal agency measures the same construct and is
absorbed where one does. The published headline of **+0.0010** is an average over
a basket that is **71% a single BLS table**, so it should never travel without
that composition attached.

---

## What not to ask

**Do not ask them to bless the feature set, or to confirm a p-value.** They
cannot, and it invites a decision they are not positioned to make. Source A's
typed block ships either way — it costs one regex pass and no model download.

**Question 1 is the only one whose answer changes what gets built.**

---

## Related

- `docs/downstream_target_assumptions.md` — the placeholder this question
  replaces, including the full size-dependence tiering and the invalidation
  conditions.
- `docs/PROJECT_GOAL.md` — open decision #1 is this question, stated from the
  repo's side.
- `docs/plans/source_a_next_steps.md` — the five plans this question selects
  among; it is question 4 there.
- `analysis-output/source-a/source-a-findings.md` §14.2a, §17.2a, §17.3 — the
  power figures and basket composition behind the claims above.
- `outputs/feature_size_dependence.csv` — every r-with-size figure quoted here.
