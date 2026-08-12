---
type: results-report
date: 2026-08-12
experiment_line: cross-source
round: 1
purpose: settle Source F's slot on a fair test, and score every pillar the same way
status: active
---

# Drop-One — What Is Each Pillar Actually Worth?

## 1. Why this exists

`docs/pillar_status.md` names Source F as the one pillar falling short. Its only
strong raw correlation in the 15-pillar-pair sweep — against Source D freight
tonnage, r = 0.495, the largest raw effect anywhere in that sweep — collapses to
**r = −0.057** once county size is controlled. `docs/PROJECT_GOAL.md` resolves
that by keeping F and reclassifying it as a "structural anchor," justified by
what the typology definitionally is rather than by measured performance.

`pillar_status.md` states the problem with that resolution plainly: against an
operating principle that says every pillar earns its slot on evidence, the
reclassification is close to a rationalization for keeping a pillar that failed
its own test. It also names the fix, and this report is that fix:

> Raw/controlled correlation is the wrong lens for a categorical structural
> variable — test whether F explains residual variance after B/C/D/E are already
> in the model, rather than raw pairwise correlation.

Two design decisions follow from taking that seriously.

**Every pillar takes the same test.** A test only the suspect sits proves
nothing in either direction, and the symmetric table is what a go/no-go deck
needs anyway.

**The verdict rests on the external arm.** Pillar-versus-pillar lift measures
coherence, not usefulness, and `analyze_pillar_matrix_signal.py` already records
why a low internal number cannot condemn a pillar: being unpredictable from the
other five is also exactly what an independent information source looks like.
So the same drop-one design runs against the five public ACS targets, which is
the only non-circular evidence this project can produce.

## 2. Pre-registration

Recorded in the implementation plan before either script was run, and reproduced
here verbatim:

> F ships as a pillar if its marginal contribution — R²(size + all pillars) −
> R²(size + all pillars except F), pooled out-of-fold over the five external ACS
> targets, with restatement columns ablated — is positive on a majority of
> targets and above the shuffled-feature noise floor. Otherwise `E_macro` ships
> five pillars and the go/no-go deck says so plainly.

Nothing about the rule was renegotiated after the numbers arrived. It is stated
first here for the same reason it was written first: the failure mode this whole
exercise exists to avoid is choosing the justification after seeing the result.

## 3. Design

Two arms, same shape, different targets.

**Arm 1 — internal** (`scripts/analyze_pillar_block_marginal.py`). For each of
the 29 targets in the matrix sweep, owned by pillar Q, and each pillar P ≠ Q:

```
reduced = size + state fixed effects + every block except Q and P
full    = reduced + block P
lift_P  = R2_out-of-fold(full) - R2_out-of-fold(reduced)
```

145 (target, block) pairs, 5-fold CV, 49-rep row-shuffle null, one
Benjamini-Hochberg correction across the sweep.

**Arm 2 — external** (`scripts/analyze_external_target.py`, extended). Same
subtraction, against five ACS outcomes outside all six pillars, under
`GroupKFold` on `state_fips` so every scored county sits in a state the model
never trained on:

```
contribution(P) = R2(size + all pillars) - R2(size + all pillars except P)
```

The noise floor is the same permutation logic: the withheld block is added back
with its rows shuffled, breaking county alignment while preserving each column's
distribution and the design's width, 20 reps per pillar per target.

Both arms report lift twice, raw and with `RESTATEMENT_COLUMNS` held out. That
ablation is load-bearing here. USDA builds Source F's industry-dependence flags
from industry employment and earnings shares, which is what Source B's location
quotients measure, so an unablated F number would credit F for restating BLS.

**Reproduction check before anything else.** The extended external script
reproduces the published round-2 headline exactly: mean ablated lift over size
**+0.1897** against the +0.190 in `external-target-findings.md` §10, and all
five per-target lifts match (+0.091, +0.154, +0.239, +0.234, +0.232). The
drop-one numbers below are therefore comparable to that published round rather
than to a moved panel.

## 4. Headline: Source F is the second most valuable pillar

Arm 2, mean contribution across the five ACS targets, restatements ablated:

| pillar | mean contribution | positive on | above noise floor | mean placebo |
|---|---|---|---|---|
| **E** Capital Flow | **+0.0582** | 5/5 | 5/5 | −0.0018 |
| **F** Structural Resilience | **+0.0413** | 5/5 | 5/5 | −0.0027 |
| **D** Trade Logistics | +0.0191 | 5/5 | 5/5 | −0.0017 |
| **B** Industrial Core | +0.0067 | 3/5 | 4/5 | −0.0072 |
| **C** Economic Velocity | +0.0054 | 5/5 | 4/5 | −0.0006 |
| **A** Place Identity | −0.0000 | 2/5 | 2/5 | −0.0037 |

The noise floor sits where it should: every pillar's shuffled block *hurts* the
fit slightly, and the largest apparent contribution any shuffle produced across
all 30 pillar × target cells is **+0.0031**. F's +0.0413 is thirteen times that.

Per target, F's contribution never goes negative:

| target | E | F | D | B | C | A |
|---|---|---|---|---|---|---|
| `broadband_rate` | +0.0404 | +0.0368 | +0.0141 | +0.0035 | +0.0039 | −0.0026 |
| `median_household_income` | +0.0802 | +0.0149 | +0.0167 | +0.0231 | +0.0035 | −0.0032 |
| `median_age` | +0.0305 | **+0.1089** | +0.0244 | −0.0046 | +0.0120 | +0.0070 |
| `median_home_value` | +0.1055 | +0.0232 | +0.0279 | −0.0016 | +0.0071 | −0.0025 |
| `mean_commute_minutes` | +0.0345 | +0.0228 | +0.0124 | +0.0131 | +0.0003 | +0.0010 |

**The pre-registered rule is met on every clause, and not narrowly.** Positive
on 5 of 5 rather than a majority; above the floor on 5 of 5; second largest of
six. **Source F ships as a pillar, on evidence.**

### The one objection worth pre-empting

Source A's `has_metro_attachment` fires when a Wikipedia intro says the county
belongs to a metropolitan statistical area, which is the OMB delineation Source
F's `metro_2023` reports directly. Left in the reduced design it stands in for
part of what F carries, understating F's contribution. Removing it from both
sides moves the number from **+0.0413 to +0.0410** — nothing. F's case does not
depend on that column being present or absent, which is worth knowing because it
is the obvious place to look for an artifact.

Note also what the external number is already robust to by construction. F's
industry flags restate Source B's location quotients, but B is *in the reference
model*, so anything F contributes is contribution over B. A block that only
restated another pillar would score zero here.

## 5. Arm 1 agrees, and shows why the old test was misleading

Arm 1, mean ablated lift per block across the targets it is eligible for:

| block | targets | carrying signal | mean ablated lift | median | max |
|---|---|---|---|---|---|
| E | 28 | 17 | +0.0134 | +0.0100 | +0.0584 |
| D | 26 | 17 | +0.0125 | +0.0061 | +0.0637 |
| **F** | 28 | **18** | +0.0063 | +0.0030 | +0.0354 |
| C | 26 | 9 | +0.0061 | +0.0014 | +0.0465 |
| B | 9 | 4 | −0.0014 | +0.0009 | +0.0251 |
| A | 28 | 2 | −0.0031 | −0.0025 | +0.0103 |

F carries signal on **18 of 28 targets, more than any other block**, while
sitting mid-table on mean lift. Block B is scored on only 9 targets because it
owns the 20 LQ columns and cannot predict itself.

The important number in this table is not F's rank but the gap between F's raw
and ablated lift: **+0.0510 raw against +0.0063 ablated.** Roughly seven eighths
of F's apparent internal contribution is USDA restating industry composition
that BLS already measures. That is precisely the redundancy the original
pillar-pair sweep should have been testing for, and it is real — but it is a
statement about F versus B inside a closed system, not about whether F is worth
serving. Against outcomes outside the six pillars, where the same ablation moves
F by 0.0003, the redundancy does not bind.

**Read the two arms together:** internally F looks unremarkable once its overlap
with B is removed; externally it is second only to E. Both can be true, and the
second is the one that answers the go/no-go question.

## 6. Open decision #2: B ↔ E is complementary, but that is not the interesting part

`docs/PROJECT_GOAL.md` open decision #2 asks whether B ↔ E deserves privileged
weight, given that it is roughly five times stronger than anything else
surviving the size control (r = 0.394 raw / 0.382 controlled).

Withholding both blocks at once costs **+0.0632**, against **+0.0649** for the
sum of withholding each separately (B +0.0067, E +0.0582). Joint ≈ sum, so the
two pillars are **complementary rather than substantially redundant** — the
pillar count does not effectively drop by one.

**But the premise behind the question does not survive the measurement.** The
B ↔ E correlation is strong and it is not worth privileging, because almost all
of the pair's external value is E's: B contributes +0.0067 and is positive on
only 3 of 5 targets. A strong correlation between two pillars is evidence they
see the same economy; it is not evidence that either predicts anything, and
here only one of them does. Weighting the pair up would amount to weighting up
a correlation that no external target rewards.

Recommended resolution: **close decision #2 as "no privileged weight."** Keep
both pillars — B is cheap, its schema is now frozen, and its LQ vector is the
natural interpretability layer for E's capital signal — but do not build a
weighting scheme around the pair.

## 7. The uncomfortable finding: Source A, not Source F

Source A contributes **−0.0000** externally, is positive on 2 of 5 targets,
clears the noise floor on 2 of 5, and is the only block with a negative mean in
both arms. It is also the pillar `docs/pillar_status.md` marks "Good shape.
Done."

This is not a harness failure. The same code path produces +0.0582 for E and
+0.0413 for F, the placebo distributions behave, and A's per-target
contributions are small in both directions (−0.0032 to +0.0070) rather than
wildly negative — the signature of a block that is genuinely redundant with the
rest of the matrix, not one that is broken.

It is also not a contradiction of the evidence already on file. Source A's typed
block was justified on a marginal lift of **+0.0010** over a baseline holding
every other pillar (`source-a-findings.md` §13–§17) — a real effect, at p =
0.010 with power 0.92, and a tiny one. A contribution indistinguishable from
zero against five external outcomes is what that effect size predicts.

What it does mean: **applied consistently, the operating principle that every
pillar earns its slot on evidence now points at Source A rather than Source F.**
Three things stop this report from recommending a cut:

1. **A is nearly free.** No API key, no model, no inference. The cost of keeping
   it is a schema doc that already exists.
2. **The targets are ACS demographics.** A's columns encode named industries,
   universities, ports and protected land — plausibly more useful for an
   ad-tech outcome than for median age, and the consumer's real label is
   unobtainable (`docs/PROJECT_GOAL.md`, operating constraints).
3. **Redundancy inside a feature store is not uselessness.** A is redundant
   *with the other five pillars*, which is exactly the position a downstream
   model can exploit when another pillar is missing for a county.

Recorded here as an open item for the go/no-go deck, not as a proposed cut. The
deck should state A's marginal contribution honestly rather than let "done"
imply "valuable."

## 8. Limitations

- **The targets are public proxies, not the consumer's label**, which is
  structurally unobtainable. Every conclusion here is by analogy, and the
  go/no-go must say so rather than imply a direct test.
- **20 placebo reps per pillar × target, 49 in arm 1.** Enough to place a floor
  near zero against contributions an order of magnitude larger. Not enough to
  resolve a borderline contribution, and B's and C's numbers (+0.0067, +0.0054)
  are close enough to the floor that their ordering should not be quoted as
  settled.
- **Contribution is not importance under a different model class.** Everything
  here is ridge on an imputed design. A gradient-boosted consumer might
  distribute credit differently, and only arm 1's sibling script carries a GBM
  cross-check.
- **Cross-sectional and single-period.** Temporal transfer — one of the things a
  geographic fixed effect genuinely fails at — is still untested anywhere in
  this repo.
- **This does not answer the fixed-effect objection.** It reallocates credit
  among pillars given that the matrix as a whole beats a size baseline on
  held-out states; `external-target-findings.md` is where that prior question
  lives.

## 9. Forbidden wording

- Do not write "Source F failed its test." It failed a pairwise correlation
  test that was the wrong instrument, and passed the residual-variance test that
  `pillar_status.md` itself proposed. Both halves travel together.
- Do not write "Source F is the second most important pillar" without "on five
  public ACS targets, measured as marginal R² over the rest of the matrix."
  Importance is relative to a target, and this is not the consumer's target.
- Do not quote F's internal lift of +0.0510 without the ablated +0.0063 beside
  it. Seven eighths of it is USDA restating BLS.
- Do not write "B and E are redundant." They are complementary; B is
  individually weak. Those are different claims.
- Do not write "Source A should be cut." This report records A's marginal
  contribution and three reasons not to act on it yet.

## 10. Artifact index

- Arm 1: `scripts/analyze_pillar_block_marginal.py` →
  `outputs/pillar_block_marginal.csv`,
  `analysis-output/cross-source/pillar_block_marginal_stats.json`
  (`uv run scripts/analyze_pillar_block_marginal.py`, ~9 minutes)
- Arm 2: `scripts/analyze_external_target.py` →
  `outputs/external_target_scores.csv` (drop-one rows carry `withheld_pillars`,
  `contribution` and `contribution_ablated`),
  `outputs/external_target_drop_one_placebo.csv`,
  `analysis-output/cross-source/external_target_stats.json` (keys `drop_one`
  and `drop_one_noise_floor`)
- Prior rounds this builds on: `analysis-output/cross-source/external-target-findings.md`,
  `analysis-output/cross-source/pillar_matrix_signal_stats.json`
