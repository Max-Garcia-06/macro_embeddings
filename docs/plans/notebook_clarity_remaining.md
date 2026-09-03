# Status notebook: what is left after the clarity pass

Companion to the four fixes landed on `notebook-clarity-pass`. This document
carries the five items that were deferred, in the order they should be done, with
enough detail to execute without re-deriving the reasoning.

The notebook is generated. **Every change below is an edit to
`scripts/build_status_notebook.py`, never to the `.ipynb`.** Rebuild with:

```bash
uv run python scripts/build_status_notebook.py && uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace analysis-output/E_macro_pillar_worth_2026-08-13.ipynb
```

## What already landed

| # | Fix | Where |
|---|---|---|
| 1 | Cross-references by name, not number; three were pointing at the wrong section | throughout |
| 2 | Evidence-basket table, and stub's 21-of-28 explained | section 1 |
| 3 | Drop-one matrix and corpus composition generated from artifacts; `analyze_source_a_section_composition.py` added | Source A tier section, representation section, A8 |
| 4 | Units on every Source E statistic; 17.4% / 9.5% restored from the artifact | Source E tier section |
| 5 | Bootstrap intervals on the representation arms — **done** | representation section |
| 6 | Source A narrated once; superseded accounts moved to A9 — **done** | sections 2, 5, 6, A2, A9 |
| 7 | Decision list with owners and dated defaults — **done** | limits section |
| 8 | Shape and refresh-cost rows, each labelled by basis — **done** | limits section |
| 9 | All five simplifications — **done** | throughout |
| — | `display.max_colwidth` — pandas was eliding every table cell past 50 chars, including the evidence-basket table from fix 2 | setup cell |

Two questions from the read-through are answered by those and do not recur below:
how many baskets exist, and why stub scores on 21 targets.

---

## 5. Confidence intervals on the representation arms

**Status: DONE.** It did change what the section concludes — see *Result*
at the end of this item.

### Why

The representation section's central claim is that two coordinate columns do 96%
of what the selected encoder does: `latlong_only` scores +0.0158 against the
encoder's +0.0164 on the 41-target decision basket, and the encoder's
contribution net of geography falls to +0.0006. Every one of those numbers is a
mean over targets reported to four decimal places with no interval attached, and
the section already concedes two things that make an interval necessary:

- the basket is clustered — 5 of 42 targets are heating-fuel shares from a single
  ACS table, 35 target pairs correlate above 0.7, and the effective sample is
  nearer 28 than 41;
- the reported power is therefore overstated.

A boss reading +0.0164 against +0.0158 cannot tell whether the gap is real. If
the interval on the geography-net contribution covers zero, the section's
wording has to change from "falls to +0.0006" to "is indistinguishable from
zero", which is a stronger statement than the one currently made.

### What to do

The per-target numbers are already persisted, so **no model needs re-fitting.**
`analysis-output/source-a/source_a_representation_marginal_stats.json` carries
`by_representation[arm]["by_target"]` and `["by_target_geo"]`, and
`outputs/source_a_representation_marginal.csv` carries the same per-target rows
with `contribution` and `contribution_geo` columns.

1. **Paired bootstrap over targets.** Resample targets with replacement — the
   target is the unit the headline mean is taken over, not the county — and
   recompute each arm's mean on the resampled basket. Pair the arms: resample the
   target set once per replicate and score every arm on the same draw, so the
   interval on a *difference* between arms is not inflated by target-level
   variance both arms share.
2. **Cluster the resample by ACS table.** `EXTERNAL_TARGETS` in
   `scripts/ingest_external_targets.py` carries a `table` field per target, so
   resampling whole tables rather than individual targets is a lookup, not new
   work. Report both the naive and the table-clustered interval; the gap between
   them is the concrete size of the clustering caveat the section currently
   states in words.
3. **Report the difference, not the ratio.** A confidence interval on "96%" is a
   ratio of two means whose denominator is small, and it will be unstable and
   unreadable. Report `encoder − latlong_only` with its interval, and keep 96% as
   the point estimate in prose.
4. **Three intervals belong in the notebook**, printed under the geography figure
   rather than in a new section: the selected encoder's plain contribution, its
   contribution net of geography, and the encoder-minus-`latlong_only`
   difference.

### Where

- `scripts/analyze_source_a_representation_marginal.py` — add the bootstrap, write
  the intervals into the existing stats artifact under a new
  `bootstrap` key per arm. Seed it off the module's existing `RANDOM_SEED` so it
  reproduces.
- `scripts/build_status_notebook.py` — the geography figure's subtitle, and the
  paragraph beginning "**The win is geography, not economic content.**"
- `docs/plans/notebook_clarity_remaining.md` — record the result here when done.

### Acceptance

- Both intervals are in the artifact and both are shown in the notebook.
- The prose states which interval it is quoting.
- If the geography-net interval covers zero, the surrounding sentences are
  rewritten to say so, and the "settled, then complicated" framing gets its third
  beat.

### Effort and risk

Half a day, most of it wording. The risk is not technical: it is that the
interval turns a clean-sounding number into a hedged one, which is the correct
outcome and should not be softened. The pre-registered verdict is unaffected
either way — a paired comparison cancels the baseline both arms share, and that
rule was fixed before scoring.

### Result

`bootstrap_representations` in `analyze_source_a_representation_marginal.py`,
10,000 replicates, seeded off `RANDOM_SEED`, written to every arm's `bootstrap`
key. Both resamples reported: naive over the 41 basket targets, and clustered
over the 28 ACS tables they come from.

| Arm | naive | table-clustered |
|---|---|---|
| selected encoder | +0.0164 [+0.0093, +0.0251] | +0.0164 [+0.0071, +0.0297] |
| `latlong_only` | +0.0158 [+0.0055, +0.0281] | +0.0158 [+0.0030, +0.0287] |
| encoder net of lat/lon | +0.0006 [−0.0081, +0.0088] | +0.0006 [−0.0070, +0.0082] |

**The geography-net interval covers zero on both resamples.** The section's
wording changed accordingly: "falls to +0.0006" now reads as a number the basket
cannot distinguish from nothing, and the heading became "settled, complicated,
then bounded" — the third beat the acceptance criteria asked for. The
pre-registered verdict is untouched, as anticipated.

**One thing the plan did not anticipate:** two of its three requested intervals
are the same statistic. `latlong_only`'s full model *is* the geo-adjusted
baseline every other arm is scored against, so

    contribution_geo(arm) = contribution(arm) − contribution(latlong_only)

holds identically. Three numbers are reported, not four, and the notebook says
why. `test_geo_net_interval_is_the_latlong_difference` keeps the identity
honest; `test_bootstrap_pairs_the_arms_within_a_replicate` keeps the pairing
honest.

**No model was re-fitted.** `summarize` is a pure function of the per-target
scores, and those are committed in `outputs/source_a_representation_marginal.csv`,
so the artifact was regenerated from the CSV through the real code path. Every
pre-existing figure round-tripped to within 1e-9 relative and the originals were
kept; only the `bootstrap` key is new. A full re-run reproduces the same
intervals.

---

## 6. Collapse the two Source A sections into one

**Status: DONE.** See *Result* at the end of this item.

### Why

Source A's representation is currently narrated three times, in the order the
work happened rather than the order a reader needs:

1. the tier section gives the encoder tie and the cost argument for the typed
   columns, then says the representation section replaces that account;
2. the pillar-worth section gives −0.044 for the embedding, then says that number
   is mostly a width artifact;
3. the representation section corrects both, and then complicates its own answer
   with the geography control.

A reader meets two superseded accounts before the current one. In a live talk the
superseded ones are what gets remembered, because they arrive first and are
stated most confidently. The honesty of keeping them on the record is worth
preserving; their position is not.

### What to do

One Source A representation section, in this order:

1. **What ships today** — the 29 typed columns and why they exist (the corpus
   diagnosis: where economic content actually lives).
2. **The encoder comparison, run correctly** — both confounds removed (reading
   scope, width), the leakage screen, the pre-registered rule, the outright win.
3. **The geography control** — two coordinate columns do 96% of it, with the
   intervals from item 5.
4. **What that means for shipping** — not "ship the encoder"; the next test is
   whether `E_macro` needs Source A at all once the sibling tiers supply location.

Move to a new **appendix A9, "Superseded Source A accounts"**:

- the encoder tie at 11-of-28 and 14-of-28 and the cost argument built on it;
- the −0.044 result and the width artifact that produced it;
- the ~9,000-character chunk cap correction already recorded in A2, which belongs
  with the other superseded numbers rather than in the method appendix.

Each entry keeps one line saying what replaced it and why, so nothing looks
quietly dropped.

The tier work — branching, the drop-one sweep, the three independent tests —
stays where it is. It is a different question and it survived.

### Where

`scripts/build_status_notebook.py`, the `md()` blocks between the section 2
banner and the branching verdict, and the whole representation-section block.
Check `docs/superpowers/specs/2026-08-13-exec-status-notebook-design.md` for the
wording constraints before editing, and
`analysis-output/cross-source/pillar-marginal-findings.md` §9, which constrains
the Source A and Source F wording specifically.

### Acceptance

- No forward reference in the talk sections tells the reader that what they are
  reading has been superseded.
- A9 exists and every superseded number is in it, with its replacement named.
- Section names in prose still resolve (item 1's convention holds — this reorder
  is exactly the event that convention was for).
- Talk length drops. Re-time it; the header currently claims 30 minutes for what
  reads closer to 45.

### Effort

Half a day. Largest diff of anything in this document, and entirely prose.

### Result

Section 6 is now one section in the four moves the plan named, as `### 1.` to
`### 4.` subheadings: what ships today, the encoder comparison run correctly, the
geography control, what that means for shipping. A9 exists and carries all three
superseded accounts — the encoder tie with both p-values and the cost argument,
the −0.044 and the width artifact, and the ~9,000-character chunk cap moved out
of A2 — each with a `> *Replaced by*` line naming what took its place.

The tier section no longer opens an account it will later withdraw: the two
paragraphs that gave the tie and then took it back are replaced by one sentence
saying the representation question is asked elsewhere. The pillar-worth section's
three paragraphs on −0.044 collapse to one that states the current position and
points at A9 for the history.

**Talk length did not drop, and it was not going to.** Items 5, 7, 8 and the
glossary all add prose to the talk by design; item 6's own edits moved ~470 words
from talk to appendix and item 9's compressions removed ~450 more, but the
additions outweigh them. Counted: 5,605 → 5,582 words of spoken prose, which is
flat, with the appendix up 2,027 → 2,571. The talk is not shorter; it is
correctly ordered, and the header now states 45 minutes instead of 30 with the
count behind it and a named 30-minute cut.

---

## 7. A decision list with a default action

**Status: DONE.** See *Result* at the end of this item.

### Why

The pillar-worth section ends by handing the room a question — does Source A
ship? — whose answer requires knowing whether the downstream target is
demographic or economic. `docs/downstream_target.md` records that the consumer is
Comcast FreeWheel Revenue Science and that one row is an impression, ad
request, auction, household or device, asserted verbally and pending written
confirmation. So the question may not come back at all, and the notebook has no
stated behaviour for that case.

Meanwhile the representation section produces a concrete third option that never
reaches the recommendation list: `data/county_centroids.parquet` already exists,
and two columns from it supply 96% of the measured gain with no model download
and no inference.

### What to do

1. Turn the "remaining path" list into the decision list, with a row per open
   decision, an owner, and a default.
2. Add the Source A row with a dated fallback, in this shape: *no answer from the
   consuming team by `<date>` → ship five pillars plus two centroid columns from
   `data/county_centroids.parquet`; Source A stays in the repo, unshipped, and
   the go/no-go deck says why.*
3. Shorten the open-question block in the pillar-worth section to a pointer at
   that row. The three-arguments analysis is good and should live in one place,
   not two.
4. State the cost of keeping A: it is the argument that justifies leaving the
   code in the repo, and it is currently asserted once as "nearly free" with no
   number. Item 8 supplies the number.

### Acceptance

- Every open decision in the notebook has a default action and a date.
- The Source A recommendation appears exactly once.
- The centroid-columns option is named in the recommendation, not only in the
  analysis that produced it.

### Effort

Two hours, once items 5 and 6 have settled the wording.

### Result

"The remaining path" is now "The decisions, and what happens if nobody makes
them" — five rows, each with an owner and a default. Two carry the date
**2026-09-18**: the join grain and whether Source A ships. **That date is a
placeholder chosen while writing this; change it to whatever the real deadline
is before presenting.**

The Source A row reads as the plan specified: ship five pillars plus two centroid
columns from `data/county_centroids.parquet`, which supply 96% of A's measured
gain; Source A stays in the repo, unshipped, and the go/no-go deck says why. The
centroid option now appears in the recommendation and not only in the analysis
that produced it.

The pillar-worth section's open-question block keeps the three-arguments analysis
and drops the recommendation, ending instead with a pointer at the decision row —
so the Source A recommendation appears exactly once. It also no longer contains
the sentence "the recommendation is to cut it from the shipped matrix", which sat
uncomfortably close to the forbidden wording in
`pillar-marginal-findings.md` §9.

---

## 8. Cost and shape rows in the readiness table

**Status: DONE.** See *Result* at the end of this item.

### Why

The readiness table has seven evidence rows and zero cost rows. A go/no-go needs
both sides. And the notebook never states what the shipped thing actually is:
3,144 rows by how many columns.

### What to do

1. **Column and row count of the shipped matrix**, computed rather than typed.
   `pillar_matrix.build_matrix()` returns the matrix and its per-pillar blocks, so
   the per-pillar column counts and the total are one call. Add them to the
   readiness display, or to a one-line print above it.
2. **Refresh cost per pillar.** What a vintage refresh costs in wall-clock,
   request count, and manual steps. Source A's is knowable from
   `ingest_source_a.py` (3,144 requests, flat); the others need either
   measurement or an explicit estimate. **Label each cell measured or estimated** —
   an estimated cost presented as a measured one is exactly the kind of thing the
   rest of this notebook refuses to do.
3. Cross-reference `outputs/pillar_vintages.csv`, which already carries cadence
   and reference period, so the cost row sits next to how often it is paid.

### Acceptance

- The readiness table states what ships and what maintaining it costs.
- Estimated figures are marked as estimates.

### Effort

Two hours for the shape rows, longer for the cost rows if they are measured
rather than estimated. Estimating is acceptable here provided it is labelled.

### Result

**Shape**, computed from `build_matrix()`: 3,144 counties × 118 pillar feature
columns (A 29, B 40, C 4, D 15, E 10, F 20) plus 3 size controls held out of every
block. Printed above a per-block table rather than typed into prose.

**Cost**, one row per pillar, with cadence and reference period joined from
`outputs/pillar_vintages.csv` and a `Basis` column on every row. Only Source B is
measured end to end (4m36s, from `logs/ingest_source_b_rerun.log`). Request counts
are counted off the ingest scripts and will not drift unless the ingest changes;
Source C's ≥63 minutes is derived from its own 100/min rate limiter. Every
wall-clock figure other than B's is marked estimated.

The A row is what item 7 needed: 3,144 API requests against credentialled
Wikimedia Enterprise access, on a source with no reference period. "Nearly free"
now has a number attached in both places it is claimed.

---

## 9. Simplifications

**Status: DONE, all five.**

None of these change a conclusion. All are independent of each other and of
everything above, and all can be dropped without consequence.

- **The three-arm branching explanation** in the Source A tier section is roughly
  450 words of prose describing three fitting schemes. Make it a three-row table:
  arm / coefficients estimated / rows each coefficient sees / penalty shared. Same
  content, one glance.
- **The drop-one sweep narration.** The table is now generated under its chart, so
  the per-cell walkthrough beneath it can drop to two sentences: stub gains from
  reading less, and the gain does not survive pooling. The `drop_thin`
  off-diagonal detail is appendix material.
- **Consolidate the caveats.** The representation section's three caveats, the
  `no_fuel_used_share` exclusion, the `typed_transformed` backfire and the 15
  unscreened targets are one category of thing scattered across three cells.
  One "known weaknesses in the decision basket" block, kept in the talk rather
  than the appendix, because the section's credibility rests on them.
- **A glossary box**, six terms: pillar, block, arm, tier, basket, restatement
  ablation. The room is mixed and will not ask.
- **The time budget.** The header claims roughly 30 minutes. Time the talk after
  item 6 and state the real number.

### Effort

An hour each. Do them when the substantive items are done, or not at all.

### Result

- Three-arm branching prose → a four-column table (arm / coefficients estimated /
  rows each coefficient sees / ridge penalty) plus two sentences on what makes the
  middle arm subtle. ~350 words → ~150.
- Drop-one narration → three sentences. The per-cell walkthrough, including
  `drop_thin`'s off-diagonal cost exceeding its own diagonal gain, moved to A2.
- Caveats consolidated into `### Known weaknesses in the decision basket`, kept in
  the talk, now four named weaknesses including the 15 unscreened targets that
  were previously buried in the leakage paragraph.
- Glossary added as `### Six words this talk uses precisely` in section 1, before
  the evidence baskets so "basket" is defined before it is used.
- Time budget re-counted: 45 minutes, with the 30-minute cut named explicitly.

### One thing found while doing these

`pd.set_option("display.max_colwidth", None)` is now in the setup cell. pandas
elides string cells past 50 characters in the **HTML** repr as well as the text
one, so every wide table in this notebook was shipping to the export with its
explanation replaced by "...", including the evidence-basket table that fix 2
added. This was pre-existing and is unrelated to the five items above.

---

## Sequencing

```
5 (intervals)  ──▶  6 (collapse Source A)  ──▶  7 (decision list)
                                                     ▲
8 (cost + shape rows) ───────────────────────────────┘
9 (simplifications) — independent, any time
```

Item 5 first because it can change what item 6 has to say. Item 7 last of the
three because it depends on both. Item 8 feeds the cost argument item 7 needs but
can be done in parallel.

**All five are done**, in that order, plus the `max_colwidth` fix. What is left
is not in this document: the 2026-09-18 placeholder date in the decision list, and
the five estimated wall-clock figures in the cost table, which become measured the
next time each ingest is run.
