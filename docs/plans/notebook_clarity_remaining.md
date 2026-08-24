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

Two questions from the read-through are answered by those and do not recur below:
how many baskets exist, and why stub scores on 21 targets.

---

## 5. Confidence intervals on the representation arms

**Status: do this first. It can change what the representation section concludes.**

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

---

## 6. Collapse the two Source A sections into one

**Depends on item 5** — do the intervals first so the conclusion is written once.

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

---

## 7. A decision list with a default action

**Depends on items 5 and 6.**

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

---

## 8. Cost and shape rows in the readiness table

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

---

## 9. Simplifications

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
