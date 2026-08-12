# Pillar status

Snapshot as of 2026-08-12, **updated the same day**: every gap this document
listed is now closed. B, C and D have schema docs, F has one too, and F's slot
was settled on the fairer test this document asked for — it passes, decisively.
Details in `analysis-output/cross-source/pillar-marginal-findings.md`. The one
new open item is Source A — see its section below.

Scope is deliberately narrow: per-pillar evidence quality only. Grain, join
key, and fusion are separate open questions —
see `docs/PROJECT_GOAL.md` open decision #1 and `docs/plans/dma_regrain.md` —
and are out of scope here by request.

Source verdicts below restate `docs/PROJECT_GOAL.md` "Where the evidence
stands," reorganized around one question per pillar: is it in good shape, or
is it falling short, and if so what fixes it or whether it should be cut.

## A — Place Identity (Wikipedia lead-section text)

Good shape on paperwork. **The weakest pillar on the drop-one test**, which is
new information as of 2026-08-12.

Embedding step (`bge-m3`) was cut: |r| = 0.041 Mantel, k-means silhouette
0.028, and a statistical tie against the 29 typed columns head-to-head
(13/28, p = 0.76) — not worth a 2.2GB model download and CPU inference over
3,144 articles for a tie. Ships 29 typed columns extracted from the lead and
economy sections instead of the old single `content_length` scalar. Schema
frozen in `docs/source_a_feature_schema.md`. `data/source_a_embeddings.parquet`
is retained but no longer regenerated, so reinstating the embedding is a
`git revert` if anyone revisits it.

**New, 2026-08-12.** Withholding Source A's whole block from a model that has
size and the other five pillars costs **−0.0000** mean R² across five external
ACS targets — positive on 2 of 5, above the shuffled-feature noise floor on 2 of
5, the lowest of the six. It is the only block negative in both the internal and
the external arm.

That is consistent with the evidence already on file rather than contradicting
it: the typed block was justified on a marginal lift of +0.0010 over a baseline
holding every other pillar, which is a real effect and a tiny one. A is
redundant with the rest of the matrix, and redundancy inside a feature store is
not the same as uselessness — a consumer can lean on A for a county where
another pillar is missing.

No action proposed. **But the go/no-go deck should state A's marginal
contribution rather than let "done" imply "valuable"**, and if the operating
principle "every pillar earns its slot on evidence" is applied consistently, A
is now the pillar it points at. Reasons not to act yet — near-zero maintenance
cost, ACS targets that are a poor match for what A's columns encode — are in
`pillar-marginal-findings.md` §7.

## B — Industrial Core (BLS QCEW location quotients)

Paperwork done. Signal real but marginally thin.

Feature fix already validated: ships the 20-dim LQ vector, not a scalar.
Strongest surviving cross-pillar link in the whole sweep is B against E
(capital-to-wage ratio), r = 0.394 raw / 0.382 size-controlled — roughly 5x
anything else that survives size control.

~~Gap: no `docs/source_b_feature_schema.md`.~~ **Written 2026-08-12**, generated
from the parquet by `scripts/export_pillar_schema.py`. It states the three-state
null semantics the pillar turns on: 27.9% of county × sector cells are BLS
suppressed, a further 6.7% have no county row at all, and **34.7% of the LQ
matrix therefore arrives null** — more than the 30.0% suppression rate the
findings quote, which counts suppression only among cells BLS reports at all.

~~Open question: does B↔E deserve privileged weight?~~ **Answered 2026-08-12:
no.** Withholding B and E together costs +0.0632 against +0.0649 for the sum of
withholding each alone, so they are complementary rather than redundant — the
pillar count does not drop. But the premise does not survive: B's own
contribution is **+0.0067, positive on 3 of 5 targets**, against E's +0.0582.
Nearly all of the pair's external value is E's, so there is nothing to
privilege. `docs/PROJECT_GOAL.md` open decision #2 is closed on that basis.

Not a cut candidate: B is the interpretability layer for E's capital signal, its
20-dim LQ vector is the strongest surviving cross-pillar link in the sweep, and
its cost is one quarterly download.

## C — Economic Velocity (FRED unemployment & real GDP slopes)

Paperwork done. Signal real but small.

Metric fix already done: every reported result uses `gdp_velocity_pct`, not
dollar-denominated `gdp_velocity`.

~~Gap: no `docs/source_c_feature_schema.md`.~~ **Written 2026-08-12.** It
documents the coverage split (3,080 counties with both series, 63 unemployment
only, 1 with neither) and the two unrelated causes behind the 64 missing GDP
series: 51 Virginia independent cities and all 9 Connecticut Planning Regions.

One thing the schema doc surfaced that was not previously written down
anywhere: **`gdp_velocity` is still inside the matrix's Source C block**, not
held out in `SIZE_COLUMNS` alongside `gdp_latest`, and it runs r = +0.420 with
log population against `gdp_velocity_pct`'s +0.101. The metric fix is done in
the sense that no reported result uses the dollar column; its continued
membership in the block is an open item, not a decision. Cheap to resolve and
worth resolving before handoff.

Marginal contribution on the drop-one test: **+0.0054, positive on 5 of 5
targets** — small, but consistently signed, and above the noise floor on 4 of 5.
Not a cut candidate.

## D — Trade Logistics (BTS FAF5 freight flows)

Good shape. Paperwork done.

Feature fix already done and validated: ships the ten commodity *shares*,
not raw per-commodity tonnages (which ran 0.52–0.97 Spearman against
population — i.e. were mostly just measuring county size). The shares are
what surfaced the freight-to-industry link the original proposal claimed:
Agriculture LQ moved from indistinguishable-from-zero to +0.0430 ablated,
Manufacturing LQ +0.067 → +0.107. The planned `tons_per_capita`
normalization was tested and found to add nothing — it's algebraically
identical to `log_total_tons − log_population`, which any model controlling
for size already has (`source-d-findings.md` §10).

~~Gap: no `docs/source_d_feature_schema.md`.~~ **Written 2026-08-12.** Source D
is the only pillar with no null policy to state — zero nulls anywhere in the
file — which the schema doc says explicitly so nobody goes looking for one.

Marginal contribution on the drop-one test: **+0.0191, positive on 5 of 5
targets and above the noise floor on 5 of 5** — third of six, and the ablation
barely moves it. Not a cut candidate.

## E — Capital Flow (IRS SOI capital-to-wage ratio)

Good shape. Done.

The raw ratio is a product of three separable drivers (R² = 0.975 on its
log) and its level tracks the market year, not the county — unweighted
county mean runs 0.095 / 0.156 / 0.108 across TY2020–TY2022. Ships the three
components plus a TY2018–TY2022 normalized mean instead. Re-scored sweep
backs the change: 24 of 29 targets carry signal against 21 before, mean lift
+0.0720 → +0.0808, and the definitional share of that lift falls from 0.683
to 0.592 — i.e. most of the gain is real, not just recombination. Two
Source E dollar totals moved into the size control at the same time and
cost only −0.0011, so the gain isn't a size artifact either. Schema frozen
in `docs/source_e_feature_schema.md`.

Confirmed 2026-08-12 as **the strongest pillar of the six**: withholding E's
block costs +0.0582 mean R² across the five external targets, positive on 5 of 5
and the largest margin over the noise floor anywhere in the sweep. Where the
schema doc already says to prefer `capital_to_wage_ratio_normalized_mean`, that
preference now has an external number behind it rather than only a
vintage-stability argument.

No action needed.

## F — Structural Resilience (USDA ERS county typology)

**Resolved 2026-08-12: F ships, on evidence.** It was the one pillar this
document flagged as falling short; option 1 below was run, and F passed it
decisively. The original text is kept underneath the verdict because it is the
pre-registration the test was run against.

### The verdict

Withholding Source F's block from a model that already holds county size and the
other five pillars costs **+0.0413 mean R²** across five external ACS targets —
**second of six pillars**, behind only Source E's +0.0582 and ahead of D
(+0.0191), B (+0.0067), C (+0.0054) and A (−0.0000). Positive on 5 of 5 targets,
above the shuffled-feature noise floor on 5 of 5, where the largest apparent
contribution any shuffled block produced anywhere in the sweep was +0.0031.

The pre-registered rule — positive on a majority of targets and above the noise
floor, restatements ablated — is met on every clause and not narrowly.

Two robustness checks, both clean:

- Source A's `has_metro_attachment` restates F's `metro_2023` and sits in the
  reduced design, covering for part of what F carries. Removing it from both
  sides moves F from +0.0413 to **+0.0410**.
- F's industry flags restate Source B's location quotients, but B is in the
  reference model, so what F contributes is contribution *over* B. A block that
  only restated another pillar would score zero by construction.

**What this does not overturn.** F still fails the pairwise hub test, exactly as
described below: r = 0.495 against Source D tonnage, −0.057 once size is
controlled. Both facts are true, and they travel together. The pairwise test was
the wrong instrument for a categorical structural variable, which is what this
document said in option 1 before the numbers existed.

The internal arm carries the caution worth quoting alongside the headline:
across the 29 in-matrix targets, F's raw lift is +0.0510 and its ablated lift is
**+0.0063** — roughly seven eighths of F's apparent internal contribution is
USDA restating industry composition BLS already measures. That redundancy is
real inside the six-pillar system and does not bind against outcomes outside it,
where the same ablation moves F by 0.0003.

Paperwork: `docs/source_f_feature_schema.md` written 2026-08-12, closing the
inconsistency where this document marked F's paperwork N/A while
`docs/PROJECT_GOAL.md` next-work item 5 listed F as still needing one.

### The original assessment, kept as the pre-registration

**This was the one actually falling short.**

What falling short means here, concretely: F's only strong raw correlation
in the full 15-pillar-pair sweep was F against D freight tonnage, r = 0.495
— the single largest raw effect in the entire sweep. Once county size is
controlled, that correlation collapses to **r = −0.057**. The apparent link
was population riding along in both variables, not a real structural
relationship. Strip that out and F has no surviving hub-test evidence — it
does not correlate meaningfully with any of the other five pillars.

Current resolution in `docs/PROJECT_GOAL.md`: keep F, but reclassify it —
stop justifying its slot by hub-test correlation (which it fails) and
justify it instead as a structural anchor, i.e. by what it definitionally
is (county typology metadata) rather than by statistical performance
against the other pillars. That's a real downgrade in justification, not a
cosmetic one.

Options, in order of preference:

1. **Run a fairer test before the go/no-go.** Raw/controlled correlation is
   the wrong lens for a categorical structural variable — test whether F
   explains residual variance after B/C/D/E are already in the model,
   rather than raw pairwise correlation. This is the honest version of
   "structural anchor" and would either vindicate the keep decision or
   confirm the cut.
2. **Cut it.** If the operating principle is "every pillar earns its slot
   on evidence" (stated explicitly in `docs/PROJECT_GOAL.md`), F hasn't met
   that bar, and the reclassification is close to a rationalization for
   keeping a pillar that failed its own test. Six sources becomes five;
   say so plainly in the go/no-go deck rather than leaving F's status
   ambiguous.
3. **Keep as-is, reclassified.** Cheapest option — USDA ERS is decennial,
   near-zero ingestion cost to maintain — but weakest for a go/no-go
   narrative that's supposed to show every pillar earned its place.

~~No fix has been applied yet; this is an open decision, not a done item like
A/C/D/E.~~ **Option 1 was run on 2026-08-12. It vindicated the keep.**

## Summary

Marginal contribution is the mean R² a model loses when the pillar's whole block
is withheld, across five external ACS targets, restatements ablated, out-of-fold
on held-out states. Full table and caveats in
`analysis-output/cross-source/pillar-marginal-findings.md`.

| Pillar | Marginal contribution | Positive on | Paperwork | Verdict |
|---|---|---|---|---|
| A | −0.0000 | 2/5 | Done | Ships; contributes nothing marginal — the new open item |
| B | +0.0067 | 3/5 | Done 2026-08-12 | Ships; individually thin, complementary to E |
| C | +0.0054 | 5/5 | Done 2026-08-12 | Ships; small but consistent |
| D | +0.0191 | 5/5 | Done 2026-08-12 | Ships |
| E | +0.0582 | 5/5 | Done | Ships; the strongest pillar |
| F | +0.0413 | 5/5 | Done 2026-08-12 | Ships; second strongest, on evidence |

Six pillars, six schema docs, every block above the shuffled-feature noise floor
except Source A's. The go/no-go can state each pillar's worth as a number rather
than as a narrative.

## Related

- `docs/PROJECT_GOAL.md` — "Where the evidence stands" section, canonical
  source for the verdicts above.
- `analysis-output/cross-source/pillar-marginal-findings.md` — the drop-one
  test behind every contribution figure above, its pre-registered decision rule,
  and its limitations.
- `docs/source_{a..f}_feature_schema.md` — all six now exist. A is generated by
  `scripts/export_source_a_schema.py`, B/C/D/F by
  `scripts/export_pillar_schema.py`; E is the one hand-written doc.
- `analysis-output/source-{a..f}/` — per-pillar findings backing each
  verdict.
- `analysis-output/E_macro_key_findings.ipynb` — detail behind the sweep
  numbers cited above.
