# Pillar status

Snapshot as of 2026-08-12. Scope is deliberately narrow: per-pillar evidence
quality only. Grain, join key, and fusion are separate open questions —
see `docs/PROJECT_GOAL.md` open decision #1 and `docs/plans/dma_regrain.md` —
and are out of scope here by request.

Source verdicts below restate `docs/PROJECT_GOAL.md` "Where the evidence
stands," reorganized around one question per pillar: is it in good shape, or
is it falling short, and if so what fixes it or whether it should be cut.

## A — Place Identity (Wikipedia lead-section text)

Good shape. Done.

Embedding step (`bge-m3`) was cut: |r| = 0.041 Mantel, k-means silhouette
0.028, and a statistical tie against the 29 typed columns head-to-head
(13/28, p = 0.76) — not worth a 2.2GB model download and CPU inference over
3,144 articles for a tie. Ships 29 typed columns extracted from the lead and
economy sections instead of the old single `content_length` scalar. Schema
frozen in `docs/source_a_feature_schema.md`. `data/source_a_embeddings.parquet`
is retained but no longer regenerated, so reinstating the embedding is a
`git revert` if anyone revisits it.

No action needed.

## B — Industrial Core (BLS QCEW location quotients)

Good shape on signal. Missing paperwork.

Feature fix already validated: ships the 20-dim LQ vector, not a scalar.
Strongest surviving cross-pillar link in the whole sweep is B against E
(capital-to-wage ratio), r = 0.394 raw / 0.382 size-controlled — roughly 5x
anything else that survives size control.

Gap: no `docs/source_b_feature_schema.md`. A and E have frozen schema docs;
B does not.

Fix: write the schema doc and freeze null semantics, same pattern as A and E.
Not a cut candidate — the signal is real.

Open question worth resolving before handoff: does B↔E deserve privileged
weight, or are the two pillars substantially redundant? Unresolved in
`docs/PROJECT_GOAL.md` open decision #2. Worth checking before go/no-go —
if B and E overlap heavily, the pillar count effectively drops by one.

## C — Economic Velocity (FRED unemployment & real GDP slopes)

Good shape on signal. Missing paperwork.

Metric fix already done: uses `gdp_velocity_pct`, not dollar-denominated
`gdp_velocity`.

Gap: no `docs/source_c_feature_schema.md`.

Fix: write the schema doc. Not a cut candidate.

## D — Trade Logistics (BTS FAF5 freight flows)

Good shape on signal. Missing paperwork.

Feature fix already done and validated: ships the ten commodity *shares*,
not raw per-commodity tonnages (which ran 0.52–0.97 Spearman against
population — i.e. were mostly just measuring county size). The shares are
what surfaced the freight-to-industry link the original proposal claimed:
Agriculture LQ moved from indistinguishable-from-zero to +0.0430 ablated,
Manufacturing LQ +0.067 → +0.107. The planned `tons_per_capita`
normalization was tested and found to add nothing — it's algebraically
identical to `log_total_tons − log_population`, which any model controlling
for size already has (`source-d-findings.md` §10).

Gap: no `docs/source_d_feature_schema.md`.

Fix: write the schema doc. Not a cut candidate.

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

No action needed.

## F — Structural Resilience (USDA ERS county typology)

**This is the one actually falling short.**

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

No fix has been applied yet; this is an open decision, not a done item like
A/C/D/E.

## Summary

| Pillar | Signal | Paperwork | Verdict |
|---|---|---|---|
| A | Good | Done | Good shape, done |
| B | Good | Missing schema doc | Good shape, needs schema doc |
| C | Good | Missing schema doc | Good shape, needs schema doc |
| D | Good | Missing schema doc | Good shape, needs schema doc |
| E | Good | Done | Good shape, done |
| F | Fails hub test | N/A | Falling short — fix the test or cut |

## Related

- `docs/PROJECT_GOAL.md` — "Where the evidence stands" section, canonical
  source for the verdicts above.
- `docs/source_a_feature_schema.md`, `docs/source_e_feature_schema.md` —
  the two schema docs that exist; B, C, D should follow this pattern.
- `analysis-output/source-{a..f}/` — per-pillar findings backing each
  verdict.
- `analysis-output/E_macro_key_findings.ipynb` — detail behind the sweep
  numbers cited above.
