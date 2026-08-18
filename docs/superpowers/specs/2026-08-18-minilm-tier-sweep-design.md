# Design — MiniLM tier sweep into section 2, 18 August 2026

**Deliverable:** four new spliced arms and per-tier scoring in
`scripts/analyze_source_a_tiered_embedding.py`, a new per-tier chart in section 2 of
`analysis-output/E_macro_pillar_worth_2026-08-13.ipynb`, and the generator edits in
`scripts/build_status_notebook.py` that make all of it survive a rebuild.

## The question

Section 2 currently settles *should the typed columns branch on tier* and answers no.
The parallel encoder question — *should the **input text** branch on tier* — is
already measured but sits in the appendix as a single pooled number per arm, mixed
into a six-bar chart alongside the typed-column scope arms.

Pooled numbers hide the thing section 2 spent chart 3 establishing: branching helped
the tier it was aimed at and lost on balance because a *bigger* tier went the other
way. Nobody has looked for that shape in the encoder arms, because the encoder scores
were never broken out by tier.

## What already exists, and is not being rebuilt

`analyze_source_a_tiered_embedding.py` encodes four text-selection rules through
`all-MiniLM-L6-v2` and scores each in raw, `_pca64`, and `_l2` form:

| Arm | stub | thin | mid | rich | mean lift |
|---|---|---|---|---|---|
| `lead_only` | lead | lead | lead | lead | +0.00169 |
| `uniform` | all | all | all | all | **+0.00322** (`_l2` +0.00351) |
| `tier_conditional` | all | all | economy | lead | +0.00180 |
| `tier_conditional_inverse` | lead | lead | economy | all | +0.00068 |
| `typed_sections` (shipped) | — | — | — | — | +0.00307 |

The verdict is already on file and is not in question: `uniform` beats every
tier-conditional rule, and the shipped typed block sits between them. This design
does not relitigate that. It asks **where** the tier-conditional loss comes from.

## Section 1 — Four sweep arms, built by splicing

A county's vector depends only on its own tier's text rule: `build_variant_texts`
selects text per county from `variant.tier_scope[tier]`, and `encode_variant`
mean-pools that county's chunks independently of every other county. Therefore an
arm defined as *"`uniform` everywhere except tier T reads its lead only"* is a
**row-wise splice of the `uniform` and `lead_only` arrays that the script already
computes**.

**Corrected 2026-08-18, after the check failed.** An earlier draft of this spec
claimed the splice is *bit-identical* to a fresh encode. It is not, and the
`--verify-splice` gate caught it: splice and real encode agree to ~1.6e-7, not to
1e-12. Measured cause, and it is **not** a GPU artifact:

| | same batch composition | different composition |
|---|---|---|
| `mps` | 0.000e+00 | 1.104e-07 |
| `cpu` | 0.000e+00 | 1.043e-07 |

`sentence-transformers` buckets by length and pads per batch, so a differently
sized `flat` chunk list changes padding and grouping and perturbs the numerics on
any device. Forcing CPU does not fix it.

What survives, and what the design now claims:

- **The splice is conceptually exact.** Identical composition reproduces
  bitwise (0.000e+00). Each arm's composition is fixed by its own text rule, so
  re-running the pipeline still reproduces its own artifacts exactly — the
  notebook's "no drift" provenance claim is unaffected. The 1e-7 appears only
  when comparing *different* arms' encodes, which the pipeline never does.
- **The splice is the more rigorous construction, not merely the cheaper one.**
  A spliced `drop_T` differs from `uniform` in tier T and nowhere else, because
  the untouched rows *are* `uniform`'s rows. Encoding each arm for real would
  perturb the untouched tiers by padding noise, putting ~1e-7 of drift into
  precisely the tiers the drop-one contrast holds fixed.

| Arm | stub | thin | mid | rich |
|---|---|---|---|---|
| `uniform` (reference) | all | all | all | all |
| `drop_stub` | **lead** | all | all | all |
| `drop_thin` | all | **lead** | all | all |
| `drop_mid` | all | all | **lead** | all |
| `drop_rich` | all | all | all | **lead** |

Each arm answers: *what does reading the full article in tier T buy, holding every
other tier at full depth?*

**Why drop-one rather than build-up.** The additive mirror — hold three tiers at
`lead_only` and upgrade one — is equally free by the same splice. Drop-one is chosen
because it is the instrument the pillar-worth analysis uses everywhere else in this
project, so the numbers read on a scale the audience has already been taught. The
additive mirror is a four-line addition and is deliberately **not** built now; it
gets built only if the drop-one result is ambiguous.

**Encoding cost: zero.** No new text rules, no new `TextVariant` entries requiring a
forward pass. The four encodes the script performs today are unchanged.

### The confound this must handle

`lead_only` vectors have mean norm ≈ 1.0 — one chunk, no averaging. `uniform` vectors
have mean norm 0.63–0.71, because many chunks partly cancel under mean-pooling. A
spliced arm therefore carries a **norm discontinuity exactly at the tier boundary**,
and a model can read the tier straight off vector magnitude without reading any text
at all.

This is not hypothetical. The script's own diagnostics already measure it:
`tier_variance_share` is 0.0607 for `tier_conditional` against 0.0121 for `uniform` —
a fivefold rise in the share of vector variance that is explained by tier alone, and
`StandardScaler` cannot remove it because it centres each *dimension* across counties
while the gradient runs across *rows*.

Consequences for this design, both mandatory:

1. Every sweep arm is scored in **both raw and `_l2`** form. `_pca64` is skipped for
   the sweep arms — it answers a width question that the sweep is not asking.
2. **The `_l2` twin is the arm the conclusion rests on.** The raw−`_l2` gap is
   reported, not hidden: it is itself the measurement of how much of any apparent
   tier effect was magnitude rather than text.

An arm that looks strong raw and collapses under `_l2` has found the tier label, not
the article.

## Section 2 — Per-tier scoring

Transplanted from `analyze_source_a_representation.py:578-593`, which already does
this for the typed-column arms. Keep the pooled out-of-fold predictions, then slice
both prediction vectors by tier mask and compute R² within the tier:

```
lift_tier = r2_score(y[mask], predictions[mask]) - r2_score(y[mask], baseline_predictions[mask])
```

Fitting stays pooled; only scoring is sliced. This is the deliberate choice — a
per-tier *fit* would change two things at once and make the lift incomparable to the
pooled arms, which is the same reasoning `_per_tier_oof_predictions` documents.

Guarded at `MIN_TIER_OBSERVATIONS = 150`, matching the typed-column path. Below that
an out-of-fold R² on a subset is dominated by which rows landed in it. Tiers falling
under the guard are omitted from the chart rather than plotted small, and the count of
targets surviving the guard is reported in the chart subtitle — chart 3 already sets
this precedent by disclosing that its stub estimate rests on 21 targets rather than 28.

**New artifact:** `outputs/source_a_tiered_embedding_by_tier.csv`, long format,
columns `pillar, column, representation, tier, n, r2_baseline, lift`. The existing
pooled `outputs/source_a_tiered_embedding.csv` is left untouched so nothing downstream
of it moves.

`source_a_tiered_embedding_stats.json` gains a `mean_lift_by_tier` block, mirroring
the key of the same name in `source_a_representation_stats.json`.

## Section 3 — Notebook changes

All three go into `scripts/build_status_notebook.py`. The `.ipynb` is a build
artifact and is never the edit surface.

**Section 2 gains one chart**, inserted after the existing chart-3 prose block
(generator line ~429) and before the section-3 transition. Reads the new by-tier CSV,
styled as a sibling of chart 3 so the typed-column and encoder stories read on the
same axes.

Grouped bars, four tier groups on the x-axis. Each group holds **five bars**:
`uniform` as the reference, then `drop_stub`, `drop_thin`, `drop_mid`, `drop_rich`.
Twenty bars is at the top of what chart 3's layout carries, so if it reads crowded at
projector size the fallback is to plot **only each tier's own drop arm against
`uniform`** — eight bars, and the diagonal is the whole question anyway. Decide from
the rendered figure, not in advance.

**The chart plots the `_l2` twins.** Raw is the confounded measurement, per section 1,
and putting it on the headline chart would invite reading a magnitude artifact as a
text effect. The raw−`_l2` gap goes in the prose beneath the chart, where it can be
labelled as what it is.

**Appendix cell 27** keeps the all-arms overview chart and gains a one-line pointer up
to section 2. It also gains `lead_only`, which is currently absent from that chart
despite being the retired embedding's actual configuration — its omission is why the
appendix chart reads as though the encoder was only ever tested at full depth.

## Section 4 — Two prose edits that must be ported

Two edits were made directly to the `.ipynb` earlier in this session and exist
**only** there. The generator still holds the superseded text, so the next run of
`build_status_notebook.py` silently reverts both. Porting them is part of this work,
not a follow-up.

1. **Generator line ~391** — the three-arm bullet list in section 2. The shipped
   version now states, for each arm, its coefficient count, whether it is one training
   run or four, and what is shared versus isolated (sample, coefficients, ridge
   penalty).
2. **Generator lines ~470-477** — the prose after chart 3. The number restate was cut
   (the chart's own title and subtitle already carry it) and the
   sparsity-already-encodes-tier aside was cut as a tangent. The bias–variance
   mechanism paragraph and the reproducibility note stay.

The `.ipynb` and the generator must hold identical text when this is done.

## Section 5 — What this can and cannot answer

**Can:** locate which tier the tier-conditional loss comes from, on the same axes as
the typed-column result, and separate that effect from the magnitude artifact via the
raw/`_l2` gap.

**Cannot:** say whether an intermediate depth — economy-titled sections only — would
suit a given tier better than either endpoint. `ECONOMY_TITLE_PATTERN` is currently
encoded only for the mid tier inside the two `tier_conditional` arms, so a
three-level sweep needs real forward passes for stub, thin, and rich. Out of scope
unless the drop-one result comes back ambiguous.

**Does not change:** the shipped design. `uniform` already beat every branching rule
and the typed block already ships on cost and interpretability rather than measured
lift. This sharpens the account of *why* branching lost; it does not reopen what ships.

## Verification

- `outputs/source_a_tiered_embedding.csv` byte-identical to its committed version
  after the run — the pooled path must be untouched by this change.
- **The splice gate, in its corrected two-part form.** Bitwise equality is not the
  claim, so it is not the test. For one arm, encoded for real and compared against
  its splice: (a) vectors agree to better than `1e-5`, comfortably above the
  ~1.6e-7 the padding noise produces and far below anything that could matter; and
  (b) — the part that actually licenses the design — the resulting **lifts** agree
  to better than `1e-6`, an order below the `1e-5` precision every lift in this
  notebook is reported at, so the difference cannot move a printed digit. A gate on
  the number we publish beats a gate on a bitwise property we never needed.
- Tier-sliced lifts reconcile against the pooled lift for `uniform` (n-weighted mean
  across tiers, within float tolerance).
- Notebook regenerated from the generator and the section-2 chart renders.
- `git diff` on the regenerated `.ipynb` shows the two ported prose edits as *no
  change*, confirming generator and artifact agree.

## Runtime

Unchanged encode (~87k chunks, four forward passes as today). Scoring grows by 8
blocks × 28 targets of ridge on 384 dimensions — seconds.
