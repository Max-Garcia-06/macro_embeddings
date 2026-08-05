# Project Goal

Orientation doc for anyone (human or agent) picking this repo up cold. Sources of
truth: `docs/geo embedding strategy rev0.2.pdf` (why `E_macro` exists),
`docs/macro_pre_scoping_spec.pdf` (what this stage is scoped to), and
`analysis-output/E_macro_key_findings.ipynb` (where the evidence stands).

## What this repo is

`E_macro` — the **regional/macro tier** of a three-tier geospatial embedding stack:

| Tier | Captures | Where it lives |
|---|---|---|
| `E_local` | Physical "vibe" — H3 res-8 + OpenStreetMap features | not this repo |
| `E_census` | Official demographics — Census/Overture joins | not this repo |
| **`E_macro`** | **Regional economic climate, county-level** | **this repo** |

The three are built as independent sub-embeddings and served from a feature store,
so a downstream model can select one or all. Deliberately *not* fused into a single
vector — OSM is current-but-indirect, Census is direct-but-stale, and training them
together would blend those failure modes.

`E_macro`'s specific job: distinguish physically identical places sitting in
different economic climates — a suburb outside New York versus one outside
Cleveland. It is keyed on `fips_code` at N ≈ 3,143 US counties and equivalents.

## Who consumes it

**This feeds downstream ML models on the Comcast FreeWheel Revenue Science
team** — ad-tech revenue modelling over video/CTV inventory. One row in their
training data is an impression, ad request, auction, household, or device, which
is what settles open decision #1 below. `E_macro` is not the deliverable
on its own — it is a geospatial feature layer those models pull from, so the bar
is production feature-store quality, not a one-off analysis: stable schema keyed
on `fips_code`, documented coverage and null semantics, and every feature
defensible on evidence rather than plausibility.

Two consequences that shape decisions in this repo:

- **Suppression and sentinel values stay explicit.** BLS suppresses ~30% of county
  × sector LQ cells and those stay null with a matching `disclosure_*` flag; IRS
  ships no suppression flag at all and that limitation is disclosed rather than
  papered over. A downstream model must be able to tell "missing" from "zero."
- **Nothing ships that hasn't earned its slot.** A feature that correlates with
  nothing, or only with county size, is a liability once it is in a production
  model — hence the validation-first stage below.

## Why six independent sources

Each pillar is a separate federal or public source, ingested by its own script into
its own `data/*.parquet`. Independence is the point: when two pillars agree, that's
two agencies with different methodologies corroborating the same underlying economy,
not one source echoing itself.

| Source | Pillar | Signal |
|---|---|---|
| A | Place Identity | Wikipedia lead-section text |
| B | Industrial Core | BLS QCEW location quotients |
| C | Economic Velocity | FRED unemployment & real GDP slopes |
| D | Trade Logistics | BTS FAF5 freight flows |
| E | Capital Flow | IRS SOI capital-to-wage ratio |
| F | Structural Resilience | USDA ERS county typology |

Update cadence spans continuous (A) to decennial (F), which is why ingestion is
offline/asynchronous batch — downstream consumers are isolated from API bottlenecks.

## Operating constraints

Recorded 2026-08-05 because they explain why the validation looks the way it
does, and because anyone picking this repo up cold will otherwise propose work
that cannot be done.

- **No access to company data or internal channels.** The project was scoped to
  **public and open-source data only**, deliberately and from the start. That is
  why every pillar is a federal or public source and why there is no downstream
  label anywhere in the repo.
- **A real downstream label is therefore unobtainable at this stage.** This is a
  structural fact, not an open question. It closes
  `docs/plans/source_a_next_steps.md` question 2 with "no," which per that
  document makes an **external public target mandatory rather than optional** —
  it is the only non-circular evidence this project can ever produce.
- **A data extract is a policy ask; a schema look is not.** Being shown rows on a
  screen during a scheduled conversation is expected to be available. What the
  open questions need is column names rather than values, so this is usually
  enough — see the checklist in `docs/plans/dma_regrain.md` §0.0. Copies of
  company data are a different request and should be planned around, not
  assumed.
- **Communication with the commissioning side is in person only**, four days a
  week, with no async channel. Questions must be batched into a written
  one-pager rather than asked one at a time; there is no "email the downstream
  team" path. Everything routes through one person.
- **No deadline and no budget cap in days.** The binding constraint is calendar
  weeks of availability, not permission. Leaving work unfinished is acceptable.

Two consequences worth stating plainly. Any claim that depends on the consuming
team's data — the fixed-effect comparison in `docs/plans/dma_regrain.md` Phase 3
above all — can only be answered **by analogy against a public proxy**, and the
go/no-go artifact must disclose that rather than imply a direct test. And every
deliverable has to stand on its own evidence without its author present to
explain it.

## Current stage: validation, not modeling

The pre-scoping spec scopes this phase to documenting boundaries and flagging risks,
not shipping the final tensor. Repo state matches that:

- All six sources ingested, coverage 3,143–3,144 counties on five of six.
- Per-pillar findings in `analysis-output/source-{a..f}/`.
- Full 15-pillar-pair crossvalidation sweep: 50 feature pairs, 499 permutations,
  one Benjamini-Hochberg correction across the sweep, every correlation recomputed
  as a partial correlation controlling for county size.
- **The fusion/assembly step does not exist yet.** By design.

Operating principle: every pillar must earn its slot on evidence before anything is
fused. Applied already — Source A's `bge-m3` embedding step was cut (|r| = 0.041
Mantel, k-means silhouette 0.028), and 15 of 33 significant correlations lose more
than half their effect once size is controlled.

## Where the evidence stands

Verdict per pillar (detail in `analysis-output/E_macro_key_findings.ipynb`):

- **A** — cut the embedding, keep the text source, and replace the single
  `content_length` scalar with 29 typed columns extracted from the lead and the
  economy section. Those beat the scalar and tie the cut embedding on mean
  cross-pillar lift, and they survive a baseline that already holds every other
  pillar (+0.0010, p = 0.010, power 0.92); see `source-a-findings.md` §13–§17.
  Schema and null semantics are frozen in `docs/source_a_feature_schema.md`.
  Done.
- **B** — keep, change the feature: ship the 20-dim LQ vector, not a scalar.
- **C** — keep, fix the metric: use `gdp_velocity_pct`, not dollar-denominated
  `gdp_velocity`. Done.
- **D** — keep, change the feature: ship the ten commodity *shares*, not the raw
  per-commodity tonnages, which run 0.52–0.97 Spearman against population. The
  shares are what let freight composition predict industry composition —
  Agriculture LQ moved from indistinguishable-from-zero to +0.0430 ablated,
  Manufacturing LQ +0.067 → +0.107 — which is the freight-to-industry link the
  proposal claimed and round 1 could not show. The `tons_per_capita`
  normalization this repo planned is **not** worth doing: it equals
  `log_total_tons − log_population` exactly, so it adds nothing to any model that
  already controls for size (`source-d-findings.md` §10). The ten raw tonnages
  moved into `SIZE_COLUMNS` on 2026-08-05 once the target was confirmed as a
  rate; they were retained until then only because a count target would have made
  them legitimately predictive. Done.
- **E** — keep, change the feature: the capital-to-wage ratio is a product of
  three separable drivers (R² = 0.975 on its log) and its *level* is set by the
  market year, not the county — the unweighted county mean runs 0.095 / 0.156 /
  0.108 across TY2020–TY2022. Ships the three components plus a TY2018–TY2022
  normalized mean; prefer that mean over the raw ratio. The re-scored sweep
  backs the change: 24 of 29 targets now carry signal against 21 before, mean
  lift +0.0720 → +0.0808, and the definitional share of that lift falls from
  0.683 to 0.592. Two Source E dollar totals moved into the size control at the
  same time (r ≈ 0.89 with log population) and cost only −0.0011, so the gain is
  not size. Schema frozen in `docs/source_e_feature_schema.md`. Done.
- **F** — keep, reclassify as a structural anchor rather than a hub-tested pillar.

Strongest surviving link: Source B Real Estate & Rental & Leasing LQ against
Source E capital-to-wage ratio, r = 0.394 raw / 0.382 size-controlled. Largest raw
effect in the sweep — D freight tonnage against F metro status, r = 0.495 — collapses
to −0.057 once size is controlled.

## Open decisions blocking the next phase

1. ~~**Is county size a control or a feature?**~~ **Answered 2026-08-05: it is a
   control.** The consumer is Comcast **FreeWheel Revenue Science**, and one row
   in their training data is an impression, ad request, auction, household, or
   device. Every one of those carries a per-row target, so county population is
   not on the left-hand side — it sets how many rows a county contributes, not
   what any row is worth. *Asserted by Max, pending written confirmation from the
   consuming team*; `docs/downstream_target.md` Part 2's invalidation conditions
   are the rollback path.

   Two things this does **not** mean. It does not mean regressing size out of
   every pillar — advertiser demand really is denser in metro markets, so a
   size-loaded column can be earning its slot causally. The operation is marginal
   lift over a `target ~ log_population + density` baseline, scored per column.
   And it does not mean the fusion step was ever as blocked as this document
   claimed: that procedure is identical under either answer.

   What it did change, concretely: Source D's ten raw per-commodity tonnages
   moved into `SIZE_COLUMNS` (`pillar_matrix.py`, 2026-08-05), which cost
   nothing — matrix-sweep mean lift +0.0847 → +0.0851, definitional share
   0.522 → 0.514, one noise-level target dropped.

   **The question that took this slot — the geo join key — is also answered:
   DMA.** Same provenance and same caveat. It is the worse of the three possible
   answers and it opens a larger problem than resolution loss: with 210 DMAs and
   millions of impressions each, a DMA fixed effect is cheap and precise, and any
   static DMA-keyed feature is exactly collinear with it. Cross-sectionally,
   `E_macro` adds zero over `C(dma)` — and cross-sectional association is what
   every test in this repo measures. The value proposition has to move to cold
   start, partial pooling, temporal transfer and interpretability, none of which
   has been tested here.

   **But the impression row almost certainly carries ZIP**, so county is
   derivable and the DMA join is a *choice* rather than a constraint. The
   reasoning for preferring county was that the fixed-effect objection scales
   with units per parameter — fatal at 210 DMAs with millions of rows each,
   weaker at 3,143 counties where most units are thin.

   **That preference is no longer supported as stated.**
   `analysis-output/cross-source/external-target-findings.md` §12 measured both
   halves of the grain penalty against five public targets: losing rows costs
   −0.122 mean lift, aggregating gains **+0.106**, and they roughly cancel. On
   three of five targets an aggregated 208-market arm matches or beats full
   county grain. County wins on `broadband_rate` and `mean_commute_minutes` only
   — the first mattering more than a 2-of-5 tally suggests, since it is closest
   to the consumer's domain and is where the aggregate arm goes negative.
   **The grain question is open; county is not established as better.** Plan and
   caveats in `docs/plans/dma_regrain.md`.

   The ZIP claim is recollection, not verification, and under the access
   constraints above it **cannot be verified from inside this repo**. It is
   recorded as an assumption to be put to the commissioning side, and the
   county-grain case is a *recommendation this project makes* rather than a
   decision it implements. Both grains are buildable here regardless — the
   ZIP→county crosswalk (HUD-USPS) and the county→DMA crosswalk are both public.
2. **Does B ↔ E get privileged weight?** It is roughly five times stronger than
   anything else surviving the size control.
3. ~~**Confirm the Source A *embedding* cut.**~~ **Closed 2026-08-05: the cut is
   final.** The decision was ours to make and it is made. The pillar was never
   cut — it ships 29 typed columns; only the `bge-m3` step is gone. Head to head
   against the typed block the embedding is a statistical tie (13/28, p = 0.76),
   and a tie does not justify a 2.2GB model download plus CPU inference over
   3,144 articles. `data/source_a_embeddings.parquet` is retained and no longer
   regenerated, so reinstating remains a `git revert` if anyone ever revisits it.

## Stage outcome: this ends in a go/no-go

Recorded 2026-08-05. The commissioning side expects a **go/no-go decision on
whether `E_macro` is worth continuing**, which fixes what the deliverable is:

**A validated claim, not a pile of documented columns.** That distinction was
open — `docs/plans/source_a_next_steps.md` question 3 asks it directly — and it
is now answered. "Here are 29 documented columns, downstream decides" would be
finished today; a go/no-go is not, because the repo cannot yet say whether the
pillars are useful to anyone.

Two things stand between here and an honest verdict:

1. **Every test in this repo is pillar-versus-pillar**, which measures coherence
   and redundancy rather than usefulness. Usefulness needs a target outside the
   six pillars.
2. **The fixed-effect objection is unanswered.** If the consumer joins at DMA
   grain, a 210-level dummy supplies everything `E_macro` could, and no
   cross-sectional correlation in `analysis-output/` is evidence against that.

Walking into the go/no-go with either unresolved means the defensible
recommendation is "no." **The external-target benchmark is therefore the
deliverable**, not a side quest — see `docs/plans/dma_regrain.md` Phase 3 and
`docs/plans/source_a_next_steps.md` Plan 4.

### The fusion step is deferred, deliberately

Decided 2026-08-05, reversing this document's earlier framing of fusion as the
next build. Two reasons:

- **Fusion does not decide the go/no-go.** Evidence does. Spending the remaining
  weeks on assembly rather than validation arrives at the decision point with the
  same unanswerable question.
- **Fusing now bakes in a grain that is not settled.** This is the same trap the
  size confound posed, one level up.

It also costs less to defer than the framing implies: `pillar_matrix.build_matrix()`
already joins all six pillars into a 3,144 × 124 matrix with per-pillar blocks.
What remains of "fusion" is packaging — freeze the schema, settle the imputation
policy, choose a serving format.

### Sibling tiers: one exists, one does not

`E_local` is under construction by another contributor at **H3 res-8**.
`E_census` does not exist. So the three-tier stack described above is currently
one tier at county grain, one at hex grain, and one missing — served side by side
by design, at grains that have never been reconciled, with nobody owning the
reconciliation. Flagged here rather than solved; it is outside this repo's scope
but it is a real gap and the go/no-go should name it.

## Next work, in order

1. Extend the sweep beyond single scalars — full 20-column B LQ vector against E.
2. ~~Re-run the size control with Census population instead of the tax-return
   proxy.~~ **Done 2026-08-04** — `scripts/county_population.py`, results in
   `source-a-findings.md` §18. Same tiering, same verdicts, one self-reference
   removed.
3. **Build the fusion — unblocked 2026-08-05.** The size question is settled, so
   the confound cannot get baked in. Structure it as
   `target ~ log_population + density` baseline, then pillars added in
   cleanliness order with permutation importance at each step and grouped,
   spatially blocked CV throughout (`docs/downstream_target.md` Part 2,
   validation plan steps 4–6).
4. **Ask the geo-key question** — county, ZIP, or DMA. It sits where the
   rate-versus-count question used to, and a DMA answer would restate every power
   figure in the repo.
5. Hand off to the FreeWheel Revenue Science models: publish the pillar parquets
   behind the feature store with a frozen schema and documented null semantics.
   Sources A and E are ready (`docs/source_a_feature_schema.md`,
   `docs/source_e_feature_schema.md`); every pillar now carries an
   `as_of_date` (`outputs/pillar_vintages.csv`). B, C, D, and F still need their
   own schema docs.

   **Hand them the grain-mismatch warning first.** Impression-level rows carry
   only 3,143 distinct feature values, so effective n is the county count. Random
   k-fold will make `E_macro` look good in evaluation and do nothing in
   production. Cluster standard errors by `fips_code`; use grouped, spatially
   blocked folds. This matters more to them than which columns ship
   (`docs/downstream_target.md` Part 2, trap 1).
