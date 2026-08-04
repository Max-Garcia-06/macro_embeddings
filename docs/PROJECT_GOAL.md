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

**This feeds downstream ML models at Comcast.** `E_macro` is not the deliverable
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
- **D** — keep. Weak, but clean and independent.
- **E** — keep, change the feature: the capital-to-wage ratio is a product of
  three separable drivers (R² = 0.975 on its log) and its *level* is set by the
  market year, not the county — the unweighted county mean runs 0.095 / 0.156 /
  0.108 across TY2020–TY2022. Ships the three components plus a TY2018–TY2022
  normalized mean; prefer that mean over the raw ratio. Schema frozen in
  `docs/source_e_feature_schema.md`. Done.
- **F** — keep, reclassify as a structural anchor rather than a hub-tested pillar.

Strongest surviving link: Source B Real Estate & Rental & Leasing LQ against
Source E capital-to-wage ratio, r = 0.394 raw / 0.382 size-controlled. Largest raw
effect in the sweep — D freight tonnage against F metro status, r = 0.495 — collapses
to −0.057 once size is controlled.

## Open decisions blocking the next phase

1. **Is county size a control or a feature?** If `E_macro` must distinguish counties
   *beyond* how big they are, size gets regressed out of every pillar before fusion,
   which shrinks most of the cross-pillar structure found so far. If size is part of
   the target, the raw correlations stand but the project is partly a population
   model. Everything downstream hangs on this. **This is not answerable in-repo —
   it is one question for the downstream team, written out in
   `docs/downstream_target.md` Part 1.**
2. **Does B ↔ E get privileged weight?** It is roughly five times stronger than
   anything else surviving the size control.
3. **Confirm the Source A *embedding* cut.** The pillar itself is not cut — it
   ships 29 typed columns. Only the `bge-m3` step is gone. Reinstating it is a
   `git revert`; re-running costs a 2.2GB model download plus CPU inference over
   3,144 articles. The case against reinstating is no longer that the embedding
   loses: head to head against the typed block it is a statistical tie (13/28,
   p = 0.76). It is that a tie does not justify the download.

## Next work, in order

1. Extend the sweep beyond single scalars — full 20-column B LQ vector against E.
2. ~~Re-run the size control with Census population instead of the tax-return
   proxy.~~ **Done 2026-08-04** — `scripts/county_population.py`, results in
   `source-a-findings.md` §18. Same tiering, same verdicts, one self-reference
   removed.
3. Build the fusion — **only after** the size question is settled, or the confound
   gets baked into the output.
4. Hand off to the Comcast downstream models: publish the pillar parquets behind
   the feature store with a frozen schema and documented null semantics. Sources
   A and E are ready (`docs/source_a_feature_schema.md`,
   `docs/source_e_feature_schema.md`); every pillar now carries an
   `as_of_date` (`outputs/pillar_vintages.csv`). B, C, D, and F still need their
   own schema docs.
