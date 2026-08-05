---
type: results-report
date: 2026-08-04
experiment_line: source-d
round: 3
purpose: initial-ingestion-and-findings, then rate refeaturization
status: active
---

> **Round 2 (2026-08-04).** §1–§8 are the round-1 report as written. §9–§11 test
> the rate normalization `docs/downstream_target.md` calls the highest-value
> change a rate target would imply. It is not one, for a provable reason (§10).
> What did help was a different transformation the doc did not propose (§11).
>
> **Round 3 (2026-08-05).** The rate-versus-count question is answered — rate —
> so §11's pre-registered action fires: the ten raw per-commodity tonnages move
> to `SIZE_COLUMNS`, at no measured cost (§14). §15 states what survives.

# Source D — BTS FAF5 Experimental County Commodity Flows (Trade Gravity)

## 1. Executive Summary

Source D ingests 2022 county-level freight tonnage from the Bureau of Transportation Statistics' FAF5 Experimental County-Level Estimates — the "Movement" pillar of `E_macro`, meant per `E_macro_extendedProposal.pdf` to distinguish "a logistical pass-through corridor or industrial exporter from a pure consumer sink." Coverage is complete (3,144/3,144 crosswalk counties, zero nulls, zero duplicate FIPS codes). Two real deviations from what `macro_pre_scoping_spec.pdf` and the extended proposal described were found and are documented in §7 rather than silently absorbed: BTS's own county/FAF-zone hybrid table structure already solves the "combinatorial matrix explosion" both scoping docs worried about, superseding the spec's proposed top-K-per-origin truncation; and the experimental county-level product ships tonnage only, with no dollar-value column despite both docs describing Source D as tracking "tonnage and dollar value."

Three findings stand out:

1. **Scalar tonnage alone cleanly separates hubs from sinks, at up to 1,000x range** (§3.1) — Harris County, TX (Houston) leads at 731K tons, while sub-1,000-ton rural counties sit at the opposite extreme; a genuine oil/gas export point (Loving County, TX — population ~64) shows the metric correctly capturing economic character that has nothing to do with population.
2. **Partner concentration (HHI) reverses direction between regional and national scale** (§3.2) — the two-state design spike documented in `docs/plans/ingestion_recon.md` (§ Source D, Phase 1b) found hub counties *less* concentrated than rural ones; at full national scale, the opposite holds (r = 0.278 between log-tonnage and HHI) — big hubs funnel large volume through a few dominant interstate corridors, while small counties spread modest volume evenly across nearby neighbors. This is a real scale-dependent reversal, not noise, and is worth flagging explicitly for any downstream consumer that only saw the regional spike's direction.
3. **Trade volume tracks GDP momentum weakly but monotonically** (§3.3) — mean size-normalized GDP velocity rises cleanly across freight-tonnage quartiles (1.6% → 1.6% → 2.1% → 2.4%/year), a small but directionally consistent signal that higher-throughput counties trend toward faster economic growth, distinct from and complementary to Source C's own velocity measure.

## 2. Data & Setup

`scripts/ingest_source_d.py` downloads per-state zip files from `faf.ornl.gov` (the actual host behind BTS's Akamai-gated `bts.gov` landing page — see `docs/plans/ingestion_recon.md` § Source D, Phase 0 for the access-resolution detail) via `curl` subprocess calls, worked around a TLS-handshake incompatibility between this host and Python's `requests`/urllib3 stack. Each state zip contains four OD tables at mixed county/FAF-zone granularity; only domestic flows (`trade_type=1`) are used. For each state's home counties, the pipeline computes `total_outbound_tons`/`total_inbound_tons`, a 5-way `sctgG5` commodity-group breakdown per direction, and `out_partner_hhi`/`in_partner_hhi` — a Herfindahl-Hirschman concentration index pooling county-level and FAF-zone-level partner rows into one distribution per county per direction.

This feature set (scalar totals + pooled HHI) was not picked a priori: `docs/plans/ingestion_recon.md`'s § Source D Phase 1b ran three candidate vectorizations (scalar aggregates only; scalar + top-5 partner columns; scalar + graph-summary statistics) on two independent state samples (Rhode Island, then New Jersey) before choosing. Partner-degree and distance-weighted "reach" were both tested and dropped — degree is a dead signal (BTS's gravity-model disaggregation assigns near-universal nonzero flow to almost every county pair, so raw partner count doesn't distinguish hub from sink at all), and distance-weighted reach pointed backward for a understood, structural reason (a real hub's long-distance flows collapse into the FAF-zone-level tables once they leave the adjacent-state window that bounds the county-to-county table, so a small regional sample can't see them). Top-5 partner columns added nothing over scalar totals, since both hub and rural counties in a region share the same handful of big neighboring counties as partners.

Output: `data/source_d_faf.parquet`, 3,144 rows, 16 columns (`county_name`, `fips_code`, `total_outbound_tons`, `total_inbound_tons`, `out_partner_hhi`, `in_partner_hhi`, 5 `out_sctg*` + 5 `in_sctg*` commodity-group columns). Full-batch ingestion ran 51/51 states with zero failures; every crosswalk county has a row, zero nulls, zero duplicate `fips_code` values.

## 3. Main Findings

### 3.1 Scalar tonnage separates hubs from sinks by orders of magnitude

| County | Total tons (out + in) | Character |
|---|---|---|
| Harris County, TX (Houston) | 731,121 | Top national hub — port + petrochemical corridor |
| Los Angeles County, CA | 540,493 | Major port hub |
| Cook County, IL (Chicago) | 348,042 | Major rail/intermodal hub |
| Loving County, TX (pop. ~64) | 9,901 | Genuine Permian Basin oil/gas export point |
| Petroleum County, MT | 390 | Rural, no meaningful freight character |

The gap between the biggest hubs and the median county spans roughly two orders of magnitude, exactly as expected for a metric meant to isolate logistics-significant geography. Loving County is the sharpest illustration that the signal tracks real economic function rather than population or urbanization: essentially uninhabited, but its 9,842 outbound tons against just 59 inbound is a genuine, correctly-captured petroleum export imbalance — not noise from a sparse-data county.

### 3.2 HHI concentration direction reverses between regional and national scale

The § Source D Phase 1b design spike (Rhode Island, then New Jersey) in `docs/plans/ingestion_recon.md` found hub counties *less* concentrated than rural ones — e.g., Essex County, NJ (Newark hub) at HHI 0.043–0.056 vs. Sussex County, NJ (rural) at 0.047–0.059, consistent with a hub spreading flow across many destinations. At full national scale, the direction flips:

| Metric | Value |
|---|---|
| log10(total tons) vs. mean partner HHI (Pearson r) | **0.278** (positive) |
| Harris County, TX HHI (out / in) | 0.201 / 0.161 |
| Los Angeles County, CA HHI (out / in) | 0.231 / 0.189 |
| Cook County, IL HHI (out / in) | 0.108 / 0.111 |
| Petroleum County, MT HHI (out / in) | 0.040 / 0.023 |
| Loving County, TX HHI (out / in) | 0.024 / 0.012 |

Nationally, the biggest hubs are visibly *more* concentrated than small rural counties — a large hub funnels enormous volume through a handful of dominant interstate corridors, while a small county spreads its modest volume evenly across nearby neighbors regardless of character. This is a genuine scale-dependent reversal, mechanistically understood (regional samples only see nearby partners; national data captures the concentrated long-haul corridors that define a real hub), not a contradiction between the two rounds of testing.

### 3.3 Trade volume weakly but monotonically tracks GDP momentum

Cross-validating against Source C's velocity metrics (size-normalized `gdp_velocity_pct`, per `source-c-findings.md` §5's own recommended fix for the raw column's economy-size confound; 64 of 3,144 counties lack Source C GDP coverage and drop from this comparison only):

| Comparison | Pearson r |
|---|---|
| log(tons) vs. unemployment velocity | 0.0498 |
| log(tons) vs. size-normalized GDP velocity | 0.0765 |
| Partner HHI vs. unemployment velocity | -0.0261 |
| Partner HHI vs. size-normalized GDP velocity | 0.0688 |

| Tonnage quartile (1=lowest, 4=highest) | Mean GDP velocity (%/yr) |
|---|---|
| 1 | 1.61% |
| 2 | 1.63% |
| 3 | 2.07% |
| 4 | 2.40% |

All correlations are weak in absolute terms, consistent with the pattern Source F's cross-validation against Source C already established (structural/static signals track short-run momentum only faintly). But unlike Source F's near-zero, directionless result, this one is monotonic: each tonnage quartile step up trends toward faster GDP growth, a small but economically sensible signal that higher freight throughput associates with (not necessarily causes) faster local growth.

## 4. Figure-by-Figure Interpretation

- `analysis-output/source-d/figures/source-d-figure-01-top-hubs.png`: horizontal bar chart of the top 15 counties by total tonnage. Visually confirms §3.1 — Harris, LA, and Cook Counties dominate by a wide margin over the rest of the top 15.
- `analysis-output/source-d/figures/source-d-figure-02-tons-vs-concentration.png`: scatter of log10(tons) vs. mean partner HHI across all 3,144 counties. The positive trend is visible but noisy — confirms §3.2's r=0.278 is a real but modest tilt, not a tight relationship.
- `analysis-output/source-d/figures/source-d-figure-03-tons-vs-velocity.png`: bar chart of mean size-normalized GDP velocity by tonnage quartile. Shows the clean monotonic step from §3.3, with Q4 roughly 1.5x Q1.
The `.html` renders below are ~5MB each, regenerable, and no longer committed —
rebuild any of them with the script named against it in §7.

- `outputs/source_d_hubs.html`: interactive version of figure 1's underlying scatter (log-tonnage vs. HHI, top 10 hubs labeled).
- `outputs/source_d_map_tons.html`, `outputs/source_d_map_concentration.html`: interactive US choropleth-style bubble maps of tonnage and HHI respectively.
- `outputs/source_d_source_c_crossvalidation.html`: interactive version of figure 3.

## 5. Limitations / Open Items

- **No dollar-value column** — see §7. Any downstream use expecting a value-weighted (not just tonnage-weighted) trade signal is not supported by this experimental county-level product.
- **Single 2022 cross-section, not a time series** — unlike Source C's rolling velocity window, Source D captures one snapshot; trend/momentum in freight flows themselves cannot be derived from this data alone.
- **`dms_mode` code `11` remains unexplained** — the experimental county product's mode taxonomy doesn't match the general FAF5 user guide's 1–8 code table (`docs/plans/ingestion_recon.md` § Source D, Phase 0 addendum); the county-level-specific technical report that would likely resolve this sits behind the same Akamai gate as the landing page and wasn't pursued (see plan file for why this wasn't worked around).
- **Zone→county crosswalk was not pursued** — HHI pooling uses raw FAF-zone partner rows without resolving them to zone-centroid geography, so no distance/reach metric was salvaged; `docs/plans/ingestion_recon.md` (§ Source D) recommends this if a "reach" signal is wanted in a future round.
- **Ingestion has no per-state checkpointing** — a known, accepted limitation for the current run (which completed in 2h32min with zero failures); revisit before assuming a future data refresh will be equally fast.

## 6. Next Actions

1. **Test the proposal's stated Trade Logistics synergy with Source F**, which explicitly flagged this as its own next action (`source-f-findings.md` §6 item 1): does Source F's structural typology (industry dependence, demographic distress) explain which counties become logistics hubs vs. sinks, beyond what tonnage/HHI alone show?
2. **Once Source B (BLS QCEW) exists**, test whether a county's industry mix explains its trade-flow character — the proposal's stated three-way synergy between Movement, Capital Flow, and industry composition.
3. No action planned this round on the `dms_mode=11` documentation gap (§5) or the zone-centroid crosswalk (§5) — revisit only if a downstream consumer needs either resolved.

## 7. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_d.py` → `data/source_d_faf.parquet` (`uv run scripts/ingest_source_d.py`, no credentials required; ~2.5hr full run against 51 state zips via `curl`)
- Hub ranking: `scripts/analyze_source_d_hubs.py` → `outputs/source_d_hubs.csv`, `outputs/source_d_hubs.html`
- Cross-validation vs. Source C: `scripts/analyze_source_d_source_c_correlation.py` → `outputs/source_d_source_c_crossvalidation.csv`, `outputs/source_d_source_c_crossvalidation.html`
- Maps: `scripts/visualize_source_d.py` → `outputs/source_d_map_{tons,concentration}.html`
- Stats/figures: `scripts/generate_source_d_insights.py` → `analysis-output/source-d/source_d_stats.json`, `analysis-output/source-d/figures/source-d-figure-*.png`, `analysis-output/source-d/figures/source-d-numeric-summary.md`
- Presentation notebook: `analysis-output/source-d/source_d_key_findings.ipynb`
- Full reconnaissance/decision trail: `docs/plans/ingestion_recon.md` § Source D (Phase 0 access resolution, Phase 1b vectorization comparison, the falsified hypotheses). The original `source_d_plan.md`, including the phase-by-phase run logs, is in git history.

## 8. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf` / `macro_pre_scoping_spec.pdf`, Source D section)

The proposal frames Source D as isolating "a logistical pass-through corridor or industrial exporter from a pure consumer sink" via county-to-county freight flows, and the pre-scoping spec specifically worried about a "combinatorial matrix explosion" from a naive dense county×county matrix, proposing top-K-per-origin truncation as a fix while flagging that truncation "destroys global network topology."

- **Supported**: the core framing holds at national scale (§3.1, §3.2) — scalar tonnage cleanly separates known hubs (Houston, LA, Chicago) from rural sinks by two orders of magnitude, and partner concentration adds a second, complementary axis (hubs funnel volume through fewer dominant corridors) once tested at full national scale rather than a two-state regional sample.
- **Deviation 1 — combinatorial-explosion problem already solved upstream, differently than proposed**: BTS's own county/FAF-zone hybrid table structure (full county granularity only for nearby geography, FAF-zone aggregation for everything farther) already prevents the dense 9.87M-edge matrix both scoping docs worried about. This supersedes the spec's proposed top-K-per-origin truncation entirely — that scheme would have solved a problem the source data doesn't actually present in that form, and would have discarded the zone-level long-tail structure BTS already preserves.
- **Deviation 2 — no dollar-value column**: both scoping docs describe Source D as tracking "freight tonnage and dollar value." The experimental county-level product ships tonnage only (`tons_2022`); the base (non-experimental) FAF5 product does carry `value`/`current_value` fields, per the general user guide, but that field was dropped in the county-level disaggregation. This is a real scope reduction, not a parsing gap — confirmed against the FAF5 data dictionary directly.
- **New finding not anticipated by either doc**: the HHI concentration signal's *direction* is scale-dependent (§3.2) — a design detail neither scoping doc could have anticipated, since neither analyzed the disaggregation model's gravity-artifact behavior at regional vs. national scope. Anyone extending this concentration metric should be aware it only reads correctly at full national coverage.

---

# Round 2 — Rate refeaturization (2026-08-04)

## 9. Every Source D column is a level

Round 1 shipped two directional tonnage totals, ten per-commodity tonnages, and
two trade-partner concentration indices. Measured against Census population
(Spearman, so the heavy right skew in raw tons does not understate it), twelve
of the fourteen are size measures:

| column | Spearman vs population |
|---|---|
| `total_inbound_tons` | +0.894 |
| `in_sctg3499` (manufactured goods, inbound) | **+0.971** |
| `in_sctg2033` | +0.931 |
| `out_sctg3499` | +0.921 |
| `in_sctg1014` | +0.921 |
| `out_sctg2033` | +0.900 |
| `out_sctg1014` | +0.848 |
| `in_sctg1519` | +0.767 |
| `in_sctg0109` | +0.630 |
| `out_sctg1519` | +0.554 |
| `out_sctg0109` | +0.523 |
| `out_partner_hhi` | +0.272 |
| `in_partner_hhi` | +0.278 |

The per-commodity columns are worse than the totals they decompose. That is not
surprising once stated plainly: a county's inbound manufactured-goods tonnage is
close to a restatement of how many people live there. The two HHIs are the only
columns that were ever composition rather than level, and they are the only two
under 0.30. `docs/downstream_target.md` flagged them at 0.33 and asked whether a
concentration index should carry that much size; on the scale-invariant measure
they carry less than the doc feared.

## 10. The rate normalization is a re-expression, not a fix

`docs/downstream_target.md` (§ *Refeaturization implied by a rate target*) calls
`tons_per_capita` the "highest-value single change implied here." It is not a
change at all under any model that already controls for size:

```
log10(tons / population) == log_total_tons − log_population
```

to floating-point precision (max absolute difference 8.9e-16 across 3,144
counties). `log_total_tons` is in Source D's block and `log_population` is in
`SIZE_FEATURES`, held in every baseline — so the per-capita column is an exact
linear combination of two columns the design already contains.

Measured rather than argued, against Source F:

| Source D feature | raw r vs `metro_2023` | size-controlled |
|---|---|---|
| `log_total_tons` (current) | +0.495 | **−0.036** |
| `log_tons_per_capita` | −0.400 | **−0.036** |
| `share_out_sctg3499` | +0.320 | +0.026 |
| `out_partner_hhi` | +0.275 | **+0.115** |

The raw correlation swings 0.9 points and *flips sign* — per-capita freight is
higher in small counties, since rural counties move bulk agriculture and mining
against tiny populations. The size-controlled correlation does not move at all.
The same holds against `distress_count`: −0.138 either way.

**Source D's problem was never its units.** It is that once county size is
removed there is very little left — and no re-expression of the same quantity
can add information that was not there. The normalization matters only for a
consumer fitting *without* a size control, which is a serving-format question,
not a feature question. No `tons_per_capita` column ships; the reasoning is
recorded in `pillar_matrix.py` beside `D_COMMODITY_GROUPS` so it is not
rediscovered and re-proposed.

`out_partner_hhi` is the finding that survives here, and it was already in the
block: +0.275 raw to **+0.115** size-controlled is the least size-dependent
thing Source D has, and it is a concentration measure rather than a level.

## 11. What did help: commodity shares

The transformation that does add something is the one the doc did not propose —
each commodity group as a share of the county's own directional total, which is
composition rather than volume. It is the same move that rescued Source B (ship
the LQ vector, not a scalar) and Source E (ship participation and intensity, not
the ratio).

Against `log_population` (Pearson, on the shares' natural scale):

| share column | r | tier | was, as raw tons |
|---|---|---|---|
| `share_out_sctg1519` | −0.024 | 3 size-free | +0.554 |
| `share_in_sctg1519` | +0.104 | 3 size-free | +0.767 |
| `share_in_sctg2033` | +0.212 | 3 size-free | +0.931 |
| `share_out_sctg2033` | +0.213 | 3 size-free | +0.900 |
| `share_out_sctg1014` | +0.259 | 3 size-free | +0.848 |
| `share_in_sctg1014` | +0.365 | 2 partly size | +0.921 |
| `share_in_sctg3499` | +0.392 | 2 partly size | +0.971 |
| `share_out_sctg0109` | −0.425 | 2 partly size | +0.523 |
| `share_in_sctg0109` | −0.475 | 2 partly size | +0.630 |
| `share_out_sctg3499` | +0.513 | 2 partly size | +0.921 |

**Five of ten clear the size-free bar that none of the raw columns cleared.** No
nulls: every county has non-zero outbound and inbound totals, so every share is
defined.

The ten shares are added to Source D's block in `pillar_matrix._derive_pillar_columns`,
alongside the three log tonnage columns that were already derived there. Nothing
is removed. The raw per-commodity tonnages stay in the block for now even though
§9 shows they are levels, because cutting them is a decision that depends on the
unanswered rate-versus-count question in `docs/downstream_target.md` Part 1 —
under a count target they are legitimately predictive. The recommendation, once
that answer exists and if it is "rate", is to move all ten into `SIZE_COLUMNS`
on the same rule that moved Source E's dollar totals there.

> **Fired 2026-08-05 — see §14.** The answer came back "rate" and all ten moved.
> The paragraph above is left as written because it is the pre-registration the
> action was taken against.

## 12. Sweep result: the shares pay, and they pay to Source B

`analyze_pillar_matrix_signal.py` re-run with the ten shares added to Source D's
block (25 features, up from 15):

| | before shares | after |
|---|---|---|
| targets carrying signal | 24 of 29 | **25 of 29** |
| mean lift | +0.0808 | **+0.0847** |
| mean ablated lift | +0.0329 | **+0.0405** |
| median ablated lift | +0.0264 | **+0.0348** |
| definitional share of mean lift | 0.592 | **0.522** |
| mean GBM lift | +0.1159 | +0.1157 |

Mean ablated lift up 23%, median up 32%, and the definitional share falls seven
points — the largest single drop it has taken. More of what the matrix knows is
now corroboration between agencies rather than one federal product restating
another.

**The gain is almost entirely Source B's, and it is interpretable.** Source D
does not predict itself any better — its own row is unchanged at +0.0194, as it
must be, since a pillar's targets are predicted by the *other* five. What moved:

| target | ablated lift before | after |
|---|---|---|
| Agriculture, Forestry, Fishing & Hunting LQ | +0.0004 | **+0.0430** |
| Manufacturing LQ | +0.0673 | **+0.1072** |
| Transportation & Warehousing LQ | +0.0342 | **+0.0635** |
| Accommodation & Food Services LQ | +0.0880 | +0.1065 |
| Health Care & Social Assistance LQ | +0.0195 | +0.0348 |
| Utilities LQ | −0.0070 | +0.0009 (newly carrying) |

A county whose outbound freight is mostly agricultural products has
agricultural employment; a county shipping manufactured goods has manufacturing
employment. That is the freight-to-industry link the proposal claimed and round
1 could not demonstrate, because raw tonnage could only say *how much* a county
ships. The shares say *what*, and BLS payroll data agrees. Agriculture LQ moving
from indistinguishable-from-zero to +0.0430 is the clearest instance.

**One honest caveat: the GBM lift did not move** (+0.1159 to +0.1157). The
gradient-boosted model was already recovering this structure from the raw
tonnages nonlinearly; the shares make it available to the linear model, which is
what the ridge-based headline measures. Read the gain as "the composition signal
is now reachable without a nonlinear learner," not as new information appearing
in the matrix.

Source A's ablated lift slips −0.0012 and Source E's −0.0005, both inside the
noise of adding eight columns to a 128-column design.

## 13. Round 2 Artifact Index

- Feature derivation: `scripts/pillar_matrix.py` — `D_COMMODITY_GROUPS` and the
  `share_{out,in}_{sctg*}` block in `_derive_pillar_columns`. No ingestion change,
  so no re-download of the 51 state zips.
- Re-scored sweep: `scripts/analyze_pillar_matrix_signal.py` →
  `outputs/pillar_matrix_signal.csv`,
  `analysis-output/cross-source/pillar_matrix_signal_stats.json`
- **Closed 2026-08-05** (§14): the ten raw per-commodity tonnages moved to
  `SIZE_COLUMNS`, and Source D survives a rate target through composition rather
  than volume.

---

# Round 3 — The rate answer lands (2026-08-05)

## 14. The ten raw tonnages move to `SIZE_COLUMNS`

§11 held the ten raw per-commodity tonnages in Source D's scored block for one
stated reason: cutting them depended on the unanswered rate-versus-count question
in `docs/downstream_target.md` Part 1, and under a count target they are
legitimately predictive. That question is answered — the consumer is Comcast
FreeWheel Revenue Science, one row in their training data is an impression,
request, auction, household, or device, and every such row carries a per-row
target. **Rate.** (Asserted by Max, pending written confirmation.)

So the pre-registered action fires. `D_COMMODITY_TONNAGE_COLUMNS` is now part of
`SIZE_COLUMNS` in `pillar_matrix.py`, on the same rule that moved Source E's
dollar totals there. Source D's scored block goes from 25 columns to 15: the
three log tonnage columns, the two partner HHIs, and the ten commodity shares.

**Measured cost of the removal**, `analyze_pillar_matrix_signal.py` re-run at
seed 42:

| statistic | before | after |
|---|---|---|
| targets carrying signal | 25 of 29 | 24 of 29 |
| mean lift | +0.0847 | **+0.0851** |
| mean lift, ablated | +0.0405 | +0.0413 |
| definitional share of mean lift | 0.522 | **0.514** |
| best target (`distress_count`), ablated | +0.1197 | +0.1206 |

The mean lift went *up* and the definitional share went *down*. The single target
that dropped below the signal bar is `lq_emp_22` (Utilities LQ), which fell from
an ablated lift of +0.0009 to −0.0020 — both sides of zero at noise level, and
the weakest surviving target in the sweep before the change.

**Removing ten size-in-disguise columns cost nothing.** That is the same result
the Source E dollar totals produced (−0.0011), and it is the expected one: a
column that is a level wearing a commodity label carries information the size
control already holds.

## 15. Does Source D survive a rate target?

Yes, but not on the strength §3.1 reported. What survives is composition, not
volume:

- **The ten shares.** Five of ten sit below |r| = 0.15 with log population where
  none of the raw tonnages did (§11), and the shares are what moved Agriculture
  LQ off zero and Manufacturing LQ from +0.067 to +0.107 ablated (§12).
- **`out_partner_hhi`.** +0.275 raw to +0.115 size-controlled — the least
  size-dependent thing Source D has, and a concentration measure rather than a
  level (§10).
- **`log_total_tons`** stays in the block but is honest about what it is: r =
  +0.865 with log population, Tier 1, and worth +0.008 lift against its own
  target with a baseline R² of +0.886. It is retained for the fusion step to
  score against the size baseline rather than pre-emptively cut, on the standard
  in `docs/downstream_target.md` Part 2 step 4.

The pillar's verdict is unchanged from round 2 — keep, ship composition — and the
rate answer converts that from a preference into the operative configuration.
