---
type: results-report
date: 2026-07-13
experiment_line: source-c
round: 1
purpose: initial-ingestion-and-findings
status: active
---

# Source C — FRED Time-Series Slope Derivatives (Economic Velocity)

## 1. Executive Summary

Source C pulls annual unemployment-rate and real-GDP series from FRED for all 3,144 US counties/county-equivalents and computes each county's rolling 3-year first derivative (Δy/Δt) — the "Economic Velocity" pillar of `E_macro`. Coverage is effectively complete: 3,080 counties (98.0%) have both series, 63 (2.0%) have unemployment only (GDP series missing), and exactly one county — Kalawao County, HI, population ~90, the smallest county in the US — has neither. **Two findings depart from what the pre-scoping spec and extended proposal assumed going in**, and both are documented rather than silently patched over:

1. **LAUS is used at annual, not monthly, cadence** (§2) — the spec's assumed monthly series naming isn't FIPS-derivable; the annual series is, and it also removes the need for deseasonalization the spec flagged as a risk.
2. **GDP velocity is measured in absolute dollars, not a growth rate**, which means the "top diverging pairs" and quadrant-extreme counties are dominated by the largest metro economies (King, Harris, Maricopa, LA, New York counties) rather than by counties experiencing unusually sharp *acceleration* (§3.4, §5). This is a real gap between the proposal's "momentum" narrative and what this metric literally measures.

The dominant macro pattern across counties is a broad, common-direction shift, not sector-specific divergence: 63.8% of counties fall in the "Growing but Loosening" quadrant (GDP still expanding, but unemployment also rising) and mean unemployment velocity is positive (+0.19 pp/year) almost everywhere. Read together with a near-zero cross-sectional correlation between the two velocities (r = 0.009), this reads as a labor market normalizing upward off historically low 2022 unemployment while GDP keeps growing — a level shift shared across nearly all counties, not a signal that differentiates fast-movers from slow-movers the way the proposal's "different quadrants" framing implied it would.

## 2. Data & Setup

`scripts/ingest_source_c.py` pulls two FRED series per county:

- **Unemployment rate** (annual, %): `LAUCN{FIPS}0000000003A`
- **Real GDP** (annual, thousands of chained-2017 $): `REALGDPALL{FIPS}`

Both are FIPS-derivable directly from `data/county_crosswalk.parquet` — no per-county lookup table, unlike Source A's Wikipedia-title edge cases. This is a deliberate deviation from the pre-scoping spec, which describes LAUS as monthly: FRED's monthly county series use ad-hoc state+county-abbreviation codes (e.g. `ALAUTA1URN`) that aren't derivable from FIPS, so using them would require either a per-county `series/search` call (3,142 extra requests) or a hand-built lookup table. The annual series avoids both, and — since annual data has no seasonal component — also makes moot the spec's flagged risk about deseasonalization padding (48-month lookback, only 12 months of padding). Velocity is a plain 3-year difference: `(value[latest_year] - value[latest_year - 3]) / 3`.

Requests are rate-limited to ~100/min (FRED's actual limit is 120/min) via a sleep-based token bucket; the full run is 2 × 3,144 = 6,288 requests, ~65 minutes. A transient FRED `500` mid-run on the first attempt (Kearney County, NE's GDP series) exposed a real bug — `_fetch_velocity` only caught "series not found" and "insufficient history," not the broader transient-failure case, so a single server hiccup crashed the entire batch. Fixed by (1) broadening the catch to isolate any per-series `SourceCError`, matching the stated per-county-isolation design principle, and (2) adding a bounded retry (3 attempts, 2s backoff) for transient network/5xx errors specifically, so a genuine blip resolves without even needing to fall back to "missing data." Both behaviors were verified with targeted mock tests before the full run.

Output: `data/source_c_fred.parquet`, 3,144 rows, columns `county_name`, `fips_code`, `unemployment_velocity`, `unemployment_rate_latest`, `unemployment_latest_year`, `gdp_velocity`, `gdp_latest`, `gdp_latest_year`.

## 3. Main Findings

### 3.1 Coverage

| | Count | % |
|---|---|---|
| Full (both series) | 3,080 | 98.0% |
| Partial (unemployment only) | 63 | 2.0% |
| Failed (neither series) | 1 | 0.03% |

The single full failure, Kalawao County, HI, is the former Kalaupapa leper colony settlement on Molokaʻi — the smallest county in the US by population and a plausible genuine absence from both FRED series, not a pipeline defect.

### 3.2 GDP coverage gap has two distinct root causes

64 counties are missing a GDP series (`scripts/analyze_source_c_gdp_coverage.py`), concentrated in two unrelated patterns:

- **Virginia independent cities (51 of 64, ~80%)**: Virginia's independent-city structure — the same edge case that required a 37-entry lookup table for Source A's Wikipedia titles — apparently falls below whatever population/data threshold BEA uses to publish county-level GDP for some (not all) of these cities. Winchester City and Harrisonburg City were the two hits found during pre-full-run sampling; the full run surfaces all ~51 at once.
- **Connecticut (9 of 64, all 9 of its geographies)**: not an independent-city issue at all. Connecticut dissolved its traditional counties as functioning administrative units in 2022, replacing them with 9 new Planning Regions (FIPS-like codes 09110–09190, visible in the crosswalk as e.g. "Capitol Planning Region, Connecticut"). BEA/FRED's county GDP series apparently hasn't backfilled data under the new geography yet — every single Connecticut geography in the crosswalk is missing GDP, none are missing unemployment.

Alaska (2) and Hawaii (2) round out the remainder — small/remote geographies plausibly below a reporting threshold, not further investigated this round.

### 3.3 Momentum quadrants

Classifying the 3,080 fully-covered counties by sign of GDP velocity (accelerating/decelerating) × sign of unemployment velocity (tightening/loosening):

| Quadrant | Count | % |
|---|---|---|
| Growing but Loosening (GDP↑, unemployment↑) | 1,966 | 63.8% |
| Decelerating & Loosening (GDP↓, unemployment↑) | 541 | 17.6% |
| Accelerating & Tightening (GDP↑, unemployment↓) | 465 | 15.1% |
| Slowing but Tightening (GDP↓, unemployment↓) | 108 | 3.5% |

Nearly two-thirds of counties share the same quadrant. This isn't the differentiated "different markets map to different quadrants" picture the proposal's narrative example describes — it reads instead as a common macro trend (unemployment normalizing upward across almost the whole country over this 3-year window, alongside continued GDP growth almost everywhere) dominating whatever county-level differentiation exists underneath it.

### 3.4 Unemployment velocity vs. GDP velocity: near-zero correlation

Pearson r = 0.009 across the 3,080 fully-covered counties — essentially no linear relationship, which runs counter to a naive Okun's-law expectation (GDP growth co-occurring with falling unemployment). Given §3.3's finding that most counties share the same quadrant (both rising), this makes sense as a composition effect: with a common direction shared broadly across the cross-section, there's little of the *opposing*-direction variation that would produce a negative correlation. The velocity data captures a level shift more than a cross-sectional relationship in this window.

### 3.5 Geographic proximity vs. economic divergence

`scripts/analyze_source_c_similarity.py` standardizes both velocity axes (z-score) and ranks geographically-close county pairs (bottom quartile of distance) by economic distance. Every one of the top-20 pairs includes **King County, WA**, paired with a wide variety of geographic neighbors across four states. This is not a "two specific neighbors have diverging economies" story — it's a single outlier appearing repeatedly, and it points to a real methodological limitation (§5), not a data error.

## 4. Figure-by-Figure Interpretation

- `analysis-output/figures/source-c-figure-01-quadrants.png`: momentum-quadrant scatter, all 3,080 counties, colored by quadrant. Visually confirms §3.3's finding — one quadrant clearly dominates by point density.
- `analysis-output/figures/source-c-figure-02-velocity-distributions.png`: side-by-side histograms of both velocity axes. Unemployment velocity is right-shifted off zero (broad increase); GDP velocity is extremely heavy-tailed (see §5).
- `analysis-output/figures/source-c-figure-03-proximity-vs-divergence.png`: geographic-distance-vs-economic-distance hexbin. The King County outlier effect is visible as a distinct band of high-economic-distance points regardless of geographic distance.
- `outputs/source_c_map_unemployment.html`, `outputs/source_c_map_gdp.html`: interactive US choropleths, one per velocity axis.
- `outputs/source_c_quadrants.html`, `outputs/source_c_similarity.html`: interactive versions of figures 1 and 3.

## 5. Limitation: Absolute-Dollar GDP Velocity Measures Economy Size, Not Acceleration Rate

The pre-scoping spec and extended proposal both specify `Δy/Δt` on the raw GDP series (thousands of dollars), which is exactly what was implemented. But the practical effect: the five largest `gdp_velocity` values belong to King County WA, Harris County TX, Maricopa County AZ, Los Angeles County CA, and New York County NY — precisely the five largest county economies by `gdp_latest`. A large economy produces a large absolute-dollar year-over-year change even at a modest *percentage* growth rate, so this metric is highly correlated with economy size, not purely with "how fast is this county's economy accelerating" the way the proposal's narrative example ("a market ... experiencing sharp macroeconomic acceleration") suggests. z-score standardization (used in §3.5) rescales variance but does not fix a heavy-tailed distribution — the same handful of mega-economies still dominate any distance-based ranking.

This isn't a bug in this round's implementation; it's a fair reading of what the spec's literal formula produces. A **relative** velocity (e.g. `gdp_velocity / gdp_latest`, a percentage growth-rate derivative) would more directly match the proposal's "acceleration" framing and is a natural follow-up metric — flagged here rather than substituted silently, since the current column matches what was scoped and tested this round.

## 6. Limitations / Open Items

- GDP coverage gap (2.0%, §3.2) is real and has two distinct causes; not further backfilled this round (no per-county fallback source identified for VA independent cities or CT planning regions).
- GDP velocity is absolute-dollar, not percentage-based (§5) — a real limitation for any downstream use that wants "acceleration rate" rather than "size of dollar change."
- No cross-source validation yet: this round doesn't join Source C against Source A (or B, once built) the way Source A's findings checked against `E_macro_extendedProposal.pdf`'s narrative claims for economic-transition text. That join is natural once at least one more numeric source (B) exists.
- The "close but diverging" analysis (§3.5) is dominated by one outlier; a version that log-transforms or percentage-normalizes GDP velocity before standardizing would likely surface different, more genuinely "neighboring counties with different momentum" pairs.

## 7. Next Actions

1. Consider adding a relative/percentage GDP velocity column (`gdp_velocity_pct = gdp_velocity / gdp_latest`) as a second metric alongside the absolute-dollar one already in `data/source_c_fred.parquet`, to give downstream `E_macro` consumers a size-invariant momentum signal.
2. Re-run `analyze_source_c_similarity.py` against that relative metric once available, to see whether the "close but diverging" story changes once King County's dollar-scale dominance is removed.
3. When Source B (BLS QCEW) is built, join against Source C to test the proposal's stated synergy claim directly — "Source B identifies the exact industrial engines driving [Source C's] speed" (`E_macro_extendedProposal.pdf` §3.2).
4. No action planned this round for the VA/CT GDP gaps; revisit if a downstream consumer needs those 64 counties fully populated.

## 8. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_c.py` → `data/source_c_fred.parquet` (`uv run --env-file .env scripts/ingest_source_c.py`)
- Quadrants: `scripts/analyze_source_c_quadrants.py` → `outputs/source_c_quadrants.{csv,html}`
- Proximity/divergence: `scripts/analyze_source_c_similarity.py` → `outputs/source_c_similarity_pairs.csv`, `outputs/source_c_similarity.html`
- GDP coverage: `scripts/analyze_source_c_gdp_coverage.py` → `outputs/source_c_gdp_coverage.csv`
- Maps: `scripts/visualize_source_c.py` → `outputs/source_c_map_{unemployment,gdp}.html`
- Stats/figures: `scripts/generate_source_c_insights.py` → `analysis-output/source_c_stats.json`, `analysis-output/figures/source-c-*.png`, `analysis-output/figures/source-c-numeric-summary.md`
- Presentation notebook: `analysis-output/source_c_key_findings.ipynb`

## 9. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf`, Source C section)

The proposal frames Source C as capturing "which direction a market is moving and how fast," illustrated with an example of two mid-income markets — one with slowing job growth, one accelerating — that should "map to a completely different quadrant." The data partially supports and partially complicates this:

- **Supported**: the quadrant mechanism works as designed and does differentiate counties (§3.3) — 15.1% of counties are in the strongest "Accelerating & Tightening" quadrant while 3.5% are in the weakest "Slowing but Tightening," a real spread.
- **Complicated**: the proposal's illustrative framing implies quadrant assignment mainly reflects *idiosyncratic* county-level momentum differences. What §3.3–3.4 actually find is a dominant *shared* macro trend (rising unemployment almost everywhere, most counties still GDP-growing) with near-zero cross-sectional correlation between the two axes — the quadrant split is real, but a large share of what determines it is common-trend timing (was a county's unemployment still above or already back to its post-2022 low at measurement time) rather than purely local economic character.
- **Not yet testable**: the proposal's stated synergy with Source B ("identifies the exact industrial engines driving that speed") requires Source B, which doesn't exist yet (§7, item 3).
- **New finding not anticipated by the proposal**: the absolute-dollar GDP velocity metric (§5) measures economy size more than acceleration rate — worth a metric-design revisit before this becomes a proposal-alignment problem in a later round.
