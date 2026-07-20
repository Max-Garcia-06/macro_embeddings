---
type: results-report
date: 2026-07-14
experiment_line: source-e
round: 1
purpose: initial-ingestion-and-findings
status: active
---

# Source E — IRS Statistics of Income Panel Data (Capital Composition)

## 1. Executive Summary

Source E pulls the IRS Statistics of Income Division's Tax Year 2022 county-level individual income tax aggregates and computes `capital_to_wage_ratio` = (net capital gains + qualified dividends) / salaries and wages -- the "Capital Flow" pillar of `E_macro`, isolating asset-rich, investment-driven counties from pure labor-dependent ones. Coverage is 3,143 of 3,144 crosswalk counties (99.97%); the one gap is Kalawao County, HI (population ~90) -- the same county Source B's QCEW ingestion was independently missing. Three findings stand out:

1. **The ratio signal is real and validates against well-known geography, before any correlation analysis -- but the top 25 do not split cleanly into just two explanatory groups.** Famous ultra-wealthy resort/second-home markets dominate the top of the list: Teton County, WY (Jackson Hole, ratio 2.80, the national maximum), Pitkin County, CO (Aspen, 1.57), Blaine County, ID (Sun Valley, 1.24), Collier County, FL (Naples, 1.13), Monroe County, FL (Florida Keys, 1.11), San Miguel County, CO (Telluride, 1.08), Summit County, UT (Park City, 0.72), Douglas County, NV (Lake Tahoe, 0.62), Routt County, CO (Steamboat Springs, 0.61), and Walton County, FL (the "30A" Gulf Coast corridor, 0.60) -- exactly the county profile the proposal's "asset-rich, investment-driven markets" framing describes. But a large-`num_returns` outlier doesn't fit that story: **Benton County, AR (ratio 0.55, 138,170 returns -- the fourth-most-populous county in the top 25, and no resort)** is home to Walmart, Tyson Foods, and J.B. Hunt's corporate headquarters, pointing to a third, distinct mechanism -- concentrated executive/founder equity compensation -- that the proposal's framing doesn't name but the ratio correctly picks up anyway.
2. **A real, undisclosed limitation not anticipated by either scoping doc: no suppression flag exists in this IRS file, unlike BLS QCEW's `disclosure_code`.** Confirmed by the Tax Year 2022 documentation guide: cells with fewer than 20 returns are excluded from the total, but the resulting county aggregate carries no marker distinguishing a genuine zero from a suppressed one. Empirically, 3 of 3,143 counties (Kenedy, King, and Loving Counties, TX -- all 40-140 total returns) show an exact `0` for both the amount *and* the underlying return-count columns for capital gains/dividends, which is consistent with either explanation and cannot be resolved from this file alone.
3. **A second, related limitation found during this round's analysis: single large one-time transactions visibly distort the ratio in very-low-return counties.** Seven of the top-25 counties have fewer than 2,200 total returns -- well below the national median of 11,700 -- and cluster at the very top of the low-`num_returns` tail: Arthur County, NE (190 returns, ratio 0.59), McPherson County, NE (210, 0.59), Garfield County, MT (550, 0.80), Harding County, SD (590, 0.51), Jeff Davis County, TX (910, 0.59), Shackelford County, TX (1,490, 1.00), and Mason County, TX (2,100, 0.53) -- each with net capital gains several times its qualified-dividend total, the signature of a small number of large one-time sales (plausibly farm/ranch land) rather than a sustained investment-income base. This is a real, disclosed noise source distinct from both finding 1's resort counties and its Benton County outlier, and downstream consumers should weight the ratio by `num_returns` or treat very-low-return counties with caution rather than reading every high-ratio county as "investment-driven" in the proposal's intended sense.

The proposal's framing of Source E as revealing whether "a market's wealth is bound to local job survival or global Wall Street performance" (`E_macro_extendedProposal.pdf` §3.1) holds up directly for finding 1's resort counties, but the same top-25 list shows at least three distinct real-world mechanisms producing a high ratio -- resort/investment wealth, concentrated executive equity (Benton County), and one-time land-sale noise (finding 3) -- that the ratio alone cannot distinguish between.

## 2. Data & Setup

`scripts/ingest_source_e.py` downloads the IRS SOI Division's Tax Year 2022 pre-aggregated county file (`22incyallnoagi.csv` -- `AGI_STUB` fixed at 0, IRS's own county totals across all 8 AGI brackets, chosen over summing the 8-bracket `22incyallagi.csv` file or looping over 51 per-state Excel files) directly via `requests` -- `irs.gov` has no bot protection on either the landing pages or the `/pub/irs-soi/*.csv` file host, the simplest access mechanism of all six `E_macro` sources.

Target columns are referenced through a `SOI_COLUMN_MAP` conceptual-name mapping (`num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`) rather than raw SOI variable codes, per the pre-scoping spec's own proposed mitigation for upstream schema mutation -- a future year renaming or dropping one of these fields raises a loud `KeyError` rather than silently misreading a shifted column. State-total rows (`COUNTYFIPS == "000"`) are dropped before computing `fips_code` = `STATEFIPS` + `COUNTYFIPS` and joining to `county_crosswalk.parquet`.

Output: `data/source_e_irs_soi.parquet`, 3,143 rows, 8 columns (`county_name`, `fips_code`, `num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`, `capital_to_wage_ratio`).

## 3. Main Findings

### 3.1 Ratio distribution and top counties: three distinct mechanisms, not two

`capital_to_wage_ratio` has mean 0.107 and median 0.082 across all 3,143 counties -- the typical county's investment income (capital gains + qualified dividends) runs about 8% of its wage income. The distribution is heavily right-skewed (max 2.798, Teton County, WY). Sorting the top 25 by `num_returns` (median 11,700 nationally) rather than by ratio makes the mechanism split explicit: 7 counties sit far below the median at under 2,200 returns (§1 finding 3 -- the one-time-transaction noise group), most of the remainder are recognizable high-wealth resort/second-home/retirement counties well above median-to-large in size (Teton WY, Pitkin CO, Blaine ID, San Miguel CO, Routt CO, Summit UT, Douglas NV, Walton FL, Monroe FL, Martin FL, Indian River FL, Collier FL, Sarasota FL, Palm Beach FL, Miami-Dade FL), and three counties don't fit either bucket cleanly: Benton County, AR (138,170 returns, Walmart/Tyson/J.B. Hunt HQ -- §1 finding 1's executive-equity outlier), and Randolph County, MO (10,830 returns) and Nemaha County, KS (4,930 returns), neither of which has an obvious resort or corporate-HQ identity found this round -- left as an unexplained residual rather than force-fit into either group.

### 3.2 No suppression flag: a real gap versus Source B's equivalent handling

The Tax Year 2022 documentation guide (pulled and parsed directly from `22incydocguide.docx`) confirms the same class of small-cell privacy suppression Source B's BLS QCEW ingestion handles ("Income and tax items with less than 20 returns for a county were excluded"). Source B could null out suppressed cells explicitly because BLS ships an explicit `disclosure_code` flag alongside each value (`source-b-findings.md` §3.3). **The IRS county file has no equivalent flag at all** -- a suppressed cell and a true zero are both written as a bare `0`. This affects a small, identifiable set of ultra-low-population counties (Kenedy, King, and Loving Counties, TX, all under 150 total returns, all showing exact `0` for both the capital-gains amount and its underlying return count) and is shipped as-is rather than guessed at, consistent with how Source B chose an honest null over a fabricated estimate when its own suppression-handling options were tested and found wanting (`source-b-findings.md` §3.3) -- the difference here is there is no flag to null against, so the limitation is simply disclosed rather than resolved.

### 3.3 Cross-validation against Source C: a weak but real GDP-velocity link, no unemployment link

Joining `capital_to_wage_ratio` onto Source C's velocity metrics (3,080 of 3,143 counties have a `gdp_velocity_pct`; the 63 missing are the same Connecticut Planning Region / no-GDP-series gap Sources B, D, and F all independently document):

| Comparison | Pearson r | permutation p |
|---|---|---|
| `capital_to_wage_ratio` vs. unemployment velocity | -0.030 | 0.112 (not significant) |
| `capital_to_wage_ratio` vs. size-normalized GDP velocity | 0.071 | 0.002 |

Mean size-normalized GDP velocity rises monotonically across capital-to-wage quartiles: 1.6% (Q1, most labor-dependent) -> 1.8% -> 2.0% -> 2.3% (Q4, most investment-driven) -- a small but clean, directionally sensible signal, similar in shape to Source B's specialization-quartile result (`source-b-findings.md` §3.4). The unemployment-velocity result is not statistically significant, which is directionally consistent with the proposal's own framing: an investment-heavy county's wealth is tied to "global Wall Street performance" rather than local job survival (`E_macro_extendedProposal.pdf` §3.1), so a weaker link to local unemployment momentum than to broader GDP growth is exactly what that framing would predict, though the effect sizes here are small enough that this should be read as directionally consistent rather than strong confirmation.

## 4. Figure-by-Figure Interpretation

- `analysis-output/source-e/figures/source-e-figure-01-top-investment-driven.png`: horizontal bar chart of the top 15 counties by `capital_to_wage_ratio`. Visually confirms §3.1's resort-county cluster (Teton, Pitkin, Blaine, Collier, Monroe at the top).
- `analysis-output/source-e/figures/source-e-figure-02-ratio-distribution.png`: histogram of `capital_to_wage_ratio` across all counties. Right-skewed, median 0.082, long tail out past 2.5, visualizing the resort/low-N-outlier extremes from §3.1.
- `analysis-output/source-e/figures/source-e-figure-03-composition-vs-velocity.png`: bar chart of mean size-normalized GDP velocity by capital-composition quartile. Visualizes §3.3's monotonic 1.6%-2.3% quartile progression.
- `outputs/source_e_map_capital_composition.html`: interactive US map, one bubble per county, colored by `capital_to_wage_ratio`, top-5 counties labeled directly.
- `outputs/source_e_capital_composition.html`, `outputs/source_e_source_c_correlation.html`: interactive versions of figures 2 and 3.

## 5. Limitations / Open Items

- **No suppression flag (§3.2).** Unlike Source B, a handful of ultra-low-population counties' zero values cannot be distinguished from suppressed values. Affects only 3 identified counties currently, all under 150 total returns, but is a structural gap in the source data, not a pipeline bug.
- **Low-return-count noise distorts the ratio at the tail (§1 finding 3, §3.1).** Counties under ~1,000 total returns are disproportionately represented in the top-25 ratio ranking, likely reflecting one-time large transactions (e.g., land sales) rather than sustained capital-market exposure. Any downstream model consuming `capital_to_wage_ratio` directly should consider weighting by `num_returns` or flagging low-`num_returns` counties rather than treating the raw ratio as equally reliable everywhere.
- **Single reference year (Tax Year 2022, the latest published as of this writing).** No multi-year trend analysis -- whether a county's capital composition is stable or itself has momentum (analogous to Source C's velocity framing) is untested this round.
- **Thousands-of-dollars units retained as reported**, not converted to whole dollars -- matches the source file's own convention, documented in `ingest_source_e.py`'s `SOI_COLUMN_MAP` column names (`*_thousands`).
- **1 county (Kalawao, HI) has no usable Source E signal at all**, the same gap Source B independently has -- an IRS-SOI-based feature will need an explicit missing-data policy for this county in any downstream model.

## 6. Next Actions

1. Cross-validate Source E's capital-composition quartile against Source B's dominant-sector labels directly (e.g., do Finance & Insurance- or Real Estate-dominant counties from Source B skew toward higher `capital_to_wage_ratio`?) -- a natural agreement check between two independently-sourced wealth/industry signals, not attempted this round.
2. Consider a `num_returns`-weighted or -filtered variant of `capital_to_wage_ratio` to directly address the low-N noise limitation (§5) rather than leaving it as a caveat for downstream consumers.
3. No action planned this round on the undisclosed-suppression gap (§5) -- it is a structural property of the source data affecting a small, identified set of counties; revisit only if a downstream consumer specifically needs it resolved.

## 7. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_e.py` -> `data/source_e_irs_soi.parquet` (`uv run scripts/ingest_source_e.py`, no credentials required)
- Capital-composition characterization: `scripts/analyze_source_e_capital_composition.py` -> `outputs/source_e_capital_composition.csv`, `outputs/source_e_capital_composition.html`
- Cross-validation vs. Source C: `scripts/analyze_source_e_source_c_correlation.py` -> `outputs/source_e_source_c_correlation.csv`, `outputs/source_e_source_c_correlation.html`
- Map: `scripts/visualize_source_e.py` -> `outputs/source_e_map_capital_composition.html`
- Stats/figures: `scripts/generate_source_e_insights.py` -> `analysis-output/source-e/source_e_stats.json`, `analysis-output/source-e/figures/source-e-figure-*.png`, `analysis-output/source-e/figures/source-e-numeric-summary.md`
- Presentation notebook: `analysis-output/source-e/source_e_key_findings.ipynb`
- Full phased research/design history (access mechanism, file-format choice, column-mapping mitigation): `source_e_plan.md`

## 8. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf` / `docs/macro_pre_scoping_spec.pdf`, Source E section)

The proposal frames Source E as computing "the exact ratio of Form 1040 Schedule D (Capital Gains) + Qualified Dividends relative to Standard W-2 Wage Income" to isolate asset-rich markets from labor-dependent ones.

- **Supported**: the mechanism works exactly as described -- IRS SOI ships pre-aggregated county-level `A01000` (net capital gain, 1040 line 7, sourced from Schedule D when required) and `A00650` (qualified dividends, 1040 line 3a) alongside `A00200` (wages, 1040 line 1z), and the resulting ratio cross-checks cleanly against real-world geography (Jackson Hole, Aspen, Sun Valley, Naples, the Florida Keys -- §3.1).
- **Deviated, minor**: the pre-scoping spec describes the access protocol as "Bulk CSV Zip Archives (irs.gov/statistics)"; the real mechanism is a direct, unzipped CSV download (`22incyallnoagi.csv`) with no zip archive step at all -- simpler than the spec assumed, not a risk.
- **New finding not anticipated by either scoping doc**: unlike the spec's own flagged risk for Source B (BLS suppression), Source E's suppression mechanism is real but carries no disclosure flag at all (§3.2) -- a strictly less transparent version of a risk the spec correctly anticipated existing for a *different* source but didn't flag here. A second, related new finding (§1 finding 3, §3.1) is that low-return-count counties introduce a distinct, non-suppression noise source: single large one-time transactions inflating the ratio in a way that reads identically to genuine investment-market exposure without a `num_returns` cross-check.
- **Reporting lag**: the spec estimates an 18-24 month reporting lag for this source; Tax Year 2022 data (covering returns filed through December 2023) being the latest available as of mid-2026 reflects a longer real-world lag than that estimate, consistent with the general pattern (also seen in Source D) that this project's scoping docs' latency estimates run optimistic relative to what's actually published.
