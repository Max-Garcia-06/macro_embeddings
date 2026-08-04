---
type: results-report
date: 2026-08-04
experiment_line: source-e
round: 2
purpose: initial-ingestion-and-findings, then multi-year and data-volume revision
status: active
---

> **Round 2 (2026-08-04).** §1–§8 are the round-1 report as written. §9–§15 add a
> multi-year panel, decompose the ratio, stratify every claim by county data
> volume, and re-score the leave-one-pillar-out sweep. Two round-1 conclusions do
> not survive that: finding 3's noise mechanism is narrower than described, and
> §5's mitigation advice ("weight by `num_returns`") is backwards. Both are
> corrected in place below and explained in §11.

# Source E — IRS Statistics of Income Panel Data (Capital Composition)

## 1. Executive Summary

Source E pulls the IRS Statistics of Income Division's Tax Year 2022 county-level individual income tax aggregates and computes `capital_to_wage_ratio` = (net capital gains + qualified dividends) / salaries and wages -- the "Capital Flow" pillar of `E_macro`, isolating asset-rich, investment-driven counties from pure labor-dependent ones. Coverage is 3,143 of 3,144 crosswalk counties (99.97%); the one gap is Kalawao County, HI (population ~90) -- the same county Source B's QCEW ingestion was independently missing. Three findings stand out:

1. **The ratio signal is real and validates against well-known geography, before any correlation analysis -- but the top 25 do not split cleanly into just two explanatory groups.** Famous ultra-wealthy resort/second-home markets dominate the top of the list: Teton County, WY (Jackson Hole, ratio 2.80, the national maximum), Pitkin County, CO (Aspen, 1.57), Blaine County, ID (Sun Valley, 1.24), Collier County, FL (Naples, 1.13), Monroe County, FL (Florida Keys, 1.11), San Miguel County, CO (Telluride, 1.08), Summit County, UT (Park City, 0.72), Douglas County, NV (Lake Tahoe, 0.62), Routt County, CO (Steamboat Springs, 0.61), and Walton County, FL (the "30A" Gulf Coast corridor, 0.60) -- exactly the county profile the proposal's "asset-rich, investment-driven markets" framing describes. But a large-`num_returns` outlier doesn't fit that story: **Benton County, AR (ratio 0.55, 138,170 returns -- the fourth-most-populous county in the top 25, and no resort)** is home to Walmart, Tyson Foods, and J.B. Hunt's corporate headquarters, pointing to a third, distinct mechanism -- concentrated executive/founder equity compensation -- that the proposal's framing doesn't name but the ratio correctly picks up anyway.
2. **A real, undisclosed limitation not anticipated by either scoping doc: no suppression flag exists in this IRS file, unlike BLS QCEW's `disclosure_code`.** Confirmed by the Tax Year 2022 documentation guide: cells with fewer than 20 returns are excluded from the total, but the resulting county aggregate carries no marker distinguishing a genuine zero from a suppressed one. Empirically, 3 of 3,143 counties (Kenedy, King, and Loving Counties, TX -- all 40-140 total returns) show an exact `0` for both the amount *and* the underlying return-count columns for capital gains/dividends, which is consistent with either explanation and cannot be resolved from this file alone.
3. **A second, related limitation found during this round's analysis: single large one-time transactions visibly distort the ratio in very-low-return counties.** Seven of the top-25 counties have fewer than 2,200 total returns -- well below the national median of 11,700 -- and cluster at the very top of the low-`num_returns` tail: Arthur County, NE (190 returns, ratio 0.59), McPherson County, NE (210, 0.59), Garfield County, MT (550, 0.80), Harding County, SD (590, 0.51), Jeff Davis County, TX (910, 0.59), Shackelford County, TX (1,490, 1.00), and Mason County, TX (2,100, 0.53) -- each with net capital gains several times its qualified-dividend total, the signature of a small number of large one-time sales (plausibly farm/ranch land) rather than a sustained investment-income base. This is a real, disclosed noise source distinct from both finding 1's resort counties and its Benton County outlier, and downstream consumers should weight the ratio by `num_returns` or treat very-low-return counties with caution rather than reading every high-ratio county as "investment-driven" in the proposal's intended sense.

> **Corrected in round 2 (§11).** The mechanism is real but is not a *low-return* mechanism — it is a concentrated-gain one, and the two are close to independent. Only 10 of the 325 low-return counties show concentrated gains, and the `num_returns` weighting recommended above is actively wrong: low-return counties are the *more* temporally stable group (median year-over-year move 0.298, against 0.393 for counties above 100k returns), so weighting by `num_returns` upweights the least stable counties in the file. `concentrated_gain_flag` and `thin_claimer_flag` now carry this instead.

The proposal's framing of Source E as revealing whether "a market's wealth is bound to local job survival or global Wall Street performance" (`E_macro_extendedProposal.pdf` §3.1) holds up directly for finding 1's resort counties, but the same top-25 list shows at least three distinct real-world mechanisms producing a high ratio -- resort/investment wealth, concentrated executive equity (Benton County), and one-time land-sale noise (finding 3) -- that the ratio alone cannot distinguish between.

## 2. Data & Setup

`scripts/ingest_source_e.py` downloads the IRS SOI Division's Tax Year 2022 pre-aggregated county file (`22incyallnoagi.csv` -- `AGI_STUB` fixed at 0, IRS's own county totals across all 8 AGI brackets, chosen over summing the 8-bracket `22incyallagi.csv` file or looping over 51 per-state Excel files) directly via `requests` -- `irs.gov` has no bot protection on either the landing pages or the `/pub/irs-soi/*.csv` file host, the simplest access mechanism of all six `E_macro` sources.

Target columns are referenced through a `SOI_COLUMN_MAP` conceptual-name mapping (`num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`) rather than raw SOI variable codes, per the pre-scoping spec's own proposed mitigation for upstream schema mutation -- a future year renaming or dropping one of these fields raises a loud `KeyError` rather than silently misreading a shifted column. State-total rows (`COUNTYFIPS == "000"`) are dropped before computing `fips_code` = `STATEFIPS` + `COUNTYFIPS` and joining to `county_crosswalk.parquet`.

Output: `data/source_e_irs_soi.parquet`, 3,143 rows, 8 columns (`county_name`, `fips_code`, `num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`, `capital_to_wage_ratio`).

> **Round 2:** the shipped file is now 3,143 × 24 and the pipeline pulls TY2018–TY2022, not TY2022 alone. Current schema in `docs/source_e_feature_schema.md`; what changed and why in §9–§13.

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
The `.html` renders below are ~5MB each, regenerable, and no longer committed --
rebuild any of them with the script named against it in §7.

- `outputs/source_e_map_capital_composition.html`: interactive US map, one bubble per county, colored by `capital_to_wage_ratio`, top-5 counties labeled directly.
- `outputs/source_e_capital_composition.html`, `outputs/source_e_source_c_correlation.html`: interactive versions of figures 2 and 3.

## 5. Limitations / Open Items

- **No suppression flag (§3.2).** Unlike Source B, a handful of ultra-low-population counties' zero values cannot be distinguished from suppressed values. Affects only 3 identified counties currently, all under 150 total returns, but is a structural gap in the source data, not a pipeline bug. **Partly mitigated in §12**: the IRS file's `N00200`/`N00650`/`N01000` return-count columns now ship, so a consumer can see how thin the sample behind each amount is instead of inferring it from `num_returns`. The flag itself still does not exist upstream.
- ~~**Low-return-count noise distorts the ratio at the tail (§1 finding 3, §3.1).** Counties under ~1,000 total returns are disproportionately represented in the top-25 ratio ranking, likely reflecting one-time large transactions (e.g., land sales) rather than sustained capital-market exposure. Any downstream model consuming `capital_to_wage_ratio` directly should consider weighting by `num_returns` or flagging low-`num_returns` counties rather than treating the raw ratio as equally reliable everywhere.~~ **Superseded by §11.** The `num_returns` weighting is backwards; the mechanism is concentrated gains, not low return counts, and it is now carried by `concentrated_gain_flag` and `thin_claimer_flag`.
- ~~**Single reference year (Tax Year 2022, the latest published as of this writing).** No multi-year trend analysis -- whether a county's capital composition is stable or itself has momentum (analogous to Source C's velocity framing) is untested this round.~~ **Resolved in §10.** TY2018–TY2022 now ship as `data/source_e_irs_soi_panel.parquet`, and the vintage effect turned out large enough to be the pillar's biggest single weakness rather than a footnote.
- **Thousands-of-dollars units retained as reported**, not converted to whole dollars -- matches the source file's own convention, documented in `ingest_source_e.py`'s `SOI_COLUMN_MAP` column names (`*_thousands`).
- **1 county (Kalawao, HI) has no usable Source E signal at all**, the same gap Source B independently has -- an IRS-SOI-based feature will need an explicit missing-data policy for this county in any downstream model.

## 6. Next Actions

1. Cross-validate Source E's capital-composition quartile against Source B's dominant-sector labels directly (e.g., do Finance & Insurance- or Real Estate-dominant counties from Source B skew toward higher `capital_to_wage_ratio`?) -- a natural agreement check between two independently-sourced wealth/industry signals, not attempted this round.
2. ~~Consider a `num_returns`-weighted or -filtered variant of `capital_to_wage_ratio` to directly address the low-N noise limitation (§5) rather than leaving it as a caveat for downstream consumers.~~ **Done, differently (§11–§12).** The weighting was tested and rejected; the ratio's three drivers ship as separate columns instead, with `thin_claimer_flag` and `concentrated_gain_flag` marking where the level cannot be trusted.
3. ~~No action planned this round on the undisclosed-suppression gap (§5)~~ **Partly closed (§12)** -- the return-count columns now ship. The upstream flag still does not exist; revisit only if a downstream consumer needs more.

## 7. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_e.py` -> `data/source_e_irs_soi.parquet` (`uv run scripts/ingest_source_e.py`, no credentials required)
- Capital-composition characterization: `scripts/analyze_source_e_capital_composition.py` -> `outputs/source_e_capital_composition.csv`, `outputs/source_e_capital_composition.html`
- Cross-validation vs. Source C: `scripts/analyze_source_e_source_c_correlation.py` -> `outputs/source_e_source_c_correlation.csv`, `outputs/source_e_source_c_correlation.html`
- Map: `scripts/visualize_source_e.py` -> `outputs/source_e_map_capital_composition.html`
- Stats/figures: `scripts/generate_source_e_insights.py` -> `analysis-output/source-e/source_e_stats.json`, `analysis-output/source-e/figures/source-e-figure-*.png`, `analysis-output/source-e/figures/source-e-numeric-summary.md`
- Presentation notebook: `analysis-output/source-e/source_e_key_findings.ipynb`
- Reconnaissance/decision trail (access mechanism, file-format choice, column-mapping mitigation): `docs/plans/ingestion_recon.md` § Source E. The original `source_e_plan.md`, including the phase-by-phase run logs, is in git history.

## 8. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf` / `docs/macro_pre_scoping_spec.pdf`, Source E section)

The proposal frames Source E as computing "the exact ratio of Form 1040 Schedule D (Capital Gains) + Qualified Dividends relative to Standard W-2 Wage Income" to isolate asset-rich markets from labor-dependent ones.

- **Supported**: the mechanism works exactly as described -- IRS SOI ships pre-aggregated county-level `A01000` (net capital gain, 1040 line 7, sourced from Schedule D when required) and `A00650` (qualified dividends, 1040 line 3a) alongside `A00200` (wages, 1040 line 1z), and the resulting ratio cross-checks cleanly against real-world geography (Jackson Hole, Aspen, Sun Valley, Naples, the Florida Keys -- §3.1).
- **Deviated, minor**: the pre-scoping spec describes the access protocol as "Bulk CSV Zip Archives (irs.gov/statistics)"; the real mechanism is a direct, unzipped CSV download (`22incyallnoagi.csv`) with no zip archive step at all -- simpler than the spec assumed, not a risk.
- **New finding not anticipated by either scoping doc**: unlike the spec's own flagged risk for Source B (BLS suppression), Source E's suppression mechanism is real but carries no disclosure flag at all (§3.2) -- a strictly less transparent version of a risk the spec correctly anticipated existing for a *different* source but didn't flag here. A second, related new finding (§1 finding 3, §3.1) is that low-return-count counties introduce a distinct, non-suppression noise source: single large one-time transactions inflating the ratio in a way that reads identically to genuine investment-market exposure without a `num_returns` cross-check.
- **Reporting lag**: the spec estimates an 18-24 month reporting lag for this source; Tax Year 2022 data (covering returns filed through December 2023) being the latest available as of mid-2026 reflects a longer real-world lag than that estimate, consistent with the general pattern (also seen in Source D) that this project's scoping docs' latency estimates run optimistic relative to what's actually published.

---

# Round 2 — Multi-year panel and data-volume stratification (2026-08-04)

## 9. The ratio is a product of three separable things

`capital_to_wage_ratio` collapses three independent quantities into one number.
Regressing its log on all three, across the 3,140 counties where each is
defined:

```
log(ratio) ~ log(capgain_participation_rate)
           + log(gain_per_claimer_thousands)
           + log(wage_per_return_thousands)          R² = 0.975
betas:       +0.963                +0.895              −0.859
```

Three near-unit elasticities and almost no residual. A county reaches a high
ratio because a large share of its filers report investment income, because the
few who do report unusually large amounts, or because there is barely any wage
income underneath — and the round-1 feature maps all three onto the same value.

Two consequences the round-1 report did not surface:

- **The numerator does all the work.** `var(log cap per return) = 0.548` against
  `var(log wage per return) = 0.053`. The "vs. wage" half of the framing
  contributes about a tenth of the variance the "capital" half does.
- **The denominator confound is real but small.** `corr(ratio, share of returns
  reporting wages) = −0.346` (Spearman −0.403). Retirement counties score high
  partly because wages are absent, not because capital is present. The ratio
  alone cannot separate "capital-rich" from "job-thin"; `wage_per_return_thousands`
  now ships so a model can.

The pillar also sits closer to a wealth-level measure than round 1 implied:
`corr(capital_to_wage_ratio, AGI per return) = 0.629` (Spearman 0.530). That
matters because `pillar_matrix` already draws `log_agi` from Source E's own
`agi_thousands` as the scale control, so the feature and the control share a
source.

**Shipped:** `capgain_participation_rate`, `dividend_participation_rate`,
`gain_per_claimer_thousands`, `wage_per_return_thousands`.

## 10. The market year sets the level, not the county

Round 1 listed the single reference year as a minor limitation. Across
TY2018–TY2022 it is the pillar's largest single weakness:

| tax year | national aggregate ratio | unweighted county mean | county median |
|---|---|---|---|
| 2018 | 0.1439 | — | — |
| 2019 | 0.1312 | — | — |
| 2020 | 0.1596 | 0.0946 | 0.0708 |
| 2021 | 0.2551 | 0.1562 | 0.1194 |
| 2022 | 0.1557 | 0.1075 | 0.0822 |

TY2021 runs 64% above TY2022 nationally. That is the equity market, not 3,143
county economies moving together. A downstream model trained on the round-1
parquet and refreshed to a later tax year sees every county's value shift at
once, with no column explaining why. Rank churn confirms the level is doing the
moving: 19 of the top-50 counties by ratio changed between TY2021 and TY2022.

**Shipped:** `capital_to_wage_ratio_normalized` (county ÷ national aggregate for
the same year, so 1.0 = national-average composition),
`capital_to_wage_ratio_normalized_mean` and `_std` across TY2018–TY2022,
`n_tax_years_observed`, and the full long-format panel at
`data/source_e_irs_soi_panel.parquet`. The normalized five-year mean is the
column a consumer should prefer.

One ingestion bug surfaced while building the panel and is worth recording,
because it is exactly the failure mode `SOI_COLUMN_MAP` was built to prevent in a
different dimension: **the TY2019 file ships both FIPS columns unpadded**
(`1`/`1` where every other year writes `01`/`001`). The round-1 transform
filtered state-total rows on `COUNTYFIPS != "000"`, which does not match `"0"`,
so state totals survived and the concatenated key landed on the wrong counties —
1,150 of 3,193 rows joined, silently. Padding now happens before the filter, and
a duplicate-key check raises rather than warns.

## 11. Data volume: four tiers, and the correction to round 1

`scripts/analyze_source_e_tiers.py` splits counties on `num_returns` at the
`low_return_flag` threshold, the national median, and 100k returns. Stats in
`analysis-output/source-e/source_e_tier_stats.json`.

| tier | n | share of national investment income | median ratio | IQR | median \|Δ\|/ratio, TY21→22 | rank stability (Spearman) | B Real Estate LQ × E |
|---|---|---|---|---|---|---|---|
| T1 thin (<2.2k returns) | 325 | **0.14%** | 0.116 | 0.117 | 0.298 | 0.861 | **−0.058** (n=49) |
| T2 small (2.2k–11.7k) | 1,246 | 2.03% | 0.073 | 0.056 | 0.280 | 0.869 | +0.380 (n=900) |
| T3 mid (11.7k–100k) | 1,246 | 15.23% | 0.080 | 0.053 | 0.333 | 0.882 | +0.410 (n=1,181) |
| T4 large (≥100k) | 326 | **82.60%** | 0.118 | 0.082 | **0.393** | 0.941 | **+0.476** (n=323) |

Four results, two of which correct round 1:

1. **Dispersion is U-shaped, not decreasing.** T1's IQR is 2.2× T3's — that part
   of round-1 finding 3 holds. But T4's IQR is also elevated, and the two tails
   are elevated for different reasons: T1 because a few filers can move a small
   county, T4 because large wealthy counties genuinely hold more market-linked
   capital.
2. **It is not sampling noise.** Regressing log dispersion on log median returns
   across deciles gives a slope of **+0.026** (r² = 0.014). Pure sampling error
   would give −0.5. Standard deviation is flat at ≈0.10 in every decile.
3. **Small counties are the more *stable* group over time — round 1 had this
   backwards.** Median year-over-year relative move rises monotonically with
   size, 0.298 (T1) → 0.393 (T4), and the share of counties moving more than 50%
   in one year is 17% in T1 against 9% in T4 only because T1's *levels* are
   noisier; on ranks, stability improves with size (0.861 → 0.941). Round 1's
   recommended mitigation — weight by `num_returns` — therefore upweights the
   counties whose values move most between vintages.
4. **The strongest surviving cross-pillar link is a large-county phenomenon.**
   B Real Estate LQ × E runs +0.394 nationally, but +0.476 in T4 and **−0.058 in
   T1**. `PROJECT_GOAL.md`'s "strongest surviving link" does not exist for the
   10% of counties with the least data. Any consumer serving rural geographies
   needs to know that before leaning on B ↔ E.

The economic asymmetry is the cleanest way to state the whole problem: **T1 and
T4 are each about 10% of counties. T1 holds 0.14% of national investment income;
T4 holds 82.6%.** An unweighted county-level feature and the economy it purports
to describe are different objects — the national aggregate ratio is 0.156 against
an unweighted county mean of 0.107.

## 12. Flags rebuilt

Round 1's `low_return_flag` (`num_returns < 2,200`) was introduced to catch
one-time-land-sale distortion. It does not: only 2 of the top-15 counties by
ratio are flagged, 27 of the top 100, and dropping every county below the
national median leaves the maximum unchanged (2.798, Teton County WY, 15,210
returns) and the standard deviation slightly *higher*. Cutting the low-N tail
does not cut the outliers, because the outliers were never mostly low-N.

The IRS file already ships what does separate the mechanisms — the counts of
returns behind each amount — and round 1 discarded them:

| county | returns | ratio | claimers | $k per claimer | participation |
|---|---|---|---|---|---|
| Shackelford County, TX | 1,490 | 1.00 | 300 | 233.7 | 0.20 |
| Arthur County, NE | 190 | 0.59 | 90 | 31.4 | 0.47 |
| national median | 11,700 | 0.082 | — | 18.6 | 0.16 |

Both are round-1 finding 3 counties and both carry `low_return_flag`, but
Shackelford is one very large realization against normal participation while
Arthur is nearly half the county reporting ordinary gains. Different mechanisms,
one flag.

**Shipped:** `n_returns_wages`, `n_returns_qualified_dividends`,
`n_returns_net_cap_gain`, plus two flags that name their mechanism —
`thin_claimer_flag` (fewer than 100 filers behind the numerator; 37 counties,
and where undisclosed suppression can hide) and `concentrated_gain_flag`
(gain per claimer above the national p95 of $45.8k; 157 counties). Only 10
counties carry both `low_return_flag` and `concentrated_gain_flag` — the two are
close to independent, which is why one could never stand in for the other.
`low_return_flag` stays, redocumented as a **materiality** flag: it marks the
counties that barely move a national aggregate, not the counties whose values are
untrustworthy.

## 13. What this changes for the handoff

Schema and null semantics are frozen in `docs/source_e_feature_schema.md`,
matching the treatment Source A got. The operative guidance for a consuming
model:

- Prefer `capital_to_wage_ratio_normalized_mean` over `capital_to_wage_ratio`.
  The raw column is a market-year level; the normalized five-year mean is not.
- Treat the ratio's three components as available features, not internals.
- Read `thin_claimer_flag` for "can I trust this level", `low_return_flag` for
  "does this county move my aggregate", `concentrated_gain_flag` for "is this a
  land sale".
- Do not weight by `num_returns` to suppress noise. It does the opposite.

Open, unchanged: Kalawao County HI still has no Source E row, TY2023 is not yet
published (HTTP 404 as of 2026-08-04), and the source still ships no suppression
flag.

## 14. Round 2 Artifact Index

- Ingestion (now multi-year): `scripts/ingest_source_e.py` -> `data/source_e_irs_soi.parquet` (3,143 × 24) and `data/source_e_irs_soi_panel.parquet` (15,686 rows, TY2018–TY2022)
- Data-volume tiers: `scripts/analyze_source_e_tiers.py` -> `outputs/source_e_tiers.csv`, `analysis-output/source-e/source_e_tier_stats.json`
- Frozen schema, including the per-column size tiering: `docs/source_e_feature_schema.md`
- Re-scored sweep (§15): `scripts/analyze_pillar_matrix_signal.py` -> `outputs/pillar_matrix_signal.csv`, `analysis-output/cross-source/pillar_matrix_signal_stats.json`

## 15. Re-scored leave-one-pillar-out sweep

Two changes to Source E's feature block landed in this round, and both move
`analyze_pillar_matrix_signal.py` for every pillar except E itself — E's own
target is predicted by the *other* five pillars, so its row is byte-identical
across all three runs (+0.0729 lift, +0.0694 ablated).

The sweep was run at each state to keep the two changes separable:

| | A: pre-round 2 (E = 4 cols) | B: + the new E columns (E = 12) | C: dollar totals moved out (E = 10) |
|---|---|---|---|
| targets carrying signal | 21 of 29 | 24 of 29 | **24 of 29** |
| mean lift | +0.0720 | +0.0818 | **+0.0808** |
| mean ablated lift | +0.0228 | +0.0333 | **+0.0329** |
| mean GBM lift | +0.1042 | +0.1139 | **+0.1159** |
| definitional share of mean lift | 0.683 | 0.593 | **0.592** |

**The components earned their place (A → B).** Three more targets clear the
carrying-signal bar, all Source B sector LQs, and the definitional share of the
mean lift falls from 0.683 to 0.592 — a larger fraction of what the matrix knows
is now corroboration rather than two federal products restating one fact.

The targets that moved most are the ones a capital-composition signal should
move, which is the part that argues this is real rather than 8 extra columns
buying out-of-fold luck:

| target | ablated lift, A | ablated lift, C |
|---|---|---|
| Professional, Scientific & Technical Services LQ | +0.0059 | **+0.0686** |
| Arts, Entertainment & Recreation LQ | −0.0287 | **+0.0161** |
| Information LQ | +0.0576 | **+0.0920** |
| Management of Companies & Enterprises LQ | −0.0107 | **+0.0157** |
| Finance & Insurance LQ | +0.0445 | **+0.0582** |
| unemployment rate (level) | +0.0232 | **+0.0492** |

**Removing the two size proxies cost almost nothing (B → C).**
`qualified_dividends_thousands` and `net_cap_gain_thousands` are dollar totals
running r = 0.894 and 0.875 against log population in logs, and r = 0.928 / 0.912
against `log_agi` — which `pillar_matrix` derives from Source E's own
`agi_thousands`. Leaving them in E's block put two near-copies of the size
control inside a block scored against that control. Source A's `n_body_sections`
was cut from its scored block at r = 0.550, so the rule already existed; it had
been applied to `wages_salaries_thousands` and not to its two siblings.

Moving them costs −0.0011 mean lift and no targets. That is the result worth
recording: **the round-2 gain was not size.** Had the gain evaporated here, the
new columns would have been a size proxy in a better costume.

One per-pillar move is worth naming. Source A's mean ablated lift fell +0.0560 →
+0.0510, the largest in the sweep. Source A's only target is `content_length`,
itself classified size-in-disguise (`source-a-findings.md` §13.1), so removing a
size proxy from the predictor pool costs most on the target most made of size.
That is the exclusion working, not a regression.

**Unchanged:** the 15-pillar-pair sweep (`pillar_pair_stats.json`) uses
`capital_to_wage_ratio` as E's feature and is untouched, so the B ↔ E headline
(r = 0.394 raw, 0.382 size-controlled) stands exactly as published.
