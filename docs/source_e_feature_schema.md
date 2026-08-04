# Source E — Frozen Feature Schema

Keyed on `fips_code`, 3,143 rows, one per US county and county-equivalent
covered by the IRS SOI county file. Written by `scripts/ingest_source_e.py`
to `data/source_e_irs_soi.parquet`, with the long-format companion
`data/source_e_irs_soi_panel.parquet` (one row per county × tax year,
TY2018–TY2022).

## What ships and why

The pillar's original single feature, `capital_to_wage_ratio`, is a product of
three separable quantities. Regressing its log on them recovers **R² = 0.975**
with near-unit elasticities (`source-e-findings.md` §9):

```
log(ratio) ≈ 0.96·log(capgain_participation_rate)
           + 0.90·log(gain_per_claimer_thousands)
           − 0.86·log(wage_per_return_thousands)
```

A county reaches a high ratio because many filers report gains, because a few
report very large ones, or because it has almost no wage base underneath — three
different economies that the single ratio maps onto the same number. All three
components ship so a consuming model can tell them apart. The ratio itself stays
for continuity with the crossvalidation sweep.

## Null semantics

**Zero means zero everywhere except `gain_per_claimer_thousands`.** The IRS file
publishes no suppression flag (unlike BLS QCEW's `disclosure_code`), so a
suppressed small cell and a genuine zero are written identically. Its practical
footprint is 3 counties — Kenedy, King, and Loving Counties, TX, all under 150
returns — where the capital-gain amount *and* its claiming-return count are both
zero. Those three are the only nulls in the file: there is no average gain over
an empty set, so `gain_per_claimer_thousands` is null rather than 0, which would
read as "gains were small."

The `n_returns_*` columns are the intended replacement for the missing
suppression flag. They state how many returns underlie each amount, so a
consumer can see a thin sample directly instead of inferring it from
`num_returns`.

**1 county has no Source E row at all**: Kalawao County, HI (`15005`, population
~90) — the same gap Source B has independently. Downstream models need an
explicit policy for it.

`n_tax_years_observed` is 5 for 3,132 counties. The 11 exceptions are boundary
changes, not data quality: the 9 Connecticut Planning Regions replaced the old
counties in TY2021 (2 years each), and Chugach / Copper River Census Areas, AK
split from Valdez-Cordova (4 years each).

## Flag policy

| flag | fires | count | what it means |
|---|---|---|---|
| `low_return_flag` | `num_returns < 2,200` | 325 (10.3%) | **Materiality, not noise.** These counties are 10.3% of rows and **0.14%** of national investment income. Their ratios are *not* less stable year over year than large counties' — see below. Use this to decide whether a county matters to an aggregate, not whether its value is trustworthy. |
| `thin_claimer_flag` | `n_returns_net_cap_gain < 100` | 37 (1.2%) | The numerator rests on fewer than 100 filers. This is the honest "do not trust the level" flag, and it is where undisclosed suppression can hide. |
| `concentrated_gain_flag` | `gain_per_claimer_thousands` above the national p95 (45.8k) | 157 (5.0%) | A few unusually large realizations, typically farm/ranch land sales, rather than a broad investment base. This is the mechanism `low_return_flag` was originally introduced to catch but does not isolate — only 10 counties carry both. |

475 counties (15.1%) carry at least one flag. Flags are diagnostics: they stay in
the parquet and out of any distance or similarity computation.

## Vintage and the market-cycle problem

`capital_to_wage_ratio` is a *level* set as much by the market year as by the
county. The unweighted county mean runs **0.095 (TY2020) → 0.156 (TY2021) →
0.108 (TY2022)**; a refresh to a new tax year moves every county at once.

`capital_to_wage_ratio_normalized` divides each county by the national aggregate
ratio for its own year, so 1.0 means "national-average composition" in any year.
`capital_to_wage_ratio_normalized_mean` averages that across TY2018–TY2022 and is
**the column a downstream model should prefer** — it is the one that survives a
vintage refresh. `capital_to_wage_ratio_normalized_std` states how much the
county moved across those five years.

`as_of_date` is `2022-12-31`, the end of the latest tax year in the file. Every
cross-year column is built only from years at or before that date, so the stamp
remains a valid upper bound for leakage checks.

## Columns

| column | dtype | nulls | description |
|---|---|---|---|
| `county_name` | str | 0 | e.g. `"Teton County, Wyoming"`, from `county_crosswalk.parquet`. |
| `fips_code` | str | 0 | 5-digit state+county FIPS. Join key. |
| `num_returns` | float64 | 0 | Total returns filed (SOI `N1`). Ranges 40 → 4,691,250. |
| `agi_thousands` | float64 | 0 | Adjusted gross income (`A00100`). Used by `pillar_matrix` as the `log_agi` scale control, held outside every predictor block. |
| `wages_salaries_thousands` | float64 | 0 | Salaries and wages (`A00200`). Ratio denominator. |
| `n_returns_wages` | float64 | 0 | Returns reporting wages (`N00200`). |
| `qualified_dividends_thousands` | float64 | 0 | Qualified dividends (`A00650`). |
| `n_returns_qualified_dividends` | float64 | 0 | Returns reporting qualified dividends (`N00650`). |
| `net_cap_gain_thousands` | float64 | 0 | Net capital gain (`A01000`). |
| `n_returns_net_cap_gain` | float64 | 0 | Returns reporting a net capital gain (`N01000`). |
| `capital_to_wage_ratio` | float64 | 0 | (net capital gain + qualified dividends) / wages, TY2022. Median 0.082, max 2.798. Market-year dependent — prefer the normalized columns. |
| `capital_to_wage_ratio_normalized` | float64 | 0 | `capital_to_wage_ratio` ÷ the national aggregate ratio for TY2022 (0.1557). 1.0 = national-average composition. |
| `national_capital_to_wage_ratio` | float64 | 0 | The TY2022 national aggregate, 0.1557, constant across rows. Carried so the normalization is invertible without refetching. |
| `capgain_participation_rate` | float64 | 0 | `n_returns_net_cap_gain / num_returns`. Median 0.160. |
| `dividend_participation_rate` | float64 | 0 | `n_returns_qualified_dividends / num_returns`. |
| `gain_per_claimer_thousands` | float64 | 3 | `net_cap_gain_thousands / n_returns_net_cap_gain`. Median 18.6. Null where no filer reported a gain. |
| `wage_per_return_thousands` | float64 | 0 | `wages_salaries_thousands / num_returns`. The ratio's denominator, per filer. |
| `low_return_flag` | bool | 0 | See flag policy. 325 fire. |
| `thin_claimer_flag` | bool | 0 | See flag policy. 37 fire. |
| `concentrated_gain_flag` | bool | 0 | See flag policy. 157 fire. |
| `capital_to_wage_ratio_normalized_mean` | float64 | 0 | Mean normalized ratio across TY2018–TY2022. **Preferred headline feature.** |
| `capital_to_wage_ratio_normalized_std` | float64 | 0 | Standard deviation of the same, across years. |
| `n_tax_years_observed` | int64 | 0 | Years contributing to the two columns above. 5 for 3,132 counties. |
| `as_of_date` | str | 0 | `2022-12-31`. See `outputs/pillar_vintages.csv`. |

## Serving policy by data volume

Source E does not behave the same way across county sizes, and a single national
statistic hides it. `scripts/analyze_source_e_tiers.py` splits counties on
`num_returns` into four tiers and writes
`analysis-output/source-e/source_e_tier_stats.json`. The operative result: the
cross-pillar link to Source B that survives the size control nationally
(r = 0.394) **does not exist in the smallest tier** (r = −0.058, n = 49), and the
smallest tier holds 0.14% of the dollars. A consumer serving rural counties gets
nothing from B ↔ E and should know that before relying on it.

The tiers are a diagnostic and a serving policy. They are deliberately **not** a
feature column: county size is the open question in `docs/PROJECT_GOAL.md`, and a
tier in the matrix would answer it by accident.
