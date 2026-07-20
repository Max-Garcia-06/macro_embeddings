# Source E: IRS Statistics of Income Panel Data (Capital Composition)

## Context

Sources A (Wikipedia embeddings), B (BLS QCEW), C (FRED velocity), D (BTS FAF5 trade flows), and F (USDA typology) are complete.

**Source E = "IRS Statistics of Income Panel Data"**: pull annual county-level individual income tax return aggregates from the IRS SOI division and compute the ratio of investment income (net capital gains + qualified dividends) to W-2 wage income. Conceptually this is the "Capital Flow" pillar of `E_macro`, per `docs/E_macro_extendedProposal.pdf` — it isolates asset-rich, investment-driven markets (sensitive to monetary policy and equity performance) from pure labor-dependent markets (sensitive to layoffs and manufacturing cuts), a distinction the Census's raw income totals collapse into a single indistinguishable number.

`docs/macro_pre_scoping_spec.pdf` flags one risk up front: "upstream schema mutation" — tax form line items get renamed/reordered/removed across fiscal years — and proposes an abstraction layer mapping raw columns to invariant conceptual names (`total_w2_wages`, `net_cap_gains`) rather than tracking literal line numbers.

## Risks (in priority order)

### Risk 1: access mechanism — resolved, simplest of all six sources
`www.irs.gov` (both the landing page and the direct `/pub/irs-soi/*.csv` file host) has **no bot protection** — confirmed via plain `curl` and Python `requests`, 200 OK, no special headers, no TLS quirk. Unlike Sources B/D, there is no landing-page/data-host split to navigate at all — same host serves everything.

### Risk 2: schema mutation across years — real, mitigated by picking the pre-aggregated file and using conceptual column mapping (per the spec's own proposed fix)
Confirmed via the live Tax Year 2022 documentation guide (`22incydocguide.docx`, pulled and parsed): line items **do** get renamed/added/removed year over year (e.g., TY2022 added `N00400`/`A00400` tax-exempt interest and dropped several TY2021 COVID-era credit fields). The columns this pipeline actually needs — `A00200` (salaries and wages, 1040 line 1z), `A00650` (qualified dividends, 1040 line 3a), `A01000` (net capital gain/loss, 1040 line 7), `A00100` (AGI, 1040 line 11) — have stable SOI variable codes across at least TY2021-2022 per the changelog, but the spec's concern is legitimate for future years. Mitigated the way the spec recommended: reference columns by name via a small `SOI_COLUMN_MAP` constant (conceptual name → SOI variable code) rather than positional indexing, so a future schema change fails loudly (`KeyError`) instead of silently misreading a shifted column.

### Risk 3 (new, not flagged in either scoping doc): no suppression flag — real, and more opaque than Source B's equivalent
The Tax Year 2022 documentation guide confirms IRS SOI applies the same kind of small-cell privacy suppression BLS QCEW does ("Income and tax items with less than 20 returns for a county were excluded"), but **unlike BLS's `disclosure_code` column, the IRS county file carries no suppression flag at all** — a suppressed cell and a genuine zero are both written as a bare `0`, indistinguishable from each other. Confirmed empirically: 3 of 3,143 counties (Kenedy, King, Loving — all ultra-low-population West Texas counties, N1 40-140 returns) show `A01000=0`/`A00650=0` for both the amount *and* the underlying return-count columns, consistent with either a true zero or full suppression; there is no way to tell which from this file alone. Decision: ship the raw values as-is (no null-passthrough possible without a flag to key off of) and document this as an accepted, disclosed limitation in the findings doc rather than fabricate a distinction the source data doesn't support.

## Phases

### Phase 0: Reconnaissance — complete (2026-07-14)
1. Confirmed `www.irs.gov/statistics/soi-tax-stats-county-data` lists per-tax-year subpages (`-2011` through `-2022`); a `-2023` guess 404s, so **Tax Year 2022 is the latest available county file** (the WebSearch tool's claim of a "TY2023 release on 2026-08-13" did not check out against the live site and was dropped — a real hallucination the live `curl`/page-scrape caught, not a source discrepancy).
2. Confirmed via the TY2022 documentation guide that the county data ships in **5 formats**: 51 per-state `.xlsx` files, a gross county `.xlsx` excluding AGI classes, `22incyallagi.csv` (all states, one row per county **per AGI bracket**, 8 brackets), `22incyallnoagi.csv` (all states, **one row per county, `AGI_STUB` fixed at 0** — i.e. IRS's own pre-aggregated county total, no need to sum 8 brackets ourselves), and a CBSA-level file (not needed here). **Chose `22incyallnoagi.csv`** — same "use the source's own pre-solved aggregate" pattern as Source B's bulk-singlefile discovery and Source D's county/FAF-zone hybrid, avoiding both a 51-file per-state download loop and a manual AGI-bracket summation.
3. Downloaded and parsed `https://www.irs.gov/pub/irs-soi/22incyallnoagi.csv` directly (`requests`, no auth, `latin-1` encoding — the file has non-UTF-8 bytes, e.g. in a handful of county names). 3,194 rows: 3,143 real counties (`COUNTYFIPS != "000"`) + 51 state totals + DC. **3,143/3,144 crosswalk coverage — the one gap is Kalawao County, HI (`15005`, pop. ~90), the exact same county Source B's QCEW ingestion was also missing**, not a new gap.
4. Confirmed real column meanings against the doc guide's field-reference table (all amounts in **thousands of dollars**, per the guide's explicit unit note):
   - `A00100` = Adjusted gross income (1040 line 11)
   - `N00200`/`A00200` = count / amount of salaries and wages (1040 line 1z) — this is the "Standard W-2 Wage Income" denominator the proposal asks for.
   - `N00650`/`A00650` = count / amount of qualified dividends (1040 line 3a)
   - `N01000`/`A01000` = count / amount of net capital gain (less loss) (1040 line 7 — the return-level total after Schedule D, when Schedule D is required) — this is the "Form 1040 Schedule D (Capital Gains)" numerator component the proposal asks for.
5. Computed the proposal's ratio (`(A01000 + A00650) / A00200`) end-to-end on the real file as a sanity check before writing pipeline code: median 0.082, mean 0.107, and **zero division-by-zero cases** (`A00200` min across all 3,143 counties is $2.872M, never zero). Top 5 counties by ratio are all well-known ultra-wealthy resort/investment counties, not noise: **Teton County, WY** (Jackson Hole, ratio 2.80), **Pitkin County, CO** (Aspen, 1.57), **Blaine County, ID** (Sun Valley, 1.24), **Collier County, FL** (Naples, 1.13), **Monroe County, FL** (Florida Keys, 1.11) — real-world validation of the signal before any pipeline code exists, same spirit as Source D's Loving County TX petroleum-export spot check.

### Phase 1: Credentials & dependencies — complete (2026-07-14)
- **No API key/credentials required** (confirmed — plain `requests`, no auth, no rate limiting observed).
- **No new dependencies** needed beyond what's already in `pyproject.toml` (`requests`, `pandas`, `pyarrow`).
- **Reference year: Tax Year 2022** (the latest published, per Phase 0 step 1).

### Phase 2: `scripts/ingest_source_e.py` — complete (2026-07-14)
Mirrors Sources A/B/C/D/F's established architecture (module docstring, `configure_logging`/`main` entrypoint, isolated download/transform/storage sections):
- Downloads `22incyallnoagi.csv` via plain `requests` (no streaming/chunking needed — 5MB, unlike Source B's 2.2GB bulk file).
- Parses with `latin-1` encoding; drops `COUNTYFIPS == "000"` state-total rows; builds `fips_code` from zero-padded `STATEFIPS` + `COUNTYFIPS`.
- References target columns via a `SOI_COLUMN_MAP` conceptual-name mapping (per Risk 2's mitigation), not raw SOI codes, in the output schema.
- Computes `capital_to_wage_ratio = (net_cap_gain + qualified_dividends) / wages_salaries`.
- Joins to `county_crosswalk.parquet` on `fips_code` (direct match, no new crosswalk needed, per Phase 0).
- Output: `data/source_e_irs_soi.parquet` — 3,143 rows, 8 columns (`fips_code`, `county_name`, `num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`, `capital_to_wage_ratio`).
- **Verified**: 3,143/3,144 crosswalk coverage exactly as Phase 0 predicted (only Kalawao County, HI missing — the same gap Source B independently has); zero nulls; ratio well-defined for all counties (`wages_salaries_thousands` never zero, min $2.872M); spot-checked Teton County, WY (ratio 2.798) and Loving County, TX (ratio 0.0) against the raw CSV, both match exactly.

### Phase 3: Analysis scripts (mirroring the per-concern script split) — complete (2026-07-14)
- `scripts/analyze_source_e_capital_composition.py` — ranks counties by `capital_to_wage_ratio` and buckets into data-driven quartiles (`capital_quartile`), analogous to Source B/D's `dominant_lq`/`tons_quartile` collapses.
- `scripts/analyze_source_e_source_c_correlation.py` — cross-validates capital composition against Source C's GDP/unemployment velocity, analogous to `analyze_source_d_source_c_correlation.py`'s single-scalar cross-validation shape.
- `scripts/visualize_source_e.py` — US bubble map of `capital_to_wage_ratio`, mirroring `visualize_source_c.py`'s single-continuous-signal map structure.
- `scripts/generate_source_e_insights.py` — headline stats to `analysis-output/source-e/source_e_stats.json`, three static figures, and a numeric-summary markdown, mirroring `generate_source_b_insights.py`.

All four scripts ran end-to-end without errors. Real result: the top 25 counties by `capital_to_wage_ratio` validate against real-world geography (Teton WY/Jackson Hole, Pitkin CO/Aspen, etc.) but do **not** split cleanly into just "resort" vs. "low-N noise" as first assumed — fact-checking caught Benton County, AR (Walmart/Tyson/J.B. Hunt HQ, 138,170 returns) as a third, distinct mechanism (concentrated executive equity), plus two unexplained residual counties (Randolph MO, Nemaha KS). See `source-e-findings.md` §1/§3.1 for the corrected breakdown. Cross-validation against Source C: `capital_to_wage_ratio` vs. size-normalized GDP velocity r=0.0705 (p=0.002, significant), vs. unemployment velocity r=-0.0297 (p=0.112, not significant) — directionally consistent with the proposal's "Wall Street performance, not local job survival" framing.

### Phase 4: Findings deliverable — complete (2026-07-14)
- `analysis-output/source-e/source-e-findings.md` — same structure/frontmatter conventions as Sources B/D's findings docs, with an explicit proposal-alignment section covering: the pre-aggregated-file access pattern (vs. the spec's assumed zip-archive ingestion), the conceptual-column-mapping mitigation for schema mutation, and two new findings not anticipated by either scoping doc (Risk 3's undisclosed-suppression gap, and the executive-equity/one-time-transaction noise sources behind the top-ratio counties).
- `analysis-output/source-e/source_e_key_findings.ipynb`, built with `nbformat` and executed in place via `jupyter nbconvert --execute --inplace`, 18 cells, zero execution errors.
- **Fact-checking pass caught a real overclaim, fixed before finalizing**: the findings doc's first draft asserted the top-25 ratio counties "split cleanly into two groups" (resort counties vs. low-return-count noise). Re-sorting the top 25 by `num_returns` surfaced Benton County, AR (138,170 returns, corporate-HQ county, not a resort) as a genuine third mechanism, plus two counties (Randolph MO, Nemaha KS) that fit neither bucket — corrected in both the executive summary and §3.1 rather than left as a false clean-split claim.

## Verification
1. Confirm Phase 0's chosen file (`22incyallnoagi.csv`) and column mapping are correct before Phase 2 code is written. **Done.**
2. Run ingestion against the real file; confirm `data/source_e_irs_soi.parquet` row count (~3,142-3,143) and crosswalk coverage. **Done — 3,143 rows, matches Phase 0 prediction exactly.**
3. Spot-check 2-3 counties' ratios manually against the raw CSV. **Done — Teton County, WY and Loving County, TX both match exactly.**
4. Run each analysis script and confirm expected output files (CSV/HTML/PNG) without errors. **Done — all 4 scripts ran clean.**
5. Read `analysis-output/source-e/source-e-findings.md` for internal consistency; confirm the proposal-alignment section gives real numbers. **Done — caught and fixed the two-group overclaim above.**

## Status
**All phases (0 through 4) complete (2026-07-14).** Access mechanism (direct, no bot-gating), reference year (Tax Year 2022), and file choice (`22incyallnoagi.csv`, pre-aggregated) all confirmed with live data. `scripts/ingest_source_e.py` produced `data/source_e_irs_soi.parquet` (3,143 rows). All 4 Phase 3 analysis scripts run cleanly. `analysis-output/source-e/source-e-findings.md` and `source_e_key_findings.ipynb` are written, executed, and fact-checked against the underlying data (one real overclaim caught and corrected). Source E is done pending final user review.
