---
type: results-report
date: 2026-07-14
experiment_line: source-b
round: 1
purpose: initial-ingestion-and-findings
status: active
---

# Source B — BLS Quarterly Census of Employment and Wages (Industrial Core)

## 1. Executive Summary

Source B pulls BLS QCEW's pre-computed Location Quotients (LQ) across the 20 primary 2-digit NAICS sectors, Private ownership, for 2025 Q4 -- the "Industrial Core" pillar of `E_macro`, identifying which industries actually drive a county's economy at scale-invariant resolution. Coverage is 3,143 of 3,144 crosswalk counties (99.97%); the one gap is Kalawao County, HI (population ~90), which has no QCEW private-sector county-level row at all. Three findings stand out:

1. **Extreme specialization is real, not a data artifact.** The most concentrated county nationally, Eureka County, NV (dominant LQ = 248.09, Mining), and the rest of the top 25 most-specialized counties -- oil/gas and mining counties across Texas, Alaska (including North Slope Borough, the 2nd-most-specialized county nationally at LQ = 136.08), Nevada, North Dakota, Oklahoma, and Georgia, plus two Nebraska agriculture counties -- are real, independently verifiable single-industry economies. The one non-mining/agriculture entry, Surry County, VA (LQ = 102.34, Utilities), is home to the Surry Nuclear Power Station -- exactly the kind of single-facility county the LQ metric is designed to surface (§3.2).
2. **A real suppression-handling bug caught during Phase 2 verification, not just Phase 1b's design decision.** BLS reports a suppressed cell's LQ as a literal `0`, not blank -- an early draft of `ingest_source_b.py` left these at `0.0` instead of null, silently reintroducing the exact false-zero problem Phase 0 flagged. Fixed and re-verified against real 2025 Q4 data (30.0% suppression rate reproduced exactly, zero suppressed-but-non-null cells remaining). The 10 counties with *no* disclosed or present sector at all are, without exception, tiny rural counties (Mineral County, CO; Banner and Hayes Counties, NE; Piute and Daggett Counties, UT; etc.) -- consistent with Phase 1b's finding that suppression tracks genuinely small establishment counts, not a data-quality artifact.
3. **Sector identity predicts growth better than specialization magnitude does.** A county's raw specialization magnitude (`dominant_lq`) is only weakly related to Source C's economic velocity (r = -0.095 vs. unemployment velocity, r = 0.046 vs. size-normalized GDP velocity -- both statistically significant, p < 0.02, but small in effect size). But which *specific* sector dominates matters more: Real Estate & Rental & Leasing LQ correlates with GDP velocity at r = 0.206, Wholesale Trade at r = -0.157, and Construction at r = 0.154 (all n > 2,300, all surviving Benjamini-Hochberg FDR correction across the full 20-sector scan). This is a genuinely informative result for the proposal's framing: it isn't *how specialized* a county is that signals growth direction, it's *which* industry the specialization is in.

The proposal's characterization of Source B as identifying "what kind of growth or decline a county is experiencing" (per `docs/E_macro_extendedProposal.pdf`) holds up: the per-sector correlation result is direct evidence for exactly that distinction, and it would have been invisible if the analysis had stopped at the single `dominant_lq` magnitude.

## 2. Data & Setup

`scripts/ingest_source_b.py` downloads BLS QCEW's bulk `2025_qtrly_singlefile.zip` (287MB compressed, all quarters/areas/ownerships/industries/size-classes in one file -- individual per-industry-slice URLs were tried first but 3 of the 20 combined NAICS codes, `31-33`/`44-45`/`48-49`, 404; see `docs/plans/ingestion_recon.md` § Source B, Phase 1b), stream-filters it in 500k-row chunks to `qtr="4"`, `own_code="5"` (Private, decided in Phase 1 -- government employment doesn't spread across NAICS sectors the way private employment does and would dilute the "industrial structure" signal without adding anything the proposal asked for), and `agglvl_code="74"` (county-level).

The long-format result (one row per county x NAICS-2 sector) is pivoted into one row per county: 20 `lq_emp_{naics2}` columns (that sector's Location Quotient) plus 20 matching `disclosure_{naics2}` boolean flags (`True` = BLS-suppressed). BLS reports a suppressed cell's LQ as a literal `0` rather than blank -- caught during Phase 2 verification (see §1, finding 2) -- so `lq_emp_{naics2}` is now explicitly nulled wherever the matching `disclosure_{naics2}` flag is `True`, rather than trusting the raw value at face value.

A join to `county_crosswalk.parquet` on `area_fips`/`fips_code` covers 3,143 of 3,144 counties; the one unmatched county is Kalawao County, HI, which has no QCEW private-sector county-level row of any kind (not suppressed -- simply absent, consistent with its population of roughly 90).

Output: `data/source_b_qcew.parquet`, 3,143 rows, 42 columns (`fips_code`, `county_name`, 20 `lq_emp_*`, 20 `disclosure_*`).

## 3. Main Findings

### 3.1 Dominant sector breakdown: production/extraction sectors, not diversified services

Collapsing each county's 20 sector LQs to a single dominant (highest-LQ) sector:

| Dominant sector | Count | % |
|---|---|---|
| Manufacturing | 695 | 22.1% |
| Agriculture, Forestry, Fishing & Hunting | 489 | 15.6% |
| Mining, Quarrying, Oil & Gas | 367 | 11.7% |
| Utilities | 268 | 8.5% |
| Retail Trade | 180 | 5.7% |
| Construction | 156 | 5.0% |
| Unclassified | 139 | 4.4% |
| Transportation & Warehousing | 136 | 4.3% |
| Wholesale Trade | 126 | 4.0% |
| Arts, Entertainment & Recreation | 107 | 3.4% |
| Accommodation & Food Services | 81 | 2.6% |
| Health Care & Social Assistance | 73 | 2.3% |
| Educational Services | 71 | 2.3% |
| Other Services (except Public Administration) | 52 | 1.7% |
| Management of Companies & Enterprises | 47 | 1.5% |
| Finance & Insurance | 44 | 1.4% |
| Real Estate & Rental & Leasing | 29 | 0.9% |
| Information | 25 | 0.8% |
| Professional, Scientific & Technical Services | 24 | 0.8% |
| Administrative & Support & Waste Management | 24 | 0.8% |
| Not classified (no disclosed/present sector) | 10 | 0.3% |

Manufacturing, Agriculture, Mining, and Utilities together account for 57.9% of all counties' dominant sector -- county-level industrial structure in the US is overwhelmingly production/extraction-driven rather than concentrated in tertiary/service sectors, which each individually account for under 6% of counties even at their most common. This is the opposite emphasis of Source F's typology findings (`source-f-findings.md` §3.1, where "Nonspecialized" was the single largest category at 50%) -- the two sources are measuring genuinely different things: Source F asks whether a county crosses a *threshold* of concentration in ERS's five predefined categories, while Source B's LQ-based dominant sector always names *some* sector (the highest LQ, whatever its absolute level), so a "diversified" county here still gets a nominal dominant sector rather than falling into a "none" bucket.

The 10 counties with no dominant sector at all (every one of the 20 sectors either suppressed or entirely absent) are Mineral County, CO; Banner and Hayes Counties, NE; Harding County, NM; Slope County, ND; Wheeler County, OR; Ziebach County, SD; Daggett and Piute Counties, UT; and Menominee County, WI -- all sparsely populated rural counties, consistent with Phase 1b's finding that suppressed cells have a median of 5 establishments versus 40 for disclosed cells.

### 3.2 Extreme specialization is real, not an artifact

The most specialized county nationally is Eureka County, NV (dominant LQ = 248.09, Mining) -- more than 248 times the national average concentration of mining employment. The rest of the top 25 most-specialized counties are, with one exception, real single-industry economies: West/South Texas oil & gas counties (Irion, Winkler, Reagan, Ward, Midland, Hockley, Shackelford in the Permian Basin proper; Dimmit in the Eagle Ford shale; Ochiltree in the Texas Panhandle/Anadarko Basin), Alaska oil counties (North Slope Borough at LQ = 136.08, the 2nd-highest nationally, and Southeast Fairbanks Census Area at LQ = 85.28), Nevada mining counties (Pershing, Esmeralda, White Pine, Humboldt), North Dakota Bakken counties (Dunn, Williams), Dewey County, OK (oil/gas), Wilkinson County, GA (kaolin mining), and two Nebraska agriculture counties (Wheeler, Grant). The one non-mining/agriculture entry is Surry County, VA (LQ = 102.34, Utilities) -- home to the Surry Nuclear Power Station, exactly the single-large-employer profile the LQ metric is designed to surface. This cross-check against real-world geography is strong evidence the ingestion and dominant-sector calculation are correct, not an artifact of the pivot or suppression handling.

### 3.3 Suppression: 30.0% of cells, shipped as honest nulls (full derivation in `docs/plans/ingestion_recon.md` § Source B, Phase 1b)

BLS suppresses 30.0% of county x sector cells nationally for small-employer privacy, ranging from 3.1% (Retail Trade, present almost everywhere) to 59.0% (Mining, naturally concentrated in few counties). Phase 1b empirically tested a state-level LQ fallback (MAE = 0.786, r = 0.334 against held-out disclosed cells) and a proportional-allocation proxy for the spec's proposed IPF matrix-completion fix (MAE = 0.786, r = 0.340) -- both barely better than a global-mean baseline (MAE = 0.947) and statistically indistinguishable from each other. Neither meaningfully recovers a suppressed cell's true value, so Source B ships null-passthrough with the `disclosure_*` flags preserved as explicit nullability markers, rather than a number that would read as more precise than it is.

### 3.4 Cross-validation against Source C: sector identity beats specialization magnitude

Joining Source B's dominant sector and its LQ magnitude onto Source C's velocity metrics (3,080 of 3,143 counties have a `gdp_velocity_pct`; 63 lack a GDP series, the same Connecticut Planning Region gap `source-c-findings.md` §3.2 and `source-f-findings.md` §3.2 both document independently):

| Comparison | Pearson r | permutation p |
|---|---|---|
| `dominant_lq` vs. unemployment velocity | -0.095 | 0.002 |
| `dominant_lq` vs. size-normalized GDP velocity | 0.046 | 0.018 |

Both are statistically significant but small in magnitude -- a county's raw degree of specialization, independent of what it's specialized in, is a weak growth signal at best.

Scanning each of the 20 individual sector LQs against size-normalized GDP velocity instead (restricted to sectors disclosed in >= 100 counties, and Benjamini-Hochberg FDR-corrected across all 20 to avoid overstating the significance of whichever sectors happen to rank highest by chance):

| Sector | Pearson r | raw p | FDR q | n counties |
|---|---|---|---|---|
| Real Estate & Rental & Leasing | 0.206 | 0.002 | 0.0067 | 2,396 |
| Wholesale Trade | -0.157 | 0.002 | 0.0067 | 2,307 |
| Construction | 0.154 | 0.002 | 0.0067 | 2,435 |

All three survive FDR correction with a comfortable margin (q = 0.0067 against a 0.05 threshold) and rest on samples well above the 100-county reliability floor. The direction of each is economically legible: Real Estate and Construction specialization tracking positive GDP momentum is consistent with growing local housing/development markets; Wholesale Trade's negative correlation is consistent with wholesale-hub counties skewing toward slower-growing, established goods-distribution geography rather than expansion. This is the concrete evidence for the proposal's framing that Source B "distinguishes *what kind* of growth ... a county is experiencing" -- the signal lives in sector identity, not in raw concentration.

## 4. Figure-by-Figure Interpretation

- `analysis-output/source-b/figures/source-b-figure-01-dominant-sector.png`: bar chart of the 21-way dominant-sector breakdown (20 NAICS-2 sectors plus "Not classified"). Visually confirms §3.1 -- Manufacturing, Agriculture, Mining, and Utilities are the four tallest bars by a wide margin over the remaining sixteen.
- `analysis-output/source-b/figures/source-b-figure-02-specialization-distribution.png`: histogram of `dominant_lq` across all counties. Heavily right-skewed -- most counties cluster at modest specialization (median 2.87) with a long tail out past 200, visualizing the Eureka County/oil-and-mining-belt extremes from §3.2.
- `analysis-output/source-b/figures/source-b-figure-03-specialization-vs-velocity.png`: bar chart of mean size-normalized GDP velocity by specialization quartile. Nearly flat across all four quartiles (1.8%-2.1%), visualizing §3.4's finding that raw specialization magnitude alone is a weak velocity signal.
The `.html` renders below are ~5MB each, regenerable, and no longer committed --
rebuild any of them with the script named against it in §7.

- `outputs/source_b_map_dominant_sector.html`: interactive US map, one bubble per county, colored by dominant NAICS-2 sector (20-color qualitative palette).
- `outputs/source_b_map_specialization.html`: interactive US map colored by specialization magnitude (`dominant_lq`), with the top-3 most specialized counties labeled directly.
- `outputs/source_b_industry_mix.html`, `outputs/source_b_source_c_correlation.html`: interactive versions of figures 1 and 3.

## 5. Limitations / Open Items

- **~30% of LQ cells are null by design (§3.3), not filled or estimated.** Any downstream consumer of `data/source_b_qcew.parquet` must treat `lq_emp_*` nulls as "unknown," not "zero" -- the `disclosure_*` flags exist specifically so this distinction isn't lost.
- **Private ownership only (`own_code="5"`).** A judgment call from Phase 1: government-dominated county economies (e.g., a state capital's public-administration concentration) are invisible to this dataset by design. Revisit if downstream analysis specifically needs a government-inclusive industrial view.
- **Per-sector correlations in §3.4 rest on suppression-biased subsamples even after the n >= 100 floor.** BLS suppression targets small-establishment counties non-randomly, so each sector's correlation is computed over the subset of counties where that sector happens to be large enough to disclose -- the n-floor guard reduces but doesn't eliminate this bias, and it isn't corrected for statistically this round.
- **Single reference quarter (2025 Q4).** No seasonality or multi-quarter trend analysis -- whether a county's specialization/velocity relationship is stable over time or itself has momentum is untested this round.
- **10 counties (§3.1) and 1 county (Kalawao, HI) have no usable Source B signal at all** -- an LQ-based industrial-structure feature will need an explicit missing-data policy for these 11 counties in any downstream model.

## 6. Next Actions

1. Cross-validate Source B's dominant sector against Source F's ERS typology industry-dependence category directly (e.g., does a county QCEW labels dominant-Manufacturing also carry ERS's `industry_dependence_manufacturing` flag?) -- a natural agreement check between two independently-sourced industrial-structure signals, not attempted this round.
2. Consider pulling a multi-quarter QCEW panel to test whether industrial specialization itself has a "velocity" component, analogous to Source C's GDP/unemployment velocity framing.
3. No action planned this round on the Private-only ownership scope (§5) or the suppression-bias caveat on per-sector correlations (§5) -- both are documented, deliberate scoping decisions; revisit if a downstream consumer specifically needs either resolved.

## 7. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_b.py` -> `data/source_b_qcew.parquet` (`uv run scripts/ingest_source_b.py`, no credentials required)
- Industry-mix / dominant-sector breakdown: `scripts/analyze_source_b_industry_mix.py` -> `outputs/source_b_industry_mix.csv`, `outputs/source_b_industry_mix.html`
- Cross-validation vs. Source C: `scripts/analyze_source_b_source_c_correlation.py` -> `outputs/source_b_source_c_correlation.csv`, `outputs/source_b_source_c_correlation.html`
- Maps: `scripts/visualize_source_b.py` -> `outputs/source_b_map_{dominant_sector,specialization}.html`
- Stats/figures: `scripts/generate_source_b_insights.py` -> `analysis-output/source-b/source_b_stats.json`, `analysis-output/source-b/figures/source-b-figure-*.png`, `analysis-output/source-b/figures/source-b-numeric-summary.md`
- Presentation notebook: `analysis-output/source-b/source_b_key_findings.ipynb`
- Reconnaissance/decision trail (access mechanism, suppression-handling comparison, own_code scope decision): `docs/plans/ingestion_recon.md` § Source B. The original `source_b_plan.md`, including the phase-by-phase run logs, is in git history.

## 8. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf` / `docs/macro_pre_scoping_spec.pdf`, Source B section)

The proposal frames Source B as pulling pre-calculated Location Quotients across the 20 primary NAICS-2 sectors to identify "which industries actually drive a county's economy at scale-invariant resolution," positioned to distinguish "what kind of growth or decline a county is experiencing" alongside Source C's velocity.

- **Supported**: the mechanism works exactly as described -- BLS ships pre-computed, scale-invariant LQ values for exactly the 20 target sectors, and the dominant-sector/magnitude extraction cross-checks cleanly against real-world geography (West Texas oil counties, the Bakken, Nevada mining belts, a named nuclear plant county -- §3.2).
- **Deviated, deliberately and with evidence**: the spec's own proposed suppression fix (Iterative Proportional Fitting) was flagged by the spec itself as flawed, and this round's empirical test of a cheaper, correctly-specified proxy for the same idea (proportional allocation respecting state totals) showed no improvement over a plain state-level fallback -- both barely beat a global-mean baseline. Source B ships honest nulls instead of attempting IPF, a direct, evidence-based deviation from the spec's suggested approach (§3.3).
- **New finding not anticipated by the proposal**: the proposal's "what kind of growth" framing is validated more specifically than stated -- it isn't a county's overall specialization *magnitude* that predicts growth direction (that signal is weak, §3.4), it's *which* sector is dominant, with Real Estate, Construction, and Wholesale Trade standing out as the sectors that actually track GDP momentum.
- **Not yet testable**: the proposal's implied cross-source synergies with Sources A, D, and F (e.g., does Source F's typology-based industry dependence agree with Source B's LQ-based dominant sector?) are not attempted this round (§6, item 1).
