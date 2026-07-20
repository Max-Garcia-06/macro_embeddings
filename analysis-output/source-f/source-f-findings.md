---
type: results-report
date: 2026-07-13
experiment_line: source-f
round: 1
purpose: initial-ingestion-and-findings
status: active
---

# Source F — USDA ERS County Typology Codes (Structural Resilience Baseline)

## 1. Executive Summary

Source F pulls the 2025-edition USDA Economic Research Service County Typology Codes for all 3,144 US counties/county-equivalents — the "Structural Resilience Baseline" anchor pillar of `E_macro`. Coverage is complete: every crosswalk county has a row, at the cost of a straightforward join decision (§2). Unlike Sources A and C, this is a single static file, so there is no missing-article or missing-series failure mode to characterize — the interesting findings instead concern how the *categorical* content of the typology actually splits the country, and how well its structural/demographic labels track the short-term economic momentum Source C measures. Three findings stand out:

1. **Half the country has no dominant industry** (§3.1) — the proposal's framing of Source F as sorting counties into "Farming-dependent, Mining-dependent, Manufacturing-dependent, Government-dependent, or Recreation-dependent" undersells how common "none of the above" is: 50.0% of counties (1,572 of 3,144) are `Nonspecialized`, more than double the next-largest category (Manufacturing, 22.5%).
2. **The exact same Connecticut geography-transition gap Source C found, independently surfaced here** (§3.2) — all 9 counties ERS leaves with a not-classified industry-dependence sentinel are Connecticut's new Planning Regions, the identical root cause `source-c-findings.md` §3.2 documents for Connecticut's missing GDP series. Two unrelated federal data providers have the same lag catching up to Connecticut's 2022 county abolition.
3. **Demographic distress flags don't predict short-term economic velocity** (§3.4) — cross-validating Source F's demographic risk flags (population loss, persistent poverty, etc.) against Source C's velocity metrics gives a near-zero correlation in both directions (r = 0.007 vs. unemployment velocity, r = −0.019 vs. size-normalized GDP velocity). This is a real, useful negative result: Source F is capturing *structural* character, not short-run momentum, and downstream consumers of `E_macro` should not expect the two pillars to substitute for each other.

The proposal's characterization of Source F as a stable, low-maintenance "baseline anchor" (§9) holds up well operationally — no API key, no rate limiting, one clean static download — but the *content* is more concentrated toward "unspecialized" and more structurally distinct from Source C's momentum signal than the proposal's narrative implies.

## 2. Data & Setup

`scripts/ingest_source_f.py` downloads the ERS County Typology Codes CSV (`ers.usda.gov/media/6174/...-2025-edition.csv`), which is published long-format — one row per (county, attribute) pair across 13 attributes: six non-exclusive "high concentration" economic flags, one mutually-exclusive `Industry_Dependence_2025` code (0–5), six non-exclusive demographic risk flags, and a metro/nonmetro indicator. The pipeline pivots this to one row per county, one-hot encodes the dependence code into `industry_dependence_{none,farming,mining,manufacturing,government,recreation}`, and converts the already-binary flags to nullable booleans — ERS's sentinel codes (`99` = not classified, `-1` = insufficient data) map to null rather than `False`, so "no signal" is never silently conflated with "confirmed absent."

The raw ERS file has 3,152 FIPS codes, 8 more than the crosswalk's 3,144. Those 8 are Connecticut's legacy counties (FIPS 09001–09015), which the crosswalk no longer carries — Connecticut abolished its counties as administrative units in 2022 in favor of 9 new Planning Regions, and the crosswalk (built from the 2024 Census Gazetteer) uses the new geography exclusively. An inner join on `fips_code` handles this correctly with no special-case code: the legacy rows simply don't match anything and are dropped, while the 9 new Planning Region rows join cleanly. Result: **3,144/3,144 crosswalk counties covered, zero unmatched.**

Output: `data/source_f_usda_typology.parquet`, 3,144 rows, 21 columns (`county_name`, `fips_code`, `metro_2023`, 6 binary economic "high concentration" flags, 6 binary demographic flags, 6 one-hot `industry_dependence_*` columns).

## 3. Main Findings

### 3.1 Industry dependence: "none" dominates

| Dependence category | Count | % |
|---|---|---|
| None (nonspecialized) | 1,572 | 50.0% |
| Manufacturing | 706 | 22.5% |
| Farming | 354 | 11.3% |
| Recreation | 267 | 8.5% |
| Government | 146 | 4.6% |
| Mining | 90 | 2.9% |
| Not classified (sentinel) | 9 | 0.3% |

Half the country has no single industry earning/employment concentration high enough to be labeled dependent on it. This one-hot column is mutually exclusive by construction (verified: every county's six `industry_dependence_*` flags sum to exactly 0 or 1, never more), so this is a real distributional fact, not a data artifact.

The metro/nonmetro split explains part of it but not all: among *classified* counties, 67.6% of metro counties are "None" versus 39.6% of nonmetro counties — nonmetro counties are, as expected, more likely to have a concentrated economic base, but a plurality of them (39.6%) are still unspecialized even before counting metro counties. The proposal's framing ("groups counties into mutually exclusive economic dependencies") is accurate about the mechanism but implicitly undersells how large the residual "none of the above" category is.

### 3.2 The Connecticut not-classified sentinel matches Source C's independent finding

All 9 counties carrying ERS's `99` (not classified) sentinel for industry dependence are Connecticut's Planning Regions:

Capitol, Greater Bridgeport, Lower Connecticut River Valley, Naugatuck Valley, Northeastern Connecticut, Northwest Hills, South Central Connecticut, Southeastern Connecticut, and Western Connecticut Planning Regions.

This is the same underlying cause `source-c-findings.md` §3.2 documents independently for Source C's missing-GDP-series gap: Connecticut dissolved its traditional counties in 2022, and multiple federal statistical agencies — BEA/FRED for GDP and now ERS for typology classification — haven't yet backfilled data under the new Planning Region geography. Two unrelated sources landing on the same root cause is a useful cross-source confirmation that this is a genuine upstream data-lag issue, not a bug specific to either ingestion pipeline.

### 3.3 Nonmetro counties carry more compounding demographic distress

Summing the six non-exclusive demographic flags (`low_postsecondary_ed`, `low_employment`, `population_loss`, `housing_stress`, `retirement_destination`, `persistent_poverty`) into a 0–6 "distress count" per county:

| Distress count | Counties | % |
|---|---|---|
| 0 | 1,194 | 38.0% |
| 1 | 1,338 | 42.6% |
| 2 | 392 | 12.5% |
| 3 | 143 | 4.5% |
| 4 | 60 | 1.9% |
| 5 | 17 | 0.5% |

No county in this vintage carries all 6 flags — the observed maximum is 5. Mean distress count is 1.069 for nonmetro counties versus 0.659 for metro counties, a ~62% relative gap, in line with what the six flags are designed to capture (rural/nonmetro structural risk).

### 3.4 Cross-validation against Source C: demographic distress does not track short-term velocity

Joining Source F's distress count onto Source C's velocity metrics for all 3,144 counties (64 lack a `gdp_velocity` per `source-c-findings.md` §3.2, so a size-normalized `gdp_velocity_pct = gdp_velocity / gdp_latest` — the same fix Source C's own findings §5 recommended but hadn't yet implemented — is computed for this comparison rather than reusing the raw absolute-dollar column, which is confounded with economy size the same way §5 describes):

| Comparison | Pearson r |
|---|---|
| Distress count vs. unemployment velocity | 0.0065 |
| Distress count vs. size-normalized GDP velocity | −0.0189 |

Both are indistinguishable from zero. This is a genuinely informative negative result: a county's structural demographic profile (population loss, persistent poverty, etc.) tells you essentially nothing about whether its economy is accelerating or decelerating *right now*. That's consistent with what the two sources are actually designed to measure — Source F is a slow-moving structural baseline, Source C is a 3-year momentum derivative — but it means `E_macro` consumers should treat them as genuinely complementary, orthogonal signals rather than expecting redundancy or reinforcement between them. Figure 3 (§4) shows a modest downward tilt at the highest distress bucket, but that bucket has only 17 counties and should not be read as a trend.

## 4. Figure-by-Figure Interpretation

- `analysis-output/figures/source-f-figure-01-dependence.png`: bar chart of the 7-way industry dependence breakdown. Visually confirms §3.1 — "None" is more than double the next-largest bar.
- `analysis-output/figures/source-f-figure-02-distress-distribution.png`: stacked histogram of distress count, metro vs. nonmetro. Nonmetro's distribution is visibly right-shifted relative to metro's, matching §3.3.
- `analysis-output/figures/source-f-figure-03-distress-vs-velocity.png`: mean size-normalized GDP velocity by distress count. Flat across 0–4, with a dip at 5 driven by a 17-county sample — the near-zero correlation from §3.4 visualized, including its noise floor.
- `outputs/source_f_map_dependence.html`: interactive US map, one bubble per county, colored by dominant industry dependence category.
- `outputs/source_f_map_distress.html`: interactive US map colored by demographic distress count (0–6).
- `outputs/source_f_typology.html`, `outputs/source_f_source_c_crossvalidation.html`: interactive versions of figures 1 and 3.

## 5. Limitations / Open Items

- **Distress count treats missing flags as absent, not unknown.** Individual demographic flags carry ERS sentinel nulls for a small number of counties (8–33 depending on the flag, mostly `persistent_poverty` at 33). `compute_distress_count` sums with `skipna=True`, so a county missing 2 of 6 flags is scored on the remaining 4 rather than penalized or excluded — likely a slight undercount for the handful of counties affected, not corrected this round.
- **The demographic flags are binary, not graded** — "low employment" is a threshold crossing, not a magnitude, so the 0–6 distress count treats a county barely over a threshold the same as one far over it. A future round could pull ERS's underlying continuous metrics (where available) instead of only the derived flags.
- **Cross-validation (§3.4) is a single-round snapshot** against Source C's current 3-year velocity window; the null result might not hold at longer horizons (e.g. 10-year population-loss trend vs. multi-decade GDP trend) — not tested this round.
- No validation yet against the proposal's specific "long-tail and rural markets" framing (§9) beyond the metro/nonmetro split in §3.1/§3.3 — a direct join against Source A (Wikipedia article length/existence) to test whether Source F really does anchor counties where Source A's text is thin would be a natural, low-cost follow-up.

## 6. Next Actions

1. Once Source D (BTS FAF5) exists, test the proposal's stated Trade Logistics synergy directly — "Source F anchors the long-tail and rural markets to their fundamental economic identities... establishes a baseline for how these regions absorb broader supply chain pressures" (`E_macro_extendedProposal.pdf` §3.3).
2. Check whether Source A's Wikipedia article length/intro quality is systematically thinner for the counties Source F flags as high-distress or non-metro — a direct test of the proposal's "for low-population or hyper-rural counties where online text data might be sparse... provides a solid baseline" claim.
3. No action planned this round on the Connecticut not-classified gap (§3.2) or the missing-flag undercount (§5) — revisit if a downstream consumer needs either fully resolved.

## 7. Artifact and Reproducibility Index

- Ingestion: `scripts/ingest_source_f.py` → `data/source_f_usda_typology.parquet` (`uv run scripts/ingest_source_f.py`, no credentials required)
- Industry dependence / distress breakdown: `scripts/analyze_source_f_typology.py` → `outputs/source_f_typology_breakdown.csv`, `outputs/source_f_typology.html`
- Cross-validation vs. Source C: `scripts/analyze_source_f_source_c_crossvalidation.py` → `outputs/source_f_source_c_crossvalidation.csv`, `outputs/source_f_source_c_crossvalidation.html`
- Maps: `scripts/visualize_source_f.py` → `outputs/source_f_map_{dependence,distress}.html`
- Stats/figures: `scripts/generate_source_f_insights.py` → `analysis-output/source_f_stats.json`, `analysis-output/figures/source-f-*.png`, `analysis-output/figures/source-f-numeric-summary.md`
- Presentation notebook: `analysis-output/source_f_key_findings.ipynb`

## 8. Proposal Alignment Assessment (`E_macro_extendedProposal.pdf`, Source F section)

The proposal frames Source F as grouping counties into "mutually exclusive economic dependencies: Farming-dependent, Mining-dependent, Manufacturing-dependent, Government-dependent, or Recreation-dependent," positioned as a stable structural anchor especially valuable "for low-population or hyper-rural counties where online text data might be sparse or uninformative."

- **Supported**: the mechanism works exactly as described — the six dependence flags are genuinely mutually exclusive, and ingestion required none of the accommodations Sources A/C needed (no auth, no rate limiting, no per-county lookup table, no missing-series failure mode). The metro/nonmetro distress gap (§3.3) is consistent with the "rural resilience baseline" framing.
- **Complicated**: the proposal's five-category list reads as though most counties land in one of the five dependent buckets. In practice, half don't (§3.1) — "Nonspecialized" is the single largest category by a wide margin, which is worth surfacing explicitly for anyone using this as a categorical feature, since it means the modal county contributes no dependence-category signal at all.
- **New finding not anticipated by the proposal**: an independent, second data source (ERS) hitting the exact same Connecticut Planning Region data-lag gap Source C already found (§3.2) — a useful signal that this is a genuine upstream federal-data-lag issue worth flagging if `E_macro` ever needs full Connecticut coverage.
- **Not yet testable**: the proposal's stated Trade Logistics synergy with Source D ("establishes a baseline for how these regions absorb broader supply chain pressures") requires Source D, which doesn't exist yet (§6, item 1). The Capital Flow synergy with Source A ("for... hyper-rural counties where online text data might be sparse") is directly testable now and flagged as next-round work (§6, item 2).
