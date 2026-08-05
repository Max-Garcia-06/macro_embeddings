# MacroEmbeddings (`E_macro`)

A six-pillar county-level macro/geo dataset for the ~3,143 U.S. counties and county-equivalents. Each pillar (Sources A-F) is an independent federal or public data source, ingested by its own script into a single `data/*.parquet` file keyed on `fips_code`. This file documents how each pillar is ingested and what it outputs.

**Start here for context:** `docs/PROJECT_GOAL.md` — what `E_macro` is for, what the six pillars are, what stage the project is in, and the open decisions blocking the fusion step. It carries the pillar table and the keep/cut/fix verdict per source; this README does not repeat them.

All six are ingested and analyzed; a cross-pillar crossvalidation sweep and per-pillar findings reports are in `analysis-output/`. Ingestion reconnaissance for the sources that needed it (B, D, E) is in `docs/plans/ingestion_recon.md`.

The `.html` map renders the visualization scripts produce are ~5MB each and are **not committed** (`.gitignore`d) — rerun the script named in the relevant findings doc's artifact index to rebuild one.

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Sources A and C need credentials in a `.env` file (see their sections below); B, D, E, and F need none.

## Source A: Wikipedia County Text

### What it does

`ingest_source_a.py` runs an end-to-end ingestion pipeline:

1. Authenticates against the **Wikimedia Enterprise API**.
2. Fetches the full article for each of the ~3,222 US counties, county-equivalents, and Puerto Rico municipios in `ALL_COUNTIES` (derived from the Census Gazetteer counties file, cached in `county_crosswalk.parquet`).
3. Isolates the **lead/introductory section** (drops infobox, later sections, and citation markers) — Wikimedia Enterprise returns a full HTML document whose body is split into `<section data-mw-section-id="N">` blocks per MediaWiki's Parsoid model; section `0` is the lead.
4. Strips self-references and boilerplate phrasing, then writes the cleaned text and its character count to `data/source_a_text_features.parquet`.
5. Persists **every body section** to `data/source_a_sections.parquet` (64,588 rows, 20.5 sections per county, 3,144 counties). Fetch cost was always flat — the full article body was always returned and then discarded — so no future section question needs another 3,144-request crawl.

Per-county failures (article not found, empty intro) are logged and skipped rather than aborting the run; only an authentication failure stops the whole pipeline.

### The pillar ships 29 typed columns, not the text

Ingestion is step one of three. Two extraction passes then enrich `source_a_text_features.parquet` in place:

- `extract_source_a_features.py` — a fixed lexicon over the lead section (industry mentions, university, port, river, interstate, tourism, military base, tribal land, founding year, …).
- `extract_source_a_section_features.py` — the same industry lexicon applied only to sections whose title marks them economic, prefixed `sec_`. This raises industry coverage from 8.2% to 18.8% of counties.

The result is the 29-column block `pillar_matrix.build_matrix` exposes as pillar A. It beats the `content_length` scalar it replaced and ties the cut embedding on mean cross-pillar lift, and it survives a baseline that already holds every other pillar (+0.0010 mean R² lift, p = 0.013, power 0.88). Full evidence, including the wording that is and is not defensible, in `analysis-output/source-a/source-a-findings.md` §13–§17.

Two columns are written to the parquet as diagnostics and **excluded** from the scored block: `n_body_sections` (r = 0.550 with county size, and removing it costs only 2.4% of the section gain) and `has_usda_echo`.

### The embedding step was removed

This pipeline previously embedded each intro with `BAAI/bge-m3` (1024-dim) and wrote `data/source_a_embeddings.parquet`. That step was cut: the embedding correlates with economic distance at |r| = 0.041 (Mantel, n = 2,786) and k-means over it peaks at a silhouette of 0.028, meaning no recoverable cluster structure. See `analysis-output/E_macro_key_findings.ipynb` §2.

`data/source_a_embeddings.parquet` is **retained** and is no longer regenerated. The new ingestion writes to a separate path so a re-run cannot clobber it. Its one live consumer is `analyze_source_a_representation.py`, which scores the embedding head-to-head against the typed features that replaced it (`analysis-output/source-a/source-a-findings.md` §13–§17). Head to head the two are a statistical tie — 13 of 28 targets, Wilcoxon p = 0.76 — so the case for the typed block is cost, interpretability, and no 2.2GB download, not measured lift. The embedding-era EDA scripts that also read it — `analyze_source_a_clusters.py`, `analyze_source_a_cluster_stability.py`, `generate_source_a_insights.py`, and the two `analyze_source_a_source_{c,f}_correlation.py` crossvalidations — were deleted 2026-08-03; recover them from git history if the cut is reversed. `visualize_source_a.py` and `analyze_source_a_similarity.py` stay, but as shared geospatial utilities: sources B–F import `fetch_county_centroids` and `haversine_distance_matrix` from them.

### Output

`data/source_a_text_features.parquet` with columns:

| column | type | notes |
|---|---|---|
| `county_name` | str | e.g. `"Allegheny County, Pennsylvania"` |
| `fips_code` | str \| None | from `FIPS_CROSSWALK`; `None` only if a county_name is somehow missing from the crosswalk |
| `raw_intro_text` | str | cleaned lead-section text |
| `embedding_text` | str | `raw_intro_text` with self-references and boilerplate phrasing stripped |
| `content_length` | int | character count of `embedding_text` |
| 19 lead-extraction columns | bool \| int | written by `extract_source_a_features.py` — `has_university`, `has_port`, `has_river`, `has_interstate`, `has_tourism`, `has_military_base`, `has_tribal_land`, `has_protected_land`, `has_namesake`, `has_metro_attachment`, `founding_year`, `n_distinct_proper_nouns`, `n_industry_mentions`, and six `has_<industry>` flags |
| 9 section-extraction columns | bool \| int | written by `extract_source_a_section_features.py` — `has_economy_section`, `sec_n_industry_mentions`, and seven `sec_has_<industry>` flags |
| `n_body_sections`, `has_usda_echo` | int \| bool | diagnostics only; **excluded** from the scored 29-column block |

Absence is encoded as `False`/`0`, never null: the schema is uniform across all 3,144 counties and sparsity is itself the signal (a county whose article says nothing is a county about which little is written).

And `data/source_a_sections.parquet` — one row per county × body section (64,588 rows), carrying `fips_code`, `county_name`, `section_id`, `section_title`, `section_text`.

### Running

Add Wikimedia Enterprise credentials to `.env`:

```
WIKIMEDIA_USERNAME=...
WIKIMEDIA_PASSWORD=...
```

```bash
uv run --env-file .env scripts/ingest_source_a.py
```

No model download is needed -- the pipeline is now HTTP fetching plus text cleaning, so a full run is bounded by the Wikimedia Enterprise API rather than by CPU inference.

## Source B: BLS QCEW Location Quotients (Industrial Core)

`ingest_source_b.py` downloads BLS's Quarterly Census of Employment and Wages bulk quarterly file and extracts pre-calculated Location Quotients (LQ) across the 20 primary 2-digit NAICS sectors for all counties -- the "Industrial Core" pillar of `E_macro`. An LQ of 2.0 means twice the national-average concentration of jobs in that sector, regardless of the county's absolute size, distinguishing *what kind* of growth or decline a county is experiencing rather than just its direction.

Scoped to **private ownership only** (`own_code="5"`) and the most recent fully-published quarter. BLS suppresses LQ cells in counties where small employer counts could expose individual company operations (~30% of county x sector cells nationally); these are left as null (with a matching `disclosure_*` flag) rather than backfilled -- tested state-level and proportional-allocation fallbacks and neither meaningfully beat a flat null (r=0.33-0.34, barely above guessing the national average).

No credentials are required (downloads the public bulk singlefile, ~2.2GB uncompressed, streamed and filtered locally):

```bash
uv run scripts/ingest_source_b.py
```

Output: `data/source_b_qcew.parquet` with columns `county_name`, `fips_code`, `lq_emp_{naics2}` (20 columns, one per 2-digit NAICS sector) and `disclosure_{naics2}` (20 matching boolean suppression flags).

## Source C: FRED Time-Series Slope Derivatives

`ingest_source_c.py` pulls county-level annual unemployment rate and real GDP series from the **FRED API** and computes the rolling 3-year first derivative (Δy/Δt) of each, rather than storing raw levels — this is the "Economic Velocity" pillar of `E_macro`.

Both series are FIPS-derivable and pulled at **annual** frequency:

- Unemployment rate: `LAUCN{FIPS}0000000003A`
- Real GDP (chained 2017 $): `REALGDPALL{FIPS}`

FRED's monthly county unemployment series use ad-hoc state/county-abbreviation codes that aren't derivable from a FIPS code, so the annual series is used instead for both — this also means there's no seasonal component to remove before differencing. Requests are rate-limited to ~100/min. GDP series coverage is not universal (a small number of independent cities have no GDP series); those counties get a partial row with unemployment data only, rather than being dropped.

Add a FRED API key to `.env`:

```
FRED_API_KEY=...
```

```bash
uv run --env-file .env scripts/ingest_source_c.py
```

Output: `data/source_c_fred.parquet` with columns `county_name`, `fips_code`, `unemployment_velocity`, `unemployment_rate_latest`, `unemployment_latest_year`, `gdp_velocity`, `gdp_velocity_pct`, `gdp_latest`, `gdp_latest_year`.

`gdp_velocity` is denominated in chained 2017 dollars, so any ranking built on it returns the largest metro economies rather than the fastest-moving counties -- the raw and normalized top-10 lists share zero counties. **Use `gdp_velocity_pct` (= `gdp_velocity` / `gdp_latest`) for anything comparative.** Three analysis scripts were recomputing this locally before it was added to the parquet.

## Source D: BTS FAF5 Freight Trade Flows

`ingest_source_d.py` downloads the Bureau of Transportation Statistics' Freight Analysis Framework (FAF5) county-to-county and county-to-zone flow tables and derives per-county trade-volume and concentration signals -- the "Trade Logistics" pillar of `E_macro`. Ships tonnage totals (Option A, validated across two regional samples) plus a partner-concentration HHI pooled across both county-level and FAF-zone-level partner rows (Option C) rather than explicit top-K partner columns or distance-weighted "reach", both of which were tested and dropped for lacking discriminating signal.

Requires shelling out to `curl` rather than `requests` for downloads (an ordinary HTTP-client TLS-fingerprint incompatibility against `faf.ornl.gov`, not a bot-detection issue); no credentials needed.

```bash
uv run scripts/ingest_source_d.py
```

Output: `data/source_d_faf.parquet` with columns `county_name`, `fips_code`, `total_outbound_tons`, `total_inbound_tons`, `out_partner_hhi`, `in_partner_hhi`, and 5-way commodity-group (`sctg`) tonnage breakdowns per direction (`out_sctg0109`, `out_sctg1014`, `out_sctg1519`, `out_sctg2033`, `out_sctg3499` and the matching `in_sctg*` columns).

## Source E: IRS SOI County Capital-to-Wage Ratio

`ingest_source_e.py` downloads the IRS Statistics of Income pre-aggregated county files for Tax Years 2018-2022 and derives each county's position on investment income (qualified dividends + net capital gain) versus wage income -- the "Capital vs. Wage Income" pillar of `E_macro`, distinguishing counties driven by market/investment performance from those driven by local employment.

Uses the IRS's own pre-aggregated `{yy}incyallnoagi.csv` files rather than summing across the 8 per-county AGI brackets manually. Target columns are referenced via a conceptual name-to-SOI-variable-code mapping so a future schema change fails loudly instead of silently misreading a shifted column. Unlike Source B, the IRS file carries no suppression flag -- a suppressed cell and a genuine zero are indistinguishable, and this is shipped as a disclosed limitation rather than papered over.

### The ratio alone was not enough

`capital_to_wage_ratio` decomposes into three separable quantities -- how many filers report investment income, how much each reports, and how much wage income sits underneath -- at R² = 0.975 with near-unit elasticities. A high ratio can mean any of the three, and the single column maps them onto the same number. All three now ship alongside it, together with the `N00200`/`N00650`/`N01000` return counts the earlier version discarded.

Its *level* is also set by the market year rather than by county economics: the unweighted county mean runs 0.095 (TY2020), 0.156 (TY2021), 0.108 (TY2022). `capital_to_wage_ratio_normalized_mean` -- each year's ratio divided by that year's national aggregate, averaged across TY2018-TY2022 -- is the column downstream models should prefer, since it survives a vintage refresh. Evidence in `analysis-output/source-e/source-e-findings.md` §9-§13; frozen schema and null semantics in `docs/source_e_feature_schema.md`.

No credentials are required:

```bash
uv run scripts/ingest_source_e.py
```

Output: `data/source_e_irs_soi.parquet` (3,143 x 24, latest tax year plus the cross-year summary) and `data/source_e_irs_soi_panel.parquet` (15,686 rows, one per county x tax year).

Three flags ship, each naming its own mechanism: `thin_claimer_flag` (fewer than 100 filers behind the numerator; 37 counties -- the "do not trust this level" flag, and where undisclosed suppression can hide), `concentrated_gain_flag` (gain per claiming return above the national p95; 157 counties -- a few large land sales rather than a broad investment base), and `low_return_flag` (under 2,200 total returns; 325 counties). `low_return_flag` is a **materiality** flag, not a noise flag: those counties are 10.3% of rows but 0.14% of national investment income, and they are no less stable year over year than large counties. Do not weight the ratio by `num_returns` to suppress noise -- the largest counties move most between vintages, so it does the opposite.

Source E does not behave the same way across county sizes. `scripts/analyze_source_e_tiers.py` splits counties into four data-volume tiers and writes `analysis-output/source-e/source_e_tier_stats.json`; the operative result is that the strongest surviving cross-pillar link, B Real Estate LQ x E ratio, runs +0.476 among counties above 100k returns and -0.058 among those below 2,200.

## Source F: USDA ERS County Typology Codes

`ingest_source_f.py` downloads the 2025-edition USDA Economic Research Service County Typology Codes and reshapes them into one row per county -- the "Structural Resilience Baseline" anchor pillar of `E_macro`. Unlike Sources A and C, this is a single static file download: no API key, no rate limiting, and (per the source) a decennial/annual-refresh baseline rather than a time series.

The ERS file is published long-format (one row per county/attribute pair) and includes six non-exclusive "high concentration" economic flags, one mutually-exclusive dominant-industry code, six non-exclusive demographic flags, and a metro/nonmetro indicator. The pipeline pivots this to one row per county, one-hot encodes the dominant-industry code into `industry_dependence_{none,farming,mining,manufacturing,government,recreation}`, and leaves the already-binary flags as nullable booleans. ERS's sentinel codes (`99` = not classified, `-1` = insufficient data) are mapped to null rather than `False`.

No credentials are required:

```bash
uv run scripts/ingest_source_f.py
```

Output: `data/source_f_usda_typology.parquet` with columns `county_name`, `fips_code`, `metro_2023`, `high_farming`, `high_mining`, `high_manufacturing`, `high_government`, `high_recreation`, `nonspecialized`, `low_postsecondary_ed`, `low_employment`, `population_loss`, `housing_stress`, `retirement_destination`, `persistent_poverty`, and the six `industry_dependence_*` one-hot columns.

## Cross-pillar crossvalidation

`analyze_pillar_pair_crossvalidation.py` runs the full pillar-to-pillar sweep: representative scalar features from all six pillars against each other (50 feature pairs spanning all 15 pillar pairs), each permutation-tested at 499 permutations with one Benjamini-Hochberg correction across the whole sweep. Earlier crossvalidation rounds only ever tested each pillar against Source C.

Every correlation is then recomputed as a partial correlation controlling for county size (log tax returns), because large counties simultaneously move more freight, carry longer Wikipedia articles, hold more capital, and are classified metro.

```bash
uv run scripts/analyze_pillar_pair_crossvalidation.py
```

Outputs `outputs/pillar_pair_crossvalidation.csv` (per feature pair) and `analysis-output/cross-source/pillar_pair_stats.json` (sweep-level counts plus the best link per pillar pair).

**The size control matters:** 19 of 50 tests lose more than half their effect size once it is applied, including 17 of the 33 that survived the FDR correction. The largest raw effect in the sweep, Source D freight tonnage against Source F metro status at r = 0.495, falls to -0.057. The strongest surviving link is Source B's Real Estate & Rental & Leasing LQ against Source E's capital-to-wage ratio -- r = 0.394 raw, 0.382 size-controlled -- two independent federal sources identifying the same underlying economy.

**The size-controlled column is the operative one.** The downstream consumer is the Comcast FreeWheel Revenue Science team, whose training rows are impressions, ad requests, auctions, households, or devices — all per-row targets, so county size is a control rather than a feature (`docs/downstream_target.md` Part 1; asserted 2026-08-05, pending written confirmation). Do not quote a raw `r` without its size-controlled partner.

## Findings

`analysis-output/E_macro_key_findings.ipynb` consolidates all six pillars into one keep/cut/fix decision per pillar, with the evidence behind each. Per-pillar detail lives in `analysis-output/source-{a..f}/`.

Source A's typed-extraction round is the most recent work: `analysis-output/source-a/source-a-findings.md` §13–§17, with the open items and the plans for them in `docs/plans/source_a_next_steps.md`.
