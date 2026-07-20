# wiki_embedding

Source A of a planned multi-source ("A-F") macro/geo embedding dataset: Wikipedia introductory-text embeddings for U.S. counties.

## What it does

`ingest_source_a.py` runs an end-to-end ingestion pipeline:

1. Authenticates against the **Wikimedia Enterprise API**.
2. Fetches the full article for each of the ~3,222 US counties, county-equivalents, and Puerto Rico municipios in `ALL_COUNTIES` (derived from the Census Gazetteer counties file, cached in `county_crosswalk.parquet`).
3. Isolates the **lead/introductory section** only (drops infobox, later sections, and citation markers) — Wikimedia Enterprise returns a full HTML document whose body is split into `<section data-mw-section-id="N">` blocks per MediaWiki's Parsoid model; section `0` is the lead.
4. Embeds the cleaned intro text with `BAAI/bge-m3` (1024-dim) via `sentence-transformers`, running on **CPU** (MPS/GPU auto-selection caused severe per-call slowdowns on this model for short inputs).
5. L2-normalizes each embedding and writes the results to `data/source_a_embeddings.parquet`.

Per-county failures (article not found, empty intro) are logged and skipped rather than aborting the run; only an authentication failure stops the whole pipeline.

## Output

`data/source_a_embeddings.parquet` with columns:

| column | type | notes |
|---|---|---|
| `county_name` | str | e.g. `"Allegheny County, Pennsylvania"` |
| `fips_code` | str \| None | from `FIPS_CROSSWALK`; `None` only if a county_name is somehow missing from the crosswalk |
| `raw_intro_text` | str | cleaned lead-section text |
| `embedding` | list[float] | 1024-dim, L2-normalized |

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with Wikimedia Enterprise credentials:

```
WIKIMEDIA_USERNAME=...
WIKIMEDIA_PASSWORD=...
```

## Running

```bash
uv run --env-file .env scripts/ingest_source_a.py
```

First run downloads the `bge-m3` model (~2.2GB) from Hugging Face and caches it locally; subsequent runs reuse the cache.

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

Output: `data/source_c_fred.parquet` with columns `county_name`, `fips_code`, `unemployment_velocity`, `unemployment_rate_latest`, `unemployment_latest_year`, `gdp_velocity`, `gdp_latest`, `gdp_latest_year`.

## Source B: BLS QCEW Location Quotients (Industrial Core)

`ingest_source_b.py` downloads BLS's Quarterly Census of Employment and Wages bulk quarterly file and extracts pre-calculated Location Quotients (LQ) across the 20 primary 2-digit NAICS sectors for all counties -- the "Industrial Core" pillar of `E_macro`. An LQ of 2.0 means twice the national-average concentration of jobs in that sector, regardless of the county's absolute size, distinguishing *what kind* of growth or decline a county is experiencing rather than just its direction.

Scoped to **private ownership only** (`own_code="5"`) and the most recent fully-published quarter. BLS suppresses LQ cells in counties where small employer counts could expose individual company operations (~30% of county x sector cells nationally); these are left as null (with a matching `disclosure_*` flag) rather than backfilled -- tested state-level and proportional-allocation fallbacks and neither meaningfully beat a flat null (r=0.33-0.34, barely above guessing the national average).

No credentials are required (downloads the public bulk singlefile, ~2.2GB uncompressed, streamed and filtered locally):

```bash
uv run scripts/ingest_source_b.py
```

Output: `data/source_b_qcew.parquet` with columns `county_name`, `fips_code`, `lq_emp_{naics2}` (20 columns, one per 2-digit NAICS sector) and `disclosure_{naics2}` (20 matching boolean suppression flags).

## Source D: BTS FAF5 Freight Trade Flows

`ingest_source_d.py` downloads the Bureau of Transportation Statistics' Freight Analysis Framework (FAF5) county-to-county and county-to-zone flow tables and derives per-county trade-volume and concentration signals -- the "Trade Logistics" pillar of `E_macro`. Ships tonnage totals (Option A, validated across two regional samples) plus a partner-concentration HHI pooled across both county-level and FAF-zone-level partner rows (Option C) rather than explicit top-K partner columns or distance-weighted "reach", both of which were tested and dropped for lacking discriminating signal.

Requires shelling out to `curl` rather than `requests` for downloads (an ordinary HTTP-client TLS-fingerprint incompatibility against `faf.ornl.gov`, not a bot-detection issue); no credentials needed.

```bash
uv run scripts/ingest_source_d.py
```

Output: `data/source_d_faf.parquet` with columns `county_name`, `fips_code`, `total_outbound_tons`, `total_inbound_tons`, `out_partner_hhi`, `in_partner_hhi`, and 5-way commodity-group (`sctg`) tonnage breakdowns per direction (`out_sctg0109`, `out_sctg1014`, `out_sctg1519`, `out_sctg2033`, `out_sctg3499` and the matching `in_sctg*` columns).

## Source E: IRS SOI County Capital-to-Wage Ratio

`ingest_source_e.py` downloads the IRS Statistics of Income pre-aggregated county file and computes each county's ratio of investment income (qualified dividends + net capital gain) to wage income -- the "Capital vs. Wage Income" pillar of `E_macro`, distinguishing counties driven by market/investment performance from those driven by local employment.

Uses the IRS's own pre-aggregated `22incyallnoagi.csv` (Tax Year 2022, the latest published) rather than summing across the 8 per-county AGI brackets manually. Target columns are referenced via a conceptual name-to-SOI-variable-code mapping so a future schema change fails loudly instead of silently misreading a shifted column. Unlike Source B, the IRS file carries no suppression flag -- a suppressed cell and a genuine zero are indistinguishable, and this is shipped as a disclosed limitation rather than papered over.

No credentials are required:

```bash
uv run scripts/ingest_source_e.py
```

Output: `data/source_e_irs_soi.parquet` with columns `county_name`, `fips_code`, `num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands`, `capital_to_wage_ratio`.

## Source F: USDA ERS County Typology Codes

`ingest_source_f.py` downloads the 2025-edition USDA Economic Research Service County Typology Codes and reshapes them into one row per county -- the "Structural Resilience Baseline" anchor pillar of `E_macro`. Unlike Sources A and C, this is a single static file download: no API key, no rate limiting, and (per the source) a decennial/annual-refresh baseline rather than a time series.

The ERS file is published long-format (one row per county/attribute pair) and includes six non-exclusive "high concentration" economic flags, one mutually-exclusive dominant-industry code, six non-exclusive demographic flags, and a metro/nonmetro indicator. The pipeline pivots this to one row per county, one-hot encodes the dominant-industry code into `industry_dependence_{none,farming,mining,manufacturing,government,recreation}`, and leaves the already-binary flags as nullable booleans. ERS's sentinel codes (`99` = not classified, `-1` = insufficient data) are mapped to null rather than `False`.

No credentials are required:

```bash
uv run scripts/ingest_source_f.py
```

Output: `data/source_f_usda_typology.parquet` with columns `county_name`, `fips_code`, `metro_2023`, `high_farming`, `high_mining`, `high_manufacturing`, `high_government`, `high_recreation`, `nonspecialized`, `low_postsecondary_ed`, `low_employment`, `population_loss`, `housing_stress`, `retirement_destination`, `persistent_poverty`, and the six `industry_dependence_*` one-hot columns.
