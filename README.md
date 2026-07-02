# wiki_embedding

Source A of a planned multi-source ("A-F") macro/geo embedding dataset: Wikipedia introductory-text embeddings for U.S. counties.

## What it does

`ingest_source_a.py` runs an end-to-end ingestion pipeline:

1. Authenticates against the **Wikimedia Enterprise API**.
2. Fetches the full article for each county in `TEST_COUNTIES`.
3. Isolates the **lead/introductory section** only (drops infobox, later sections, and citation markers) — Wikimedia Enterprise returns a full HTML document whose body is split into `<section data-mw-section-id="N">` blocks per MediaWiki's Parsoid model; section `0` is the lead.
4. Embeds the cleaned intro text with `BAAI/bge-m3` (1024-dim) via `sentence-transformers`, running on **CPU** (MPS/GPU auto-selection caused severe per-call slowdowns on this model for short inputs).
5. L2-normalizes each embedding and writes the results to `source_a_embeddings.parquet`.

Per-county failures (article not found, empty intro) are logged and skipped rather than aborting the run; only an authentication failure stops the whole pipeline.

## Output

`source_a_embeddings.parquet` with columns:

| column | type | notes |
|---|---|---|
| `county_name` | str | e.g. `"Allegheny County, Pennsylvania"` |
| `fips_code` | str \| None | placeholder, not yet populated |
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
uv run --env-file .env ingest_source_a.py
```

First run downloads the `bge-m3` model (~2.2GB) from Hugging Face and caches it locally; subsequent runs reuse the cache.
