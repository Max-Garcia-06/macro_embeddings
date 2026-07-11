"""Re-embed Source A counties using a local LLM (gemma2:9b via Ollama) to strip
boilerplate from raw_intro_text before bge-m3 encoding.

Replaces the regex/corpus-frequency boilerplate-stripping approach with a single
LLM rewrite pass per county: keep only county-specific facts, drop generic
templated phrasing. Validated on tracked pairs from
analysis-output/source-a-findings.md sec 3.4 plus a 128-county random sample
(see the corresponding findings section) before running on the full corpus.

Progress is appended to PROGRESS_PATH after every county, so an interrupted
run (e.g. the machine sleeping or crashing) can be resumed without re-cleaning
already-completed counties.

Usage:
    uv run python reembed_source_a_llm.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from ingest_source_a import BgeM3EmbeddingGenerator, configure_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PARQUET_PATH = REPO_ROOT / "data" / "source_a_embeddings.parquet"
OUTPUT_PARQUET_PATH = REPO_ROOT / "data" / "source_a_embeddings_llm.parquet"
PROGRESS_PATH = REPO_ROOT / "outputs" / "source_a_llm_cleaning_progress.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:9b"

# May contend with other local Ollama clients (e.g. prototype scripts) for the
# server's single request slot, so a request can sit queued for a while before
# it starts generating; generous timeout avoids spurious failures.
OLLAMA_TIMEOUT_SECONDS = 600

CLEAN_PROMPT_TEMPLATE = """The following is the Wikipedia introduction for {county_name}. \
Rewrite it keeping ONLY the facts specific to this county (geography, history, economy, \
notable places/people, distinguishing details). Remove generic templated phrasing that \
almost every US county article shares, such as "X is a county in the U.S. state of Y", \
census population boilerplate, and "the county seat is Z" sentences -- but keep the actual \
values (population figures, seat name, etc.) if they appear elsewhere in a distinctive sentence. \
Do NOT add any fact, date, place name, or detail that is not literally stated in the text below. \
Do not infer or guess at geography, economy, or history that isn't written there. If removing \
boilerplate leaves very little content, output only that little content -- do not pad it. \
Output only the rewritten text, no preamble or commentary.

Text:
{raw_text}"""


def clean_with_llm(raw_text: str, county_name: str) -> str:
    """Strip boilerplate from one county's intro text via a local Ollama model.

    Args:
        raw_text: Raw (uncleaned) Wikipedia intro text.
        county_name: County display name, used in the prompt for context.

    Returns:
        LLM-rewritten text with generic boilerplate phrasing removed.
    """
    prompt = CLEAN_PROMPT_TEMPLATE.format(county_name=county_name, raw_text=raw_text)
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def load_progress(progress_path: Path) -> dict[str, dict]:
    """Load already-cleaned counties from the progress file, keyed by county_name.

    Args:
        progress_path: Path to the JSONL progress file; need not exist yet.

    Returns:
        Mapping of county_name to its completed result row. Empty if the file
        doesn't exist, i.e. this is a fresh run.
    """
    if not progress_path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in progress_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        completed[entry["county_name"]] = entry
    return completed


def append_progress(progress_path: Path, entry: dict) -> None:
    """Append one completed county's result to the JSONL progress file.

    Args:
        progress_path: Path to the JSONL progress file.
        entry: Completed result row for one county.
    """
    with progress_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    """Clean and re-embed every county's raw_intro_text with the local LLM."""
    configure_logging()

    df = pd.read_parquet(SOURCE_PARQUET_PATH)
    completed = load_progress(PROGRESS_PATH)
    logger.info("Resuming with %d/%d counties already cleaned", len(completed), len(df))

    embedder = BgeM3EmbeddingGenerator(device="cpu")

    for _, row in df.iterrows():
        county_name = row["county_name"]
        if county_name in completed:
            continue

        logger.info("Cleaning '%s'...", county_name)
        try:
            cleaned_text = clean_with_llm(row["raw_intro_text"], county_name)
        except requests.exceptions.RequestException as exc:
            logger.warning("Skipping '%s' after request failure: %s", county_name, exc)
            continue
        raw_vector = embedder.encode(cleaned_text)
        normalized_vector = embedder.l2_normalize(raw_vector)

        entry = {
            "county_name": county_name,
            "fips_code": row["fips_code"],
            "raw_intro_text": row["raw_intro_text"],
            "embedding_text": cleaned_text,
            "embedding": normalized_vector.tolist(),
        }
        append_progress(PROGRESS_PATH, entry)
        completed[county_name] = entry

    result_df = pd.DataFrame(list(completed.values()))
    result_df.to_parquet(OUTPUT_PARQUET_PATH, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(result_df), OUTPUT_PARQUET_PATH)


if __name__ == "__main__":
    main()
