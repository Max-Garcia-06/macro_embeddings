"""Structural features of a county's Wikipedia article -- shape, not content.

Everything here is derived from two things: a section's title and how many
characters it contains. No section text is read. That restriction is the point
of the round: `analyze_source_a_representation.py` scores what the article
*says*, and this module asks what the article's skeleton knows on its own.

The prior is that the answer is "county size". `n_body_sections` was computed
during the section round, correlated r = 0.550 against log tax returns -- above
`content_length`'s 0.359 -- and cut from the scored block for that reason
(`pillar_matrix.SOURCE_A_DIAGNOSTIC_COLUMNS`). So this module only builds the
block; `analyze_source_a_structure.py` scores it on a baseline that already
holds three size measures, where a pure size proxy is worth approximately
nothing.

Output: `data/source_a_structure_features.parquet`, one row per county.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from extract_source_a_section_features import SECTIONS_PARQUET_PATH
from pillar_matrix import DATA_DIR

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

STRUCTURE_FEATURES_PATH: Path = DATA_DIR / "source_a_structure_features.parquet"
STRUCTURE_FEATURE_STATS_PATH: Path = ANALYSIS_DIR / "source_a_structure_feature_stats.json"

# A section shorter than this is a stub -- a heading with a sentence under it.
# Not arbitrary: the corpus-wide first quartile of section length is 108
# characters and the median is 340, so 200 splits the bottom of the
# distribution rather than trimming a tail.
STUB_CHAR_THRESHOLD: int = 200

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def normalize_titles(sections: pd.DataFrame) -> pd.Series:
    """Strip and case-fold section titles.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized title per row, aligned to `sections.index`. Untitled
        sections -- 2,009 rows corpus-wide -- become the empty string rather
        than being dropped.
    """
    return sections["section_title"].fillna("").str.strip().str.lower()


def gini(values: np.ndarray) -> float:
    """Gini coefficient over a county's section lengths.

    Measures how unevenly the article's characters are distributed across its
    sections: 0 when every section is the same length, approaching 1 when one
    section holds nearly everything.

    Args:
        values: Section lengths for one county.

    Returns:
        Gini coefficient, or 0.0 for an empty or all-zero input.
    """
    if len(values) == 0:
        return 0.0
    ordered = np.sort(values.astype("float64"))
    total = float(ordered.sum())
    if total <= 0:
        return 0.0
    n = len(ordered)
    index = np.arange(1, n + 1)
    return float(2.0 * float((index * ordered).sum()) / (n * total) - (n + 1) / n)


def count_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Count how many sections a county's article has, and of what kind.

    `section_id` is Parsoid's numbering and is not contiguous within a county:
    ids are skipped where sections nest or were dropped during ingestion, so
    `n_id_gaps` is a free structural signal about how deep the article's
    hierarchy goes.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with the five count columns.
    """
    titles = normalize_titles(sections)
    frame = sections.assign(_title=titles, _untitled=(titles == "").astype("int64"))
    grouped = frame.groupby("fips_code")

    counts = pd.DataFrame(
        {
            "n_body_sections": grouped.size(),
            "n_distinct_titles": grouped["_title"].nunique(),
            "n_untitled_sections": grouped["_untitled"].sum(),
            "max_section_id": grouped["section_id"].max(),
        }
    )
    counts["n_id_gaps"] = counts["max_section_id"] - counts["n_body_sections"]
    counts.index.name = "fips_code"
    return counts


def length_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Summarize how long a county's sections are and how evenly.

    Spread statistics are filled rather than left null. A one-section county has
    an undefined sample standard deviation, and "undefined" here means "there is
    no spread", not "unknown" -- letting it through as NaN would hand the
    imputer a median from counties that are not comparable.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with the nine length columns.
    """
    chars = sections["section_text"].fillna("").str.len().astype("float64")
    frame = sections[["fips_code"]].assign(
        _chars=chars, _stub=(chars < STUB_CHAR_THRESHOLD).astype("int64")
    )
    grouped = frame.groupby("fips_code")

    lengths = pd.DataFrame(
        {
            "total_body_chars": grouped["_chars"].sum(),
            "mean_section_chars": grouped["_chars"].mean(),
            "median_section_chars": grouped["_chars"].median(),
            "max_section_chars": grouped["_chars"].max(),
            "sd_section_chars": grouped["_chars"].std().fillna(0.0),
            "n_stub_sections": grouped["_stub"].sum(),
            "section_length_gini": grouped["_chars"].apply(lambda s: gini(s.to_numpy())),
        }
    )
    totals = lengths["total_body_chars"].replace(0.0, np.nan)
    lengths["share_in_largest_section"] = (lengths["max_section_chars"] / totals).fillna(0.0)
    lengths["share_stub_sections"] = lengths["n_stub_sections"] / grouped.size()
    lengths.index.name = "fips_code"
    return lengths
