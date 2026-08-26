# Source A Structural Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much of a county can be read off the *shape* of its Wikipedia article — section counts, section lengths, which section titles are present — without reading any of the text.

**Architecture:** One extraction module turns `data/source_a_sections.parquet` into a county-keyed structural feature parquet. One analysis module scores that block against the 28-target cross-pillar basket in four arms, reusing the residual-ridge protocol from `analyze_source_a_representation.py` rather than reimplementing it. One builder script emits the notebook from the committed artifacts.

**Tech Stack:** Python 3.12, `uv`, pandas, numpy, scikit-learn, scipy, matplotlib, nbformat, pytest.

**Design:** `docs/superpowers/specs/2026-08-25-source-a-structure-features-design.md`

> ## Correction (2026-08-25, post-review) — read before using this plan as a reference
>
> This plan was executed as written. A whole-branch review then falsified two
> claims it carries, both of which appear verbatim in the docstrings and notebook
> prose the tasks below dictate. The code has been corrected; the tasks below are
> left as they were executed, because a plan edited to match its outcome stops
> being a record of what was done. Do not copy either claim forward.
>
> 1. **"a pure size proxy is worth approximately nothing"** (in the extraction
>    module docstring at Task 1, and again in the analysis module docstring) —
>    false. It is true only of a *linear* size proxy. The baseline holds
>    `log_population`, `log_agi` and `log_gdp_latest` linearly, in logs. A block
>    of squares, cubes and pairwise products of those same three columns, adding
>    no information the baseline lacks, scores **+0.01748** mean lift on 26 of 28
>    targets (p = 1.3e-06) against the structural block's **+0.00269** — six and
>    a half times as much. A fourth arm, `size_nonlinear`, now reports this
>    calibration in every artifact. "Lift" in this round means "beyond a
>    linear-in-logs size model", not "beyond county size".
> 2. **"sets the expectation: most of these columns are size measurements"** (the
>    notebook builder's docstring and its Part-two prose, Task 5) — false. The
>    audit found **6 of 64** columns clearing |r| = 0.4 against any size measure.
>    The audit was also linear-only, and so blind to the channel correction 1
>    describes; it now carries an out-of-fold R² against a degree-3 size basis
>    and a variance column beside the correlation.
>
> What the round actually found, in the house format, is
> `analysis-output/source-a/source-a-findings.md` §23.

## Global Constraints

- Python 3.12; every command runs under `uv run` from the repo root.
- The extraction module reads `section_title` and `len(section_text)` **only**. Section text never survives past the character count. This is the premise of the round, not a style preference.
- Every module gets a module docstring stating what it computes and why, and every public function gets a Google-style docstring with `Args:` and `Returns:`. This repo's scripts are written that way without exception.
- Type annotations on every signature; `from __future__ import annotations` at the top of each module.
- Constants are module-level, uppercase, annotated, and carry a comment explaining any number that could have been chosen differently.
- No number is typed into the notebook by hand. The notebook reads parquet and JSON artifacts.
- Matplotlib only, never plotly — plotly's mimetype output needs a JupyterLab extension and renders as blank space without it.
- `data/source_a_structure_features.parquet` must never enter `pillar_matrix.build_matrix`. Verified safe: that function loads explicit paths and does not glob.
- Scoring protocol is fixed: unpenalized OLS baseline of `pillar_matrix.SIZE_FEATURES` (`log_population`, `log_agi`, `log_gdp_latest`) plus state dummies, ridge fitted to its residuals, penalty by nested crossvalidation, `N_FOLDS = 5`, `RANDOM_SEED = 42`, identical folds and rows across arms.
- Work happens on branch `source-a-structure-features`, already created.

---

### Task 1: Count and length features

**Files:**
- Create: `scripts/extract_source_a_structure_features.py`
- Create: `tests/test_source_a_structure.py`

**Interfaces:**
- Consumes: `data/source_a_sections.parquet` — columns `fips_code`, `county_name`, `section_id`, `section_text`, `section_title`; 64,588 rows over 3,144 counties.
- Produces:
  - `normalize_titles(sections: pd.DataFrame) -> pd.Series` — stripped, case-folded titles aligned to `sections.index`
  - `gini(values: np.ndarray) -> float`
  - `count_features(sections: pd.DataFrame) -> pd.DataFrame` — indexed by `fips_code`
  - `length_features(sections: pd.DataFrame) -> pd.DataFrame` — indexed by `fips_code`
  - `STUB_CHAR_THRESHOLD: int = 200`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_source_a_structure.py`. The existing `tests/conftest.py` already puts `scripts/` on `sys.path` and provides a session-scoped `sections_frame` fixture, so the import below works and the real parquet is available without reloading it per test.

```python
"""Structural features derived from Wikipedia section titles and lengths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import extract_source_a_structure_features as structure


def make_sections(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Build a section frame from (fips_code, section_id, title, text) tuples."""
    return pd.DataFrame(
        rows, columns=["fips_code", "section_id", "section_title", "section_text"]
    ).assign(county_name="Test County")


def test_titles_are_stripped_and_casefolded() -> None:
    sections = make_sections([("01001", 1, "  Demographics ", "x"), ("01001", 2, "ECONOMY", "y")])

    assert list(structure.normalize_titles(sections)) == ["demographics", "economy"]


def test_counts_cover_sections_titles_and_blanks() -> None:
    sections = make_sections(
        [
            ("01001", 1, "History", "a"),
            ("01001", 2, "History", "b"),
            ("01001", 3, "   ", "c"),
        ]
    )

    counts = structure.count_features(sections)

    assert counts.loc["01001", "n_body_sections"] == 3
    assert counts.loc["01001", "n_distinct_titles"] == 2  # "history" and ""
    assert counts.loc["01001", "n_untitled_sections"] == 1


def test_id_gaps_are_zero_when_ids_are_contiguous() -> None:
    sections = make_sections([("01001", i, f"S{i}", "x") for i in range(1, 6)])

    assert structure.count_features(sections).loc["01001", "n_id_gaps"] == 0


def test_id_gaps_count_skipped_parsoid_ids() -> None:
    """Parsoid numbers nested sections it does not emit; the gap is the signal."""
    sections = make_sections([("01001", 1, "A", "x"), ("01001", 2, "B", "y"), ("01001", 9, "C", "z")])

    assert structure.count_features(sections).loc["01001", "n_id_gaps"] == 6


def test_length_summary_uses_character_counts() -> None:
    sections = make_sections([("01001", 1, "A", "a" * 100), ("01001", 2, "B", "b" * 300)])

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "total_body_chars"] == 400
    assert lengths.loc["01001", "mean_section_chars"] == 200
    assert lengths.loc["01001", "max_section_chars"] == 300
    assert lengths.loc["01001", "share_in_largest_section"] == pytest.approx(0.75)


def test_stub_threshold_splits_at_200_characters() -> None:
    sections = make_sections(
        [("01001", 1, "A", "a" * 199), ("01001", 2, "B", "b" * 200), ("01001", 3, "C", "c" * 400)]
    )

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "n_stub_sections"] == 1
    assert lengths.loc["01001", "share_stub_sections"] == pytest.approx(1 / 3)


def test_single_section_county_has_zero_spread_not_nan() -> None:
    """A one-section county has an undefined sample sd; it must not reach the model as NaN."""
    sections = make_sections([("01001", 1, "A", "a" * 500)])

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "sd_section_chars"] == 0.0
    assert lengths.loc["01001", "section_length_gini"] == 0.0


def test_gini_is_zero_for_equal_sections() -> None:
    assert structure.gini(np.array([300.0, 300.0, 300.0])) == pytest.approx(0.0)


def test_gini_rises_when_one_section_dominates() -> None:
    even = structure.gini(np.array([100.0, 100.0, 100.0, 100.0]))
    lopsided = structure.gini(np.array([10.0, 10.0, 10.0, 5000.0]))

    assert lopsided > even
    assert lopsided < 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'extract_source_a_structure_features'`.

- [ ] **Step 3: Write the module with count and length features**

Create `scripts/extract_source_a_structure_features.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_structure_features.py tests/test_source_a_structure.py
git commit -m "feat(source-a): count and measure article sections without reading them"
```

---

### Task 2: Title-presence flags

**Files:**
- Modify: `scripts/extract_source_a_structure_features.py`
- Modify: `tests/test_source_a_structure.py`

**Interfaces:**
- Consumes: `normalize_titles` from Task 1.
- Produces:
  - `TITLE_FLAG_MIN_SHARE: float = 0.05`
  - `slugify(title: str) -> str`
  - `flag_vocabulary(sections: pd.DataFrame) -> list[str]` — normalized titles present in more than `TITLE_FLAG_MIN_SHARE` of counties, most common first
  - `title_flag_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame` — indexed by `fips_code`, columns `has_section_<slug>`, float64 0.0/1.0

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_structure.py`:

```python
def test_vocabulary_keeps_titles_above_the_share_floor() -> None:
    """Four counties: 'geography' is in three, 'quirk' in one. The floor is 5%."""
    rows = [(f"0100{i}", 1, "Geography", "x") for i in range(1, 4)]
    rows.append(("01004", 1, "Quirk", "x"))
    rows.append(("01004", 2, "Geography", "x"))

    vocabulary = structure.flag_vocabulary(make_sections(rows))

    assert "geography" in vocabulary
    assert "quirk" in vocabulary  # 1 of 4 counties = 25%, above the floor


def test_vocabulary_drops_a_title_below_the_share_floor() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 41)]
    rows.append(("00007", 2, "One Off", "x"))  # 1 of 40 counties = 2.5%

    vocabulary = structure.flag_vocabulary(make_sections(rows))

    assert "geography" in vocabulary
    assert "one off" not in vocabulary


def test_vocabulary_is_derived_not_hardcoded() -> None:
    """A corpus with a different title distribution produces different flags."""
    rows = [(f"0100{i}", 1, "Volcanology", "x") for i in range(1, 5)]

    assert structure.flag_vocabulary(make_sections(rows)) == ["volcanology"]


def test_vocabulary_is_ordered_by_county_count() -> None:
    rows = [(f"0100{i}", 1, "Geography", "x") for i in range(1, 5)]
    rows += [(f"0100{i}", 2, "Economy", "x") for i in range(1, 3)]

    assert structure.flag_vocabulary(make_sections(rows)) == ["geography", "economy"]


def test_untitled_sections_do_not_enter_the_vocabulary() -> None:
    rows = [(f"0100{i}", 1, "   ", "x") for i in range(1, 5)]

    assert structure.flag_vocabulary(make_sections(rows)) == []


def test_slugify_produces_a_valid_column_name() -> None:
    assert structure.slugify("2020 census") == "2020_census"
    assert structure.slugify("law and government") == "law_and_government"
    assert structure.slugify("census-designated places") == "census_designated_places"


def test_slugify_survives_a_title_with_no_usable_characters() -> None:
    """2,009 sections are untitled; slugification must not produce an empty column name."""
    assert structure.slugify("") == "untitled"
    assert structure.slugify("---") == "untitled"


def test_flags_are_binary_per_county() -> None:
    rows = [
        ("01001", 1, "Geography", "x"),
        ("01001", 2, "Geography", "x"),  # twice in one county is still one flag
        ("01002", 1, "Economy", "x"),
    ]

    flags = structure.title_flag_features(make_sections(rows), ["geography", "economy"])

    assert flags.loc["01001", "has_section_geography"] == 1.0
    assert flags.loc["01001", "has_section_economy"] == 0.0
    assert flags.loc["01002", "has_section_geography"] == 0.0
    assert set(flags["has_section_geography"].unique()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 8 failures with `AttributeError: module 'extract_source_a_structure_features' has no attribute 'flag_vocabulary'` (and `slugify`, `title_flag_features`).

- [ ] **Step 3: Add the vocabulary and flag builders**

Append to `scripts/extract_source_a_structure_features.py`, after `length_features`:

```python
# A title earns a flag when it appears in more than this share of counties.
# 5% of 3,144 counties is a ~157-county floor, which keeps the head of the
# distribution -- `demographics` at 3,142 counties down through the mid-tail --
# and drops the one-off titles that are really county names in disguise.
TITLE_FLAG_MIN_SHARE: float = 0.05

# Prefix marking a column as "this section title was present", so a consumer can
# tell these apart from the shipped lexicon flags (`has_university`,
# `has_economy_section`) that describe what the text says.
TITLE_FLAG_PREFIX: str = "has_section_"


def slugify(title: str) -> str:
    """Convert a section title into a valid, stable column suffix.

    Args:
        title: Normalized section title.

    Returns:
        Lowercase alphanumeric-and-underscore slug, or `"untitled"` when the
        title has no usable characters.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "untitled"


def flag_vocabulary(sections: pd.DataFrame) -> list[str]:
    """Choose which section titles are common enough to flag.

    Computed from the corpus rather than hardcoded, so the vocabulary moves when
    the corpus does. The chosen set is written to the stats file, which is what
    makes a shifting vocabulary auditable instead of silent.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized titles held by more than `TITLE_FLAG_MIN_SHARE` of counties,
        most common first. Untitled sections are excluded -- their count is
        already carried by `n_untitled_sections`.
    """
    titles = normalize_titles(sections)
    n_counties = int(sections["fips_code"].nunique())
    titled = sections.assign(_title=titles).loc[titles != ""]
    per_title = titled.groupby("_title")["fips_code"].nunique()
    kept = per_title[per_title / n_counties > TITLE_FLAG_MIN_SHARE]
    return list(kept.sort_values(ascending=False).index)


def title_flag_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Flag which of the common section titles each county's article carries.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.
        vocabulary: Normalized titles to flag, from `flag_vocabulary`.

    Returns:
        DataFrame indexed by `fips_code`, one float64 0.0/1.0 column per title.
        A title appearing twice in one article is still one flag: this asks
        which structures are present, not how many times.
    """
    titles = normalize_titles(sections)
    frame = sections[["fips_code"]].assign(_title=titles)
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")

    flags = pd.DataFrame(index=index)
    for title in vocabulary:
        holders = frame.loc[frame["_title"] == title, "fips_code"].unique()
        flags[f"{TITLE_FLAG_PREFIX}{slugify(title)}"] = index.isin(holders).astype("float64")
    return flags
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_structure_features.py tests/test_source_a_structure.py
git commit -m "feat(source-a): flag the section titles common enough to mean something"
```

---

### Task 3: Bucket character shares

**Files:**
- Modify: `scripts/extract_source_a_structure_features.py`
- Modify: `tests/test_source_a_structure.py`

**Interfaces:**
- Consumes: `normalize_titles` from Task 1; the title patterns already defined elsewhere in the repo.
- Produces:
  - `GEOGRAPHY_TITLE_PATTERN: str`, `GOVERNMENT_TITLE_PATTERN: str`
  - `STRUCTURE_CATEGORIES: tuple[tuple[str, str], ...]` — ordered `(bucket key, regex)` pairs
  - `BUCKET_KEYS: tuple[str, ...]` — the seven category keys plus `"other"`
  - `bucket_share_features(sections: pd.DataFrame) -> pd.DataFrame` — indexed by `fips_code`, columns `share_chars_<bucket>`, rows summing to 1

Background the implementer needs: the five reused patterns live in three modules — `CENSUS_TITLE_PATTERN`, `LIST_TITLE_PATTERN` and `HIGHWAY_TITLE_PATTERN` in `scripts/source_a_text_leakage.py`; `NARRATIVE_TITLE_PATTERN` in `scripts/analyze_source_a_section_scope.py`; `ECONOMY_TITLE_PATTERN` in `scripts/extract_source_a_section_features.py`. Each is fully anchored (`^(?:...)$`), so `Series.str.match` is the right call. Order is load-bearing: `population ranking` appears in both the census and list patterns, and `transportation` reads as highway rather than economy.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_structure.py`:

```python
def test_bucket_shares_sum_to_one_for_every_county(sections_frame: pd.DataFrame) -> None:
    shares = structure.bucket_share_features(sections_frame)

    assert len(shares) == sections_frame["fips_code"].nunique()
    assert np.allclose(shares.sum(axis=1).to_numpy(), 1.0)


def test_census_wins_the_population_ranking_collision() -> None:
    """'population ranking' matches both the census and list patterns."""
    sections = make_sections([("01001", 1, "Population ranking", "x" * 100)])

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_census"] == pytest.approx(1.0)
    assert shares.loc["01001", "share_chars_lists"] == pytest.approx(0.0)


def test_transportation_is_a_highway_not_an_economy_section() -> None:
    sections = make_sections([("01001", 1, "Transportation", "x" * 100)])

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_highways"] == pytest.approx(1.0)
    assert shares.loc["01001", "share_chars_economy"] == pytest.approx(0.0)


def test_highways_are_their_own_bucket_not_folded_into_lists() -> None:
    sections = make_sections(
        [("01001", 1, "Major highways", "x" * 100), ("01001", 2, "Communities", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_highways"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_lists"] == pytest.approx(0.5)


def test_shares_are_weighted_by_characters_not_section_count() -> None:
    sections = make_sections(
        [("01001", 1, "Demographics", "x" * 900), ("01001", 2, "Economy", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_census"] == pytest.approx(0.9)
    assert shares.loc["01001", "share_chars_economy"] == pytest.approx(0.1)


def test_unmatched_titles_fall_to_other() -> None:
    sections = make_sections([("01001", 1, "Volcanology", "x" * 100)])

    assert structure.bucket_share_features(sections).loc["01001", "share_chars_other"] == pytest.approx(1.0)


def test_geography_and_government_are_split_out_of_other() -> None:
    sections = make_sections(
        [("01001", 1, "Geography", "x" * 100), ("01001", 2, "Government", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_geography"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_government"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_other"] == pytest.approx(0.0)


def test_other_is_not_the_largest_bucket_in_the_real_corpus(sections_frame: pd.DataFrame) -> None:
    """Splitting geography and government out of `other` is the point of doing it."""
    corpus_share = structure.bucket_share_features(sections_frame).mean()

    assert corpus_share["share_chars_other"] < corpus_share.drop("share_chars_other").max()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 8 failures with `AttributeError: module 'extract_source_a_structure_features' has no attribute 'bucket_share_features'`.

- [ ] **Step 3: Add the patterns and the share builder**

Add these imports to the import block of `scripts/extract_source_a_structure_features.py`:

```python
from analyze_source_a_section_scope import NARRATIVE_TITLE_PATTERN
from extract_source_a_section_features import ECONOMY_TITLE_PATTERN, SECTIONS_PARQUET_PATH
from source_a_text_leakage import (
    CENSUS_TITLE_PATTERN,
    HIGHWAY_TITLE_PATTERN,
    LIST_TITLE_PATTERN,
)
```

(The `SECTIONS_PARQUET_PATH` import from Task 1 folds into that line — do not import the module twice.)

Then append after `title_flag_features`:

```python
# Physical-setting sections. `national protected area` is deliberately absent:
# `LIST_TITLE_PATTERN` already claims it and runs first, so listing it here
# would be a dead alternative that reads as if it did something.
GEOGRAPHY_TITLE_PATTERN: str = (
    r"^(?:geography|geography and climate|climate|geology|topography|"
    r"terrain|environment|physical geography)$"
)

# Civic sections. Education sits here rather than in its own bucket: it is a
# county-government function in these articles, and splitting it would produce
# a bucket too thin to carry a share.
GOVERNMENT_TITLE_PATTERN: str = (
    r"^(?:government|politics|government and politics|law and government|"
    r"politics and government|education|law enforcement|elections|voting)$"
)

# Ordered: the first pattern to claim a title wins, and the order is
# load-bearing rather than alphabetical. Inherited from
# `analyze_source_a_section_composition.CATEGORIES`, with two deliberate
# differences:
#
# - Highways get their own bucket instead of folding into `lists`. The
#   composition script merges them because it is asking how much of the corpus
#   is content-free for an encoder; this round is asking which *structures* are
#   present, and a highway section is a different structure from a list of towns.
# - `geography` and `government` are split out of the `other` residual. They are
#   its two largest occupants, and leaving them in would put most of the corpus
#   in a bucket named "other".
STRUCTURE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("census", CENSUS_TITLE_PATTERN),
    ("lists", LIST_TITLE_PATTERN),
    ("highways", HIGHWAY_TITLE_PATTERN),
    ("narrative", NARRATIVE_TITLE_PATTERN),
    ("economy", ECONOMY_TITLE_PATTERN),
    ("geography", GEOGRAPHY_TITLE_PATTERN),
    ("government", GOVERNMENT_TITLE_PATTERN),
)

BUCKET_KEYS: tuple[str, ...] = tuple(key for key, _ in STRUCTURE_CATEGORIES) + ("other",)

# Prefix for the character-share columns.
BUCKET_SHARE_PREFIX: str = "share_chars_"


def assign_buckets(sections: pd.DataFrame) -> pd.Series:
    """Label every section with the first thematic bucket that claims it.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Bucket key per row, aligned to `sections.index`. Anything no pattern
        claims is `"other"`.
    """
    titles = normalize_titles(sections)
    buckets = pd.Series("other", index=sections.index)
    claimed = pd.Series(False, index=sections.index)
    for key, pattern in STRUCTURE_CATEGORIES:
        matched = titles.str.match(pattern, na=False) & ~claimed
        buckets[matched] = key
        claimed |= matched
    return buckets


def bucket_share_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Split each county's body characters across the thematic buckets.

    Shares rather than counts, so the block describes the article's composition
    rather than its length -- length is already carried, and carried better, by
    `total_body_chars`.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with one `share_chars_<bucket>` column
        per bucket, each county's row summing to 1.
    """
    chars = sections["section_text"].fillna("").str.len().astype("float64")
    frame = sections[["fips_code"]].assign(_chars=chars, _bucket=assign_buckets(sections))

    per_bucket = (
        frame.pivot_table(index="fips_code", columns="_bucket", values="_chars", aggfunc="sum")
        .reindex(columns=list(BUCKET_KEYS))
        .fillna(0.0)
    )
    totals = per_bucket.sum(axis=1).replace(0.0, np.nan)
    shares = per_bucket.div(totals, axis=0).fillna(0.0)
    shares.columns = [f"{BUCKET_SHARE_PREFIX}{key}" for key in shares.columns]
    shares.index.name = "fips_code"
    return shares
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 25 passed. If `test_other_is_not_the_largest_bucket_in_the_real_corpus` fails, do **not** loosen the assertion — inspect which titles dominate `other` with the snippet below and widen `GEOGRAPHY_TITLE_PATTERN` or `GOVERNMENT_TITLE_PATTERN` to cover the real occupants, then re-run.

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
import pandas as pd
import extract_source_a_structure_features as s
sections = pd.read_parquet('data/source_a_sections.parquet')
titles = s.normalize_titles(sections)
buckets = s.assign_buckets(sections)
chars = sections['section_text'].str.len()
print(chars[buckets == 'other'].groupby(titles[buckets == 'other']).sum().sort_values(ascending=False).head(20))
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_structure_features.py tests/test_source_a_structure.py
git commit -m "feat(source-a): split article characters across eight thematic buckets"
```

---

### Task 4: Assemble the block and write the parquet

**Files:**
- Modify: `scripts/extract_source_a_structure_features.py`
- Modify: `tests/test_source_a_structure.py`
- Creates at runtime: `data/source_a_structure_features.parquet`, `analysis-output/source-a/source_a_structure_feature_stats.json`

**Interfaces:**
- Consumes: `count_features`, `length_features`, `flag_vocabulary`, `title_flag_features`, `bucket_share_features` from Tasks 1–3.
- Produces:
  - `build_structure_features(sections: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` — `(features, vocabulary)` where `features` carries `fips_code` as a column, every other column float64
  - `structure_feature_columns(features: pd.DataFrame) -> list[str]` — every column except `fips_code`
  - `summarize(features: pd.DataFrame, vocabulary: list[str]) -> dict[str, object]`
  - `main() -> None`
  - `STRUCTURE_FEATURES_PATH`, `STRUCTURE_FEATURE_STATS_PATH` (declared in Task 1)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_structure.py`:

```python
def test_every_county_appears_exactly_once(sections_frame: pd.DataFrame) -> None:
    features, _ = structure.build_structure_features(sections_frame)

    assert len(features) == sections_frame["fips_code"].nunique()
    assert features["fips_code"].is_unique
    assert set(features["fips_code"]) == set(sections_frame["fips_code"])


def test_feature_columns_are_numeric_and_finite(sections_frame: pd.DataFrame) -> None:
    features, _ = structure.build_structure_features(sections_frame)
    block = features[structure.structure_feature_columns(features)]

    assert (block.dtypes == "float64").all()
    assert np.isfinite(block.to_numpy()).all(), "imputation must not be papering over NaNs here"


def test_the_block_carries_counts_lengths_flags_and_shares(sections_frame: pd.DataFrame) -> None:
    features, vocabulary = structure.build_structure_features(sections_frame)
    columns = set(features.columns)

    assert {"n_body_sections", "n_id_gaps", "total_body_chars", "section_length_gini"} <= columns
    assert f"{structure.TITLE_FLAG_PREFIX}demographics" in columns
    assert all(f"{structure.BUCKET_SHARE_PREFIX}{key}" in columns for key in structure.BUCKET_KEYS)
    assert len(vocabulary) > 10


def test_no_section_text_survives_into_the_block(sections_frame: pd.DataFrame) -> None:
    """The premise of the round: shape only, never content."""
    features, _ = structure.build_structure_features(sections_frame)

    assert "section_text" not in features.columns
    non_numeric = set(features.dtypes[features.dtypes == "object"].index)
    assert non_numeric == {"fips_code"}, f"unexpected text column(s): {non_numeric}"


def test_structure_parquet_cannot_reach_the_pillar_matrix() -> None:
    """A structural block that leaked into the matrix would predict itself."""
    import pillar_matrix

    source = pathlib.Path(pillar_matrix.__file__).read_text()

    assert "source_a_structure_features" not in source


def test_summary_records_the_vocabulary_it_chose(sections_frame: pd.DataFrame) -> None:
    features, vocabulary = structure.build_structure_features(sections_frame)

    stats = structure.summarize(features, vocabulary)

    assert stats["n_counties"] == len(features)
    assert stats["title_flag_vocabulary"] == vocabulary
    assert stats["title_flag_min_share"] == structure.TITLE_FLAG_MIN_SHARE
    assert stats["stub_char_threshold"] == structure.STUB_CHAR_THRESHOLD
```

`test_structure_parquet_cannot_reach_the_pillar_matrix` needs `pathlib` — add `import pathlib` to the test file's import block, above `import numpy as np`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: failures with `AttributeError: module 'extract_source_a_structure_features' has no attribute 'build_structure_features'`.

- [ ] **Step 3: Add assembly, summary, and the CLI entry point**

Append to `scripts/extract_source_a_structure_features.py`:

```python
def structure_feature_columns(features: pd.DataFrame) -> list[str]:
    """List the scored columns of an assembled structural block.

    Args:
        features: Output of `build_structure_features`.

    Returns:
        Every column except the `fips_code` key.
    """
    return [column for column in features.columns if column != "fips_code"]


def build_structure_features(sections: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the full structural block for every county.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Tuple of (features, vocabulary). `features` has `fips_code` as a column
        and every feature as float64; `vocabulary` is the title set the flags
        were built from, which the caller records so a shifting corpus is
        auditable.
    """
    vocabulary = flag_vocabulary(sections)
    parts = [
        count_features(sections),
        length_features(sections),
        title_flag_features(sections, vocabulary),
        bucket_share_features(sections),
    ]
    features = pd.concat(parts, axis=1).astype("float64")
    features.index.name = "fips_code"
    return features.reset_index(), vocabulary


def summarize(features: pd.DataFrame, vocabulary: list[str]) -> dict[str, object]:
    """Describe the block for the notebook and for later auditing.

    Args:
        features: Output of `build_structure_features`.
        vocabulary: Title set the flags were built from.

    Returns:
        Counts, the chosen vocabulary and thresholds, and per-column summary
        statistics keyed by column name.
    """
    columns = structure_feature_columns(features)
    block = features[columns]
    return {
        "n_counties": int(len(features)),
        "n_features": len(columns),
        "title_flag_vocabulary": vocabulary,
        "title_flag_min_share": TITLE_FLAG_MIN_SHARE,
        "stub_char_threshold": STUB_CHAR_THRESHOLD,
        "bucket_keys": list(BUCKET_KEYS),
        "mean_bucket_share": {
            f"{BUCKET_SHARE_PREFIX}{key}": float(block[f"{BUCKET_SHARE_PREFIX}{key}"].mean())
            for key in BUCKET_KEYS
        },
        "column_summary": {
            column: {
                "mean": float(block[column].mean()),
                "sd": float(block[column].std()),
                "min": float(block[column].min()),
                "max": float(block[column].max()),
            }
            for column in columns
        },
    }


def main() -> None:
    """Build the structural block from the section parquet and write it out."""
    configure_logging()

    try:
        sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    except FileNotFoundError:
        logger.error("Need %s -- run ingest_source_a.py first.", SECTIONS_PARQUET_PATH)
        raise

    features, vocabulary = build_structure_features(sections)
    stats = summarize(features, vocabulary)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(STRUCTURE_FEATURES_PATH, index=False)
    STRUCTURE_FEATURE_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info(
        "wrote %d structural features for %d counties to %s",
        stats["n_features"],
        stats["n_counties"],
        STRUCTURE_FEATURES_PATH,
    )
    logger.info("title flags: %d titles above %.0f%% of counties", len(vocabulary), TITLE_FLAG_MIN_SHARE * 100)
    logger.info("wrote %s", STRUCTURE_FEATURE_STATS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 31 passed.

- [ ] **Step 5: Build the parquet**

```bash
uv run scripts/extract_source_a_structure_features.py
```

Expected: a log line reporting ~60 features across 3,144 counties, and both output files written.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_source_a_structure_features.py tests/test_source_a_structure.py data/source_a_structure_features.parquet analysis-output/source-a/source_a_structure_feature_stats.json
git commit -m "feat(source-a): assemble and persist the structural feature block"
```

---

### Task 5: Score the four arms

**Files:**
- Create: `scripts/analyze_source_a_structure.py`
- Modify: `tests/test_source_a_structure.py`
- Creates at runtime: `outputs/source_a_structure_scores.csv`, `outputs/source_a_structure_by_pillar.csv`, `analysis-output/source-a/source_a_structure_stats.json`

**Interfaces:**
- Consumes:
  - `build_structure_features` is *not* called here — the parquet from Task 4 is read.
  - From `analyze_source_a_representation`: `_baseline_oof_predictions(base_design, y, folds)`, `_residual_oof_predictions(base_design, block, y, folds, n_components)`, `build_non_a_targets(blocks, naics_labels)`.
  - From `analyze_pillar_matrix_signal`: `N_FOLDS`, `RANDOM_SEED`, `Target` (fields `pillar`, `column`, `label`), `build_baseline_design(matrix)`.
  - From `pillar_matrix`: `build_matrix()` returning `(matrix, blocks)`.
  - From `extract_source_a_features`: `VARIANT_COLUMNS`; from `extract_source_a_section_features`: `section_feature_columns()`. Together these are the shipped 29 typed columns: `VARIANT_COLUMNS["extracted_full"]` plus `section_feature_columns()`.
- Produces: `ARMS`, `typed_columns()`, `attach_structure(matrix)`, `score_target(...)`, `run_sweep(...)`, `summarize(...)`, `main()`.

Two facts the implementer should not have to rediscover: `build_matrix()` returns a matrix that does **not** carry `n_body_sections` (it is in `pillar_matrix.SOURCE_A_DIAGNOSTIC_COLUMNS` and is dropped), so merging the structural block in cannot collide on that name. And `_residual_oof_predictions` takes `n_components` for PCA; pass `None` — no arm here reduces.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_structure.py`:

```python
def test_three_scored_arms_are_declared() -> None:
    """The fourth arm is the baseline, which is the comparison rather than a block."""
    import analyze_source_a_structure as scoring

    assert [arm.key for arm in scoring.ARMS] == [
        "structure",
        "typed",
        "typed_plus_structure",
    ]


def test_typed_block_is_the_shipped_29_columns() -> None:
    import analyze_source_a_structure as scoring

    assert len(scoring.typed_columns()) == 29


def test_structure_attaches_to_every_matrix_row() -> None:
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    attached, columns = scoring.attach_structure(matrix)

    assert len(attached) == len(matrix)
    assert attached[columns].notna().all().all()
    assert "n_body_sections" in columns


def test_structure_columns_do_not_collide_with_matrix_columns() -> None:
    """A collision would silently rename a column to `_x`/`_y` and score the wrong block."""
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    _, columns = scoring.attach_structure(matrix)

    assert not set(columns) & set(matrix.columns)


def test_arms_share_folds_and_rows() -> None:
    """Paired comparison is the headline statistic; unpaired folds would void it."""
    import analyze_source_a_structure as scoring

    assert scoring.N_FOLDS == 5
    assert scoring.RANDOM_SEED == 42
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'analyze_source_a_structure'`.

- [ ] **Step 3: Write the scoring module**

Create `scripts/analyze_source_a_structure.py`:

```python
"""Does the shape of a Wikipedia article know anything county size does not?

`extract_source_a_structure_features.py` builds a block from section counts,
section lengths and section titles -- never from section text. This module
scores it against the same 28-target cross-pillar basket, the same folds and
the same protocol as `analyze_source_a_representation.py`, whose pipeline
helpers are imported rather than reimplemented so the numbers stay directly
comparable to that sweep's.

Four arms:

- `baseline`             -- size (`log_population`, `log_agi`, `log_gdp_latest`)
                            plus state fixed effects, and nothing else
- `structure`            -- baseline plus the structural block
- `typed`                -- baseline plus the shipped 29 typed columns
- `typed_plus_structure` -- baseline plus both

Two comparisons carry the round. `structure` against `baseline` asks what the
skeleton knows. `typed_plus_structure` against `typed` asks whether it knows
anything the shipped block does not already have, which is the fusion-relevant
question and the one most likely to come back at zero.

**The baseline is doing real work here, not decoration.** `n_body_sections`
correlates r = 0.550 against log tax returns and was cut from the scored matrix
for exactly that reason. Fitting each arm to the *residuals* of an unpenalized
size-plus-state model is what makes a pure size proxy worth approximately
nothing instead of worth a headline.

**Per-pillar is reported beside the aggregate, never instead of it.** Findings
§14.2b established that 20 of the 28 targets are one QCEW table, so a basket-wide
mean is 71% one pillar and reads as a breadth claim the basket does not support.

Run after `extract_source_a_structure_features.py`.

Outputs: `outputs/source_a_structure_scores.csv`,
`outputs/source_a_structure_by_pillar.csv`,
`analysis-output/source-a/source_a_structure_stats.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import (
    N_FOLDS,
    RANDOM_SEED,
    Target,
    build_baseline_design,
)
from analyze_source_a_representation import (
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from extract_source_a_features import VARIANT_COLUMNS
from extract_source_a_section_features import section_feature_columns
from extract_source_a_structure_features import (
    STRUCTURE_FEATURES_PATH,
    structure_feature_columns,
)
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_structure_scores.csv"
OUTPUT_PILLAR_CSV_PATH: Path = OUTPUTS_DIR / "source_a_structure_by_pillar.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_structure_stats.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arm:
    """One block scored on top of the shared baseline.

    Attributes:
        key: Short identifier used as the result column suffix.
        label: Human-readable description used in reports.
        against: Arm key this one is paired against, or None for a comparison
            made directly against the baseline.
    """

    key: str
    label: str
    against: str | None


ARMS: tuple[Arm, ...] = (
    Arm("structure", "structural block (shape only)", None),
    Arm("typed", "shipped 29 typed columns", None),
    Arm("typed_plus_structure", "typed columns + structural block", "typed"),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def typed_columns() -> list[str]:
    """List the shipped Source A typed columns.

    Assembled here rather than imported as one name because neither extraction
    module knows about the other: the lead features live in
    `extract_source_a_features` and the section features in
    `extract_source_a_section_features`.

    Returns:
        The 29 columns `pillar_matrix` exposes as pillar A.
    """
    return [*VARIANT_COLUMNS["extracted_full"], *section_feature_columns()]


def attach_structure(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge the structural block onto the pillar matrix.

    Args:
        matrix: Feature matrix from `build_matrix`.

    Returns:
        Tuple of (matrix with the structural columns attached, their names).

    Raises:
        FileNotFoundError: If the structural parquet is absent.
        ValueError: If a structural column name already exists in the matrix,
            which would rename both to `_x`/`_y` and score the wrong block.
    """
    try:
        features = pd.read_parquet(STRUCTURE_FEATURES_PATH)
    except FileNotFoundError:
        logger.error(
            "Need %s -- run extract_source_a_structure_features.py first.",
            STRUCTURE_FEATURES_PATH,
        )
        raise

    columns = structure_feature_columns(features)
    collisions = sorted(set(columns) & set(matrix.columns))
    if collisions:
        raise ValueError(f"Structural columns already in the matrix: {collisions}")

    attached = matrix.merge(features, on="fips_code", how="left")
    # Every county in the matrix has a Wikipedia article, so a null here means
    # the two files disagree about the panel rather than that a value is missing.
    missing = int(attached[columns].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} matrix rows have no structural features")
    return attached, columns


def build_arm_blocks(
    matrix: pd.DataFrame, structure_cols: list[str], rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Assemble every arm's feature array for one target's usable rows.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        rows: Boolean mask of rows where the target is observed.

    Returns:
        Mapping of arm key to feature array.
    """
    typed = typed_columns()
    return {
        "structure": matrix.loc[rows, structure_cols].to_numpy(dtype="float64"),
        "typed": matrix.loc[rows, typed].to_numpy(dtype="float64"),
        "typed_plus_structure": matrix.loc[rows, [*typed, *structure_cols]].to_numpy(
            dtype="float64"
        ),
    }


def score_target(
    matrix: pd.DataFrame,
    structure_cols: list[str],
    baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score every arm against one target.

    Every arm sees identical folds and identical rows, which is what makes the
    per-target differences paired and the Wilcoxon test across targets legible.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        baseline: Size-plus-state design from `build_baseline_design`.
        target: The column to predict.

    Returns:
        One record with the baseline R2 and each arm's lift over it.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_baseline = float(r2_score(y, _baseline_oof_predictions(base_design, y, folds)))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
    }

    blocks = build_arm_blocks(matrix, structure_cols, rows)
    for arm in ARMS:
        predictions = _residual_oof_predictions(base_design, blocks[arm.key], y, folds, None)
        record[f"lift_{arm.key}"] = float(r2_score(y, predictions)) - r2_baseline
    return record


def run_sweep(
    matrix: pd.DataFrame, structure_cols: list[str], targets: list[Target]
) -> pd.DataFrame:
    """Score every target against every arm.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        targets: Targets to score.

    Returns:
        Per-target results, sorted by the structural arm's lift.
    """
    baseline = build_baseline_design(matrix)
    records = []
    for target in targets:
        record = score_target(matrix, structure_cols, baseline, target)
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  structure=%+.4f  typed=%+.4f  both=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["lift_structure"],
            record["lift_typed"],
            record["lift_typed_plus_structure"],
        )
    return pd.DataFrame(records).sort_values("lift_structure", ascending=False).reset_index(drop=True)


def summarize_by_pillar(results: pd.DataFrame) -> pd.DataFrame:
    """Mean lift per arm within each target's owning pillar.

    Args:
        results: Output of `run_sweep`.

    Returns:
        One row per pillar with the target count and each arm's mean lift.
    """
    aggregated = results.groupby("pillar").agg(
        n_targets=("column", "count"),
        **{arm.key: (f"lift_{arm.key}", "mean") for arm in ARMS},
    )
    return aggregated.reset_index()


def _paired_test(results: pd.DataFrame, arm: Arm) -> dict[str, object]:
    """Test one arm against its comparison across every target.

    Args:
        results: Output of `run_sweep`.
        arm: The arm to test. `arm.against` names the arm it is paired with;
            None means it is compared against the baseline, where the lift
            column is already the difference.

    Returns:
        Mean lift, mean paired difference, win count and the Wilcoxon
        signed-rank p-value.
    """
    lifts = results[f"lift_{arm.key}"]
    differences = lifts if arm.against is None else lifts - results[f"lift_{arm.against}"]
    statistic, p_value = wilcoxon(differences)
    return {
        "label": arm.label,
        "compared_against": arm.against or "baseline",
        "mean_lift": float(lifts.mean()),
        "median_lift": float(lifts.median()),
        "mean_paired_difference": float(differences.mean()),
        "n_wins": int((differences > 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def summarize(results: pd.DataFrame, n_structure_features: int) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    Args:
        results: Output of `run_sweep`.
        n_structure_features: Width of the structural block.

    Returns:
        Target count, block widths, per-arm paired tests, and the per-pillar
        breakdown as records.
    """
    return {
        "n_targets": int(len(results)),
        "n_structure_features": n_structure_features,
        "n_typed_features": len(typed_columns()),
        "mean_r2_baseline": float(results["r2_baseline"].mean()),
        "arms": {arm.key: _paired_test(results, arm) for arm in ARMS},
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def main() -> None:
    """Attach the structural block, run the four arms, and write the artifacts."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix, structure_cols = attach_structure(matrix)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets: %d structural columns against %d shipped typed columns",
        len(targets),
        len(structure_cols),
        len(typed_columns()),
    )

    results = run_sweep(matrix, structure_cols, targets)
    pillar_results = summarize_by_pillar(results)
    stats = summarize(results, len(structure_cols))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    pillar_results.to_csv(OUTPUT_PILLAR_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    # Per-pillar first: the aggregate below is 71% QCEW and reads as a breadth
    # claim the basket does not support unless its composition is visible.
    logger.info("per-pillar mean lift:")
    for row in pillar_results.itertuples():
        logger.info(
            "  pillar %s  %2d targets  structure %+.5f | typed %+.5f | both %+.5f",
            row.pillar,
            row.n_targets,
            row.structure,
            row.typed,
            row.typed_plus_structure,
        )

    for arm in ARMS:
        test = stats["arms"][arm.key]
        logger.info(
            "%-22s mean lift %+.5f | vs %-8s mean diff %+.5f | wins %2d/%d | p=%.4f",
            arm.key,
            test["mean_lift"],
            test["compared_against"],
            test["mean_paired_difference"],
            test["n_wins"],
            stats["n_targets"],
            test["wilcoxon_p"],
        )

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_structure.py -v
```

Expected: 36 passed.

- [ ] **Step 5: Run the sweep**

```bash
uv run scripts/analyze_source_a_structure.py
```

Expected: 28 target lines, a per-pillar block, three arm summary lines, three files written. Runtime is a few minutes — three arms × 28 targets × 5 folds with a nested penalty search.

- [ ] **Step 6: Sanity-check the result before trusting it**

```bash
uv run python -c "
import json
stats = json.load(open('analysis-output/source-a/source_a_structure_stats.json'))
print('targets:', stats['n_targets'], '| structure cols:', stats['n_structure_features'])
for key, arm in stats['arms'].items():
    print(f\"{key:22} lift {arm['mean_lift']:+.5f}  vs {arm['compared_against']:8} diff {arm['mean_paired_difference']:+.5f}  p={arm['wilcoxon_p']:.4f}\")
"
```

`n_targets` must be 28. If the `structure` arm's mean lift is large and positive, that is a finding to report, not a bug to fix — but confirm first that `test_structure_columns_do_not_collide_with_matrix_columns` still passes, since a silent merge collision is the one defect that would manufacture it.

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_source_a_structure.py tests/test_source_a_structure.py outputs/source_a_structure_scores.csv outputs/source_a_structure_by_pillar.csv analysis-output/source-a/source_a_structure_stats.json
git commit -m "feat(source-a): score article shape against the cross-pillar basket"
```

---

### Task 6: The notebook

**Files:**
- Create: `scripts/build_source_a_structure_notebook.py`
- Creates at runtime: `analysis-output/source-a/source_a_structure_round.ipynb`

**Interfaces:**
- Consumes: `data/source_a_structure_features.parquet`, `data/source_a_sections.parquet`, `analysis-output/source-a/source_a_structure_feature_stats.json`, `analysis-output/source-a/source_a_structure_stats.json`, `outputs/source_a_structure_scores.csv`, `outputs/source_a_structure_by_pillar.csv`.
- Produces: the notebook. Nothing consumes this script's output programmatically.

The builder follows `scripts/build_status_notebook.py`: `nbformat` cells appended by `md()` and `code()` helpers, then written and executed with `nbconvert`. Read that file's first 120 lines before starting — the helper shape and the repo's figure conventions are there. Do not copy its `--for-html` staging path; this notebook is a working artifact, not a presentation deliverable.

- [ ] **Step 1: Write the builder**

Create `scripts/build_source_a_structure_notebook.py`:

```python
"""Generate analysis-output/source-a/source_a_structure_round.ipynb.

The exploratory round on what an article's *shape* knows about a county. Reads
the committed artifacts -- the structural parquet, both stats files and the
scores CSV -- and computes every figure from them. No number is typed in by
hand, which is the standing rule for this project's notebooks: a number that
moves upstream has to move here.

Order is deliberate. The size-proxy audit runs *before* the scoring section,
not after, so the reader forms the expectation ("most of these columns are size
measurements") before seeing the result rather than being talked into it
afterwards.

Matplotlib, not plotly: plotly's mimetype output needs a JupyterLab extension
and renders as blank space without it.

Build and execute:

    uv run scripts/build_source_a_structure_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO: Path = Path(__file__).resolve().parent.parent
OUT: Path = REPO / "analysis-output" / "source-a" / "source_a_structure_round.ipynb"

cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: Cell body; surrounding blank lines are trimmed.
    """
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell.

    Args:
        text: Cell body; surrounding blank lines are trimmed.
    """
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Source A — What the Shape of an Article Knows

**The question:** how much of a county can be read off the *structure* of its
Wikipedia article — how many sections it has, how long they are, which ones are
present — without reading a word of the text?

**The prior:** the answer is "county size". `n_body_sections` was computed
during the section round, correlated r = 0.550 against log tax returns, and cut
from the scored block for exactly that reason. This notebook is built so that
prior can be checked rather than assumed: the size audit runs before the
scoring, and every arm sits on a baseline that already holds three size
measures.

Everything below is computed from committed artifacts. Nothing is typed in.
""")

code("""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path.cwd()
while not (REPO / "data").exists() and REPO != REPO.parent:
    REPO = REPO.parent
DATA, ANALYSIS, OUTPUTS = REPO / "data", REPO / "analysis-output", REPO / "outputs"
SOURCE_A = ANALYSIS / "source-a"

mpl.rcParams.update({"figure.figsize": (11, 5.5), "figure.dpi": 110, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

sections = pd.read_parquet(DATA / "source_a_sections.parquet")
features = pd.read_parquet(DATA / "source_a_structure_features.parquet")
feature_stats = json.loads((SOURCE_A / "source_a_structure_feature_stats.json").read_text())
scoring_stats = json.loads((SOURCE_A / "source_a_structure_stats.json").read_text())
scores = pd.read_csv(OUTPUTS / "source_a_structure_scores.csv")
by_pillar = pd.read_csv(OUTPUTS / "source_a_structure_by_pillar.csv")

print(f"{feature_stats['n_counties']:,} counties x {feature_stats['n_features']} structural features")
print(f"{len(sections):,} sections, {len(feature_stats['title_flag_vocabulary'])} title flags")
""")

md("""
## Part one — what the corpus is shaped like

Three facts set up everything after them: how many sections an article has, how
unevenly its characters are spread across them, and how quickly the title
vocabulary thins out.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].hist(features["n_body_sections"], bins=40, color="#3b6ea5")
axes[0].set_title("Sections per county")
axes[0].set_xlabel("body sections")
axes[0].axvline(features["n_body_sections"].median(), color="#c44", lw=1.5,
                label=f"median {features['n_body_sections'].median():.0f}")
axes[0].legend()

section_chars = sections["section_text"].str.len()
axes[1].hist(np.log10(section_chars.clip(lower=1)), bins=50, color="#3b6ea5")
axes[1].set_title("Section length (log10 characters)")
axes[1].set_xlabel("log10 characters")
axes[1].axvline(np.log10(section_chars.median()), color="#c44", lw=1.5,
                label=f"median {section_chars.median():,.0f} chars")
axes[1].legend()

fig.tight_layout()
plt.show()

print(f"mean section length {section_chars.mean():,.0f} chars, median {section_chars.median():,.0f}, "
      f"max {section_chars.max():,.0f}")
print("The mean sits well above the median: a handful of very long sections carry it.")
""")

code("""
titles = sections["section_title"].fillna("").str.strip().str.lower()
per_title = titles[titles != ""].groupby(titles[titles != ""]).size().sort_values(ascending=False)
county_counts = (sections.assign(t=titles).loc[titles != ""]
                 .groupby("t")["fips_code"].nunique().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(11, 5))
top = county_counts.head(25)[::-1]
ax.barh(top.index, top.to_numpy(), color="#3b6ea5")
ax.axvline(feature_stats["title_flag_min_share"] * feature_stats["n_counties"], color="#c44",
           lw=1.5, ls="--", label=f"flag floor ({feature_stats['title_flag_min_share']:.0%} of counties)")
ax.set_title("Most common section titles, by counties carrying them")
ax.set_xlabel("counties")
ax.legend()
fig.tight_layout()
plt.show()

print(f"{len(county_counts):,} distinct titles; {len(feature_stats['title_flag_vocabulary'])} clear the floor.")
""")

code("""
bucket_means = pd.Series(feature_stats["mean_bucket_share"]).sort_values(ascending=False)
bucket_means.index = [c.replace("share_chars_", "") for c in bucket_means.index]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(bucket_means.index, bucket_means.to_numpy(), color="#3b6ea5")
ax.set_title("Mean share of a county's body characters, by section theme")
ax.set_ylabel("share of characters")
for x, y in zip(bucket_means.index, bucket_means.to_numpy()):
    ax.text(x, y + 0.005, f"{y:.1%}", ha="center", fontsize=9)
fig.tight_layout()
plt.show()
""")

md("""
## Part two — the size-proxy audit

This runs before the scoring, on purpose. Most of these columns are ways of
measuring how much has been written about a county, and how much has been
written about a county is mostly how big it is. The columns that correlate
*weakly* with all three size measures are the ones with anything left to
contribute once the baseline has taken its share.

Three size measures, not one: `n_body_sections` was cut on its correlation with
log tax returns specifically, and a table showing only population would have
understated it.
""")

code("""
import sys
sys.path.insert(0, str(REPO / "scripts"))
from pillar_matrix import SIZE_FEATURES, build_matrix

matrix, _ = build_matrix()
merged = matrix[["fips_code", *SIZE_FEATURES]].merge(features, on="fips_code", how="inner")
structure_cols = [c for c in features.columns if c != "fips_code"]

correlations = pd.DataFrame(
    {size: merged[structure_cols].corrwith(merged[size]) for size in SIZE_FEATURES}
)
correlations["max_abs"] = correlations.abs().max(axis=1)
correlations = correlations.sort_values("max_abs", ascending=False)

print("Most size-dependent structural columns:")
display(correlations.head(12).round(3))
print("\\nLeast size-dependent — the columns with headroom:")
display(correlations.tail(12).round(3))
""")

code("""
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(correlations["max_abs"], bins=30, color="#3b6ea5")
ax.set_title("How much each structural column is a size measurement")
ax.set_xlabel("largest |r| against log_population, log_agi or log_gdp_latest")
ax.set_ylabel("columns")
fig.tight_layout()
plt.show()

n_strong = int((correlations["max_abs"] > 0.4).sum())
print(f"{n_strong} of {len(correlations)} columns correlate above |r| = 0.4 with at least one size measure.")
""")

md("""
## Part three — the four arms

Every arm sits on the same unpenalized baseline of three size measures plus
state fixed effects, and is fitted to that baseline's residuals with a ridge
whose penalty is chosen by nested crossvalidation. Identical folds, identical
rows, five folds, seed 42. A block that knows nothing therefore costs
approximately nothing rather than dragging the controls down.
""")

code("""
arms = pd.DataFrame(scoring_stats["arms"]).T
display(arms[["label", "compared_against", "mean_lift", "mean_paired_difference", "n_wins", "wilcoxon_p"]])

print(f"{scoring_stats['n_targets']} targets | "
      f"{scoring_stats['n_structure_features']} structural columns vs "
      f"{scoring_stats['n_typed_features']} shipped typed columns")
""")

code("""
fig, ax = plt.subplots(figsize=(11, 4.5))
keys = list(scoring_stats["arms"])
values = [scoring_stats["arms"][k]["mean_lift"] for k in keys]
ax.bar(keys, values, color=["#3b6ea5", "#7a9cc6", "#2f5d8a"])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean out-of-fold R² lift over the size-plus-state baseline")
ax.set_ylabel("mean lift")
for x, y in zip(keys, values):
    ax.text(x, y, f"{y:+.4f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=10)
fig.tight_layout()
plt.show()
""")

md("""
### Per-pillar, because the aggregate is a property of the basket

Twenty of the twenty-eight targets are one QCEW table. A basket-wide mean is
therefore 71% one pillar, and reading it as a breadth claim is a mistake this
project has made before.
""")

code("""
display(by_pillar.round(5))

fig, ax = plt.subplots(figsize=(11, 4.5))
width = 0.26
x = np.arange(len(by_pillar))
for offset, key, color in zip((-width, 0, width), keys, ("#3b6ea5", "#7a9cc6", "#2f5d8a")):
    ax.bar(x + offset, by_pillar[key], width, label=key, color=color)
ax.set_xticks(x, [f"{p}\\n({n} targets)" for p, n in zip(by_pillar["pillar"], by_pillar["n_targets"])])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean lift by owning pillar")
ax.legend()
fig.tight_layout()
plt.show()
""")

code("""
best = scores.head(10)[["pillar", "label", "n", "r2_baseline", "lift_structure", "lift_typed",
                        "lift_typed_plus_structure"]]
print("Targets where article shape helps most:")
display(best.round(4))

worst = scores.tail(5)[["pillar", "label", "lift_structure", "lift_typed"]]
print("\\nAnd where it hurts most:")
display(worst.round(4))
""")

md("""
## What this round does and does not settle

- It does not propose shipping these columns. If `typed_plus_structure` beats
  `typed`, that is an argument for a follow-up round, not a change to
  `pillar_matrix`.
- It reads no section text. Any lexicon question belongs to the section-scope
  round, which already exists.
- It does not revisit the `n_body_sections` cut on its own authority. That
  decision stands unless the numbers above argue otherwise.
""")


def main() -> None:
    """Write the notebook and execute it in place."""
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(OUT))
    print(f"wrote {OUT.relative_to(REPO)} ({len(cells)} cells)")

    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace", str(OUT),
        ],
        check=True,
    )
    print(f"executed {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build and execute the notebook**

```bash
uv run scripts/build_source_a_structure_notebook.py
```

Expected: `wrote analysis-output/source-a/source_a_structure_round.ipynb (N cells)` followed by nbconvert's execution log and no traceback. An execution error here is a real failure — fix the offending cell in the builder and re-run; never hand-edit the `.ipynb`.

- [ ] **Step 3: Confirm every cell actually produced output**

```bash
uv run python -c "
import json
nb = json.load(open('analysis-output/source-a/source_a_structure_round.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
empty = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print(f'{len(code_cells)} code cells, {len(empty)} with no output: {empty}')
"
```

Expected: 0 cells with no output. A silent empty cell is the failure mode this check exists for.

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: everything passes, including the pre-existing suites — nothing in this round modifies a shipped artifact, so a failure elsewhere means something was touched that should not have been.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_source_a_structure_notebook.py analysis-output/source-a/source_a_structure_round.ipynb
git commit -m "docs(source-a): notebook for the article-shape round"
```

---

## Verification

After Task 6, the round is complete when all of these hold:

```bash
uv run pytest tests/ -v
uv run scripts/extract_source_a_structure_features.py
uv run scripts/analyze_source_a_structure.py
uv run scripts/build_source_a_structure_notebook.py
```

- Full test suite green.
- `data/source_a_structure_features.parquet` has 3,144 rows and no nulls.
- `analysis-output/source-a/source_a_structure_stats.json` reports `n_targets = 28`.
- The notebook executes end to end with output in every code cell.
- `grep -r source_a_structure_features scripts/pillar_matrix.py` returns nothing.
