# Source A Shape Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out how much can be extracted from the shape of a county's Wikipedia article, and report every number in three framings at once so none of them can be read in a framing its reader cannot see.

**Architecture:** A new extraction module adds four feature families the previous round never built, into its own parquet — round one's committed artifact is cited by findings §23 and is never touched. A new analysis module scores five arms under two learners against three framings, reusing round one's baselines and null control. A separate diagnostic inverts the question §23 left open: predict county size *from* the shape block.

**Tech Stack:** Python 3.12, `uv`, pandas, numpy, scikit-learn (`RidgeCV`, `HistGradientBoostingRegressor`), scipy, matplotlib, nbformat, pytest.

**Design:** `docs/superpowers/specs/2026-08-26-source-a-shape-profile-design.md`

## Global Constraints

- Python 3.12; every command runs under `uv run` from the repo root.
- **`data/source_a_structure_features.parquet` is never modified.** Findings §23 cites it. The new families go in a separate parquet.
- The extraction module reads `section_title` and the *characters* of `section_text`. Character-class densities are the boundary and it is not crossed further — no lexicon, no word matching, no meaning.
- Google-style docstrings with `Args:`/`Returns:` on every public function; type annotations on every signature; `from __future__ import annotations` at the top of each module.
- Constants are module-level, uppercase, annotated, and carry a comment explaining any number that could have been chosen differently.
- **No number is typed into the notebook by hand.** Every figure and printed value reads an artifact.
- Matplotlib only, never plotly.
- **Every arm × learner row carries all three framings** — `r2_alone`, `lift`, `lift_flexbase`. A partial row is a defect: round one's failure was a correct number quoted in an invisible framing.
- Protocol constants come from the existing modules and do not drift: `N_FOLDS = 5`, `RANDOM_SEED = 42`, one `KFold` per target shared by every arm, every learner, and both baselines.
- The ridge learner must delegate to the *imported* `_residual_oof_predictions` / `_alone_oof_r2`, not to a local reimplementation, so the ridge numbers stay bit-comparable to §23.
- Work happens on a new branch off `source-a-structure-features` (that branch is unmerged and this round builds on it).

## What already exists, and what it is called

`scripts/extract_source_a_structure_features.py` exports:
`normalize_titles(sections) -> pd.Series` (stripped, lowercased; untitled → `""`), `slugify(title) -> str`, `flag_vocabulary(sections) -> list[str]` (titles held by >5% of counties, most common first), `assign_buckets(sections) -> pd.Series` (bucket key per row, first pattern wins), `BUCKET_KEYS: tuple[str, ...]` (census, lists, highways, narrative, economy, geography, government, other), `SECTIONS_PARQUET_PATH`, `STRUCTURE_FEATURES_PATH`, `structure_feature_columns(features) -> list[str]`, `TITLE_FLAG_PREFIX = "has_section_"`, `BUCKET_SHARE_PREFIX = "share_chars_"`.

`scripts/analyze_source_a_structure.py` exports:
`Arm` (frozen dataclass: `key`, `label`, `against`), `typed_columns() -> list[str]` (the shipped 29), `size_nonlinear_block(frame) -> pd.DataFrame` (nine information-free columns), `size_curvature_directions(frame) -> pd.DataFrame`, `build_flexible_baseline_design(matrix, baseline) -> pd.DataFrame`, `attach_structure(matrix) -> tuple[pd.DataFrame, list[str]]`, `FLEXIBLE_SUFFIX = "_flexbase"`, `NULL_ARM_KEY = "size_nonlinear"`.

`scripts/analyze_source_a_representation.py` exports the protocol helpers:
`_baseline_oof_predictions(base_design, y, folds)`, `_residual_oof_predictions(base_design, block, y, folds, n_components)`, `_alone_oof_r2(block, y, folds, n_components)`, `build_non_a_targets(blocks, naics_labels)`.

`scripts/analyze_pillar_matrix_signal.py` exports `N_FOLDS`, `RANDOM_SEED`, `Target`, `build_baseline_design(matrix)`.
`scripts/pillar_matrix.py` exports `build_matrix()`, `SIZE_FEATURES`, `DATA_DIR`.

`data/source_a_sections.parquet`: 64,588 rows × 3,144 counties, columns `fips_code`, `county_name`, `section_id`, `section_title`, `section_text`. `section_id` is Parsoid numbering and is not contiguous.

---

### Task 1: Section order and position

**Files:**
- Create: `scripts/extract_source_a_shape_profile.py`
- Create: `tests/test_source_a_shape_profile.py`

**Interfaces:**
- Consumes: `normalize_titles`, `flag_vocabulary`, `slugify`, `assign_buckets` from `extract_source_a_structure_features`.
- Produces:
  - `POSITION_ABSENT: float = -1.0`, `POSITION_PREFIX: str = "pos_"`
  - `ordered_sections(sections: pd.DataFrame) -> pd.DataFrame` — `sections` plus a `_position` column, normalized to `[0, 1]` within each county
  - `position_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame` — indexed by `fips_code`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_source_a_shape_profile.py`:

```python
"""Shape-profile features: where sections sit, how standard they are, what they look like."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import extract_source_a_shape_profile as shape


def make_sections(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Build a section frame from (fips_code, section_id, title, text) tuples."""
    return pd.DataFrame(
        rows, columns=["fips_code", "section_id", "section_title", "section_text"]
    ).assign(county_name="Test County")


def test_position_runs_from_zero_to_one_in_section_id_order() -> None:
    sections = make_sections(
        [("01001", 5, "C", "x"), ("01001", 1, "A", "x"), ("01001", 3, "B", "x")]
    )

    ordered = shape.ordered_sections(sections)
    by_title = dict(zip(ordered["section_title"], ordered["_position"]))

    assert by_title["A"] == pytest.approx(0.0)
    assert by_title["B"] == pytest.approx(0.5)
    assert by_title["C"] == pytest.approx(1.0)


def test_a_single_section_county_sits_at_zero_not_nan() -> None:
    """One section is both first and last; 0.0 is the defensible reading."""
    sections = make_sections([("01001", 1, "A", "x")])

    assert shape.ordered_sections(sections)["_position"].iloc[0] == pytest.approx(0.0)


def test_absent_sections_get_the_sentinel_not_zero() -> None:
    """Position 0.0 means 'first', which is the opposite of absent."""
    sections = make_sections([("01001", 1, "Geography", "x"), ("01002", 1, "Economy", "x")])

    positions = shape.position_features(sections, ["geography", "economy"])

    assert positions.loc["01001", "pos_geography"] == pytest.approx(0.0)
    assert positions.loc["01001", "pos_economy"] == shape.POSITION_ABSENT
    assert shape.POSITION_ABSENT < 0.0


def test_position_of_the_longest_section_is_found() -> None:
    sections = make_sections(
        [("01001", 1, "A", "x" * 10), ("01001", 2, "B", "x" * 900), ("01001", 3, "C", "x" * 10)]
    )

    positions = shape.position_features(sections, [])

    assert positions.loc["01001", "pos_longest_section"] == pytest.approx(0.5)


def test_history_before_economy_reads_the_actual_order() -> None:
    early = make_sections([("01001", 1, "History", "x"), ("01001", 2, "Economy", "x")])
    late = make_sections([("01002", 1, "Economy", "x"), ("01002", 2, "History", "x")])

    assert shape.position_features(early, []).loc["01001", "history_before_economy"] == 1.0
    assert shape.position_features(late, []).loc["01002", "history_before_economy"] == 0.0


def test_history_before_economy_is_zero_when_either_is_absent() -> None:
    sections = make_sections([("01001", 1, "History", "x"), ("01001", 2, "Geography", "x")])

    assert shape.position_features(sections, []).loc["01001", "history_before_economy"] == 0.0


def test_position_spread_is_zero_for_one_flagged_title() -> None:
    sections = make_sections([("01001", 1, "Geography", "x"), ("01001", 2, "Unflagged", "x")])

    positions = shape.position_features(sections, ["geography"])

    assert positions.loc["01001", "position_spread"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'extract_source_a_shape_profile'`.

- [ ] **Step 3: Write the module with the position family**

Create `scripts/extract_source_a_shape_profile.py`:

```python
"""Shape profile: where an article's sections sit, and what they look like.

Round one (`extract_source_a_structure_features.py`) asked how many sections a
county's article has, how long they are, and which titles are present. It found
+0.00269 mean lift, and the branch review then showed roughly three quarters of
that was curvature in county size that a linear-in-logs control could not
absorb (`analysis-output/source-a/source-a-findings.md` §23).

This module adds the four families round one never built:

- **order and position** -- where each section sits, which is editorial priority
  and the one signal here with no obvious reading as a volume measure
- **template conformity** -- how far the article departs from the house skeleton
  county articles follow, which is editorial attention rather than county size
- **surface statistics** -- character-class densities; a census table rendered as
  prose is roughly 30% digits, and that is a fact about shape, not content
- **length curve** -- the sorted section-length curve beyond round one's Gini

It reads section titles and the *characters* of section text. It never reads
meaning: no lexicon, no word matching. That is the boundary and it is not
crossed.

**This module does not touch `data/source_a_structure_features.parquet`.** §23
cites that artifact; mutating it would silently invalidate a committed finding.
The new families go to `data/source_a_shape_profile.parquet` and
`analyze_source_a_shape_profile.py` joins the two.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from extract_source_a_structure_features import (
    SECTIONS_PARQUET_PATH,
    assign_buckets,
    flag_vocabulary,
    normalize_titles,
    slugify,
)
from pillar_matrix import DATA_DIR

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

SHAPE_PROFILE_PATH: Path = DATA_DIR / "source_a_shape_profile.parquet"
SHAPE_PROFILE_STATS_PATH: Path = ANALYSIS_DIR / "source_a_shape_profile_stats.json"

# Sentinel for "this county has no such section". Deliberately outside the
# [0, 1] range a real position occupies, and deliberately *not* 0.0: position
# 0.0 means "this section comes first", which is the opposite of absent. A tree
# can split the sentinel off cleanly; a linear model reads it as one step below
# the earliest possible position, which is the correct direction.
POSITION_ABSENT: float = -1.0

POSITION_PREFIX: str = "pos_"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def ordered_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Attach each section's normalized position within its county's article.

    Order is `section_id` order, which is the order the sections appear in the
    rendered article. Position is normalized to `[0, 1]` so a 40-section article
    and a 6-section article are comparable: what matters is whether the economy
    section sits a fifth of the way down or four fifths.

    A county with one section gets position 0.0 -- it is both first and last, and
    0.0 is the reading that keeps the sentinel at `POSITION_ABSENT` unambiguous.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        `sections` with `_position`, `_title` and `_chars` columns added, sorted
        by county and `section_id`.
    """
    frame = sections.assign(
        _title=normalize_titles(sections),
        _chars=sections["section_text"].fillna("").str.len().astype("float64"),
    ).sort_values(["fips_code", "section_id"])

    rank = frame.groupby("fips_code").cumcount().astype("float64")
    size = frame.groupby("fips_code")["section_id"].transform("size").astype("float64")
    # size - 1 is the number of gaps between sections; a one-section county has
    # none, and dividing by zero there would produce NaN rather than 0.0.
    span = (size - 1.0).replace(0.0, np.nan)
    frame["_position"] = (rank / span).fillna(0.0)
    return frame


def position_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Where each common section sits in the article.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.
        vocabulary: Titles to locate, from `flag_vocabulary`. Passed in rather
            than re-derived so `pos_<title>` and round one's `has_section_<title>`
            can never cover different title sets.

    Returns:
        DataFrame indexed by `fips_code` with one `pos_<title>` column per
        vocabulary entry plus `pos_longest_section`, `pos_first_economy`,
        `pos_first_census`, `pos_first_narrative`, `history_before_economy` and
        `position_spread`.
    """
    frame = ordered_sections(sections)
    frame["_bucket"] = assign_buckets(frame)
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    features = pd.DataFrame(index=index)

    for title in vocabulary:
        matched = frame.loc[frame["_title"] == title].groupby("fips_code")["_position"].min()
        features[f"{POSITION_PREFIX}{slugify(title)}"] = matched.reindex(index).fillna(
            POSITION_ABSENT
        )

    longest = frame.loc[frame.groupby("fips_code")["_chars"].idxmax()]
    features["pos_longest_section"] = longest.set_index("fips_code")["_position"].reindex(index)

    first_by_bucket: dict[str, pd.Series] = {}
    for bucket in ("economy", "census", "narrative"):
        first = frame.loc[frame["_bucket"] == bucket].groupby("fips_code")["_position"].min()
        first_by_bucket[bucket] = first.reindex(index)
        features[f"pos_first_{bucket}"] = first.reindex(index).fillna(POSITION_ABSENT)

    # Both present, and the narrative one earlier. A county missing either gets
    # 0.0: "no, history does not come first here" is the honest reading of an
    # article that has no history section.
    both = first_by_bucket["narrative"].notna() & first_by_bucket["economy"].notna()
    earlier = first_by_bucket["narrative"] < first_by_bucket["economy"]
    features["history_before_economy"] = (both & earlier).astype("float64")

    flagged = frame.loc[frame["_title"].isin(vocabulary)]
    features["position_spread"] = (
        flagged.groupby("fips_code")["_position"].std().reindex(index).fillna(0.0)
    )
    return features
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_shape_profile.py tests/test_source_a_shape_profile.py
git commit -m "feat(source-a): read where an article's sections sit, not just that they exist"
```

---

### Task 2: Template conformity

**Files:**
- Modify: `scripts/extract_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`

**Interfaces:**
- Consumes: `normalize_titles` from `extract_source_a_structure_features`; `ordered_sections` from Task 1.
- Produces:
  - `MODAL_TITLE_MIN_SHARE: float = 0.5`, `UNUSUAL_TITLE_MAX_SHARE: float = 0.01`
  - `title_county_shares(sections: pd.DataFrame) -> pd.Series` — share of counties holding each title, indexed by normalized title
  - `modal_title_set(sections: pd.DataFrame) -> list[str]`
  - `template_features(sections: pd.DataFrame) -> pd.DataFrame` — indexed by `fips_code`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def make_template_corpus() -> pd.DataFrame:
    """Four counties. 'geography' and 'history' are modal; 'oddity' is not."""
    rows = []
    for i in range(1, 5):
        rows.append((f"0100{i}", 1, "Geography", "x"))
        rows.append((f"0100{i}", 2, "History", "x"))
    rows.append(("01004", 3, "Oddity", "x"))
    return make_sections(rows)


def test_modal_set_is_the_titles_most_counties_hold() -> None:
    modal = shape.modal_title_set(make_template_corpus())

    assert set(modal) == {"geography", "history"}


def test_modal_set_is_derived_not_hardcoded() -> None:
    """A corpus with a different skeleton produces a different modal set."""
    rows = [(f"0100{i}", 1, "Volcanology", "x") for i in range(1, 5)]

    assert shape.modal_title_set(make_sections(rows)) == ["volcanology"]


def test_jaccard_is_one_for_a_county_holding_exactly_the_modal_set() -> None:
    features = shape.template_features(make_template_corpus())

    assert features.loc["01001", "template_jaccard"] == pytest.approx(1.0)


def test_jaccard_falls_when_a_county_adds_an_unusual_section() -> None:
    features = shape.template_features(make_template_corpus())

    # 01004 holds {geography, history, oddity} against a modal {geography, history}
    assert features.loc["01004", "template_jaccard"] == pytest.approx(2 / 3)
    assert features.loc["01004", "template_jaccard"] < features.loc["01001", "template_jaccard"]


def test_missing_core_sections_are_counted() -> None:
    corpus = make_template_corpus()
    thin = pd.concat([corpus, make_sections([("01009", 1, "Geography", "x")])])

    features = shape.template_features(thin)

    assert features.loc["01009", "n_core_missing"] == 1.0  # has geography, lacks history
    assert features.loc["01001", "n_core_missing"] == 0.0


def test_unusual_sections_are_the_rare_ones() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 201)]
    rows.append(("00007", 2, "One Off", "x"))  # 1 of 200 counties = 0.5%

    features = shape.template_features(make_sections(rows))

    assert features.loc["00007", "n_unusual_sections"] == 1.0
    assert features.loc["00001", "n_unusual_sections"] == 0.0


def test_title_rarity_is_higher_for_a_county_with_rare_titles() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 201)]
    rows.append(("00007", 2, "One Off", "x"))

    features = shape.template_features(make_sections(rows))

    assert features.loc["00007", "mean_title_rarity"] > features.loc["00001", "mean_title_rarity"]


def test_title_word_count_is_averaged_over_the_county() -> None:
    sections = make_sections(
        [("01001", 1, "Law and government", "x"), ("01001", 2, "Economy", "x")]
    )

    assert shape.template_features(sections).loc["01001", "n_title_words"] == pytest.approx(2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 8 failures with `AttributeError: module 'extract_source_a_shape_profile' has no attribute 'modal_title_set'` (and `template_features`).

- [ ] **Step 3: Add the template family**

Append to `scripts/extract_source_a_shape_profile.py`:

```python
# A title is part of the house skeleton if more than half of counties carry it.
# Half is the natural cut for "modal": below it, the set stops describing what a
# typical county article looks like and starts describing a large minority.
MODAL_TITLE_MIN_SHARE: float = 0.5

# A title held by under 1% of counties is unusual -- roughly 31 counties. Set
# well below round one's 5% flag floor on purpose: this measures the tail that
# floor excludes, so the two cuts describe different populations rather than
# two views of the same one.
UNUSUAL_TITLE_MAX_SHARE: float = 0.01


def title_county_shares(sections: pd.DataFrame) -> pd.Series:
    """Share of counties holding each section title.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Share per normalized title, indexed by title. Untitled sections are
        excluded -- round one's `n_untitled_sections` already counts them.
    """
    titles = normalize_titles(sections)
    n_counties = int(sections["fips_code"].nunique())
    titled = sections.assign(_title=titles).loc[titles != ""]
    return titled.groupby("_title")["fips_code"].nunique() / n_counties


def modal_title_set(sections: pd.DataFrame) -> list[str]:
    """The house skeleton: titles more than half of counties carry.

    Computed from the corpus rather than hardcoded, and written to the stats
    file, so a shifting skeleton is auditable rather than silent.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized titles above `MODAL_TITLE_MIN_SHARE`, most common first.
    """
    shares = title_county_shares(sections)
    kept = shares[shares > MODAL_TITLE_MIN_SHARE]
    return list(kept.sort_values(ascending=False).index)


def template_features(sections: pd.DataFrame) -> pd.DataFrame:
    """How far the county's article departs from the house skeleton.

    Deviation from the template is editorial attention, which is a different
    quantity from county size: a small county someone cared about carries
    sections a large county's boilerplate article does not.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `template_jaccard`,
        `n_core_missing`, `n_unusual_sections`, `share_unusual_sections`,
        `mean_title_rarity` and `n_title_words`.
    """
    titles = normalize_titles(sections)
    shares = title_county_shares(sections)
    modal = set(modal_title_set(sections))
    unusual = set(shares[shares < UNUSUAL_TITLE_MAX_SHARE].index)

    frame = sections.assign(_title=titles).loc[titles != ""]
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    grouped = frame.groupby("fips_code")

    held = grouped["_title"].apply(set).reindex(index)
    held = held.where(held.notna(), other=pd.Series([set()] * len(index), index=index))

    features = pd.DataFrame(index=index)
    features["template_jaccard"] = [
        len(county & modal) / len(county | modal) if county | modal else 1.0 for county in held
    ]
    features["n_core_missing"] = [float(len(modal - county)) for county in held]
    features["n_unusual_sections"] = (
        frame.assign(_unusual=frame["_title"].isin(unusual).astype("float64"))
        .groupby("fips_code")["_unusual"]
        .sum()
        .reindex(index)
        .fillna(0.0)
    )
    # Denominator is titled sections, matching the numerator: an untitled section
    # cannot be unusual because it has no title to be rare.
    n_titled = grouped.size().reindex(index).fillna(0.0)
    features["share_unusual_sections"] = (
        features["n_unusual_sections"] / n_titled.replace(0.0, np.nan)
    ).fillna(0.0)
    features["mean_title_rarity"] = (
        frame.assign(_rarity=1.0 - frame["_title"].map(shares).astype("float64"))
        .groupby("fips_code")["_rarity"]
        .mean()
        .reindex(index)
        .fillna(0.0)
    )
    features["n_title_words"] = (
        frame.assign(_words=frame["_title"].str.split().str.len().astype("float64"))
        .groupby("fips_code")["_words"]
        .mean()
        .reindex(index)
        .fillna(0.0)
    )
    return features
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_shape_profile.py tests/test_source_a_shape_profile.py
git commit -m "feat(source-a): measure how far an article departs from the county template"
```

---

### Task 3: Surface statistics and length curve

**Files:**
- Modify: `scripts/extract_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`

**Interfaces:**
- Consumes: `assign_buckets`, `BUCKET_KEYS` from `extract_source_a_structure_features`; `ordered_sections` from Task 1.
- Produces:
  - `DENSITY_BUCKETS: tuple[str, ...] = ("census", "lists", "narrative", "geography")`
  - `surface_features(sections: pd.DataFrame) -> pd.DataFrame`
  - `length_curve_features(sections: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def test_digit_density_is_zero_for_letters_and_one_for_digits() -> None:
    letters = make_sections([("01001", 1, "A", "abcdef")])
    digits = make_sections([("01002", 1, "A", "123456")])

    assert shape.surface_features(letters).loc["01001", "digit_density"] == pytest.approx(0.0)
    assert shape.surface_features(digits).loc["01002", "digit_density"] == pytest.approx(1.0)


def test_capital_ratio_counts_letters_only() -> None:
    """Digits are neither upper nor lower, so they must not dilute the ratio."""
    sections = make_sections([("01001", 1, "A", "AB12ab")])

    assert shape.surface_features(sections).loc["01001", "capital_ratio"] == pytest.approx(0.5)


def test_mean_word_length_ignores_whitespace() -> None:
    sections = make_sections([("01001", 1, "A", "aa bbbb cc")])

    assert shape.surface_features(sections).loc["01001", "mean_word_length"] == pytest.approx(
        8 / 3
    )


def test_bucket_density_is_computed_only_where_that_bucket_has_characters() -> None:
    """A density over zero characters is not a number; it must not be invented."""
    sections = make_sections([("01001", 1, "Geography", "abc123")])

    surface = shape.surface_features(sections)

    assert surface.loc["01001", "digit_density_geography"] == pytest.approx(0.5)
    assert surface.loc["01001", "digit_density_census"] == pytest.approx(0.0)


def test_a_county_with_no_characters_gets_zero_not_nan() -> None:
    sections = make_sections([("01001", 1, "A", "")])

    surface = shape.surface_features(sections)

    assert np.isfinite(surface.loc["01001"].to_numpy()).all()


def test_top3_share_is_the_three_longest_sections() -> None:
    sections = make_sections(
        [
            ("01001", 1, "A", "x" * 400),
            ("01001", 2, "B", "x" * 300),
            ("01001", 3, "C", "x" * 200),
            ("01001", 4, "D", "x" * 100),
        ]
    )

    curve = shape.length_curve_features(sections)

    assert curve.loc["01001", "top3_length_share"] == pytest.approx(0.9)


def test_top3_share_is_one_when_a_county_has_three_or_fewer_sections() -> None:
    sections = make_sections([("01001", 1, "A", "x" * 10), ("01001", 2, "B", "x" * 20)])

    assert shape.length_curve_features(sections).loc["01001", "top3_length_share"] == pytest.approx(
        1.0
    )


def test_decay_slope_is_negative_when_lengths_fall_off() -> None:
    steep = make_sections(
        [("01001", i, f"S{i}", "x" * n) for i, n in enumerate([1000, 100, 10, 5], start=1)]
    )
    flat = make_sections([("01002", i, f"S{i}", "x" * 100) for i in range(1, 5)])

    curve = shape.length_curve_features(pd.concat([steep, flat]))

    assert curve.loc["01001", "length_decay_slope"] < curve.loc["01002", "length_decay_slope"]
    assert curve.loc["01002", "length_decay_slope"] == pytest.approx(0.0)


def test_absolute_bucket_characters_are_reported_for_every_bucket() -> None:
    sections = make_sections([("01001", 1, "Geography", "x" * 250)])

    curve = shape.length_curve_features(sections)

    assert curve.loc["01001", "chars_geography"] == pytest.approx(250.0)
    assert curve.loc["01001", "chars_census"] == pytest.approx(0.0)


def test_recomputed_top_one_share_matches_round_one(sections_frame: pd.DataFrame) -> None:
    """Cross-module consistency: the same quantity under two names must agree.

    Round one ships `share_in_largest_section`; this module deliberately does not
    ship a duplicate of it. Asserting the equality here catches a drift in either
    module without putting the column in the block twice.
    """
    import extract_source_a_structure_features as structure

    round_one = structure.length_features(sections_frame)["share_in_largest_section"]

    chars = sections_frame["section_text"].fillna("").str.len().astype("float64")
    grouped = sections_frame.assign(_chars=chars).groupby("fips_code")["_chars"]
    recomputed = (grouped.max() / grouped.sum().replace(0.0, np.nan)).fillna(0.0)

    assert np.allclose(recomputed.to_numpy(), round_one.reindex(recomputed.index).to_numpy())
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 9 failures with `AttributeError: module 'extract_source_a_shape_profile' has no attribute 'surface_features'` (and `length_curve_features`). `test_recomputed_top_one_share_matches_round_one` passes immediately — it exercises only round-one code and is a regression guard, not a driver.

- [ ] **Step 3: Add both families**

Add `BUCKET_KEYS` to the import from `extract_source_a_structure_features`, then append:

```python
# Buckets that get their own character-class densities. The four largest only:
# `economy`, `government`, `highways` and `other` are absent or near-empty for a
# large share of counties, and a density over zero characters is not a number --
# it is a zero standing where "no data" belongs, which a model reads as a
# measurement.
DENSITY_BUCKETS: tuple[str, ...] = ("census", "lists", "narrative", "geography")


def _class_counts(text: pd.Series) -> pd.DataFrame:
    """Count character classes per section.

    Args:
        text: Section text.

    Returns:
        DataFrame with `chars`, `digits`, `letters`, `uppers`, `punct` and
        `words` columns, aligned to `text.index`.
    """
    filled = text.fillna("")
    return pd.DataFrame(
        {
            "chars": filled.str.len().astype("float64"),
            "digits": filled.str.count(r"\d").astype("float64"),
            "letters": filled.str.count(r"[A-Za-z]").astype("float64"),
            "uppers": filled.str.count(r"[A-Z]").astype("float64"),
            "punct": filled.str.count(r"[^\w\s]").astype("float64"),
            "words": filled.str.split().str.len().fillna(0.0).astype("float64"),
        },
        index=filled.index,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, treating a zero denominator as zero rather than as NaN or an error.

    Args:
        numerator: Top of the ratio.
        denominator: Bottom of the ratio.

    Returns:
        The ratio, with zero-denominator rows set to 0.0.
    """
    return (numerator / denominator.replace(0.0, np.nan)).fillna(0.0)


def surface_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Character-class densities, overall and for the four largest buckets.

    These read characters and never meaning. The signal they are after is
    documented: a census table rendered as prose is roughly 30% digits, a list of
    place names is short-worded and heavily capitalized, and narrative prose is
    neither -- all three are facts about an article's shape.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `digit_density`, `punct_density`,
        `capital_ratio`, `mean_word_length`, `numeral_to_letter`, and
        `digit_density_<bucket>` for each of `DENSITY_BUCKETS`.
    """
    counts = _class_counts(sections["section_text"])
    frame = counts.assign(fips_code=sections["fips_code"].to_numpy(), _bucket=assign_buckets(sections))
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    totals = frame.groupby("fips_code")[["chars", "word_chars", "digits", "letters", "uppers", "punct", "words"]].sum().reindex(index).fillna(0.0)  # `word_chars` per the correction below

    features = pd.DataFrame(index=index)
    features["digit_density"] = _safe_ratio(totals["digits"], totals["chars"])
    features["punct_density"] = _safe_ratio(totals["punct"], totals["chars"])
    features["capital_ratio"] = _safe_ratio(totals["uppers"], totals["letters"])
    # POST-HOC CORRECTION (2026-08-28, branch review finding I5): this plan's
    # formula was `chars / words`, which counts whitespace and punctuation into
    # the numerator and so reports "characters per word including the spaces
    # between them". The shipped code corrected it to `word_chars / words`,
    # where `word_chars` excludes whitespace, and the test below pins the
    # corrected value (8 / 3 for "aa bbbb cc", not 10 / 3). The code is right;
    # this line is left in place with the correction noted rather than
    # rewritten, so the record shows what changed and why.
    features["mean_word_length"] = _safe_ratio(totals["word_chars"], totals["words"])
    features["numeral_to_letter"] = _safe_ratio(totals["digits"], totals["letters"])

    for bucket in DENSITY_BUCKETS:
        within = frame.loc[frame["_bucket"] == bucket].groupby("fips_code")[["digits", "chars"]].sum().reindex(index).fillna(0.0)
        features[f"digit_density_{bucket}"] = _safe_ratio(within["digits"], within["chars"])
    return features


def length_curve_features(sections: pd.DataFrame) -> pd.DataFrame:
    """The shape of the sorted section-length curve, and absolute bucket lengths.

    Round one shipped a Gini over section lengths and the share in the largest
    section. Both are scale-free, so neither can express "this county has a long
    economy section" -- only "a large fraction of this county's article is its
    economy section". The absolute `chars_<bucket>` columns close that gap.

    `chars_<bucket>` is `share_chars_<bucket>` times `total_body_chars`, both of
    which round one already ships, so it is derivable rather than new. It earns
    its place for the ridge learner, which cannot form products, and is redundant
    for the boosting learner, which can.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `top3_length_share`,
        `length_decay_slope` and `chars_<bucket>` for every bucket.
    """
    frame = sections.assign(
        _chars=sections["section_text"].fillna("").str.len().astype("float64"),
        _bucket=assign_buckets(sections),
    )
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    grouped = frame.groupby("fips_code")["_chars"]

    totals = grouped.sum().reindex(index).fillna(0.0)
    top3 = grouped.apply(lambda s: float(np.sort(s.to_numpy())[::-1][:3].sum())).reindex(index).fillna(0.0)

    features = pd.DataFrame(index=index)
    features["top3_length_share"] = _safe_ratio(top3, totals)
    features["length_decay_slope"] = (
        grouped.apply(lambda s: _decay_slope(s.to_numpy())).reindex(index).fillna(0.0)
    )

    per_bucket = (
        frame.pivot_table(index="fips_code", columns="_bucket", values="_chars", aggfunc="sum")
        .reindex(index=index, columns=list(BUCKET_KEYS))
        .fillna(0.0)
    )
    for bucket in BUCKET_KEYS:
        features[f"chars_{bucket}"] = per_bucket[bucket]
    return features


def _decay_slope(lengths: np.ndarray) -> float:
    """OLS slope of log length on rank, over a county's sections longest-first.

    Measures how fast the article falls away from its main section: a steep
    negative slope is one substantial section and a tail of stubs, a flat slope
    is an evenly developed article.

    Args:
        lengths: Section lengths for one county.

    Returns:
        The slope, or 0.0 when the county has fewer than two sections.
    """
    if len(lengths) < 2:
        return 0.0
    ordered = np.sort(lengths.astype("float64"))[::-1]
    ranks = np.arange(len(ordered), dtype="float64")
    slope, _ = np.polyfit(ranks, np.log1p(ordered), 1)
    return float(slope)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_source_a_shape_profile.py tests/test_source_a_shape_profile.py
git commit -m "feat(source-a): add character-class densities and length-curve shape"
```

---

### Task 4: Assemble the shape profile and write the parquet

**Files:**
- Modify: `scripts/extract_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`
- Creates at runtime: `data/source_a_shape_profile.parquet`, `analysis-output/source-a/source_a_shape_profile_stats.json`

**Interfaces:**
- Consumes: `position_features`, `template_features`, `surface_features`, `length_curve_features` from Tasks 1–3.
- Produces:
  - `build_shape_profile(sections: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]` — `(features with fips_code as a column, metadata)`
  - `shape_profile_columns(features: pd.DataFrame) -> list[str]`
  - `summarize(features: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]`
  - `main() -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def test_every_county_appears_exactly_once(sections_frame: pd.DataFrame) -> None:
    features, _ = shape.build_shape_profile(sections_frame)

    assert len(features) == sections_frame["fips_code"].nunique()
    assert features["fips_code"].is_unique


def test_all_feature_columns_are_finite(sections_frame: pd.DataFrame) -> None:
    features, _ = shape.build_shape_profile(sections_frame)
    block = features[shape.shape_profile_columns(features)]

    assert (block.dtypes == "float64").all()
    assert np.isfinite(block.to_numpy()).all()


def test_the_profile_shares_no_column_with_round_one(sections_frame: pd.DataFrame) -> None:
    """The analysis module joins the two parquets; a shared name would collide."""
    import extract_source_a_structure_features as structure

    profile, _ = shape.build_shape_profile(sections_frame)
    round_one, _ = structure.build_structure_features(sections_frame)

    shared = (set(profile.columns) & set(round_one.columns)) - {"fips_code"}
    assert shared == set(), f"colliding column names: {sorted(shared)}"


def test_position_columns_cover_round_ones_flag_vocabulary(sections_frame: pd.DataFrame) -> None:
    """`pos_x` and `has_section_x` must never describe different title sets."""
    import extract_source_a_structure_features as structure

    profile, _ = shape.build_shape_profile(sections_frame)

    flagged = {
        c[len(structure.TITLE_FLAG_PREFIX):]
        for c in structure.build_structure_features(sections_frame)[0].columns
        if c.startswith(structure.TITLE_FLAG_PREFIX)
    }
    positioned = {
        c[len(shape.POSITION_PREFIX):]
        for c in profile.columns
        if c.startswith(shape.POSITION_PREFIX) and not c.startswith("pos_first_")
        and c != "pos_longest_section"
    }

    assert positioned == flagged


def test_summary_records_the_sets_it_derived(sections_frame: pd.DataFrame) -> None:
    features, metadata = shape.build_shape_profile(sections_frame)

    stats = shape.summarize(features, metadata)

    assert stats["n_counties"] == len(features)
    assert stats["modal_title_set"] == metadata["modal_title_set"]
    assert stats["modal_title_min_share"] == shape.MODAL_TITLE_MIN_SHARE
    assert stats["unusual_title_max_share"] == shape.UNUSUAL_TITLE_MAX_SHARE
    assert len(stats["modal_title_set"]) > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 5 failures with `AttributeError: module 'extract_source_a_shape_profile' has no attribute 'build_shape_profile'`.

- [ ] **Step 3: Add assembly, summary and the entry point**

Append to `scripts/extract_source_a_shape_profile.py`:

```python
def shape_profile_columns(features: pd.DataFrame) -> list[str]:
    """List the scored columns of an assembled shape profile.

    Args:
        features: Output of `build_shape_profile`.

    Returns:
        Every column except the `fips_code` key.
    """
    return [column for column in features.columns if column != "fips_code"]


def build_shape_profile(sections: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assemble all four new families for every county.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Tuple of (features, metadata). `features` carries `fips_code` as a
        column and every feature as float64; `metadata` records the two title
        sets that were derived from the corpus, so a shift in either is
        auditable rather than silent.
    """
    vocabulary = flag_vocabulary(sections)
    modal = modal_title_set(sections)
    parts = [
        position_features(sections, vocabulary),
        template_features(sections),
        surface_features(sections),
        length_curve_features(sections),
    ]
    features = pd.concat(parts, axis=1).astype("float64")
    features.index.name = "fips_code"
    metadata: dict[str, object] = {
        "flag_vocabulary": vocabulary,
        "modal_title_set": modal,
    }
    return features.reset_index(), metadata


def summarize(features: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    """Describe the profile for the notebook and for later auditing.

    Args:
        features: Output of `build_shape_profile`.
        metadata: The derived title sets from `build_shape_profile`.

    Returns:
        Counts, the derived sets and thresholds, and per-column summary
        statistics keyed by column name.
    """
    columns = shape_profile_columns(features)
    block = features[columns]
    return {
        "n_counties": int(len(features)),
        "n_features": len(columns),
        "modal_title_set": metadata["modal_title_set"],
        "flag_vocabulary": metadata["flag_vocabulary"],
        "modal_title_min_share": MODAL_TITLE_MIN_SHARE,
        "unusual_title_max_share": UNUSUAL_TITLE_MAX_SHARE,
        "position_absent_sentinel": POSITION_ABSENT,
        "density_buckets": list(DENSITY_BUCKETS),
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
    """Build the shape profile from the section parquet and write it out."""
    configure_logging()

    try:
        sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    except FileNotFoundError:
        logger.error("Need %s -- run ingest_source_a.py first.", SECTIONS_PARQUET_PATH)
        raise

    features, metadata = build_shape_profile(sections)
    stats = summarize(features, metadata)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(SHAPE_PROFILE_PATH, index=False)
    SHAPE_PROFILE_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info(
        "wrote %d shape-profile features for %d counties to %s",
        stats["n_features"],
        stats["n_counties"],
        SHAPE_PROFILE_PATH,
    )
    logger.info(
        "modal skeleton: %d titles above %.0f%% of counties",
        len(metadata["modal_title_set"]),
        MODAL_TITLE_MIN_SHARE * 100,
    )
    logger.info("wrote %s", SHAPE_PROFILE_STATS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 30 passed.

- [ ] **Step 5: Build the parquet**

```bash
uv run scripts/extract_source_a_shape_profile.py
```

Expected: roughly 60 features across 3,144 counties, and the modal-skeleton line. Report the actual counts.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_source_a_shape_profile.py tests/test_source_a_shape_profile.py data/source_a_shape_profile.parquet analysis-output/source-a/source_a_shape_profile_stats.json
git commit -m "feat(source-a): assemble and persist the shape profile"
```

---

### Task 5: The scoring module — five arms, ridge, three framings

**Files:**
- Create: `scripts/analyze_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`

**Interfaces:**
- Consumes: everything named in **What already exists** above, plus `SHAPE_PROFILE_PATH` and `shape_profile_columns` from Task 4.
- Produces:
  - `SHAPE_ARMS: tuple[Arm, ...]` — keys `shape_v1`, `shape_v2`, `typed`, `typed_plus_shape_v2`, `size_nonlinear`
  - `attach_blocks(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]` — `(matrix, v1_columns, profile_columns)`
  - `build_arm_blocks(matrix, v1_cols, profile_cols, rows) -> dict[str, np.ndarray]`
  - `score_target(matrix, v1_cols, profile_cols, baseline, flexible_baseline, target) -> dict[str, float | str | int]`
  - `run_sweep(matrix, v1_cols, profile_cols, targets) -> pd.DataFrame`

Two facts the implementer should not rediscover: `attach_structure` from `analyze_source_a_structure` already merges round one's parquet with `validate="one_to_one"`, a collision check and a row-count check — reuse it rather than writing a second merge. And `build_matrix()` does not carry `n_body_sections`, so neither merge can collide on it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def test_five_arms_are_declared() -> None:
    import analyze_source_a_shape_profile as scoring

    assert [arm.key for arm in scoring.SHAPE_ARMS] == [
        "shape_v1",
        "shape_v2",
        "typed",
        "typed_plus_shape_v2",
        "size_nonlinear",
    ]


def test_shape_v2_strictly_contains_shape_v1() -> None:
    """v2 is v1 plus the new families; if it were not, the comparison is meaningless."""
    import analyze_source_a_shape_profile as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    matrix, v1_cols, profile_cols = scoring.attach_blocks(matrix)
    rows = np.ones(len(matrix), dtype=bool)

    blocks = scoring.build_arm_blocks(matrix, v1_cols, profile_cols, rows)

    assert blocks["shape_v1"].shape[1] == len(v1_cols)
    assert blocks["shape_v2"].shape[1] == len(v1_cols) + len(profile_cols)


def test_both_blocks_attach_without_collision() -> None:
    import analyze_source_a_shape_profile as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    attached, v1_cols, profile_cols = scoring.attach_blocks(matrix)

    assert len(attached) == len(matrix)
    assert not set(v1_cols) & set(profile_cols)
    assert attached[v1_cols + profile_cols].notna().all().all()


def test_every_arm_carries_all_three_framings() -> None:
    """A row missing a framing is how round one's number got quoted wrong."""
    import analyze_source_a_shape_profile as scoring

    record = scoring.empty_record_keys()

    for arm in scoring.SHAPE_ARMS:
        assert f"r2_alone_{arm.key}" in record
        assert f"lift_{arm.key}" in record
        assert f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}" in record
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'analyze_source_a_shape_profile'`.

- [ ] **Step 3: Write the scoring module**

Create `scripts/analyze_source_a_shape_profile.py`:

```python
"""How much can be pulled from article shape -- in all three framings at once.

Round one scored a 64-column structural block and reported +0.00269 mean lift
over a linear size-plus-state baseline. The branch review then showed that an
*information-free* nonlinear reshaping of that baseline's own size columns
scores +0.01748 through the same protocol, and that roughly three quarters of
the structural lift disappears once the baseline is allowed to be curved
(`analysis-output/source-a/source-a-findings.md` §23).

The lesson was not that the number was wrong. It was that a correct number was
quoted in a framing its readers could not see. So this module reports every arm
in three framings and never one without the others:

- `r2_alone_<arm>`  -- out-of-fold R2 with the block as the *only* predictor.
                       No controls. This is the raw-power reading: how much of a
                       county is recoverable from article shape, size and
                       geography included.
- `lift_<arm>`      -- lift over the linear size-plus-state baseline. Comparable
                       to §13 through §23.
- `lift_<arm>_flexbase` -- lift over the curvature-augmented baseline. The
                       strict reading, and the one §23 showed matters.

Five arms, with `shape_v1` present as a regression check rather than a finding:
it re-scores round one's exact block through this module, and its
`lift_shape_v1` must reproduce §23's `lift_structure`. If it does not, something
drifted and the rest of the sweep is not trustworthy.

Run after `extract_source_a_structure_features.py` and
`extract_source_a_shape_profile.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import N_FOLDS, RANDOM_SEED, Target, build_baseline_design
from analyze_source_a_representation import (
    _alone_oof_r2,
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from analyze_source_a_structure import (
    FLEXIBLE_SUFFIX,
    Arm,
    attach_structure,
    build_flexible_baseline_design,
    size_nonlinear_block,
    typed_columns,
)
from extract_source_a_shape_profile import SHAPE_PROFILE_PATH, shape_profile_columns
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_shape_profile_scores.csv"
OUTPUT_PILLAR_CSV_PATH: Path = OUTPUTS_DIR / "source_a_shape_profile_by_pillar.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_shape_profile_stats_scoring.json"

SHAPE_ARMS: tuple[Arm, ...] = (
    # First, and not a finding either: this reproduces round one's block so a
    # drift anywhere in the shared machinery shows up as a changed number here
    # rather than as a silently different result downstream.
    Arm("shape_v1", "REGRESSION CHECK: round one's 64 structural columns", None),
    Arm("shape_v2", "structural block + the four new shape families", "shape_v1"),
    Arm("typed", "shipped 29 typed columns", None),
    Arm("typed_plus_shape_v2", "typed columns + full shape profile", "typed"),
    Arm(
        "size_nonlinear",
        "NULL CONTROL: nonlinear transforms of the baseline's own size columns",
        None,
    ),
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def empty_record_keys() -> dict[str, float]:
    """The full column set one scored target produces, with zeros.

    Exists so a test can assert that every arm carries every framing without
    running a sweep. A row missing a framing is exactly how round one's number
    came to be quoted in a reading its audience could not see.

    Returns:
        Mapping of every per-arm result column to 0.0.
    """
    keys: dict[str, float] = {}
    for arm in SHAPE_ARMS:
        keys[f"r2_alone_{arm.key}"] = 0.0
        keys[f"lift_{arm.key}"] = 0.0
        keys[f"lift_{arm.key}{FLEXIBLE_SUFFIX}"] = 0.0
    return keys


def attach_blocks(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge round one's structural block and the new shape profile onto the matrix.

    Round one's merge is delegated to `analyze_source_a_structure.attach_structure`,
    which already carries the collision check, the `validate="one_to_one"` guard
    and the row-count assertion. The profile merge repeats those guards for the
    same reasons.

    Args:
        matrix: Feature matrix from `build_matrix`.

    Returns:
        Tuple of (matrix with both blocks attached, round-one column names, shape
        profile column names).

    Raises:
        FileNotFoundError: If either parquet is absent.
        ValueError: On a column collision, a non-one-to-one merge, or a row whose
            profile is missing.
    """
    attached, v1_columns = attach_structure(matrix)

    try:
        profile = pd.read_parquet(SHAPE_PROFILE_PATH)
    except FileNotFoundError:
        logger.error(
            "Need %s -- run extract_source_a_shape_profile.py first.", SHAPE_PROFILE_PATH
        )
        raise

    profile_columns = shape_profile_columns(profile)
    collisions = sorted(set(profile_columns) & set(attached.columns))
    if collisions:
        raise ValueError(f"Shape profile columns already in the matrix: {collisions}")

    merged = attached.merge(profile, on="fips_code", how="left", validate="one_to_one")
    if len(merged) != len(attached):
        raise ValueError(
            f"Profile merge changed the row count: {len(attached)} -> {len(merged)}"
        )
    missing = int(merged[profile_columns].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} matrix rows have no shape profile")
    return merged, v1_columns, profile_columns


def build_arm_blocks(
    matrix: pd.DataFrame, v1_cols: list[str], profile_cols: list[str], rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Assemble every arm's feature array for one target's usable rows.

    Every arm is sliced with the same `rows` mask, which is what makes the
    per-target differences paired.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        rows: Boolean mask of rows where the target is observed.

    Returns:
        Mapping of arm key to feature array, one entry per member of `SHAPE_ARMS`.
    """
    typed = typed_columns()
    v2 = [*v1_cols, *profile_cols]
    return {
        "shape_v1": matrix.loc[rows, v1_cols].to_numpy(dtype="float64"),
        "shape_v2": matrix.loc[rows, v2].to_numpy(dtype="float64"),
        "typed": matrix.loc[rows, typed].to_numpy(dtype="float64"),
        "typed_plus_shape_v2": matrix.loc[rows, [*typed, *v2]].to_numpy(dtype="float64"),
        "size_nonlinear": size_nonlinear_block(matrix.loc[rows]).to_numpy(dtype="float64"),
    }


def score_target(
    matrix: pd.DataFrame,
    v1_cols: list[str],
    profile_cols: list[str],
    baseline: pd.DataFrame,
    flexible_baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score every arm against one target, in all three framings.

    One splitter is constructed here and handed to every fit, so every arm sees
    identical folds and identical rows under both baselines and in the
    no-baseline reading.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        baseline: Linear design from `build_baseline_design`.
        flexible_baseline: The curvature-augmented design.
        target: The column to predict.

    Returns:
        One record with both baselines' R2 and, per arm, the raw R2 and the lift
        over each baseline.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")
    flexible_design = flexible_baseline.loc[rows].to_numpy(dtype="float64")

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_baseline = float(r2_score(y, _baseline_oof_predictions(base_design, y, folds)))
    r2_flexible = float(r2_score(y, _baseline_oof_predictions(flexible_design, y, folds)))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
        "r2_baseline_flexible": r2_flexible,
    }

    blocks = build_arm_blocks(matrix, v1_cols, profile_cols, rows)
    for arm in SHAPE_ARMS:
        block = blocks[arm.key]
        record[f"r2_alone_{arm.key}"] = _alone_oof_r2(block, y, folds, None)
        record[f"lift_{arm.key}"] = (
            float(r2_score(y, _residual_oof_predictions(base_design, block, y, folds, None)))
            - r2_baseline
        )
        record[f"lift_{arm.key}{FLEXIBLE_SUFFIX}"] = (
            float(
                r2_score(
                    y, _residual_oof_predictions(flexible_design, block, y, folds, None)
                )
            )
            - r2_flexible
        )
    return record


def run_sweep(
    matrix: pd.DataFrame, v1_cols: list[str], profile_cols: list[str], targets: list[Target]
) -> pd.DataFrame:
    """Score every target against every arm.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        targets: Targets to score.

    Returns:
        Per-target results, sorted by the `shape_v2` arm's raw R2.
    """
    baseline = build_baseline_design(matrix)
    flexible_baseline = build_flexible_baseline_design(matrix, baseline)

    records = []
    for target in targets:
        record = score_target(
            matrix, v1_cols, profile_cols, baseline, flexible_baseline, target
        )
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  alone=%.4f  lift=%+.4f  flex=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["r2_alone_shape_v2"],
            record["lift_shape_v2"],
            record[f"lift_shape_v2{FLEXIBLE_SUFFIX}"],
        )
    return (
        pd.DataFrame(records)
        .sort_values("r2_alone_shape_v2", ascending=False)
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 34 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_source_a_shape_profile.py tests/test_source_a_shape_profile.py
git commit -m "feat(source-a): score the shape arms in all three framings at once"
```

---

### Task 6: The boosting learner

**Files:**
- Modify: `scripts/analyze_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`

**Interfaces:**
- Consumes: `SHAPE_ARMS`, `build_arm_blocks`, `score_target` from Task 5.
- Produces:
  - `LEARNERS: tuple[str, ...] = ("ridge", "boost")`, `BOOST_SUFFIX: str = "_boost"`
  - `make_booster() -> HistGradientBoostingRegressor`
  - `boost_residual_oof(base_design, block, y, folds) -> np.ndarray`
  - `boost_alone_oof_r2(block, y, folds) -> float`
  - `score_target` extended to emit `<column>_boost` alongside every ridge column

Why a local residual routine rather than the imported one: `_residual_oof_predictions` hardcodes `_residual_pipeline`, which is `RidgeCV`. The ridge path must keep delegating to the import so its numbers stay bit-comparable to §23; the boosting path needs the same fold structure with a different estimator, so it is written once here and mirrors the import's shape exactly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def test_boost_residual_matches_the_imported_routines_fold_structure() -> None:
    """The two learners must differ in estimator only, never in fold handling."""
    import analyze_source_a_shape_profile as scoring
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 3))
    block = rng.normal(size=(200, 5))
    y = base[:, 0] * 2.0 + block[:, 1] ** 2 + rng.normal(scale=0.1, size=200)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    predictions = scoring.boost_residual_oof(base, block, y, folds)

    assert predictions.shape == (200,)
    assert np.isfinite(predictions).all()


def test_boost_recovers_a_curve_that_ridge_cannot() -> None:
    """The reason this learner exists: a step function is invisible to a linear fit."""
    import analyze_source_a_shape_profile as scoring
    from analyze_source_a_representation import _alone_oof_r2
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=600)
    block = x.reshape(-1, 1)
    y = np.where(x > 0.0, 1.0, -1.0) + rng.normal(scale=0.05, size=600)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    assert scoring.boost_alone_oof_r2(block, y, folds) > _alone_oof_r2(block, y, folds, None)


def test_both_learners_appear_in_every_record_key() -> None:
    import analyze_source_a_shape_profile as scoring

    keys = scoring.empty_record_keys()

    for arm in scoring.SHAPE_ARMS:
        assert f"r2_alone_{arm.key}{scoring.BOOST_SUFFIX}" in keys
        assert f"lift_{arm.key}{scoring.BOOST_SUFFIX}" in keys
        assert f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{scoring.BOOST_SUFFIX}" in keys
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 3 failures with `AttributeError: module 'analyze_source_a_shape_profile' has no attribute 'boost_residual_oof'`.

- [ ] **Step 3: Add the learner**

Add `from sklearn.ensemble import HistGradientBoostingRegressor` to the imports, then insert before `score_target`:

```python
# Both learners run for every arm and every framing. Ridge is primary: it is what
# §13 through §23 used, so it is the only reading directly comparable to them.
# Boost is secondary and exists because "as much as possible" is bounded by the
# model class -- a stub-heavy article with 30 sections is a different object from
# a stub-heavy article with 5, and a linear learner cannot say so.
LEARNERS: tuple[str, ...] = ("ridge", "boost")

BOOST_SUFFIX: str = "_boost"

# Fixed rather than searched, deliberately. A nested search would be the honest
# tuned number, but it multiplies runtime by the grid and the point of this arm
# is a ceiling estimate. Fixed settings make the reported ceiling a *lower*
# bound on what boosting could reach, which is the safe direction for a number
# that will be quoted. `min_samples_leaf` is high for the panel size because
# 3,144 counties over 5 folds leaves ~2,500 training rows and a shallow leaf
# would fit fold noise.
BOOST_PARAMS: dict[str, object] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}


def make_booster() -> HistGradientBoostingRegressor:
    """Build the boosting estimator, at fixed hyperparameters.

    Returns:
        An unfitted `HistGradientBoostingRegressor`. It handles NaNs natively,
        so no imputer is needed -- unlike the ridge path, which imputes inside
        `_residual_pipeline`.
    """
    return HistGradientBoostingRegressor(**BOOST_PARAMS)


def boost_residual_oof(
    base_design: np.ndarray, block: np.ndarray, y: np.ndarray, folds: KFold
) -> np.ndarray:
    """Out-of-fold predictions from the controls plus a boosted block.

    Mirrors `_residual_oof_predictions` exactly -- controls fitted unpenalized on
    the training rows, their residuals become the boosting target, the two
    predictions summed on the held-out rows -- with the estimator swapped. It is
    written here rather than reused because the imported routine hardcodes
    `RidgeCV`, and the ridge path must keep calling the import so its numbers
    stay bit-comparable to §23.

    Args:
        base_design: Control array.
        block: Feature block for the same rows.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        Out-of-fold prediction per row.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(base_design):
        controls = _baseline_pipeline().fit(base_design[train_idx], y[train_idx])
        residuals = y[train_idx] - controls.predict(base_design[train_idx])
        model = make_booster().fit(block[train_idx], residuals)
        predictions[test_idx] = controls.predict(base_design[test_idx]) + model.predict(
            block[test_idx]
        )
    return predictions


def boost_alone_oof_r2(block: np.ndarray, y: np.ndarray, folds: KFold) -> float:
    """Out-of-fold R2 of a boosted block with no controls at all.

    Args:
        block: Feature block.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        R2 over the concatenated out-of-fold predictions.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(block):
        model = make_booster().fit(block[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(block[test_idx])
    return float(r2_score(y, predictions))
```

Add `_baseline_pipeline` to the `analyze_source_a_representation` import list. Then extend `empty_record_keys` to emit both learners:

```python
def empty_record_keys() -> dict[str, float]:
    """The full column set one scored target produces, with zeros.

    Exists so a test can assert that every arm carries every framing under every
    learner without running a sweep. A row missing a framing is exactly how round
    one's number came to be quoted in a reading its audience could not see.

    Returns:
        Mapping of every per-arm result column to 0.0.
    """
    keys: dict[str, float] = {}
    for arm in SHAPE_ARMS:
        for suffix in ("", BOOST_SUFFIX):
            keys[f"r2_alone_{arm.key}{suffix}"] = 0.0
            keys[f"lift_{arm.key}{suffix}"] = 0.0
            keys[f"lift_{arm.key}{FLEXIBLE_SUFFIX}{suffix}"] = 0.0
    return keys
```

And extend the arm loop in `score_target`, immediately after the three ridge assignments:

```python
        record[f"r2_alone_{arm.key}{BOOST_SUFFIX}"] = boost_alone_oof_r2(block, y, folds)
        record[f"lift_{arm.key}{BOOST_SUFFIX}"] = (
            float(r2_score(y, boost_residual_oof(base_design, block, y, folds))) - r2_baseline
        )
        record[f"lift_{arm.key}{FLEXIBLE_SUFFIX}{BOOST_SUFFIX}"] = (
            float(r2_score(y, boost_residual_oof(flexible_design, block, y, folds)))
            - r2_flexible
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_source_a_shape_profile.py tests/test_source_a_shape_profile.py
git commit -m "feat(source-a): add a boosting learner beside the ridge one"
```

---

### Task 7: The joint-size diagnostic, the summary, and the sweep run

**Files:**
- Modify: `scripts/analyze_source_a_shape_profile.py`
- Modify: `tests/test_source_a_shape_profile.py`
- Creates at runtime: `outputs/source_a_shape_profile_scores.csv`, `outputs/source_a_shape_profile_by_pillar.csv`, `analysis-output/source-a/source_a_shape_profile_stats_scoring.json`

**Interfaces:**
- Consumes: everything from Tasks 5–6.
- Produces:
  - `size_recoverability(matrix, blocks_by_key) -> dict[str, dict[str, float]]`
  - `summarize_by_pillar(results) -> pd.DataFrame`
  - `summarize(results, size_recovery, n_v1, n_profile) -> dict[str, object]`
  - `main() -> None`

The diagnostic is what §23 said did not exist. §23's per-column audit clears each column individually and clears nothing else, because the size dependence is joint across the block. Inverting the question — predict size *from* shape — measures the joint channel in one number per (size measure, block, learner), and bounds how much of any lift could be size in disguise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_a_shape_profile.py`:

```python
def test_size_recoverability_finds_a_planted_size_signal() -> None:
    """A block that is a noisy copy of size must score high; noise must not."""
    import analyze_source_a_shape_profile as scoring

    rng = np.random.default_rng(0)
    size = rng.normal(size=400)
    matrix = pd.DataFrame(
        {"log_population": size, "log_agi": size, "log_gdp_latest": size}
    )
    blocks = {
        "copy": (size + rng.normal(scale=0.1, size=400)).reshape(-1, 1),
        "noise": rng.normal(size=(400, 3)),
    }

    recovery = scoring.size_recoverability(matrix, blocks)

    assert recovery["copy"]["log_population_ridge"] > 0.9
    assert recovery["noise"]["log_population_ridge"] < 0.1


def test_size_recoverability_reports_every_size_measure_and_learner() -> None:
    import analyze_source_a_shape_profile as scoring

    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        {
            "log_population": rng.normal(size=200),
            "log_agi": rng.normal(size=200),
            "log_gdp_latest": rng.normal(size=200),
        }
    )

    recovery = scoring.size_recoverability(matrix, {"block": rng.normal(size=(200, 2))})

    for measure in ("log_population", "log_agi", "log_gdp_latest"):
        for learner in ("ridge", "boost"):
            assert f"{measure}_{learner}" in recovery["block"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 2 failures with `AttributeError: module 'analyze_source_a_shape_profile' has no attribute 'size_recoverability'`.

- [ ] **Step 3: Add the diagnostic, the summary and `main`**

Add `from sklearn.impute import SimpleImputer` and `from pillar_matrix import SIZE_FEATURES, build_matrix` to the imports, then append:

```python
def size_recoverability(
    matrix: pd.DataFrame, blocks_by_key: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """How much of county size each block can reconstruct.

    §23 closed on an open problem: the per-column size audit comes back clean
    while the block as a whole carries size, because the dependence is *joint*
    across columns and no per-column statistic can see it. This inverts the
    question. Predicting size *from* shape measures the joint channel directly,
    in one number, and bounds how much of any reported lift could be size in
    disguise.

    Both learners run, because the channel §23 found is curved and a linear
    reading of it would understate it -- which is the same mistake round one made
    one level up.

    Args:
        matrix: Any frame carrying `SIZE_FEATURES`.
        blocks_by_key: Feature arrays to test, keyed by name. Every array must
            have one row per row of `matrix`.

    Returns:
        Nested mapping of block key to `{"<size measure>_<learner>": R2}`.
    """
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sizes = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(matrix[list(SIZE_FEATURES)]),
        columns=list(SIZE_FEATURES),
        index=matrix.index,
    )

    recovery: dict[str, dict[str, float]] = {}
    for key, block in blocks_by_key.items():
        scores: dict[str, float] = {}
        for measure in SIZE_FEATURES:
            y = sizes[measure].to_numpy(dtype="float64")
            scores[f"{measure}_ridge"] = _alone_oof_r2(block, y, folds, None)
            scores[f"{measure}_boost"] = boost_alone_oof_r2(block, y, folds)
        recovery[key] = scores
    return recovery


def summarize_by_pillar(results: pd.DataFrame) -> pd.DataFrame:
    """Mean lift per arm within each target's owning pillar.

    Reported beside the aggregate and never instead of it: 20 of the 28 targets
    are one QCEW table, so a basket-wide mean is 71% one pillar.

    Args:
        results: Output of `run_sweep`.

    Returns:
        One row per pillar with the target count and each arm's mean lift under
        each learner.
    """
    aggregations: dict[str, tuple[str, str]] = {"n_targets": ("column", "count")}
    for arm in SHAPE_ARMS:
        for suffix in ("", BOOST_SUFFIX):
            aggregations[f"{arm.key}{suffix}"] = (f"lift_{arm.key}{suffix}", "mean")
    return results.groupby("pillar").agg(**aggregations).reset_index()


def _paired_test(results: pd.DataFrame, arm: Arm, column: str) -> dict[str, object]:
    """Test one arm's lift column against its comparison across every target.

    Args:
        results: Output of `run_sweep`.
        arm: The arm to test. `arm.against` names the arm it is paired with;
            None means the comparison is against the baseline, where the lift
            column is already the difference.
        column: Full lift column name, carrying whichever suffixes apply.

    Returns:
        Mean lift, mean paired difference, win count and the Wilcoxon
        signed-rank p-value.
    """
    lifts = results[column]
    if arm.against is None:
        differences = lifts
    else:
        differences = lifts - results[column.replace(arm.key, arm.against, 1)]
    statistic, p_value = wilcoxon(differences)
    return {
        "mean_lift": float(lifts.mean()),
        "median_lift": float(lifts.median()),
        "mean_paired_difference": float(differences.mean()),
        "compared_against": arm.against or "baseline",
        "n_wins": int((differences > 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def summarize(
    results: pd.DataFrame,
    size_recovery: dict[str, dict[str, float]],
    n_v1: int,
    n_profile: int,
) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    Args:
        results: Output of `run_sweep`.
        size_recovery: Output of `size_recoverability`.
        n_v1: Width of round one's block.
        n_profile: Width of the new shape profile.

    Returns:
        Target counts, block widths, the size diagnostic, and per-arm results in
        all three framings under both learners.
    """
    arms: dict[str, object] = {}
    for arm in SHAPE_ARMS:
        for learner, suffix in (("ridge", ""), ("boost", BOOST_SUFFIX)):
            arms[f"{arm.key}_{learner}"] = {
                "label": arm.label,
                "learner": learner,
                "mean_r2_alone": float(results[f"r2_alone_{arm.key}{suffix}"].mean()),
                "linear": _paired_test(results, arm, f"lift_{arm.key}{suffix}"),
                "flexible": _paired_test(
                    results, arm, f"lift_{arm.key}{FLEXIBLE_SUFFIX}{suffix}"
                ),
            }
    return {
        "n_targets": int(len(results)),
        "n_shape_v1_features": n_v1,
        "n_shape_profile_features": n_profile,
        "n_shape_v2_features": n_v1 + n_profile,
        "n_typed_features": len(typed_columns()),
        "mean_r2_baseline": float(results["r2_baseline"].mean()),
        "mean_r2_baseline_flexible": float(results["r2_baseline_flexible"].mean()),
        "size_recoverability": size_recovery,
        "arms": arms,
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def main() -> None:
    """Attach both blocks, run the sweep and the diagnostic, write the artifacts."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix, v1_cols, profile_cols = attach_blocks(matrix)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets: shape_v1=%d, profile=%d, shape_v2=%d, typed=%d",
        len(targets),
        len(v1_cols),
        len(profile_cols),
        len(v1_cols) + len(profile_cols),
        len(typed_columns()),
    )

    results = run_sweep(matrix, v1_cols, profile_cols, targets)

    all_rows = np.ones(len(matrix), dtype=bool)
    diagnostic_blocks = build_arm_blocks(matrix, v1_cols, profile_cols, all_rows)
    size_recovery = size_recoverability(
        matrix, {key: diagnostic_blocks[key] for key in ("shape_v1", "shape_v2")}
    )

    pillar_results = summarize_by_pillar(results)
    stats = summarize(results, size_recovery, len(v1_cols), len(profile_cols))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    pillar_results.to_csv(OUTPUT_PILLAR_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("size recoverable from the shape block (out-of-fold R2):")
    for key, scores in size_recovery.items():
        for measure in SIZE_FEATURES:
            logger.info(
                "  %-9s %-16s ridge %.4f | boost %.4f",
                key,
                measure,
                scores[f"{measure}_ridge"],
                scores[f"{measure}_boost"],
            )

    for name, arm_stats in stats["arms"].items():
        logger.info(
            "%-26s alone %.4f | linear %+.5f (p=%.4f) | flexible %+.5f (p=%.4f)",
            name,
            arm_stats["mean_r2_alone"],
            arm_stats["linear"]["mean_lift"],
            arm_stats["linear"]["wilcoxon_p"],
            arm_stats["flexible"]["mean_lift"],
            arm_stats["flexible"]["wilcoxon_p"],
        )

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_source_a_shape_profile.py -v
```

Expected: 39 passed.

- [ ] **Step 5: Run the sweep**

```bash
uv run scripts/analyze_source_a_shape_profile.py
```

This is the expensive step: 5 arms × 3 framings × 2 learners × 28 targets × 5 folds. Expect on the order of 20–40 minutes, dominated by boosting. Let it finish.

- [ ] **Step 6: Check the regression arm before trusting anything else**

```bash
uv run python -c "
import json, pandas as pd
new = pd.read_csv('outputs/source_a_shape_profile_scores.csv').set_index('column')['lift_shape_v1']
old = pd.read_csv('outputs/source_a_structure_scores.csv').set_index('column')['lift_structure']
joined = new.to_frame('new').join(old.to_frame('old'))
joined['delta'] = joined['new'] - joined['old']
print(joined.sort_values('delta', key=abs, ascending=False).head())
print('max abs delta:', joined['delta'].abs().max())
"
```

Expected: `max abs delta` at or near zero — `shape_v1` is round one's block through the same protocol and must reproduce §23's `lift_structure`. **A non-trivial delta means the shared machinery drifted; stop and report it rather than continuing.**

- [ ] **Step 7: Report the diagnostic**

```bash
uv run python -c "
import json
s = json.load(open('analysis-output/source-a/source_a_shape_profile_stats_scoring.json'))
print('targets', s['n_targets'], '| v1', s['n_shape_v1_features'], '| profile', s['n_shape_profile_features'])
print()
print('SIZE RECOVERABLE FROM SHAPE (out-of-fold R2):')
for block, scores in s['size_recoverability'].items():
    for k, v in scores.items():
        print(f'  {block:9} {k:24} {v:.4f}')
print()
for name, a in s['arms'].items():
    print(f\"{name:26} alone {a['mean_r2_alone']:.4f} | linear {a['linear']['mean_lift']:+.5f} p={a['linear']['wilcoxon_p']:.4f} | flex {a['flexible']['mean_lift']:+.5f} p={a['flexible']['wilcoxon_p']:.4f}\")
"
```

Record every number in the task report, whatever it says.

- [ ] **Step 8: Commit**

```bash
git add scripts/analyze_source_a_shape_profile.py tests/test_source_a_shape_profile.py outputs/source_a_shape_profile_scores.csv outputs/source_a_shape_profile_by_pillar.csv analysis-output/source-a/source_a_shape_profile_stats_scoring.json
git commit -m "feat(source-a): measure how much county size the shape block encodes"
```

---

### Task 8: The notebook

**Files:**
- Create: `scripts/build_source_a_shape_profile_notebook.py`
- Creates at runtime: `analysis-output/source-a/source_a_shape_profile_round.ipynb`

**Interfaces:**
- Consumes: `data/source_a_shape_profile.parquet`, `data/source_a_structure_features.parquet`, `data/source_a_sections.parquet`, `analysis-output/source-a/source_a_shape_profile_stats.json`, `analysis-output/source-a/source_a_shape_profile_stats_scoring.json`, `outputs/source_a_shape_profile_scores.csv`, `outputs/source_a_shape_profile_by_pillar.csv`.
- Produces: the notebook. Nothing consumes it programmatically.

Read `scripts/build_source_a_structure_notebook.py` first — this follows its pattern exactly: `md()`/`code()` helpers appending `nbformat` cells, written and executed via `nbconvert`, every number read from an artifact, matplotlib only.

Section order is fixed by the design and is not a stylistic choice: **the new families, then the size diagnostic, then the arms, then the ceiling.** The diagnostic precedes the arms so the reader knows how much size is in the block before seeing what the block scores.

- [ ] **Step 1: Write the builder**

Create `scripts/build_source_a_shape_profile_notebook.py`:

```python
"""Generate analysis-output/source-a/source_a_shape_profile_round.ipynb.

How much can be pulled from the shape of a county's Wikipedia article, reported
in three framings at once so no number can be read in a framing its reader
cannot see. That was round one's actual failure: §23's +0.00269 was correct and
still misleading, because the linear-in-logs baseline it was measured against
was not visible in the sentence that quoted it.

**The size diagnostic runs before the arms, deliberately.** §23 closed on an open
problem -- the per-column size audit clears each column individually and clears
nothing else, because the dependence is joint across the block. Predicting size
*from* shape measures that joint channel in one number, and the reader should
have it before seeing what the block scores, not after.

Every number is read from a committed artifact. Matplotlib, not plotly: plotly's
mimetype output needs a JupyterLab extension and renders as blank space without
it.

Build and execute:

    uv run scripts/build_source_a_shape_profile_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO: Path = Path(__file__).resolve().parent.parent
OUT: Path = REPO / "analysis-output" / "source-a" / "source_a_shape_profile_round.ipynb"

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
# Source A — How Much Is In the Shape of an Article

**The question:** how much of a county can be recovered from the *shape* of its
Wikipedia article — how many sections, how long, which ones, in what order, how
far from the county template, and what the characters look like — without
reading a word for meaning?

**Why every number here carries three readings.** Round one reported +0.00269
mean lift for article shape and it was arithmetically correct. It was also
misleading, because the baseline it was measured against controlled for county
size *linearly, in logs*, and an information-free curve on those same size
columns scores +0.01748 through the identical protocol. The number was right;
its framing was invisible. So nothing below is reported in one framing:

- **`r2_alone`** — the block as the only predictor, no controls. How much of a
  county is recoverable from article shape, size and geography included.
- **`lift`** — over the linear size-plus-state baseline. Comparable to §13–§23.
- **`lift_flexbase`** — over the curvature-augmented baseline. The strict reading.

Two arms are not findings and are labelled so wherever they appear:
`shape_v1` re-scores round one's block as a regression check, and
`size_nonlinear` is the information-free null control that prices the unit.
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

profile = pd.read_parquet(DATA / "source_a_shape_profile.parquet")
structure = pd.read_parquet(DATA / "source_a_structure_features.parquet")
feature_stats = json.loads((SOURCE_A / "source_a_shape_profile_stats.json").read_text())
stats = json.loads((SOURCE_A / "source_a_shape_profile_stats_scoring.json").read_text())
scores = pd.read_csv(OUTPUTS / "source_a_shape_profile_scores.csv")
by_pillar = pd.read_csv(OUTPUTS / "source_a_shape_profile_by_pillar.csv")

profile_cols = [c for c in profile.columns if c != "fips_code"]
v1_cols = [c for c in structure.columns if c != "fips_code"]

print(f"{feature_stats['n_counties']:,} counties")
print(f"shape_v1 {stats['n_shape_v1_features']} cols + profile {stats['n_shape_profile_features']} "
      f"= shape_v2 {stats['n_shape_v2_features']} cols, vs typed {stats['n_typed_features']}")
print(f"{stats['n_targets']} targets | modal skeleton: {len(feature_stats['modal_title_set'])} titles")
""")

md("""
## Part one — the four new families

Round one measured how many sections an article has, how long they are, and
which titles are present. These four families measure things it never looked at.
""")

code("""
POSITION_ABSENT = feature_stats["position_absent_sentinel"]
pos_cols = [c for c in profile_cols if c.startswith("pos_") and not c.startswith("pos_first_")
            and c != "pos_longest_section"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

present = profile[pos_cols].replace(POSITION_ABSENT, np.nan)
axes[0, 0].hist(present.mean(axis=1).dropna(), bins=40, color="#3b6ea5")
axes[0, 0].set_title("Mean position of a county's common sections")
axes[0, 0].set_xlabel("normalized position (0 = top of article)")

absent_share = (profile[pos_cols] == POSITION_ABSENT).mean().sort_values()
axes[0, 1].hist(absent_share, bins=30, color="#3b6ea5")
axes[0, 1].set_title("How often each common section is simply absent")
axes[0, 1].set_xlabel("share of counties lacking it")

axes[1, 0].hist(profile["template_jaccard"], bins=40, color="#3b6ea5")
axes[1, 0].set_title("Template conformity (Jaccard vs the modal skeleton)")
axes[1, 0].set_xlabel("1.0 = exactly the house template")

axes[1, 1].hist(profile["digit_density"], bins=40, color="#3b6ea5")
axes[1, 1].set_title("Digit density of the article body")
axes[1, 1].set_xlabel("digits / characters")

fig.tight_layout()
plt.show()

print(f"median template conformity {profile['template_jaccard'].median():.3f}; "
      f"median digit density {profile['digit_density'].median():.3f}")
print(f"history precedes economy in {profile['history_before_economy'].mean():.1%} of counties")
""")

code("""
merged = structure.merge(profile, on="fips_code", validate="one_to_one")
cross = pd.Series(
    {c: merged[v1_cols].corrwith(merged[c]).abs().max() for c in profile_cols}
).sort_values()

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(cross, bins=30, color="#3b6ea5")
ax.set_title("How much each new column duplicates round one's block")
ax.set_xlabel("largest |r| against any round-one column")
ax.set_ylabel("columns")
fig.tight_layout()
plt.show()

print(f"{(cross < 0.5).sum()} of {len(cross)} new columns stay under |r| = 0.5 against everything "
      "round one already had.")
print("Most duplicated:", ", ".join(cross.tail(3).index))
print("Most novel:", ", ".join(cross.head(3).index))
""")

md("""
## Part two — how much of county size is *in* the block

§23 closed on an open problem. Its per-column size audit came back clean — no
single structural column is a hidden curved size proxy — while the block as a
whole demonstrably carried size. Both are true: the dependence is **joint across
columns**, and no per-column statistic can see it.

So invert the question. Instead of asking whether each column looks like size,
ask how much of county size the whole block can reconstruct. One number, no
statistics argument, and it bounds how much of any lift below could be size in
disguise.
""")

code("""
recovery = pd.DataFrame(stats["size_recoverability"]).T

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(recovery.columns))
width = 0.38
for offset, block, color in zip((-width / 2, width / 2), recovery.index, ("#7a9cc6", "#2f5d8a")):
    ax.bar(x + offset, recovery.loc[block], width, label=block, color=color)
ax.set_xticks(x)
ax.set_xticklabels(recovery.columns, rotation=30, ha="right")
ax.set_ylabel("out-of-fold R²")
ax.set_title("How much of county size the shape block reconstructs")
ax.legend()
fig.tight_layout()
plt.show()

display(recovery.round(4))

peak = recovery.max().max()
where = recovery.stack().idxmax()
print(f"Peak: {peak:.3f} — {where[0]} reconstructing {where[1]}.")
print(f"So article shape encodes county size to R² = {peak:.3f}. Any lift below is what "
      "remains after a control has already removed size — read it with that in mind.")
""")

md("""
## Part three — the arms

Five arms, two learners, three framings, and no arm reported in fewer than all
three. `shape_v1` is a regression check on round one's block; `size_nonlinear` is
the information-free null control. Neither is a finding.
""")

code("""
arms = pd.DataFrame([
    {
        "arm": name,
        "learner": a["learner"],
        "r2_alone": a["mean_r2_alone"],
        "lift_linear": a["linear"]["mean_lift"],
        "p_linear": a["linear"]["wilcoxon_p"],
        "lift_flexible": a["flexible"]["mean_lift"],
        "p_flexible": a["flexible"]["wilcoxon_p"],
        "vs": a["linear"]["compared_against"],
    }
    for name, a in stats["arms"].items()
])
display(arms.round(5))
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, column, title in zip(
    axes,
    ("r2_alone", "lift_linear", "lift_flexible"),
    ("R² alone (no controls)", "Lift over linear baseline", "Lift over flexible baseline"),
):
    for offset, learner, color in zip((-0.2, 0.2), ("ridge", "boost"), ("#7a9cc6", "#2f5d8a")):
        subset = arms[arms["learner"] == learner]
        positions = np.arange(len(subset)) + offset
        bars = ax.bar(positions, subset[column], 0.4, label=learner, color=color)
        for bar, arm_name in zip(bars, subset["arm"]):
            if "size_nonlinear" in arm_name or "shape_v1" in arm_name:
                bar.set_hatch("//")
    ax.set_xticks(np.arange(len(arms[arms["learner"] == "ridge"])))
    ax.set_xticklabels(
        [n.replace("_ridge", "") for n in arms[arms["learner"] == "ridge"]["arm"]],
        rotation=35, ha="right",
    )
    ax.axhline(0, color="#333", lw=1)
    ax.set_title(title)
    ax.legend()
fig.suptitle("Hatched bars are not findings: shape_v1 is a regression check, "
             "size_nonlinear is the null control")
fig.tight_layout()
plt.show()
""")

md("""
### Per pillar, because the aggregate is a property of the basket

Twenty of the twenty-eight targets are one QCEW table, so a basket-wide mean is
71% one pillar. Reading it as a breadth claim is a mistake this project has made
before.
""")

code("""
display(by_pillar.round(5))

fig, ax = plt.subplots(figsize=(11, 4.5))
keys = [c for c in by_pillar.columns if c not in ("pillar", "n_targets") and not c.endswith("_boost")]
x = np.arange(len(by_pillar))
width = 0.8 / len(keys)
for i, key in enumerate(keys):
    ax.bar(x + i * width - 0.4, by_pillar[key], width, label=key)
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\\n({n})" for p, n in zip(by_pillar["pillar"], by_pillar["n_targets"])])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean lift by owning pillar (ridge, linear baseline)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

md("""
## Part four — where the ceiling is

The targets article shape predicts best on its own, and what survives as the
control tightens. The gap between the first column and the last is the answer to
"how much of this is really about the article."
""")

code("""
ceiling = scores[[
    "pillar", "label", "n",
    "r2_alone_shape_v2", "r2_alone_shape_v2_boost",
    "lift_shape_v2", "lift_shape_v2_flexbase",
]].head(12)
display(ceiling.round(4))

fig, ax = plt.subplots(figsize=(12, 5))
top = scores.head(12).iloc[::-1]
ax.barh(top["label"], top["r2_alone_shape_v2"], color="#7a9cc6", label="R² alone")
ax.barh(top["label"], top["lift_shape_v2_flexbase"], color="#2f5d8a", label="lift, flexible baseline")
ax.set_title("Best targets: raw predictive power vs what survives the strict control")
ax.legend()
fig.tight_layout()
plt.show()

print(f"Mean R² alone across all {stats['n_targets']} targets: "
      f"{scores['r2_alone_shape_v2'].mean():.4f}")
print(f"Mean lift over the flexible baseline: {scores['lift_shape_v2_flexbase'].mean():+.5f}")
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

The markdown cells above contain claims with no number in them ("no arm reported
in fewer than all three", "the dependence is joint"). Those are statements about
the method and are safe. **Any sentence you add that characterizes a magnitude
must be checked against the output beneath it in Step 4.**

- [ ] **Step 2: Build and execute the notebook**

```bash
uv run scripts/build_source_a_shape_profile_notebook.py
```

Expected: the write line, then nbconvert's execution log, no traceback. A failing cell is fixed in the builder and rebuilt — never by hand-editing the `.ipynb`.

- [ ] **Step 3: Confirm every cell produced output**

```bash
uv run python -c "
import json
nb = json.load(open('analysis-output/source-a/source_a_shape_profile_round.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
empty = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print(f'{len(code_cells)} code cells, {len(empty)} with no output: {empty}')
"
```

Expected: 0 cells with no output.

- [ ] **Step 4: Check the notebook's prose against its own output**

Read the executed notebook's markdown cells beside the outputs they introduce. Every claim about magnitude — "most", "small", "large", "survives", "collapses" — must match the number in the cell below it. This is the check round one failed twice: prose asserting "most of these columns are size measurements" over a table showing 6 of 64.

Fix any mismatch in the builder and rebuild. Report every sentence you changed.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: everything passes. Nothing in this round modifies a shipped artifact, so a failure elsewhere means something was touched that should not have been.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_source_a_shape_profile_notebook.py analysis-output/source-a/source_a_shape_profile_round.ipynb
git commit -m "docs(source-a): notebook for the shape-profile round"
```

---

## Verification

The round is complete when all of these hold:

```bash
uv run pytest tests/ -v
uv run scripts/extract_source_a_shape_profile.py
uv run scripts/analyze_source_a_shape_profile.py
uv run scripts/build_source_a_shape_profile_notebook.py
```

- Full test suite green.
- `data/source_a_shape_profile.parquet` has 3,144 rows, no nulls, and shares no column name with `data/source_a_structure_features.parquet`.
- `data/source_a_structure_features.parquet` is **unmodified** — `git status` shows it untouched, and §23's numbers still reproduce.
- `lift_shape_v1` reproduces §23's `lift_structure` to within floating-point noise.
- The scoring stats JSON reports `n_targets = 28` and a populated `size_recoverability` block.
- Every arm carries all three framings under both learners.
- The notebook executes end to end with output in every code cell, and no markdown claim contradicts the output beneath it.

## A note for whoever writes up the result

This round is likely to produce a large `r2_alone` — article shape probably reconstructs a good deal of a county, because it reconstructs county size. That is a real and useful finding (a cheap universal proxy) and it is *not* the same finding as "article shape knows something about counties." The size diagnostic exists precisely so those two can be told apart in one number.

Whatever the arms say, the write-up belongs in `analysis-output/source-a/source-a-findings.md` as §24, in the house format with **Allowed wording**, **Forbidden wording** and **Status** — the same format §23 used. §23's forbidden list already bans "article shape knows something county size does not"; check the new numbers against it before writing any sentence that resembles it.
