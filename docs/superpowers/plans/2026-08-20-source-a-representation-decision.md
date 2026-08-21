# Source A Representation Decision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, on evidence rather than on cost-and-interpretability, whether Source A ships as 29 typed columns or as a MiniLM encoding of the same articles.

**Architecture:** Widen the external ACS target basket from 5 to ≥40 non-circular targets, close the text-leakage channel that lets the encoder read its own answer out of Wikipedia census sections, give both representations a pre-registered capacity-equalizing pass, and score them width-matched at 29 dimensions each on marginal contribution. Scope selection runs on the in-repo 28-target basket; the decision runs on the external basket; the two never touch.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, sentence-transformers (`all-MiniLM-L6-v2`), pytest (new), `uv` for dependency management.

**Spec:** `docs/superpowers/specs/2026-08-20-source-a-representation-decision-design.md`

## Global Constraints

- Python `>=3.12`. Run everything through `uv run`, never a bare `python`.
- Random seed is `42` and folds are `N_FOLDS`, both imported from `analyze_pillar_matrix_signal`. Never redefine them locally.
- Out-of-fold scoring groups on `state_fips`. Never shuffle across states.
- ACS vintage is fixed at `ACS 2023 5-year`, `as_of_date` `2023-12-31`.
- Census downloads use the keyless `www2.census.gov` table-based summary file. Never the Census data API — it requires a key.
- Negative ACS values are sentinels (`-666666666` and relatives). Every read masks `where(value >= 0)`.
- Suppression and missingness stay explicit. Never fill a null with zero to make a join work.
- `analysis-output/*.ipynb` is a build artifact. Edit `scripts/build_status_notebook.py` and regenerate; never edit the `.ipynb`.
- Existing published numbers must not move. `outputs/source_a_tiered_embedding.csv` and the five original external-target contributions are regression-locked in Task 1.
- No target peeking. Parts 3 and 4 of the spec are pre-registered; arms are chosen by the rule in Task 9, not by their scores on the decision basket.

## Correction carried from the spec

The spec's character-share table is measured over *all* sections. `uniform` already excludes narrative titles — `build_variant_texts` hardcodes `NARRATIVE_TITLE_PATTERN` as its exclusion — so census tables are **~42%** of what `uniform` actually reads, not 36.4%. Consequently `prose_plus_history` *adds* text to `uniform` rather than trimming it. Task 5 must not describe it as a narrowing arm.

## File Structure

| File | Responsibility |
|---|---|
| `tests/conftest.py` (create) | Shared fixtures: cached section frame, cached panel |
| `tests/test_external_targets.py` (create) | ACS target arithmetic, Autauga gate, circularity registry completeness |
| `tests/test_text_scopes.py` (create) | Section-selection scope rules and their coverage |
| `tests/test_regression_locks.py` (create) | Published numbers that must not move |
| `scripts/ingest_external_targets.py` (modify) | Per-table download cache; ~40 new `ExternalTarget` entries |
| `scripts/source_a_text_leakage.py` (create) | Screens targets for verbatim restatement in census sections |
| `scripts/analyze_external_target.py` (modify) | `TARGET_RESTATEMENTS` entries for the new targets |
| `scripts/analyze_source_a_tiered_embedding.py` (modify) | `TextVariant.exclude` field; four new scopes; common-component removal |
| `scripts/analyze_source_a_representation_marginal.py` (modify) | PCA-29 arm, typed-transform arm, new scope arms |
| `scripts/source_a_typed_transform.py` (create) | The pre-registered typed capacity pass |
| `docs/source_a_representation_decision.md` (create) | Pre-registered decision rule, written before Task 10 runs |

---

### Task 1: Test harness and regression locks

Nothing in this repo is currently under test, and every later task edits code that produces published numbers. This task exists so those numbers cannot move silently.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/test_regression_locks.py`

**Interfaces:**
- Produces: `sections_frame` and `tiered_embedding_results` pytest fixtures, used by Tasks 3, 4, 5.

- [ ] **Step 1: Add pytest to the dev dependency group**

In `pyproject.toml`, under `[dependency-groups]`, add to the `dev` list:

```toml
    "pytest>=8.0.0",
```

Then run:

```bash
uv sync
```

- [ ] **Step 2: Write the shared fixtures**

Create `tests/conftest.py`:

```python
"""Fixtures shared across the Source A representation test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="session")
def sections_frame() -> pd.DataFrame:
    """Long-format Wikipedia section frame, loaded once per session."""
    return pd.read_parquet(REPO_ROOT / "data" / "source_a_sections.parquet")


@pytest.fixture(scope="session")
def tiered_embedding_results() -> pd.DataFrame:
    """Committed pooled results from the tiered embedding sweep."""
    return pd.read_csv(REPO_ROOT / "outputs" / "source_a_tiered_embedding.csv")
```

- [ ] **Step 3: Write the failing regression lock**

Create `tests/test_regression_locks.py`:

```python
"""Published numbers that later tasks must not move.

Every value here is copied from a committed artifact, not recomputed. If a
change to the encoder or the harness moves one of these, that is a finding to
investigate, not a number to update.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# From analysis-output/source-a/source_a_representation_marginal_stats.json,
# the arm that reproduces the published -0.0000 from a separate harness.
TYPED_MARGINAL_MEAN = -4.573467945852005e-05

# From outputs/source_a_tiered_embedding.csv, mean pooled lift across 28 targets.
POOLED_LIFT_EXPECTED = {
    "typed_sections": 0.003072,
    "uniform": 0.003218,
    "uniform_l2": 0.003514,
    "lead_only": 0.001686,
}


def test_typed_marginal_mean_unchanged() -> None:
    path = REPO_ROOT / "analysis-output" / "source-a" / "source_a_representation_marginal_stats.json"
    stats = json.loads(path.read_text())
    actual = stats["by_representation"]["typed"]["mean_contribution"]
    assert actual == pytest.approx(TYPED_MARGINAL_MEAN, abs=1e-12)


@pytest.mark.parametrize(("representation", "expected"), POOLED_LIFT_EXPECTED.items())
def test_pooled_lift_unchanged(
    tiered_embedding_results, representation: str, expected: float
) -> None:
    subset = tiered_embedding_results[
        tiered_embedding_results["representation"] == representation
    ]
    assert len(subset) == 28, f"{representation} should score 28 targets"
    assert subset["lift"].mean() == pytest.approx(expected, abs=5e-7)
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
uv run pytest tests/test_regression_locks.py -v
```

Expected: 5 passed. If `test_typed_marginal_mean_unchanged` fails, the stats file has already drifted — stop and investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "test: lock the published Source A numbers before the representation sweep"
```

---

### Task 2: Cache ACS table downloads by table

`_derive` calls `_download_table` once per target, and again for a same-table denominator. Each call downloads ~118MB. At 40+ targets drawn from overlapping tables this is several redundant gigabytes. Fix before Task 3 multiplies it.

**Files:**
- Modify: `scripts/ingest_external_targets.py`
- Create: `tests/test_external_targets.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_download_table(table: str) -> pd.DataFrame` — signature changes to drop the `columns` argument and return **all** county rows for the table, memoized per process.

- [ ] **Step 1: Write the failing test**

Create `tests/test_external_targets.py`:

```python
"""ACS external-target arithmetic and admission gates."""
from __future__ import annotations

import pandas as pd
import pytest

import ingest_external_targets as iet


def test_download_table_is_memoized(monkeypatch) -> None:
    """A second request for the same table must not hit the network."""
    calls: list[str] = []

    def fake_fetch(table: str) -> pd.DataFrame:
        calls.append(table)
        return pd.DataFrame(
            {"B00000_E001": [1.0], "B00000_M001": [0.1]},
            index=pd.Index(["01001"], name="fips_code"),
        )

    monkeypatch.setattr(iet, "_fetch_table_uncached", fake_fetch)
    iet._download_table.cache_clear()

    first = iet._download_table("b00000")
    second = iet._download_table("b00000")

    assert calls == ["b00000"], "second call should be served from cache"
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_external_targets.py::test_download_table_is_memoized -v
```

Expected: FAIL with `AttributeError: module 'ingest_external_targets' has no attribute '_fetch_table_uncached'`.

- [ ] **Step 3: Split the download into a cached wrapper**

In `scripts/ingest_external_targets.py`, add `from functools import lru_cache` to the imports. Rename the existing `_download_table` body into `_fetch_table_uncached`, dropping its `columns` parameter so it retains every column:

```python
def _fetch_table_uncached(table: str) -> pd.DataFrame:
    """Fetch one ACS summary-file table and reduce it to county rows.

    Retains every column, so one download serves every target drawn from this
    table. Callers select the lines they need.

    Args:
        table: Lowercase table identifier as it appears in the file name.

    Returns:
        DataFrame indexed by `fips_code` carrying all numeric table columns.

    Raises:
        requests.HTTPError: If the Census download fails.
    """
    response = requests.get(ACS_BASE_URL.format(table=table), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    raw = pd.read_csv(
        io.BytesIO(response.content),
        sep="|",
        dtype={"GEO_ID": str},
        na_values=["", ".", "-", "null"],
        low_memory=False,
    )
    counties = raw[raw["GEO_ID"].str.startswith(COUNTY_GEO_PREFIX)].copy()
    counties["fips_code"] = counties["GEO_ID"].str.removeprefix(COUNTY_GEO_PREFIX)
    counties = counties.set_index("fips_code").drop(columns=["GEO_ID"])
    return counties.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=None)
def _download_table(table: str) -> pd.DataFrame:
    """Memoized `_fetch_table_uncached`, so one table downloads once per run.

    Args:
        table: Lowercase table identifier.

    Returns:
        The cached county-row frame for that table.
    """
    return _fetch_table_uncached(table)
```

- [ ] **Step 4: Update the three call sites in `_derive`**

In `_derive`, replace each `_download_table(table, columns)` call with `_download_table(table)`. Add a missing-column check where the columns are first selected, so a changed line numbering still raises rather than silently producing NaN:

```python
    frame = _download_table(target.table)
    missing = [c for c in numerator_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{target.column}: {target.table.upper()} is missing {missing}; "
            "the table's line numbering changed and the mapping needs revisiting"
        )
```

Apply the same check to the denominator columns.

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run pytest tests/test_external_targets.py -v
```

Expected: PASS.

- [ ] **Step 6: Verify the five existing targets still derive identically**

```bash
rm -f data/external_targets.parquet
uv run python scripts/ingest_external_targets.py
uv run python -c "
import pandas as pd
d = pd.read_parquet('data/external_targets.parquet')
print(d[['broadband_rate','median_household_income','median_age']].describe().round(4))
print('rows', len(d))
"
```

Expected: 3,144-ish rows and finite describe output. Compare against `git stash`-ed original if any column shifts.

- [ ] **Step 7: Commit**

```bash
git add scripts/ingest_external_targets.py tests/test_external_targets.py
git commit -m "perf(acs): memoize summary-file downloads by table"
```

---

### Task 3: Expand the external basket to ≥40 non-circular targets

**Files:**
- Modify: `scripts/ingest_external_targets.py`
- Modify: `tests/test_external_targets.py`

**Interfaces:**
- Consumes: `_download_table(table)` from Task 2.
- Produces: `EXTERNAL_TARGETS` grown to ≥40 entries; `TARGET_CIRCULARITY: dict[str, str]` mapping every target column to one of `"clean"`, `"ablated"`, or a rejection reason.

**Verified line numbers.** Every table below was downloaded and checked against Autauga County, AL (FIPS `01001`) during planning. Values in parentheses are the observed Autauga figures — an implementer who sees different numbers should stop, because the line numbering has moved.

| construct | table | line | Autauga | kind |
|---|---|---|---|---|
| per-capita income | b19301 | E001 | 36,227 | median |
| median family income | b19113 | E001 | 83,452 | median |
| median gross rent | b25064 | E001 | 1,200 | median |
| median contract rent | b25058 | E001 | 1,020 | median |
| median monthly housing cost | b25105 | E001 | 1,048 | median |
| median year structure built | b25035 | E001 | 1,993 | median |
| mean household size | b25010 | E001 | 2.61 | median |
| owner-occupied share | b25003 | E002 / E001 | 16,872 / 22,523 | proportion |
| poverty rate | b17001 | E002 / E001 | 6,275 / 58,731 | proportion |
| family-household share | b11001 | E002 / E001 | 15,674 / 22,523 | proportion |
| single-unit-detached share | b25024 | E002 / E001 | 18,289 / 24,731 | proportion |
| bachelor's-degree share | b15003 | E022 / E001 | 6,518 / 40,767 | proportion |
| graduate-degree share | b15003 | E023 / E001 | 4,006 / 40,767 | proportion |
| labour-force participation | b23025 | E002 / E001 | 28,020 / 47,508 | proportion |

Cross-check that must hold: `B11001_E001` and `B25003_E001` are both 22,523 for Autauga — households and occupied housing units are the same universe. If they diverge, one of the two mappings is wrong.

- [ ] **Step 1: Write the failing admission-gate test**

Append to `tests/test_external_targets.py`:

```python
AUTAUGA = "01001"

# Observed during planning. A mismatch means the line numbering moved.
AUTAUGA_EXPECTED = {
    "per_capita_income": 36227.0,
    "median_family_income": 83452.0,
    "median_gross_rent": 1200.0,
    "mean_household_size": 2.61,
    "owner_occupied_share": 16872 / 22523,
    "poverty_rate": 6275 / 58731,
    "bachelors_share": 6518 / 40767,
    "labor_force_participation": 28020 / 47508,
}


@pytest.fixture(scope="module")
def targets_frame() -> pd.DataFrame:
    return iet.fetch_external_targets().set_index("fips_code")


def test_basket_is_large_enough() -> None:
    assert len(iet.EXTERNAL_TARGETS) >= 40


def test_no_table_family_dominates() -> None:
    """The 28-target basket's defect was 71% one table. Cap this one at 6."""
    counts: dict[str, int] = {}
    for target in iet.EXTERNAL_TARGETS:
        counts[target.table] = counts.get(target.table, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1])
    assert worst[1] <= 6, f"{worst[0]} contributes {worst[1]} targets"


def test_every_target_has_a_circularity_verdict() -> None:
    verdicts = iet.TARGET_CIRCULARITY
    for target in iet.EXTERNAL_TARGETS:
        assert target.column in verdicts, f"{target.column} has no circularity verdict"
        assert verdicts[target.column] in {"clean", "ablated"}


@pytest.mark.parametrize(("column", "expected"), AUTAUGA_EXPECTED.items())
def test_autauga_reconciles(targets_frame, column: str, expected: float) -> None:
    actual = float(targets_frame.loc[AUTAUGA, column])
    assert actual == pytest.approx(expected, rel=1e-4)


def test_every_target_ships_a_standard_error(targets_frame) -> None:
    for target in iet.EXTERNAL_TARGETS:
        se = f"{target.column}_se"
        assert se in targets_frame.columns
        assert targets_frame[se].notna().sum() > 2500
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_external_targets.py -v -k "basket or circularity"
```

Expected: FAIL — `len(EXTERNAL_TARGETS)` is 5 and `TARGET_CIRCULARITY` does not exist.

- [ ] **Step 3: Add the circularity registry**

In `scripts/ingest_external_targets.py`, above `EXTERNAL_TARGETS`:

```python
# Every target's standing against the six pillars, decided before it was
# admitted. "clean" means no pillar measures the construct; "ablated" means a
# pillar column is close enough to restate it, and `analyze_external_target.py`
# must carry a matching `TARGET_RESTATEMENTS` entry that removes it from the
# design. Constructs a pillar measures outright are not listed because they were
# rejected rather than admitted:
#
#   - county unemployment rate     -> Source C measures it directly
#   - any sector employment share  -> Source B location quotients
#   - freight or logistics volume  -> Source D
#
TARGET_CIRCULARITY: dict[str, str] = {
    "broadband_rate": "clean",
    "median_household_income": "ablated",   # Source E wage_per_return_thousands
    "median_age": "ablated",                # Source F retirement_destination
    "median_home_value": "clean",
    "mean_commute_minutes": "clean",
    "per_capita_income": "ablated",         # Source E wage_per_return_thousands
    "median_family_income": "ablated",      # Source E wage_per_return_thousands
    "median_gross_rent": "clean",
    "median_contract_rent": "clean",
    "median_monthly_housing_cost": "ablated",  # Source F housing_stress
    "median_year_built": "clean",
    "mean_household_size": "clean",
    "owner_occupied_share": "clean",
    "poverty_rate": "ablated",              # Source F persistent_poverty
    "family_household_share": "clean",
    "single_unit_share": "clean",
    "bachelors_share": "ablated",           # Source F low_education
    "graduate_share": "ablated",            # Source F low_education
    "labor_force_participation": "ablated", # Source F low_employment
}
```

- [ ] **Step 4: Add the verified target entries**

Append to `EXTERNAL_TARGETS` in `scripts/ingest_external_targets.py`. Each entry follows the existing dataclass exactly:

```python
    ExternalTarget(
        column="per_capita_income",
        table="b19301",
        numerator="B19301_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="per capita income",
    ),
    ExternalTarget(
        column="median_family_income",
        table="b19113",
        numerator="B19113_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median family income",
    ),
    ExternalTarget(
        column="median_gross_rent",
        table="b25064",
        numerator="B25064_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median gross rent",
    ),
    ExternalTarget(
        column="median_contract_rent",
        table="b25058",
        numerator="B25058_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median contract rent",
    ),
    ExternalTarget(
        column="median_monthly_housing_cost",
        table="b25105",
        numerator="B25105_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median monthly owner housing cost",
    ),
    ExternalTarget(
        column="median_year_built",
        table="b25035",
        numerator="B25035_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median year structure built",
    ),
    ExternalTarget(
        column="mean_household_size",
        table="b25010",
        numerator="B25010_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="mean household size",
    ),
    ExternalTarget(
        column="owner_occupied_share",
        table="b25003",
        numerator="B25003_E002",
        denominator="B25003_E001",
        denominator_table=None,
        kind="proportion",
        label="owner-occupied housing share",
    ),
    ExternalTarget(
        column="poverty_rate",
        table="b17001",
        numerator="B17001_E002",
        denominator="B17001_E001",
        denominator_table=None,
        kind="proportion",
        label="share below the poverty line",
    ),
    ExternalTarget(
        column="family_household_share",
        table="b11001",
        numerator="B11001_E002",
        denominator="B11001_E001",
        denominator_table=None,
        kind="proportion",
        label="family-household share",
    ),
    ExternalTarget(
        column="single_unit_share",
        table="b25024",
        numerator="B25024_E002",
        denominator="B25024_E001",
        denominator_table=None,
        kind="proportion",
        label="single-unit detached housing share",
    ),
    ExternalTarget(
        column="bachelors_share",
        table="b15003",
        numerator="B15003_E022",
        denominator="B15003_E001",
        denominator_table=None,
        kind="proportion",
        label="bachelor's degree share, age 25+",
    ),
    ExternalTarget(
        column="graduate_share",
        table="b15003",
        numerator="B15003_E023",
        denominator="B15003_E001",
        denominator_table=None,
        kind="proportion",
        label="graduate degree share, age 25+",
    ),
    ExternalTarget(
        column="labor_force_participation",
        table="b23025",
        numerator="B23025_E002",
        denominator="B23025_E001",
        denominator_table=None,
        kind="proportion",
        label="labour force participation rate",
    ),
```

- [ ] **Step 5: Extend to ≥40 using the gate as the admission rule**

The 14 above plus the original 5 is 19. Reach 40 by adding entries from the table families below, which are the same construct space at finer grain. For **each** candidate, run this probe before writing the dataclass entry — never write an entry from a guessed line number:

```bash
uv run python - <<'EOF'
import sys
sys.path.insert(0, "scripts")
from ingest_external_targets import _download_table

TABLE = "b25004"          # <-- the candidate table
LINES = ["E001", "E002"]  # <-- the candidate lines

frame = _download_table(TABLE)
for line in LINES:
    col = f"{TABLE.upper()}_{line}"
    if col not in frame.columns:
        print(f"{col}: ABSENT — reject")
        continue
    values = frame[col].where(frame[col] >= 0)
    print(f"{col}: n={values.notna().sum()} autauga={values.get('01001')}")
EOF
```

Admit the candidate only if county coverage exceeds 2,500 and the Autauga value is arithmetically plausible for the construct. Record the observed Autauga figure in `AUTAUGA_EXPECTED` as you go.

Candidate families, all within the eight constructs the spec requires and none measured by a pillar:
`b25004` (vacancy status), `b25040` (house heating fuel), `b08301` (means of transportation to work), `b05002` (nativity), `b07001` (geographic mobility), `b19052`/`b19055` (earnings and social-security receipt), `b25081` (mortgage status), `b28003` (computer availability), `b09002` (own children by family type).

Stop at 40 or when the eight-construct spread is satisfied, whichever is later.

- [ ] **Step 6: Run the full gate**

```bash
rm -f data/external_targets.parquet
uv run python scripts/ingest_external_targets.py
uv run pytest tests/test_external_targets.py -v
```

Expected: all pass. A failing `test_autauga_reconciles` means a line number is wrong — fix the mapping, never the expectation.

- [ ] **Step 7: Commit**

```bash
git add scripts/ingest_external_targets.py tests/test_external_targets.py
git commit -m "feat(acs): widen the external basket to 40+ non-circular targets"
```

---

### Task 4: Close the text-leakage channel

Wikipedia census sections state the targets verbatim — `The median age was 38.9 years` — and 2,589 counties carry one mentioning median income. The encoder reads this; the typed block cannot. Nothing currently guards it.

**Files:**
- Create: `scripts/source_a_text_leakage.py`
- Modify: `scripts/analyze_external_target.py`
- Create: `tests/test_text_scopes.py`

**Interfaces:**
- Consumes: `EXTERNAL_TARGETS`, `TARGET_CIRCULARITY` from Task 3.
- Produces:
  - `CENSUS_TITLE_PATTERN: str` — full-match regex over lowercased section titles.
  - `LIST_TITLE_PATTERN: str`
  - `HIGHWAY_TITLE_PATTERN: str`
  - `PROSE_EXCLUDE_PATTERN: str` — the three above alternated, plus `NARRATIVE_TITLE_PATTERN`.
  - `restated_targets(sections: pd.DataFrame) -> dict[str, int]` — target column to count of counties whose census sections restate it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_scopes.py`:

```python
"""Section-scope rules and the text-leakage screen."""
from __future__ import annotations

import re

import pytest

import source_a_text_leakage as leak


def _matches(pattern: str, title: str) -> bool:
    return re.match(pattern, title, flags=re.IGNORECASE) is not None


@pytest.mark.parametrize(
    "title",
    ["2020 census", "2010 census", "2000 census", "demographics",
     "racial and ethnic composition", "population ranking"],
)
def test_census_titles_are_matched(title: str) -> None:
    assert _matches(leak.CENSUS_TITLE_PATTERN, title)


@pytest.mark.parametrize("title", ["economy", "geography", "politics", "education"])
def test_substantive_titles_are_not_census(title: str) -> None:
    assert not _matches(leak.CENSUS_TITLE_PATTERN, title)


@pytest.mark.parametrize(
    "title",
    ["communities", "unincorporated communities", "cities", "towns",
     "townships", "ghost towns", "adjacent counties", "census-designated places"],
)
def test_list_titles_are_matched(title: str) -> None:
    assert _matches(leak.LIST_TITLE_PATTERN, title)


def test_economy_survives_the_prose_exclusion() -> None:
    """The scope must never drop the 1.5% of text it exists to preserve."""
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "economy")
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "agriculture")
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "industry")


def test_restatement_screen_finds_median_age(sections_frame) -> None:
    """median_age is stated verbatim in census sections; the screen must see it."""
    found = leak.restated_targets(sections_frame)
    assert found.get("median_age", 0) > 1000


def test_prose_scope_drops_at_least_a_third_of_characters(sections_frame) -> None:
    titles = sections_frame["section_title"].str.strip().str.lower()
    dropped = titles.str.match(leak.PROSE_EXCLUDE_PATTERN, na=False)
    share = sections_frame[dropped]["section_text"].str.len().sum() / (
        sections_frame["section_text"].str.len().sum()
    )
    assert share > 0.33, f"prose exclusion only drops {share:.1%} of characters"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_text_scopes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'source_a_text_leakage'`.

- [ ] **Step 3: Write the module**

Create `scripts/source_a_text_leakage.py`:

```python
"""Which Wikipedia sections restate the external targets, and how to drop them.

`analyze_external_target.py` already guards one direction of target restatement:
`TARGET_RESTATEMENTS` ablates pillar columns that define rather than predict a
target. Nothing guarded the other direction. Wikipedia census sections state the
targets in words -- "The median age was 38.9 years" -- and the MiniLM arms read
those sections while the typed block, which extracts lexicon counts and no
numbers, cannot.

Dropping census sections is therefore a leakage control rather than a tuning
choice, and it is a precondition of widening the external basket rather than an
arm to be scored against it.
"""
from __future__ import annotations

import re

import pandas as pd

from analyze_source_a_section_scope import NARRATIVE_TITLE_PATTERN

# Sections that render a census table as prose. These carry the target values
# verbatim and are 36.4% of all section characters -- about 42% of what the
# `uniform` arm reads, since `uniform` already excludes narrative titles.
CENSUS_TITLE_PATTERN: str = (
    r"^(?:(?:19|20)\d0 census|census|demographics|population|"
    r"racial and ethnic composition|population ranking|"
    r"race and ethnicity|income and poverty)$"
)

# Name lists. Near content-free for a sentence encoder, and `adjacent counties`
# additionally acts as a geographic identifier rather than an economic signal.
LIST_TITLE_PATTERN: str = (
    r"^(?:communities|cities|towns?|townships|villages?|city|village|"
    r"unincorporated communities|other unincorporated communities|"
    r"census-designated places?|ghost towns?|adjacent counties|"
    r"national protected areas?|protected areas|lakes|population ranking)$"
)

HIGHWAY_TITLE_PATTERN: str = (
    r"^(?:major highways|major roads|highways|transportation|"
    r"airports?|railroads?|transit)$"
)

# Everything the `prose_only` family removes. `NARRATIVE_TITLE_PATTERN` is
# included because `build_variant_texts` applies it to every variant today, so
# folding it in here keeps one exclusion rule rather than two.
PROSE_EXCLUDE_PATTERN: str = (
    r"^(?:"
    + r"|".join(
        p.removeprefix("^(?:").removesuffix(")$")
        for p in (
            CENSUS_TITLE_PATTERN,
            LIST_TITLE_PATTERN,
            HIGHWAY_TITLE_PATTERN,
            NARRATIVE_TITLE_PATTERN,
        )
    )
    + r")$"
)

# Phrases that restate a target in prose. Deliberately narrow: this flags a
# target as leakage-exposed, it does not attempt to parse the value.
RESTATEMENT_PHRASES: dict[str, tuple[str, ...]] = {
    "median_age": (r"median age",),
    "median_household_income": (r"median income for a household", r"median household income"),
    "median_family_income": (r"median income for a family", r"median family income"),
    "per_capita_income": (r"per capita income",),
    "poverty_rate": (r"below the poverty line", r"poverty line", r"poverty level"),
    "median_home_value": (r"median value of",  r"median home value"),
    "mean_household_size": (r"average household size", r"average family size"),
    "owner_occupied_share": (r"owner-occupied",),
}


def census_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Subset `sections` to the census-table sections.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Rows whose title matches `CENSUS_TITLE_PATTERN`.
    """
    titles = sections["section_title"].str.strip().str.lower()
    return sections[titles.str.match(CENSUS_TITLE_PATTERN, na=False)]


def restated_targets(sections: pd.DataFrame) -> dict[str, int]:
    """Count counties whose census sections restate each target.

    Args:
        sections: Long-format section frame.

    Returns:
        Target column to the number of distinct counties restating it. Targets
        with no configured phrase are absent from the mapping.
    """
    census = census_sections(sections)
    counts: dict[str, int] = {}
    for column, phrases in RESTATEMENT_PHRASES.items():
        pattern = "|".join(phrases)
        hit = census["section_text"].str.contains(pattern, case=False, na=False, regex=True)
        counts[column] = int(census[hit]["fips_code"].nunique())
    return counts
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_text_scopes.py -v
```

Expected: all pass. If `test_prose_scope_drops_at_least_a_third_of_characters` fails, a pattern is too narrow — print the unmatched high-volume titles and widen.

- [ ] **Step 5: Record the screen's output**

```bash
uv run python - <<'EOF'
import sys, json
sys.path.insert(0, "scripts")
import pandas as pd
from source_a_text_leakage import restated_targets

sections = pd.read_parquet("data/source_a_sections.parquet")
counts = restated_targets(sections)
print(json.dumps(counts, indent=2))
EOF
```

Paste the output into `docs/source_a_representation_decision.md` (created in Task 9) under a `Text-leakage screen` heading. Any target above 500 counties is tagged `restated_in_text`.

- [ ] **Step 6: Commit**

```bash
git add scripts/source_a_text_leakage.py tests/test_text_scopes.py
git commit -m "feat(source-a): screen external targets for restatement in article text"
```

---

### Task 5: Four new encoder scopes

**Files:**
- Modify: `scripts/analyze_source_a_tiered_embedding.py`
- Modify: `tests/test_text_scopes.py`

**Interfaces:**
- Consumes: `PROSE_EXCLUDE_PATTERN` from Task 4.
- Produces: `TextVariant` gains `exclude: str | None = None`; `TEXT_VARIANTS` gains `prose_only`, `prose_plus_history`, `economy_all_tiers`, `prose_by_tier`.

`build_variant_texts` currently hardcodes `NARRATIVE_TITLE_PATTERN` as the exclusion for every variant. That is why `prose_plus_history` **adds** text rather than trimming it: history is already excluded from `uniform` today.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_text_scopes.py`:

```python
import analyze_source_a_tiered_embedding as tiered


def test_new_variants_are_registered() -> None:
    keys = {variant.key for variant in tiered.TEXT_VARIANTS}
    assert {"prose_only", "prose_plus_history", "economy_all_tiers", "prose_by_tier"} <= keys


def test_variant_exclusion_defaults_to_narrative() -> None:
    """Existing variants must keep the behaviour they were measured under."""
    uniform = next(v for v in tiered.TEXT_VARIANTS if v.key == "uniform")
    assert uniform.exclude is None


def test_prose_plus_history_readmits_narrative() -> None:
    variant = next(v for v in tiered.TEXT_VARIANTS if v.key == "prose_plus_history")
    assert "history" not in variant.exclude


def test_prose_by_tier_reads_lead_for_thin_tiers() -> None:
    variant = next(v for v in tiered.TEXT_VARIANTS if v.key == "prose_by_tier")
    assert variant.tier_scope["stub"] is None
    assert variant.tier_scope["thin"] is None
    assert variant.tier_scope["mid"] is not None
    assert variant.tier_scope["rich"] is not None
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_text_scopes.py -v -k variant
```

Expected: FAIL — `TextVariant` has no attribute `exclude`.

- [ ] **Step 3: Add the exclusion field**

In `scripts/analyze_source_a_tiered_embedding.py`, extend the dataclass:

```python
@dataclass(frozen=True)
class TextVariant:
    """One rule for assembling a county's encoder input.

    Attributes:
        key: Short identifier used in result columns and stats.
        label: Human-readable description for reports.
        tier_scope: Tier to section-selection regex, or None for lead only.
        exclude: Full-match regex of titles to drop after `tier_scope` selects.
            None keeps the historical behaviour of excluding narrative titles,
            which is what every previously measured arm was scored under.
    """

    key: str
    label: str
    tier_scope: dict[str, str | None]
    exclude: str | None = None
```

And in `build_variant_texts`, replace the hardcoded exclusion:

```python
        exclude = variant.exclude if variant.exclude is not None else NARRATIVE_TITLE_PATTERN
        selected = select_sections(sections, pattern, exclude)
```

- [ ] **Step 4: Register the four scopes**

Add `from source_a_text_leakage import PROSE_EXCLUDE_PATTERN` to the imports, then append to `TEXT_VARIANTS`:

```python
    # Drops census tables, place lists and highway lists on top of the narrative
    # exclusion every arm already carries. This is the leakage-controlled arm:
    # census sections state the external targets verbatim.
    TextVariant(
        "prose_only",
        "lead + substantive prose, no census tables or name lists",
        {tier: ALL_TITLES for tier in ("stub", "thin", "mid", "rich")},
        exclude=PROSE_EXCLUDE_PATTERN,
    ),
    # `prose_only` plus history and notable people. The typed block excluded
    # narrative deliberately -- a lexicon hit inside History is usually a defunct
    # industry -- but that reasoning may not transfer to an encoder that reads the
    # surrounding sentence. History is 13.6% of all section characters.
    TextVariant(
        "prose_plus_history",
        "prose_only plus history and notable people",
        {tier: ALL_TITLES for tier in ("stub", "thin", "mid", "rich")},
        exclude=(
            r"^(?:"
            + r"|".join(
                p.removeprefix("^(?:").removesuffix(")$")
                for p in (CENSUS_TITLE_PATTERN, LIST_TITLE_PATTERN, HIGHWAY_TITLE_PATTERN)
            )
            + r")$"
        ),
    ),
    # The clean signal-only endpoint. Economy-titled sections are 1.5% of the
    # text and exist for only 660 of 3,144 counties, so 79% of counties fall back
    # to their lead. Expected weak; measured rather than assumed.
    TextVariant(
        "economy_all_tiers",
        "economy-titled sections everywhere, lead fallback",
        {tier: ECONOMY_TITLE_PATTERN for tier in ("stub", "thin", "mid", "rich")},
    ),
    # Branches on content availability rather than on depth, which is what
    # separates it from `tier_conditional` and `tier_conditional_inverse`. Both of
    # those branched on depth and both lost. A stub county's body is almost
    # entirely census table and place list, so there is no prose there to read.
    TextVariant(
        "prose_by_tier",
        "rich and mid read substantive prose; stub and thin read their lead",
        {"stub": None, "thin": None, "mid": ALL_TITLES, "rich": ALL_TITLES},
        exclude=PROSE_EXCLUDE_PATTERN,
    ),
```

Add `CENSUS_TITLE_PATTERN, LIST_TITLE_PATTERN, HIGHWAY_TITLE_PATTERN` to the `source_a_text_leakage` import.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_text_scopes.py -v
```

Expected: all pass.

- [ ] **Step 6: Verify the existing arms did not move**

```bash
uv run python scripts/analyze_source_a_tiered_embedding.py
uv run pytest tests/test_regression_locks.py -v
```

Expected: the Task 1 locks still pass. If `uniform` moved, the exclusion default is wrong — it must resolve to `NARRATIVE_TITLE_PATTERN` when `exclude is None`.

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_source_a_tiered_embedding.py tests/test_text_scopes.py
git commit -m "feat(source-a): add four content-selected encoder scopes"
```

---

### Task 6: Common-component removal

All 3,144 articles open with the same template sentence, so the mean-pooled vectors share a large common component that consumes representational budget.

**Files:**
- Modify: `scripts/analyze_source_a_tiered_embedding.py`

**Interfaces:**
- Produces: `remove_common_component(vectors: np.ndarray) -> np.ndarray` — subtracts the corpus mean row, then L2-normalizes. Scored as a `_ccr` suffixed block alongside `_l2`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_text_scopes.py`:

```python
import numpy as np


def test_common_component_removal_centres_the_corpus() -> None:
    rng = np.random.default_rng(42)
    shared = np.ones(8) * 5.0
    vectors = shared + rng.normal(scale=0.1, size=(100, 8))

    result = tiered.remove_common_component(vectors)

    assert result.shape == vectors.shape
    assert np.abs(result.mean(axis=0)).max() < 0.2, "shared component should be gone"
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_common_component_removal_handles_a_zero_row() -> None:
    """Counties with no text get a zero vector; it must not produce NaN."""
    vectors = np.vstack([np.ones((5, 4)), np.zeros((1, 4))])
    result = tiered.remove_common_component(vectors)
    assert np.isfinite(result).all()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_text_scopes.py -v -k common_component
```

Expected: FAIL — no attribute `remove_common_component`.

- [ ] **Step 3: Implement it**

In `scripts/analyze_source_a_tiered_embedding.py`, beside `l2_normalize`:

```python
def remove_common_component(vectors: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean vector, then row-normalize.

    Every county article opens with the same template sentence, so mean-pooled
    vectors share a large component that carries no between-county information
    while consuming representational budget. `StandardScaler` in the scoring
    pipeline centres each dimension across counties, which is not the same
    operation: this removes a shared *direction* before normalization, so the
    row norms that survive describe deviation from the corpus rather than
    absolute position.

    Args:
        vectors: One row per county.

    Returns:
        Centred, row-normalized vectors of the same shape. Zero rows stay zero
        rather than becoming NaN.
    """
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    return l2_normalize(centred)
```

Then in `main`, alongside the existing `_l2` block registration:

```python
        blocks[f"{variant.key}_ccr"] = (remove_common_component(vectors), None)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_text_scopes.py -v -k common_component
```

Expected: PASS.

- [ ] **Step 5: Re-run the sweep and check the locks**

```bash
uv run python scripts/analyze_source_a_tiered_embedding.py
uv run pytest tests/test_regression_locks.py -v
```

Expected: locks pass; the results CSV gains `_ccr` rows.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_source_a_tiered_embedding.py tests/test_text_scopes.py
git commit -m "feat(source-a): score a common-component-removed embedding block"
```

---

### Task 7: Width-matched PCA-29 arm in the marginal harness

The 700× gap in §21 cannot be attributed to content while the embedding carries 384 columns against the typed block's 29. §21.2 states this in writing.

**Files:**
- Modify: `scripts/analyze_source_a_representation_marginal.py`

**Interfaces:**
- Consumes: `build_source_a_embedding(fips_order) -> dict[str, np.ndarray]`.
- Produces: that mapping gains `minilm_uniform_pca29` and `minilm_uniform_pca64` keys.

PCA must be fitted **inside** the cross-validation, not on the full matrix, or the reduction sees held-out states and inflates the very arm it exists to control.

`out_of_fold_predictions` (`analyze_external_target.py:378`) delegates to `cross_val_predict`, which gives no hook for reducing one column group and not another. `_pipeline()` (`analyze_external_target.py:358`) is an `impute -> scale -> RidgeCV` `Pipeline`, and PCA cannot simply be prepended to it: the imputer exists because BLS suppresses a large share of the pillar columns, and those NaNs must reach it unreduced. The embedding columns carry no NaN — counties with no text get a zero row.

So the reduction is applied to the embedding block only, per fold, before the hstack. Step 4 spells this out; no branch is left to the implementer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_marginal_arms.py`:

```python
"""Arms entering the marginal decision."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyze_source_a_representation_marginal as marginal


def test_pca_arms_are_declared() -> None:
    assert "minilm_uniform_pca29" in marginal.EMBEDDING_ARMS
    assert "minilm_uniform_pca64" in marginal.EMBEDDING_ARMS


def test_pca_widths_match_their_names() -> None:
    assert marginal.EMBEDDING_ARMS["minilm_uniform_pca29"] == 29
    assert marginal.EMBEDDING_ARMS["minilm_uniform_pca64"] == 64


def test_reduce_fits_only_on_training_rows() -> None:
    """A reduction fitted on all rows leaks held-out states into the design."""
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 10))
    test = rng.normal(size=(20, 10)) + 100.0

    fitted = marginal.fit_reduction(train, n_components=3)
    reduced_test = fitted.transform(test)

    assert reduced_test.shape == (20, 3)
    assert np.abs(reduced_test).max() > 10, "test rows should sit far from the training centre"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_marginal_arms.py -v
```

Expected: FAIL — no attribute `EMBEDDING_ARMS`.

- [ ] **Step 3: Implement the arm registry and the reduction**

In `scripts/analyze_source_a_representation_marginal.py`:

```python
from sklearn.decomposition import PCA

# Embedding arms entering the marginal comparison, and the width each is reduced
# to. `None` means the native 384 dimensions. The 29-dimension arm exists so the
# comparison against the typed block is width-matched: findings §21.2 states that
# an unknown share of the embedding's penalty is width rather than content, and
# names this arm as the missing control.
EMBEDDING_ARMS: dict[str, int | None] = {
    "minilm_uniform": None,
    "minilm_uniform_l2": None,
    "minilm_uniform_pca29": 29,
    "minilm_uniform_pca64": 64,
}


def fit_reduction(train_vectors: np.ndarray, n_components: int) -> PCA:
    """Fit a PCA reduction on training rows only.

    Fitting on the full matrix would let the reduction see the held-out states
    the fold is scored on, which inflates the arm it is meant to control.

    Args:
        train_vectors: Rows in the fold's training split.
        n_components: Target width.

    Returns:
        The fitted reducer, ready to transform both splits.
    """
    reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reducer.fit(train_vectors)
    return reducer
```

Import `RANDOM_SEED` from `analyze_pillar_matrix_signal` at the top of the module.

- [ ] **Step 4: Wire the reduction into the fold loop**

Add a sibling to `out_of_fold_predictions` that reduces the embedding block inside each fold and leaves the pillar block untouched, so the imputer still sees the suppressed BLS cells:

```python
def out_of_fold_predictions_with_reduction(
    size_and_others: np.ndarray,
    vectors: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Out-of-fold predictions with the embedding reduced inside each fold.

    Args:
        size_and_others: The reduced-model design, unreduced.
        vectors: Full-width embedding rows aligned to `size_and_others`.
        y: Target values.
        groups: State FIPS per row.
        n_components: Width to reduce the embedding to.

    Returns:
        Out-of-fold predictions, one per row.
    """
    from sklearn.model_selection import GroupKFold

    predictions = np.zeros_like(y, dtype=float)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for train_idx, test_idx in splitter.split(size_and_others, y, groups):
        reducer = fit_reduction(vectors[train_idx], n_components)
        design_train = np.hstack([size_and_others[train_idx], reducer.transform(vectors[train_idx])])
        design_test = np.hstack([size_and_others[test_idx], reducer.transform(vectors[test_idx])])
        model = _pipeline()
        model.fit(design_train, y[train_idx])
        predictions[test_idx] = model.predict(design_test)
    return predictions
```

Import `N_FOLDS` and `_pipeline` from `analyze_external_target`. In `score_representation`, route arms whose `EMBEDDING_ARMS` width is not `None` through this function instead of building a design up front.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_marginal_arms.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_source_a_representation_marginal.py tests/test_marginal_arms.py
git commit -m "feat(source-a): add width-matched PCA arms to the marginal harness"
```

---

### Task 8: The pre-registered typed capacity pass

Both arms are scored under ridge, so 29 raw columns against 384 dense dimensions is not an equal-capacity comparison. This equalizes it; it does not tilt it.

**Files:**
- Create: `scripts/source_a_typed_transform.py`
- Modify: `tests/test_marginal_arms.py`

**Interfaces:**
- Produces: `transform_typed(frame: pd.DataFrame, typed_columns: list[str], tier: pd.Series) -> tuple[np.ndarray, list[str]]` — returns the expanded design and its column names.

The transform is fixed here, chosen from how the columns are constructed rather than from their scores. No target is consulted.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_marginal_arms.py`:

```python
import source_a_typed_transform as typed_transform


def test_counts_are_log_transformed() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [0.0, 9.0], "has_economy_section": [0.0, 1.0]})
    tier = pd.Series(["stub", "rich"])

    design, names = typed_transform.transform_typed(
        frame, ["sec_n_industry_mentions", "has_economy_section"], tier
    )

    assert "log1p_sec_n_industry_mentions" in names
    column = design[:, names.index("log1p_sec_n_industry_mentions")]
    assert column[0] == pytest.approx(0.0)
    assert column[1] == pytest.approx(np.log1p(9.0))


def test_binary_columns_are_not_log_transformed() -> None:
    frame = pd.DataFrame({"has_economy_section": [0.0, 1.0]})
    tier = pd.Series(["stub", "rich"])
    _, names = typed_transform.transform_typed(frame, ["has_economy_section"], tier)
    assert "log1p_has_economy_section" not in names


def test_industry_mentions_interacts_with_tier() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [2.0, 4.0]})
    tier = pd.Series(["stub", "rich"])
    design, names = typed_transform.transform_typed(frame, ["sec_n_industry_mentions"], tier)

    assert "sec_n_industry_mentions_x_rich" in names
    column = design[:, names.index("sec_n_industry_mentions_x_rich")]
    assert column[0] == pytest.approx(0.0)
    assert column[1] == pytest.approx(4.0)


def test_original_columns_survive() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [2.0, 4.0]})
    tier = pd.Series(["stub", "rich"])
    _, names = typed_transform.transform_typed(frame, ["sec_n_industry_mentions"], tier)
    assert "sec_n_industry_mentions" in names
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_marginal_arms.py -v -k typed
```

Expected: FAIL — `ModuleNotFoundError: No module named 'source_a_typed_transform'`.

- [ ] **Step 3: Implement it**

Create `scripts/source_a_typed_transform.py`:

```python
"""The pre-registered capacity pass over Source A's typed columns.

Both representations are scored under ridge, so 29 raw columns against 384 dense
dimensions is not an equal-capacity comparison. A typed win under that setup
would be partly an artifact of the encoder's extra flexibility, and a typed loss
would be partly an artifact of the typed block's rigidity.

The two transforms here are fixed before any decision-basket target is scored,
and are chosen from how the columns are constructed rather than from what they
predict:

1. **`log1p` on count columns.** Lexicon counts are bounded below at zero and
   right-skewed; ridge fits a linear coefficient to them. This is the standard
   remedy and needs no justification from the data.
2. **`sec_n_industry_mentions` x tier.** That single column carries 97.6% of the
   section block's gain (`source_a_next_steps.md`), and tier is defined by
   article length, so the same count means something different in a stub and in a
   rich article. The interaction says so explicitly.

Nothing else is added. This is a capacity control, not a feature search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Tiers that get an interaction term. `stub` is the reference level and is
# omitted, as a dummy-coded set must be to stay full rank alongside the main
# effect.
INTERACTION_TIERS: tuple[str, ...] = ("thin", "mid", "rich")

# The column whose tier interaction is pre-registered.
INTERACTION_COLUMN: str = "sec_n_industry_mentions"


def _is_count_column(values: pd.Series) -> bool:
    """Decide whether a column is a count rather than a flag or a ratio.

    Args:
        values: The column.

    Returns:
        True when the column is non-negative and takes more than two values.
    """
    finite = values.dropna()
    if finite.empty:
        return False
    return bool((finite >= 0).all() and finite.nunique() > 2)


def transform_typed(
    frame: pd.DataFrame, typed_columns: list[str], tier: pd.Series
) -> tuple[np.ndarray, list[str]]:
    """Expand the typed block with its pre-registered transforms.

    Args:
        frame: Rows carrying every column in `typed_columns`.
        typed_columns: Source A's shipped column names.
        tier: Tier label per row, aligned to `frame`.

    Returns:
        The expanded design and its column names, in matching order.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []

    for column in typed_columns:
        values = frame[column].astype(float)
        columns.append(values.to_numpy())
        names.append(column)
        if _is_count_column(values):
            columns.append(np.log1p(values.clip(lower=0.0)).to_numpy())
            names.append(f"log1p_{column}")

    if INTERACTION_COLUMN in typed_columns:
        base = frame[INTERACTION_COLUMN].astype(float).to_numpy()
        tier_values = tier.to_numpy()
        for label in INTERACTION_TIERS:
            columns.append(np.where(tier_values == label, base, 0.0))
            names.append(f"{INTERACTION_COLUMN}_x_{label}")

    return np.column_stack(columns), names
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_marginal_arms.py -v -k typed
```

Expected: PASS.

- [ ] **Step 5: Wire it into the marginal harness as a `typed_transformed` arm**

In `score_representation` in `scripts/analyze_source_a_representation_marginal.py`, beside the existing `typed` design:

```python
        transformed, _ = transform_typed(usable, typed, usable["tier"])
        designs["typed_transformed"] = np.hstack([size_and_others, transformed])
```

`usable` must carry a `tier` column; add it in `load_panel`'s caller by merging `assign_tiers(matrix["content_length"])` if it is absent.

- [ ] **Step 6: Commit**

```bash
git add scripts/source_a_typed_transform.py scripts/analyze_source_a_representation_marginal.py tests/test_marginal_arms.py
git commit -m "feat(source-a): pre-registered capacity pass over the typed block"
```

---

### Task 9: Write the decision rule, then select the scope

This task must complete **before** Task 10 runs. Its whole purpose is that the rule exists in git before any decision-basket number is seen.

**Files:**
- Create: `docs/source_a_representation_decision.md`

**Interfaces:**
- Consumes: `outputs/source_a_tiered_embedding.csv`, regenerated in Tasks 5 and 6.
- Produces: the named winning scope, recorded in the document.

- [ ] **Step 1: Write the decision document with the rule and an empty results section**

Create `docs/source_a_representation_decision.md` containing, verbatim, the five rules from the spec's Part 5, plus the Task 4 leakage-screen counts. Leave the results section headed and empty.

- [ ] **Step 2: Commit it before looking at anything**

```bash
git add docs/source_a_representation_decision.md
git commit -m "docs(source-a): pre-register the representation decision rule"
```

This commit is the pre-registration. It must land before Step 3.

- [ ] **Step 3: Select the embedding scope on the 28-target basket only**

```bash
uv run python - <<'EOF'
import pandas as pd

results = pd.read_csv("outputs/source_a_tiered_embedding.csv")
family = results[results["representation"].str.startswith(("prose_only", "prose_plus_history", "economy_all_tiers", "prose_by_tier"))]
print(family.groupby("representation")["lift"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False).round(6).to_string())
EOF
```

The highest mean lift is the embedding representative. Record its name in the decision document under `Selected scope`, with the table above pasted beneath it.

This basket is the in-repo 28. The decision basket is the external 40+. They share no targets, so selection cannot contaminate the decision.

- [ ] **Step 4: Commit the selection**

```bash
git add docs/source_a_representation_decision.md
git commit -m "docs(source-a): record the selected embedding scope"
```

---

### Task 10: Run the decision and write it up

**Files:**
- Modify: `scripts/analyze_source_a_representation_marginal.py`
- Modify: `docs/source_a_representation_decision.md`
- Modify: `analysis-output/source-a/source-a-findings.md`

- [ ] **Step 1: Point the marginal harness at the selected scope**

Change `VARIANT_KEY` in `scripts/analyze_source_a_representation_marginal.py` from `"uniform"` to the scope selected in Task 9, and add the selected scope's `_pca29` reduction to `EMBEDDING_ARMS`. Keep `minilm_uniform_pca29` registered as the unselected reference — it is the width-matched twin of the leakage-carrying arm.

- [ ] **Step 2: Run the harness**

```bash
uv run python scripts/analyze_source_a_representation_marginal.py 2>&1 | tail -40
```

- [ ] **Step 3: Report the primary comparison both ways**

```bash
uv run python - <<'EOF'
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from source_a_text_leakage import restated_targets

scores = pd.read_csv("outputs/source_a_representation_marginal.csv")
sections = pd.read_parquet("data/source_a_sections.parquet")
exposed = {k for k, v in restated_targets(sections).items() if v > 500}

for label, subset in [
    ("full basket", scores),
    ("leakage-clean subset", scores[~scores["target"].isin(exposed)]),
]:
    table = subset.groupby("representation")["contribution"].agg(["mean", "median", "count"])
    print(f"\n=== {label} ===")
    print(table.sort_values("mean", ascending=False).round(6).to_string())
EOF
```

- [ ] **Step 4: Apply the pre-registered rule**

Compute the paired difference between `typed_transformed` and the selected `_pca29` arm across the decision basket, with a bootstrap 95% CI. If the CI includes zero, the verdict is a tie and the decision falls back to cost and interpretability — write that in those words, per rule 4.

- [ ] **Step 5: Write the findings section**

Add a new numbered section to `analysis-output/source-a/source-a-findings.md` following the file's established shape: what was asked, what was measured, the table, then explicit **Allowed wording** and **Forbidden wording** bullets. Forbidden wording must include any claim that ignores the leakage split, and any figure quoted without its basket size.

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run pytest -v
git add -A
git commit -m "feat(source-a): settle the representation question on the external basket"
```

---

## Self-Review

**Spec coverage.** Part 1 → Tasks 2, 3. Part 2 → Task 7. Part 3 → Tasks 5, 6. Part 4 → Task 8. Part 5 → Tasks 9, 10. The spec's leakage screen, which sits inside Part 1, → Task 4. Verification section → Task 1's regression locks plus the per-task gates.

**Gap found and closed.** The spec's verification list requires that "every proportion target reconciles against Autauga County, AL" — Task 3 Step 1 now encodes that as `test_autauga_reconciles` with real observed values rather than leaving it to reviewer discipline.

**Type consistency.** `remove_common_component`, `fit_reduction`, `transform_typed`, `restated_targets`, `EMBEDDING_ARMS`, `PROSE_EXCLUDE_PATTERN` are each defined once and referenced under the same name in every later task.

**Resolved during review.** An earlier draft of Task 7 asked the implementer to read `_pipeline()` and decide how to insert PCA. That has been checked and settled in the plan instead: `_pipeline()` is an `impute -> scale -> RidgeCV` Pipeline, and PCA must **not** be prepended to it, because the imputer exists to receive the suppressed BLS cells unreduced. The reduction applies to the embedding block only, per fold. No branch is left open.

**Residual risk, stated rather than hidden.** Task 3 Step 5 reaches 40 targets by admitting candidates through a probe rather than by listing all 40 here. That is deliberate — line numbers guessed at planning time are exactly the error class the Autauga gate exists to catch — but it means the final basket composition is decided during execution. The gate tests (`test_basket_is_large_enough`, `test_no_table_family_dominates`, `test_every_target_has_a_circularity_verdict`) constrain it mechanically, so the discretion is bounded.
