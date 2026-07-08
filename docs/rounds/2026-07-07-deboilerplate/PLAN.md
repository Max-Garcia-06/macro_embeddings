# Source A Embedding Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the boilerplate-driven false similarity documented in `analysis-output/source-a-findings.md` §3.4 by extending text cleaning, re-embedding offline from the stored `raw_intro_text`, and adopting the best variant only if it passes a pre-registered evaluation gate.

**Architecture:** Two candidate cleaning variants are built on top of the existing `strip_self_reference` → `strip_boilerplate_phrasing` pipeline: **v2** adds three targeted regex patterns (eponym clauses, metro/micropolitan-area sentences, formation connectives — the exact mechanisms behind the findings' top-5 far-but-similar pairs), and **v3** adds a corpus-frequency sentence filter on top of v2 (drop any sentence whose number/proper-noun-masked "template" appears in ≥5% of counties). Both variants re-embed the 3,144 stored intro texts with the same `BAAI/bge-m3` model — no Wikimedia API access needed. A fixed evaluation harness compares each variant against the current baseline on the same county set, and a decision gate determines adoption.

**Tech Stack:** Python ≥3.12, uv, sentence-transformers (`BAAI/bge-m3`), pandas/pyarrow, numpy, scikit-learn, pytest (new, dev-only).

## Global Constraints

- Python `>=3.12` (from `pyproject.toml`); all commands run via `uv run`.
- `RANDOM_SEED = 42` everywhere; Mantel test uses 499 permutations (matches `analyze_source_a_clusters.py`).
- **No Wikimedia API calls.** All variants re-clean and re-embed from the `raw_intro_text` column already stored in `source_a_embeddings.parquet` (3,144 rows).
- Embedding model: `BAAI/bge-m3`, 1024-dim, L2-normalized, `device="cpu"` (README notes CPU beats MPS for this model on short inputs), single-text `encode()` loop matching `ingest_source_a.py`.
- Parquet schema is additive-only: existing columns `county_name`, `fips_code`, `raw_intro_text`, `embedding` keep their names/types; variants add `embedding_text`.
- `pytest` is added to the `dev` dependency group only — no new runtime dependencies.
- Conventional Commits; end every commit message with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- All functions have type hints and Google-style docstrings; module-level `logger = logging.getLogger(__name__)`; no `print()` in library code.
- Do not re-open closed directions: no section-expansion variants (findings §4), no claims that clusters are "regions" (findings §3.6).

## Scope and Assumptions (read first)

1. **"Improve" is defined as measurable de-boilerplating**, per the findings' own next-lever recommendation (§8.3, §10.2): kill the shared-eponym and generic-template similarity mechanisms while preserving the corpus's one confirmed signal (weak negative Mantel correlation). Success criteria are pre-registered in the Decision Gate below.
2. **Economic ground-truth validation is out of scope.** The proposal's real claim (embedding distance ↔ economic variables) needs Sources E/B, which are not in-repo (findings §8.5). This plan improves the text→embedding pipeline; it cannot validate economic signal.
3. **Better differentiation metrics are a proxy, not proof.** The plan's gate guards against making things *worse*; whether the adopted variant helps E_macro is decided later against Source E/B.

## Decision Gate (pre-registered, used in Task 7)

A candidate variant **passes** iff, on the fixed analysis county set:

1. `tracked_pair_mean` (mean cosine similarity of the 5 tracked boilerplate pairs from findings §3.4) drops by **≥ 0.03** vs. baseline;
2. `pairwise_similarity_std` is **≥ baseline** (differentiation did not shrink);
3. Mantel `r < 0` with `p < 0.05` (the one real signal is preserved).

If both v2 and v3 pass, adopt the one with the lower `tracked_pair_mean` (tiebreak: higher `pairwise_similarity_std`). If neither passes, keep the baseline and record the negative result in the findings doc — that outcome is a valid deliverable.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `text_cleaning.py` | **Create** | All pure-text/HTML cleaning: existing regexes + functions moved out of `ingest_source_a.py`, the three new boilerplate patterns, and `clean_for_embedding` (full offline cleaning with empty-result fallback). No network, no model, importable by tests without side effects. |
| `boilerplate_frequency.py` | **Create** | Corpus-frequency sentence filter: sentence splitting, template masking, common-template detection, sentence dropping with fallback. |
| `reembed_source_a.py` | **Create** | Offline re-embedding CLI: parquet → clean (v2/v3) → bge-m3 → `source_a_embeddings_{variant}.parquet` with `embedding_text` column. |
| `evaluate_source_a_variants.py` | **Create** | Evaluation harness: baseline vs. variants on a fixed county set → printed table + `analysis-output/variant-eval.json`. |
| `ingest_source_a.py` | **Modify** | Remove the moved preprocessing block; import (and thereby re-export) the cleaning functions from `text_cleaning` so `from ingest_source_a import strip_self_reference, strip_boilerplate_phrasing` keeps working for `analyze_source_a_similarity.py`. |
| `tests/test_text_cleaning.py` | **Create** | Characterization tests for existing cleaning + TDD tests for new patterns and fallback. |
| `tests/test_boilerplate_frequency.py` | **Create** | TDD tests for the frequency filter. |
| `tests/test_evaluate_variants.py` | **Create** | Unit test for `evaluate_variant` on a synthetic fixture. |
| `pyproject.toml` | **Modify** | Add `pytest` to `[dependency-groups] dev`. |
| `analysis-output/source-a-findings.md` | **Modify (Task 7)** | Round-4 section recording the experiment outcome. |

Why the extraction: `ingest_source_a.py` is 790 lines (the 800-line hard cap leaves no room for new patterns), and importing it executes `fetch_county_crosswalk` at module scope — tests of pure text functions shouldn't depend on that. Moving the preprocessing block is the minimal change that fixes both; everything else in `ingest_source_a.py` is untouched.

**Interfaces between new modules (single source of truth):**

```python
# text_cleaning.py
def clean_intro_text(article_html: str) -> str                       # "" when empty (was: raised)
def strip_self_reference(text: str, county_name: str) -> str
def strip_boilerplate_phrasing(text: str) -> str
def clean_for_embedding(raw_intro_text: str, county_name: str) -> str

# boilerplate_frequency.py
DEFAULT_MIN_COUNTY_FRACTION: float = 0.05
def split_sentences(text: str) -> list[str]
def mask_sentence_template(sentence: str) -> str
def find_common_templates(texts: list[str], min_fraction: float = DEFAULT_MIN_COUNTY_FRACTION) -> set[str]
def drop_common_sentences(text: str, common_templates: set[str]) -> str

# evaluate_source_a_variants.py
def evaluate_variant(variant_df: pd.DataFrame, analysis_df: pd.DataFrame, distance_matrix: np.ndarray) -> dict
```

---

### Task 1: Test scaffold + extract `text_cleaning.py`

**Files:**
- Modify: `pyproject.toml` (dev group)
- Create: `tests/test_text_cleaning.py`
- Create: `text_cleaning.py` (moved from `ingest_source_a.py:376-538`)
- Modify: `ingest_source_a.py` (remove moved block, import from `text_cleaning`, inline the empty-text raise in `process_county`)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `text_cleaning.clean_intro_text(article_html: str) -> str` (returns `""` when empty instead of raising), `strip_self_reference(text: str, county_name: str) -> str`, `strip_boilerplate_phrasing(text: str) -> str`, plus `isolate_lead_section` / `strip_non_narrative_elements`. `ingest_source_a` re-exports the three public names so existing importers are unaffected.

One deliberate behavior change: `clean_intro_text` currently raises `EmptyIntroError` (defined in `ingest_source_a.py`) on empty output. To avoid a circular import, the moved version returns `""` and `process_county` raises `EmptyIntroError` at the call site. `run_pipeline` catches at the same boundary, so observable pipeline behavior is identical. The backfill scripts only import `run_pipeline`-level names (verified), so they are unaffected.

- [ ] **Step 1: Add pytest to the dev group**

```bash
uv add --group dev pytest
```

- [ ] **Step 2: Write the tests (characterization + new contract)**

Create `tests/test_text_cleaning.py`:

```python
"""Tests for text_cleaning: characterization of existing behavior + new patterns."""

from text_cleaning import (
    clean_intro_text,
    strip_boilerplate_phrasing,
    strip_self_reference,
)

# Real corpus text (Lincoln County, Kansas — source_a_embeddings.parquet).
LINCOLN_KS_INTRO = (
    "County in Kansas, United States County in Kansas Lincoln County is a "
    "county located in the U.S. state of Kansas . Its county seat and largest "
    "city is Lincoln Center . As of the 2020 census , the county population "
    "was 2,939. The county was named after Abraham Lincoln , the 16th "
    "president of the United States."
)


class TestExistingBehavior:
    def test_self_reference_removes_name_state_and_breadcrumb(self) -> None:
        out = strip_self_reference(LINCOLN_KS_INTRO, "Lincoln County, Kansas")
        assert "Lincoln County" not in out
        assert "Kansas" not in out
        assert not out.startswith("County in")  # breadcrumb before first mention dropped
        assert "Lincoln Center" in out  # unrelated proper noun kept

    def test_census_clause_stripped_value_kept(self) -> None:
        out = strip_boilerplate_phrasing(
            "As of the 2020 census , the county population was 2,939."
        )
        assert "As of the 2020 census" not in out
        assert "2,939" in out

    def test_county_seat_clause_stripped(self) -> None:
        out = strip_boilerplate_phrasing("Its county seat and largest city is Lincoln Center .")
        assert "county seat" not in out
        assert "Lincoln Center" in out

    def test_clean_intro_text_returns_empty_string_when_no_content(self) -> None:
        html = '<section data-mw-section-id="0"><table><tr><td>x</td></tr></table></section>'
        assert clean_intro_text(html) == ""
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_text_cleaning.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'text_cleaning'`.

- [ ] **Step 4: Create `text_cleaning.py` by moving the preprocessing block**

Cut lines 376–538 of `ingest_source_a.py` (everything from `_WIKI_LINK_PATTERN` through the end of `strip_boilerplate_phrasing`, **excluding** `extract_article_html`, which stays — it reads the API payload shape) into a new `text_cleaning.py`:

```python
"""Pure text/HTML cleaning for Source A county intro texts.

Extracted from ingest_source_a.py so cleaning can be tested and iterated on
without importing the ingestion module (whose import triggers the county
crosswalk load). No network access, no model loading.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# ... paste the moved regex constants and functions here unchanged, with ONE
# edit: clean_intro_text's final lines become
#
#     if not text:
#         return ""
#     return text
#
# and its docstring's Raises: section is replaced by
#     Returns: Cleaned text, or "" if no narrative content remains.
```

The moved names: `_WIKI_LINK_PATTERN`, `_CITATION_BRACKET_PATTERN`, `_WHITESPACE_PATTERN`, `_LEADING_PUNCTUATION_PATTERN`, `_OPENING_TOPIC_SENTENCE_PATTERN`, `_CENSUS_CLAUSE_PATTERN`, `_COUNTY_SEAT_CLAUSE_PATTERN`, `isolate_lead_section`, `strip_non_narrative_elements`, `clean_intro_text`, `strip_self_reference`, `strip_boilerplate_phrasing`.

In `ingest_source_a.py`, replace the removed block with:

```python
from text_cleaning import (
    clean_intro_text,
    strip_boilerplate_phrasing,
    strip_self_reference,
)
```

(placed in the local-modules import group), remove the now-unused `re` and `BeautifulSoup` imports **if nothing else in the file uses them** (check: `extract_article_html` doesn't; nothing else does — remove both), and change `process_county`:

```python
    intro_text = clean_intro_text(article_html)
    if not intro_text:
        raise EmptyIntroError("Cleaned introduction text is empty.")
    embedding_text = strip_self_reference(intro_text, county_name)
```

- [ ] **Step 5: Run tests to verify they pass, plus import smoke checks**

```bash
uv run pytest tests/test_text_cleaning.py -v
uv run python -c "import ingest_source_a, analyze_source_a_similarity, analyze_source_a_clusters, backfill_virginia_cities, backfill_remaining_19; print('imports OK')"
```

Expected: 4 tests PASS; `imports OK`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_text_cleaning.py text_cleaning.py ingest_source_a.py
git commit -m "refactor(source-a): extract text cleaning into text_cleaning module"
```

---

### Task 2: New boilerplate patterns (eponym, metro-area, formation)

**Files:**
- Modify: `text_cleaning.py`
- Test: `tests/test_text_cleaning.py`

**Interfaces:**
- Consumes: `strip_boilerplate_phrasing` from Task 1.
- Produces: the same `strip_boilerplate_phrasing(text: str) -> str` signature, now also removing the three new pattern families. No new public names.

These target the two mechanisms behind all five far-but-similar pairs in findings §3.4: shared-eponym sentences ("named for/after Abraham Lincoln, 16th president…") and templated metro/micropolitan-area sentences ("X comprises the Y micropolitan statistical area"). The formation connective ("The county was established on") follows the module's existing philosophy: strip the shared connective, keep the county-specific value (the date). The eponym clause is removed *including* the person's name — the shared name token is precisely what creates false cross-country similarity.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_text_cleaning.py`)

```python
class TestNewBoilerplatePatterns:
    def test_eponym_clause_removed_including_name(self) -> None:
        text = (
            "The county was named after Abraham Lincoln , the 16th president "
            "of the United States."
        )
        out = strip_boilerplate_phrasing(text)
        assert "Abraham Lincoln" not in out
        assert "named" not in out

    def test_eponym_clause_removed_is_named_for_variant(self) -> None:
        text = "The county is named for Abraham Lincoln , 16th president of the United States ."
        out = strip_boilerplate_phrasing(text)
        assert "Abraham Lincoln" not in out

    def test_metro_area_sentence_removed_entirely(self) -> None:
        text = (
            "comprises the Jamestown micropolitan statistical area . "
            "The population was 21,593."
        )
        out = strip_boilerplate_phrasing(text)
        assert "micropolitan" not in out
        assert "Jamestown" not in out
        assert "21,593" in out

    def test_metropolitan_area_sentence_removed(self) -> None:
        text = "is included in the Montgomery metropolitan area ."
        assert strip_boilerplate_phrasing(text) == ""

    def test_formation_connective_removed_date_kept(self) -> None:
        text = "The county was established on May 9, 1838, and named for Benjamin Franklin ."
        out = strip_boilerplate_phrasing(text)
        assert "established" not in out
        assert "1838" in out
        assert "Benjamin Franklin" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_text_cleaning.py -v -k TestNewBoilerplatePatterns
```

Expected: 5 FAIL (assertions, not errors).

- [ ] **Step 3: Implement the patterns**

In `text_cleaning.py`, add after `_COUNTY_SEAT_CLAUSE_PATTERN` (with a comment in the existing style explaining the mechanism, citing findings §3.4):

```python
# Shared-eponym clauses ("named for/after Abraham Lincoln, 16th president
# ...") reproduce near-identical sentences in every county named for the same
# figure and were the top mechanism behind far-apart-but-similar pairs
# (analysis-output/source-a-findings.md section 3.4). The eponym's name is
# removed along with the clause: the shared name token is what creates the
# false similarity.
_EPONYM_CLAUSE_PATTERN = re.compile(
    r"(?:(?:The (?:county|parish)|It) (?:was|is) )?"
    r"named (?:for|after|in honor of)\b[^.]*\.?",
    re.IGNORECASE,
)
# Metro/micropolitan-area sentences follow one near-verbatim template
# ("X comprises / is included in the Y (metropolitan|micropolitan)
# (statistical) area"); the whole sentence is dropped.
_METRO_AREA_SENTENCE_PATTERN = re.compile(
    r"[^.]*\b(?:metropolitan|micropolitan|combined) (?:statistical )?area\b[^.]*\.?",
    re.IGNORECASE,
)
# Formation connective ("The county was established on/in") is shared across
# nearly all counties; the county-specific date it introduces is kept.
_FORMATION_CONNECTIVE_PATTERN = re.compile(
    r"\b(?:The (?:county|parish)|It) was "
    r"(?:formed|established|created|organized|founded)(?: (?:on|in))?\b",
    re.IGNORECASE,
)
```

Extend `strip_boilerplate_phrasing` (add the three `sub` calls before the existing `_LEADING_PUNCTUATION_PATTERN` line, and extend the docstring to mention the new families):

```python
    text = _OPENING_TOPIC_SENTENCE_PATTERN.sub("", text)
    text = _CENSUS_CLAUSE_PATTERN.sub("", text)
    text = _COUNTY_SEAT_CLAUSE_PATTERN.sub("", text)
    text = _METRO_AREA_SENTENCE_PATTERN.sub("", text)
    text = _EPONYM_CLAUSE_PATTERN.sub("", text)
    text = _FORMATION_CONNECTIVE_PATTERN.sub("", text)
    text = _LEADING_PUNCTUATION_PATTERN.sub("", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()
```

- [ ] **Step 4: Run the full test file to verify all pass (no regressions)**

```bash
uv run pytest tests/test_text_cleaning.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Spot-check against the real corpus**

```bash
uv run python -c "
import pandas as pd
from text_cleaning import strip_self_reference, strip_boilerplate_phrasing
df = pd.read_parquet('source_a_embeddings.parquet')
for name in ['Lincoln County, Kansas', 'Stutsman County, North Dakota', 'Franklin County, Maine']:
    raw = df.loc[df.county_name == name, 'raw_intro_text'].iloc[0]
    print('---', name)
    print(strip_boilerplate_phrasing(strip_self_reference(raw, name)))
"
```

Expected: no "named for/after …" clauses, no "metropolitan/micropolitan … area" sentences, population figures and dates still present. Eyeball that the residue is not gibberish (dangling "and"/commas are acceptable — embeddings don't need fluent prose — but the county-specific values must survive).

- [ ] **Step 6: Commit**

```bash
git add text_cleaning.py tests/test_text_cleaning.py
git commit -m "feat(source-a): strip eponym, metro-area, and formation boilerplate"
```

---

### Task 3: `clean_for_embedding` with empty-result fallback

**Files:**
- Modify: `text_cleaning.py`
- Test: `tests/test_text_cleaning.py`

**Interfaces:**
- Consumes: `strip_self_reference`, `strip_boilerplate_phrasing` (Tasks 1–2).
- Produces: `clean_for_embedding(raw_intro_text: str, county_name: str) -> str` — the single entry point Task 5's re-embed script calls. Never returns `""` for non-empty input.

Rationale: the stronger stripping can plausibly empty a stub county's text entirely; embedding an empty string would be garbage. Fallback ladder: fully-stripped → self-reference-stripped → raw.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_text_cleaning.py`, and add `clean_for_embedding` to the file's `from text_cleaning import (...)` block)

```python
class TestCleanForEmbedding:
    def test_applies_both_strip_stages(self) -> None:
        out = clean_for_embedding(LINCOLN_KS_INTRO, "Lincoln County, Kansas")
        assert "Lincoln County" not in out
        assert "Abraham Lincoln" not in out
        assert "2,939" in out

    def test_falls_back_when_stripping_empties_text(self) -> None:
        # Text that is *entirely* boilerplate: full stripping leaves nothing.
        text = "Foo County is a county located in the U.S. state of Kansas ."
        out = clean_for_embedding(text, "Foo County, Kansas")
        assert out != ""

    def test_falls_back_to_raw_when_everything_empties(self) -> None:
        # County name == entire text: self-reference stripping empties it too.
        out = clean_for_embedding("Foo County", "Foo County, Kansas")
        assert out == "Foo County"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_text_cleaning.py -v -k TestCleanForEmbedding
```

Expected: FAIL with `ImportError: cannot import name 'clean_for_embedding'`.

- [ ] **Step 3: Implement**

Append to `text_cleaning.py`:

```python
def clean_for_embedding(raw_intro_text: str, county_name: str) -> str:
    """Run the full offline cleaning pipeline with an empty-result fallback.

    Applies strip_self_reference then strip_boilerplate_phrasing. If the
    stronger stripping empties the text (possible for stub articles that are
    entirely boilerplate), falls back to the self-reference-stripped text,
    then to the raw text, so no county is ever embedded from an empty string.

    Args:
        raw_intro_text: Stored intro text, as in the parquet's raw_intro_text.
        county_name: County display name, e.g. "Lincoln County, Kansas".

    Returns:
        Non-empty text to embed (assuming raw_intro_text is non-empty).
    """
    dereferenced = strip_self_reference(raw_intro_text, county_name)
    stripped = strip_boilerplate_phrasing(dereferenced)
    if stripped:
        return stripped
    if dereferenced:
        return dereferenced
    return raw_intro_text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_text_cleaning.py -v
```

Expected: 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add text_cleaning.py tests/test_text_cleaning.py
git commit -m "feat(source-a): add clean_for_embedding with empty-result fallback"
```

---

### Task 4: Corpus-frequency sentence filter

**Files:**
- Create: `boilerplate_frequency.py`
- Test: `tests/test_boilerplate_frequency.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure module).
- Produces: `split_sentences(text: str) -> list[str]`, `mask_sentence_template(sentence: str) -> str`, `find_common_templates(texts: list[str], min_fraction: float = DEFAULT_MIN_COUNTY_FRACTION) -> set[str]`, `drop_common_sentences(text: str, common_templates: set[str]) -> str`, constant `DEFAULT_MIN_COUNTY_FRACTION: float = 0.05`.

Rationale: regexes can't enumerate every template; findings §3.4 mechanism (b) (generic-boilerplate convergence of short articles) is shape-level, not phrase-level. Masking numbers and proper-noun runs turns "the population was 2,939" and "the population was 50,395" into the same template; any template present in ≥5% of counties (~157) is definitionally boilerplate and is dropped. Deliberately aggressive — the harness (Task 6), not intuition, judges whether v3 beats v2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_boilerplate_frequency.py`:

```python
"""Tests for the corpus-frequency sentence filter."""

from boilerplate_frequency import (
    drop_common_sentences,
    find_common_templates,
    mask_sentence_template,
    split_sentences,
)


def test_split_sentences() -> None:
    assert split_sentences("First one. Second one. Third") == [
        "First one.",
        "Second one.",
        "Third",
    ]


def test_mask_numbers_and_proper_noun_runs() -> None:
    masked = mask_sentence_template("The population was 2,939 in Lincoln Center .")
    assert masked == "<name> population was <num> in <name> ."


def test_same_shape_different_values_share_template() -> None:
    a = mask_sentence_template("the population was 2,939.")
    b = mask_sentence_template("the population was 50,395.")
    assert a == b


def test_find_common_templates_by_county_fraction() -> None:
    # NOTE: the "unique" sentences must differ in *shape*, not just values —
    # masking collapses numbers/names, so vary the word count per text.
    common = "the population was 100."
    texts = [f"{common} {'alpha ' * (i + 1)}omega." for i in range(10)]
    templates = find_common_templates(texts, min_fraction=0.5)
    assert mask_sentence_template(common) in templates
    assert mask_sentence_template("alpha alpha alpha omega.") not in templates


def test_repeats_within_one_county_count_once() -> None:
    # One county repeating a sentence 10 times must not make it "common".
    texts = ["same thing here. " * 10] + [f"different {i} alpha beta." for i in range(9)]
    templates = find_common_templates(texts, min_fraction=0.5)
    assert mask_sentence_template("same thing here.") not in templates


def test_drop_common_sentences_keeps_rare_ones() -> None:
    templates = {mask_sentence_template("the population was 100.")}
    out = drop_common_sentences(
        "the population was 2,939. It hosts the state's only alligator farm.", templates
    )
    assert "alligator farm" in out
    assert "2,939" not in out


def test_drop_common_sentences_falls_back_when_all_dropped() -> None:
    text = "the population was 2,939."
    templates = {mask_sentence_template(text)}
    assert drop_common_sentences(text, templates) == text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_boilerplate_frequency.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'boilerplate_frequency'`.

- [ ] **Step 3: Implement**

Create `boilerplate_frequency.py`:

```python
"""Corpus-frequency sentence filter for Source A intro texts.

Masks numbers and proper-noun runs in each sentence to a shape "template";
sentences whose template recurs across a configurable fraction of counties
are treated as boilerplate and dropped. Complements the regex patterns in
text_cleaning.py by catching templated sentence shapes not explicitly
enumerated (analysis-output/source-a-findings.md section 3.4, mechanism b).
"""

from __future__ import annotations

import re
from collections import Counter

_NUMBER_PATTERN = re.compile(r"\d[\d,./-]*")
# A run of one or more capitalized words collapses to a single <NAME> token,
# so "Lincoln Center" and "Newport" mask identically. Sentence-initial words
# get masked too; acceptable, since masking is applied uniformly to every
# sentence being compared.
_PROPER_NOUN_RUN_PATTERN = re.compile(r"\b[A-Z][a-zA-Z'’]*(?:\s+[A-Z][a-zA-Z'’]*)*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# A sentence template present in >=5% of counties (~157 of 3,144) cannot be
# carrying county-specific narrative; it is templated boilerplate.
DEFAULT_MIN_COUNTY_FRACTION: float = 0.05


def split_sentences(text: str) -> list[str]:
    """Split whitespace-normalized intro text into sentences.

    Args:
        text: Single-line, whitespace-normalized text.

    Returns:
        Non-empty sentence strings, terminal punctuation retained.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def mask_sentence_template(sentence: str) -> str:
    """Reduce a sentence to its shape template.

    Numbers become <NUM>, proper-noun runs become <NAME>, whitespace is
    normalized, and the result is lowercased, so sentences differing only in
    county-specific values map to the same template.

    Args:
        sentence: One sentence.

    Returns:
        Lowercased template string.
    """
    masked = _NUMBER_PATTERN.sub("<NUM>", sentence)
    masked = _PROPER_NOUN_RUN_PATTERN.sub("<NAME>", masked)
    return _WHITESPACE_PATTERN.sub(" ", masked).strip().lower()


def find_common_templates(
    texts: list[str],
    min_fraction: float = DEFAULT_MIN_COUNTY_FRACTION,
) -> set[str]:
    """Find sentence templates shared by at least min_fraction of texts.

    Each text contributes each template at most once, so repetition within a
    single county cannot promote a template to "common".

    Args:
        texts: One cleaned intro text per county.
        min_fraction: Minimum fraction of counties a template must appear in.

    Returns:
        Set of common (boilerplate) templates.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update({mask_sentence_template(s) for s in split_sentences(text)})
    threshold = max(2, int(min_fraction * len(texts)))
    return {template for template, count in counts.items() if count >= threshold}


def drop_common_sentences(text: str, common_templates: set[str]) -> str:
    """Remove sentences whose template is in the common (boilerplate) set.

    Falls back to the input text unchanged if every sentence would be
    dropped, so no county ends up with an empty embedding input.

    Args:
        text: Cleaned intro text.
        common_templates: Output of find_common_templates.

    Returns:
        Text with boilerplate-template sentences removed.
    """
    kept = [s for s in split_sentences(text) if mask_sentence_template(s) not in common_templates]
    if not kept:
        return text
    return " ".join(kept)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_boilerplate_frequency.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add boilerplate_frequency.py tests/test_boilerplate_frequency.py
git commit -m "feat(source-a): add corpus-frequency sentence template filter"
```

---

### Task 5: Offline re-embedding script

**Files:**
- Create: `reembed_source_a.py`

**Interfaces:**
- Consumes: `clean_for_embedding` (Task 3); `find_common_templates`, `drop_common_sentences`, `DEFAULT_MIN_COUNTY_FRACTION` (Task 4); `BgeM3EmbeddingGenerator`, `configure_logging` from `ingest_source_a`.
- Produces: `source_a_embeddings_v2.parquet` / `source_a_embeddings_v3.parquet` with columns `county_name`, `fips_code`, `raw_intro_text`, `embedding_text`, `embedding` — the inputs Task 6 evaluates. Helper `build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]`.

No unit test for this script (it is a thin orchestration of already-tested functions plus a model call); verification is the post-run integrity check in Step 3. Note: ~3,144 bge-m3 CPU encodes take tens of minutes per variant — run in the background and append to a log (`>>`).

- [ ] **Step 1: Write the script**

Create `reembed_source_a.py`:

```python
"""Offline re-embedding of Source A from stored intro texts.

Reads raw_intro_text from source_a_embeddings.parquet (no Wikimedia API
access), applies a cleaning variant, re-embeds with BAAI/bge-m3, and writes
source_a_embeddings_{variant}.parquet including the embedding_text column so
the exact embedded text is auditable.

Variants:
  v2: strip_self_reference + strip_boilerplate_phrasing (incl. the new
      eponym / metro-area / formation patterns), with empty-text fallback.
  v3: v2, then drop sentences whose masked template appears in >=5% of
      counties (boilerplate_frequency).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from boilerplate_frequency import (
    DEFAULT_MIN_COUNTY_FRACTION,
    drop_common_sentences,
    find_common_templates,
)
from ingest_source_a import BgeM3EmbeddingGenerator, configure_logging
from text_cleaning import clean_for_embedding

INPUT_PARQUET_PATH: Path = Path(__file__).resolve().parent / "source_a_embeddings.parquet"
OUTPUT_TEMPLATE: str = "source_a_embeddings_{variant}.parquet"
LOG_EVERY: int = 250

logger = logging.getLogger(__name__)


def build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]:
    """Produce the per-county text to embed for a cleaning variant.

    Args:
        df: DataFrame with county_name and raw_intro_text columns.
        variant: "v2" (regex cleaning) or "v3" (v2 + frequency filter).

    Returns:
        One non-empty text per row, in row order.

    Raises:
        ValueError: If variant is not "v2" or "v3".
    """
    if variant not in ("v2", "v3"):
        raise ValueError(f"Unknown variant: {variant!r}")
    texts = [
        clean_for_embedding(raw, name)
        for name, raw in zip(df["county_name"], df["raw_intro_text"])
    ]
    if variant == "v3":
        templates = find_common_templates(texts, DEFAULT_MIN_COUNTY_FRACTION)
        logger.info("Frequency filter: %d common templates found", len(templates))
        texts = [drop_common_sentences(t, templates) for t in texts]
    return texts


def main() -> None:
    """Re-embed the full corpus for one cleaning variant."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["v2", "v3"], required=True)
    args = parser.parse_args()

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    logger.info("Loaded %d counties from %s", len(df), INPUT_PARQUET_PATH)
    texts = build_embedding_texts(df, args.variant)

    embedder = BgeM3EmbeddingGenerator(device="cpu")
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        if i % LOG_EVERY == 0:
            logger.info("Embedding %d/%d", i, len(texts))
        vector = embedder.l2_normalize(embedder.encode(text))
        embeddings.append(vector.tolist())

    out = df[["county_name", "fips_code", "raw_intro_text"]].copy()
    out["embedding_text"] = texts
    out["embedding"] = embeddings
    output_path = Path(__file__).resolve().parent / OUTPUT_TEMPLATE.format(variant=args.variant)
    out.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(out), output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run both variants (background, appending logs)**

```bash
uv run reembed_source_a.py --variant v2 >> reembed_run.log 2>&1
uv run reembed_source_a.py --variant v3 >> reembed_run.log 2>&1
```

Expected: `reembed_run.log` ends with `Wrote 3144 rows to .../source_a_embeddings_v2.parquet` (and `_v3`). Each run takes tens of minutes on CPU.

- [ ] **Step 3: Integrity check**

```bash
uv run python -c "
import numpy as np, pandas as pd
for v in ['v2', 'v3']:
    df = pd.read_parquet(f'source_a_embeddings_{v}.parquet')
    emb = np.vstack(df['embedding'].to_numpy())
    assert len(df) == 3144, len(df)
    assert emb.shape == (3144, 1024), emb.shape
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert (df['embedding_text'].str.len() > 0).all()
    print(v, 'OK — mean embedding_text len:', int(df.embedding_text.str.len().mean()))
"
```

Expected: `v2 OK ...` and `v3 OK ...`; v3's mean length noticeably below v2's (the filter removed sentences).

- [ ] **Step 4: Commit**

```bash
git add reembed_source_a.py source_a_embeddings_v2.parquet source_a_embeddings_v3.parquet
git commit -m "feat(source-a): offline re-embedding CLI + v2/v3 variant parquets"
```

(If variant parquets are too large to track comfortably, commit the script only and note the artifacts in the findings doc instead — check the size first with `ls -lh source_a_embeddings*.parquet`; the baseline parquet is already tracked, so matching its treatment is the default.)

---

### Task 6: Evaluation harness

**Files:**
- Create: `evaluate_source_a_variants.py`
- Test: `tests/test_evaluate_variants.py`

**Interfaces:**
- Consumes: variant parquets (Task 5 schema); `drop_stub_counties`, `haversine_distance_matrix` from `analyze_source_a_similarity`; `filter_to_fifty_states`, `mantel_test` from `analyze_source_a_clusters`; `fetch_county_centroids`, `CENTROIDS_CACHE_PATH` from `visualize_source_a`.
- Produces: `evaluate_variant(variant_df: pd.DataFrame, analysis_df: pd.DataFrame, distance_matrix: np.ndarray) -> dict` and `build_analysis_frame(baseline_df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame`; CLI writing `analysis-output/variant-eval.json`.

Protocol: the analysis county set is fixed **once from the baseline parquet** (stub filter → 50-states filter → centroid match) and reused verbatim for every variant, so metric deltas reflect embeddings, not set composition. Note: `drop_stub_counties` measures content length with the *current* (post-Task-2, stronger) stripping code, so the analysis n may come out slightly below the findings' 2,849 — expected, and consistent across all variants.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate_variants.py`:

```python
"""Unit test for evaluate_variant on a synthetic 6-county fixture."""

import numpy as np
import pandas as pd

from evaluate_source_a_variants import evaluate_variant


def _make_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    names = [f"County {i}, Somestate" for i in range(6)]
    emb = rng.normal(size=(6, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    variant_df = pd.DataFrame({"county_name": names, "embedding": list(emb)})
    analysis_df = pd.DataFrame(
        {"county_name": names, "lat": np.linspace(30, 45, 6), "lon": np.linspace(-120, -75, 6)}
    )
    n = 6
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) * 500.0
    return variant_df, analysis_df, dist


def test_evaluate_variant_metrics() -> None:
    variant_df, analysis_df, dist = _make_fixture()
    result = evaluate_variant(variant_df, analysis_df, dist)
    emb = np.vstack(variant_df["embedding"].to_numpy())
    sims = (emb @ emb.T)[np.triu_indices(6, k=1)]
    assert result["n_counties"] == 6
    assert result["pairwise_similarity_mean"] == float(sims.mean())
    assert result["pairwise_similarity_std"] == float(sims.std())
    assert -1.0 <= result["mantel_r"] <= 1.0
    assert 0.0 < result["mantel_p"] <= 1.0
    assert result["tracked_pair_mean"] is None  # fixture has no tracked pairs
    assert len(result["top_far_similar_pairs"]) >= 1


def test_evaluate_variant_rejects_missing_counties() -> None:
    variant_df, analysis_df, dist = _make_fixture()
    import pytest

    with pytest.raises(ValueError):
        evaluate_variant(variant_df.iloc[:4], analysis_df, dist)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evaluate_variants.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_source_a_variants'`.

- [ ] **Step 3: Implement**

Create `evaluate_source_a_variants.py`:

```python
"""Compare Source A embedding variants against the intro-text baseline.

Usage:
    uv run evaluate_source_a_variants.py BASELINE.parquet VARIANT.parquet [VARIANT2.parquet ...]

The analysis county set is fixed once from the baseline parquet (stub filter,
50-states filter, centroid match) and reused for every variant, so metric
differences come from the embeddings, not from set composition. Results are
printed as a comparison table and persisted to analysis-output/variant-eval.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analyze_source_a_clusters import filter_to_fifty_states, mantel_test
from analyze_source_a_similarity import drop_stub_counties, haversine_distance_matrix
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 499
MIN_CONTENT_LENGTH: int = 100
FAR_APART_PERCENTILE: float = 75.0
TOP_FAR_PAIR_COUNT: int = 5
OUTPUT_JSON_PATH: Path = (
    Path(__file__).resolve().parent / "analysis-output" / "variant-eval.json"
)

# The five far-apart-but-similar pairs from
# analysis-output/source-a-findings.md section 3.4 — the boilerplate
# mechanisms this experiment line is trying to remove.
TRACKED_BOILERPLATE_PAIRS: list[tuple[str, str]] = [
    ("Lincoln County, Kansas", "Lincoln County, Oregon"),
    ("Montgomery County, Alabama", "Stutsman County, North Dakota"),
    ("Stutsman County, North Dakota", "Williamsburg County, South Carolina"),
    ("Franklin County, Maine", "Franklin County, Nebraska"),
    ("Stutsman County, North Dakota", "Providence County, Rhode Island"),
]

REPORT_METRICS: list[str] = [
    "n_counties",
    "pairwise_similarity_mean",
    "pairwise_similarity_std",
    "mantel_r",
    "mantel_p",
    "silhouette_k2",
    "tracked_pair_mean",
]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_analysis_frame(baseline_df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Fix the analysis county set from the baseline parquet.

    Args:
        baseline_df: Baseline parquet contents (county_name, fips_code,
            raw_intro_text, embedding).
        centroids: Output of fetch_county_centroids (fips_code, lat, lon).

    Returns:
        DataFrame with county_name, fips_code, lat, lon for every county in
        the fixed analysis set.
    """
    df = drop_stub_counties(baseline_df, MIN_CONTENT_LENGTH)
    df = filter_to_fifty_states(df)
    df = df.merge(centroids, on="fips_code", how="inner")
    logger.info("Analysis set fixed at %d counties", len(df))
    return df[["county_name", "fips_code", "lat", "lon"]].reset_index(drop=True)


def evaluate_variant(
    variant_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> dict:
    """Compute the comparison metrics for one embedding variant.

    Args:
        variant_df: DataFrame with county_name and embedding columns.
        analysis_df: Fixed analysis frame (county_name, lat, lon) whose row
            order matches distance_matrix.
        distance_matrix: (n, n) pairwise haversine distances for analysis_df.

    Returns:
        Metrics dict (see REPORT_METRICS, plus tracked_pair_similarity and
        top_far_similar_pairs).

    Raises:
        ValueError: If the variant is missing any analysis-set county.
    """
    merged = analysis_df.merge(
        variant_df[["county_name", "embedding"]], on="county_name", how="left"
    )
    if merged["embedding"].isna().any():
        missing = int(merged["embedding"].isna().sum())
        raise ValueError(f"Variant is missing {missing} analysis-set county(ies)")

    embeddings = np.vstack(merged["embedding"].to_numpy())
    names = merged["county_name"].to_numpy()
    similarity = embeddings @ embeddings.T
    row_idx, col_idx = np.triu_indices(len(merged), k=1)
    sims_flat = similarity[row_idx, col_idx]
    dist_flat = distance_matrix[row_idx, col_idx]

    mantel_r, mantel_p = mantel_test(similarity, distance_matrix, N_PERMUTATIONS, RANDOM_SEED)
    labels = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit_predict(embeddings)
    silhouette = float(silhouette_score(embeddings, labels, random_state=RANDOM_SEED))

    name_to_idx = {name: i for i, name in enumerate(names)}
    tracked: dict[str, float] = {}
    for a, b in TRACKED_BOILERPLATE_PAIRS:
        if a in name_to_idx and b in name_to_idx:
            tracked[f"{a} | {b}"] = float(similarity[name_to_idx[a], name_to_idx[b]])

    far_mask = dist_flat >= np.percentile(dist_flat, FAR_APART_PERCENTILE)
    order = np.argsort(sims_flat[far_mask])[::-1][:TOP_FAR_PAIR_COUNT]
    far_rows = row_idx[far_mask][order]
    far_cols = col_idx[far_mask][order]
    top_far_pairs = [
        {
            "pair": f"{names[i]} | {names[j]}",
            "similarity": float(similarity[i, j]),
            "distance_km": float(distance_matrix[i, j]),
        }
        for i, j in zip(far_rows, far_cols)
    ]

    return {
        "n_counties": int(len(merged)),
        "pairwise_similarity_mean": float(sims_flat.mean()),
        "pairwise_similarity_std": float(sims_flat.std()),
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "silhouette_k2": silhouette,
        "tracked_pair_similarity": tracked,
        "tracked_pair_mean": float(np.mean(list(tracked.values()))) if tracked else None,
        "top_far_similar_pairs": top_far_pairs,
    }


def main() -> None:
    """Evaluate every given parquet against the first (baseline) one."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquets", nargs="+", type=Path, help="baseline first, then variants")
    args = parser.parse_args()

    centroids = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    baseline_df = pd.read_parquet(args.parquets[0])
    analysis_df = build_analysis_frame(baseline_df, centroids)
    distance_matrix = haversine_distance_matrix(
        analysis_df["lat"].to_numpy(dtype=float), analysis_df["lon"].to_numpy(dtype=float)
    )

    results: dict[str, dict] = {}
    for path in args.parquets:
        logger.info("Evaluating %s ...", path.name)
        results[path.name] = evaluate_variant(pd.read_parquet(path), analysis_df, distance_matrix)

    header = f"{'metric':<28}" + "".join(f"{name:>36}" for name in results)
    logger.info("\n%s", header)
    for metric in REPORT_METRICS:
        row = f"{metric:<28}" + "".join(
            f"{(results[name][metric] if results[name][metric] is not None else float('nan')):>36.4f}"
            for name in results
        )
        logger.info("%s", row)
    for name, result in results.items():
        logger.info("Top far-similar pairs — %s:", name)
        for entry in result["top_far_similar_pairs"]:
            logger.info(
                "  %.3f @ %5.0f km  %s", entry["similarity"], entry["distance_km"], entry["pair"]
            )

    payload = {
        "generated": datetime.date.today().isoformat(),
        "baseline": args.parquets[0].name,
        "seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "results": results,
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass (full suite, to catch regressions)**

```bash
uv run pytest -v
```

Expected: all tests PASS (12 text_cleaning + 7 boilerplate_frequency + 2 evaluate).

- [ ] **Step 5: Commit**

```bash
git add evaluate_source_a_variants.py tests/test_evaluate_variants.py
git commit -m "feat(source-a): variant evaluation harness with tracked boilerplate pairs"
```

---

### Task 7: Run the experiment, apply the decision gate, record the outcome

**Files:**
- Create: `analysis-output/variant-eval.json` (harness output)
- Modify: `analysis-output/source-a-findings.md` (new Round-4 section)
- Possibly modify (adoption branch only): `source_a_embeddings.parquet` and all regenerated artifacts (`stats.json`, `figures/*`, `source_a_map.html`, `source_a_similarity.html`/`_pairs.csv`, `source_a_clusters.html`/`_summary.csv`, `source_a_key_findings.ipynb`)

**Interfaces:**
- Consumes: `source_a_embeddings.parquet`, `source_a_embeddings_v2.parquet`, `source_a_embeddings_v3.parquet` (Task 5); the harness CLI (Task 6); the Decision Gate defined at the top of this plan.
- Produces: a recorded, committed experiment outcome — either an adopted new baseline with regenerated artifacts, or a documented negative result.

- [ ] **Step 1: Run the evaluation**

```bash
uv run evaluate_source_a_variants.py source_a_embeddings.parquet source_a_embeddings_v2.parquet source_a_embeddings_v3.parquet 2>&1 | tee -a variant_eval_run.log
```

Expected: comparison table printed; `analysis-output/variant-eval.json` written. The Mantel test on ~2,800 counties × 499 permutations takes a few minutes per variant.

- [ ] **Step 2: Apply the Decision Gate** (from the top of this plan — do not improvise new criteria)

Read the three variants' rows from the table / JSON and check, for each candidate:
1. `tracked_pair_mean` ≤ baseline `tracked_pair_mean` − 0.03
2. `pairwise_similarity_std` ≥ baseline `pairwise_similarity_std`
3. `mantel_r < 0` and `mantel_p < 0.05`

Pick the winner per the gate's tiebreak rules, or conclude "no adoption".

- [ ] **Step 3a (adoption branch): promote the winner and regenerate every artifact**

```bash
cp source_a_embeddings_<winner>.parquet source_a_embeddings.parquet   # git history keeps the old baseline
uv run visualize_source_a.py
uv run analyze_source_a_similarity.py
uv run analyze_source_a_clusters.py
uv run analyze_source_a_cluster_stability.py
uv run generate_source_a_insights.py
uv run python -m nbconvert --to notebook --execute --inplace analysis-output/source_a_key_findings.ipynb
```

Expected: each script exits 0 and rewrites its artifacts. **Caution:** `drop_stub_counties` now uses the stronger stripping, so the clustering/Mantel n will differ from 2,849 — that is expected and must be stated in the findings update, not silently absorbed.

- [ ] **Step 3b (no-adoption branch): keep the baseline**

Delete nothing; the variant parquets and `variant-eval.json` are the record. Skip regeneration.

- [ ] **Step 4: Update the findings document**

Append a new section to `analysis-output/source-a-findings.md` (bump frontmatter `round: 3` → `round: 4`) titled `## 12. Round 4 — Targeted Boilerplate Stripping (2026-07-XX)` containing, in the document's existing style:
- the two variants tested (v2 regex families, v3 = v2 + ≥5%-frequency sentence filter) and the pre-registered gate;
- the actual numbers table from `variant-eval.json` (baseline / v2 / v3 rows for the seven REPORT_METRICS);
- the before/after similarities of the five tracked pairs from §3.4;
- the decision taken and, on adoption, a note that all §9 artifacts were regenerated against the new baseline (with the new analysis n);
- a claim-candidate block with allowed/forbidden wording mirroring §6's format (forbidden: any claim that de-boilerplating validated Source A for the proposal's economic-narrative role — that still requires the Source E/B correlation test of §10).

- [ ] **Step 5: Run the full test suite one last time**

```bash
uv run pytest -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

Adoption branch:

```bash
git add source_a_embeddings.parquet analysis-output/ source_a_map.html source_a_similarity.html source_a_similarity_pairs.csv source_a_clusters.html source_a_cluster_summary.csv
git commit -m "feat(source-a): adopt <winner> de-boilerplated embeddings after gated evaluation"
```

No-adoption branch:

```bash
git add analysis-output/variant-eval.json analysis-output/source-a-findings.md
git commit -m "docs(source-a): record negative result for de-boilerplating variants"
```

---

## Out of Scope (explicitly)

- **Economic ground-truth correlation** (Source E capital-gains/W-2 ratio, Source B location quotients) — the proposal's actual claim; blocked on those sources being ingested (findings §8.5, §10). First follow-up once they land.
- **Non-Wikipedia text sources** — the other lever named in findings §8.3; a separate experiment line.
- **Alternative clustering robustness checks** (findings §7.4) and any frozen-LLM-encoder ablation (findings §11) — analysis work, not embedding improvement.
- **Section-expansion variants** — closed (findings §4).
