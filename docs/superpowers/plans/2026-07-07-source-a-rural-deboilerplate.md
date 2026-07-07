# Source A Rural-County Boilerplate-Filter Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Exception:** Tasks 4 and 5's CLI runs (re-embedding the full 3,144-county
> corpus, running the evaluation harness) are mechanical, multi-minute data
> generation, not implementation judgment — run them directly in the
> orchestrating session rather than dispatching a fresh subagent for them,
> matching the precedent already set for this repo's v2/v3 round (see
> `docs/rounds/2026-07-07-deboilerplate/SUMMARY.md` after Task 3 relocates it).

**Goal:** Fix the rural/short-county over-stripping regression introduced by
the `v3` corpus-frequency boilerplate filter, without losing `v3`'s
demonstrated gains, and clean up dead-weight files from the prior round.

**Architecture:** Add a `raw` (no-cleaning) and `v4` (outcome-gated
frequency filter) variant to the existing `reembed_source_a.py` /
`evaluate_source_a_variants.py` pipeline. `v4` reuses `v3`'s regex + frequency
filter exactly, except a per-county frequency-filtered result is only kept
if it still clears the existing `MIN_CONTENT_LENGTH` stub-content bar;
otherwise the pre-filter (v2) text is used for that county. Evaluate `v4`
against `raw` and the current baseline (`v3`) through the existing harness,
apply a five-criterion decision gate, then either adopt `v4` (regenerate all
downstream artifacts) or record a negative result.

**Tech Stack:** Python 3.12, pandas, `BAAI/bge-m3` via `sentence-transformers`
(unchanged), pytest.

## Global Constraints

- Reuse `MIN_CONTENT_LENGTH = 100` imported from `analyze_source_a_similarity.py`
  — do not introduce a new/separate length constant.
- `RANDOM_SEED = 42` throughout (already the project-wide convention; no
  change needed since no new randomness is introduced).
- Gate criteria (exact values, from `docs/superpowers/specs/2026-07-07-source-a-rural-deboilerplate-design.md`):
  1. `tracked_pair_mean` drop ≥ 0.03 vs. `raw`
  2. `pairwise_similarity_std` ≥ `raw`
  3. `mantel_r < 0` with `mantel_p < 0.05`
  4. `v4`'s worst `top_far_similar_pairs` entry ≤ `0.9607` (v3's worst)
  5. Stutsman ND ↔ Providence RI tracked-pair similarity does not increase
     vs. `raw` (v3 regressed it by +0.0071; v4 must not)
- All existing tests (21 as of this round's start) must keep passing
  unmodified — this round does not change `v2`/`v3` behavior or the
  `strip_self_reference/strip_boilerplate_phrasing` regex layer.
- Cleanup file list (exact names): delete `source_a_embeddings_v2.parquet`,
  `source_a_embeddings_v3.parquet`, `reembed_run.log`,
  `variant_eval_run.log`; move `PLAN.md` and `SUMMARY.md` to
  `docs/rounds/2026-07-07-deboilerplate/`.

---

### Task 1: Add `raw` (no-cleaning) variant to `reembed_source_a.py`

**Files:**
- Modify: `reembed_source_a.py:38-61` (`build_embedding_texts`), `:64-69`
  (argparse `--variant` choices), `:8-13` (module docstring)
- Test: `tests/test_reembed_source_a.py` (new file)

**Interfaces:**
- Produces: `build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]`
  now accepts `variant="raw"`, returning `df["raw_intro_text"]` values
  unchanged (as a `list[str]`). This is consumed by Task 2 and Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reembed_source_a.py`:

```python
"""Tests for build_embedding_texts variant selection."""

import pandas as pd
import pytest

from reembed_source_a import build_embedding_texts


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_name": ["Lincoln County, Kansas"],
            "raw_intro_text": [
                "Lincoln County is a county in the U.S. state of Kansas."
            ],
        }
    )


def test_raw_variant_returns_unmodified_text() -> None:
    texts = build_embedding_texts(_fixture_df(), "raw")
    assert texts == ["Lincoln County is a county in the U.S. state of Kansas."]


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        build_embedding_texts(_fixture_df(), "bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reembed_source_a.py -v`
Expected: `test_raw_variant_returns_unmodified_text` FAILS (current code raises
`ValueError: Unknown variant: 'raw'` since `"raw"` is not in `("v2", "v3")`);
`test_unknown_variant_raises` PASSES already (that's fine, it's a
regression guard, not new behavior).

- [ ] **Step 3: Implement the `raw` variant**

Replace `reembed_source_a.py`'s `build_embedding_texts` (lines 38-61) with:

```python
def build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]:
    """Produce the per-county text to embed for a cleaning variant.

    Args:
        df: DataFrame with county_name and raw_intro_text columns.
        variant: "raw" (no cleaning), "v2" (regex cleaning), or "v3" (v2 +
            frequency filter).

    Returns:
        One non-empty text per row, in row order.

    Raises:
        ValueError: If variant is not one of the supported names.
    """
    if variant not in ("raw", "v2", "v3"):
        raise ValueError(f"Unknown variant: {variant!r}")
    if variant == "raw":
        return df["raw_intro_text"].tolist()
    texts = [
        clean_for_embedding(raw, name)
        for name, raw in zip(df["county_name"], df["raw_intro_text"])
    ]
    if variant == "v3":
        templates = find_common_templates(texts, DEFAULT_MIN_COUNTY_FRACTION)
        logger.info("Frequency filter: %d common templates found", len(templates))
        texts = [drop_common_sentences(t, templates) for t in texts]
    return texts
```

Update the module docstring (lines 8-13) to add:

```python
"""Offline re-embedding of Source A from stored intro texts.

Reads raw_intro_text from source_a_embeddings.parquet (no Wikimedia API
access), applies a cleaning variant, re-embeds with BAAI/bge-m3, and writes
source_a_embeddings_{variant}.parquet including the embedding_text column so
the exact embedded text is auditable.

Variants:
  raw: no cleaning -- embeds raw_intro_text unchanged. Used to reconstruct
      the true pre-cleaning baseline for gate comparisons.
  v2: strip_self_reference + strip_boilerplate_phrasing (incl. the new
      eponym / metro-area / formation patterns), with empty-text fallback.
  v3: v2, then drop sentences whose masked template appears in >=5% of
      counties (boilerplate_frequency).
"""
```

Update the argparse choices (around line 68):

```python
    parser.add_argument("--variant", choices=["raw", "v2", "v3"], required=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reembed_source_a.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -v`
Expected: all tests PASS (existing 21 plus these 2 new ones).

- [ ] **Step 6: Commit**

```bash
git add reembed_source_a.py tests/test_reembed_source_a.py
git commit -m "feat(source-a): add raw (no-cleaning) re-embedding variant

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Add outcome-gated `v4` variant

**Files:**
- Modify: `reembed_source_a.py` (`build_embedding_texts`, argparse choices,
  imports, module docstring) — building on Task 1's version
- Test: `tests/test_reembed_source_a.py` (extend)

**Interfaces:**
- Consumes: `MIN_CONTENT_LENGTH: int` from `analyze_source_a_similarity`
  (existing constant, value `100`, already used by
  `evaluate_source_a_variants.py` the same way).
- Consumes: `drop_common_sentences(text: str, common_templates: set[str]) -> str`
  and `find_common_templates(texts: list[str], min_fraction: float) -> set[str]`
  from `boilerplate_frequency` (existing, unchanged).
- Produces: `build_embedding_texts(df, "v4") -> list[str]`, consumed by
  Task 4's CLI run.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reembed_source_a.py`:

```python
def test_v4_skips_frequency_filter_when_result_too_short() -> None:
    common_sentence = (
        "This is a shared boilerplate sentence repeated across many "
        "counties for testing purposes today."
    )
    long_unique = (
        "This county has a very long and detailed unique history full of "
        "specific narrative content that goes well beyond the minimum "
        "content length threshold easily."
    )
    short_unique = "Short unique bit."

    df = pd.DataFrame(
        {
            "county_name": [f"County {i}, Somestate" for i in range(10)]
            + ["Rural County, Somestate"],
            "raw_intro_text": [f"{common_sentence} {long_unique}" for _ in range(10)]
            + [f"{common_sentence} {short_unique}"],
        }
    )

    texts = build_embedding_texts(df, "v4")

    # Long-article counties: boilerplate stripped as normal (matches v3).
    assert "shared boilerplate" not in texts[0]
    assert long_unique in texts[0]
    # Short/rural county: filtering would leave < MIN_CONTENT_LENGTH chars,
    # so the boilerplate sentence is kept instead of stripped.
    assert "shared boilerplate" in texts[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reembed_source_a.py::test_v4_skips_frequency_filter_when_result_too_short -v`
Expected: FAIL with `ValueError: Unknown variant: 'v4'`.

- [ ] **Step 3: Implement the `v4` variant**

Add the import near the top of `reembed_source_a.py` (alongside the existing
`boilerplate_frequency` import):

```python
from analyze_source_a_similarity import MIN_CONTENT_LENGTH
```

Replace `build_embedding_texts` with:

```python
def build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]:
    """Produce the per-county text to embed for a cleaning variant.

    Args:
        df: DataFrame with county_name and raw_intro_text columns.
        variant: "raw" (no cleaning), "v2" (regex cleaning), "v3" (v2 +
            frequency filter), or "v4" (v2 + outcome-gated frequency
            filter, protecting short/rural counties from over-stripping).

    Returns:
        One non-empty text per row, in row order.

    Raises:
        ValueError: If variant is not one of the supported names.
    """
    if variant not in ("raw", "v2", "v3", "v4"):
        raise ValueError(f"Unknown variant: {variant!r}")
    if variant == "raw":
        return df["raw_intro_text"].tolist()
    texts = [
        clean_for_embedding(raw, name)
        for name, raw in zip(df["county_name"], df["raw_intro_text"])
    ]
    if variant in ("v3", "v4"):
        templates = find_common_templates(texts, DEFAULT_MIN_COUNTY_FRACTION)
        logger.info("Frequency filter: %d common templates found", len(templates))
        filtered = [drop_common_sentences(t, templates) for t in texts]
        if variant == "v3":
            texts = filtered
        else:
            texts = [
                candidate if len(candidate) >= MIN_CONTENT_LENGTH else original
                for original, candidate in zip(texts, filtered)
            ]
    return texts
```

Update the module docstring's variant list to add:

```python
  v4: v2, then drop sentences whose masked template appears in >=5% of
      counties, UNLESS doing so would leave a county's text below
      MIN_CONTENT_LENGTH -- in that case the v2 text is kept unfiltered for
      that county (see analysis-output/source-a-findings.md section 12's
      rural-county over-stripping finding).
```

Update argparse choices:

```python
    parser.add_argument("--variant", choices=["raw", "v2", "v3", "v4"], required=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reembed_source_a.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reembed_source_a.py tests/test_reembed_source_a.py
git commit -m "feat(source-a): add v4 variant with outcome-gated frequency filter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Repo cleanup — remove dead-weight files, archive round docs

**Files:**
- Delete: `source_a_embeddings_v2.parquet`, `source_a_embeddings_v3.parquet`
  (tracked), `reembed_run.log`, `variant_eval_run.log` (untracked)
- Move: `PLAN.md` → `docs/rounds/2026-07-07-deboilerplate/PLAN.md`,
  `SUMMARY.md` → `docs/rounds/2026-07-07-deboilerplate/SUMMARY.md`
  (both currently untracked)

**Interfaces:** None — no code depends on these files at import time
(verify in Step 1).

- [ ] **Step 1: Verify nothing imports or reads these files by path**

Run: `grep -rn "source_a_embeddings_v2\|source_a_embeddings_v3" --include="*.py" .`
Expected: no matches, or only matches inside `reembed_source_a.py`'s output
path template (`source_a_embeddings_{variant}.parquet`), which writes, not
reads, that path — confirming deletion is safe.

- [ ] **Step 2: Delete the dead-weight files**

```bash
git rm source_a_embeddings_v2.parquet source_a_embeddings_v3.parquet
rm reembed_run.log variant_eval_run.log
```

- [ ] **Step 3: Archive the round's planning docs**

```bash
mkdir -p docs/rounds/2026-07-07-deboilerplate
mv PLAN.md docs/rounds/2026-07-07-deboilerplate/PLAN.md
mv SUMMARY.md docs/rounds/2026-07-07-deboilerplate/SUMMARY.md
git add docs/rounds/2026-07-07-deboilerplate/PLAN.md docs/rounds/2026-07-07-deboilerplate/SUMMARY.md
```

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: all tests PASS (none of the deleted/moved files are test
fixtures or imports).

- [ ] **Step 5: Commit**

```bash
git status
git commit -m "chore(source-a): remove superseded variant parquets/logs, archive round docs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Re-embed the corpus for `raw` and `v4`

> Run this task directly (not via a dispatched subagent) — it is CLI
> execution of code already implemented and tested in Tasks 1-2, taking
> tens of minutes on CPU per variant (same as the v2/v3 round).

**Files:**
- Produces: `source_a_embeddings_raw.parquet`, `source_a_embeddings_v4.parquet`
  (both via `reembed_source_a.py`'s existing `OUTPUT_TEMPLATE`)

**Interfaces:**
- Consumes: `uv run reembed_source_a.py --variant raw|v4` (Task 1/2's CLI,
  unchanged invocation pattern).

- [ ] **Step 1: Re-embed the `raw` variant**

```bash
uv run --env-file .env reembed_source_a.py --variant raw >> reembed_raw_run.log 2>&1
```

Expected: exits 0, logs `Wrote 3144 rows to .../source_a_embeddings_raw.parquet`.

- [ ] **Step 2: Re-embed the `v4` variant**

```bash
uv run --env-file .env reembed_source_a.py --variant v4 >> reembed_v4_run.log 2>&1
```

Expected: exits 0, logs a `Frequency filter: N common templates found` line
and `Wrote 3144 rows to .../source_a_embeddings_v4.parquet`.

- [ ] **Step 3: Spot-check outputs**

```bash
uv run python -c "
import pandas as pd
raw = pd.read_parquet('source_a_embeddings_raw.parquet')
v4 = pd.read_parquet('source_a_embeddings_v4.parquet')
assert len(raw) == 3144 and len(v4) == 3144
assert (raw['embedding_text'] == raw['raw_intro_text']).all()
print('raw and v4 parquets OK, 3144 rows each')
"
```

Expected: prints `raw and v4 parquets OK, 3144 rows each`, no assertion error.

(No commit here — these are large generated data files evaluated in Task 5;
whether they're kept depends on Task 6's adopt/reject decision.)

---

### Task 5: Run the evaluation harness and apply the decision gate

> Run this task directly, same rationale as Task 4.

**Files:**
- Reads: `source_a_embeddings_raw.parquet`, `source_a_embeddings.parquet`
  (current baseline, i.e. `v3`), `source_a_embeddings_v4.parquet`
- Produces: `analysis-output/variant-eval.json` (overwritten with this
  round's 3-way comparison)

**Interfaces:**
- Consumes: `evaluate_source_a_variants.py` (existing, unmodified) — its CLI
  takes baseline-first-then-variants positional args.

- [ ] **Step 1: Run the harness over all three parquets**

```bash
uv run --env-file .env evaluate_source_a_variants.py source_a_embeddings_raw.parquet source_a_embeddings.parquet source_a_embeddings_v4.parquet >> variant_eval_v4_run.log 2>&1
cat analysis-output/variant-eval.json
```

Expected: exits 0; JSON contains a `results` object keyed by
`source_a_embeddings_raw.parquet`, `source_a_embeddings.parquet`, and
`source_a_embeddings_v4.parquet`, each with the `REPORT_METRICS` fields plus
`tracked_pair_similarity` and `top_far_similar_pairs`.

- [ ] **Step 2: Evaluate the five gate criteria against the JSON output**

Using `results["source_a_embeddings_raw.parquet"]` as `raw`,
`results["source_a_embeddings.parquet"]` as `v3`, and
`results["source_a_embeddings_v4.parquet"]` as `v4`:

| # | Criterion | Check |
|---|---|---|
| 1 | `v4.tracked_pair_mean` ≤ `raw.tracked_pair_mean - 0.03` | |
| 2 | `v4.pairwise_similarity_std` ≥ `raw.pairwise_similarity_std` | |
| 3 | `v4.mantel_r < 0` and `v4.mantel_p < 0.05` | |
| 4 | `max(p["similarity"] for p in v4.top_far_similar_pairs) <= 0.9607` | |
| 5 | `v4.tracked_pair_similarity["Stutsman County, North Dakota | Providence County, Rhode Island"] <= raw.tracked_pair_similarity[...]` | |

Record the actual numbers and pass/fail for each row — this table (with
real values filled in) becomes the gate-check table in Task 6's
`source-a-findings.md` section either way.

- [ ] **Step 3: Decide adopt or reject**

If **all 5 criteria pass**: proceed to Task 6, Adopt path.
If **any criterion fails**: proceed to Task 6, Reject path.

(No commit here — `variant-eval.json` is committed as part of Task 6
regardless of which path is taken.)

---

### Task 6: Execute the gate decision

Only one of the two paths below is executed, per Task 5 Step 3's outcome.
Both are fully specified here since the actual result isn't known until
Task 5 runs.

**Files (Adopt path):**
- Modify: `source_a_embeddings.parquet` (overwritten with `v4` content)
- Delete: `source_a_embeddings_v4.parquet`, `source_a_embeddings_raw.parquet`
- Modify: `source_a_map.html`, `source_a_similarity.html`,
  `source_a_similarity_pairs.csv`, `source_a_clusters.html`,
  `source_a_cluster_summary.csv`, `analysis-output/stats.json`,
  `analysis-output/figures/*`, `analysis-output/source_a_key_findings.ipynb`
  (all regenerated)
- Modify: `analysis-output/source-a-findings.md` (new §13 section)

**Files (Reject path):**
- Delete: `source_a_embeddings_v4.parquet`, `source_a_embeddings_raw.parquet`
- Modify: `analysis-output/source-a-findings.md` (new §13 section,
  negative result)

#### Adopt path (if Task 5 Step 3 said all 5 criteria passed)

- [ ] **Step A1: Overwrite the baseline with v4**

```bash
cp source_a_embeddings_v4.parquet source_a_embeddings.parquet
rm source_a_embeddings_v4.parquet source_a_embeddings_raw.parquet
```

- [ ] **Step A2: Regenerate every downstream artifact**

```bash
uv run --env-file .env generate_source_a_insights.py
uv run --env-file .env visualize_source_a.py
uv run --env-file .env analyze_source_a_similarity.py
uv run --env-file .env analyze_source_a_clusters.py
uv run --env-file .env analyze_source_a_cluster_stability.py
uv run --env-file .env jupyter nbconvert --execute --to notebook --inplace analysis-output/source_a_key_findings.ipynb
```

Expected: every command exits 0. Note the printed PC1 variance, k-selection,
silhouette, and Mantel r/p from each script's output for the findings.md
table (Step A3).

- [ ] **Step A3: Add a new §13 section to `analysis-output/source-a-findings.md`**

Append (adjust the bracketed values with the real numbers from Task 5 Step
2 and Step A2's output):

```markdown
## 13. Round 5 — Rural-County Boilerplate-Filter Protection (2026-07-07)

**Question**: §12 documented that v3's corpus-frequency filter, while
passing its adoption gate, introduced a worse-than-baseline worst-case
outlier (0.9607 vs. baseline's 0.8327) by over-stripping short/rural
articles. Could an outcome-gated version of the same filter fix this
without giving up v3's tracked-pair and dispersion gains?

**Method**: a `v4` variant, identical to v3's regex + frequency-filter
pipeline, except each county's frequency-filtered result is only used if it
still meets the existing MIN_CONTENT_LENGTH (100 char) stub-content bar;
otherwise that county keeps its v2 (regex-only) text. Evaluated via
`evaluate_source_a_variants.py` against a freshly reconstructed `raw`
(no-cleaning) baseline and the current v3 baseline, n=[N] counties.

**Results**:

| Metric | raw | v3 (baseline) | v4 |
|---|---|---|---|
| tracked_pair_mean | [x] | 0.75410 | [x] |
| pairwise_similarity_std | [x] | 0.08830 | [x] |
| mantel_r | [x] | -0.04999 | [x] |
| mantel_p | [x] | 0.002 | [x] |
| worst top_far_similar_pairs | 0.8327 | 0.9607 | [x] |
| Stutsman/Providence tracked pair | 0.8255 | 0.8327 | [x] |

**Gate check**: [paste the 5-row table from Task 5 Step 2 with real
pass/fail values]

**Decision**: v4 adopted as the new baseline
(`source_a_embeddings.parquet` overwritten; git history retains v3).

[Add any tradeoffs the real numbers surface, following this document's own
established practice of disclosing costs rather than only gains.]
```

- [ ] **Step A4: Commit**

```bash
git add source_a_embeddings.parquet source_a_map.html source_a_similarity.html \
  source_a_similarity_pairs.csv source_a_clusters.html source_a_cluster_summary.csv \
  analysis-output/stats.json analysis-output/figures/ \
  analysis-output/source_a_key_findings.ipynb analysis-output/variant-eval.json \
  analysis-output/source-a-findings.md
git commit -m "feat(source-a): adopt v4 rural-protected embeddings after gated evaluation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

#### Reject path (if Task 5 Step 3 said any criterion failed)

- [ ] **Step R1: Discard the candidate parquets**

```bash
rm source_a_embeddings_v4.parquet source_a_embeddings_raw.parquet
```

- [ ] **Step R2: Add a new §13 section to `analysis-output/source-a-findings.md`**

Append (fill in real numbers and the specific failing criterion/criteria):

```markdown
## 13. Round 5 — Rural-County Boilerplate-Filter Protection (2026-07-07)

**Question**: [same as Adopt path's Question]

**Method**: [same as Adopt path's Method]

**Gate check**: [paste the 5-row table from Task 5 Step 2 with real
pass/fail values, marking which criterion/criteria failed]

**Decision**: v4 rejected — baseline remains v3
(`source_a_embeddings.parquet` unchanged). The rural/short-county
over-stripping regression documented in §12 remains an open, undismissed
limitation of the current baseline; see Next Actions below for what would
need to change for a future attempt.

**Next Actions**:
1. [Concrete next lever, informed by which criterion failed — e.g. if
   criterion 4/5 failed, the outcome-gate's MIN_CONTENT_LENGTH floor may be
   too low to actually protect the worst-case counties, suggesting a
   higher, purpose-specific floor rather than reusing the stub-detection
   threshold.]
```

- [ ] **Step R3: Commit**

```bash
git add analysis-output/source-a-findings.md
git commit -m "docs(source-a): record v4 rural-protection variant as rejected

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes

- **Spec coverage**: §1 (outcome-gated filter) → Task 2; §2 (raw variant) →
  Task 1; §3 (v4 variant naming) → Task 2; §4 (decision gate) → Task 5/6;
  §5 (cleanup) → Task 3. All spec sections have a task.
- **Type consistency**: `build_embedding_texts(df: pd.DataFrame, variant: str) -> list[str]`
  signature is unchanged across Tasks 1-2, only the accepted `variant`
  values grow (`raw` in Task 1; `v4` in Task 2) — matches the design doc.
- **No placeholders**: Task 6's Adopt/Reject paths are both fully written
  out; the only bracketed values are real numbers not knowable until Task 5
  actually runs (an evaluation result, not an unresolved design decision).
