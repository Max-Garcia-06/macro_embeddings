# Source A Structural Features — Design

**Date:** 2026-08-25
**Status:** implemented; two claims below were falsified in review and are
corrected inline. See the **Corrections (2026-08-25, post-review)** section at
the foot of this document, and `analysis-output/source-a/source-a-findings.md`
§23 for what the round actually found.

## The question

How much of a county can be read off the *shape* of its Wikipedia article — how
many sections it has, how long they are, which ones are present — without
reading a word of the text?

Source A has been scored three ways so far: a 1024-dim `bge-m3` embedding, a
`content_length` scalar, and the shipped 29 typed columns from a fixed lexicon.
All three read content. None of them asks whether the article's skeleton, on its
own, carries anything.

The question is not idle. `n_body_sections` was computed during the section
round, found to correlate r = 0.550 against log tax returns — above
`content_length`'s 0.359 — and cut from the scored block for exactly that
reason (`pillar_matrix.SOURCE_A_DIAGNOSTIC_COLUMNS`).
That single result is the prior: structural features are size proxies wearing
hats. This round tests whether *any* of them survive once size is controlled
for, and it is designed so that a pure size proxy scores approximately zero
rather than scoring well and looking like a finding.

> **Correction (post-review).** That last clause is false as written, and the
> design was approved on it. The baseline controls for size *linearly, in logs*.
> A block of squares, cubes and pairwise products of the baseline's own three
> size columns — carrying no information the baseline lacks — scores +0.01748
> mean lift on 26 of 28 targets, against +0.00269 for the structural block. A
> pure size proxy scores approximately zero only if it is a *linear* one. The
> round therefore measures "beyond a linear-in-logs size model", not "beyond
> county size". A fourth arm, `size_nonlinear`, was added to
> `analyze_source_a_structure.py` to make that calibration visible in every
> artifact rather than arguable in prose.

## What gets built

Three artifacts, following the existing convention: a script computes, a stats
file records, a notebook reads. No number is typed into the notebook by hand.

### 1. `scripts/extract_source_a_structure_features.py`

Reads `data/source_a_sections.parquet` (64,588 rows, 3,144 counties, one row per
county × body section) and writes `data/source_a_structure_features.parquet`,
one row per county keyed on `fips_code`.

The module reads `section_title` and `len(section_text)`. It never reads the
section text itself. That restriction is the whole point of the round and should
be enforced by keeping the text column out of every derived frame after the
character count is taken.

**Count features**

| column | definition |
|---|---|
| `n_body_sections` | rows for the county |
| `n_distinct_titles` | distinct case-folded, stripped titles |
| `n_untitled_sections` | sections whose title is blank (2,009 such rows corpus-wide) |
| `max_section_id` | largest `section_id` |
| `n_id_gaps` | `max_section_id - n_body_sections` |

`section_id` is Parsoid's numbering and is **not contiguous** within a county —
verified, not assumed. The gaps mark sections that nest or were dropped during
ingestion, which makes `n_id_gaps` a free structural signal that costs nothing
to compute.

**Length features**

| column | definition |
|---|---|
| `total_body_chars` | sum of section lengths |
| `mean_section_chars`, `median_section_chars`, `max_section_chars`, `sd_section_chars` | per-county summary of section lengths |
| `share_in_largest_section` | `max_section_chars / total_body_chars` |
| `section_length_gini` | Gini coefficient over the county's section lengths |
| `n_stub_sections` | sections under 200 characters |
| `share_stub_sections` | `n_stub_sections / n_body_sections` |

The stub threshold is not arbitrary: the corpus-wide first quartile of section
length is 108 characters and the median is 340, so a 200-character cut splits
the bottom of the distribution rather than trimming a tail.

**Title-presence flags**

Binary `has_<title>` columns for every case-folded title appearing in more than
5% of counties (roughly 40 flags, from `demographics` at 3,142 counties down to
a ~157-county floor). The exact set is computed at runtime from the corpus and
written to the stats file, never hardcoded, so it moves when the corpus moves.
Titles are slugified to valid column names.

**Bucket character shares**

Character share per thematic bucket, summing to 1 across buckets. Precedence
order is load-bearing and inherited from `analyze_source_a_section_composition.CATEGORIES`,
which already resolves the known collisions (`population ranking` matches both
census and list patterns; `transportation` reads as highway, not economy).

Patterns reused as-is: `CENSUS_TITLE_PATTERN`, `LIST_TITLE_PATTERN`,
`HIGHWAY_TITLE_PATTERN` (`source_a_text_leakage`), `NARRATIVE_TITLE_PATTERN`
(`analyze_source_a_section_scope`), `ECONOMY_TITLE_PATTERN`
(`extract_source_a_section_features`).

Two differences from `CATEGORIES`, both deliberate and both to be documented in
the module docstring:

- **Highways get their own bucket** rather than folding into `lists`. The
  composition script merges them because it is answering "how much of this is
  content-free for an encoder"; this round is asking which *structures* are
  present, and a highway section is a different structure from a list of towns.
- **Two new patterns** are defined in the new module and inserted before the
  `other` fallback: `GEOGRAPHY_TITLE_PATTERN` (geography, climate, geology,
  national protected area, adjacent counties already being claimed by `lists`)
  and `GOVERNMENT_TITLE_PATTERN` (government, politics, law and government,
  education). These are the two largest occupants of the current `other`
  residual, and leaving them unsplit would put most of the corpus in a bucket
  named "other".

Final buckets: `census`, `lists`, `highways`, `narrative`, `economy`,
`geography`, `government`, `other`.

**Where the file lives.** `data/source_a_structure_features.parquet` sits beside
the other derived Source A parquets. It cannot leak into the scored matrix by
accident: `pillar_matrix._load_pillar_frames` loads explicit paths and does not
glob, and this path is not among them.

### 2. `scripts/analyze_source_a_structure.py`

Scores the structural block against the cross-pillar target basket. Writes
`outputs/source_a_structure_scores.csv` (per target × arm) and
`analysis-output/source-a/source_a_structure_stats.json`.

Protocol is the one `analyze_source_a_representation.py` established, reused
rather than reimplemented — its `_baseline_pipeline`, `_residual_pipeline`,
`_baseline_oof_predictions`, `_residual_oof_predictions` and `build_non_a_targets`
are imported, not copied:

- baseline is `pillar_matrix.SIZE_FEATURES` (`log_population`, `log_agi`,
  `log_gdp_latest`) plus state fixed effects, fitted **unpenalized** — ordinary
  least squares, since it is three size measures and ~50 dummies against
  thousands of rows
- each Source A block is fitted to the baseline's **residuals**, so a block that
  knows nothing costs approximately nothing instead of dragging the controls down
- ridge penalty chosen by **nested** crossvalidation inside each training fold
- 5 folds, seed 42, identical folds and identical rows across arms so the
  per-target differences are paired
- headline statistic is a Wilcoxon signed-rank test across targets, not any
  single target's lift

Targets are the 28-target cross-pillar basket that
`analyze_source_a_representation.build_non_a_targets` derives from
`pillar_matrix.build_matrix` — the 20 NAICS location quotients from Source B
plus the static targets in pillars C through F.

The degenerate target from findings §22.1, `no_fuel_used_share`, is **not** in
this basket and needs no exclusion here: it belongs to the separate 42-target
external basket in `ingest_external_targets.EXTERNAL_TARGETS`. If this round is
later extended to the external targets, that exclusion becomes live.

**Four arms:**

| arm | block on top of the baseline |
|---|---|
| `baseline` | — |
| `structure` | the structural columns |
| `typed` | the shipped 29 typed columns |
| `typed_plus_structure` | both |
| `size_nonlinear` | **null control, added in review:** squares, cubes and pairwise products of the baseline's own three size columns |

Two comparisons carry the round:

1. `structure` vs `baseline` — what does the skeleton know that population and
   state do not?
2. `typed_plus_structure` vs `typed` — does it know anything the shipped block
   does not already have? This is the fusion-relevant one, and the one most
   likely to come back near zero.

**Per-pillar breakdown is not optional.** Findings §14.2b established that 20 of
the 28 targets are one table, and §14.2c that the aggregate is therefore a
property of the basket rather than of the pillar. The stats file carries the
per-pillar summary and the notebook shows it next to the aggregate.

### 3. `analysis-output/source-a/source_a_structure_round.ipynb`

Reads the parquet and the stats JSON. Computes its figures; quotes no number
that is not in an artifact.

**Part one — what the corpus looks like.**

- Distribution of sections per county (mean 20.5, sd 7.8, range 1–73)
- Distribution of section lengths and its skew (mean 848 chars, median 340,
  max 55,228 — the mean is four standard deviations of nothing, and saying so is
  the point of the figure)
- Title frequency curve: the head (`demographics` 3,142, `geography` 3,104,
  `2020 census` 3,104) against the long tail
- Title co-occurrence among the flagged titles — which sections travel together
- Structure against population, scatter, for the headline count and length
  features

**Part two — the size-proxy audit.**

A correlation table of every structural column against each of the three
`SIZE_FEATURES` the baseline controls for, sorted by largest absolute
correlation. Three columns, not one: `n_body_sections` was cut on its
correlation with log tax returns specifically, not with population, and a table
that showed only population would have understated it. This runs *before* the
scoring section, so the reader has the size question in hand before seeing the
result. Columns that correlate weakly with all three size measures are the ones
with anything left to contribute, and naming them in advance makes the scoring
result checkable rather than surprising.

> **Correction (post-review).** This section originally read "sets the
> expectation: most of these columns are size measurements." The audit found the
> opposite: 6 of 64 columns clear |r| = 0.4 against any size measure. A separate
> defect: a Pearson table is a purely *linear* diagnostic, and a column can be a
> near-deterministic curved function of log population while showing |r| ≈ 0.04
> — which is exactly the channel the correction above says the round is exposed
> to. The audit therefore also reports out-of-fold R² of each column against a
> degree-3 polynomial basis in the same three size features, and the difference
> between that and a straight-line fit. It comes back clean per column (largest
> R² 0.340, largest curvature gain +0.076), which clears each column
> individually and not the block. A third defect: sorting by |r| alone put three
> near-constant flags — `has_section_demographics` fires for 3,140 of 3,144
> counties — at the top of the "columns with headroom" list, where low |r| means
> no variance rather than spare information. Prevalence and sd are now shown
> beside the correlation.

**Part three — the four arms.**

Mean out-of-fold R² lift per arm, the two headline paired tests, the per-pillar
table, and the per-target detail. Then a short section on which individual
structural columns carry whatever lift exists.

Matplotlib, not plotly — plotly's mimetype output needs a JupyterLab extension
and renders as blank space without it.

## Testing

`tests/test_source_a_structure.py`, following the existing test modules:

- shares across the eight buckets sum to 1 for every county
- precedence holds: a section titled `population ranking` lands in `census`, not
  `lists`; `transportation` lands in `highways`, not `economy`
- `n_id_gaps` is zero for a synthetic county with contiguous ids and positive
  for one with a gap
- untitled sections are counted, not dropped, and do not crash slugification
- every county in `source_a_sections.parquet` appears exactly once in the output
- Gini is 0 for equal-length sections and approaches 1 as one section dominates
- the title-flag vocabulary is derived from the corpus, not hardcoded — a
  fixture with a different title distribution produces different flags

## What this round does not do

- It does not propose shipping these columns. If `typed_plus_structure` beats
  `typed`, that is an argument for a follow-up round, not a change to
  `pillar_matrix`.
- It does not read section text. Any lexicon question belongs to the section
  scope round, which already exists.
- It does not revisit the `n_body_sections` cut. That decision stands until a
  result argues otherwise, and this round is the thing that would produce such a
  result.

## Corrections (2026-08-25, post-review)

This design was approved and implemented as written. A whole-branch review then
falsified two of its claims. Both are corrected inline above rather than edited
out, because a design document that quietly stops saying what it said is not a
review artifact any more.

1. **"designed so that a pure size proxy scores approximately zero"** — false.
   True only of a *linear* size proxy. An information-free block of curves on the
   baseline's own three size columns scores +0.01748 against the structural
   block's +0.00269. Fixed by adding the `size_nonlinear` null arm, which reports
   that number in the scores CSV, the stats JSON, the log and the notebook.
2. **"sets the expectation: most of these columns are size measurements"** —
   false. The audit found 6 of 64. The audit was also linear-only, and therefore
   blind to the channel correction 1 describes; it now carries a nonlinear
   diagnostic and a variance column.

The round's finding, restated honestly, is in
`analysis-output/source-a/source-a-findings.md` §23. Short version: the
structural lift is real but is measured against a linear-in-logs size model, and
roughly a quarter of it survives a flexible size control — concentrated in a few
consumer-facing location quotients, with Accommodation & Food Services the one
target that clearly holds up. The fusion comparison does not survive.
