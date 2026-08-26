# Source A Shape Profile — Design

**Date:** 2026-08-26
**Status:** approved, not yet implemented
**Predecessor:** `2026-08-25-source-a-structure-features-design.md`, and
`analysis-output/source-a/source-a-findings.md` §23, which this round extends.

## The question

Round one asked whether the shape of a county's Wikipedia article knows anything
beyond county size, and answered: a little, mostly not, and the apparent signal
was three-quarters curvature in size that a linear control could not absorb.

This round asks the other question, and asks it harder: **how much can be
extracted from article shape at all** — and it reports the answer in both
framings at once, so no number can be read in the wrong one.

Two things make this more than a rerun:

1. **Four new feature families** that round one never built, none of which is an
   obvious volume measure.
2. **A diagnostic §23 said did not exist.** §23 closed on an open problem: the
   per-column size audit comes back clean while the block as a whole is
   size-dependent, because the dependence is *joint* across columns and no
   per-column statistic can see it. This round measures it directly.

## What gets built

### 1. `scripts/extract_source_a_shape_profile.py`

Reads `data/source_a_sections.parquet` and writes
`data/source_a_shape_profile.parquet`, one row per county keyed on `fips_code`.

**This module does not touch `data/source_a_structure_features.parquet`.** That
artifact is cited by §23; mutating it would silently invalidate a committed
finding. The new families live in their own file and the analysis joins the two.

Roughly 50 new columns in four families.

**Order and position.** Where a section sits, normalized to `[0, 1]` by its
index among the county's body sections.

| column | definition |
|---|---|
| `pos_<title>` | normalized position of each title in the round-one flag vocabulary, or `-1.0` when absent |

| `pos_longest_section` | normalized position of the longest section |
| `pos_first_economy`, `pos_first_census`, `pos_first_narrative` | normalized position of the first section in each of those buckets |
| `history_before_economy` | 1.0 when a narrative-bucket section precedes the first economy-bucket section |
| `position_spread` | standard deviation of the positions of the flagged titles present |

Absence is encoded as `-1.0` rather than null, so the sentinel is outside the
`[0, 1]` range a present section can take and a tree can split on it cleanly.
This differs from round one's convention (absence as `0`/`False`) for a reason:
position `0.0` means "first", which is the opposite of absent.

The title list for `pos_<title>` is **round one's flag vocabulary**, imported by
calling `extract_source_a_structure_features.flag_vocabulary(sections)` rather
than re-deriving it. Two independently-derived vocabularies that drift apart
would silently give `has_section_x` and `pos_x` different membership, and the
join in the analysis module would not complain.

Order is editorial priority — which part of a county an editor put at the top —
and it is the one signal in this corpus with no obvious reading as a volume
measure.

**Template conformity.** County articles follow a house skeleton. Deviation is
editorial attention, which is not the same quantity as county size.

| column | definition |
|---|---|
| `template_jaccard` | Jaccard similarity between the county's title set and the corpus-modal title set |
| `n_core_missing` | count of modal-set titles the county lacks |
| `n_unusual_sections` | sections whose title is held by under 1% of counties |
| `share_unusual_sections` | `n_unusual_sections / n_body_sections` |
| `mean_title_rarity` | mean over the county's titles of `1 - (counties holding it / all counties)` |
| `n_title_words` | mean word count of the county's section titles |

The modal title set is computed at runtime as the titles held by more than half
of counties, and written to the stats file — never hardcoded, on the same rule
as round one's flag vocabulary.

**Surface statistics.** Character-class densities, computed overall and per
thematic bucket. These read characters; they never read meaning. A census table
rendered as prose is roughly 30% digits, and that is a fact about the article's
shape rather than its content.

| column | definition |
|---|---|
| `digit_density`, `digit_density_<bucket>` | digits ÷ characters |
| `punct_density` | punctuation ÷ characters |
| `capital_ratio` | uppercase letters ÷ letters |
| `mean_word_length` | characters ÷ whitespace-delimited tokens |
| `numeral_to_letter` | digits ÷ letters |

Per-bucket densities are computed for the four largest buckets only — census,
lists, narrative, geography — because the economy and government buckets are
absent for a large share of counties and a density over zero characters is not a
number.

**Length curve.** The sorted section-length curve, beyond the Gini round one
already has.

| column | definition |
|---|---|
| `top3_length_share` | share of body characters in the longest three sections |
| `length_decay_slope` | OLS slope of `log(length)` on rank over the sorted sections |
| `chars_<bucket>` | absolute characters per bucket, all eight |

Two notes on what is deliberately *not* here:

- **No `top1_length_share`.** It is exactly round one's `share_in_largest_section`
  under a different name, and shipping it would put the same quantity in the
  block twice. The equality is worth asserting in a test — it catches a drift in
  either module — but not worth a column.
- **`chars_<bucket>` is `share_chars_<bucket> × total_body_chars`**, so it is
  derivable from two columns round one already ships. It earns its place for the
  ridge arm, which cannot form products, and is redundant for the boosting arm,
  which can. Stating that here so a reader does not mistake eight columns for
  eight new facts.

### 2. `scripts/analyze_source_a_shape_profile.py`

Scores the arms. Reuses the round-one machinery rather than restating it —
`analyze_source_a_structure` already exports `Arm`, `size_nonlinear_block`,
`size_curvature_directions`, `build_flexible_baseline_design`, `typed_columns`,
`FLEXIBLE_SUFFIX` and `NULL_ARM_KEY`, and `analyze_source_a_representation`
exports `_baseline_oof_predictions`, `_residual_oof_predictions`,
`_alone_oof_r2` and `build_non_a_targets`.

**Five arms:**

| arm | block |
|---|---|
| `shape_v1` | round one's 64 structural columns — a regression check that nothing drifted; must reproduce §23 |
| `shape_v2` | `shape_v1` plus the ~50 new columns — the push |
| `typed` | the shipped 29 typed columns, for scale |
| `typed_plus_shape_v2` | both |
| `size_nonlinear` | the information-free null control, carried forward unchanged |

**Two learners.** "As much as possible" is bounded by the model class, and ridge
is linear. Shape features are the kind that interact — a stub-heavy article with
30 sections is a different object from a stub-heavy article with 5 — so a linear
learner cannot express most of what this block might know.

- `ridge` — `_residual_pipeline`'s nested-CV RidgeCV, unchanged. Primary, because
  it is what every prior Source A round used and therefore the only arm directly
  comparable to §13 through §23.
- `boost` — `HistGradientBoostingRegressor`, fitted the same way (to the
  baseline's residuals, inside each fold, hyperparameters fixed rather than
  searched). Secondary, and reported beside ridge rather than instead of it.

Fixing the boosting hyperparameters rather than searching them is deliberate. A
search inside each fold would be the honest version, but it multiplies runtime
by the grid size and the point of this arm is a ceiling estimate, not a tuned
model. The fixed settings are stated in the module and the ceiling is therefore
a lower bound on what boosting could reach.

**Three framings, reported for every arm × learner, never one without the others:**

| column | meaning |
|---|---|
| `r2_alone` | out-of-fold R² with the shape block as the only predictor. No controls. The raw-power number, via `_alone_oof_r2`. |
| `lift` | lift over the linear size-plus-state baseline. Comparable to §23 and every earlier round. |
| `lift_flexbase` | lift over the curvature-augmented baseline. The strict number. |

This is the round's central discipline. Round one's failure was not a wrong
number; it was a right number quoted in a framing its readers could not see. A
row that carries all three cannot be misquoted the same way.

Outputs: `outputs/source_a_shape_profile_scores.csv`,
`outputs/source_a_shape_profile_by_pillar.csv`,
`analysis-output/source-a/source_a_shape_profile_stats.json`.

### 3. The joint-size diagnostic

A separate, cheap computation in the same module, and the thing this round is
most likely to be remembered for.

§23 established that no per-column statistic can detect the size dependence that
actually carries the lift, because the dependence is joint. So invert the
question: **predict size from shape.**

For each of `log_population`, `log_agi`, `log_gdp_latest`, report the out-of-fold
R² of that size measure regressed on the whole shape block — once per block
(`shape_v1`, `shape_v2`) and once per learner. Same folds, same seed.

This yields one number per (size measure, block, learner) that says how much of
county size the shape block encodes. It bounds how much of any reported lift
could be size in disguise, it is interpretable without a statistics argument,
and it answers §23's open item directly rather than by proxy.

Written to the stats file under `size_recoverability`, and shown in the notebook
immediately before the arms.

### 4. `analysis-output/source-a/source_a_shape_profile_round.ipynb`

Built by `scripts/build_source_a_shape_profile_notebook.py`, on the round-one
pattern: `nbformat` cells, executed with `nbconvert`, every number read from an
artifact, matplotlib only.

Order:

1. **The new families** — what order, conformity, surface and curve features look
   like across the corpus, and how they correlate with the round-one block
2. **The joint-size diagnostic** — how much of county size is recoverable from
   shape, before any arm is scored
3. **The arms** — all five, both learners, all three framings
4. **Where the ceiling is** — the best targets under `r2_alone`, and what falls
   away under each stricter framing

The size diagnostic precedes the arms for the same reason round one's audit
did: the reader should know how much size is in the block before seeing what
the block scores.

## What this round does not do

- It does not read section text for meaning. Character-class densities are the
  boundary and are not crossed further.
- It does not propose shipping anything. If `shape_v2` beats `shape_v1`, that is
  an argument for a follow-up, not a change to `pillar_matrix`.
- It does not revisit round one's numbers. `shape_v1` exists to prove they
  reproduce; if they do not, that is a defect to report, not a result.
- It does not tune the boosting arm. The ceiling it reports is a lower bound.

## Testing

`tests/test_source_a_shape_profile.py`:

- position sentinels: an absent section is `-1.0`, a first section is `0.0`, and
  the two are never confused
- `history_before_economy` is 1.0 when narrative precedes economy and 0.0 when it
  does not, on synthetic frames with both orders
- `template_jaccard` is 1.0 for a county holding exactly the modal set and falls
  as titles are removed
- the modal title set is derived from the corpus, not hardcoded — a fixture with
  a different distribution yields a different set
- digit density is 0.0 for letters-only text and 1.0 for digits-only text
- per-bucket densities are not computed over zero characters
- a recomputed top-one length share equals round one's
  `share_in_largest_section` for every county — a cross-module consistency check
  that catches a drift in either, without shipping the duplicate column
- `pos_<title>` covers exactly the titles in round one's `has_section_<title>`
  set, since both come from the same `flag_vocabulary` call
- every county in the sections parquet appears exactly once in the output
- the shape parquet and the round-one structural parquet share no column name,
  so the join in the analysis module cannot collide
- `shape_v1` scored through the new module reproduces §23's committed lifts
