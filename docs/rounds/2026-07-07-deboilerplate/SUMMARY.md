# Source A De-Boilerplating — Execution Summary

Executed `PLAN.md` end-to-end using subagent-driven development (one implementer + task-reviewer cycle per task, plus a final whole-branch review) in an isolated worktree/branch (`worktree-source-a-deboilerplate`).

## What was built

1. **`text_cleaning.py`** — extracted pure text-cleaning logic out of `ingest_source_a.py` so it's testable without triggering the county-crosswalk network load.
2. **Three new regex families** (eponym, metro-area, formation-connective) added to `strip_boilerplate_phrasing`, targeting the exact mechanisms behind the findings doc's top-5 false-similarity pairs.
3. **`clean_for_embedding()`** — single entry point with a three-tier fallback so cleaning never empties a county's text.
4. **`boilerplate_frequency.py`** — corpus-frequency sentence-template filter (masks numbers/proper-nouns, drops sentences whose template recurs in ≥5% of counties).
5. **`reembed_source_a.py`** — offline CLI; re-embedded all 3,144 counties for two variants (v2 = regex-only, v3 = v2 + frequency filter), same `BAAI/bge-m3` model, no Wikimedia API calls.
6. **`evaluate_source_a_variants.py`** — harness comparing baseline vs. v2 vs. v3 on a fixed analysis county set.

## Decision gate outcome

Both v2 and v3 passed all three pre-registered criteria (tracked-pair similarity drop ≥0.03, similarity std ≥ baseline, Mantel r<0 & p<0.05). **v3 adopted** as the new baseline — lower tracked-pair similarity (0.754 vs. baseline 0.829, a 0.075 drop). All downstream artifacts (visualizations, clustering, insights, notebook) regenerated against it.

Documented tradeoffs of adoption (surfaced by the final review, added to the findings doc):
- v3's worst extreme far-similar outlier pair (0.961) is *higher* than baseline's worst (0.833).
- Mantel r magnitude dropped ~59% (−0.122 → −0.050), though it still clears the gate's floor.

Full record: `analysis-output/source-a-findings.md` §12.

## Process notes

- 7 tasks, all individually reviewed; 2 tasks needed a fix-and-reverify round (missing test-fixture commit, missing commit trailers, a regex character-class bug on curly apostrophes).
- Task 6 deviated from the plan's literal reference code by importing shared constants (`RANDOM_SEED`, `MIN_CONTENT_LENGTH`, `configure_logging`) instead of duplicating them — flagged in pre-flight review and agreed before implementation.
- Task 5's actual embedding runs (tens of minutes on CPU) were executed directly rather than through a subagent, since that's mechanical execution, not implementation judgment.
- Final whole-branch review (opus) found two Important documentation-completeness gaps (the tradeoffs above) — fixed in one follow-up commit, re-verified, approved.

## Result

- 21/21 tests passing.
- 11 commits on `worktree-source-a-deboilerplate`, pushed.
- PR opened: https://github.com/Max-Garcia-06/wiki_embedding/pull/1
