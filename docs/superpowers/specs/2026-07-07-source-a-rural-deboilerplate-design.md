# Source A / Rural-County Protection for the Boilerplate Frequency Filter / 2026-07-07

## Context

`analysis-output/source-a-findings.md` §12 documents adopting a `v3` cleaning
variant (regex stripping + a corpus-frequency sentence filter,
`boilerplate_frequency.py`) as the new Source A embedding baseline. `v3`
passed its pre-registered decision gate, but the gate did not measure
worst-case behavior, and a real regression surfaced after adoption:

- The single worst far-apart-but-similar pair got *more* similar under `v3`
  (0.8327 → 0.9607, Allamakee County, IA ↔ Clatsop County, OR), the opposite
  of the round's goal.
- One of the four evaluable tracked boilerplate pairs (Stutsman ND ↔
  Providence RI) moved in the wrong direction under `v3` (+0.0071).

Root cause: `drop_common_sentences` removes any sentence whose masked
template recurs in ≥5% of counties, with no regard for how much of a given
county's *own* article that sentence represents. For a short (rural) article
of 2-3 sentences, dropping even one boilerplate sentence can remove most of
the remaining content, collapsing the residual text toward a generic
centroid — differentiating counties less, not more, for exactly the
population the round was trying to help.

## Goal

Keep the boilerplate-removal gains `v3` demonstrated (tracked-pair similarity
down, dispersion up, geography signal still real) while eliminating the
rural/short-article over-stripping failure mode, without introducing an
arbitrary/unvalidated length cutoff to decide which counties count as
"rural."

## Design

### 1. Outcome-gated frequency filter (the core change)

In `reembed_source_a.py`'s `build_embedding_texts`, per county, after
computing `v2_text = clean_for_embedding(raw, name)` (unchanged) and the
corpus-wide `common_templates` (unchanged):

```python
candidate = drop_common_sentences(v2_text, common_templates)
final_text = candidate if len(candidate) >= MIN_CONTENT_LENGTH else v2_text
```

`MIN_CONTENT_LENGTH` is the existing stub-content constant already imported
from `analyze_source_a_similarity.py` elsewhere in this codebase (the same
bar used to decide whether a county has "enough real content to analyze" —
294/868 stub counties dropped in prior rounds used this same threshold). No
new magic number is introduced.

This is a per-county binary decision (frequency filter fully applied, or
fully skipped) — not partial sentence reinstatement. The trigger is tied
directly to the diagnosed harm (ending up stub-thin after filtering), not a
proxy like raw article length, so it protects any county the filter would
have hurt, not just ones pre-classified as "rural."

`find_common_templates` itself is unchanged — the corpus-wide template
frequencies are still computed from every county's `v2_text`, so short
counties' sentences still count toward what "common" means; only the *drop*
decision becomes conditional.

### 2. New `raw` re-embedding variant

`reembed_source_a.py` currently supports `--variant v2` and `--variant v3`
only; the true pre-cleaning baseline embeddings no longer exist as a
standalone file (v3 was renamed over `source_a_embeddings.parquet` at
adoption). Add `--variant raw` that embeds `raw_intro_text` with no cleaning
at all, so the original baseline is reproducible on demand from the current
repo state rather than recovered from git history. Reusable for any future
variant round, not single-purpose to this one.

### 3. New variant: `v4`

`build_embedding_texts` gets a fourth variant, `v4`: identical to `v3`
(regex cleaning + frequency filter) except the frequency-filter step uses
the outcome-gated logic from §1 instead of applying unconditionally.

### 4. Decision gate for `v4`

Evaluated via the existing `evaluate_source_a_variants.py` harness, run once
over `[raw, v3, v4]` parquets so all three are compared against the same
fixed analysis set in one pass.

**Carried over from the original gate** (v4 vs. `raw`, i.e., v4 must still
qualify as a real improvement over doing nothing):
1. `tracked_pair_mean` drops by ≥0.03 vs. `raw`
2. `pairwise_similarity_std` ≥ `raw`
3. Mantel `r < 0` with `p < 0.05`

**New, specific to this round's actual goal** (v4 vs. `v3`, i.e., v4 must
fix what v3 broke):
4. v4's worst `top_far_similar_pairs` entry must not exceed v3's (0.9607) —
   report where it lands relative to `raw`'s worst (0.8327) rather than
   requiring it beat that exact number, since some regression tolerance
   relative to `raw` is expected from any boilerplate removal at all
5. The Stutsman/Providence tracked pair must not have increased vs. `raw`
   the way it did under `v3` (+0.0071) — flat or decreased is acceptable

If v4 passes all five: adopt v4 as the new baseline, following the same
adoption mechanics as v3 (overwrite `source_a_embeddings.parquet`,
regenerate every downstream artifact listed in findings.md §9, append a new
dated round section to `source-a-findings.md` with the full metrics table,
gate check, and any new tradeoffs — same rigor as every prior round).

If v4 fails: record the negative result in `source-a-findings.md`, keep `v3`
as baseline, and note that the rural-outlier regression is a known,
undismissed limitation rather than silently dropping the finding.

### 5. Repo cleanup (independent of the v4 decision)

- Delete `source_a_embeddings_v2.parquet` and `source_a_embeddings_v3.parquet`
  (tracked, 21MB each) — `v2` lost the original adoption tiebreak, `v3` is
  now a byte-identical duplicate of the adopted baseline. Git history
  retains both if ever needed again, matching the precedent already set in
  findings.md when six prior round-report files were consolidated and
  removed.
- Delete `reembed_run.log` (917KB) and `variant_eval_run.log` (27KB) —
  untracked raw stdout; their useful content already lives in
  `analysis-output/variant-eval.json` and `source-a-findings.md`.
- Move `PLAN.md` and `SUMMARY.md` (untracked) into
  `docs/rounds/2026-07-07-deboilerplate/`, preserving them as a historical
  record without cluttering repo root.
- Apply the same "delete the loser" pattern to whichever of `v4`'s
  intermediate parquet or `v3`'s (if v4 is adopted) is no longer the live
  baseline, once the gate decision is made.

## Testing

- `boilerplate_frequency.py`: unit test that a county whose frequency-filtered
  text would fall below `MIN_CONTENT_LENGTH` keeps its unfiltered `v2` text
  instead (construct a short synthetic article where the only non-boilerplate
  sentence is very short).
- `reembed_source_a.py`: unit test for `build_embedding_texts(..., variant="raw")`
  returning `raw_intro_text` unchanged, and `variant="v4"` applying the
  outcome-gated skip for a short synthetic county while still filtering a
  long synthetic county normally.
- Existing test suite (21 tests per SUMMARY.md) must continue passing
  unchanged, since `v2`/`v3` behavior is not modified by this round.

## Out of scope

- No change to `strip_self_reference` or `strip_boilerplate_phrasing`
  (the regex layer) — only the frequency-filter step is gated.
- No attempt to fix rural counties' underlying thin Wikipedia coverage
  (§4/§11 of findings.md already flag this as a source-text limitation, not
  a cleaning-pipeline bug) — this round only stops the cleaning pipeline
  from making that thinness worse.
- No re-litigation of the `v2` vs. `v3` tiebreak — `v4` is built on top of
  `v3`'s approach, not a fresh alternative to it.
