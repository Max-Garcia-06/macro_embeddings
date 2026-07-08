# v5: LLM-Assisted Review of Frequency-Filter Drops — Design

**Status:** approved by user, pending implementation plan.

## Motivation

`boilerplate_frequency.py` (v3's filter) drops any sentence whose masked
template (numbers → `<NUM>`, proper-noun runs → `<NAME>`) recurs across
`>=5%` of counties. This is a lexical proxy for "boilerplate" — it cannot
distinguish a truly generic sentence from a sentence that happens to share
a common shape but carries a specific county's only real content.

`analysis-output/source-a-findings.md` §12 documents the resulting
over-stripping failure via the Stutsman County, ND ↔ Providence County, RI
tracked pair (similarity 0.8327 vs. real-baseline 0.8255). A follow-up
round (`v4`, this doc's predecessor) tried to patch this with an outcome
gate: keep the unfiltered v2 text if the v3-filtered text falls below
`MIN_CONTENT_LENGTH` (100 chars). **v4 was rejected** — 4 of 5
pre-registered gate criteria passed, but the length floor never engaged
for the Stutsman/Providence pair (its filtered text never dropped below
100 chars), so that pair was identical under v3 and v4. The real problem
was never length; it was that the frequency filter conflates "common
template" with "no content," and a length floor can't fix a content
problem.

**Goal:** replace the length-floor heuristic with direct semantic
judgment. For every county, ask a local Gemma model to look at each
sentence the frequency filter would drop and decide whether it adds
county-specific information the kept text doesn't already have. Restore
sentences it judges non-boilerplate. This can only add text back relative
to v3 — it never strips more than v3 already does.

## Non-Goals

- Not a general-purpose LLM rewrite/summary of intro text (rejected in
  favor of extractive keep/drop — see options below).
- Not a full-corpus reprocessing change to v2's regex stripping or v3's
  frequency-filter threshold; those stages are reused unchanged.
- Not a replacement for the existing evaluation gate; v5 must clear the
  same style of pre-registered criteria as v2/v3/v4 did.
- Economic ground-truth validation (Source E/B) remains out of scope, per
  the original plan's Scope and Assumptions §2.

## Options Considered

1. **Local Gemma, per-county batched, extractive keep/drop (chosen).**
   One Gemma call per county with ≥1 dropped sentence; sees the kept text
   as context plus the literal dropped sentences; returns a keep/drop
   verdict per sentence. No new wording is ever introduced (output is
   always a subset of original Wikipedia sentences), which keeps this
   auditable against the source text and keeps the "embedding input is
   Wikipedia text" property the pipeline has had since v1.
2. **Hosted API (Gemini/similar).** Rejected: adds a paid, networked
   dependency and weakens run-to-run reproducibility guarantees versus a
   pinned local checkpoint at `temperature=0`.
3. **Abstractive rewrite/summary.** Rejected: introduces paraphrase and
   hallucination risk, and produces embedding input that is no longer
   literal source text, a much bigger departure from this pipeline's
   character than extractive filtering.
4. **Length-floor threshold, re-tuned (v4's own suggested next step).**
   Rejected: same class of magic-number risk that got v4 rejected in the
   first place, just with a different number; doesn't address the root
   cause (frequency filter conflating shape with content).
5. **Per-sentence isolated Gemma calls (no county context).** Rejected:
   loses the context needed to judge "generic and redundant given what's
   already kept" vs. "generic-shaped but this county's only real detail,"
   and multiplies call count for counties with several dropped sentences.

## Architecture

### Trigger & scope

Runs for every county with ≥1 sentence flagged by `find_common_templates`
in the existing v3 pipeline (most counties). Counties with zero dropped
sentences are untouched and byte-identical to v3 — no LLM call is made for
them.

### New module: `llm_boilerplate_review.py`

Mirrors the existing project convention (pure-logic module, importable
without network/model side effects at import time, matching
`text_cleaning.py` and `boilerplate_frequency.py`'s shape):

```python
def build_review_prompt(kept_text: str, dropped_sentences: list[str]) -> str
    """Build the Gemma prompt: county's kept (v3) text as context, plus the
    literal dropped sentences, requesting a JSON keep/drop verdict per
    sentence index."""

def parse_review_response(raw: str, n_sentences: int) -> list[bool]
    """Parse and validate the model's JSON response into one bool per
    dropped sentence (True = restore). Raises ValueError on malformed
    output or a length mismatch; caller handles the fallback."""

def review_dropped_sentences(
    kept_text: str,
    dropped_sentences: list[str],
    client: GemmaClient,
) -> list[str]
    """Return the subset of dropped_sentences to restore. Looks up each
    (kept_text, sentence) pair in the on-disk cache first; only calls
    client on a cache miss."""
```

`GemmaClient` wraps the local Ollama call (model `gemma2:9b`, pinned by
tag/digest, `temperature=0`). Its interface is a single `generate(prompt:
str) -> str` method so it can be swapped for a fake in tests.

### Determinism & caching

- Pinned Gemma checkpoint (tagged digest, not `:latest`), `temperature=0`.
- Every verdict is cached in `analysis-output/llm_review_cache.jsonl`,
  keyed by a hash of `(kept_text, dropped_sentence)`. Re-running the eval
  harness or a partial rebuild reuses cached verdicts instead of
  re-querying the model. This file doubles as the audit trail for what
  Gemma decided and why, matching the project's existing evidence-logging
  pattern (`analysis-output/variant-eval.json`).

### Pipeline integration

Add `variant == "v5"` to `reembed_source_a.py:build_embedding_texts`:

1. Compute v2 text and the frequency-filter template set exactly as v3/v4
   already do.
2. For each county, identify which sentences the template set would drop
   (not just the kept text — need the actual dropped sentence list, which
   today only exists implicitly inside `drop_common_sentences`). Add a
   small helper alongside the existing filter (in
   `boilerplate_frequency.py` or the new module) that returns the dropped
   sentences for a text, without changing `drop_common_sentences`'s
   existing signature or behavior.
3. Call `review_dropped_sentences` for counties with ≥1 dropped sentence.
4. Reconstruct the final text preserving **original sentence order**
   (kept-by-frequency-filter ∪ restored-by-Gemma) — not appended at the
   end.

### Error handling

If a Gemma call fails (timeout, model unavailable) or returns unparseable
output for a county, that county falls back to plain v3 behavior (all
disputed sentences stay dropped) — consistent with this codebase's
existing "never fail the run, never regress below the last known-good
baseline" pattern (`clean_for_embedding`'s three-tier fallback, v4's
per-county gating). Failures are counted and logged once at the end of the
run, not per-county.

## Evaluation Plan

Reuse `evaluate_source_a_variants.py` unchanged —
`source_a_embeddings_v5.parquet` gated against the real §12 historical
baseline (`tracked_pair_mean=0.82926`, `pairwise_similarity_std=0.06251`,
`mantel_r=-0.12171`) with the same first four criteria v4 was gated on:

1. `tracked_pair_mean` drops by ≥0.03 vs. real baseline.
2. `pairwise_similarity_std` ≥ real baseline.
3. Mantel `r < 0`, `p < 0.05`.
4. Worst `top_far_similar_pairs` entry ≤ v3's worst (0.9607).

**Plus one new pre-registered criterion specific to this round's
motivation:**

5. Stutsman ND ↔ Providence RI tracked-pair similarity must **decrease**
   relative to v3's 0.8327 (not merely "not increase," which is the
   weaker bar v4 already cleared trivially by being numerically identical
   to v3). Failure to improve on this pair is this round's rejection
   condition — it is the specific case this round exists to fix, mirroring
   v4's own rejection postmortem in §12's Next Actions.

Per this experiment line's standing rule, if any criterion fails, v5 is
rejected and the baseline remains v3.

## Testing Plan

- Unit tests for `build_review_prompt` / `parse_review_response` using a
  fake `GemmaClient` (no real model calls in CI) — malformed JSON, length
  mismatch, well-formed response cases.
- Test confirming reconstructed text preserves original sentence order
  (kept + restored, not restored-appended-at-end).
- Regression test: v5 output equals v3 output byte-for-byte for any county
  with zero dropped sentences.
- Cache test: a repeated `(kept_text, dropped_sentence)` pair hits the
  cache and does not call the client a second time.

## Open Risks / Assumptions

- Requires Ollama (or equivalent local runtime) with `gemma2:9b` (or
  chosen checkpoint) downloaded and available on the machine running
  `reembed_source_a.py --variant v5`; this is a new operational
  dependency beyond `sentence-transformers`/`bge-m3`.
- CPU inference cost for ~3,144 county-level calls (fewer if many counties
  have zero dropped sentences) is unknown until measured; if prohibitively
  slow, may need batching or a smaller checkpoint — to be assessed at
  implementation/execution time, not blocking this design.
- Prompt-following reliability (valid per-sentence JSON, no drift) is a
  known open risk for smaller local checkpoints; the parse-and-fallback
  design absorbs failures without blocking the run, but a high failure
  rate would undermine the round's evaluation validity and should be
  reported alongside the gate result.
