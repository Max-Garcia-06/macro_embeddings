---
type: session-log
date: 2026-08-03
experiment_line: source-a
purpose: what-was-done-and-why
status: complete
companion_notebook: source_a_extraction_round.ipynb
canonical_findings: source-a-findings.md §13–§15
---

# Session Log — Source A: Replacing the Cut Embedding With Typed Extraction

Narrative record of one working session: what was asked, what was decided, what
was measured, and where the reasoning changed mid-flight. The canonical numbers
live in `source-a-findings.md` §13–§15; the presentation version with figures is
`source_a_extraction_round.ipynb`. This file exists to explain the *path*,
including the parts that went wrong, which neither of those captures.

---

## 1. The starting position

Source A turns each US county's Wikipedia article into features. Its original
mechanism — a 1,024-dimension `bge-m3` embedding of the article's lead section —
had been **cut**, and the cut rested on cost rather than absence of signal:

| | mean R² lift over a size+state baseline |
|---|---|
| `content_length` (character count) | +0.00098 |
| the 1024-dim embedding | +0.00273 |

The embedding won on 23 of 28 cross-pillar targets (Wilcoxon p = 4.2e-5), but for
a 2.2GB model download and CPU inference over 3,144 articles. Source A was left
shipping **one scalar**.

**The brief:** improve on that scalar.

**The hypothesis brought into the session:** county articles are heterogeneous —
some are boilerplate stubs, others have much more to give — so counties should be
split into 3–4 groups and each group handled differently.

---

## 2. Testing the hypothesis before building on it

The hypothesis was checked against the data before any design work. It held, and
sharpened into something more specific than "some articles are longer."

Splitting all 3,144 counties into content tiers and scanning for what their
articles actually contain:

| tier | n | names an industry | distinct proper nouns | says "named for/after" |
|---|---|---|---|---|
| stub (<100 chars) | 294 | 1.0% | 2.0 | 6.8% |
| thin (100–283) | 1,274 | 1.1% | 5.2 | 42.5% |
| mid (284–461) | 788 | 5.5% | 8.8 | 53.7% |
| rich (462+) | 788 | **25.2%** | 17.8 | 51.7% |

**Two readings drove every subsequent decision:**

1. **Economic content is 23× denser in the rich tier**, and only 6.5% of the
   corpus carries it at all. An embedding averaged over 3,144 articles is
   dominated by the ~93% with nothing economic to say. This is the quantitative
   form of the hypothesis, and it explains the embedding's weak performance
   mechanically rather than by assertion.
2. **Founding-narrative content is flat across thin/mid/rich.** It has mass
   everywhere and separates nothing — which is exactly what the earlier PCA
   analysis found its leading component to be (§3.2: a Texas-concentrated
   Wikipedia editorial artifact). Confirming this from a second angle explained
   why the embedding's strongest structure had been useless.

### A correction that had to be made mid-session

The first version of this table was **wrong**, and the error was caught by a
follow-up check rather than by the original analysis:

- Patterns were **case-sensitive**, so "Metropolitan Statistical Area" was missed
  while "metropolitan" was caught — undercounting metro attachment as 23% when it
  is 51%.
- Patterns lacked **word boundaries**, so `port` matched "im**port**ant",
  "air**port**", and "trans**port**" — inflating the industry rate to 19.7%.

The corrected gradient is **steeper** (23× rather than the originally reported
4.6×) but the absolute level is **much lower** (6.5% rather than 19.7%). Both
errors were recorded in `source-a-findings.md` §13.1 so the bad numbers cannot be
reused from the planning documents that quote them.

---

## 3. The design decision, and where it departed from the brief

The brief implied a **branching pipeline**: group counties, then extract
differently per group. The design that shipped instead uses **one uniform schema
for all counties**, with the tiers used to decide *what to extract* and to break
out results.

**Reasons given at the time:**

- A ragged per-group schema introduces a third null category — "not applicable to
  this county's group" — on top of "missing" and "zero". `PROJECT_GOAL.md`
  requires the downstream Comcast feature store to distinguish those, so ragged
  output is a real cost at handoff.
- Four extraction strategies means four things to maintain, validate, and check
  for precision.
- Sparsity already encodes the group. A stub county returns `False` across the
  board, and that emptiness *is* its tier membership — no branching needed to
  express it.
- Tier membership tracks county size (`content_length` correlates with metro
  status at r = 0.247), and whether size is a control or part of the target is an
  open project-level question. Keeping tiers out of the shipped schema avoids
  smuggling a size proxy into the features.

This departure was surfaced and approved before implementation, and **it was
tested rather than assumed** — see §7 below, which is the part of the original
brief that turned out to be genuinely answerable.

---

## 4. What was built

Four pieces, in dependency order:

| script | role |
|---|---|
| `ingest_source_a.py` (modified) | now persists every article section, not just the lead |
| `extract_source_a_features.py` (new) | 20 typed columns from the lead |
| `extract_source_a_section_features.py` (new) | industry lexicon over economy-titled sections |
| `analyze_source_a_tiers.py` (new) | tier assignment + composition report |
| `analyze_source_a_representation.py` (extended) | scores every variant on the existing 28-target harness |

**Conventions the extraction follows:**

- **Absence is `False`, never null.** Every flag is populated for every county, so
  a consumer can distinguish "this county has no university" from "we don't know".
  `founding_year` is the sole exception — genuinely unknown when no founding
  clause exists, so it stays nullable.
- **Extraction reads `raw_intro_text`, never `embedding_text`.** The corpus
  stripper that produces the latter removes the county name, the state name, and
  the phrase "U.S. state of" along with boilerplate:

  ```
  RAW:   Nelson County is a county in the U.S. state of North Dakota. As of the
         2020 census, the population was 3,015... county seat is Lakota.
  CLEAN: the population was 3,015, and was estimated to be 2,963 in 2025. Lakota .
  ```

  Anything built on the cleaned column works from damaged input. This was found
  by sampling texts across the length distribution, not by reading the code.

### Precision checking caught two broken features

Sampling each flag's actual matches — before scoring anything — found two that
were measuring the wrong thing entirely:

| flag | before | after | what was actually matching |
|---|---|---|---|
| `has_military_base` | 163 | 21 | "Fort Wayne", "Fort Yates" (city names), "Fort Lemhi" (1855 Mormon settlement), `Army` in Civil War prose |
| `has_tribal_land` | 157 | 79 | "American Indian Wars", reservations dissolved in the 1830s |

Five of six sampled `has_military_base` hits were false. Both patterns were
tightened to require an installation or present-tense land term. **This step is
what separates lexicon extraction from plausible-looking noise**, and it is the
main reason a regex approach is defensible here at all.

---

## 5. Reopening a question that had been closed

The largest wins came from named-industry features, which reached only 8.2% of
counties from the lead alone. An earlier round (§4/§4.1) had already tested
reading more of each article and **closed it as a negative result**.

Reopening it was justified on three specific grounds rather than optimism:

1. **The old result used a rejected yardstick.** §4/§4.1 measured Mantel-r against
   *geographic* distance. The project had already concluded — in
   `analyze_source_a_representation.py`'s own header — that geography-routed and
   Source-C-routed verdicts are broken yardsticks, and re-tested the embedding cut
   against the pillar matrix for exactly that reason. The same argument applies.
2. **Its finding penalizes embeddings far more than extraction.** §4 found body
   sections are *more templated* than the lead. Templating is fatal to a dense
   vector, which absorbs boilerplate into every dimension. Targeted extraction
   reads named facts and ignores prose.
3. **Its failure mode cannot occur here.** §4.1 diagnosed per-county
   inconsistency from an LLM cleaner keeping geography for some counties and
   dropping it for others. A fixed lexicon cannot be inconsistent that way.

**And the cost objection was based on a false premise.** `extract_article_html`
had always returned the *full* article body from the API; `isolate_lead_section`
discarded everything after the lead at parse time. The article bodies had been
downloaded and thrown away on every run since the pipeline was written. The
refetch cost the same 3,144 requests as the original ingest.

**Result:** 64,588 sections across all 3,144 counties, 20.5 per county, 0
failures. Industry coverage rose **8.2% → 18.8% (+332 counties)**.

The per-tier yield inverted the obvious expectation:

| tier | has economy section | industry in intro | **added by sections** |
|---|---|---|---|
| stub | 10.5% | 0.7% | +5.4% |
| thin | 14.2% | 1.1% | **+8.6%** (≈8× relative) |
| mid | 21.2% | 5.5% | +12.6% |
| rich | 35.7% | 25.3% | +13.7% |

Sections help most, in relative terms, for counties whose **lead says least**.

---

## 6. Results

Same protocol throughout: 28 targets from pillars B–F, unpenalized
size-plus-state baseline, each representation fitted to that baseline's
residuals, ridge penalty by nested crossvalidation, seed 42, 5 folds.

| variant | columns | mean R² lift | raw R² alone | beats incumbent | Wilcoxon p |
|---|---|---|---|---|---|
| `content_length` (incumbent) | 1 | +0.00117 | 0.020 | — | — |
| typed, minimal | 4 | +0.00254 | 0.042 | 13/28 | 0.493 |
| typed, mid | 8 | +0.00243 | 0.042 | 19/28 | 0.066 |
| typed, intro only | 20 | +0.00263 | 0.044 | 16/28 | 0.339 |
| **typed + economy section** | **29** | **+0.00320** | 0.048 | 19/28 | 0.082 |
| typed × tier | 120 | +0.00265 | 0.042 | 19/28 | 0.066 |
| `bge-m3` PCA-50 | 50 | +0.00171 | 0.085 | 13/28 | 0.678 |
| `bge-m3` full | 1024 | +0.00273 | 0.112 | 19/28 | 0.014 |

**29 readable columns exceed the 1,024-dimension embedding**, at 2.7× the
incumbent scalar and no model download.

### The caveat that does not go away

The paired Wilcoxon is **p = 0.082**, short of 0.05. Mean difference against the
incumbent is +0.00203 while the **median is +0.00061** — a mean more than three
times the median, meaning extraction wins large on a handful of targets and only
marginally on the rest. The rank-based test weighs consistency rather than
magnitude, so it sees the marginal wins, not the large ones. The embedding shows
the opposite profile: smaller gains spread more evenly, which is what makes its
test significant at p = 0.014.

**Extraction is the better and far cheaper representation on average. It is not
uniformly better target by target.** The findings file forbids the stronger claim
explicitly.

### Evidence the wins are real rather than fitted

| | mean Accommodation & Food Services LQ | n |
|---|---|---|
| article mentions tourism | **1.407** | 78 |
| article does not | 1.010 | 2,291 |

Counties whose Wikipedia article mentions resorts, skiing, or casinos genuinely
have more tourism-sector employment (r = 0.157) — and that sector is the single
largest target gain in the sweep. The mechanism is visible, not inferred.

### One feature was a size proxy in disguise

`n_body_sections` correlates with county size at r = 0.550 — higher than
`content_length`'s 0.359, making it the most size-dependent column in Source A.
An ablation isolated it:

| feature set | mean lift |
|---|---|
| intro only (20 cols) | +0.00263 |
| + sections, including `n_body_sections` (30) | +0.00328 |
| + sections, `n_body_sections` dropped (29) | +0.00320 |
| + sections, both structural columns dropped (28) | +0.00320 |

**97.6% of the section gain survives removing it.** It ships as a diagnostic, not
a feature — 2.4% of the gain does not justify that much size dependence in a
feature set whose central open question is whether size is a control or a target.

---

## 7. The original brief, tested directly

The design in §3 departed from the brief by not branching per group. That
departure was then **tested rather than left as an assertion**, in both available
forms:

| approach | width | mean R² lift |
|---|---|---|
| one model, one global coefficient per feature | 29 cols | **+0.00320** |
| one model, coefficients free to vary by tier | 120 cols | +0.00265 |
| four independent models, one per tier | 29 cols × 4 fits | **−0.01595** |

**Both branching forms lose, and the loss scales with how much branching there
is.** Tier-specific slopes cost 17% of the lift. Fully separate per-tier models go
*negative* — worse than dropping Source A entirely.

The mechanism is ordinary bias–variance: crossing 29 features with 4 tiers puts
120 columns against targets whose smallest sample is n ≈ 1,026, and the ridge
penalty large enough to control that width over-shrinks the coefficients doing the
work. Splitting the fit entirely removes even the shared penalty's protection.

**This is easy to read backwards.** The tier structure was real and it mattered —
the 23× industry gradient identified industry as the feature family worth
building, and the per-tier yield justified refetching article bodies. **The tiers
were the right diagnostic; they are not the right architecture.** Heterogeneity is
better handled by features that are simply absent when a county has nothing to
say, because sparsity already encodes the tier and the model has more data to
learn from when it is not partitioned.

---

## 8. What ships

**`extracted_sections`** — 29 interpretable columns, one uniform schema across all
3,144 counties, absence as `False`.

Written into `data/source_a_text_features.parquet`, flowing into
`pillar_matrix.build_matrix()` as Source A's block (31 columns including
diagnostics) with **no change to that module** — it already forwarded every Source
A column except the two raw-text ones.

### Open items

1. **p = 0.082 is a judgment call, not a result.** If the bar is "better on
   average at far lower cost", this is done. If it is "reliably better across the
   board", it is not, and the honest next move is more targets rather than more
   features.
2. **The pillar-pair sweep needs re-checking.** Source A's block went from 1 to 31
   columns, and `analyze_pillar_matrix_signal.py` uses Source A on *both* the
   predictor and target sides. Whether the new columns should be excluded there
   should be settled before that sweep is trusted again.
3. **The refetch pulled live Wikipedia three weeks newer**, shifting figures
   slightly (`content_length` mean 388.1 → 390.0, incumbent lift 0.00098 →
   0.00117). The embedding's +0.00273 comes from a frozen July parquet, so it
   remains a valid reference point but is measured against marginally different
   text than the extraction variants.
4. **Unrelated, noted not touched:** unused imports in
   `analyze_source_e_source_c_correlation.py:27` and
   `generate_source_e_insights.py:25`.

### Reproduction

Seed 42 throughout. The ingest step rewrites `source_a_text_features.parquet`, so
the extraction steps must follow it.

```bash
uv run --env-file .env python scripts/ingest_source_a.py          # ~16 min, 3,144 articles
uv run python scripts/extract_source_a_features.py
uv run python scripts/extract_source_a_section_features.py
uv run python scripts/analyze_source_a_tiers.py
uv run python scripts/analyze_source_a_representation.py          # ~5 min, 8 variants
```

### Artifacts

| path | contents |
|---|---|
| `data/source_a_sections.parquet` | 64,588 article sections, all 3,144 counties |
| `data/source_a_text_features.parquet` | text + 30 extracted columns |
| `outputs/source_a_representation.csv` | per-target lift, every variant |
| `outputs/source_a_representation_by_tier.csv` | per-tier breakout |
| `outputs/source_a_tiers.csv` | per-county tier assignment |
| `analysis-output/source-a/source_a_*_stats.json` | extraction, tier, section, representation summaries |
| `analysis-output/source-a/source_a_extraction_round.ipynb` | figures and presentation |
| `analysis-output/source-a/source-a-findings.md` | §13–§15, canonical claims and forbidden wording |
