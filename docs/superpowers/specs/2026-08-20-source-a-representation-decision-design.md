# Design — Breaking the Source A representation tie, 20 August 2026

**Deliverable:** a pre-registered, non-circular, width-matched head-to-head between
Source A's 29 typed columns and a MiniLM encoding of the same articles, on the
measurement that governs Source A's slot in the matrix — marginal contribution
against external targets — with both arms given their best honest shot first.

## The question, and why the current answer is not one

Two measurements exist and they disagree.

| measurement | typed (29 col) | MiniLM `uniform_l2` (384) | verdict |
|---|---|---|---|
| standalone cross-pillar lift, 28 in-repo targets | +0.00307 | +0.00351 | tie (median −0.00001, 14/28, p = 0.762) |
| marginal contribution, 5 external ACS targets | −0.00005 | −0.03499 | typed, by ~700× |

Neither settles the shipping decision.

The first is **circular and underpowered**. Every one of its 28 targets is another
pillar's feature and 22 of the 28 are BLS location quotients from one table, so the
basket penalizes a source for agreeing with the pillars it ships alongside
(`ingest_external_targets.py` module docstring). Power is 0.51 at the observed
dz = 0.323; 61 targets are needed for 0.80 (`source_a_next_steps.md`).

The second is **thin and confounded**. Five targets, and the embedding arms carry 384
columns against the typed block's 29, so an unknown share of the penalty is width
rather than content. `source-a-findings.md` §21.2 states this in writing and calls
the missing PCA arm out by name.

So the shipping decision currently rests on cost and interpretability alone. That is
a defensible position but it is not evidence, and §20.3 says so explicitly.

## What this design adds that is new

Three things, of which the second was not previously known to anyone.

### 1. The encoder's input is mostly tables and lists

Measured over `data/source_a_sections.parquet` — 64,588 sections, 54.8M characters —
by share of characters the `uniform` arm reads:

| category | % of input | counties |
|---|---|---|
| **census tables** (2020/2010/2000 census, demographics, racial composition) | **36.4** | 3,142 |
| other | 21.0 | 2,659 |
| narrative (history, notable people) | 13.6 | 2,695 |
| geography / climate | 7.3 | 3,118 |
| government / politics | 7.0 | 3,022 |
| place lists (communities, cities, townships, ghost towns, adjacent counties) | 6.7 | 3,081 |
| education | 3.5 | 2,272 |
| highways / transport | 3.1 | 2,757 |
| **economy-titled** | **1.5** | **660** |

Roughly 46% of what the encoder reads is census tables, place-name lists, and highway
lists. Economy-titled sections are 1.5% of the text and exist for only 660 of 3,144
counties (21%). Under mean pooling the 1.5% is averaged into the 46%.

This is a concrete mechanism for the encoder underperforming, and it also explains why
`ECONOMY_TITLE_PATTERN` applied everywhere cannot be the fix on its own: 79% of
counties would fall back to their lead.

### 2. The encoder reads the answer; the typed block does not

Wikipedia census sections restate the external targets verbatim. Actual text from the
parquet:

> `2020 census As of the 2020 census, the county had a population of 58,805. The
> median age was 38.9 years.`

`median_age` is one of the five external targets. 2,589 counties carry a census
section mentioning median income. The typed block cannot exploit this — it is regex
lexicon counts, no numeric extraction — but the encoder's input contains it.

The repo already guards the pillar side of this problem: `TARGET_RESTATEMENTS` in
`analyze_external_target.py` ablates `wage_per_return_thousands` against
`median_household_income` and `retirement_destination` against `median_age`, and
reports every target twice, with and without. **Nothing currently guards the Source A
text side.** Expanding the external basket without this control would build the
leakage channel out rather than closing it.

Consequence: dropping census sections is a **leakage control**, not a tuning choice,
and it is a precondition for Part 1 rather than an arm to be scored against it.

### 3. The power problem belongs to the other axis

The 61-targets-for-80%-power figure is computed at dz = 0.323, the effect size of the
*standalone lift* comparison. The marginal comparison's observed effect is −0.00005
against −0.03499 — three orders of magnitude, sign-consistent across all five targets.
An expansion sized for the standalone axis is therefore generous for the marginal
axis, which is the axis the decision rests on. Both are powered here, and the
distinction is reported rather than blurred.

---

## Part 1 — Expand the external basket, 5 → ~60 targets

`ingest_external_targets.py` gains ~55 `ExternalTarget` entries. Same keyless
`www2.census.gov` table-based summary file path, same `_se` companion per target, same
ACS 2023 5-year vintage. Measured cost: ~118MB and ~4s per table download; the cache
retains only county rows and selected columns, so disk footprint stays small.

**Composition rule.** The current 28-target basket's defect is that 71% of it is one
BLS table. The new basket is capped at **no more than 6 targets per ACS table family**
and must span at least 8 constructs: income, housing cost, housing stock, education,
labour-force structure, household composition, technology access, and commuting.

**Line-numbering risk, and how it is managed.** Asserting that `B17001_E002` is the
below-poverty count is exactly the class of error that produces a confident wrong
number. Two rules:

1. **Prefer published medians and means at line `_E001`.** These need no hierarchy
   assumption and no division. B19013, B19301, B25064, B25058, B25105, B01002, B25035,
   B19113 and their siblings are all of this form.
2. **Every proportion target is verified against Autauga County, AL** before it is
   admitted, the same check the existing five carry in comments — the parts must sum
   to the published universe. A target whose arithmetic does not reconcile is dropped,
   not patched.

**Circularity screen.** Each target is checked against all six pillars and admitted,
ablated, or rejected:

- **Rejected outright** — county unemployment rate (Source C measures it directly),
  any sector employment share (Source B location quotients), freight or logistics
  volume (Source D).
- **Admitted with a `TARGET_RESTATEMENTS` entry** — poverty rate against Source F's
  `persistent_poverty`; educational attainment against `low_education`; housing cost
  burden against `housing_stress`; age-65+ share against `retirement_destination`.
  These are related but not definitional, which is the standard the existing
  `housing_stress` comment already sets. Reported twice, with and without, and the
  ablated figure is the defensible one.
- **Admitted clean** — everything else.

**Text-leakage screen.** Each candidate target is regex-searched against the census
sections of `source_a_sections.parquet`. Targets restated in the article text are
retained but **tagged `restated_in_text`**, and every headline figure is reported both
over the full basket and over the clean subset. If the two disagree, the clean subset
governs.

## Part 2 — Width control

The marginal harness gains `uniform_pca29` — an exact 29-against-29 match with the
typed block — alongside `uniform_pca64`. `uniform` and `uniform_l2` stay as the
unreduced references.

This closes the confound §21.2 states in writing. Until it is closed, the 700× gap
cannot be attributed to content.

## Part 3 — The embedding's best honest shot

A typed win over a strawman encoder decides nothing. Two changes, then five scopes.

**Common-component removal.** Mean-pooled sentence embeddings over a corpus of
near-identical documents concentrate a large shared component; all 3,144 articles open
with the same template sentence. Subtract the corpus mean vector before scoring. No
re-encode, effectively free.

**Scopes**, motivated by the character-share table:

| arm | rule |
|---|---|
| `uniform` | everything — reference, and the leakage-carrying arm |
| `prose_only` | drop census tables, place lists, highways — **the leakage-controlled arm** |
| `prose_plus_history` | `prose_only` plus history and notable people |
| `economy_all_tiers` | `ECONOMY_TITLE_PATTERN` everywhere, lead fallback |
| `prose_by_tier` | rich and mid read `prose_only`; stub and thin read lead only |

`prose_plus_history` exists because the typed block excluded history deliberately — a
lexicon hit inside a History section is usually a *defunct* industry
(`analyze_source_a_section_scope.py`) — and that reasoning may not transfer to an
encoder that reads the surrounding sentence. History is 13.6% of the text. Measured
rather than assumed.

`prose_by_tier` is a different hypothesis from the two tier arms already on file.
`tier_conditional` and `tier_conditional_inverse` branched on **depth** and both lost
to `uniform`. This branches on **content availability**: a stub county's body is
almost entirely census table and place list, so there is no prose there to read. If it
also loses, the tier question is closed by three independent tests rather than two.

## Part 4 — The typed block's best honest shot

Both arms are scored under ridge, so 29 columns against 384 is not an equal-capacity
comparison. One pre-registered transform pass: `log1p` on the count columns, and a
`sec_n_industry_mentions` × tier interaction.

Pre-registered means written into this document before the harness runs, chosen from
the columns' construction rather than from their per-target scores. No target peeking.
This equalizes capacity; it does not tilt the comparison.

## Part 5 — Pre-registered decision rule

Parts 3 and 4 introduce roughly ten scored arms. Without a rule fixed in advance that
is a multiple-comparisons machine, and the tie would be broken by selection rather
than by evidence.

Fixed before any scoring runs:

1. **Primary metric** — mean marginal contribution across the expanded external
   basket, drop-one against a model holding county size and pillars B–F, out-of-fold
   on held-out states.
2. **Exactly two arms enter the primary comparison, both fixed before scoring.**
   - *Typed arm*: the Part 4 transformed block. It supersedes the raw 29 columns
     rather than competing with them — Part 4 exists to equalize capacity, so
     entering both would reintroduce the selection problem rule 5 is written to
     prevent. The raw block is reported in the secondary table for continuity with
     §21.
   - *Embedding arm*: the winning Part 3 scope, PCA-reduced to **29 dimensions** to
     match the typed arm's width exactly. Scope selection runs on the *in-repo
     28-target* basket, which is disjoint from the decision basket, so selection and
     decision never touch the same targets. `uniform_pca29` is scored too, but as the
     unselected reference — it is the width-matched twin of the leakage-carrying arm
     and is not eligible to be the embedding representative.
3. **Secondary** — paired Wilcoxon across the expanded basket, reported with its
   observed power, plus the full arm table for the record.
4. **Tie threshold** — a difference whose 95% CI includes zero is reported as a tie
   and the decision falls back to cost and interpretability, explicitly and in those
   words. A tie is a permitted outcome of this design, not a failure of it.
5. **Reporting split** — every headline reported over the full basket and over the
   `restated_in_text`-clean subset.

## What this can and cannot answer

**Can:** whether Source A's marginal contribution depends on its representation, at a
target count that can resolve the question, with width controlled and the text-leakage
channel closed.

**Cannot:** whether either representation helps the actual downstream FreeWheel model.
No downstream label is obtainable — that is a structural constraint of this project
(`PROJECT_GOAL.md`), not an omission. ACS targets are a public proxy and the writeup
must say so.

**Cannot:** rescue Source A if its near-zero contribution is a fact about the pillar.
§21.3 already concluded that it is. This design tests that conclusion against a
stronger encoder and a wider basket; it may confirm it rather than overturn it, and
confirming it is a result.

## Verification

- Existing five targets reproduce their published contributions after the basket
  expansion — same values, since the harness scores per target.
- `outputs/source_a_tiered_embedding.csv` unchanged for the arms that already exist.
- Every proportion target reconciles against Autauga County, AL.
- Scope selection (rule 2) demonstrably run on the 28-target basket, decision on the
  external basket, with no target appearing in both.
- Power reported at the observed effect size for both axes, not assumed.

## Phasing

Part 1 is a prerequisite for everything else meaning anything — at n = 5 no arm from
Parts 3 or 4 can be told from noise. Build order: Part 1, then Part 2, then Parts 3
and 4 in parallel, then Part 5 applied once to the finished arm table.
