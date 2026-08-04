# Source A: Next Steps — Resolving the Two Open Items

## Context

Source A ships 29 interpretable columns extracted from county Wikipedia articles
(`analysis-output/source-a/source-a-findings.md` §13–§17). The work is complete
and the shipping decision is made. Two open items remain, recorded in §14.5,
§17.3, and the session log's §8:

1. **`p = 0.082`** on the paired comparison of typed extraction against the
   incumbent `content_length` scalar — short of 0.05, described in the findings
   file as "a judgment call, not a result."
2. **Every target is another pillar's feature.** The 28-target harness is the
   closest in-repo proxy for downstream usefulness, not an answer to it.

This document does not re-derive §13–§17. It states what those two items
actually are once the test's statistical properties and the target basket's
composition are examined, lays out five possible responses, and lists the
questions whose answers select among them.

Everything below is computed from `outputs/source_a_representation.csv` and
`outputs/source_a_marginal.csv` at seed 42.

---

## What the two open items actually are

### Item 1 is an underpowered test, not an ambiguous result

Paired difference, `extracted_sections` (29 columns) minus `content_length`,
thin baseline, 28 targets:

| statistic | value |
|---|---|
| mean | +0.00203 |
| median | +0.00061 |
| sd | 0.00605 |
| Cohen dz | 0.335 |
| **power at n = 28, α = 0.05 one-sided** | **0.53** |
| n targets for 80% power | 57 |
| n targets for 90% power | 78 |

Power of 0.53 means that if the observed effect is the true effect, this test
detects it about half the time. `p = 0.082` is the ordinary output of a real
effect measured with roughly half the sample the effect size requires. It is not
evidence against the effect, and it should not be written up as though the
question were close.

**This is not a test-choice problem.** The suspicion that Wilcoxon misses the
result because it discards magnitude was checked directly:

| test | p |
|---|---|
| Wilcoxon signed-rank | 0.0815 |
| paired t (magnitude-weighted) | 0.0873 |

Both land in the same place. Switching to a magnitude-weighted test changes the
third decimal. Any proposal to "use a better test" is proposing a rounding
error.

**The mean is carried by five targets.** Mean is 3.3× the median because the
distribution is concentrated:

| target | `content_length` | typed | difference |
|---|---|---|---|
| Accommodation & Food Services LQ | −0.00001 | +0.02618 | **+0.02619** |
| demographic distress count | +0.00803 | +0.02158 | +0.01356 |
| Information LQ | −0.00514 | +0.00269 | +0.00783 |
| capital-to-wage ratio | +0.00090 | +0.00680 | +0.00590 |
| Transportation & Warehousing LQ | +0.00135 | +0.00563 | +0.00428 |

Losses are similarly concentrated: Professional Services −0.00653, Retail Trade
−0.00294, Educational Services −0.00284. Dropping Accommodation alone roughly
halves the mean. Its mechanism is documented and independently verified (§13.4:
counties whose articles mention tourism average 1.407 Accommodation LQ against
1.010 for those that do not, r = 0.157), so the concentration is explicable
rather than suspicious — but the headline rests heavily on one target.

**A number not yet recorded anywhere.** Typed extraction against the `bge-m3`
embedding: mean difference +0.00047, wins **13 of 28**, Wilcoxon **p = 0.76**.
By rank the two are a dead tie. §14.5 already forbids "significantly beats the
embedding," which is correct, but the true relationship is weaker than that
phrasing implies: typed extraction is *statistically indistinguishable* from the
1024-dim embedding and wins on cost, interpretability, and the absence of a
2.2GB model download. Those are the defensible arguments. The +0.00320 against
+0.00273 gap is noise and should not be leaned on.

### Item 2 has a structural component that is not yet written down

**The 28 targets are not 28 independent targets.**

| target pillar | count |
|---|---|
| **B (QCEW location quotients)** | **20** |
| C (velocity series) | 3 |
| D (freight) | 3 |
| E (capital-to-wage) | 1 |
| F (typology) | 1 |

Seventy-one percent of the target basket is a single table, and QCEW location
quotients are compositional — each is a share measured against a national base,
so they are mechanically coupled. Both the Wilcoxon and the paired t treat these
as 28 independent draws. Effective n is materially lower, plausibly nearer 8–12.

This makes Item 1 worse rather than better: the "57 targets for 80% power"
figure is computed on nominal n, and the effective n behind the current 28 is
far below 28. It also means every published table's claim to 28-target breadth
overstates the evidence.

**The crowded-baseline test is the well-powered one.** §17's marginal analysis,
which asks whether Source A adds anything over a baseline already containing
every other pillar, has better properties than the headline test:

| test | dz | power at n = 28 | p |
|---|---|---|---|
| typed block vs zero, crowded baseline | 0.549 | **0.88** | 0.013 |
| `content_length` vs zero, crowded baseline | 0.339 | 0.54 | 0.014 |
| typed vs scalar, crowded baseline | 0.264 | 0.39 | 0.295 |

Two consequences:

- **"Source A carries marginal value over all other pillars combined" is a
  well-powered, significant result** (power 0.88, p = 0.013). This is the
  load-bearing finding of the whole experiment line and it is solid.
- **"The typed block beats the scalar" is powered at 0.39 and needs 91 targets.**
  It will not be settled in this repo by adding targets. The typed block ships
  on cost and interpretability, not on that comparison.

**Retention varies by an order of magnitude across the basket**, which means the
headline +0.0010 is an artifact of the basket's composition:

| target pillar | thin lift | crowded lift | retained |
|---|---|---|---|
| C (velocities) | +0.00171 | +0.00118 | **69%** |
| D (freight) | +0.00212 | +0.00133 | 63% |
| B (QCEW) | +0.00317 | +0.00087 | 28% |
| F (typology) | +0.02172 | +0.00401 | 19% |
| E (capital-to-wage) | +0.00786 | +0.00002 | **0.2%** |

Retention is highest where the other pillars know least — Source C's velocity
series are near-orthogonal to county size and to the rest of the matrix — and
collapses where a federal agency measures the same construct directly. Since the
basket is 71% QCEW, the worst-retaining large block, **the single published
number is a property of the target mix, not of Source A.** Publishing +0.0010
without the composition alongside it is the most likely way to mislead a
downstream reader.

**What the proxy structurally cannot see.** Because every target is a pillar
feature, a source is penalized precisely for agreeing with the pillars it will
ship alongside. For assembling a non-redundant feature store that is arguably
the correct penalty. For predicting an external outcome it is not: Source A and
Source F can be redundant with each other and both predictive of churn. Relatedly,
`has_metro_attachment` is ablated in the sweeps as a restatement of Source F's
`metro_2023` (§16.2) — justified for pillar-versus-pillar work, unjustified
against an external target, where it would be a legitimate free feature.

---

## Five plans

### Plan 1 — Declare done, stop measuring

Ship the 29 columns. Publish +0.0010 with its caveats. Close both items as
answered as well as this repo can answer them.

- **Cost:** zero.
- **Buys:** no further evidence.
- **Case for it:** the shipping decision is already made and no p-value changes
  it. The block costs one regex pass and no model download; the cost of shipping
  it if it is worthless is 29 sparse boolean columns of clutter. That asymmetry
  is large enough that the significance test is being asked to gate a decision it
  does not gate.
- **Risk:** +0.0010 travels downstream carrying more authority than a
  71%-single-table basket earns.

### Plan 2 — Fix effective-n reporting

Do not chase power. Stop overstating the power already in hand.

- Report per-pillar breakouts as the primary result, aggregate as secondary.
- Cluster or block the significance test by pillar, or state effective n
  explicitly alongside nominal n.
- Add the power figures above to §14.5 and §17.3 so `p = 0.082` reads as
  "underpowered" rather than "failed."
- Record the typed-vs-embedding rank tie (13/28, p = 0.76).
- Refresh `docs/downstream_target_assumptions.md`, which predates §13–§17: it
  still states "A | No change (stays cut)" and still cites 41 feature pairs
  against the current 50.

- **Cost:** roughly half a day, documentation and one analysis-script change.
- **Buys:** every published claim becomes defensible. No new signal.

### Plan 3 — Expand the in-repo target set

Push 28 → 45–55 targets: QCEW wage and establishment columns, Source E's
capital-composition columns, Source F's flags split out of `distress_count`,
additional Source C series.

- **Cost:** 1–2 days.
- **Buys:** nominal power, and dilutes B's 71% dominance — the more valuable of
  the two effects.
- **Ceiling:** new targets are still pillar features and still correlated.
  Effective n grows far more slowly than nominal n. The plausible end state is
  80% nominal power at an effective n near 15.
- **Does nothing for Item 2.** The basket remains a proxy.

### Plan 4 — Ingest an external county-level target

A public, county-keyed outcome outside all six pillars. Candidates: ACS
household internet subscription rate, FCC broadband adoption, Census Business
Formation Statistics, IRS SOI migration flows, County Health Rankings.

- **Cost:** 2–4 days, one new ingest script plus a scoring harness.
- **Buys:** the only plan that attacks Item 2 at all, and it resolves Item 1 as a
  side effect — external targets are uncorrelated with QCEW, so they add real
  effective n rather than nominal n.
- **Nearest analogue to the actual consumer:** broadband or internet adoption
  sits in the same domain as a Comcast downstream model.
- **Answers a question no in-repo test can:** does Source A carry information
  about a non-pillar outcome.
- **Risk:** a public proxy is still not the real label. It can produce a
  confident answer to a slightly wrong question.

### Plan 5 — Obtain a downstream label

The only thing that settles Item 2. Outside current repo scope. Even a
500-county sample carrying any real outcome would outweigh every in-repo test
combined.

- **Cost:** not a repo cost. An organizational ask.
- **Buys:** the actual answer.
- **Blocker:** `docs/downstream_target_assumptions.md` exists because no target
  has been supplied, and the project is not scoped to obtain one.

---

## Questions that decide which plan

Ordered by how much the answer changes the work.

### 1. Would anything be built differently if `p` were 0.03 instead of 0.082?

If nothing changes — and the block appears to ship either way — then Item 1 is a
documentation problem rather than a research problem, and Plan 2 closes it.
Spending Plan 3's effort to move a number that gates no decision is the most
likely way to waste the next week.

### 2. Is there any path to a real downstream label, at any horizon?

- **Yes, with meaningful probability** → every in-repo test is provisional.
  Minimize spend now (Plans 1 and 2) and hold budget for the real thing.
- **No, never** → Plan 4 is not optional. A public external proxy becomes the
  only non-circular evidence this project will ever produce.

### 3. Is the deliverable a feature set or a validated claim?

- *"Here are 29 documented columns, downstream decides what helps"* → finished
  today with Plans 1 and 2.
- *"E_macro certifies Source A adds value"* → certification against pillar
  features is circular by construction. Plan 4 is the minimum.

### 4. Is the downstream target a rate or a count?

Already identified in `docs/downstream_target_assumptions.md` as the property
that decides everything. It also flips Source A's internal verdict:

- **Rate** → county size is a control → `content_length` (r = 0.356 with size)
  is compromised, and the typed block's advantage grows, because
  `sec_n_industry_mentions` sits at r = 0.108.
- **Count** → size is a feature → the scalar is defensible and the typed block's
  case weakens.

This is the typed-vs-scalar comparison that is powered at 0.39 and cannot be
resolved statistically in this repo. A single answer here settles what 91
targets could not.

### 5. Who consumes +0.0010, and does the target basket match their targets?

Retention swings from 0.2% (Source E) to 69% (Source C) depending on the target.
If the downstream model cares about dynamics and velocity, Source A is worth
considerably more than the headline; if about industry composition, considerably
less. Any handoff of the single number must carry the basket composition with
it.

---

## Recommendation

Plan 2 first — it is cheap and it makes every existing claim defensible.
Then Plan 1, unless question 2 returns "no label, ever," in which case Plan 4
with broadband adoption as the external target.

Skip Plan 3. It buys nominal power against a dependence problem it does not
touch.

---

## Related

- `analysis-output/source-a/source-a-findings.md` §13–§17 — canonical claims,
  allowed and forbidden wording.
- `analysis-output/source-a/SESSION_2026-08-03_source_a_extraction.md` — how the
  extraction round was reasoned through, including what went wrong.
- `docs/downstream_target_assumptions.md` — the rate-versus-count question in
  question 4. Predates §13–§17 and needs the refresh described in Plan 2.
- `docs/PROJECT_GOAL.md` — open decision #1, size as control or feature.
- `outputs/source_a_representation.csv`, `outputs/source_a_marginal.csv` — the
  per-target data every figure above is computed from.
