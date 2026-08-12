# `E_macro` — five questions before the go/no-go

**For:** the FreeWheel Revenue Science conversation. **Date:** 2026-08-12.
**Ask:** ~20 minutes, plus a look at one row of the training table on a screen.
**Not asking for:** a data extract, a copy of anything, or access. Column names
answer four of the five questions; the fifth is a decision, not a lookup.

---

## Where the project is

Six public federal sources, ingested and validated at county grain (N ≈ 3,143),
every pillar with a frozen schema and documented null semantics. Nothing is
fused yet — deliberately, because fusion bakes in a join grain that question 1
below has not settled.

**What the evidence says.** Against five public county-level outcomes (ACS
broadband, income, age, home value, commute), scored out-of-fold on **states the
model never trained on**, `E_macro` beats a county-size baseline by **+0.190
mean R²**, positive on all five. Each pillar's own worth, measured as the R² a
model loses when that pillar is withheld:

| E capital | F typology | D freight | B industry | C velocity | A text |
|---|---|---|---|---|---|
| +0.058 | +0.041 | +0.019 | +0.007 | +0.005 | −0.000 |

Two pillars have moved on that evidence: Source A's embedding step was cut, and
Source F was kept only after a test it could fail. The honest caveat travels
with the headline — **these are public proxies, not your label**, so this is an
argument by analogy until it meets a real target.

---

## The questions, most decisive first

### 1. Does the impression row carry ZIP (or lat/long) at serving time?

**Why it decides everything else.** If the join is DMA-only, a 210-level
geographic dummy — which you can estimate precisely and for free from millions
of impressions per market — supplies everything a static DMA-keyed feature
could. Cross-sectionally, `E_macro` would add nothing over `C(dma)`, and no
correlation measured in this project is evidence against that.

If the row carries ZIP, county is derivable (HUD-USPS crosswalk, public), the
feature ships at 3,143 units instead of 210, and the fixed-effect objection
weakens because most counties are thin.

Three parts, and the second matters more than the first:

- Is there a ZIP / postal / lat-long column **next to** the DMA column?
- Is it **usually populated**, or often null? One row proves existence, not
  coverage.
- Is it on the **serving-time payload**, or only in an enriched reporting table
  downstream? Only serving-time geo supports a feature join.

### 2. Does the current model already include a DMA-level effect, dummy, or learned embedding?

If **yes** — that is the true baseline, and `E_macro` has to beat it rather than
beat nothing. If **no** — why not? If the answer is "not enough volume per
market to estimate it reliably," that is precisely the cold-start regime where
this feature layer wins, and it becomes the headline of the pitch rather than a
footnote.

### 3. What is one row, and is the target a rate or a count?

Recorded on my assertion, not yours: one row is an impression, ad request,
auction, household, or device, and the target is rate-shaped (revenue per
request, eCPM, ARPU, margin %). **Everything downstream of that assumption
changes if it is wrong** — county size flips from control to feature, and ten
Source D columns plus three Source E columns move back into the feature set.
Written confirmation closes it; a "yes, that's right" out loud is enough to
proceed.

### 4. Would you join at a finer grain than you target?

This is a decision, not a fact, and it is yours. Targeting and reporting at DMA
does not require the feature join to be at DMA. Measured cost of aggregating up
instead: the row-count loss (3,143 → 210) costs −0.122 mean lift and the
aggregation itself *gains* +0.099 — they roughly cancel, and on three of five
targets a 208-market arm matches or beats full county grain. **County grain is
this project's recommendation, not an established win.** The exception is
`broadband_rate`, the target closest to your domain, where the aggregate arm
goes negative.

### 5. Can I have your DMA crosswalk?

Nielsen definitions are proprietary; public county→DMA mappings circulate with
murky licensing. Yours is the definition that governs the join, and a boundary
mismatch would silently corrupt every aggregate. Please also confirm the DMA
count — 210 is standard, vendor mappings differ at the margin, and some leave
counties unassigned.

---

## While looking at the table — glance for these

| Look for | What it settles |
|---|---|
| Fine-geo column next to the DMA column | Q1. DMA becomes a *choice* rather than a constraint |
| What one row is (impression / bid / auction / pre-bid) | Q3, independently of the answer given aloud |
| Target-shaped columns (`price`, `cpm`, `revenue`, `won`, `filled`) | What is actually predictable |
| A geo-source or precision column (`geo_type`, `location_precision`) | Whether lat/long is genuine or an IP-derived centroid — otherwise unknowable |
| Timestamp / date-partition columns | Whether models train on rolling windows, which decides if pillar vintage spread is a leakage defect |
| Any volume or count column | Whether impressions-per-geo-unit is derivable, so thin-unit tests use your distribution rather than population as a proxy |

---

## What each answer changes on my side

| Answer | Next two weeks |
|---|---|
| ZIP present at serving time | Build the ZIP→county join spec, keep county grain, benchmark against a DMA fixed effect |
| DMA only | Build the aggregation layer, re-measure every published figure at 210 units, and say plainly that the cross-sectional case is weak |
| Model already has a DMA effect | The benchmark becomes the deliverable — beat it or recommend no-go |
| Target is count-shaped, not rate | Restate the size tiering; thirteen columns move back into the feature set |
| No appetite for a finer join | Same as DMA-only, and the recommendation is likely no-go on cross-sectional grounds alone |

---

## What is not blocked

Ingestion, validation, schema freeze and the external benchmark are done and do
not depend on any answer above. If none of these questions can be answered, the
project can still hand over six documented pillars and an honest statement of
what they are worth against public targets — it just cannot say whether they are
worth anything to *you*, which is what a go/no-go needs.

Detail behind every number here:
`analysis-output/cross-source/external-target-findings.md`,
`analysis-output/cross-source/pillar-marginal-findings.md`, and
`docs/plans/dma_regrain.md` §0 for the long form of these questions.
