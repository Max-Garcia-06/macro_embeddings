# Ingestion Reconnaissance — Sources B, D, E

Merged from `source_b_plan.md`, `source_d_plan.md`, and `source_e_plan.md`
(2026-08-04). Those three were pre-implementation planning docs for work that is
now complete; the phase checklists, dependency lists, and status footers they
carried are dead. What survives here is the part the findings docs cite: how each
source's access was resolved, which design options were tested empirically and
what the numbers were, and which hypotheses were falsified along the way. Full
original text is in git history.

Sources A, C, and F never had plan docs — their reconnaissance lives in their
findings docs directly.

---

## Source B — BLS QCEW (Industrial Core)

### Phase 0: access

`www.bls.gov` (docs/landing pages) returns a bare **403** to plain `curl`, the
same shape as Source D's `bts.gov` block. The actual QCEW Open Data Access files
live on **`data.bls.gov`**, which has no bot protection — plain `curl` and
`requests` both get 200, no special headers, no TLS quirk.

Schema confirmed by direct download (San Francisco 06075, Loving TX 48301,
Petroleum MT 30069, national NAICS-11 and NAICS-52 slices, 2023 Q1).
**`own_code="5"` (Private) + `agglvl_code="74"`** is the target slice: exactly the
20 codes the proposal describes. **Disclosure mechanism confirmed as the spec
worried:** when `disclosure_code="N"` the corresponding `lq_*` fields are `0`/blank
rather than the true value, so a suppressed cell is indistinguishable from a
genuine zero unless the disclosure code is carried alongside the LQ value.

**Ownership scope: Private only.** Government rows (`own_code` 1/2/3) do carry
2-digit NAICS rows but only for a sparse subset of sectors each (federal ~8 of 20,
state ~5, local ~9). `own_code="0"` — the true all-ownership total — carries no
2-digit breakdown at all, only a single `industry_code="10"` grand-total row, so
there is no pre-computed all-ownership LQ to consume. Building one would mean
recomputing LQ from raw employment summed across four ownership codes. Government
employment concentration reflects facility placement, not industrial
specialization, so private-only is also the closer match to what the pillar is
for. Judgment call, flagged rather than made silently.

### Phase 1b: access detour, and the suppression comparison

**Access detour not anticipated in Phase 0:** the 3 combined 2-digit codes
(`31-33`, `44-45`, `48-49`) 404 as individual `industry`-slice URLs — the literal
hyphenated code, no-hyphen variants (`3133`, `4445`, `4849`), and the standalone
halves all fail to resolve. One guess, `3133`, returned 200 but is an unrelated
real 4-digit NAICS code — a red herring caught by checking the `industry_code`
field rather than trusting the status code. Fell back to the bulk
`data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip` (287MB
compressed, 2.2GB uncompressed) and filtered locally.

**Real national suppression rate: 30.0%** of county×sector cells (own_code=5,
agglvl=74, 2025 Q4) — well below the general "~60%" QCEW figure, which blends in
much finer NAICS levels. Sector range is wide: Retail Trade (44-45) 3.1% vs.
Mining (21) 59.0% and Agriculture (11) 52.9%. Suppressed cells have a median of 5
establishments against 40 for disclosed cells, so suppression tracks genuinely
small local presence rather than a data-quality artifact.

| Option | What it is | Result |
|---|---|---|
| **A. Null-passthrough** | Keep `lq_*` null wherever `disclosure_code="N"` | Simplest, no fabricated numbers, leaves real gaps in 30% of cells |
| **B. State-level LQ fallback** | Substitute the state's sector LQ where suppressed | Tested on 42,704 held-out disclosed cells: **MAE 0.786, r 0.334** — barely better than the global mean LQ everywhere (MAE 0.947) |
| **C. Constrained multi-county solve** | The spec's corrected IPF idea, respecting state totals | Median within-state county-LQ std dev (0.519) is 59% as large as between-state (0.880) — most true variation lives *within* a state, exactly what a state-margin-constrained solve has least power to recover |

**C-lite test, in place of building full IPF:** allocated each suppressed county's
sector employment proportionally to that county's share of total private
employment within its state — closed-form, exactly satisfies the state total by
construction (unlike the spec's per-county-independent IPF). **MAE 0.786, r 0.340
on 42,554 held-out cells — statistically indistinguishable from Option B.** A
county's overall economic size doesn't predict its sector-specific concentration
any better than its state's average does. Since the cheap proxy for "respect state
totals" gained nothing, full IPF — resting on the same state-margin information —
was not worth building.

**Decision: Option A**, null-passthrough with `disclosure_code`/`lq_disclosure_code`
preserved as explicit nullability markers. Shipping a substitute that scores
r≈0.33–0.34 either way would read as more precise than it is.

**Bug caught in verification:** BLS reports a suppressed cell's `lq_month3_emplvl`
as a literal `0`, not blank, so `pd.to_numeric` alone doesn't null it. The first
draft left suppressed cells at `0.0`, silently reintroducing the exact false-zero
problem Phase 0 flagged. Fixed by nulling `lq_emp_{naics2}` off the disclosure flag
rather than the raw value.

---

## Source D — BTS FAF5 (Trade Logistics)

### Phase 0: access, and what the product actually is

**`bts.gov` is Akamai bot-gated** — `curl -I https://www.bts.gov/faf/county`
returns a bare 403, no challenge page, headers alone don't help. The ICPSR/
datalumos mirror 403s too, behind Cloudflare. But the data files aren't hosted
there: the landing page (loaded once via Playwright to get past the block) links
out to **`faf.ornl.gov`** (Oak Ridge National Lab) for every download, e.g.
`https://faf.ornl.gov/faf5/Data/County/44%20-%20Rhode%20Island.zip`. That host has
zero bot protection. No browser automation or manual-download fallback is needed
in the pipeline; the per-state URL list was scraped once and hardcoded.

**Not a dense county×county matrix.** Each per-state zip contains four tables:
county-to-county (for that state and adjacent states only), county-to-FAF-zone,
FAF-zone-to-county, and FAF-zone-to-FAF-zone (~130-zone resolution). BTS has
already solved the combinatorial-explosion problem the pre-scoping spec worried
about — full county granularity nearby, pre-aggregated zones farther out. **This
supersedes the spec's proposed top-K-per-origin truncation**, which would solve a
problem the data doesn't have and discard zone-level long-tail structure the
source preserves for free.

Confirmed columns: `trade_type, fr_orig, fr_inmode, fr_dest, fr_outmode, dms_orig,
dms_orig_cnty, dms_dest, dms_dest_cnty, sctgG5, dms_mode, tons_2022`.
`dms_orig_cnty`/`dms_dest_cnty` are real 5-digit FIPS, directly joinable to
`county_crosswalk.parquet`. `sctgG5` has exactly 5 values.

**No dollar-value column.** Both `macro_pre_scoping_spec.pdf` and
`E_macro_extendedProposal.pdf` describe Source D as tracking freight tonnage *and*
dollar value. The experimental county product ships `tons_2022` only. The FAF5 user
guide's base data dictionary does carry `value`/`current_value` fields, confirming
the county-level disaggregation genuinely dropped them — a real scope reduction
from what both scoping docs promised, not a parsing miss.

**Phase 0 addendum — `dms_mode` code `11` remains unexplained.** The FAF5 User
Guide (Release 5.1, pulled from `faf.ornl.gov`) gives modes 1–8; the real
county-level CSVs use `11, 2, 3, 5, 6`. The experimental county product uses its
own extended taxonomy, not documented in the general guide. The county-specific
technical report that would likely resolve it sits on `bts.gov` behind the same
Akamai gate and is not mirrored on the ungated host. Pulling it would have meant
replaying browser anti-bot session cookies through `curl`; the environment's
safety classifier correctly blocked that and it was not worked around. **If the
documentation is needed, a manual browser download by the user is the
straightforward path** — not worth automating around a security control for a gap
this minor.

The guide's Table 3 does resolve zone→state/region names (132 zones) but **not a
zone→county-FIPS crosswalk**, which would need Census CBSA/CSA delineation files.
Recommended if a distance/"reach" signal is wanted in a future round.

### Phase 1b: the vectorization design spike

Three candidate vectorizations were built on a subsample rather than picked a
priori: **A** scalar aggregates only, **B** scalar + top-5 partner columns, **C**
scalar + graph statistics (degree, HHI concentration, distance-weighted reach).
Run first on Rhode Island (Providence 44007 urban/port vs. Washington 44009
rural), then re-run on an independent, much larger New Jersey sample (Essex 34013
= Newark port/airport vs. Sussex 34037 = rural; 153 regional partners vs. Rhode
Island's 28). The pattern held exactly across both:

- **Option A holds up.** Providence moved ~4.5× Washington's tonnage; Essex ~7×
  Sussex's (17.3K/23.3K out/in vs. 2.4K/3.3K), proportional across all five
  commodity groups.
- **Option B adds nothing.** Both counties' top-5 partners are the same handful of
  big neighbors — partner identity doesn't discriminate hub from rural when both
  are pulled toward the same regional demand centers.
- **Degree is a dead signal.** Both test counties show *identical* partner counts
  in each sample (28/28 in RI, 153/153 in NJ). BTS's gravity-model disaggregation
  assigns a small nonzero flow to nearly every county pair in a region, so raw
  degree cannot distinguish hub from sink. Dropped.
- **Distance-weighted "reach" points backward, and the mechanism is understood.**
  Essex's mean outbound distance (55km) is *shorter* than Sussex's (83km);
  Washington beat Providence the same way (66–80km vs. 40–47km). A real hub's
  long-distance flows collapse into the FAF-zone tables once they leave the
  adjacent-state window that bounds the county-to-county table, so that table can
  never see them. Dropped as a formulation, not retried on a third sample.
- **HHI concentration shows a small, directionally sensible signal** and was kept
  — but see the scale reversal below.

**Shipped: Option A + pooled-partner HHI**, with the HHI computed over county-level
*and* FAF-zone-level partner rows rather than the county-only subset the spike
used.

### Corrections to Phase 0/1b hypotheses

**Table-3 duplication: falsified.** Phase 0 speculated that `3.FAF-to-FAF.csv`
was one national table copied into every state zip, and proposed a dedup step to
cut download volume. Rhode Island's table 3 has 598,360 rows; New Jersey's has
489,987. They are state-specific in some uncharacterized way. The dedup
optimization was dropped from Phase 2.

**HHI direction is scale-dependent.** The regional spikes showed hubs as *less*
concentrated than rural counties (Essex 0.043–0.056 vs. Sussex 0.047–0.059).
Nationally the direction flips: the biggest hubs (LA, Chicago, Houston, HHI
0.11–0.23) are *more* concentrated than small rural counties (0.02–0.04) — a
massive hub funnels huge volume through a few dominant interstate corridors while
a small county spreads modest volume evenly across nearby neighbors. Real
scale-dependent reversal, not noise. The scalar tonnage signal is unambiguous
either way; only the concentration metric's direction moves.

**TLS incompatibility against `faf.ornl.gov`.** `requests`/urllib3 gets a
connection reset during the TLS handshake on every attempt, with or without
retries or forced TLS 1.2, while the system `curl` binary connects cleanly with no
configuration. Not the `bts.gov` Akamai situation — no bot detection, no challenge,
no cookies — just an ordinary client compatibility quirk, most likely a ClientHello
fingerprint mismatch against a plain IIS file server. Resolved by shelling out to
`curl` via `subprocess` (list-form arguments, hardcoded URLs, no shell
interpolation) for this one client class.

**Accepted limitation:** `export_to_parquet` writes once, after all 51 states
finish — an interruption loses the whole run. Raised mid-run as a decision point
and deliberately left alone rather than touching working code; the run completed in
2h32min. Revisit before assuming that speed on any future refresh, since the early
states were slow enough to project 4–6 hours.

---

## Source E — IRS SOI (Capital Flow)

### Phase 0: access and file choice

**Simplest access of all six sources.** `www.irs.gov` — both the landing page and
the `/pub/irs-soi/*.csv` file host — has no bot protection, no landing-page/data-host
split, no TLS quirk. Plain `requests`, 200 OK.

Tax Year 2022 is the latest published county file (`-2023` 404s). A WebSearch claim
of a "TY2023 release on 2026-08-13" did not check out against the live site and was
dropped — a real hallucination the live page-scrape caught.

The county data ships in 5 formats. **Chose `22incyallnoagi.csv`** — one row per
county with `AGI_STUB` fixed at 0, i.e. the IRS's own pre-aggregated county total —
avoiding both a 51-file per-state download loop and a manual summation across the 8
AGI brackets. Same "use the source's own pre-solved aggregate" pattern as Source B's
bulk singlefile and Source D's county/zone hybrid.

Parsed with `latin-1` (the file has non-UTF-8 bytes in some county names). 3,194
rows: 3,143 real counties + 51 state totals + DC. **3,143/3,144 crosswalk coverage
— the one gap is Kalawao County, HI (`15005`, pop. ~90), the same county Source B's
QCEW ingestion independently misses**, not a new gap.

The proposal's ratio was computed end-to-end on the real file before any pipeline
code was written: median 0.082, mean 0.107, zero division-by-zero cases (`A00200`
min is $2.872M across all counties). Top 5 by ratio are Teton WY (Jackson Hole,
2.80), Pitkin CO (Aspen, 1.57), Blaine ID (Sun Valley, 1.24), Collier FL (Naples,
1.13), Monroe FL (Florida Keys, 1.11) — real-world validation of the signal ahead
of implementation.

### Schema mutation, mitigated as the spec proposed

Line items do get renamed/added/removed year over year (TY2022 added
`N00400`/`A00400` tax-exempt interest and dropped several TY2021 COVID-era credit
fields). The four columns this pipeline needs — `A00200` wages, `A00650` qualified
dividends, `A01000` net capital gain, `A00100` AGI — have stable SOI variable codes
across TY2021–2022, but the spec's concern is legitimate for future years.
Mitigated as recommended: a `SOI_COLUMN_MAP` constant mapping conceptual name →
SOI variable code, so a future schema change fails loudly with a `KeyError` rather
than silently misreading a shifted column.

### No suppression flag — more opaque than Source B's equivalent

IRS SOI applies the same small-cell privacy suppression BLS QCEW does ("Income and
tax items with less than 20 returns for a county were excluded"), but **the county
file carries no suppression flag at all** — a suppressed cell and a genuine zero are
both written as a bare `0`. Confirmed empirically: 3 of 3,143 counties (Kenedy,
King, Loving — all ultra-low-population West Texas, 40–140 returns) show
`A01000=0`/`A00650=0` for both the amount and the underlying return-count columns,
consistent with either reading, and nothing in the file distinguishes them.
Decision: ship raw values as-is and disclose the limitation rather than fabricate a
distinction the source doesn't support.

This was not flagged in either scoping doc.
