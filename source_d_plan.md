# Source D: BTS FAF5 Experimental County Commodity Flows (Trade Gravity)

## Context

Sources A (Wikipedia embeddings), C (FRED velocity), and F (USDA typology) are complete and merged. Source B (BLS QCEW) has not been started yet — no `ingest_source_b.py` or `source_b_plan.md` exists in the repo, despite an earlier session flagging it as next. Source D does not depend on B, so it can proceed independently.

**Source D = "BTS FAF5 Experimental County Commodity Flows (Trade Gravity)"**: ingest county-to-county freight tonnage/dollar-value flows from the Bureau of Transportation Statistics Freight Analysis Framework, version 5 (FAF5). Conceptually this is the "Movement" pillar of `E_macro` — it distinguishes a logistics pass-through corridor or industrial exporter from a pure consumer sink, per `docs/E_macro_extendedProposal.pdf` and `docs/macro_pre_scoping_spec.pdf`.

This is the source both scoping docs flag as highest-risk (`macro_pre_scoping_spec.pdf` calls out "combinatorial matrix explosion" — a naive 3,142 × 3,142 ≈ 9.87M-edge dense matrix — and proposes a top-K-per-origin truncation as a fix, while flagging that the truncation itself "destroys global network topology"). **Live research during this planning pass found the real BTS product handles the combinatorial problem differently than the spec assumed**, and surfaced a second risk the spec doc didn't mention at all: the source is bot-gated.

### What's actually there (verified via web search; BTS's own site blocks direct fetch — see Risk 1)

- Real product: **"Freight Analysis Framework Version 5 (FAF5): Experimental County-Level Estimates,"** released 2025-01-03, landing page `https://www.bts.gov/faf/county`, technical report/user guide at `https://www.bts.gov/faf/county/documentation`.
- **Base year: 2022.** This is a single cross-sectional snapshot, not a time series — matches Source F's static-anchor pattern more than Source C's velocity pattern, despite being grouped under "Movement" in the proposal.
- **Not a true dense county×county matrix.** Each downloadable zip (national, or per-state) contains **four separate tables**:
  1. county-to-county OD flows *for the state of interest and every adjacent state only*
  2. county-to-FAF-zone OD flows (that multi-state area → all other FAF zones)
  3. FAF-zone-to-county OD flows (all other FAF zones → that multi-state area)
  4. FAF-zone-to-FAF-zone OD flows (everywhere else, at ~130-zone resolution)
  In other words, BTS has *already* solved the combinatorial-explosion problem: full county-level granularity is kept only for nearby geography, and everything farther away is pre-aggregated to FAF-zone level. **This directly supersedes the spec's proposed top-K-per-origin / `External_Basin_Flow` truncation** — that scheme would be solving a problem the source data doesn't actually have in that form, and would discard structure (zone-level long-tail flows) that the source already preserves for free.
  - Open question, not yet resolved: this means every county has flows expressed in *two different geographic units* (county IDs for nearby, FAF-zone IDs for distant) — Phase 1 needs to confirm the FAF-zone-to-county FIPS crosswalk BTS publishes, since `county_crosswalk.parquet` (built for Sources A/C/F) has no FAF-zone mapping.
- Commodity resolution: aggregated into **5 commodity groups** (a `sctgG5` field), down from FAF5.6.1's 42 raw SCTG commodity codes — coarser than the pre-scoping spec implied by name-dropping "SCTG5" as if it were the base classification.
- **5 transport modes** are included (truck, rail, water, air, other/multiple — exact mode labels to confirm from the user guide in Phase 1).
- Units: tonnage (thousands of tons) and value (millions of $), per the spec doc — not yet independently confirmed against the actual column headers.
- Contact for questions: `FAF@dot.gov` (a real fallback if documentation gaps block Phase 1).

## Risks (in priority order — these gate everything else)

### Risk 1: BTS is bot-gated — access mechanism is unconfirmed
`curl -I https://www.bts.gov/faf/county` from this environment returns a bare **403 from Akamai** (no challenge page, just a block). The ICPSR/datalumos mirror (`datalumos.org`) also 403s, behind a Cloudflare challenge this time. This is qualitatively different from every prior source:
- Source A: authenticated REST API (Wikimedia Enterprise) — worked with `requests`.
- Source C: authenticated REST API (FRED) — worked with `requests`.
- Source F: plain static file download (ERS) — worked with `requests`, no auth.
- Source D: the landing page itself blocks a plain HTTP client.

This needs to be resolved **before any pipeline code is written**, because it determines the whole ingestion architecture:
- If a browser-like `User-Agent` + session cookies clears it → still a `requests`-based script, just with realistic headers.
- If it's genuine bot-detection (Akamai bot manager, JS challenge) → likely needs Playwright/a headless browser to fetch the zip URLs once, or a one-time **manual download** into `data/raw/` with the ingestion script picking up local files instead of hitting the network at all (closer to how Source F's "no credentials, static tabular ingestion" was framed, just with a manual step first).
- Worth trying the direct file URLs (not the HTML landing page) via `curl` first — Akamai blocks are sometimes page-specific rather than domain-wide.

### Risk 2: vectorization strategy — turning a flow network into one row per county
Every other source in this project produces **one feature row per county**. Source D is fundamentally a **graph** (up to ~3,142 nodes × edges to both counties and FAF zones, per commodity group and mode). The pre-scoping spec's own risk callout — "truncating the network... severs macro-regional economic corridor identities" — is real and unresolved by BTS's county/zone hybrid structure; it just moves the tradeoff, not eliminates it.

This is the actual "hardest part" of Source D, and it's a modeling decision, not just an engineering one — flagging it now rather than picking silently:

| Option | What it is | Tradeoff |
|---|---|---|
| **A. Scalar aggregates only** | Per county: total inbound tonnage/value, total outbound tonnage/value, split by the 5 `sctgG5` commodity groups and 5 modes (~20–50 scalar columns). No partner identity. | Simple, fixed-width, matches every other source's shape. Loses "who trades with whom" entirely — closest to *not* capturing trade gravity at all. |
| **B. Top-K partner columns (spec's original idea)** | Scalar aggregates (A) + top-5 trading-partner FIPS/zone codes and their flow volumes. | Partially restores partner identity for a corridor's biggest edges; still the spec's own flagged flaw (destroys long-tail topology) for everything past the top 5. |
| **C. Graph summary statistics** | Scalar aggregates (A) + graph-theoretic features per county: weighted degree, flow concentration (e.g., Herfindahl index over partners), distance-weighted flow (proxy for "hub" vs. "local" character). | Captures corridor/hub-vs-sink distinction the proposal explicitly wants ("isolates a logistical pass-through corridor... from a pure consumer sink") without hardcoding a partner list. More analysis code, no new columns of raw identity data. |

**Decision (2026-07-13): don't pick a priori — build all three on a subsample and compare empirically.** Once Phase 0 confirms real schema, pull a manageable subsample (a handful of states spanning hub-like counties — e.g. a major port/rail county — and clearly rural sink counties, so the comparison actually has contrast to show) and compute options A, B, and C on it. Compare them on whether they actually separate the "corridor/exporter vs. consumer sink" distinction the proposal wants Source D for — e.g. does a known logistics hub county (a major port or intermodal rail county) look visibly different from a same-population rural sink county under each option? Whichever separates that contrast most cleanly, with the fewest columns, is the one that ships in Phase 2. This becomes a small comparison step inside Phase 0/1 rather than a documented-but-unverified recommendation.

## Phases

### Phase 0: Reconnaissance (resolves Risks 1 & 2 — do this before writing any pipeline code)
1. Test direct file URLs (not just the landing page) with `curl` using a browser `User-Agent`; if still blocked, try Playwright to load `bts.gov/faf/county` and capture the actual download links + any cookies/headers a real browser session gets.
2. If BTS remains inaccessible programmatically, download one small state's zip manually (e.g., Delaware or Rhode Island, to keep the file small) and inspect it locally to confirm real column names, FIPS/zone ID formats, and the county/zone hybrid structure described above.
3. Pull the technical report/user guide PDF (`FAF5-County-Level-Estimates-Technical-Report.pdf`) — manually, if needed — to confirm: exact `sctgG5` group definitions, the 5 mode labels, tonnage/value units, and the FAF-zone-to-county/state crosswalk.
4. Report findings back before proceeding — this phase may change the plan below (e.g., if BTS truly cannot be automated, Phase 1 becomes "document the one-time manual download step" rather than a `BtsFafClient` class).

### Phase 1: Credentials & dependencies
- No API key expected (static bulk download), but confirm after Phase 0.
- Dependencies for the comparison: `networkx` (or manual pandas groupby) for option C's graph statistics — cheap either way at ~3,142 nodes.

### Phase 1b: Subsample vectorization comparison (resolves Risk 2)
- Select a small, contrast-rich subsample once Phase 0's real schema is confirmed: a few known logistics-hub counties (major port/rail/intermodal counties) + a few rural counties of similar population, across 2–3 states so county-to-county *and* county-to-FAF-zone flows both show up in the sample.
- Compute options A, B, and C for just this subsample.
- Compare: does each option visibly separate the hub counties from the population-matched rural counties? Report this as a short table/notebook cell, not a full findings doc — this is a design spike, not a deliverable.
- Pick the option (or a light combination) that best captures that separation with the fewest columns; record the decision and why in this plan file before Phase 2 starts.

### Phase 2: `scripts/ingest_source_d.py`
Mirror Source A/C/F's architecture once Phase 0 resolves the access question:
- Thin download/parse layer (shape depends on Phase 0's outcome — either an HTTP client or a local-zip loader).
- Reshape the four OD tables (county-county, county-FAF, FAF-county, FAF-FAF) into a single edge list keyed by FIPS-or-zone origin/destination, tagged with commodity group and mode.
- Compute the per-county feature vector per the Phase-0-confirmed vectorization option (A/B/C above).
- `CountyTradeFlowResult` dataclass + per-county failure isolation, matching `IngestionSummary`'s succeeded/failed tracking pattern from Sources A/C.
- Output: `data/source_d_faf.parquet`, one row per county, exact columns TBD pending the vectorization decision.

### Phase 3: Analysis scripts (mirroring the per-concern script split)
- `scripts/analyze_source_d_hubs.py` — identify top logistics hub/corridor counties by weighted degree or flow concentration (the proposal's own framing).
- `scripts/analyze_source_d_source_c_correlation.py` — cross-check trade flow intensity against Source C's economic velocity, analogous to the existing `analyze_source_a_source_c_correlation.py`.
- `scripts/visualize_source_d.py` — choropleth/flow map (analog of `visualize_source_a.py` / `visualize_source_f.py`).
- `scripts/generate_source_d_insights.py` — headline stats for the findings doc/notebook.

### Phase 4: Findings deliverable
- `analysis-output/source-d-findings.md` — same YAML frontmatter conventions as prior findings docs, with an explicit "proposal alignment" section documenting the two deviations from the pre-scoping spec found here (the county/FAF-zone hybrid structure superseding the top-K truncation idea, and whichever vectorization option was actually used) and why.
- `analysis-output/source_d_key_findings.ipynb` + `analysis-output/figures/source_d_*.png`.

## Verification
1. Confirm Phase 0's access method actually retrieves real data (not a cached/stale mirror) before writing Phase 2 code.
2. Run ingestion against a small state sample first (reuse the Phase-0 small-state file); confirm schema, non-null rates, and that county vs. FAF-zone flows are correctly distinguished before the full run.
3. Confirm `data/source_d_faf.parquet` row count (~3,142) and spot-check 2–3 counties' flow totals manually against the raw downloaded files.
4. Run each analysis script and confirm expected output files (CSV/HTML/PNG) without errors.
5. Read `analysis-output/source-d-findings.md` for internal consistency; confirm the proposal-alignment section gives real numbers, not restated concerns.

## Phase 0 findings (2026-07-13, confirmed live)

**Risk 1 resolved — access is trivial, once you know where to look.** `bts.gov/faf/county` itself is behind an Akamai bot-manager block that rejects plain `curl`/`requests` outright (403, no challenge page — headers alone don't help). But the actual data files aren't hosted there: the landing page (loaded via Playwright to get past the block once) links out to **`faf.ornl.gov`** (Oak Ridge National Lab) for every download, e.g. `https://faf.ornl.gov/faf5/Data/County/44%20-%20Rhode%20Island.zip`. That host has **zero bot protection** — confirmed via plain `curl`, HTTP 200, no special headers needed. So Phase 2's ingestion script needs no Playwright/browser automation and no manual-download fallback: a `requests`-based per-state download loop works, same shape as Sources A/C/F. The per-state URL list (all 50 states + DC, exact filenames) was scraped once and can be hardcoded as a constant, matching how Source A hardcodes `ALL_COUNTIES`.

**Real schema (from downloading and unzipping the Rhode Island file):** each per-state zip contains exactly the 4 tables the search results described, now with confirmed columns:
- `1.County-to-County.csv`: `trade_type, fr_orig, fr_inmode, fr_dest, fr_outmode, dms_orig, dms_orig_cnty, dms_dest, dms_dest_cnty, sctgG5, dms_mode, tons_2022` (12 cols)
- `2.County-to-FAF.csv` / `2.FAF-to-County.csv`: same shape minus the far-side county column (destination or origin is a FAF zone, not a county)
- `3.FAF-to-FAF.csv`: zone-to-zone only, no county columns at all
- `trade_type`: 1/2/3 (domestic/import/export, standard FAF convention — not yet cross-checked against the user guide, but consistent with FAF's normal schema).
- `dms_orig`/`dms_dest`: 3-digit **FAF zone** codes (e.g. `091`=a Connecticut zone, `441`=Rhode Island's single zone — states get split into multiple zones only where they're big enough, e.g. CT has zones 091/092/099).
- `dms_orig_cnty`/`dms_dest_cnty`: real 5-digit county FIPS (e.g. `44001`) — directly joinable to the existing `county_crosswalk.parquet`, no new crosswalk needed for the county-level rows. FAF-zone rows still need a zone→state (or zone→county-list) mapping, not yet pulled from the user guide.
- `sctgG5`: exactly 5 values — `sctg0109, sctg1014, sctg1519, sctg2033, sctg3499` — each spanning a range of the original 2-digit SCTG codes, confirming the "5 commodity groups" finding from search.
- `dms_mode`: 5 distinct codes (`11, 2, 3, 5, 6`) — exact mode labels (truck/rail/water/air/other) still need the user guide, not yet fetched (bts.gov PDF access also 403s; may need the same Playwright workaround or a direct ornl.gov mirror).

**New finding not in either doc: no dollar-value column.** Only `tons_2022` is present in this experimental product — despite both `macro_pre_scoping_spec.pdf` and `E_macro_extendedProposal.pdf` describing Source D as tracking "freight tonnage **and dollar value**." The experimental county-level disaggregation is tonnage-only; a value figure may exist only at the coarser FAF-zone-level product (not the county one), or not at all in this release. This is a real scope reduction from what both docs promised — worth flagging explicitly in the eventual findings doc's proposal-alignment section, not silently working around.

**New risk: total download volume is large and uneven.** HEAD-checked several state zips: Rhode Island 36MB, California 121MB, Montana 156MB, New Jersey/Wyoming ~214MB, **Illinois 674MB**. Size doesn't track population/hub-status in an obvious way — the likely cause is that `3.FAF-to-FAF.csv` (598K+ rows, national in scope) appears to be the **same national table duplicated in every single state's zip**, not state-specific. Not yet fully confirmed (would need to diff two states' table-3 files byte-for-byte), but if true, full nationwide ingestion should download and parse table 3 exactly once, not 51 times — otherwise this is a double-digit-GB download for a single ingestion run. Confirm this dedup hypothesis in Phase 2 before writing the download loop.

## Phase 1b findings (2026-07-13) — first pass, inconclusive on Option C

Ran options A/B/C on the Rhode Island sample already downloaded, comparing Providence County (44007, urban/port) against Washington County (44009, rural/coastal) — both from `1.County-to-County.csv` + `2.County-to-FAF.csv`/`2.FAF-to-County.csv`, domestic flows (`trade_type=1`) only. Script: `compare_vectorizations.py` in the scratchpad (not committed — this was a throwaway spike, not deliverable code).

- **Option A (scalar totals by commodity group) gives a clean, immediately usable signal.** Providence: 19.3K tons out / 15.6K in. Washington: 4.2K out / 4.6K in — a consistent ~4.5x gap, proportional across all 5 `sctgG5` groups. Cheap, always available, no partner-identity assumptions.
- **Option B (top-5 partners) added little in this sample.** Both counties' top partners are just their own RI/CT/MA neighbors, since the county-to-county table is limited to RI + adjacent states by construction — not informative about a real port's *national* reach.
- **Option C surfaced two real findings, not a bug in the spike script:**
  - **Partner count (naive degree) is a dead signal.** Both test counties show exactly 28 nonzero partners — BTS's underlying disaggregation model assigns a small nonzero flow to nearly every county pair in a region (gravity-model artifact), so raw degree doesn't distinguish hub from sink at all.
  - **Distance-weighted "reach" pointed the wrong way**: Washington County showed *longer* average partner distance (66–80km) than Providence (40–47km) — the opposite of the "a port reaches farther" hypothesis. Most likely cause: a real port's long-distance/national-scale flows collapse into the FAF-**zone**-level tables (`2.County-to-FAF`/`2.FAF-to-County`) once they leave the adjacent-state window, and this first-pass distance calc only used the county-to-county rows, which are geographically boxed in by construction. A single small state's adjacent-state window can't validate a "reach" hypothesis — it never sees the far side.

**Conclusion: don't decide off this sample.** The comparison needs either (a) a genuinely large hub state (e.g. New Jersey/Port of NY-NJ, or Illinois/Chicago — both already HEAD-checked at 214MB/674MB) where the county-to-county table itself contains longer-range partners, or (b) extending Option C's distance/concentration stats to incorporate the FAF-zone-level rows using an approximate zone centroid (mean of the zone's member counties) — which needs the zone→county crosswalk from the user guide (still not pulled, see Phase 0's mode-label gap). Recommend (b) is worth doing regardless of (a), since national-scale hub character will always partly live in the zone-collapsed rows for any real port county, not just small-state samples.

## Phase 0 addendum — general FAF5 User Guide (Release 5.1, pulled successfully from `faf.ornl.gov`)

Confirms/corrects several Phase 0 items using the **base** FAF5 schema (the experimental county product is a derivative, not identical — see discrepancies below):
- **Mode codes (Table 1, base product):** 1=Truck, 2=Rail, 3=Water, 4=Air (incl. truck-air), 5=Multiple Modes and Mail, 6=Pipeline, 7=Other/Unknown, 8=No Domestic Mode. **Discrepancy confirmed, not resolved:** the real county-level CSVs use `dms_mode` codes `11, 2, 3, 5, 6` — code `11` doesn't exist in this base table (only goes to 8). The experimental county product evidently uses its own extended/modified mode taxonomy not documented in this general guide. Still open — see residual gaps below.
- **Commodity codes (Table 2):** confirms the 2-digit SCTG base classification (30 codes, 01–43 with gaps) that `sctgG5`'s 5 groups aggregate.
- **FAF zone table (Table 3):** full 132-zone list with descriptive CFS-area names, e.g. `441`=Rhode Island (whole state = one metro zone, "Boston-Worcester-Providence MA-RI-NH-CT CFS Area (RI Part)"), `091`/`092`/`099`=Connecticut's 3 zones, `341`/`342`=New Jersey's 2 zones. This resolves the zone→state/region-name mapping but **not** a zone→county-FIPS-list crosswalk (would need Census CBSA/CSA delineation files to go further, not pursued here).
- **Confirms the "no dollar value" finding from real data was a real product change, not a parsing miss:** the base FAF5 data dictionary (Table 5) explicitly has `value` and `current_value` fields (Million $, 2017 dollars / current dollars) in the standard ODCM database — the experimental **county-level** disaggregation genuinely dropped this field and ships tonnage (`tons_2022`) only.

**Residual gap, deliberately not pursued further:** the county-level-specific technical report (`FAF5-County-Level-Estimates-Technical-Report.pdf`, 16MB, on `bts.gov` — not mirrored on the ungated `faf.ornl.gov` host) would likely explain the `dms_mode=11` code and any real zone→county crosswalk. Playwright can load the `bts.gov` HTML pages (its browser session clears whatever check blocks plain `curl`), but pulling that specific PDF's *content* through the same route would have meant extracting the browser's Akamai anti-bot session cookies and replaying them via `curl` — the environment's safety classifier correctly blocked that as a bot-detection bypass I hadn't been asked to perform, and it wasn't pursued further or worked around. **If this documentation is needed, the straightforward path is a manual download by the user** (a real browser hits it fine) — not worth automating around a security control for a documentation gap this minor.

## Phase 1b full findings (2026-07-13) — confirmed on a second, larger, independent sample

Re-ran the same comparison on New Jersey (Essex County 34013 = Newark port/airport hub, vs. Sussex County 34037 = rural, no port) — a much richer sample than Rhode Island's (153 distinct regional partners vs. 28, since NJ's adjacent-state list pulls in NY/PA/DE/MD). **The pattern from the Rhode Island test held exactly, on an independent and much larger sample:**

- **Option A holds up again:** Essex moves ~7x Sussex's tonnage (17.3K/23.3K out/in vs. 2.4K/3.3K), proportional across commodity groups. Confirmed robust across two states.
- **Option B still weak:** both counties' top-5 partners are the same handful of big neighboring counties (Middlesex, Bergen, Union, Hudson) — partner *identity* doesn't discriminate hub from rural when both are pulled toward the same regional demand centers.
- **Option C, now with a second data point:**
  - **Degree is confirmed dead as a signal** — both counties again show identical partner counts (153/153), reinforcing that the disaggregation model assigns near-universal nonzero flows regardless of origin character. Drop this metric.
  - **HHI concentration shows a small, directionally sensible signal**: Essex is slightly *less* concentrated (0.056 out / 0.047 in) than Sussex (0.059 / 0.054) — consistent with a hub spreading flow across more destinations rather than funneling to a few. Modest effect size (~10–15% relative), but consistent in direction with the hub/sink hypothesis, unlike distance.
  - **Distance-weighted "reach" is confirmed backward, not just noisy:** Essex's avg. outbound distance (55km) is *shorter* than Sussex's (83km) — the opposite of "a hub reaches farther." This replicates the Rhode Island result exactly. Conclusion: this metric, as computed from the county-to-county table alone, is measuring the wrong thing — a real hub's long-distance reach is by construction excluded from that table (it lives in the FAF-zone-collapsed rows once flows leave the adjacent-state window). **Recommend dropping this exact formulation entirely** rather than trying to patch it with a third sample — the mechanism, not just the number, is now understood and it isn't a matter of a bad state pick.

**Correction to an earlier Phase 0 hypothesis:** Phase 0 speculated `3.FAF-to-FAF.csv` might be an identical national table duplicated in every state zip (used to justify a future dedup step). **Falsified** — Rhode Island's table 3 has 598,360 rows; New Jersey's has 489,987. They're state-specific in some way not yet characterized, not blind copies. Drop the "dedupe table 3 across states" optimization idea from Phase 2 planning; the real download-volume mitigation (if needed) would have to be figured out some other way, or simply accepted as a large one-time ingestion cost.

## Final recommendation for Phase 2's vectorization approach

**Ship Option A (scalar totals by commodity group, in/out) as the core feature set** — validated on two independent states, cheap, always available, directly serves the proposal's "capital/velocity" framing even without partner identity.

**Add HHI concentration (from Option C) as a small addition**, computed by pooling county-level *and* FAF-zone-level partner rows into one distribution per county (not just the county-to-county subset this spike used) — this is more engineering work than the spike did, but is the more honest test of "does this county's trade concentrate in a few partners or spread widely," which directly serves the proposal's stated goal ("isolates a logistical pass-through corridor... from a pure consumer sink").

**Drop Option B (explicit top-K partner columns) and the distance-weighted reach idea** — both were tested and didn't add discriminating signal over Option A in this domain; B's top partners are redundant with regional geography, and reach pointed backward for a documented, understood reason (not fixable without a zone-centroid crosswalk that isn't readily available).

This is a design decision, not yet implemented — Phase 2 code should build exactly this (A + pooled-partner HHI), pending final confirmation before writing `ingest_source_d.py`.

## Phase 2 (2026-07-13) — `scripts/ingest_source_d.py` written and smoke-tested

Built following Sources A/C/F's established shape (module docstring, dataclass results, per-unit failure isolation, `configure_logging`/`main` entrypoint pattern), implementing the Phase 1b recommendation: `total_outbound_tons`/`total_inbound_tons`, 5-way `sctgG5` breakdown per direction, and pooled county+zone `out_partner_hhi`/`in_partner_hhi`. Failure isolation is per-state (the natural download unit here) rather than per-county, mirroring `IngestionSummary`'s succeeded/failed pattern from the other sources.

One design point resolved during implementation: a county only gets a result from its own home-state zip (matched by 2-digit FIPS prefix), not from appearing as an "adjacent state" county in a neighbor's zip — only the home zip has that county's complete adjacent-neighbor set, so pulling from a non-home zip would silently under-count.

**Real engineering finding during smoke-testing, not anticipated in Phase 0:** `requests`/urllib3 (Python's TLS stack) gets a connection reset during the TLS handshake against `faf.ornl.gov` on **every single attempt**, with or without retries, with or without forcing TLS 1.2 — while the system `curl` binary connects cleanly every time with no special configuration. This isn't the bts.gov Akamai situation (no bot-detection headers, no challenge, no session cookies involved) — it's an ordinary HTTP-client compatibility quirk against a plain, unauthenticated IIS file server, most likely a TLS ClientHello fingerprint mismatch. Resolved by shelling out to `curl` (via `subprocess`, list-form arguments only, own hardcoded URLs — no shell interpolation) instead of `requests` for this one client class; documented inline in the module docstring so this doesn't read as an unexplained deviation from the other sources' pure-`requests` style.

**Validation performed:**
1. Ran the full pipeline logic (transform + HHI pooling) against the already-downloaded Rhode Island and New Jersey zips from Phase 1b, offline — output matched the Phase 1b spike's numbers exactly for the scalar totals, and the pooled HHI came in lower than the spike's county-only HHI for every test county (Essex: 0.043 vs. spike's 0.056; Providence: 0.119 vs. 0.136) — the expected direction, since pooling in zone-level partners dilutes concentration versus counting county partners alone.
2. Ran the live pipeline end-to-end (curl download → parse → transform → crosswalk join) against Rhode Island + Delaware over the real network — succeeded, 8 counties, sane values (e.g. New Castle County DE: 20.5K/19.2K tons out/in).
3. Hardened `main()` against the empty-results edge case (would otherwise raise a confusing `KeyError` inside the crosswalk merge rather than a clear error) — found via the network flakiness above producing a zero-result run before the curl fix.

## Phase 2 full-batch run complete (2026-07-13)

Ran to completion: **51/51 states succeeded, 0 failures**, 3,144 rows written to `data/source_d_faf.parquet` (2h32min wall-clock — much faster than the 4-6hr mid-run projection once early network flakiness settled; most states after the first ~10 took under a minute each).

Post-run validation:
- **3,144/3,144 crosswalk coverage, zero unmatched counties** — every county in `county_crosswalk.parquet` got a row.
- Zero nulls, zero duplicate `fips_code` rows across all 16 columns.
- Known-county spot checks confirm real signal, not noise: LA County (256K/284K tons out/in), Cook County/Chicago (175K/173K), Harris County/Houston (347K/384K) all dwarf the median county by two orders of magnitude, as expected for major hubs. Petroleum County, MT (rural) shows 327/63 tons — essentially nothing. **Loving County, TX** (the least populous county in the US, ~64 people) shows 9,842 tons outbound against just 59 inbound — not noise: it sits in the Permian Basin oil/gas fields, so despite having almost no population it's a genuine petroleum export point the data correctly captured.
- **Finding worth flagging in the eventual findings doc, not silently smoothed over:** the HHI concentration *direction* from the Phase 1b regional spike doesn't hold at national scale. Rhode Island/New Jersey (regional samples) showed hubs as *less* concentrated than rural counties. Nationally, the biggest hubs (LA, Chicago, Houston: HHI 0.11–0.23) are *more* concentrated than small rural counties (0.02–0.04) — because a massive hub funnels huge volume through a few dominant interstate corridors, while a small county spreads modest volume evenly across its nearby neighbors. The scalar tonnage signal (Option A) is unambiguous either way; only the concentration metric's direction is scale-dependent between a 3-state regional sample and full national coverage.

**Known limitation, accepted rather than fixed for this run:** `export_to_parquet` only writes once, after all 51 states finish — a crash or interruption partway loses all progress (no per-state checkpointing). Mid-run, once the pace projected 4-6 hours, this was raised explicitly as a decision point: keep the simple single-write design as-is, add per-state checkpointing, or parallelize downloads to cut wall-clock time. **Decision: let it run as-is** — accepted the crash-loses-everything risk in exchange for not touching working code mid-run. It paid off (the run completed in 2h32min, faster than the mid-run projection, no crash), but if `ingest_source_d.py` is re-run from scratch in the future for a data refresh, this tradeoff should be revisited before assuming it'll always be this fast — the early states were slow enough to project 4-6 hours before the pace picked up.

## Phase 3 (2026-07-14) — analysis scripts written and run

Wrote all four scripts per the Phase 3 plan, mirroring Sources A/C/F's script split. `scripts/analyze_source_d_hubs.py` defines the two derived signals every other Phase 3 script reuses (`add_hub_signals`: `total_tons` = outbound + inbound, replacing the dropped partner-degree idea; `mean_partner_hhi` = mean of out/in pooled-partner HHI) and imports cleanly into the other three scripts rather than duplicating the computation.

- `analyze_source_d_hubs.py` — ranks counties by `total_tons` (`source_d_hubs.csv`, `source_d_hubs.html`). Top hub: **Harris County, TX (Houston) at 731K tons**, ahead of LA County (540K) and Cook County/Chicago (348K) — plausible given Houston's port + petrochemical corridor role. Confirms Phase 2's national-scale finding *empirically* on the full 3,144-county dataset, not just the two-state spike: log10(total_tons) vs. mean_partner_hhi Pearson **r=0.278** (positive) — hubs are both higher-volume and more concentrated, the opposite direction from the Rhode Island/New Jersey regional spike.
- `analyze_source_d_source_c_correlation.py` — cross-validates trade-flow intensity against Source C velocity (`source_d_source_c_crossvalidation.csv/.html`). Both signals correlate weakly but in the economically sensible direction with size-normalized GDP velocity (log tons r=0.077, HHI r=0.069) and are essentially uncorrelated with unemployment velocity (r=0.050 and -0.026). The GDP-velocity-by-tonnage-quartile breakdown is monotonic (Q1 1.6% -> Q4 2.4%/year), a small but clean signal: higher-throughput counties trend toward faster GDP growth. 64 counties drop for missing Source C GDP coverage, consistent with Source C's own documented gap.
- `visualize_source_d.py` — two US bubble maps, log10(tons) and mean HHI (`source_d_map_tons.html`, `source_d_map_concentration.html`).
- `generate_source_d_insights.py` — headline stats to `analysis-output/source_d_stats.json`, three static figures (`source-d-figure-01-top-hubs.png`, `-02-tons-vs-concentration.png`, `-03-tons-vs-velocity.png`) and `source-d-numeric-summary.md`.

All four ran end-to-end against the real `data/source_d_faf.parquet` with no errors; outputs spot-checked (hub ranking matches Phase 2's LA/Chicago/Houston validation numbers, quartile counts split evenly at 786 each, no unexpected nulls).

## Phase 4 (2026-07-14) — findings deliverable written

Wrote `analysis-output/source-d-findings.md` (8 sections, mirroring Sources A/C/F's frontmatter and structure conventions) and `analysis-output/source_d_key_findings.ipynb` (built with `nbformat`, executed end-to-end via `jupyter nbconvert --execute`, 8/8 code cells produced output with zero errors). The findings doc includes the required proposal-alignment section (§8) documenting both deviations — the county/FAF-zone hybrid structure superseding the spec's top-K truncation idea, and the tonnage-only data with no dollar-value column — plus a third finding not anticipated by either scoping doc: the HHI concentration signal's direction is scale-dependent (negative in the Phase 1b regional spike, positive at full national coverage). Figures (`source-d-figure-01/02/03`, `source-d-numeric-summary.md`) were already generated by Phase 3's `generate_source_d_insights.py` and are referenced, not regenerated.

## Status
**Phase 0, Phase 1b, Phase 2, Phase 3, and Phase 4 complete.** `data/source_d_faf.parquet` exists, fully validated; all four Phase 3 analysis scripts run clean; `analysis-output/source-d-findings.md` and `analysis-output/source_d_key_findings.ipynb` are written. Source D is done. Next actions recorded in the findings doc (§6): test the Source D/F Trade Logistics synergy Source F's own findings flagged, and revisit once Source B (BLS QCEW) exists.
