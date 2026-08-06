"""Generate analysis-output/weekly-brief-2026-08-06.ipynb.

Narrative-first brief. Code cells only read outputs/ and analysis-output/ and plot;
no analysis is re-run here.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path("/Users/maxgarcia/Desktop/MacroEmbeddings")
OUT = REPO / "analysis-output" / "weekly-brief-2026-08-06.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------
md("""
# `E_macro` — week of 3 August 2026

**A brief, not an archive.** The evidence and the per-column detail live in
`analysis-output/E_macro_key_findings.ipynb` and the per-source findings
documents. This is what changed this week and what it means.

Every number below is read from committed artifacts in `outputs/` and
`analysis-output/`. Nothing is re-fit here.
""")

code('''
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path.cwd().parent if Path.cwd().name == "analysis-output" else Path.cwd()
OUTPUTS, ANALYSIS = REPO / "outputs", REPO / "analysis-output"

scores = pd.read_csv(OUTPUTS / "external_target_scores.csv")
grain = pd.read_csv(OUTPUTS / "grain_effect.csv")
decile = pd.read_csv(OUTPUTS / "external_target_by_decile.csv")
ext = json.loads((ANALYSIS / "cross-source" / "external_target_stats.json").read_text())
gst = json.loads((ANALYSIS / "cross-source" / "grain_effect_stats.json").read_text())

LABELS = {
    "broadband_rate": "Broadband adoption",
    "median_household_income": "Median household income",
    "median_age": "Median age",
    "median_home_value": "Median home value",
    "mean_commute_minutes": "Mean commute",
}
ORDER = list(LABELS)
INK, BASE, LIFT, WARN = "#1f2a37", "#c7cdd6", "#2563eb", "#dc2626"

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#d1d5db", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
print(f"{ext['n_targets']} targets · {ext['fold_strategy']} · "
      f"{gst['n_markets']} market groups")
''')

# --------------------------------------------------------------------------
md("""
---

## 1. Where it landed

Two positions, both of which I think the evidence now supports:

1. **`E_macro` earns its slot.** Tested against five public outcomes that live
   outside all six pillars, on counties in states the model was never trained on,
   it adds **+0.19 R² over a county-size baseline** — on all five, with no
   exceptions. That is the first result in this project that measures usefulness
   rather than internal agreement.
2. **Grain is no longer a blocker.** Monday's read said joining at market grain
   would destroy the signal. That turned out to be half a finding. Aggregating to
   market level *helps* about as much as the row-count loss hurts, and on three of
   five outcomes the market-level version is the better predictor. Net cost of
   joining at market grain instead of county grain: **0.017 R²**, against a headline
   of +0.19.

Both were open questions on Monday. Neither is now.
""")

# --------------------------------------------------------------------------
md("""
---

## 2. The problem we actually fixed

Every validation in this repo before this week was **pillar against pillar**. We
predicted one federal source's features from the other five. That measures whether
six agencies agree with each other. It cannot say whether any of them is *useful*,
and it has a bias baked in: it penalises a source precisely for agreeing with the
others, which is the wrong penalty when the question is "does this predict an
outcome."

There is no way to fix that with a real downstream label — the project is scoped to
public data only, so no such label exists here. The substitute: pick public outcomes
that **no pillar measures**, and predict those.

The five: household broadband adoption, median household income, median age, median
home value, and mean commute time. All from ACS, none constructed from any pillar's
inputs.

**The test is built around one specific objection.** The consumer joins on DMA and
holds millions of impressions per market, so it can estimate a geographic fixed
effect essentially for free — and a fixed effect makes any static geo-keyed feature
look redundant. But a fixed effect has exactly one weakness: **it has no parameter
for a place it has never seen.** So we hold out whole states, and compare against a
model that knows only county size. That is the seam.
""")

code('''
piv = (scores[scores.model.isin(["size", "size_emacro"])]
       .pivot(index="target", columns="model", values="r2_ablated")
       .reindex(ORDER))
lift = (scores[scores.model.eq("size_emacro")]
        .set_index("target")["lift_over_size_ablated"].reindex(ORDER))

fig, ax = plt.subplots(figsize=(9.2, 4.4))
y = range(len(ORDER))
ax.barh([i + 0.19 for i in y], piv["size"], height=0.36, color=BASE,
        label="County size only", zorder=3)
ax.barh([i - 0.19 for i in y], piv["size_emacro"], height=0.36, color=LIFT,
        label="County size + $E_{macro}$", zorder=3)

for i, t in enumerate(ORDER):
    ax.text(piv["size"][t] + 0.012, i + 0.19, f"{piv['size'][t]:.2f}",
            va="center", fontsize=9, color="#6b7280")
    ax.text(piv["size_emacro"][t] + 0.012, i - 0.19, f"{piv['size_emacro'][t]:.2f}",
            va="center", fontsize=9, color=LIFT, fontweight="bold")
    ax.text(0.985, i, f"+{lift[t]:.3f}", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=10, fontweight="bold", color=INK)

ax.set_yticks(list(y), [LABELS[t] for t in ORDER])
ax.invert_yaxis()
ax.set_xlabel("R² on held-out states")
ax.set_xlim(0, 1.0)
ax.set_title("Five outcomes outside every pillar. Five for five.", pad=26, loc="left")
ax.text(0, 1.045, f"Mean gain over size alone: +{ext['mean_lift_over_size_ablated']:.3f} R²"
        "        (gain per outcome at right)",
        transform=ax.transAxes, fontsize=10, color="#6b7280")
ax.legend(frameon=False, fontsize=9.5, loc="upper left",
          bbox_to_anchor=(0, -0.14), ncol=2)
ax.grid(axis="x", color="#eef0f3", zorder=0)
plt.tight_layout()
plt.show()
''')

md("""
Read the grey bar as the fixed-effect model's position and the blue bar as what
`E_macro` adds on top of it. The gap on the right of each row is the whole result.

One sanity check worth stating: an intercept-only model scores **≈0** on these
held-out states. That is the fixed effect being handed a county it has never seen,
behaving exactly as predicted — it has nothing to say.
""")

# --------------------------------------------------------------------------
md("""
---

## 3. The discount I applied to my own result

The raw number was better: +0.212. I am reporting **+0.190**, and the difference is
worth a paragraph because it is the kind of thing that gets caught in review rather
than found by the author.

Two pillar columns don't *predict* their target so much as **restate** it:

- `wage_per_return_thousands` (IRS) is average wage income per tax return, which is
  very close to a definition of median household income. Removing it drops that
  outcome's gain from +0.247 to **+0.154** — one column was carrying 38% of the
  apparent result.
- `retirement_destination` (USDA) flags counties with heavy in-migration of people
  aged 60+, which restates age structure. Smaller effect: +0.256 → +0.239.

Both are dropped from their own target's run and kept everywhere else. The headline
is the discounted number.
""")

# --------------------------------------------------------------------------
md("""
---

## 4. The grain reversal

This is the part I got wrong on Monday and corrected twice.

The consumer joins at DMA level — about 210 markets, versus 3,143 counties. That
penalty has **two separable halves**, and Monday I only measured one:

- **Fewer rows to learn from.** Measured by retraining on random county subsets.
  Verdict: real and bad. Worth **−0.121** on average, and badly unstable at that
  size — on broadband, the outcome closest to the consumer's own domain, it goes
  slightly negative.
- **Aggregation blurring within-market detail.** Not measured Monday. I said it
  "could cut either way."

It cuts the other way. Aggregating the *inputs* to market level — not averaging the
outputs — is worth **+0.105**, which very nearly cancels the row-count loss.
""")

code('''
by_size = pd.DataFrame(ext["by_training_size"])

TICKS = [210, 400, 800, 1600, 3000]
palette = ["#2563eb", "#0f766e", "#b45309", "#7c3aed", "#be185d"]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                               gridspec_kw={"width_ratios": [1.35, 1]})
for c, t in zip(palette, ORDER):
    d = by_size[by_size.target.eq(t)].sort_values("n_train_units")
    axl.plot(d.n_train_units, d.mean_lift_over_size, "-o", color=c, ms=4.5,
             lw=2.0, label=LABELS[t], zorder=3)
    axr.plot(d.n_train_units, d.sd_lift_over_size, "-o", color=c, ms=4.5,
             lw=2.0, zorder=3)

for ax in (axl, axr):
    ax.set_xscale("log")
    ax.set_xticks(TICKS, ["210", "400", "800", "1,600", "3,000"])
    ax.axvline(210, color=WARN, lw=1.3, ls="--", zorder=4)
    ax.set_xlabel("Counties available for training  (log)")
    ax.grid(color="#eef0f3", zorder=0)

axl.axhline(0, color="#9ca3af", lw=1)
axl.text(228, axl.get_ylim()[1] * 0.95, "≈ DMA count", color=WARN,
         fontsize=9.5, fontweight="bold", va="top")
axl.set_ylabel("Gain over size-only baseline (R²)")
axl.set_title("The gain shrinks…", fontsize=11.5, loc="left", pad=8)
axr.set_ylabel("Spread across 10 random draws (sd)")
axr.set_title("…and stops being reliable", fontsize=11.5, loc="left", pad=8)
axr.text(228, axr.get_ylim()[1] * 0.95, "≈ DMA count", color=WARN,
         fontsize=9.5, fontweight="bold", va="top")

fig.suptitle("Half the story: fewer rows hurt, and get unreliable",
             x=0.008, y=1.06, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.008, 0.99, "At 210 units the spread on some outcomes is wider than the "
         "effect being measured.", ha="left", fontsize=9.5, color="#6b7280")
fig.legend(frameon=False, fontsize=9.5, loc="lower left",
           bbox_to_anchor=(0.008, -0.12), ncol=3)
plt.tight_layout()
plt.show()
''')

md("""
The right panel matters as much as the left. At 210 units the answer depends heavily
on which units you happen to have — that instability, more than the size of the drop,
is what made a market-grain join look unacceptable.

Then the other half got measured:
""")

code('''
arms = (grain.pivot(index="target", columns="arm", values="mean_lift_over_size")
        .reindex(ORDER))
ARMS = [("county_full", "All 3,143 counties", "#2563eb"),
        ("county_subsample", "208 counties (row-count loss only)", "#c7cdd6"),
        ("market_aggregate", "208 aggregated markets", "#0f766e")]

fig, ax = plt.subplots(figsize=(9.6, 4.6))
x = range(len(ORDER))
for k, (col, lab, c) in enumerate(ARMS):
    off = (k - 1) * 0.27
    vals = arms[col]
    ax.bar([i + off for i in x], vals, width=0.25, color=c, label=lab, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i + off, v + (0.012 if v >= 0 else -0.028), f"{v:+.2f}",
                ha="center", fontsize=8.6,
                color=WARN if v < 0 else "#374151",
                fontweight="bold" if col == "market_aggregate" else "normal")

ax.axhline(0, color="#6b7280", lw=1)
ax.set_xticks(list(x), [LABELS[t].replace(" ", "\\n", 1) for t in ORDER])
ax.set_ylabel("Gain over size-only baseline (R²)")
ax.set_title("The other half: aggregation helps, and nearly cancels the loss",
             pad=26, loc="left")
ax.text(0, 1.05, f"Row-count effect {gst['row_count_effect']:+.3f}   ·   "
        f"aggregation effect {gst['aggregation_effect']:+.3f}   ·   "
        "market arm wins on 3 of 5",
        transform=ax.transAxes, fontsize=10, color="#6b7280")
ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
ax.grid(axis="y", color="#eef0f3", zorder=0)
plt.tight_layout()
plt.show()
''')

md("""
Compare the grey bar to the teal one — same 208 rows in both, the only difference is
whether those rows are lone counties or aggregated markets. **Median home value goes
+0.12 → +0.34. Median age goes +0.21 → +0.41.** The mechanism is not exotic:
population-weighted aggregation turns sparse, noisy county columns — suppressed BLS
cells, single-article Wikipedia flags — into stable continuous shares, and does the
same favour to the outcome being predicted.

**Then I tried to break it.** The obvious objection was that most columns were being
*approximated* at market level rather than properly re-derived, which would make the
market arm look artificially good. So Source B was re-ingested to ship raw employment
levels, moving 72 of 118 columns from approximated to correctly re-derived. Re-running
everything moved the aggregation effect by **0.001**, and no outcome changed sign.

That was expected to cost 2–3 days. It cost one download and two script changes.

**What still stands as a caveat:** the 208 market groups are k-means clusters of
county centroids, matched to DMA cardinality — they are *not* DMAs, because that
delineation is proprietary. Real markets follow media boundaries and are less
spatially compact, and the aggregated outcome is genuinely less noisy than a county
one. Both biases favour the market arm, so **+0.105 is an upper bound.**

How much does that matter? Two different thresholds, worth keeping apart:

- For market grain to be a **blocker** again — signal destroyed rather than merely
  reduced — essentially the whole +0.105 would have to be an artifact of the proxy.
  That is a large claim and I don't think the biases named above are anywhere near
  big enough to support it.
- For county grain to be **strictly better**, the overstatement only has to be
  **0.017** — that is the gap between the full-county arm (+0.212) and the market
  arm (+0.195). That is a small enough margin that I would not argue the market arm
  is *better*, only that it is not disqualifying.
""")

# --------------------------------------------------------------------------
md("""
---

## 5. Where the model cannot win, and why that is fine

One result looks alarming until you see what is under it: on the smallest counties,
the size-only baseline scores **negative** R². Something is badly wrong there — but
it is wrong with the data, not the model.

ACS publishes a margin of error with every estimate. Those are now ingested alongside
the values, which lets us split each outcome's variance into signal and sampling
noise. In the smallest population decile, **30% of the variance is sampling noise** —
error no model can ever explain. By the largest decile that is under 1%.
""")

code('''
dec = (decile.groupby("population_decile")
       .agg(median_population=("median_population", "mean"),
            noise_share=("noise_share", "mean"),
            r2_size=("r2_size", "mean"),
            r2_size_emacro=("r2_size_emacro", "mean"))
       .reset_index())

fig, ax = plt.subplots(figsize=(9.4, 4.5))
ax.bar(dec.population_decile, dec.noise_share, width=0.62, color="#fde2e2",
       edgecolor="#f5b8b8", label="Share of variance that is ACS sampling noise", zorder=2)
ax.plot(dec.population_decile, dec.r2_size, "-o", color="#9ca3af", ms=5, lw=1.9,
        label="County size only", zorder=4)
ax.plot(dec.population_decile, dec.r2_size_emacro, "-o", color=LIFT, ms=5, lw=2.2,
        label="County size + $E_{macro}$", zorder=5)
ax.axhline(0, color="#6b7280", lw=1, zorder=3)

worst = dec.iloc[0]
ax.annotate(f"{worst.r2_size:.2f}", xy=(1, worst.r2_size), xytext=(1.35, -0.40),
            fontsize=9.5, color=WARN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=WARN, lw=1))
ax.text(1.0, worst.noise_share + 0.03, f"{worst.noise_share:.0%}\\nnoise",
        ha="center", fontsize=9, color="#b91c1c", fontweight="bold")

ax.set_xticks(range(1, 11),
              [f"{int(p):,}" for p in dec.median_population.round(-2)], fontsize=8.4)
ax.set_xlabel("County population decile  (median population in each)")
ax.set_ylabel("R², averaged over the five outcomes")
ax.set_title("Small counties are mostly measurement error", pad=24, loc="left")
ax.text(0, 1.045, "Where the grey line dives, the data is noise — and $E_{macro}$ still "
        "recovers usable signal there.",
        transform=ax.transAxes, fontsize=9.5, color="#6b7280")
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.grid(axis="y", color="#eef0f3", zorder=0)
plt.tight_layout()
plt.show()
''')

md("""
Two things follow. First, the negative baseline on tiny counties is a property of ACS,
not a defect in the pipeline. Second — and more useful — `E_macro` stays positive in
exactly the decile where the size baseline collapses, which is what you would want
from a feature meant to describe places that a size proxy describes badly.

It also sets an honest ceiling: on the smallest counties no model can exceed R² ≈ 0.70
no matter how good the features are.
""")

# --------------------------------------------------------------------------
md("""
---

## 6. Plumbing, briefly

Work that doesn't change the story but does change what ships:

- **Source D's freight tonnages were county size wearing a freight label.** All ten
  raw tonnage columns moved into the size control. Measured cost of removing them:
  nothing. Normalising them per capita turned out to be a re-expression, not a fix.
- **Commodity shares replaced them** — 5 of 10 clear the size-free bar that none of
  the raw columns cleared, and the gain routes almost entirely to Source B, which is
  interpretable rather than mysterious.
- **Source E's capital-to-wage ratio was decomposed** into its components, plus a
  five-year panel and volume tiers. Its remaining dollar totals moved into the size
  control on the same principle as D's.
- **Source B now ships raw employment levels**, so its 40 location quotients can be
  re-derived at any grain instead of approximated. This is what made the grain
  re-test above trustworthy.
- **Source A** was measured for marginal value against a baseline that already
  contains the other five pillars — the harder question — and the last
  embedding-era artifacts were retired repo-wide.
- **The archive notebook was re-read against regenerated outputs.** Several Section 8
  numbers had drifted after D and E moved into the size control; corrected.

### What's next, regardless

1. Re-derive Source D's two partner-concentration indices from the partner-level flow
   table — the last columns that are still approximated at market grain with an
   underlying quantity to re-sum.
2. Build the assembly step. The go/no-go evidence now exists, which was the thing
   gating it.
3. Keep the grain caveat live: if a real DMA delineation ever becomes available, the
   market-arm result is worth re-running against it once.

---

*Sources: `analysis-output/cross-source/external-target-findings.md` (§10–§20),
`analysis-output/source-d/source-d-findings.md`,
`analysis-output/source-e/source-e-findings.md`, `docs/plans/dma_regrain.md`.*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
