"""Generate analysis-output/E_macro_pillar_worth_2026-08-13.ipynb.

An executive notebook for the commissioning side and their leadership. A progress
artifact, not the go/no-go: it reports what the project knows, and does not ask
for the decision. Content only -- no timing notes, run stamps or other framing
about the document itself.

**Present from the HTML, not from the notebook.**

    uv run scripts/build_status_notebook.py --for-html

The notebook is the reproducible source of truth and belongs in git; the HTML in
outputs/ is what goes on a screen. The code cells here carry
`jupyter.source_hidden`, which folds them in JupyterLab but is **ignored by the
VS Code and Cursor notebook editors** — there they render expanded, and the only
remedy inside the editor is the "Notebook: Collapse All Cell Inputs" command,
re-run every time the file is reopened. `--for-html` removes the cells outright
instead, so there is nothing to forget.

Figures are matplotlib, not plotly. Plotly's mimetype output needs a JupyterLab
extension to render and degrades to blank space when that extension is absent,
which is what happens in a `uv run --with jupyterlab` environment. Matplotlib
emits a PNG that every notebook client renders and that embeds directly in the
HTML export, so no network and no extension are involved.

Design: docs/superpowers/specs/2026-08-13-exec-status-notebook-design.md

**The spine is the assignment.** The commissioning side asked for Sources A and E
to be split into groups and tested for whether they should be modelled separately.
Sections 1-4 answer that (no, for both, and here is what the groups exposed
anyway); section 5 generalises the same question to all six pillars; section 6 settles how Source A should be represented, finds the win is
geography, and puts an interval on what is left; section 7 is
limits and status. The through-line is that a pillar has to earn the right to
branch, and therefore also has to earn its slot at all.

This notebook **supersedes and absorbs** analysis-output/weekly-brief-2026-08-06.ipynb,
which was written for a conversation that never happened. The brief and its
generator were removed in the commit that added this file; recover them from git
history if needed.

Every figure and every table is computed from the committed artifacts in
outputs/ and analysis-output/, which were regenerated from data/*.parquet on
2026-08-13. Numbers quoted inline in prose are transcribed from those same
artifacts by hand and are the one thing here that does not move on its own.

**Prose refers to sections by name, never by number.** The section order has
changed twice and every numbered cross-reference in the file was stale by the
second change. Names survive a reorder; numbers do not.

Wording in the Source A and Source F sections is constrained by
analysis-output/cross-source/pillar-marginal-findings.md section 9. See the
spec's "Wording constraints" section before editing those cells.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path("/Users/maxgarcia/Desktop/MacroEmbeddings")
STEM = "E_macro_pillar_worth_2026-08-13"
OUT = REPO / "analysis-output" / f"{STEM}.ipynb"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--for-html", action="store_true",
    help="Build, execute and export a code-free HTML to outputs/ instead of "
         "writing the committed notebook. Use this to present without any code "
         "cells at all; the notebook route keeps them, merely folded.",
)
args = parser.parse_args()

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell tagged to open with its source folded.

    `jupyter.source_hidden` is a JupyterLab convention and JupyterLab honours it.
    The VS Code and Cursor notebook editors do not — they have their own collapse
    state and ignore this metadata, so the cells open expanded there. The brief
    this notebook replaces claimed otherwise in its own docstring; that claim was
    wrong and is corrected here.

    The metadata is kept because it costs nothing and helps in JupyterLab, but the
    presenting path is `--for-html`, which drops the cells entirely.
    """
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    cell.metadata["jupyter"] = {"source_hidden": True}
    cell.metadata["collapsed"] = True
    cells.append(cell)


# ==========================================================================
# Title
# ==========================================================================
md("""
# `E_macro` — which pillars earn their slot

The assignment was to split Sources A and E into groups and see whether they
should be modelled separately. The answer is **no, for both** — and the groups
were worth having anyway, because of what they exposed on the way to that answer.

The same question then turned out to generalise: if a pillar has to earn the right
to branch, it also has to earn its slot at all. The pillar-worth section asks
that of all six.
""")

code('''
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Walk up to the repo root rather than assuming a directory name: this notebook
# is executed from analysis-output/ when it is the deliverable and from build/
# when it is staged for the HTML export, and both have to resolve the same way.
REPO = Path.cwd()
while not (REPO / "analysis-output").is_dir() and REPO != REPO.parent:
    REPO = REPO.parent
OUTPUTS, ANALYSIS = REPO / "outputs", REPO / "analysis-output"
XSRC = ANALYSIS / "cross-source"

# ---- artifacts -----------------------------------------------------------
ext = json.loads((XSRC / "external_target_stats.json").read_text())
blk = json.loads((XSRC / "pillar_block_marginal_stats.json").read_text())
# The five original ACS targets. The external sweep now scores 42, and the
# pillar-worth figure stays on the five its prose describes; the wide basket is
# reported separately under `geo_control` and `bootstrap`.
HEADLINE = ext["headline_basket"]
gst = json.loads((XSRC / "grain_effect_stats.json").read_text())
scope = json.loads((ANALYSIS / "source-a" / "source_a_section_scope_stats.json").read_text())
embed = json.loads((ANALYSIS / "source-a" / "source_a_tiered_embedding_stats.json").read_text())
asec = json.loads((ANALYSIS / "source-a" / "source_a_section_stats.json").read_text())
atier = json.loads((ANALYSIS / "source-a" / "source_a_tier_stats.json").read_text())
arep = json.loads((ANALYSIS / "source-a" / "source_a_representation_stats.json").read_text())
mrep = json.loads(
    (ANALYSIS / "source-a" / "source_a_representation_marginal_stats.json").read_text())
acomp = json.loads(
    (ANALYSIS / "source-a" / "source_a_section_composition_stats.json").read_text())
etier = json.loads((ANALYSIS / "source-e" / "source_e_tier_stats.json").read_text())

# Tables here carry sentence-length cells. pandas elides anything past 50
# characters in BOTH the text and the HTML repr, so an un-widened table ships to
# the export with its explanation replaced by "...".
pd.set_option("display.max_colwidth", None)

A_TIERS = ["stub", "thin", "mid", "rich"]

placebo = pd.read_csv(OUTPUTS / "external_target_drop_one_placebo.csv")
decile = pd.read_csv(OUTPUTS / "external_target_by_decile.csv")
vintages = pd.read_csv(OUTPUTS / "pillar_vintages.csv")

PILLARS = ["A", "B", "C", "D", "E", "F"]
PILLAR_NAME = {
    "A": "A · Place identity",
    "B": "B · Industrial core",
    "C": "C · Economic velocity",
    "D": "D · Trade logistics",
    "E": "E · Capital flow",
    "F": "F · Structural resilience",
}
TARGET_LABEL = {
    "broadband_rate": "Broadband adoption",
    "median_household_income": "Median household income",
    "median_age": "Median age",
    "median_home_value": "Median home value",
    "mean_commute_minutes": "Mean commute",
}
TARGET_ORDER = list(TARGET_LABEL)

# ---- palette -------------------------------------------------------------
# Same validated reference palette as the brief this notebook replaces.
# Categorical slots are fixed, never cycled: a series keeps its slot when
# others are added or removed.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, AQUA, CRITICAL = SERIES[0], SERIES[2], "#d03b3b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID = "#d5d4d0", "#ecebe6"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 110,
})


def style(ax, title, subtitle="", grid_axis="y", legend=False):
    """House chart style, sized for a projector rather than a laptop.

    Matplotlib rather than plotly: plotly's mimetype output needs a JupyterLab
    extension to render, and silently produces blank space when that extension
    is missing. Matplotlib emits a PNG that every notebook client and the HTML
    export display without any additional machinery.

    Chrome is recessive — no spines, a hairline grid on one axis only, tick
    labels a step down from the body — so the bars carry the figure.
    """
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=INK2, labelsize=10.5)

    # Subtitles run long; wrapping by hand keeps them off the plot area, which
    # matplotlib will not do on its own. Both title and subtitle are positioned
    # in points offset from the axes edge rather than in axes fractions -- mixing
    # the two units makes them overlap, and by how much depends on figure height.
    sub = textwrap.fill(subtitle, 104) if subtitle else ""
    n_lines = sub.count("\\n") + 1 if subtitle else 0
    line_h = 14.5
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK,
                 pad=(8 + n_lines * line_h + 8) if subtitle else 12)
    if subtitle:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    va="bottom", ha="left", fontsize=10, color=INK2,
                    linespacing=1.45)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2,
                  frameon=False, fontsize=10.5, labelcolor=INK2,
                  handlelength=1.4, columnspacing=1.8)
    return ax


def label_bars(ax, xs, ys, fmt="{:+.4f}", size=10.5, pad=0.012, horizontal=False):
    """Write each bar's value just past its end, outside the bar.

    `pad` is a fraction of the axis span, so labels sit the same visual distance
    from the bar regardless of the units on the axis.
    """
    lo, hi = (ax.get_xlim() if horizontal else ax.get_ylim())
    off = (hi - lo) * pad
    for x, y in zip(xs, ys):
        if horizontal:
            ax.text(y + (off if y >= 0 else -off), x, fmt.format(y),
                    va="center", ha="left" if y >= 0 else "right",
                    fontsize=size, color=INK)
        else:
            ax.text(x, y + (off if y >= 0 else -off), fmt.format(y),
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=size, color=INK)


# ---- evidence base -------------------------------------------------------
print(f"Evidence base · {ext['headline_basket']['n_targets']} external targets for the "
      f"pillar-worth figure, {ext['n_targets']} for the geography control "
      f"· {ext['fold_strategy']} · {blk['n_null_reps']} null reps "
      f"· seed {ext['random_seed']}")
''')

# ==========================================================================
# 1. The assignment, and the evidence baskets
# ==========================================================================
md("""
---

## 1. The assignment

**The task, as set:** split Source A and Source E into groups and find out whether
each pillar should be modelled separately by group.

**Why it was a real question.** Both pillars are built from sources whose *quality
varies enormously across counties*. A Wikipedia article can be three sentences or
three pages; an IRS county file can cover 900 tax returns or 900,000. If a pillar
behaves differently enough at the two ends, one global model fitted across all
3,144 counties is the wrong shape, and the fix is to let the groups have their own
parameters.

**The answer, for both pillars: no.** Neither should branch. The two tier sections
below are what the groups exposed; the branching verdict after them is why
branching loses anyway.

**What the groups were worth regardless.** They picked the feature family Source A
now ships, showed where to go looking for it, established how Source E's ratio
behaves across county sizes, and forced a disclosure that now travels with the
pillar. The tiers were never going to be the deliverable — they were the
instrument.

### Six terms, used precisely

| Term | What it means here |
|---|---|
| **pillar** | One of the six sources (A–F), each a named group of columns in the shipped matrix. |
| **block** | A pillar's columns treated as one unit for scoring — you add or drop a block, never a single column. |
| **arm** | One version of a thing being compared, holding everything else fixed. "The 29-dimension arm" is the same vectors as the 384-dimension arm, reduced. |
| **tier** | A group *inside* one pillar, cut on how much raw material a county has: Source A's stub/thin/mid/rich, Source E's T1–T4. |
| **basket** | The set of target columns a number is averaged over. Four of them exist and they are not interchangeable — the next table is which is which. |
| **restatement ablation** | Removing a pillar column that simply restates the target, before scoring. Without it a pillar gets credit for repeating the answer back. |

### The evidence baskets

Every result in this notebook is an average over a set of target columns — the
things being predicted. Four such sets appear, and **a number from one cannot be
compared with a number from another**: different targets, different baselines,
different models. Each is named wherever its numbers are quoted.

The first two are internal — pillars predicting each other, which measures whether
the six sources agree. The last two are external ACS outcomes, which is the only
way to ask whether any of them is useful. The tier sections and the branching
verdict use the first basket; the ACS ones arrive, and are explained, in the
pillar-worth section.
""")

code("""
# The wide ACS basket carries two uses -- the geography control and the Source A
# representation decision -- but it is one target set: both artifacts pull the
# same 42 candidates and exclude the same degenerate one. Listing it twice was
# what made this table read as five baskets when the text says four.
assert ext["n_targets"] == mrep["n_targets"] - len(mrep["excluded_targets"])

baskets = pd.DataFrame([
    ("Internal — Source A vs the rest", arep["n_targets"],
     "features of the other five pillars",
     "both tier sections, the branching verdict, appendix A2"),
    ("Internal — every pillar vs the rest", blk["n_targets"],
     "features inside the six-pillar matrix",
     "pillar coherence, and Source F's internal number in appendix A5"),
    ("External ACS — the five", ext["headline_basket"]["n_targets"],
     "public survey outcomes no pillar is built from",
     "the +0.190 headline and the pillar-worth figure"),
    ("External ACS — the wide pull", ext["n_targets"],
     f"the same survey, widened to {ext['n_targets_scored']} targets less "
     f"{len(ext['excluded_targets'])} too degenerate to score",
     "the geography control, the intervals, and the Source A "
     "representation decision"),
], columns=["Basket", "Targets", "What is predicted", "Where it is used"])
display(baskets.set_index("Basket"))
""")

md("""
**One of them is not always complete.** Within-tier scores on the in-repo basket
use 21 of its 28 targets for the stub tier: seven BLS location quotients
(`lq_emp_11`, `21`, `22`, `55`, `61`, `62`, `99`) are suppressed in enough small
counties that stub falls under the 150-row floor the scorer requires before it
will report a within-tier lift. Every chart that inherits this says so.

---

## 2. Source A — four content tiers

Counties split on the length of their Wikipedia lead section: **stub** (<100
characters), **thin** (100–283), **mid** (284–461), **rich** (≥462).

**One piece of context first, because it is why Source A looks the way it does.**
Source A used to ship a 1,024-dimension `bge-m3` embedding of each county's lead
section. It was cut and replaced by 29 typed columns, extracted with a fixed
lexicon:

| | the 29 columns |
|---|---|
| **Industry, from the lead** (7) | `has_` + `manufacturing` · `mining` · `oil_gas` · `agriculture` · `tourism` · `timber` · `logistics` |
| **The same seven, re-read from economy sections** (7) | the same names with a `sec_` prefix |
| **Place and infrastructure** (7) | `has_` + `university` · `military_base` · `protected_land` · `tribal_land` · `river` · `interstate` · `port` |
| **Counts and scalars** (8) | `content_length` · `n_industry_mentions` · `sec_n_industry_mentions` · `n_distinct_proper_nouns` · `founding_year` · `has_metro_attachment` · `has_namesake` · `has_economy_section` |

The seven industry flags appearing twice — once from the lead, once from the
economy sections — is the whole subject of the next chart.

**Why columns and not the encoder — two reasons, of which one survived.** At the
time of the cut, neither encoder tested could beat the typed block and neither
lost to it: `bge-m3` at 1,024 dimensions and 2.2GB and `all-MiniLM-L6-v2` at 384
dimensions and 90MB both came out ties on the in-repo basket (Wilcoxon **p =
0.52** and **p = 0.76**). With nothing separating them on lift, the tie broke on
cost and interpretability — no model download, no inference pass, and columns
whose names a reader can check against the article.

**That comparison was confounded, and what keeps the columns is a different
reason.** Both arms read whole articles — ~46% census tables and place-name lists —
and neither was width-matched against 29 columns. Corrected for both, the encoder
wins; but the win is **mostly geography**, and two latitude/longitude columns
already in the repo score **+0.0158** on the decision basket against the encoder's
**+0.0164**. The representation section asks whether typed columns are the right
representation at all. This section takes the 29 as given and asks only whether
they should be fitted separately by tier.

**What the tiers then contributed was the answer to what should be built
instead.** The split showed where economic content actually lives in the corpus.
""")

code('''
tiers = [t for t in A_TIERS if t in asec["by_tier"]]
lead = [asec["by_tier"][t]["share_intro_industry"] for t in tiers]
# `share_section_industry` OVERLAPS the lead measure — a county can name an
# industry in both places — so it is NOT an "after" value, and charting the two
# side by side implies a before/after that does not exist. `share_industry_added`
# is the genuine increment (section_has & ~intro_has), so lead + added is the
# union: the share of counties covered once economy sections are read as well.
added = [asec["by_tier"][t]["share_industry_added"] for t in tiers]
total = [l + a for l, a in zip(lead, added)]
n_by_tier = [asec["by_tier"][t]["n_counties"] for t in tiers]

x = np.arange(len(tiers))
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.bar(x, lead, 0.55, color=BLUE, label="named in the lead section")
ax.bar(x, added, 0.55, bottom=lead, color=AQUA,
       label="added by reading economy sections")
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\\n{n:,} counties" for t, n in zip(tiers, n_by_tier)])
ax.set_ylim(0, max(total) * 1.28)

# Total above each bar, and the multiple it represents -- the multiple is the
# point of the figure, so it is stated rather than left to be eyeballed.
for xi, l, t in zip(x, lead, total):
    ax.text(xi, t + max(total) * 0.03, f"{t:.1%}", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=INK)
    # One decimal: every multiple here is under 10, and rounding 1.54 to "2×"
    # overstates the rich tier's gain in the direction the section argues.
    ax.text(xi, t + max(total) * 0.11, f"{t / l:.1f}× the lead alone", ha="center",
            va="bottom", fontsize=10, color=INK2)

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_ylabel("counties naming an industry", fontsize=10.5, color=INK2)
style(ax,
      "Reading past the lead is what rescues the sparse tiers",
      "Share of counties whose article names an industry. Blue is the lead section "
      "alone; green is what economy sections add. Corpus-wide this took industry "
      "coverage from 8.2% to 18.8%, and it is why the article bodies were refetched.",
      legend=True)
plt.show()
''')

md("""
**Two decisions came out of this chart, and neither is the embedding one.**

**It picked the feature family.** The gradient in the blue bars — 0.7% of stub
articles naming an industry against 25.3% of rich — is what identified *industry*
as the thing worth extracting at all. The 29 typed columns Source A ships are built
around that choice.

**It showed where to find more of it, and that the fix was uniform.** The green
segments are what reading economy-titled sections adds. They help the sparse tiers
most — stub goes from 0.7% to **6.1%**, thin from 1.1% to **9.7%**, against rich's
25.3% to 39.0%. So the thin tiers are not empty pages; they are pages whose
economic content sits below the lead. **Reading *more* for everyone beat reading
*differently* by tier** — the first hint of the answer the branching verdict gives.

The chart is a diagnosis of the corpus, not a verdict on the embedding. It explains
why a lead-text vector would be information-poor; the measurements above are what
actually retired it.

**Then branching lost on its own terms, and the loss scaled with how much branching
there was.** Three ways to fit the *same* 29 columns, from most shared to least.
Exactly one thing varies across them — how many copies of each coefficient the
model estimates, and how much data each copy gets to see.

| Arm | Coefficients estimated | Rows each coefficient sees | Ridge penalty |
|---|---|---|---|
| **One model** — what ships | 29, in one training run | all 3,144 counties | one, shared |
| **Coefficients vary by tier** | 120 — 4 × 29 slopes, plus one intercept per tier — still one training run | only its own tier's counties | one, shared across all four tiers |
| **Four separate models** | 29 × 4, in four training runs | only its own tier's counties | four, each chosen from one tier's data |

The middle arm is the one worth a sentence: each county's features are copied into
its own tier's column slot and zeroed in the other three, so `has_port_stub` sees
stub rows only — that is the 116 slopes, and the four tier dummies prepended to
them let the intercept shift per tier as well — but the penalty strength is still chosen once across all 3,144
rows, so a tier with nothing to say is shrunk by a penalty the other three helped
pick. The bottom arm gives up the shared fit entirely: nothing is borrowed across
tiers, not the coefficients, not the penalty, not the sample.

The three arms are ordered by how much sharing survives: full sharing (one
coefficient set, one sample, one penalty) → shared sample and penalty but split
coefficients → nothing shared at all.
""")

code('''
ARMS = [
    ("One model,\\none coefficient per feature", "extracted_sections", BLUE),
    ("One model, coefficients\\nfree to vary by tier", "sections_x_tier", MUTED),
    ("Four models,\\none fitted per tier", "sections_per_tier", CRITICAL),
]
x = np.arange(len(ARMS))
vals = [arep["variants"][k]["mean_lift"] for _, k, _ in ARMS]

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(x, vals, 0.5, color=[c for _, _, c in ARMS])
ax.set_xticks(x)
ax.set_xticklabels([a for a, _, _ in ARMS])
ax.set_ylim(0, max(vals) * 1.2)
label_bars(ax, x, vals, "{:+.5f}", size=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
ax.set_ylabel("mean R² lift over the baseline", fontsize=10.5, color=INK2)
style(ax,
      "Every step away from one shared fit costs lift",
      f"Same 28 targets, same folds, same seed, same 29 columns. Only the way they "
      f"are fitted changes. Separate fits keep "
      f"{vals[2] / vals[0]:.0%} of what the flat model achieves.")
plt.show()
''')

md("""
Letting coefficients vary by tier costs **9%** of the lift. Fitting four separate
models costs **60%** — it is still positive, but it gives back most of what the
section refetch bought, landing barely above the single `content_length` scalar
Source A used to ship.

**And the aggregate hides something more interesting.** Branching did not fail
everywhere. It failed on balance.
""")

code('''
by_tier = pd.read_csv(OUTPUTS / "source_a_representation_by_tier.csv")
keys = [k for _, k, _ in ARMS]
per_tier = (by_tier[by_tier["variant"].isin(keys)]
            .groupby(["variant", "tier"])["lift"].mean().unstack())
tiers = [t for t in A_TIERS if t in per_tier.columns]
n_by_tier = [asec["by_tier"][t]["n_counties"] for t in tiers]

x = np.arange(len(tiers))
w = 0.26
fig, ax = plt.subplots(figsize=(11.5, 4.8))
for i, (label, key, colour) in enumerate(ARMS):
    vals = [per_tier.loc[key, t] for t in tiers]
    ax.bar(x + (i - 1) * w, vals, w, color=colour,
           label=label.replace("\\n", " "))
    label_bars(ax, x + (i - 1) * w, vals, "{:+.4f}", size=8.5)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\\n{n:,} counties" for t, n in zip(tiers, n_by_tier)])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
ax.set_ylabel("mean R² lift within the tier", fontsize=10.5, color=INK2)
style(ax,
      "Branching helps the tier it was aimed at, and hurts a bigger one",
      "Lift measured inside each tier. The more the fit splits, the better stub "
      "does and the worse mid does — and mid holds 2.7× as many counties. The stub "
      "estimate rests on 21 targets rather than 28.",
      legend=True)
plt.show()
''')

md("""
**So the instinct was right and the arithmetic was not.** Branching helps the
tier it was aimed at — the sparser articles are exactly where more flexible
fitting pays off. But mid moves the opposite way at every step, and mid holds
2.7× as many counties as stub. The weighted result follows the bigger tier, not
the one branching was built for.

**The mechanism is ordinary bias–variance.** Crossing 29 features with 4 tiers
puts **120 columns** against targets whose smallest sample is n ≈ 1,026. The
ridge penalty large enough to control that width also **over-shrinks the
coefficients that were doing the work in the flat model** — so the crossed arm
pays for its extra expressiveness everywhere in order to help one tier. Fitting
four separate models removes even the shared penalty's protection, which is why
that arm goes properly negative rather than merely slightly worse.

The negative result is kept reproducible rather than left as folklore:
`sections_x_tier` is retained as a scored variant in
`analyze_source_a_representation.py`, so anyone can re-run it.
""")

code('''
emb_tier = pd.read_csv(OUTPUTS / "source_a_tiered_embedding_by_tier.csv")
SWEEP = [
    ("uniform_l2", "every tier reads all", SERIES[0]),
    ("drop_stub_l2", "stub reads lead only", SERIES[1]),
    ("drop_thin_l2", "thin reads lead only", SERIES[2]),
    ("drop_mid_l2", "mid reads lead only", SERIES[3]),
    ("drop_rich_l2", "rich reads lead only", SERIES[4]),
]
piv = (emb_tier[emb_tier["representation"].isin([k for k, _, _ in SWEEP])]
       .groupby(["representation", "tier"])["lift"].mean().unstack())
tiers = [t for t in A_TIERS if t in piv.columns]
n_by_tier = [asec["by_tier"][t]["n_counties"] for t in tiers]

# Twenty bars in one cluster is what the arms produce and more than a reader can
# hold. The claim has two halves, so it gets two panels: what withholding a tier's
# own text does to that tier, and what stub's withholding costs everyone else.
# Every cell the panels omit is in the table below.
x = np.arange(len(tiers))
w = 0.34
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

base = [piv.loc["uniform_l2", t] for t in tiers]
own = [piv.loc[f"drop_{t}_l2", t] for t in tiers]
spill = [piv.loc["drop_stub_l2", t] for t in tiers]

for ax, other, other_label in ((ax1, own, "that tier reads lead only"),
                               (ax2, spill, "stub reads lead only")):
    ax.bar(x - w / 2, base, w, color=SERIES[0], label="every tier reads all")
    ax.bar(x + w / 2, other, w, color=SERIES[1], label=other_label)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
    label_bars(ax, x - w / 2, base, "{:+.4f}", size=8.5, pad=0.02)
    label_bars(ax, x + w / 2, other, "{:+.4f}", size=8.5, pad=0.02)

ax1.set_ylabel("mean R² lift within the tier", fontsize=10.5, color=INK2)
style(ax1, "Withholding a tier's own text",
      "Only stub improves. Thin and mid lose what they had.", legend=True)
# ax2's orange series is a DIFFERENT arm from ax1's, so it carries its own
# legend rather than borrowing the one under the left panel.
style(ax2, "What stub's gain costs the others",
      "The same arm, read across every tier.", legend=True)
fig.subplots_adjust(wspace=0.08)
plt.show()

# The same numbers the prose below reads off. Printed from the pivot the chart
# is drawn from rather than transcribed into the markdown beside it, so the two
# cannot drift apart.
display(piv.loc[[k for k, _, _ in SWEEP], tiers]
        .rename(index={k: lbl for k, lbl, _ in SWEEP})
        .rename_axis(index="arm", columns="tier")
        .map(lambda v: f"{v:+.5f}"))
''')

md("""
**The encoder side of the same question, and the sign is not what the plan
expected.** The chart asks of the input text what the arms above asked of the
columns: does it help to let one tier read only its lead while the others read the
whole article? Mean L2 lift within each tier, drop-one against `uniform`.

**Stub is predicted *better* when it stops reading its own article** — its own-tier
lift rises from +0.00021 to **+0.00388** — while thin and mid, the tiers actually
carrying the signal, drop from clearly positive to flat or negative when theirs is
withheld. **And stub's gain does not survive pooling:** `drop_stub` also costs mid
and rich through the one fit all four tiers share, so pooled it loses to `uniform`,
**+0.00343** against **+0.00351** — the same cross-tier interference that beat the
typed columns above.

**So branching loses on the encoder side too.** `uniform` beats three of the four
drop arms; `drop_rich` edges it by +0.00014, too narrow to carry a decision and
picked up again in the branching verdict. Method detail — the per-cell walk through
the table, why the chart plots the row-normalised arm, and how the drop arms are
spliced — is in A2.

What this sweep *did* reopen is the reading scope. Every arm above varies **how
much** of the article each tier reads while holding the section filter fixed.
Varying **which sections** are read turns out to matter considerably more, and
that is the representation section.
""")

md("""
---

## 3. Source E — four volume tiers

Counties split on `num_returns`: **T1** (<2,200), **T2** (2,200–11,700), **T3**
(11,700–100,000), **T4** (≥100,000).

Here the split did something more uncomfortable than guide a feature choice. It
showed that **Source E reports a capital figure for 3,143 of the 3,144 counties,
while the capital itself sits in a small fraction of them.** For most of the
country the pillar is doing exact arithmetic on a rounding error: the feature is
defined everywhere, and the thing it measures effectively is not.
""")

code('''
tiers = list(etier["tiers"].keys())
short = [t.split()[0] for t in tiers]
cty = [etier["tiers"][t]["share_of_counties"] for t in tiers]
inv = [etier["tiers"][t]["share_of_investment_income"] for t in tiers]

x = np.arange(len(tiers))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(x - w / 2, cty, w, color=MUTED, label="share of counties")
ax.bar(x + w / 2, inv, w, color=BLUE, label="share of national investment income")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\\n{etier['tiers'][t]['n_counties']:,} counties"
                    for s, t in zip(short, tiers)])
ax.set_ylim(0, max(cty + inv) * 1.2)
label_bars(ax, x - w / 2, cty, "{:.1%}", size=10)
label_bars(ax, x + w / 2, inv, "{:.2%}", size=10)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
style(ax,
      "One county in ten holds five-sixths of the capital",
      f"T1 and T4 are each about a tenth of all counties. T1 carries "
      f"{inv[0]:.2%} of national investment income; T4 carries {inv[-1]:.1%}.",
      legend=True)
plt.show()
''')

md("""
**Why that matters beyond the chart.** The model counts a tiny county and a huge
one as one observation each, when one of them holds almost none of the capital the
feature is measuring. That is not a reason to reweight it — weighting was tested
and rejected below — but it is a reason to *say so*, which Source E's schema doc
now does.

**What the tiers establish about the ratio.** The ratio every number below
describes is `capital_to_wage_ratio`: a county's net capital gains plus qualified
dividends, over its wages and salaries.

- **Small counties' ratios are the steadier ones year to year, and the volatility
  that exists is in the tail.** A typical T1 county's ratio moves *less* between
  tax years than a typical T4 county's — median |Δratio| ÷ prior-year ratio,
  TY2021→TY2022, of **0.298 in T1 against 0.393 in T4** — while holding its place
  in the national ordering less well: the Spearman correlation between a tier's
  TY2021 and TY2022 ratios is **0.861 in T1 against 0.941 in T4**. The tail runs
  the other way from the median: **17.4%** of T1 counties move by more than half
  their own prior-year ratio in one year, against **9.5%** of T4. Weighting the
  feature by `num_returns` damps none of this — it gives *more* weight to the
  counties with more returns, which are the ones whose ratios move most in the
  typical case, and that is why the weighted arm was rejected.
- **Small counties genuinely differ from each other — the spread is not noise.**
  If it came from averaging few tax returns, it would shrink as counties get
  bigger, and shrink fast: sampling noise falls with the square root of sample
  size, a slope of **−0.5**. Measured across population deciles the slope is
  **+0.026** — flat. There is nothing to smooth away.

**And one caveat that now ships with the pillar.** The strongest link between any
two pillars in this project — how concentrated a county's real-estate employment
is, against how much capital it holds per dollar of wages — turns out to exist
only in large counties: **+0.476 in T4 against −0.058 in T1**. Anyone serving
rural inventory needs that said out loud before leaning on it.

---

## 4. Why neither pillar branches

**Two mechanisms, both measured.** Partitioning 3,144 counties costs more in
pooled evidence than heterogeneity costs in bias: each per-tier model trains on a
fraction of the rows, and the variance that buys exceeds the bias it removes. And
the construction rule leaks into the representation — across every arm that reads
by tier, tier membership alone explains **3.7–7.0%** of the vector's variance
against **0.9–1.2%** under a uniform rule, so the model was partly learning how the
input was built rather than anything about the counties.

**Three tests, three mechanisms, one answer.** The tier section branched the
*model* on tier. The encoder sweep branched the *input* by depth — how much of each
article a tier reads — and uniform wins there too, **+0.00322** against
**+0.00180** tier-conditional; its drop-one arm shows the first mechanism in
miniature, since withholding stub's article helps stub (**+0.00367** inside its own
tier) and still loses pooled once mid and rich are charged for it. A third arm
branched on *content availability* — rich and mid read prose, stub and thin read
their lead, on the theory that a stub's body is all census table and place-name
list — and scores **+0.00192** at the same raw depth. "Do not branch on tier" is a
firmer conclusion than the two tests originally on file supported.

**One caveat.** Withholding the *rich* tier's text edges reading everything,
**+0.00365** against **+0.00351** — a margin that decides nothing, against an arm
that is no longer the best on the board anyway: changing *which sections* are read
beats everything here by a wider margin (the representation section).

**So what shipped instead:**

- **Source A ships one uniform schema** — 29 typed columns, same for every county,
  absence encoded as `False`/`0` rather than null. Sparsity is itself the signal: a
  county whose article says nothing is a county about which little is written.
- **Source E ships with an explicit size-conditionality warning**, and prefers the
  vintage-normalised ratio over the raw one.

The groups changed **what gets shipped and what gets disclosed** rather than
becoming part of the model.

---

## 5. The same question, asked of all six

If a pillar has to earn the right to branch, it has to earn its slot at all.

**Every validation before this was pillar against pillar** — predict one federal
source's features from the other five. That measures whether six agencies agree
with each other; it cannot say whether any of them is *useful*, and the bias runs
backwards, penalising a source precisely for agreeing with the others. No real
label is available: the project is scoped to public data only.

**The substitute is five public outcomes no pillar measures.** They come from the
Census Bureau's **American Community Survey**, 2023 5-year estimates — which
publish for every county, including the small ones a 1-year release skips:
broadband adoption, median household income, median age, median home value, mean
commute. None is built from any pillar's inputs, and everything from here on is
measured against ACS.

**The design answers one objection.** The consuming team joins on DMA with millions
of impressions per market, so it gets a geographic fixed effect essentially for
free — which makes any static geo-keyed feature look redundant. A fixed effect has
exactly one weakness: **no parameter for a place it has never seen.** So the test
holds out **whole states**, against a model that knows only county size.

**Then the same seam, one level down.** Withhold a single pillar's block from a
model already holding county size *and the other five*, and measure the R² lost.
Every pillar takes the same test, restatements are ablated so no pillar is paid for
repeating a neighbour, and the noise floor is measured by shuffling each block
rather than assumed. The pass/fail rule was fixed before the numbers arrived —
full text in appendix A3, *Drop-one method*.
""")

# ==========================================================================
# 5. Pillar worth: the result and the discount
# ==========================================================================
md("""
### First, the discount applied to the result

The raw number is **+0.212**; the number reported is **+0.190**. Two pillar columns
don't so much predict their target as restate it. `wage_per_return_thousands` (IRS)
is average wage income per tax return, very close to a definition of median
household income — removing it drops that outcome's gain from +0.247 to **+0.154**,
so one column was carrying **38%** of the apparent result. `retirement_destination`
(USDA) restates age structure, at a smaller +0.256 → +0.239. Both are dropped from
their own target's run and kept everywhere else. **The headline is the discounted
number.**
""")


md("""
**And within that result, what each pillar is worth.** Read the figure as: how much
predictive power the matrix loses when this pillar is taken out of it. The grey
band is the noise floor — the most any *shuffled* version of a block managed.
""")

code('''
# `HEADLINE` is the five-target basket, bound in the setup cell. Reading
# `ext["drop_one"]` here instead would move the bars onto the 41-target basket
# while the prose around them still described the five -- exactly the
# cross-basket read the evidence-basket table exists to forbid.
floor = HEADLINE["noise_floor"]
# Contribution comes from drop_one, NOT from the noise-floor dict: the latter
# carries F's `_no_ametro` robustness variant (+0.0410, scored against a
# different reference model), and quoting it here would silently understate the
# headline figure the findings report and docs both use (+0.0413).
rows = [{"pillar": p,
         "label": PILLAR_NAME[p],
         "contribution": HEADLINE["drop_one"][f"size_emacro_drop_{p}"]["mean_contribution_ablated"],
         "floor": floor[p]["max_placebo"],
         "above": floor[p]["n_targets_above_floor"],
         "positive": HEADLINE["drop_one"][f"size_emacro_drop_{p}"]["n_positive_ablated"]}
        for p in PILLARS]
worth = pd.DataFrame(rows).sort_values("contribution")

band = float(worth["floor"].max())
hi = float(worth["contribution"].max())
lo = min(-band, float(worth["contribution"].min()))

y = np.arange(len(worth))
fig, ax = plt.subplots(figsize=(11.5, 4.9))
ax.barh(y, worth["contribution"], height=0.62,
        color=[CRITICAL if v <= 0 else BLUE for v in worth["contribution"]])
ax.set_yticks(y)
ax.set_yticklabels(worth["label"])
ax.set_xlim(lo * 1.3, hi * 1.34)

# The noise floor is a property of the measurement, not a series — a shaded band
# rather than a legend entry, so it reads as the bar to clear.
ax.axvspan(-band, band, color=MUTED, alpha=0.5, linewidth=0, zorder=0)
ax.text(band, len(worth) - 0.4, " noise floor", fontsize=10, color=INK2,
        va="center", ha="left")

# Values sit in a fixed right-hand column rather than just past each bar end.
# Tracking the bar end puts Source A's near-zero value on top of its own category
# label and pushes Source E's off the canvas.
for i, v in enumerate(worth["contribution"]):
    ax.text(hi * 1.1, i, f"{v:+.4f}", va="center", ha="left",
            fontsize=11.5, fontweight="bold", color=INK)

ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
ax.set_xlabel("marginal R²", fontsize=10.5, color=INK2)
style(ax,
      "Most of the matrix's value sits in two pillars, and one adds nothing",
      "Mean R² lost when the block is withheld · 5 public ACS targets · "
      "out-of-fold on held-out states · restatements ablated",
      grid_axis="x")
plt.show()
''')

code('''
table = worth.sort_values("contribution", ascending=False).copy()
table["Marginal R²"] = table["contribution"].map(lambda v: f"{v:+.4f}")
table["Positive on"] = table["positive"].map(lambda v: f"{v} of 5")
table["Above noise floor on"] = table["above"].map(lambda v: f"{v} of 5")
display(table[["label", "Marginal R²", "Positive on", "Above noise floor on"]]
        .rename(columns={"label": "Pillar"})
        .set_index("Pillar"))
''')

md("""
**The caveat travels with the headline.** These five targets are **public proxies,
not the consuming team's label**, which is unobtainable under this project's scope.
Everything above is an argument by analogy — the strongest non-circular evidence
available here, and still an analogy. Appendix A1 takes the objection seriously.
""")

md("""
**Source F, in one line:** on the five public ACS targets, measured as marginal R²
over the rest of the matrix, F contributes **+0.0413** — second of the six.
Appendix A5 carries the full story, including the pairwise correlation test it
fails and the seven eighths of its *internal* number that is USDA restating BLS.

### Source A — the uncomfortable finding

Source A ships 29 validated typed columns under the first schema this project
froze. It contributes **−0.0000**.
""")

code('''
a_stats = HEADLINE["drop_one"]["size_emacro_drop_A"]
a_pl = placebo[placebo["pillar"] == "A"].set_index("target")
labels = [TARGET_LABEL[t] for t in TARGET_ORDER]
vals = [a_stats["by_target"][t] for t in TARGET_ORDER]
band = [float(a_pl.loc[t, "placebo_p95"]) if t in a_pl.index else 0.0 for t in TARGET_ORDER]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(11.5, 4.5))
# One colour, not red-for-negative: sign is already carried by position against
# the zero line, and a two-colour split would dramatise numbers whose whole point
# is that they are small.
ax.bar(x, vals, 0.52, color=BLUE, label="Source A contribution")
ax.hlines(band, x - 0.26, x + 0.26, color=INK2, linewidth=2.5,
          label="same block shuffled (95th percentile)")
ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(labels)
span = max(vals + band) - min(vals + band)
ax.set_ylim(min(vals + band) - span * 0.22, max(vals + band) + span * 0.22)
label_bars(ax, x, vals, size=11)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.4f}"))
ax.set_ylabel("marginal R²", fontsize=10.5, color=INK2)
style(ax,
      "Source A is redundant, not broken",
      "A broken block looks wildly negative. This is the signature of one that is "
      "genuinely redundant with the rest of the matrix.",
      legend=True)
plt.show()
''')

md("""
**Not a harness failure.** The same code path produces +0.0582 for E and +0.0413
for F, the placebo distributions behave, and A's per-target numbers are small in
*both* directions rather than wildly negative.

**Consistent with what the in-repo basket measures.** A's typed block carries a
marginal lift of **+0.0010** over a baseline holding every other pillar — a real
effect, at p = 0.010 with power 0.92, and a tiny one. A contribution
indistinguishable from zero against five external outcomes is what that effect
size predicts. A is also the only block negative in **both** arms: −0.0031
internally, −0.0000 externally.

**What it means.** Applied consistently, the operating principle that every pillar
earns its slot on evidence points at Source A. That is uncomfortable and it is the
honest reading.

**The two Source A results here are independent.** The tier section says A should
not branch; this one says A adds nothing marginal. The tier work asked whether one
model or four fits A best and would have returned the same answer if A were the
strongest pillar in the matrix — nothing in it anticipated −0.0000. What is fair to
say is narrower: the pillar this project spent the most instrument time on is the
one that adds least.

**Read this as a fact about the typed columns, not the pillar.** 29 regex columns
are a thin way to represent an article, and a properly scoped, width-matched
encoder does lift A clear of zero on a wider, non-circular basket. It still does
not rescue A: nearly all of that lift is geography, and two latitude/longitude
columns supply 96% of it more cheaply. The comparison, the geography control and
the intervals are in the representation section.

### The open question this leaves

> **Does Source A ship?** Three arguments defend keeping it: one is about cost
> rather than worth, one is neutralised by the geography control, and one is
> untested.
>
> *A is nearly free* is about cost, not worth — it justifies leaving the code in
> the repo, not calling A a pillar, and the limits section prices "nearly".
> *A better representation would rescue it* gets halfway, then loses the lift to
> geography (above). *Redundancy is insurance for a county missing another pillar*
> is plausible and untested — no coverage-failure scenario here has been scored
> with and without A.
>
> **One argument is still live, and this project cannot settle it.** A encodes
> named industries, universities, ports and protected land. Whether that is worth
> anything depends on whether the downstream target is closer to "who lives here"
> — where A adds nothing — or "what happens here economically". The five ACS
> proxies are squarely the former. Only the commissioning side has that answer.
>
> **The decision, its owner and its default are one row in the decision list** at
> the end of the limits section, so the recommendation lives in one place.
""")

# ==========================================================================
# 6. The representation question, run properly
# ==========================================================================
md("""
---

## 6. How Source A should be represented

Four moves: what ships today, what a correctly run encoder comparison says about
it, what survives a geography control, and what that leaves to do. Two earlier
answers, both from comparisons with uncontrolled confounds, are in appendix A9.

### 1. What ships today, and why

Source A ships 29 typed columns extracted with a fixed lexicon (the table in the
Source A tier section lists them). They exist because of what the corpus is
actually made of. Every body section, by share of characters:
""")

code("""
# Economy last, because it is the row the section is about.
order = ["census", "lists", "narrative", "other", "economy"]
comp = pd.DataFrame({
    "what the encoder reads": [acomp["labels"][k] for k in order],
    "share of characters": [f"{acomp['share_of_characters'][k]:.1%}" for k in order],
})
print(f"{acomp['n_sections']:,} body sections across "
      f"{acomp['n_counties']:,} counties")
display(comp.set_index("what the encoder reads"))
""")

md("""
**Economy-titled prose is 1.5% of the corpus; census tables and place-name lists
are ~46%.** That is what the typed columns answer: a lexicon extracts the 1.5% and
ignores the rest, with no model download, no inference, and columns you can read by
name. Mean-pooling an encoder over the same text averages that 1.5% into the
boilerplate instead.

"Just read the economy sections" is not the fix either — they exist for only 660 of
3,144 counties, so it scores **+0.0017**, indistinguishable from reading nothing but
the lead (**+0.0017**), because 79% of counties fall back to their lead anyway.

### 2. The encoder comparison, run correctly

**Confound one: the encoder was reading mostly boilerplate** — the composition
above. Every earlier arm read the whole article, so the 1.5% that carries economic
content was averaged into census tables and lists of place names.

**Confound two: nobody controlled for width.** Every embedding arm carried 384 or
1,024 columns against the typed block's 29, so part of every measured penalty was
width rather than content. Reducing the *identical* vectors to 29 dimensions —
fitted inside each fold, never on the full corpus — moves the same arm from
**−0.021 to +0.011**. That single control accounts for most of the **−0.044** an
earlier account of Source A reported (appendix A9).

**A third thing, not a confound but a leak.** County census sections state the ACS
targets verbatim — *"The median age was 38.9 years"* — and 2,589 counties carry one
quoting median income. The encoder reads that; the typed block, which extracts
lexicon counts and no numbers, cannot. Dropping census sections is a leakage
control, not a tuning choice; 8 of 42 targets are flagged as restated in article
text and drop out of the screened subset.

**The decision rule was fixed before scoring.** The text scope was chosen on the
in-repo 28-target basket, the decision run on 41 external ACS targets sharing none
of them. Under that rule the encoder wins outright, on both the full basket and the
leakage-screened subset.

### 3. The geography control

`GroupKFold` on state stops a county's own row leaking into its training fold. It
does nothing about regional vocabulary: a held-out New England county's article
shares dialect, place names and climate description with training-fold New England
counties, so an encoder can place it regionally without ever seeing its state. The
selected arm's largest per-target gains are what that predicts — electric heating
+0.125, fuel-oil heating +0.101, gas heating +0.061, foreign-born share +0.041,
median year built +0.038. So the arms were re-scored against a baseline that
already holds latitude and longitude.
""")

code('''
R = mrep["by_representation"]
ARMS = [("latlong_only", "lat/lon only (2 cols)"),
        ("minilm_prose_plus_history_ccr_pca29", "selected encoder (29 dims)"),
        ("minilm_uniform_pca29", "uniform scope (29 dims)"),
        ("typed", "typed columns (29 cols)"),
        ("minilm_uniform", "uniform scope (384 dims)")]
plain = [R[k]["decision_basket_mean_contribution"] for k, _ in ARMS]
geo = [R[k]["decision_basket_mean_contribution_geo"] for k, _ in ARMS]
labels = [lbl for _, lbl in ARMS]

y = np.arange(len(ARMS))
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.barh(y - 0.2, plain, 0.38, color=BLUE, label="vs size + other pillars")
ax.barh(y + 0.2, geo, 0.38, color=CRITICAL, label="also holding lat/lon")
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.invert_yaxis()
for yy, v in zip(y - 0.2, plain):
    ax.text(v + (0.0006 if v >= 0 else -0.0006), yy, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10)
for yy, v in zip(y + 0.2, geo):
    ax.text(v + (0.0006 if v >= 0 else -0.0006), yy, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10)
ax.set_xlim(min(geo) * 1.35, max(plain) * 1.30)
ax.set_xlabel("mean marginal contribution, 41 external targets", fontsize=10.5, color=INK2)
style(ax,
      "Two coordinate columns do 96% of what the encoder does",
      "Blue: contribution over size plus the other five pillars. Red: the same "
      "arm once latitude and longitude are already in the baseline.",
      grid_axis="x", legend=True)
plt.show()
''')

md("""
**The win is geography, not economic content.** Net of two float columns the
selected encoder's contribution falls from **+0.0164 to +0.0006**, and the count of
targets it helps falls from 37 of 41 to 21 — a coin flip. `latlong_only` on its own
scores **+0.0158**, which is 96% of the encoder's own headline. The typed columns
go *negative* net of geography, at −0.0127, so their information was partly
geographic too; they simply carried less of it.

**And what is left of the encoder cannot be told from zero.** Every figure above is
a mean over 41 targets quoted to four decimals, a precision the basket does not
have. Each now carries a bootstrap interval — resampled over targets, and again
over whole ACS tables, so the clustering is priced in rather than described.
Nothing was re-fitted; the per-target contributions behind the chart are what get
resampled.
""")

code('''
def band(block, scheme, statistic):
    """One interval, formatted point [low, high]."""
    interval = block[scheme][statistic]
    return f"{interval['point']:+.4f}   [{interval['low']:+.4f}, {interval['high']:+.4f}]"

sel_boot = R["minilm_prose_plus_history_ccr_pca29"]["bootstrap"]
lat_boot = R["latlong_only"]["bootstrap"]
CI_ROWS = [
    ("selected encoder", sel_boot, "contribution"),
    ("lat/lon only", lat_boot, "contribution"),
    ("selected encoder, net of lat/lon", sel_boot, "contribution_geo"),
]
ci = pd.DataFrame(
    {"resampling targets": [band(b, "naive", s) for _, b, s in CI_ROWS],
     "resampling whole ACS tables": [band(b, "table_clustered", s) for _, b, s in CI_ROWS]},
    index=[label for label, _, _ in CI_ROWS])
ci.index.name = (f"95% interval, {sel_boot['n_replicates']:,} replicates over "
                 f"{sel_boot['n_targets']} targets in {sel_boot['n_tables']} ACS tables")
display(ci)
''')

md("""
The encoder's plain contribution clears zero on both resamples. Its contribution
**net of geography does not**: +0.0006, with a table-clustered interval of
**[−0.0070, +0.0082]**. So the sentence to carry out of this section is not that
the encoder's geography-free contribution is small — it is that this basket
cannot distinguish it from nothing.

The third row is also the encoder-minus-`latlong_only` difference, not merely
close to it. `latlong_only`'s full model *is* the geography-adjusted baseline
every other arm is scored against, so "net of lat/lon" and "minus lat/lon" are
the same subtraction. Reporting a difference rather than a ratio is deliberate:
an interval on "96%" divides by a mean this small and is unreadable, so 96% stays
a point estimate in prose and the interval is quoted on the gap.

This does not overturn the pre-registered verdict — a paired comparison cancels any
baseline both arms share, so the encoder still beats the typed block by the rule as
written. What it overturns is the *interpretation*. `E_macro`'s stated job is
distinguishing physically similar places sitting in different **economic** climates.
What this measured is a pillar that mostly encodes **where the county is**.

### 4. What that means for shipping

Not "ship the encoder." A competitor gets 96% of
the measured gain from two columns already sitting in
`data/county_centroids.parquet`, with no model download, no inference cost, and no
opaque dimensions. The question worth asking next is not which encoding of Source A
is better — it is whether `E_macro` needs Source A at all once `E_local` and
`E_census` supply location, and that is a drop-one test against the sibling tiers
rather than another representation sweep.

### Known weaknesses in the decision basket

**The basket is clustered.** 5 of 42 targets are heating-fuel shares from one ACS
table and 35 target pairs correlate above 0.7, so the effective sample is nearer
28 than 42. The table-clustered column above is what that costs, measured rather
than asserted: drawing 28 whole tables instead of 41 independent targets widens
the encoder's plain contribution from **[+0.0093, +0.0251]** to
**[+0.0071, +0.0297]**.

**15 of the 42 targets were never leakage-screened.** They are reported as
unscreened rather than counted clean, which is why the pre-registered rule is
checked on the screened subset as well as the full basket.

**One target was excluded outright.** `no_fuel_used_share`: both its models score
worse than predicting the mean, so its apparent contribution was the gap between
two useless fits rather than a gain.

**The pre-registered typed transform backfired.** `typed_transformed` scored
**+0.0017** against the raw block's **+0.0031**. It is reported rather than
quietly swapped for the better arm, because the rule named it in advance.
""")

# ==========================================================================
# 7. Three honest limits
# ==========================================================================
md("""
---

## 7. Limits, and where this leaves the project

**1. The fixed-effect objection is unanswered.** Holding out states shows `E_macro`
beats a size baseline on places it has never seen; it does not show it beats a DMA
dummy on places the consumer sees constantly. What the grain question costs in each
direction:
""")

code('''
arms = [("County grain, all rows", gst["mean_lift_county_full"], BLUE),
        ("County grain, subsampled to market row count", gst["mean_lift_county_subsample"], MUTED),
        (f"Aggregated to {gst['n_markets']} markets", gst["mean_lift_market_aggregate"], AQUA)]
x = np.arange(len(arms))
vals = [a[1] for a in arms]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x, vals, 0.52, color=[a[2] for a in arms])
ax.set_xticks(x)
ax.set_xticklabels([textwrap.fill(a[0], 26) for a in arms])
ax.set_ylim(0, max(vals) * 1.18)
label_bars(ax, x, vals, "{:+.3f}", size=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}"))
ax.set_ylabel("mean lift over size baseline", fontsize=10.5, color=INK2)
style(ax,
      "Coarsening the join: two effects that roughly cancel",
      f"Losing rows costs {gst['row_count_effect']:+.3f}; aggregating itself gains "
      f"{gst['aggregation_effect']:+.3f}. County grain is this project's "
      "recommendation, not an established win.")
plt.show()
''')

md("""
**2. On the smallest counties the model cannot win, and that is a data fact rather
than a model failure.** ACS publishes a margin of error with every estimate, so
ingesting those alongside the values splits each outcome's variance into signal and
sampling noise. In the smallest population decile **30% of the variance is sampling
noise** — error no model can explain — against under 1% in the largest. The
size-only baseline scores a *negative* R² there.
""")

code('''
d = (decile.groupby("population_decile")
     .agg(noise_share=("noise_share", "mean"),
          median_population=("median_population", "median"))
     .reset_index())
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(d["population_decile"], d["noise_share"], 0.62,
       color=[CRITICAL if v > 0.15 else BLUE for v in d["noise_share"]])
ax.set_xticks(d["population_decile"])
ax.set_ylim(0, float(d["noise_share"].max()) * 1.2)
label_bars(ax, d["population_decile"], d["noise_share"], "{:.0%}", size=11)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("population decile", fontsize=10.5, color=INK2)
ax.set_ylabel("share of variance that is noise", fontsize=10.5, color=INK2)
style(ax,
      "On the smallest counties, a third of the outcome is unexplainable by anyone",
      "Share of outcome variance that is ACS sampling error, by county population "
      "decile (1 = smallest). Averaged over the five targets.")
plt.show()
''')

md("""
**3. Three more, in a line each.** No **downstream label**, and none possible under
this scope. Everything here is **cross-sectional** — temporal transfer, the one
thing a fixed effect genuinely fails at, is untested. And the **sibling tiers do
not line up**: `E_local` is at H3 res-8, `E_census` does not exist, and nobody owns
the reconciliation.

### Where this leaves the project
""")

code('''
readiness = pd.DataFrame([
    ("Six sources ingested, validated, schema frozen", "Done",
     "3,143–3,144 counties; six schema docs; vintage per pillar"),
    ("Evidence against a target outside the six pillars", "Done",
     f"{HEADLINE['mean_lift_over_size_ablated']:+.3f} mean R² over a size baseline, "
     f"{HEADLINE['targets_with_positive_lift']} of {HEADLINE['n_targets']} positive"),
    ("Each pillar's worth measured, not asserted", "Done",
     "drop-one, restatements ablated, noise floor measured"),
    ("Evidence against the consumer's real target", "Not possible in scope",
     "no downstream label; public proxies only"),
    ("Answer to the DMA fixed-effect objection", "Open",
     "needs the join grain settled"),
    ("Temporal transfer tested", "Not started",
     "cross-sectional only; the strongest untested argument"),
    ("Fusion / serving format", "Deferred by design",
     "packaging, not evidence; grain must settle first"),
], columns=["What a go/no-go needs", "Status", "Where it stands"])
display(readiness.set_index("What a go/no-go needs"))
''')

md("""
**What "it" actually is.** The readiness table says the matrix is done without
ever saying how big it is. Counted from `build_matrix()` rather than typed:
""")

code('''
import sys
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from pillar_matrix import build_matrix, SIZE_FEATURES

matrix, blocks = build_matrix()
shape = pd.DataFrame(
    [(PILLAR_NAME[p], len(blocks[p])) for p in PILLARS],
    columns=["Block", "Feature columns"])
shape.loc[len(shape)] = ("TOTAL, pillar features", int(shape["Feature columns"].sum()))
shape.loc[len(shape)] = ("Size controls, held out of every block",
                         len(SIZE_FEATURES))
print(f"{len(matrix):,} counties x {sum(len(b) for b in blocks.values()):,} pillar "
      f"feature columns, plus {len(SIZE_FEATURES)} size controls "
      f"and {matrix.shape[1] - sum(len(b) for b in blocks.values()):,} identifier "
      f"and size columns carried alongside.")
display(shape.set_index("Block"))
''')

md("""
**And what maintaining it costs**, since every row above is evidence rather than
cost. Cadence and reference period come from `outputs/pillar_vintages.csv`; each
refresh figure is stated with its basis, because one measured number beside five
guesses would otherwise read as six measurements.
""")

code('''
# Refresh cost per pillar. `basis` is not decoration: only Source B has been
# timed end to end. Request counts marked "counted" are structural -- one
# request per county, per state or per year, read off the ingest script -- and
# will not drift unless the ingest changes. Everything marked "estimated" is a
# guess and is labelled as one.
COST = {
    "A": ("3,144 article requests (one per county) + 1 auth call",
          "~25 min", "Wikimedia Enterprise credentials",
          "requests counted; wall-clock estimated"),
    "B": ("1 bulk QCEW zip, stream-filtered to county rows",
          "4m36s", "none",
          "measured end to end, 2026-08-05 run log"),
    "C": ("6,288 FRED requests (2 series x 3,144 counties)",
          ">=63 min", "FRED_API_KEY",
          "requests counted; floor derived from the script's own 100/min limiter"),
    "D": ("51 state zip downloads via curl",
          "~20 min", "none; host is flaky, retries are built in",
          "downloads counted; wall-clock estimated"),
    "E": ("5 CSV downloads, one per tax year",
          "~5 min", "none",
          "downloads counted; wall-clock estimated"),
    "F": ("1 static CSV download",
          "<1 min", "none",
          "download counted; wall-clock estimated"),
}
v = vintages.set_index("pillar")
cost = pd.DataFrame(
    [(PILLAR_NAME[p], v.loc[p, "cadence"], COST[p][0], COST[p][1],
      COST[p][2], COST[p][3]) for p in PILLARS],
    columns=["Pillar", "How often it is paid", "What a refresh fetches",
             "Wall-clock", "Manual steps", "Basis"])
display(cost.set_index("Pillar"))
''')

md("""
**The row that matters for the open decision is A's.** "Source A is nearly free"
has been asserted in this project without a number attached; the number is 3,144
API requests against credentialled Wikimedia Enterprise access, on a source with
no reference period, refetched whenever someone decides the corpus has moved.
That is cheap, and it is not free, and it is the whole of the cost argument for
keeping A in the repo.

### The decisions, and what happens if nobody makes them

Every open item below has an owner and a **default** — what this project does if
no answer comes back. A default is not a preference; it is what removes the
decision from the critical path.
""")

code('''
decisions = pd.DataFrame([
    ("Join grain: county or market?",
     "Consuming team",
     "Ship at county grain and document the aggregation path. Coarsening costs "
     f"{gst['row_count_effect']:+.3f} and gains {gst['aggregation_effect']:+.3f}; "
     "they roughly cancel, so county is the reversible choice.",
     "2026-09-18"),
    ("Does Source A ship?",
     "Consuming team (target is demographic or economic?)",
     "Ship five pillars plus two centroid columns from "
     "data/county_centroids.parquet, which supply 96% of A's measured gain. "
     "Source A stays in the repo, unshipped, and the go/no-go deck says why.",
     "2026-09-18"),
    ("Benchmark against a geographic fixed effect",
     "This project",
     "Run it at county grain against the public proxies and report the result "
     "either way. It is the test that decides whether E_macro earns a slot in a "
     "production model.",
     "Starts once grain settles"),
    ("Temporal transfer",
     "This project",
     "Remains untested and stays on the limits list. It is the one argument a "
     "fixed effect cannot answer, so it is named rather than quietly dropped.",
     "Not started"),
    ("Fusion / serving format",
     "This project",
     "Stays deferred until grain settles. Days of work, not weeks: "
     "build_matrix() already joins all six pillars.",
     "After grain"),
], columns=["Open decision", "Owner", "Default if no answer comes back", "By"])
display(decisions.set_index("Open decision"))
''')

md("""
**Not blocked by any of it:** ingestion, validation, schema freeze and the external
benchmark are complete and stand on their own.
""")

# ==========================================================================
# Appendix
# ==========================================================================
md("""
---

# Appendix

## A1 — Do public proxies mean anything?

The strongest objection available against the pillar-worth section, and it
deserves a straight answer rather than a footnote.

**The objection.** `E_macro` is scored against ACS broadband adoption, median
household income, median age, median home value and mean commute. The consuming
team predicts none of those. A feature layer that explains median age tells you
nothing about whether it explains revenue per ad request.

**Conceded immediately.** The objection is correct on its own terms. No result here
is a direct test of usefulness to the consuming team, and none is presented as one.
The label that would make such a test possible is unobtainable: the project was
scoped to public and open-source data only, deliberately and from the start.

**Why the proxies are still worth what they cost.** The alternative was not a better
test — it was the pillar-versus-pillar sweep, which is *circular*. It can establish
that Source B and Source E see the same economy and cannot establish that either
predicts anything at all. Moving to an external target replaces a circular
measurement with a non-circular one. That is a real gain in evidential status even
though the target is a proxy.

**What the proxies license.** That the six pillars carry information counties'
*size* does not already carry — the baseline is population and density and the lift
is measured on held-out states, so it is not memorisation. That the pillars are not
interchangeable, since the drop-one design holds the other five constant. And that
the ordering among them is not arbitrary — E and F clear the noise floor on 5 of 5
targets by an order of magnitude, A does not.

**What they do not license.** Any claim about **magnitude** against an ad-tech
target; +0.190 on ACS outcomes forecasts nothing. Any claim that the **ordering
transfers** — Source A could rank higher against an economically-flavoured target,
which is precisely the argument in that section for not cutting it unilaterally. And any
answer to the **fixed-effect objection**, which is a different question.

**What would settle it.** One pass of the same drop-one design against a real
downstream target, at the grain the consuming team actually joins on. That needs
either a label or a collaborator inside that team, and is the single most valuable
thing that could be added to this project.

## A2 — Tier method detail

**Source A tier edges**, on lead-section character count: stub <100, thin 100–283,
mid 284–461, rich ≥462 — quartile-ish cuts on the observed distribution, not round
numbers chosen in advance.

**Source E tier edges**, on `num_returns`: T1 <2,200, T2 2,200–11,700, T3
11,700–100,000, T4 ≥100,000. Five tax years, TY2018–TY2022.

**What each branching arm actually consumed.** The flat arm fits one coefficient per
feature across all counties. The crossed arm fits 29 features × 4 tiers = 120
columns against targets whose smallest sample is n ≈ 1,026. The separate arm fits
four independent models, each on roughly a quarter of the rows, with no shared
penalty across them.

**Why the crossed arm loses even though it is strictly more expressive.** Ordinary
bias–variance: the ridge penalty large enough to control 120 columns also
over-shrinks the coefficients that matter. Expressiveness the data cannot pay for
is a cost, not a capability.

**A p-value worth stating rather than burying.** On a paired rank test the crossed
variant reads p = 0.040 against the flat block's p = 0.115. That does not reverse
the section: the flat block still carries the higher mean lift (+0.00307 against
+0.00279), which is the quantity the shipping decision uses. It does mean the flat
block's advantage is a point estimate rather than a demonstrated gap.

**The other Source A arms tested in the same round**, all of which lost to the
shipped design:
""")

code('''
arms = [
    ("Typed columns (shipped)", scope["scopes"]["economy"]["mean_lift"], BLUE),
    ("All sections", scope["scopes"]["all_sections"]["mean_lift"], MUTED),
    ("All except narrative", scope["scopes"]["no_narrative"]["mean_lift"], MUTED),
    ("384-d embedding, lead only", embed["representations"]["lead_only"]["mean_lift"], AQUA),
    ("384-d embedding, uniform", embed["representations"]["uniform"]["mean_lift"], AQUA),
    ("384-d, tier-conditional", embed["representations"]["tier_conditional"]["mean_lift"], AQUA),
    ("384-d, tier inverted", embed["representations"]["tier_conditional_inverse"]["mean_lift"], AQUA),
    # The section-6 scopes. These vary WHICH sections are read rather than how
    # much, which is why they clear every arm above them.
    ("384-d, economy sections only", embed["representations"]["economy_all_tiers"]["mean_lift"], SERIES[3]),
    ("384-d, prose by tier", embed["representations"]["prose_by_tier"]["mean_lift"], SERIES[3]),
    ("384-d, prose only", embed["representations"]["prose_only_ccr"]["mean_lift"], SERIES[3]),
    ("384-d, prose + history (selected)", embed["representations"]["prose_plus_history_ccr"]["mean_lift"], SERIES[1]),
]
arms = sorted(arms, key=lambda a: a[1])
y = np.arange(len(arms))
vals = [a[1] for a in arms]
fig, ax = plt.subplots(figsize=(11.5, 4.3))
ax.barh(y, vals, height=0.62, color=[a[2] for a in arms])
ax.set_yticks(y)
ax.set_yticklabels([a[0] for a in arms])
ax.set_xlim(0, max(vals) * 1.2)
label_bars(ax, y, vals, "{:+.5f}", size=11, horizontal=True)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.4f}"))
ax.set_xlabel("mean lift", fontsize=10.5, color=INK2)
style(ax,
      "Every Source A arm tested, against the shipped design",
      "Mean lift over the crowded baseline. Reading all sections scores higher, but "
      "67% of the hits it adds sit in historical framing.",
      grid_axis="x")
plt.show()
''')

md("""
Reading **all** sections scores higher than the shipped economy-only rule
(+0.00403 against +0.00307) and was still not shipped: 67% of the hits it adds sit
in historical framing, which makes it a defunct-industry detector wearing a
current-economy label. That is a precision judgement, not a scoring one, and it is
the one place in this project where a higher number was deliberately declined.
**The per-tier breakdown of the embedding arms is in the Source A tier
section**, including the
drop-one sweep that locates which tier the tier-conditional rule actually costs.

**The drop-one sweep, cell by cell.** Withholding stub's body text raises stub's
own lift from +0.00021 to +0.00388, a diagonal gain of **+0.00367** — its lead
names an industry in 0.7% of counties, so there is next to nothing past it to
lose. Dropping thin's text costs its own tier **−0.00190** (+0.00196 → +0.00006)
and mid's costs **−0.00323** (+0.00288 → −0.00035). rich barely moves,
**−0.00028** (+0.00708 → +0.00680): the tier with the most text to read is close
to redundant over its own lead. `drop_stub`'s off-diagonal cost falls on mid
(+0.00288 → +0.00235) and rich (+0.00708 → +0.00671). **The off-diagonal cost is
not always smaller than the diagonal gain:** `drop_thin` gives its own tier
+0.00006 and costs the stub tier **−0.00247**, a larger move in the opposite
direction.

**Why the tier chart plots the row-normalised arm.** `tier_variance_share` — how
much of the embedding's variance tracks which tier a county is in rather than
anything else — sits at 0.0121 for `uniform` and 0.0481–0.0698 for the drop arms.
Splicing a lead-only row into an otherwise uniform arm manufactures a norm
discontinuity at the tier boundary: `lead_only` rows carry a norm near 1.0 against
`uniform`'s 0.63–0.71, so an unnormalised drop arm partly encodes tier membership
through vector length rather than content. `drop_stub` reads +0.00256 raw against
+0.00343 row-normalised — a 0.00087 artifact.

**How the drop arms are built.** Each `drop_*` arm splices that tier's `lead_only`
rows into `uniform`'s vectors rather than re-encoding the corpus under a different
reading rule, so the three untouched tiers are exactly `uniform`'s own vectors.
Re-encoding would perturb each by ~1e-7 of padding noise — drift injected into
precisely the tiers the comparison holds fixed.

**The MiniLM arms above are the uncapped run.** An earlier chunk cap truncated
their input and made them look worse than they are; that correction is recorded
with the other superseded Source A numbers in A9.

## A3 — Drop-one method

**The pre-registered rule**, verbatim from the implementation plan, written before
either script was run and not renegotiated afterwards:

> F ships as a pillar if its marginal contribution — R²(size + all pillars) −
> R²(size + all pillars except F), pooled out-of-fold over the five external ACS
> targets, with restatement columns ablated — is positive on a majority of targets
> and above the shuffled-feature noise floor. Otherwise `E_macro` ships five
> pillars and the go/no-go deck says so plainly.

**Design.** For each pillar, fit two models: one holding county size plus all six
pillars, one holding county size plus five. The difference in out-of-fold R² is that
pillar's marginal contribution. Repeated for all six and for the B+E pair.

**Estimator.** Ridge regression on an imputed design matrix. Missing values are
imputed rather than dropped, because the null patterns are themselves informative
(BLS suppresses ~35% of the LQ matrix) and listwise deletion would silently change
the county population under test.

**Folds.** `GroupKFold` on `state_fips` — spatially blocked, so a model never
predicts a county in a state it trained on. Stricter than random k-fold, and the
right strictness: county features are spatially autocorrelated, and random folds
would let a neighbouring county leak the answer.

**Restatement ablation.** Where a column in one pillar restates a column in another
(Source A's `has_metro_attachment` against Source F's `metro_2023`; Source F's
industry flags against Source B's location quotients), it is removed from both the
full and the reduced model. A pillar that only restated a neighbour then scores zero
by construction, which is the intended behaviour.

**Noise floor.** Each block is shuffled and re-scored — 49 reps in the internal arm,
20 per pillar × target in the external arm. The largest contribution any shuffled
block produced anywhere is +0.0031, and that is the bar drawn on the pillar-worth
figure.

**Two arms.** The internal arm scores against 29 in-matrix targets and measures
coherence. The external arm scores against the five ACS targets and is the one the
verdicts rest on. Where they disagree the external arm wins — being unpredictable
from the other five pillars is also exactly what an independent information source
looks like.

## A4 — Limitations, carried over unchanged

- **The targets are public proxies, not the consumer's label**, which is
  structurally unobtainable. Every conclusion is by analogy.
- **20 placebo reps per pillar × target, 49 in the internal arm.** Enough to place a
  floor near zero against contributions an order of magnitude larger. **Not enough
  to resolve a borderline contribution — B's and C's ordering (+0.0067, +0.0054)
  should not be quoted as settled.**
- **Contribution is not importance under a different model class.** Everything here
  is ridge on an imputed design. A gradient-boosted consumer might distribute credit
  differently; only the internal arm carries a GBM cross-check.
- **Cross-sectional and single-period.** Temporal transfer is untested.
- **This does not answer the fixed-effect objection.** It reallocates credit among
  pillars, given that the matrix as a whole beats a size baseline on held-out states.

## A5 — Source F, in full

F is the pillar a previous status doc flagged as falling short, and the reason was
real. Its one strong relationship in the whole 15-pair sweep was against Source D
freight tonnage, **r = 0.495 raw — the largest raw effect anywhere in that sweep —
collapsing to r = −0.057 once county size is controlled.** The apparent link was
population riding along in both variables.

The resolution then on file was to keep F and reclassify it as a "structural
anchor," justified by what county typology definitionally *is* rather than by
measured performance. That was a rationalisation, and it was withdrawn.

**What replaced it.** The status doc named the fairer test itself — does F explain
residual variance once B/C/D/E are already in the model — and that test was run. F
contributes **+0.0413**, second of the six, positive on 5 of 5 targets and above the
noise floor on 5 of 5, where the largest contribution any shuffled block produced
anywhere was +0.0031.

**Both facts travel together.** F still fails the pairwise hub test. It passed the
residual-variance test that was pre-registered for it. The first instrument was the
wrong one for a categorical structural variable, and that was said before the
numbers existed, not after.

**The caution that belongs beside the headline:**
""")

code('''
fb = blk["by_block"]["F"]
# Headline external figure, not the `_no_ametro` robustness variant — see the
# comment on the pillar-worth figure.
f_ext = HEADLINE["drop_one"]["size_emacro_drop_F"]["mean_contribution_ablated"]
# NOTE: newlines in these labels are escaped because this cell's source is a
# non-raw triple-quoted string in the builder — an unescaped \\n would be
# consumed at build time and split the string literal.
bars = [
    ("Internal, raw\\n29 in-matrix targets", fb["mean_lift"], MUTED),
    ("Internal, restatements ablated\\nthe honest internal number",
     fb["mean_lift_ablated"], AQUA),
    ("External\\n5 public ACS targets", f_ext, BLUE),
]
x = np.arange(len(bars))
vals = [b[1] for b in bars]
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.bar(x, vals, 0.52, color=[b[2] for b in bars])
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars])
ax.set_ylim(0, max(vals) * 1.18)
label_bars(ax, x, vals, size=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
ax.set_ylabel("mean R² contribution", fontsize=10.5, color=INK2)
style(ax,
      "Source F: the internal number is mostly USDA restating BLS",
      "Roughly seven eighths of F's apparent internal contribution disappears once "
      "columns that restate Source B are removed. The external number does not move.")
plt.show()
''')

md("""
**Seven eighths of F's apparent internal contribution is USDA restating industry
composition BLS already measures.** That redundancy is real inside the six-pillar
system and does not bind against outcomes outside it — the same ablation moves F's
external figure by 0.0003, which is why the external arm is the one the verdict
rests on.

## A6 — Pillar vintages
""")

code('''
v = vintages.copy()
v["Pillar"] = v["pillar"].map(PILLAR_NAME)
v = v.rename(columns={"as_of_date": "As of", "reference_period": "Reference period",
                      "cadence": "Update cadence"})
display(v[["Pillar", "As of", "Reference period", "Update cadence"]].set_index("Pillar"))
''')

md("""
## A7 — For the consuming team, when it comes to that

**The warning that matters more than which columns ship.** An impression-level
training table joined to `E_macro` carries only 3,144 distinct feature values, so
the effective sample size is the county count — not the row count. Random k-fold
will make this feature layer look good in evaluation and do nothing in production.
**Cluster standard errors by `fips_code`; use grouped, spatially blocked folds.**

Two shorter notes. **Nulls are explicit**: BLS suppresses ~35% of the Source B LQ
matrix, those cells stay null with a `disclosure_*` flag, and IRS publishes no
suppression flag at all — a model must be able to tell "missing" from "zero."
**Size columns are held out deliberately** in `SIZE_COLUMNS`, so a pillar can be
re-derived at a coarser geography; they are not features.

## A8 — Artifact index

**Every figure and every table above is computed from a committed artifact.**
Numbers quoted inline in prose are transcribed from those same artifacts by hand,
and are the one thing in this notebook that does not move when the artifact moves.

Rows are keyed by section *name*. The section order has changed twice, and
numbered references went stale both times.

| Where | Reads | Produced by |
|---|---|---|
| The assignment · evidence baskets | the `n_targets` field of the four stats artifacts below | — |
| Source A tiers · section yield | `source_a_section_stats.json` | `scripts/extract_source_a_section_features.py` |
| Source A tiers · branching arms | `source_a_representation_stats.json`, `outputs/source_a_representation_by_tier.csv` | `scripts/analyze_source_a_representation.py` |
| Source A tiers · encoder sweep | `outputs/source_a_tiered_embedding_by_tier.csv` | `scripts/analyze_source_a_tiered_embedding.py` |
| Source E tiers | `source_e_tier_stats.json` | `scripts/analyze_source_e_tiers.py` |
| Pillar worth · the discount | `external_target_stats.json` (`by_target`) | `scripts/analyze_external_target.py` |
| Pillar worth · the figure | `external_target_stats.json` (`drop_one`, `drop_one_noise_floor`) | `scripts/analyze_external_target.py` |
| Pillar worth · Source A | `external_target_drop_one_placebo.csv` | `scripts/analyze_external_target.py` |
| Representation · corpus composition | `source_a_section_composition_stats.json` | `scripts/analyze_source_a_section_composition.py` |
| Representation · the decision | `source_a_representation_marginal_stats.json`, `outputs/source_a_representation_marginal.csv` | `scripts/analyze_source_a_representation_marginal.py` |
| Representation · leakage screen | `scripts/source_a_text_leakage.py` (no artifact; screen is computed inline) | `scripts/source_a_text_leakage.py` |
| Representation · pre-registered rule | `docs/source_a_representation_decision.md` | committed before scoring |
| Limits · grain | `grain_effect_stats.json` | `scripts/analyze_grain_effect.py` |
| Limits · sampling noise | `outputs/external_target_by_decile.csv` | `scripts/analyze_external_target.py` |
| A2 · arms ruled out | `source_a_section_scope_stats.json`, `source_a_tiered_embedding_stats.json` | `scripts/analyze_source_a_section_scope.py`, `scripts/analyze_source_a_tiered_embedding.py` |
| A9 · superseded accounts | `source-a-findings.md` §20, §21.2; `source_a_representation_stats.json` | the scripts that produced each, named in the entry |
| A5 · Source F | `pillar_block_marginal_stats.json` (`by_block`) | `scripts/analyze_pillar_block_marginal.py` |
| A6 · vintages | `outputs/pillar_vintages.csv` | `scripts/pillar_vintage.py` |

Long-form evidence:
`analysis-output/cross-source/pillar-marginal-findings.md`,
`analysis-output/cross-source/external-target-findings.md`,
`analysis-output/E_macro_key_findings.ipynb`,
`docs/pillar_status.md`, `docs/PROJECT_GOAL.md`.

## A9 — Superseded Source A accounts

Three accounts of Source A that earlier versions of this notebook stated, and
stated confidently. Each was measured honestly and each is now wrong. They are
kept because a conclusion that moved is more informative than one that was always
right, and because a reader carrying a number from an earlier version deserves to
find it here rather than meet its silent absence.

**1. "Two encoders were tested and both tied, so the typed columns shipped on
cost."** bge-m3 at 1,024 dimensions and 2.2GB scored 11 of 28 targets, p = 0.52.
`all-MiniLM-L6-v2` at 384 dimensions and 90MB, reading the full article rather
than the lead, scored 14 of 28 targets, p = 0.76, with the median favouring the
typed columns. Neither encoder was beaten and neither won, so the typed block was
justified on cost and interpretability rather than measured lift.

> *Replaced by* the representation section. Both arms read the whole article, of
> which ~46% is census tables and place-name lists against 1.5% economy prose,
> and both were scored at full width against the typed block's 29 columns. Fix
> the reading scope and match the width and the tie becomes an outright win for
> the encoder. The cost argument survives as an argument about cost; it is no
> longer an argument about lift, because there is now a lift difference to
> explain.

**2. "A richer representation makes Source A worse: −0.044."** Swapping the
384-dimension embedding in for the typed columns moved A's marginal contribution
from −0.0000 to −0.044, negative on all five external targets. Read at the time
as closing the "a better representation would rescue A" defence.

> *Replaced by* the width control. The embedding carried 384 columns against the
> typed block's 29 and nothing in the comparison held width fixed. Reducing the
> *identical* vectors to 29 dimensions inside each fold moves the same arm from
> −0.021 to +0.011. The defence was not closed; it was mismeasured.

**3. "The MiniLM arms lose to the typed block."** Corrected 2026-08-17. Those arms
were first run with a chunk cap that truncated input at ~9,000 characters, which
bound on **1,871 of 3,144 counties (59.5%)** and discarded 43% of the uniform
arm's own text.

> *Replaced by* the uncapped run. Raised to ~57,600 characters, where it binds on
> 47 counties, the uniform arm rises from +0.00226 to **+0.00322** and its
> row-normalised twin to **+0.00351** — level with the typed block rather than
> behind it. The conclusion that the *input* should not branch on tier is
> unaffected and is cleaner uncapped. Full detail in `source-a-findings.md` §20.

**What did not move.** The tier work — branching, the drop-one sweep, and the
third independent test behind the branching verdict — is a different question and
survived all three corrections unchanged.
""")

# --------------------------------------------------------------------------
nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}


def nbconvert(*argv: str) -> None:
    subprocess.run([sys.executable, "-m", "nbconvert", *argv], check=True)


if not args.for_html:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells)")
    print("execute it with:")
    print(f"  uv run --with nbconvert --with ipykernel jupyter nbconvert "
          f"--to notebook --execute --inplace {OUT.relative_to(REPO)}")
else:
    # The intermediate notebook is a build artifact, not the deliverable, and
    # must never overwrite the committed copy. build/ and outputs/*.html are
    # both already gitignored.
    staging = REPO / "build" / f"{STEM}.ipynb"
    html = REPO / "outputs" / f"{STEM}.html"
    staging.parent.mkdir(parents=True, exist_ok=True)
    html.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, staging)
    print(f"staged {staging} ({len(cells)} cells)")

    nbconvert("--to", "notebook", "--execute", "--inplace", str(staging))
    # --no-input drops the code cells outright, rather than folding them the way
    # the notebook's source_hidden metadata does.
    nbconvert("--to", "html", "--no-input",
              "--output-dir", str(html.parent), "--output", html.name,
              str(staging))
    size = html.stat().st_size / 1e6
    print(f"\nwrote {html} ({size:.1f} MB, no code cells)")
    print("Figures are embedded PNGs — the page needs no network.")
