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
    """Append a code cell, marked to open collapsed.

    `jupyter.source_hidden` is the standard metadata flag for "show the output,
    not the source". JupyterLab, nbclassic and the VS Code / Cursor notebook
    editor all honour it on open, which is what makes this readable as a
    document rather than as a program.
    """
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    cell.metadata["jupyter"] = {"source_hidden": True}
    cell.metadata["collapsed"] = True
    cells.append(cell)


# --------------------------------------------------------------------------
md("""
# `E_macro` — week of 3 August 2026

- **A brief, not an archive.** Evidence and per-column detail live in
  `analysis-output/E_macro_key_findings.ipynb` and the per-source findings documents.
- **Scope:** what changed this week, and what it means.
- **Provenance:** every number is read from committed artifacts in `outputs/` and
  `analysis-output/`. Nothing is re-fit here.
""")

code('''
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO = Path.cwd().parent if Path.cwd().name == "analysis-output" else Path.cwd()
OUTPUTS, ANALYSIS = REPO / "outputs", REPO / "analysis-output"

scores = pd.read_csv(OUTPUTS / "external_target_scores.csv")
grain = pd.read_csv(OUTPUTS / "grain_effect.csv")
decile = pd.read_csv(OUTPUTS / "external_target_by_decile.csv")
ext = json.loads((ANALYSIS / "cross-source" / "external_target_stats.json").read_text())
gst = json.loads((ANALYSIS / "cross-source" / "grain_effect_stats.json").read_text())
a_tiers = json.loads((ANALYSIS / "source-a" / "source_a_tier_stats.json").read_text())
e_tiers = json.loads((ANALYSIS / "source-e" / "source_e_tier_stats.json").read_text())
scope = json.loads((ANALYSIS / "source-a" / "source_a_section_scope_stats.json").read_text())
embed = json.loads((ANALYSIS / "source-a" / "source_a_tiered_embedding_stats.json").read_text())

LABELS = {
    "broadband_rate": "Broadband adoption",
    "median_household_income": "Median household income",
    "median_age": "Median age",
    "median_home_value": "Median home value",
    "mean_commute_minutes": "Mean commute",
}
ORDER = list(LABELS)

# Categorical slots in fixed order, from the validated reference palette. Never
# cycled: a series keeps its slot when other series are added or removed.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, AQUA, CRITICAL = SERIES[0], SERIES[2], "#d03b3b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID = "#d5d4d0", "#ecebe6"
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"

pio.renderers.default = "plotly_mimetype"


def style(fig, title, subtitle="", height=440, legend=True):
    """Apply the house chart style: hairline grid, recessive chrome, hover on."""
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += f"<br><span style='font-size:12.5px;color:{INK2}'>{subtitle}</span>"
    fig.update_layout(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        title=dict(text=heading, x=0, xanchor="left", y=0.96, yanchor="top",
                   font=dict(size=17, color=INK)),
        margin=dict(l=12, r=28, t=96 if subtitle else 72, b=104 if legend else 56),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.24, x=0,
                    font=dict(size=12, color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12.5)),
        bargap=0.34,
        bargroupgap=0.06,
    )
    # automargin so category labels on a horizontal bar chart are never clipped
    # by a fixed left margin -- the plot area shrinks to fit the labels instead.
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                showline=False, ticks="", automargin=True,
                tickfont=dict(size=11.5, color=INK2),
                title_font=dict(size=12, color=INK2))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


print(f"{ext['n_targets']} targets · {ext['fold_strategy']} · {gst['n_markets']} market groups")
''')

# --------------------------------------------------------------------------
md("""
---

## 1. The task you set on Monday: split A and E into groups

Both pillars cut into four groups and pushed on.

- **Verdict:** the groups are worth a great deal as a *diagnostic* and nothing as an
  *architecture*.
- Source E is the exception that matters — there the groups changed what I think the
  pillar is measuring.

### Source A — four content tiers

Split on Wikipedia intro length: **stub** (<100 chars), **thin** (100–283), **mid**
(284–461), **rich** (≥462).

**As a diagnostic, it paid immediately:**

- Named industry content: **1.1% of thin-tier counties, 25.2% of rich** — a 23×
  gradient, with fewer than one county in ten carrying any at all.
- That killed the dense embedding — averaging 1,024 dimensions over 3,144 articles
  gives a vector dominated by counties saying nothing economic.
- It also picked the feature family worth building: *industry*.

**As an architecture, it lost three times.** Should the tier decide…

| …what? | Answer |
|---|---|
| which **model** a county gets | **No.** Tier-specific slopes cost 9% of the lift. Four separate per-tier fits: **−0.01595** against the flat model's +0.00307 — worse than dropping Source A entirely. |
| how much of its **page** is read | **Not conditionally.** Uniform widening beat the shipped rule (+0.00351 vs +0.00307). A tier-conditional rule makes `has_agriculture` mean different things at different county sizes. |
| what goes through an **encoder** | **No, either direction.** Thin reads more: +0.00162. Rich reads more: +0.00073. Everyone reads the same: **+0.00226**. |

**One reason underneath all three:**

- Partitioning 3,144 counties costs more in pooled evidence than heterogeneity costs
  in bias.
- Measurable in the embedding run: tier membership alone explains **0.9–1.0%** of
  vector variance under a uniform rule, **3.7–6.6%** under a conditional one.
- That's the construction rule leaking into the space, 4–7×. And tier is a size proxy
  the baseline already controls for.

**The premise was wrong anyway:**

- Reading the same sections for everyone is worth **+0.00153** over reading them only
  for rich counties — that gap *is* the stub and thin contribution.
- Their pages are not empty. Uniform reading takes stub industry coverage 6.1% →
  **34.0%**, thin 9.7% → 34.9%.

**One caution before widening further:**

- The outright winner — read every section, +0.00403, the only arm at *p* < 0.05 —
  gets there partly on History.
- **67%** of the hits it adds sit in historical framing (*"The South Bronx was a
  manufacturing center for many years"*). A defunct-industry detector wearing a
  current-economy label.
- Excluding narrative sections keeps +0.00044 of the +0.00096, at *p* = 0.22.

**What ships: unchanged.** One uniform schema, one model, economy-titled sections.
Method detail for all three experiments in the appendix.

### The one we talked about: a smaller embedding, fed by tier

The version we discussed: embedding back, narrower than bge-m3's 1024 dimensions, tier
decides which sections go through the encoder.

- **Encoder:** `all-MiniLM-L6-v2`, **384 dimensions**, 90MB against 2.2GB.
- **Chunked and mean-pooled**, so a long article is not silently truncated.
- **Four input rules:** the tier-conditional one you'd expect, its mirror image, and
  two controls — `lead_only` isolates the change of encoder and width, `uniform` reads
  the same sections for everyone.
""")

code('''
REP_ORDER = ["typed_sections", "uniform", "lead_only", "tier_conditional",
             "tier_conditional_inverse", "uniform_pca64", "lead_only_pca64"]
REP_LABEL = {
    "typed_sections": "typed features (shipped)",
    "uniform": "384-d, uniform input",
    "lead_only": "384-d, lead only",
    "tier_conditional": "384-d, thin reads more",
    "tier_conditional_inverse": "384-d, rich reads more",
    "uniform_pca64": "→ 64-d, uniform",
    "lead_only_pca64": "→ 64-d, lead only",
}
lifts = {k: embed["representations"][k]["mean_lift"] for k in REP_ORDER}
VAR_KEYS = ["lead_only", "uniform", "tier_conditional", "tier_conditional_inverse"]
VAR_LABEL = ["lead<br>only", "uniform", "thin reads<br>more", "rich reads<br>more"]
leak = [embed["text_diagnostics"][k]["tier_variance_share"] for k in VAR_KEYS]

# Emphasis, not identity: the two tier-conditional arms are the subject, the rest
# is context. Colour marks which is which; the labels carry the values.
lift_colour = {"typed_sections": INK, "uniform": BLUE, "lead_only": MUTED,
               "tier_conditional": CRITICAL, "tier_conditional_inverse": "#8c2020",
               "uniform_pca64": "#b7d3f6", "lead_only_pca64": "#e6e5e1"}

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.13,
    subplot_titles=("<b>Both directions lose</b>",
                    "<b>…and both leak the rule into the space</b>"),
)
fig.add_trace(go.Bar(
    y=[REP_LABEL[k] for k in REP_ORDER][::-1],
    x=[lifts[k] for k in REP_ORDER][::-1],
    orientation="h", marker_color=[lift_colour[k] for k in REP_ORDER][::-1],
    marker_cornerradius=4, showlegend=False,
    text=[f"{lifts[k]:+.5f}" for k in REP_ORDER][::-1], textposition="outside",
    textfont=dict(size=11.5, color=INK),
    hovertemplate="<b>%{y}</b><br>mean R² lift %{x:+.5f}<extra></extra>",
), row=1, col=1)
fig.add_vline(x=lifts["typed_sections"], line=dict(color=INK, width=1, dash="dot"),
              row=1, col=1)

fig.add_trace(go.Bar(
    x=VAR_LABEL, y=leak,
    marker_color=[MUTED, BLUE, CRITICAL, "#8c2020"], marker_cornerradius=4,
    showlegend=False, text=[f"{v:.3f}" for v in leak], textposition="outside",
    textfont=dict(size=11.5, color=INK),
    hovertemplate="<b>%{x}</b><br>%{y:.1%} of vector variance is tier alone<extra></extra>",
), row=1, col=2)

fig.update_xaxes(title_text="Mean R² lift, 28 targets",
                 range=[0, max(lifts.values()) * 1.3], row=1, col=1)
fig.update_yaxes(showgrid=False, row=1, col=1)
fig.update_yaxes(title_text="Variance explained by tier", tickformat=".2f",
                 range=[0, max(leak) * 1.25], row=1, col=2)
style(fig, "A 384-d embedding, fed four ways",
      f"{embed['encoder'].split('/')[-1]} · dotted line is what Source A ships today",
      height=470, legend=False)
fig.update_annotations(font=dict(size=13, color=INK))
fig.show()
''')

md("""
**Both directions lose; the inverse loses hardest** — worst arm in the run, below
encoding nothing but the lead.

**This killed my first explanation, and that's worth flagging:**

- I had said the original rule failed because it read least from the rich tier, where
  the industry content lives.
- If true, inverting it should have recovered most of `uniform`'s gain. It recovered
  none.
- What survived testing is the leakage in the right-hand panel. The appendix has the
  mechanism I ruled out.

**Two things worth keeping regardless:**

- **Uniform reading helps** — +0.00057 over lead-only.
- **Compression hurts, and this one is firm** — 64 dimensions costs −0.00063 at
  ***p* = 0.015**. Worth having, since "just make the vector smaller" is the reflexive
  answer to a feature-store cost problem.

**Bottom line:**

- A 384-d encoder is a real option — ~¾ of the typed block's lift, 4% of bge-m3's
  disk footprint — **with uniform input only**.
- Still ship the typed features. They win on lift, cost, and interpretability at once.

""")

md("""
### Source E — four volume tiers, and a finding I did not expect

Counties split on `num_returns`: **T1** (<2.2k), **T2** (2.2k–11.7k), **T3**
(11.7k–100k), **T4** (≥100k). Here the groups did more than guide feature choice.
""")

code('''
a_sum, a_lab = a_tiers["summary"], a_tiers["tier_labels"]
e_lab, e_sum = list(e_tiers["tiers"]), e_tiers["tiers"]

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.13,
    subplot_titles=("<b>A: the corpus is uneven, and economically so</b>",
                    "<b>E: equal county counts, unequal economies</b>"),
)

shares = [a_sum[t]["share_any_industry"] for t in a_lab]
fig.add_trace(go.Bar(
    x=[f"{t.capitalize()}<br>n={a_sum[t]['n_counties']:,}" for t in a_lab],
    y=shares, marker_color=BLUE,
    marker_cornerradius=4, showlegend=False,
    text=[f"{s:.1%}" for s in shares], textposition="outside",
    textfont=dict(size=12, color=INK),
    customdata=[[a_sum[t]["n_counties"], a_sum[t]["mean_length"]] for t in a_lab],
    hovertemplate="%{y:.1%} name an industry<br>%{customdata[0]:,} counties"
                  "<br>%{customdata[1]:.0f} chars mean intro<extra></extra>",
), row=1, col=1)

cty = [e_sum[t]["share_of_counties"] for t in e_lab]
inc = [e_sum[t]["share_of_investment_income"] for t in e_lab]
tiers_short = ["T1 thin", "T2 small", "T3 mid", "T4 large"]
for name, values, colour in (("Share of counties", cty, MUTED),
                             ("Share of national investment income", inc, AQUA)):
    fig.add_trace(go.Bar(
        x=tiers_short, y=values, name=name, marker_color=colour, marker_cornerradius=4,
        text=[f"{v:.2%}" if v < 0.05 else f"{v:.0%}" for v in values],
        textposition="outside", textfont=dict(size=11, color=INK),
        hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2%}}<extra></extra>",
    ), row=1, col=2)

fig.update_yaxes(tickformat=".0%", title_text="Intro names an industry",
                 range=[0, max(shares) * 1.25], row=1, col=1)
fig.update_yaxes(tickformat=".0%", range=[0, 0.95], row=1, col=2)
fig.update_xaxes(title_text="Content tier", row=1, col=1)
fig.update_xaxes(title_text="Volume tier (tax returns filed)", row=1, col=2)
style(fig, "What the tiers exposed",
      "Source A splits on article length; Source E on returns filed.", height=460)
fig.update_annotations(font=dict(size=13, color=INK))
fig.show()
''')

md("""
**Sit with the right panel.**

- T1 and T4 are each ~10% of counties. T1 holds **0.14%** of national investment
  income; T4 holds **82.6%**.
- So an unweighted county feature and the economy it claims to describe are not the
  same object — national aggregate ratio **0.156** against an unweighted county mean
  of **0.107**.

**Three more things the split turned up. Two correct earlier conclusions:**

- **The strongest cross-pillar link in the whole project is a large-county
  phenomenon.** B Real Estate LQ × E capital-to-wage runs +0.394 nationally, but
  **+0.476 in T4 and −0.058 in T1**. It does not exist for the counties with the least
  data. Anyone serving rural inventory needs that stated before leaning on B ↔ E.
- **Round 1 had the stability backwards.** I had said small counties were the volatile
  ones. On ranks it is the opposite — Spearman stability *improves* with size
  (0.861 → 0.941), and median year-over-year moves *rise* with size (0.298 → 0.393).
  Which means round 1's proposed fix, weighting by `num_returns`, would have upweighted
  exactly the counties whose values move most between vintages.
- **The dispersion is not sampling noise.** Regressing log dispersion on log median
  returns gives a slope of **+0.026**; pure sampling error would give −0.5. Small
  counties' spread is real economic variation, not thin-data artifact.

**Net from the assigned work:**

- Neither pillar should branch on its groups.
- Source A ships one uniform schema.
- Source E ships with an explicit warning that its best cross-pillar result is
  conditional on county size.
- The groups did their job by changing what gets **shipped** and what gets
  **disclosed** — not by becoming part of the model.
""")

# --------------------------------------------------------------------------
md("""
---

## 2. The rest of the week, in one line each

- **Grain moved from blocker to non-issue.** Monday's read said joining at market
  grain would destroy the signal; that was half a finding, and the other half
  reverses it. Net cost of a market-grain join: **0.022 R²**. Section 5.
- **The validation stopped being circular.** Everything before this week scored the
  pillars against each other. `E_macro` is now scored against five public outcomes
  outside all six pillars, on held-out states. Sections 3, 4 and 6.
- **Two pillars got cleaned up** — Source D's freight tonnages were county size in
  disguise, Source E's dollar totals likewise. Section 7.

---

## 3. Getting out of the circle

**The problem with every validation before this week:**

- All of them were **pillar against pillar** — predict one federal source's features
  from the other five.
- That measures whether six agencies agree with each other. It cannot say whether any
  of them is *useful*.
- Worse, the bias runs the wrong way: it penalises a source precisely for agreeing
  with the others.

**Why we can't just use a real label:** the project is scoped to public data only, so
no downstream label exists here.

**The substitute — five public outcomes no pillar measures:**

- Household broadband adoption, median household income, median age, median home
  value, mean commute time.
- All ACS. None constructed from any pillar's inputs.

**Built around one specific objection:**

- The consumer joins on DMA with millions of impressions per market, so it estimates a
  geographic fixed effect essentially for free — which makes any static geo-keyed
  feature look redundant.
- A fixed effect has exactly one weakness: **no parameter for a place it has never
  seen.**
- So: hold out whole states, compare against a model that knows only county size.
  That's the seam.
""")

code('''
piv = (scores[scores.model.isin(["size", "size_emacro"])]
       .pivot(index="target", columns="model", values="r2_ablated").reindex(ORDER))
lift = (scores[scores.model.eq("size_emacro")]
        .set_index("target")["lift_over_size_ablated"].reindex(ORDER))
names = [LABELS[t] for t in ORDER][::-1]

fig = go.Figure()
for label, column, colour in (("County size only", "size", MUTED),
                              ("County size + E_macro", "size_emacro", BLUE)):
    fig.add_trace(go.Bar(
        y=names, x=piv[column][::-1], orientation="h", name=label,
        marker_color=colour, marker_cornerradius=4,
        text=[f"{v:.2f}" for v in piv[column][::-1]], textposition="outside",
        textfont=dict(size=11, color=INK2 if column == "size" else INK),
        hovertemplate=f"<b>%{{y}}</b><br>{label}: R² %{{x:.3f}}<extra></extra>",
    ))
for i, target in enumerate(ORDER[::-1]):
    fig.add_annotation(x=1.0, y=i, xref="x", yref="y", xanchor="right",
                       text=f"<b>+{lift[target]:.3f}</b>", showarrow=False,
                       font=dict(size=12.5, color=INK))

fig.update_xaxes(title_text="R² on held-out states", range=[0, 1.03])
fig.update_yaxes(showgrid=False)
style(fig, "Five outcomes outside every pillar. Five for five.",
      f"Mean gain over size alone: +{ext['mean_lift_over_size_ablated']:.3f} R²"
      "  ·  gain per outcome at right", height=470)
fig.show()
''')

md("""
- **Grey bar** = the fixed-effect model's position. **Blue bar** = what `E_macro`
  adds on top. The gap on the right of each row is the whole result.
- **Sanity check passing:** an intercept-only model scores **≈0** on held-out states —
  the fixed effect handed a county it has never seen, with nothing to say.
""")

# --------------------------------------------------------------------------
md("""
---

## 4. The discount I applied to my own result

- Raw number: **+0.212**. Reported: **+0.190**.
- The difference matters because it's the kind of thing that gets caught in review
  rather than found by the author.

**Two pillar columns don't *predict* their target so much as restate it:**

- `wage_per_return_thousands` (IRS) is average wage income per tax return, which is
  very close to a definition of median household income. Removing it drops that
  outcome's gain from +0.247 to **+0.154** — one column was carrying 38% of the
  apparent result.
- `retirement_destination` (USDA) flags counties with heavy in-migration of people
  aged 60+, which restates age structure. Smaller effect: +0.256 → +0.239.

Both dropped from their own target's run, kept everywhere else. **The headline is the
discounted number.**
""")

# --------------------------------------------------------------------------
md("""
---

## 5. The grain reversal

The part I got wrong on Monday and corrected twice.

### First — what "grain" means here

**Grain is what one row stands for** — not how detailed the data inside a row is, the
identity of the row itself. Three are in play:

| Grain | One row is | Count |
|---|---|---|
| What `E_macro` produces | one US county (`fips_code`) | 3,143 |
| The consumer's training data | one impression / ad request / household | millions |
| The consumer's *geo key* | one Nielsen DMA (media market) | ~210 |

**`E_macro` is a lookup table: geo key → vector.**

- The downstream model never consumes a county. It consumes an impression row and
  joins the vector on whatever geo key that row carries.
- So the grain has to match the key available at join time. That's the entire question.
- Row carries only `dma` → the table must be re-keyed to 210 rows; 3,143 vectors
  collapse into 210.
- Row carries ZIP → county is derivable, table ships unchanged. **Current read: it does
  carry ZIP**, which makes county grain a live option rather than wishful thinking.

**Why this is load-bearing for a geo embedding specifically:**

- The consumer holds millions of impressions per market, so a 210-level geographic
  fixed effect is free and precise.
- At DMA grain any static geo-keyed feature is *exactly collinear* with it by
  construction — a DMA dummy already captures everything `E_macro` could say.
- At county grain it isn't, because 3,143 units is too thin for the consumer to fit its
  own per-county effect.

Finer grain is therefore not a nice-to-have. **Below DMA is where the feature stops
being redundant with something the consumer already has.** (Section 3's held-out-state
design is the other escape from the same fixed effect: it has no parameter for a
county it has never seen.)

### The penalty has two halves

Collapsing 3,143 keys to ~210 costs something in **two separable ways**, and on
Monday I measured only one:

1. **Fewer distinct rows.** 210 keys means the downstream model can ever see at most
   210 distinct values of this feature block. Measured by retraining on random county
   subsets. Verdict: real and bad — worth **−0.121** on average, and badly unstable at
   that size. On broadband, the outcome closest to the consumer's own domain, it goes
   slightly negative.
2. **Aggregation blur.** Averaging ~15 counties into one market destroys
   within-market variation — and the suburb-outside-New-York-versus-Cleveland
   distinction `E_macro` exists to capture can partly live *inside* a single market.
   Not measured Monday. I said it "could cut either way."

**It cuts the other way.** Aggregation is worth **+0.099** — very nearly cancelling
the row-count loss.

**One implementation detail decides whether that number is real: aggregate the inputs,
not the outputs.**

- A market's location quotient must be re-derived from summed employment, not averaged
  from fifteen counties' quotients.
- That's why Source B had to ship raw employment levels this week — see the re-test
  below.
""")

code('''
import numpy as np

by_size = pd.DataFrame(ext["by_training_size"])
TICKS = [210, 400, 800, 1600, 3000]

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.11,
    subplot_titles=("<b>The gain shrinks…</b>", "<b>…and stops being reliable</b>"),
)
for slot, target in zip(SERIES, ORDER):
    d = by_size[by_size.target.eq(target)].sort_values("n_train_units")
    fig.add_trace(go.Scatter(
        x=d.n_train_units, y=d.mean_lift_over_size, name=LABELS[target],
        mode="lines+markers", line=dict(color=slot, width=2),
        marker=dict(size=8, color=slot, line=dict(color=SURFACE, width=2)),
        legendgroup=target,
        hovertemplate=f"<b>{LABELS[target]}</b><br>%{{x:,}} counties"
                      "<br>lift %{y:+.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d.n_train_units, y=d.sd_lift_over_size, name=LABELS[target],
        mode="lines+markers", line=dict(color=slot, width=2),
        marker=dict(size=8, color=slot, line=dict(color=SURFACE, width=2)),
        legendgroup=target, showlegend=False,
        hovertemplate=f"<b>{LABELS[target]}</b><br>%{{x:,}} counties"
                      "<br>spread %{y:.3f}<extra></extra>",
    ), row=1, col=2)

for col in (1, 2):
    fig.add_vline(x=210, line=dict(color=CRITICAL, width=1.3, dash="dot"), row=1, col=col)
    fig.update_xaxes(type="log", tickvals=TICKS, ticktext=[f"{t:,}" for t in TICKS],
                     title_text="Counties available for training", row=1, col=col)
fig.add_annotation(x=np.log10(215), y=1.0, xref="x", yref="paper", xanchor="left",
                   yanchor="bottom", text="≈ DMA count", showarrow=False,
                   font=dict(size=11.5, color=CRITICAL))
fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=1, col=1)
fig.update_yaxes(title_text="Gain over size-only baseline (R²)", row=1, col=1)
fig.update_yaxes(title_text="Spread across 10 random draws", row=1, col=2)
style(fig, "Half the story: fewer rows hurt, and get unreliable",
      "At 210 units the spread on some outcomes is wider than the effect being measured.",
      height=470)
fig.update_layout(hovermode="x unified")
fig.update_annotations(font=dict(size=13, color=INK))
fig.show()
''')

md("""
- **The right panel matters as much as the left.** At 210 units the answer depends
  heavily on which units you happen to have.
- That instability — more than the size of the drop — is what made a market-grain join
  look unacceptable.

Then the other half got measured:
""")

code('''
arms = (grain.pivot(index="target", columns="arm", values="mean_lift_over_size")
        .reindex(ORDER))
ARMS = (("county_full", "All 3,143 counties", BLUE),
        ("county_subsample", "208 counties (row-count loss only)", MUTED),
        ("market_aggregate", "208 aggregated markets", AQUA))

fig = go.Figure()
for column, label, colour in ARMS:
    values = arms[column]
    fig.add_trace(go.Bar(
        x=[LABELS[t] for t in ORDER], y=values, name=label,
        marker_color=colour, marker_cornerradius=4,
        text=[f"{v:+.2f}" for v in values], textposition="outside",
        textfont=dict(size=10.5, color=INK),
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:+.3f}}<extra></extra>",
    ))
fig.add_hline(y=0, line=dict(color=INK2, width=1))
fig.update_yaxes(title_text="Gain over size-only baseline (R²)",
                 range=[min(arms.min()) - 0.09, max(arms.max()) + 0.09])
fig.update_xaxes(showgrid=False)
style(fig, "The other half: aggregation helps, and nearly cancels the loss",
      f"Row-count effect {gst['row_count_effect']:+.3f}  ·  aggregation effect "
      f"{gst['aggregation_effect']:+.3f}  ·  market arm wins on 3 of 5", height=480)
fig.show()
''')

md("""
**Compare the grey bar to the teal one** — same 208 rows in both, the only difference
is whether those rows are lone counties or aggregated markets.

- Median home value: **+0.12 → +0.34**. Median age: **+0.21 → +0.41**.
- Mechanism is not exotic — population-weighted aggregation turns sparse, noisy county
  columns (suppressed BLS cells, single-article Wikipedia flags) into stable continuous
  shares, and does the same favour to the outcome being predicted.

**Then I tried to break it:**

- Obvious objection: most columns were *approximated* at market level rather than
  properly re-derived, which would flatter the market arm.
- So the inputs were re-ingested to make the aggregation honest. Source B now ships
  raw employment levels (72 of 118 columns re-derived instead of 49), and Source D now
  ships the partner-tons distribution behind its two concentration indices (**74 of
  118**).
- Re-running after Source B moved the aggregation effect by **0.001**. Re-running
  after Source D moved it by **0.006**, to +0.099. **No outcome changed sign either
  time.**
- The 40 columns still approximated are Source F's typology flags, which have no
  underlying quantity to re-sum. That is the floor, not a to-do.
- Estimated cost of the Source B fix: 2–3 days. Actual: one download and two script
  changes.

**The caveat that still stands:**

- The 208 groups are k-means clusters of county centroids at DMA cardinality. They are
  **not** DMAs — that delineation is proprietary.
- Real markets follow media boundaries and are less spatially compact; the aggregated
  outcome is genuinely less noisy than a county one.
- Both biases favour the market arm, so **+0.099 is an upper bound.** Re-deriving
  the inputs fixed a third bias; it did not touch these two.

**Two thresholds, worth keeping apart:**

- For market grain to be a **blocker** again — signal destroyed rather than merely
  reduced — essentially the whole +0.099 would have to be an artifact of the proxy.
  That is a large claim, and the biases named above are nowhere near big enough for it.
- For county grain to be **strictly better**, the overstatement only has to be
  **0.022** — the gap between the full-county arm (+0.212) and the market arm
  (+0.189). Small enough that I would not argue the market arm is *better*, only that
  it is not disqualifying.
""")

# --------------------------------------------------------------------------
md("""
---

## 6. Where the model cannot win, and why that is fine

**One result looks alarming until you see what's under it:** on the smallest counties
the size-only baseline scores **negative** R².

- Something is badly wrong there — but wrong with the *data*, not the model.
- ACS publishes a margin of error with every estimate; those are now ingested alongside
  the values, which splits each outcome's variance into signal and sampling noise.
- Smallest population decile: **30% of variance is sampling noise** — error no model
  can ever explain. Largest decile: under 1%.
""")

code('''
dec = (decile.groupby("population_decile")
       .agg(median_population=("median_population", "mean"),
            noise_share=("noise_share", "mean"),
            r2_size=("r2_size", "mean"),
            r2_size_emacro=("r2_size_emacro", "mean"))
       .reset_index())
ticks = [f"{int(round(p, -2)):,}" for p in dec.median_population]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=dec.population_decile, y=dec.noise_share, name="ACS sampling noise (share of variance)",
    marker_color="#f7dcdc", marker_cornerradius=4,
    hovertemplate="<b>%{customdata:,} median population</b><br>"
                  "sampling noise %{y:.1%} of variance<extra></extra>",
    customdata=dec.median_population.round(-2).astype(int),
))
for label, column, colour, width in (("County size only", "r2_size", MUTED, 2),
                                     ("County size + E_macro", "r2_size_emacro", BLUE, 2.4)):
    fig.add_trace(go.Scatter(
        x=dec.population_decile, y=dec[column], name=label, mode="lines+markers",
        line=dict(color=colour, width=width),
        marker=dict(size=8, color=colour, line=dict(color=SURFACE, width=2)),
        hovertemplate=f"{label}: R² %{{y:.3f}}<extra></extra>",
    ))
fig.add_hline(y=0, line=dict(color=INK2, width=1))
fig.add_annotation(x=1, y=dec.r2_size.iloc[0], text=f"<b>{dec.r2_size.iloc[0]:.2f}</b>",
                   showarrow=True, arrowhead=0, arrowcolor=CRITICAL, ax=34, ay=26,
                   font=dict(size=12, color=CRITICAL))
fig.add_annotation(x=1, y=dec.noise_share.iloc[0] + 0.05,
                   text=f"<b>{dec.noise_share.iloc[0]:.0%} noise</b>", showarrow=False,
                   font=dict(size=11.5, color="#b91c1c"))
fig.update_xaxes(tickvals=dec.population_decile, ticktext=ticks,
                 title_text="County population decile (median population)", showgrid=False)
fig.update_yaxes(title_text="R², averaged over the five outcomes")
style(fig, "Small counties are mostly measurement error",
      "Where the grey line dives, the data is noise — and E_macro still recovers "
      "usable signal there.", height=470)
fig.update_layout(hovermode="x unified")
fig.show()
''')

md("""
**Two things follow:**

- The negative baseline on tiny counties is a property of ACS, not a defect in the
  pipeline.
- More useful: `E_macro` stays **positive** in exactly the decile where the size
  baseline collapses — which is what you'd want from a feature meant to describe places
  a size proxy describes badly.
- It also sets an honest ceiling: on the smallest counties no model can exceed
  **R² ≈ 0.70**, however good the features.
""")

# --------------------------------------------------------------------------
md("""
---

## 7. Plumbing, briefly

Doesn't change the story; does change what ships.

- **Source D's freight tonnages were county size wearing a freight label.** All ten
  raw tonnage columns moved into the size control. Measured cost of removing them:
  nothing. Normalising them per capita turned out to be a re-expression, not a fix.
- **Commodity shares replaced them** — 5 of 10 clear the size-free bar that none of
  the raw columns cleared, and the gain routes almost entirely to Source B, which is
  interpretable rather than mysterious.
- **Source E's capital-to-wage ratio was decomposed** into its components, plus a
  five-year panel — which is what made the tier work in Section 1 possible. Its
  remaining dollar totals moved into the size control on the same principle as D's.
- **Source B now ships raw employment levels**, so its 40 location quotients can be
  re-derived at any grain instead of approximated. This is what made the grain
  re-test above trustworthy.
- **Source A** was also measured for marginal value against a baseline that already
  contains the other five pillars — the harder question than the tier split — and the
  last embedding-era artifacts were retired repo-wide.
- **The archive notebook was re-read against regenerated outputs.** Several Section 8
  numbers had drifted after D and E moved into the size control; corrected.

### What's next, regardless

1. **Done 2026-08-07** — Source D's two partner-concentration indices are re-derived
   from the partner-tons distribution it now ships. Provenance 72/42/4 → **74/40/4**,
   aggregation effect +0.105 → **+0.099**, no target changing sign. What stays
   approximated is Source F's typology flags, which have no underlying quantity to
   re-sum.
2. **Build the assembly step.** The go/no-go evidence exists and `PROJECT_GOAL.md` is
   explicit that this was never blocked on the size decision. This is the next real
   piece of work.
3. **Keep the grain caveat live.** If a real DMA delineation ever becomes available,
   the market-arm result is worth re-running against it once.

""")

# --------------------------------------------------------------------------
md("""
---

## Appendix — method detail for Section 1

Kept out of the argument, kept in the document. Nothing here changes a conclusion;
it is what you would need to check one.

### A1. What "a model" is in the branching test

Each of the 28 targets is a feature belonging to one of the other five pillars, and
every target is scored the same way:

1. **Control first.** Ordinary least squares on three size measures
   (`log_population`, `log_agi`, `log_gdp_latest`) plus one dummy per state.
   Deliberately unpenalized — ~50 controls against 3,144 rows has nothing to
   regularize, and shrinking them would let a wide Source A block degrade them.
2. **Source A predicts only what the control missed.** Its 29 columns are fit to the
   control's *residual*: median-impute → standardize → ridge, with the penalty chosen
   by an inner 5-fold search over 10⁰–10⁶.
3. **Score = out-of-fold R² gain over the control.** Outer 5-fold, shuffled, seed 42.

Only step 2 differs across the three architectures:

- **Flat, 29 columns.** One ridge over all 3,144 counties. A stub county and a rich
  county share every slope.
- **Tier-crossed, 120 columns.** Four tier dummies, plus each feature copied into four
  slots — populated in the county's own tier's slot, zero elsewhere. Every tier gets
  its own slope and intercept, but under **one fit and one shared penalty**, so a tier
  with nothing to say is shrunk toward zero using evidence pooled from tiers that do.
- **Four independent fits.** The corpus is partitioned and a separate ridge fit inside
  each partition, each selecting its own penalty on its own rows: **294 / 1,274 / 788
  / 788** counties. No pooling, no shared shrinkage.

The partitioned form loses hardest for two structural reasons. The stub tier's 294
counties contain almost no industry content, so its private fit has nothing to find
and no shared penalty pulling its coefficients toward zero — it emits noise across its
whole slice. And no tier can borrow strength: a slope estimated on 788 rich counties
can no longer inform the 788 mid ones.

**One detail worth keeping straight**, because it is the difference between a real
tier effect and a manufactured one: the per-tier *results* quoted anywhere in this
project come from scoring the single global model's out-of-fold predictions on tier
subsets, never from refitting per tier.

*Caveat on the table in Section 1: the flat and tier-crossed numbers are the
2026-08-04 re-score against Census population; the four-fits number was measured once
against the retired baseline and not re-run, since a result that far negative does not
turn on a fourth-decimal baseline change.*

### A2. What the pipeline actually consumes, per tier

Every county gets the same rule — read the lead, plus any section whose title marks it
economic — but what that yields is unequal by an order of magnitude:

| tier | median lead chars | has an economy section | econ chars when present | mean total chars used |
|---|---|---|---|---|
| stub | 70 | 10.5% | 405 | **127** |
| thin | 191 | 14.2% | 445 | **303** |
| mid | 354 | 21.2% | 564 | **588** |
| rich | 686 | 35.7% | 1,001 | **1,267** |

Rich counties are 25% of the corpus and 57% of all economy-section text read. The
pipeline reads ~2.0M characters of the ~56M already downloaded and sitting in
`data/source_a_sections.parquet`, so widening scope costs an extraction pass, not a
refetch — which is why there is no budget argument for spending depth by tier.

### A3. Section scope, full results

`scripts/analyze_source_a_section_scope.py`. Four scopes, 28 targets, same protocol.
The shipped whitelist reproduces +0.00307 exactly, which is the new harness agreeing
with the old one.

| scope | mean lift | coverage | paired *p* | new hits in historical framing |
|---|---|---|---|---|
| economy-titled only *(shipped)* | +0.00307 | 18.8% | — | — |
| \\+ transportation, government, infrastructure | +0.00312 | 21.6% | 0.76 | 22% |
| everything except History and Notable People | +0.00351 | 42.5% | 0.22 | 38% |
| every section | **+0.00403** | 55.2% | **0.048** | **67%** |

My hand-picked widening — Transportation, Government, Infrastructure — bought
essentially nothing. The value sits in sections I would not have nominated: Geography
carries "planar areas largely devoted to agriculture", Recreation carries tourism.

Coverage by tier under the widest safe scope: stub 6.1% → 34.0%, thin 9.7% → 34.9%,
mid 18.0% → 40.2%, rich 39.0% → 60.3%.

The precision check samples the hits each widening adds and flags historical framing
by a crude marker — a pre-1990 year or a past-tense cessation phrase — for human
review rather than for a decision. Examples it caught: *"The South Bronx was a
manufacturing center for many years"*; a county "settled between 1870 and 1880 as a
ranching hub"; an oil flag set by a driller born in 1819, from a Notable People list.
Rows are in `outputs/source_a_section_scope_precision.csv`.

Any adoption would need a human-labelled precision sample rather than that heuristic,
and probably a recency filter. Nothing was rewritten.

### A4. Embedding diagnostics, including the mechanism I ruled out

`scripts/analyze_source_a_tiered_embedding.py`. Encoder `all-MiniLM-L6-v2`, 384
dimensions, text chunked at 900 characters and mean-pooled per county.

**Magnitude — tested, not the cause.** Mean-pooling *k* unit-length chunks shrinks the
result, so a pooled vector's length reports how much text its county contributed: flat
at ~0.72 under `uniform`, but spread 0.69–1.00 across tiers under the conditional
rules. Standardization runs down columns and cannot remove a gradient running across
rows, so this looked like a clean culprit. Scoring direction alone, every vector
renormalized, recovers **+0.0001 of a 0.0015 deficit**.

**Leakage — the one that held.** Share of vector variance explained by tier membership
alone: 0.009 (lead-only), 0.010 (uniform), 0.066 (thin reads more), 0.037 (rich reads
more). It explains why neither conditional arm beats either uniform one; it does *not*
rank the two conditional arms, since the original leaks more yet scores higher. For
the inverse specifically the plain reading is best: it dilutes the 788 rich counties
that carry the economic content into nine chunks of geography and demographics, and
gives the thin tier nothing new.

**Caps and noise.** The uniform arm hit its 10-chunk cap on 1,871 counties and dropped
17M characters, so it is a lower bound on "read everything" rather than a measurement
of it. Every difference in the run except the PCA-64 result sits inside the pillar's
noise band, with *p* between 0.04 and 0.17.

---

*Sources: `analysis-output/cross-source/external-target-findings.md` (§10–§20),
`analysis-output/source-a/source-a-findings.md` (§13–§15),
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
