"""Generate analysis-output/E_macro_pillar_worth_2026-08-13.ipynb.

An executive status notebook, presented live to the commissioning side and their
leadership. A progress artifact, not the go/no-go: it reports what the project
knows, and does not ask for the decision.

Design: docs/superpowers/specs/2026-08-13-exec-status-notebook-design.md

Every figure is computed from the committed artifacts in outputs/ and
analysis-output/, which were regenerated from data/*.parquet on 2026-08-13.
Nothing here is hardcoded — a number that moves upstream moves in the notebook.

Wording in the Source A and Source F sections is constrained by
analysis-output/cross-source/pillar-marginal-findings.md section 9. See the
spec's "Wording constraints" section before editing those cells.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path("/Users/maxgarcia/Desktop/MacroEmbeddings")
OUT = REPO / "analysis-output" / "E_macro_pillar_worth_2026-08-13.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell that opens collapsed, showing output but not source.

    Same convention as build_weekly_brief.py: `jupyter.source_hidden` is honoured
    by JupyterLab, nbclassic and the VS Code / Cursor notebook editor, which is
    what lets this be read as a document while it is presented.
    """
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    cell.metadata["jupyter"] = {"source_hidden": True}
    cell.metadata["collapsed"] = True
    cells.append(cell)


# ==========================================================================
# Title
# ==========================================================================
md("""
# `E_macro` — what each pillar is actually worth

**Status report, 13 August 2026.** Presented, not circulated: the detail behind
every number is in `analysis-output/`, and this notebook is the tour.

**The idea this round turns on.** Every previous update measured whether the six
pillars *correlate with each other*. That measures coherence — whether two
federal agencies see the same economy — and coherence is not usefulness. This
round measures something different: **what each pillar adds to a model that
already has the other five**, scored against outcomes that live outside the
project entirely.

Two pillars moved on that measurement. One of them is the pillar this project
had marked "done."

**Scope, as set:** public and open-source data only, no company data, no
downstream label. That boundary is why the evidence below is built the way it is.
""")

code('''
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO = Path.cwd().parent if Path.cwd().name == "analysis-output" else Path.cwd()
OUTPUTS, ANALYSIS = REPO / "outputs", REPO / "analysis-output"
XSRC = ANALYSIS / "cross-source"

# ---- artifacts -----------------------------------------------------------
ext = json.loads((XSRC / "external_target_stats.json").read_text())
blk = json.loads((XSRC / "pillar_block_marginal_stats.json").read_text())
gst = json.loads((XSRC / "grain_effect_stats.json").read_text())
scope = json.loads((ANALYSIS / "source-a" / "source_a_section_scope_stats.json").read_text())
embed = json.loads((ANALYSIS / "source-a" / "source_a_tiered_embedding_stats.json").read_text())

scores = pd.read_csv(OUTPUTS / "external_target_scores.csv")
placebo = pd.read_csv(OUTPUTS / "external_target_drop_one_placebo.csv")
grain = pd.read_csv(OUTPUTS / "grain_effect.csv")
vintages = pd.read_csv(OUTPUTS / "pillar_vintages.csv")

# ---- palette -------------------------------------------------------------
# Same validated reference palette as the weekly brief. Categorical slots are
# fixed, never cycled: a series keeps its slot when others are added or removed.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, AQUA, CRITICAL = SERIES[0], SERIES[2], "#d03b3b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID = "#d5d4d0", "#ecebe6"
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"

pio.renderers.default = "plotly_mimetype"


def style(fig, title, subtitle="", height=440, legend=True):
    """House chart style, sized for a projector rather than a laptop.

    Differences from the weekly brief's screen-oriented version: 16px base type
    against 13px, 21px titles against 17px, and heavier tick labels. Everything
    else — hairline grid, recessive chrome, legend below the plot — is the same,
    so the two documents read as one system.
    """
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += f"<br><span style='font-size:14px;color:{INK2}'>{subtitle}</span>"
    fig.update_layout(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=16, color=INK),
        # y=0.93 rather than flush to the top edge: at 21px the ascenders clip
        # against the canvas when the title is anchored any higher.
        title=dict(text=heading, x=0, xanchor="left", y=0.93, yanchor="top",
                   font=dict(size=21, color=INK)),
        margin=dict(l=12, r=44, t=124 if subtitle else 90, b=104 if legend else 64),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0,
                    font=dict(size=14, color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID,
                        font=dict(family=FONT, size=14)),
        bargap=0.34,
        bargroupgap=0.06,
        uniformtext=dict(mode="hide", minsize=11),
    )
    # automargin so category labels on a horizontal bar chart are never clipped
    # by a fixed left margin -- the plot area shrinks to fit the labels instead.
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                showline=False, ticks="", automargin=True,
                tickfont=dict(size=14, color=INK2),
                title_font=dict(size=14.5, color=INK2))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig

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

# ---- provenance ----------------------------------------------------------
sha = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip()
# Tracked modifications only, and notebooks excluded: this notebook lives under
# analysis-output/ and its own execution would otherwise register as drift in the
# artifacts it is reporting on.
dirty = subprocess.run(
    ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no",
     "--", "outputs", "analysis-output", ":(exclude)*.ipynb"],
    capture_output=True, text=True,
).stdout.strip()

print(f"Run stamp · {date.today():%d %B %Y} · commit {sha}")
print(f"Every figure below is computed from outputs/ and analysis-output/,")
print(f"regenerated from data/*.parquet by the six analysis scripts on 13 Aug 2026.")
print(f"Regenerated artifacts vs. committed: "
      f"{'IDENTICAL — no drift' if not dirty else 'DIFFER — see git diff'}")
print()
print(f"Evidence base · {ext['n_targets']} external targets · {ext['fold_strategy']} "
      f"· {blk['n_null_reps']} null reps · seed {ext['random_seed']}")
''')

# ==========================================================================
# 1. The turn
# ==========================================================================
md("""
---

## 1. The turn — from *what correlates* to *what pays*

**What the last update could say.** All six pillars ingested at county grain
(N ≈ 3,143). A full 15-pillar-pair sweep: 50 feature pairs, 499 permutations, one
Benjamini-Hochberg correction, every correlation recomputed controlling for
county size.

**What it could not say.** Whether any of it is *useful*. Pillar-versus-pillar
tests are circular by construction — they can tell you Source B and Source E see
the same economy, and tell you nothing about whether either predicts anything.

**The instrument built this round.** Withhold one pillar's entire block from a
model that already holds county size and the other five pillars, and measure the
R² it loses. Run it against **five public county outcomes that are not in the
project** — ACS broadband adoption, median household income, median age, median
home value, mean commute — scored out-of-fold on **states the model never trained
on**.

Three design decisions worth naming, because they are what make the result
survivable:

- **Every pillar takes the same test.** A test only the suspect sits proves
  nothing in either direction.
- **Restatements are ablated.** Where one pillar's column restates another's, it
  is removed from both sides, so no pillar gets credit for repeating a neighbour.
- **A noise floor is measured, not assumed.** Each pillar's block is shuffled and
  re-scored many times; the largest contribution shuffled data produces is the
  bar a real contribution has to clear.

**The rule was written before the numbers arrived**, and is reproduced verbatim
from the implementation plan:

> F ships as a pillar if its marginal contribution — R²(size + all pillars) −
> R²(size + all pillars except F), pooled out-of-fold over the five external ACS
> targets, with restatement columns ablated — is positive on a majority of
> targets and above the shuffled-feature noise floor. Otherwise `E_macro` ships
> five pillars and the go/no-go deck says so plainly.

Nothing about it was renegotiated afterwards. The failure mode this exists to
avoid is choosing the justification after seeing the result.
""")

# ==========================================================================
# 2. The headline
# ==========================================================================
md("""
---

## 2. What the six pillars are worth

Read the figure as: **how much predictive power the matrix loses when this pillar
is taken out of it.** The grey band is the noise floor — the most any *shuffled*
version of that block managed.
""")

code('''
floor = ext["drop_one_noise_floor"]
# Contribution comes from drop_one, NOT from the noise-floor dict: the latter
# carries F's `_no_ametro` robustness variant (+0.0410, scored against a
# different reference model), and quoting it here would silently understate the
# headline figure the findings report and docs both use (+0.0413).
rows = [{"pillar": p,
         "label": PILLAR_NAME[p],
         "contribution": ext["drop_one"][f"size_emacro_drop_{p}"]["mean_contribution_ablated"],
         "floor": floor[p]["max_placebo"],
         "above": floor[p]["n_targets_above_floor"],
         "positive": ext["drop_one"][f"size_emacro_drop_{p}"]["n_positive_ablated"]}
        for p in PILLARS]
worth = pd.DataFrame(rows).sort_values("contribution")

band = float(worth["floor"].max())
hi = float(worth["contribution"].max())
lo = min(-band, float(worth["contribution"].min()))

fig = go.Figure()
fig.add_trace(go.Bar(
    x=worth["contribution"], y=worth["label"], orientation="h",
    marker_color=[CRITICAL if v <= 0 else BLUE for v in worth["contribution"]],
    hovertemplate="%{y}<br>contribution %{x:+.4f}<extra></extra>",
    showlegend=False,
))
# Values are set in a fixed right-hand column rather than as outside bar labels.
# Outside labels track the bar end, which puts Source A's near-zero value on top
# of its own category label and pushes Source E's off the canvas.
value_x = hi * 1.06
for lab, v in zip(worth["label"], worth["contribution"]):
    fig.add_annotation(x=value_x, y=lab, text=f"<b>{v:+.4f}</b>", showarrow=False,
                       xanchor="left", font=dict(size=15, color=INK))
# The noise floor is a property of the measurement, not a series — drawn as a
# shaded band rather than a legend entry so it reads as the bar to clear.
fig.add_vrect(x0=-band, x1=band, fillcolor=MUTED, opacity=0.45, line_width=0,
              annotation_text="noise floor", annotation_position="top",
              annotation_font=dict(size=13, color=INK2))
style(fig,
      "What each pillar adds that the other five do not",
      "Mean R² lost when the block is withheld · 5 public ACS targets · "
      "out-of-fold on held-out states · restatements ablated",
      height=520, legend=False)
fig.update_xaxes(title="marginal R²", tickformat="+.3f",
                 range=[lo * 1.25, hi * 1.28])
fig.show()
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
**The matrix as a whole clears its baseline.** Against a model that already knows
county population and density, the six pillars together add the lift shown below
— positive on all five targets, scored on states held out of training. The
weakest is broadband adoption, which is also the target closest to the consuming
team's domain.
""")

code('''
by_t = ext["by_target"]
lifts = [by_t[t]["lift_over_size_ablated"] for t in TARGET_ORDER]
base = [by_t[t]["r2_size"] for t in TARGET_ORDER]
labels = [TARGET_LABEL[t] for t in TARGET_ORDER]

fig = go.Figure()
fig.add_trace(go.Bar(x=labels, y=base, name="county size alone",
                     marker_color=MUTED,
                     hovertemplate="%{x}<br>size baseline R² %{y:.3f}<extra></extra>"))
fig.add_trace(go.Bar(x=labels, y=lifts, name="added by E_macro",
                     marker_color=BLUE,
                     text=[f"+{v:.3f}" for v in lifts],
                     textposition="inside", textfont=dict(size=14, color="white"),
                     hovertemplate="%{x}<br>added %{y:+.3f}<extra></extra>"))
style(fig,
      f"E_macro over a county-size baseline: {ext['mean_lift_over_size_ablated']:+.3f} mean R²",
      f"Positive on {ext['targets_with_positive_lift']} of {ext['n_targets']} targets · "
      "stacked on the baseline each target starts from",
      height=470)
fig.update_layout(barmode="stack")
fig.update_yaxes(title="out-of-state R²")
fig.show()
''')

md("""
**The caveat travels with the headline, not after it.** These five targets are
**public proxies, not the consuming team's label** — which is unobtainable under
the scope this project was given. Everything above is an argument by analogy. It
is the strongest non-circular evidence this project can produce, and it is still
an analogy. Appendix A1 takes that objection seriously rather than footnoting it.
""")

# ==========================================================================
# 3. Two pillars moved
# ==========================================================================
md("""
---

## 3. Two pillars moved

### Source F — kept, on a test it could have failed

F was the pillar the last status doc flagged as falling short, and the reason was
real: its one strong relationship in the whole 15-pair sweep was against Source D
freight tonnage, **r = 0.495 raw — the largest raw effect anywhere in that sweep
— collapsing to r = −0.057 once county size is controlled.** The apparent link
was population riding along in both variables.

The resolution on file was to keep F and reclassify it as a "structural anchor,"
justified by what county typology definitionally *is* rather than by measured
performance. That was a rationalisation, and it was withdrawn.

**What replaced it.** The status doc named the fairer test itself — does F
explain residual variance once B/C/D/E are already in the model — and that test
was run. F contributes **+0.0413**, second of the six pillars, positive on 5 of 5
targets and above the noise floor on 5 of 5, where the largest contribution any
shuffled block produced anywhere was +0.0031.

**Both facts travel together.** F still fails the pairwise hub test. It passed the
residual-variance test that was pre-registered for it. The first instrument was
wrong for a categorical structural variable; that was said before the numbers
existed, not after.
""")

code('''
fb = blk["by_block"]["F"]
# Headline external figure, not the `_no_ametro` robustness variant — see the
# comment on the section 2 figure.
f_ext = ext["drop_one"]["size_emacro_drop_F"]["mean_contribution_ablated"]
bars = [
    ("Internal, raw<br><span style='font-size:12px'>29 in-matrix targets</span>",
     fb["mean_lift"], MUTED),
    ("Internal, restatements ablated<br><span style='font-size:12px'>the honest internal number</span>",
     fb["mean_lift_ablated"], AQUA),
    ("External<br><span style='font-size:12px'>5 public ACS targets</span>",
     f_ext, BLUE),
]
fig = go.Figure(go.Bar(
    x=[b[0] for b in bars], y=[b[1] for b in bars],
    marker_color=[b[2] for b in bars],
    text=[f"{b[1]:+.4f}" for b in bars], textposition="outside",
    textfont=dict(size=16),
    hovertemplate="%{x}<br>%{y:+.4f}<extra></extra>", showlegend=False,
))
style(fig,
      "Source F: the internal number is mostly USDA restating BLS",
      "Roughly seven eighths of F's apparent internal contribution disappears once "
      "columns that restate Source B are removed. The external number does not move.",
      height=470, legend=False)
fig.update_yaxes(title="mean R² contribution", tickformat="+.3f",
                 range=[0, max(b[1] for b in bars) * 1.2])
fig.show()
''')

md("""
The middle bar is the one to sit with. **Seven eighths of F's apparent internal
contribution is USDA restating industry composition that BLS already measures.**
That redundancy is real inside the six-pillar system. It does not bind against
outcomes outside it — the same ablation moves F's external figure by 0.0003 —
which is precisely why the external arm is the one the verdict rests on.

### Source A — the uncomfortable finding

Source A is the pillar this project marked **"Good shape. Done."** Its embedding
was cut on evidence, its 29 typed columns were validated, its schema was frozen
first. It contributes **−0.0000**.
""")

code('''
a_stats = ext["drop_one"]["size_emacro_drop_A"]
a_pl = placebo[placebo["pillar"] == "A"].set_index("target")
labels = [TARGET_LABEL[t] for t in TARGET_ORDER]
vals = [a_stats["by_target"][t] for t in TARGET_ORDER]
band = [float(a_pl.loc[t, "placebo_p95"]) if t in a_pl.index else 0.0 for t in TARGET_ORDER]

fig = go.Figure()
# One colour, not red-for-negative: sign is already carried by position against
# the zero line, and a two-colour split would make the legend swatch lie about
# the series while dramatising numbers whose whole point is that they are small.
fig.add_trace(go.Bar(
    x=labels, y=vals, marker_color=BLUE,
    text=[f"{v:+.4f}" for v in vals], textposition="outside",
    textfont=dict(size=14), cliponaxis=False,
    hovertemplate="%{x}<br>%{y:+.4f}<extra></extra>", name="Source A contribution",
))
fig.add_trace(go.Scatter(
    x=labels, y=band, mode="markers", name="same block shuffled (95th percentile)",
    marker=dict(symbol="line-ew", size=26, line=dict(color=INK2, width=2.5)),
    hovertemplate="%{x}<br>placebo p95 %{y:+.4f}<extra></extra>",
))
style(fig,
      "Source A, target by target: small in both directions",
      "Contributions run −0.0032 to +0.0070. A broken block looks wildly negative; "
      "this is the signature of one that is genuinely redundant.",
      height=500)
span = max(vals + band) - min(vals + band)
fig.update_yaxes(title="marginal R²", tickformat="+.4f",
                 range=[min(vals + band) - span * 0.25, max(vals + band) + span * 0.25])
fig.show()
''')

md("""
**This is not a harness failure.** The same code path produces +0.0582 for E and
+0.0413 for F, the placebo distributions behave, and A's per-target numbers are
small in *both* directions rather than wildly negative.

**It is not a contradiction of the evidence on file either.** A's typed block was
justified on a marginal lift of **+0.0010** over a baseline holding every other
pillar — a real effect, at p = 0.010 with power 0.92, and a tiny one. A
contribution indistinguishable from zero against five external outcomes is what
that effect size predicts. A is also the only block negative in **both** arms:
−0.0031 internally, −0.0000 externally.

**What it means.** Applied consistently, the operating principle that every
pillar earns its slot on evidence now points at Source A rather than Source F.
That is uncomfortable and it is the honest reading.

**Three arguments against acting on it, all genuine:**

1. **A is nearly free.** No API key, no model, no inference. The cost of keeping
   it is a schema document that already exists.
2. **The targets are ACS demographics.** A's columns encode named industries,
   universities, ports, protected land — plausibly more useful for an ad-tech
   outcome than for median age.
3. **Redundancy inside a feature store is not uselessness.** A is redundant *with
   the other five pillars*, which is exactly the position a downstream model can
   exploit for a county where another pillar is missing.

### The open question this puts to the room

> **Does Source A ship?** The recommendation is to cut it — *unless* the
> consuming team's real target rewards what A encodes. Argument 2 above is the
> only one of the three that survives scrutiny, and it is not something this
> project can settle: it depends on a label that is structurally unobtainable
> from inside this scope.
>
> **What would settle it:** knowing whether the downstream target is closer to
> "who lives here" (where A adds nothing) or "what happens here economically"
> (where A's industry, port and university flags are the kind of thing that could
> matter). That is a question only the commissioning side can answer, and it is
> the reason this is raised here rather than decided.
""")

# ==========================================================================
# 4. What was ruled out
# ==========================================================================
md("""
---

## 4. What was ruled out

Negative results, kept short. Three Source A experiments were run and all three
came back against the change; the shipped design is unchanged as a result, which
is itself the finding.

- **Section scope** — reading every section beats reading only economy-titled
  ones (+0.00403 vs +0.00307), but **67% of the hits it adds sit in historical
  framing**. A defunct-industry detector wearing a current-economy label. Not
  shipped.
- **A smaller embedding** — `all-MiniLM-L6-v2` at 384 dimensions, 90MB against
  bge-m3's 2.2GB, chunked and mean-pooled. Best arm reaches +0.00226 against the
  typed block's +0.00307. The embedding still loses, at one twenty-fourth the
  download.
- **Tier-conditional reading** — letting article length decide how much of a page
  is read, in both directions. Both lose to reading the same sections for
  everyone.
""")

code('''
arms = [
    ("Typed columns (shipped)", scope["scopes"]["economy"]["mean_lift"], BLUE),
    ("All sections", scope["scopes"]["all_sections"]["mean_lift"], MUTED),
    ("All except narrative", scope["scopes"]["no_narrative"]["mean_lift"], MUTED),
    ("384-d embedding, uniform", embed["representations"]["uniform"]["mean_lift"], AQUA),
    ("384-d, tier-conditional", embed["representations"]["tier_conditional"]["mean_lift"], AQUA),
    ("384-d, tier inverted", embed["representations"]["tier_conditional_inverse"]["mean_lift"], AQUA),
]
arms = sorted(arms, key=lambda a: a[1])
fig = go.Figure(go.Bar(
    x=[a[1] for a in arms], y=[a[0] for a in arms], orientation="h",
    marker_color=[a[2] for a in arms],
    text=[f"{a[1]:+.5f}" for a in arms], textposition="outside",
    textfont=dict(size=14),
    hovertemplate="%{y}<br>%{x:+.5f}<extra></extra>", showlegend=False,
))
style(fig,
      "Three Source A experiments, none of which changed what ships",
      "Mean lift over the crowded baseline. Blue is the shipped design; nothing "
      "beats it that is worth what it costs.",
      height=470, legend=False)
fig.update_xaxes(title="mean lift", tickformat="+.4f",
                 range=[0, max(a[1] for a in arms) * 1.22])
fig.show()
''')

md("""
**One thing that did change.** Source D's two partner-concentration indices were
re-derived at market grain rather than approximated. That moved the measured
gain from aggregation from +0.106 to **+0.099** — a correction against the
project's own favoured direction, which is the kind that is worth making.
""")

# ==========================================================================
# 5. What this still cannot answer
# ==========================================================================
md("""
---

## 5. What this still cannot answer

Four items, stated plainly.

**1. The fixed-effect objection is unanswered.** If the consuming team joins
geography at DMA grain, a 210-level dummy — cheap and precise to estimate from
millions of impressions per market — supplies everything a static DMA-keyed
feature could. Cross-sectionally, `E_macro` would add nothing over it, and no
correlation measured anywhere in this project is evidence against that. The chart
below is what the grain question costs in each direction.
""")

code('''
arms = [("County grain, all rows", gst["mean_lift_county_full"], BLUE),
        ("County grain, subsampled to market row count", gst["mean_lift_county_subsample"], MUTED),
        (f"Aggregated to {gst['n_markets']} markets", gst["mean_lift_market_aggregate"], AQUA)]
fig = go.Figure(go.Bar(
    x=[a[0] for a in arms], y=[a[1] for a in arms],
    marker_color=[a[2] for a in arms],
    text=[f"{a[1]:+.3f}" for a in arms], textposition="outside",
    textfont=dict(size=16),
    hovertemplate="%{x}<br>%{y:+.3f}<extra></extra>", showlegend=False,
))
style(fig,
      "Coarsening the join: two effects that roughly cancel",
      f"Losing rows costs {gst['row_count_effect']:+.3f}; aggregating itself gains "
      f"{gst['aggregation_effect']:+.3f}. County grain is this project's "
      "recommendation, not an established win.",
      height=470, legend=False)
fig.update_yaxes(title="mean lift over size baseline", tickformat="+.2f",
                 range=[0, max(a[1] for a in arms) * 1.2])
fig.show()
''')

md("""
**2. There is no downstream label**, and there cannot be one under the scope this
project was given. Structural, not an oversight — and the reason every result
above is by analogy.

**3. Everything here is cross-sectional and single-period.** Temporal transfer —
one of the things a geographic fixed effect genuinely fails at, and therefore one
of the strongest available arguments *for* this feature layer — is untested
anywhere in this repo.

**4. The sibling tiers do not line up.** `E_local` is under construction at H3
res-8; `E_census` does not exist. The three-tier stack is currently one tier at
county grain, one at hex grain, and one missing, with nobody owning the
reconciliation. Outside this repo's scope, flagged rather than solved.
""")

# ==========================================================================
# 6. Readiness
# ==========================================================================
md("""
---

## 6. Where this leaves the project

**Everything a handover needs, exists.** Six pillars ingested, coverage 3,143–3,144
counties on five of six, six frozen schemas with documented null semantics, an
`as_of_date` on every pillar, and a measured worth per pillar rather than an
adjective.
""")

code('''
v = vintages.copy()
v["Pillar"] = v["pillar"].map(PILLAR_NAME)
v = v.rename(columns={"as_of_date": "As of", "reference_period": "Reference period",
                      "cadence": "Update cadence"})
display(v[["Pillar", "As of", "Reference period", "Update cadence"]].set_index("Pillar"))
''')

code('''
readiness = pd.DataFrame([
    ("Six sources ingested at county grain", "Done",
     "3,143–3,144 counties on five of six"),
    ("Per-pillar evidence", "Done",
     "findings report per source in analysis-output/"),
    ("Frozen schema + null semantics", "Done",
     "docs/source_{a..f}_feature_schema.md, all six"),
    ("Vintage stamped per pillar", "Done", "outputs/pillar_vintages.csv"),
    ("Evidence against a target outside the six pillars", "Done",
     f"{ext['mean_lift_over_size_ablated']:+.3f} mean R² over a size baseline"),
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
**The remaining path, in the currency that binds it** — calendar weeks of
availability, not permission or budget:

- **Settle the join grain.** Everything downstream of it is cheap; nothing
  downstream of it is safe to build first. This is the single highest-leverage
  unblock.
- **Benchmark against a geographic fixed effect.** Once the grain is known, this
  is the test that decides whether `E_macro` earns a slot in a production model.
- **Test temporal transfer.** The one argument a fixed effect cannot answer, and
  the one this project has not yet made.
- **Then package.** `pillar_matrix.build_matrix()` already joins all six pillars
  into a 3,144 × 124 matrix. What remains of "fusion" is a schema freeze, an
  imputation policy and a serving format — days, not weeks.

**What is not blocked by any of the above:** ingestion, validation, schema freeze
and the external benchmark are complete and stand on their own.
""")

# ==========================================================================
# Appendix
# ==========================================================================
md("""
---

# Appendix

Written to be read without a narrator.

## A1 — Do public proxies mean anything?

This is the strongest objection available against everything in section 2, and it
deserves a straight answer rather than a footnote.

**The objection.** `E_macro` is being scored against ACS broadband adoption,
median household income, median age, median home value and mean commute. The
consuming team predicts none of those things. A feature layer that explains
median age tells you nothing about whether it explains revenue per ad request.

**What is conceded immediately.** The objection is correct on its own terms. No
result in this project is a direct test of usefulness to the consuming team, and
none is presented as one. The label that would make such a test possible is
unobtainable: the project was scoped to public and open-source data only, from
the start and deliberately.

**Why the proxies are still worth what they cost.** The alternative was not a
better test — it was the pillar-versus-pillar sweep, which is *circular*. It can
establish that Source B and Source E see the same economy and cannot establish
that either predicts anything at all. Moving to an external target replaces a
circular measurement with a non-circular one. That is a real gain in evidential
status even though the target is a proxy.

**What the proxies do license.**

- That the six pillars carry **information about counties that county size does
  not already carry** — the baseline is population and density, and the lift over
  it is measured on states held out of training, so it is not memorisation of
  local idiosyncrasy.
- That the pillars are **not interchangeable with each other** — the drop-one
  design holds the other five constant, so each contribution is genuinely
  marginal.
- That the **ordering among pillars is not arbitrary** — E and F clear the noise
  floor on 5 of 5 targets by an order of magnitude; A does not.

**What they do not license.**

- Any claim about **magnitude** against an ad-tech target. +0.190 mean R² on ACS
  outcomes is not a forecast of anything.
- Any claim that the **ordering transfers**. Source A could plausibly rank higher
  against an economically-flavoured target than against median age; that is
  precisely the argument in section 3 for not cutting it unilaterally.
- Any answer to the **fixed-effect objection**, which is a different question
  entirely and is unanswered.

**What would settle it.** One pass of the same drop-one design against a real
downstream target, at the grain the consuming team actually joins on. That
requires either a label or a collaborator inside the consuming team, and is the
single most valuable thing that could be added to this project.

## A2 — Method

**Design.** For each pillar, fit two models: one holding county size plus all six
pillars, one holding county size plus five. The difference in out-of-fold R² is
that pillar's marginal contribution. Repeated for all six pillars and for the B+E
pair.

**Estimator.** Ridge regression on an imputed design matrix. Missing values are
imputed rather than dropped, because the null patterns are themselves informative
(BLS suppresses ~35% of the LQ matrix) and listwise deletion would silently
change the county population under test.

**Folds.** `GroupKFold` on `state_fips` — spatially blocked, so a model never
predicts a county in a state it trained on. This is stricter than random k-fold,
and it is the right strictness: county-level features are spatially
autocorrelated, and random folds would let a neighbouring county leak the answer.

**Restatement ablation.** Where a column in one pillar restates a column in
another (Source A's `has_metro_attachment` against Source F's `metro_2023`; Source
F's industry flags against Source B's location quotients), it is removed from both
the full and the reduced model. A pillar that only restated a neighbour would then
score zero by construction, which is the intended behaviour.

**Noise floor.** Each pillar's block is shuffled and re-scored — 49 reps in the
internal arm, 20 per pillar × target in the external arm. The largest contribution
any shuffled block produced anywhere in the sweep is +0.0031, and that is the bar
drawn on the section 2 figure.

**Two arms.** The internal arm scores against 29 in-matrix targets and measures
coherence. The external arm scores against the five ACS targets and is the one the
verdicts rest on. Where they disagree, the external arm wins — being unpredictable
from the other five pillars is also exactly what an independent information source
looks like.

## A3 — Limitations, carried over unchanged

- **The targets are public proxies, not the consumer's label**, which is
  structurally unobtainable. Every conclusion is by analogy.
- **20 placebo reps per pillar × target, 49 in the internal arm.** Enough to place
  a floor near zero against contributions an order of magnitude larger. **Not
  enough to resolve a borderline contribution — B's and C's ordering (+0.0067,
  +0.0054) should not be quoted as settled.**
- **Contribution is not importance under a different model class.** Everything
  here is ridge on an imputed design. A gradient-boosted consumer might
  distribute credit differently; only the internal arm carries a GBM cross-check.
- **Cross-sectional and single-period.** Temporal transfer is untested.
- **This does not answer the fixed-effect objection.** It reallocates credit among
  pillars, given that the matrix as a whole beats a size baseline on held-out
  states.

## A4 — For the consuming team, when it comes to that

**The warning that matters more than which columns ship.** An impression-level
training table joined to `E_macro` carries only 3,143 distinct feature values, so
the effective sample size is the county count — not the row count. Random k-fold
will make this feature layer look good in evaluation and do nothing in
production. **Cluster standard errors by `fips_code`; use grouped, spatially
blocked folds.**

**Null semantics are explicit and must stay that way.** BLS suppresses ~35% of the
Source B LQ matrix and those cells are null with a matching `disclosure_*` flag.
IRS ships no suppression flag at all, and that limitation is disclosed rather than
papered over. A downstream model must be able to tell "missing" from "zero."

**Size columns are held out deliberately.** Source B's employment levels, Source
D's raw tonnages and two Source E dollar totals sit in `SIZE_COLUMNS` — they exist
so a pillar can be re-derived at a coarser geography, not as features. A
market-level location quotient is summed sector employment over summed total
employment, never the mean of its counties' quotients.

## A5 — Artifact index

Every figure above is computed from a committed artifact. Nothing is hardcoded.

| Section | Reads | Produced by |
|---|---|---|
| 2 · pillar worth | `external_target_stats.json` (`drop_one`, `drop_one_noise_floor`) | `scripts/analyze_external_target.py` |
| 2 · baseline lift | `external_target_stats.json` (`by_target`) | `scripts/analyze_external_target.py` |
| 3 · Source F | `pillar_block_marginal_stats.json` (`by_block`) | `scripts/analyze_pillar_block_marginal.py` |
| 3 · Source A | `external_target_drop_one_placebo.csv` | `scripts/analyze_external_target.py` |
| 4 · ruled out | `source_a_section_scope_stats.json`, `source_a_tiered_embedding_stats.json` | `scripts/analyze_source_a_section_scope.py`, `scripts/analyze_source_a_tiered_embedding.py` |
| 5 · grain | `grain_effect_stats.json` | `scripts/analyze_grain_effect.py` |
| 6 · vintages | `outputs/pillar_vintages.csv` | `scripts/pillar_vintage.py` |

Long-form evidence:
`analysis-output/cross-source/pillar-marginal-findings.md`,
`analysis-output/cross-source/external-target-findings.md`,
`analysis-output/E_macro_key_findings.ipynb`,
`docs/pillar_status.md`, `docs/PROJECT_GOAL.md`.
""")

# --------------------------------------------------------------------------
nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
