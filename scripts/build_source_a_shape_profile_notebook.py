"""Generate analysis-output/source-a/source_a_shape_profile_round.ipynb.

How much can be pulled from the shape of a county's Wikipedia article, reported
in three framings at once so no number can be read in a framing its reader
cannot see. That was round one's actual failure: §23's +0.00269 was correct and
still misleading, because the linear-in-logs baseline it was measured against
was not visible in the sentence that quoted it.

**The size diagnostic runs before the arms, deliberately.** §23 closed on an open
problem -- the per-column size audit clears each column individually and clears
nothing else, because the dependence is joint across the block. Predicting size
*from* shape measures that joint channel in one number, and the reader should
have it before seeing what the block scores, not after.

**A boost lift is never shown without its floor.** HistGradientBoosting scores
every arm's *baseline* comparison worse than ridge does on this dataset -- the
boost floor itself sits at roughly -0.078 (linear) / -0.071 (flexible) -- so a
raw boost lift is mostly that offset, not a verdict on the arm. Every boost
number here is shown beside `vs_boost_floor`, which is the comparison that
actually answers whether the arm adds anything under that learner.

Every number is read from a committed artifact. Matplotlib, not plotly: plotly's
mimetype output needs a JupyterLab extension and renders as blank space without
it.

Build and execute:

    uv run scripts/build_source_a_shape_profile_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO: Path = Path(__file__).resolve().parent.parent
OUT: Path = REPO / "analysis-output" / "source-a" / "source_a_shape_profile_round.ipynb"

cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: Cell body; surrounding blank lines are trimmed.
    """
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell.

    Args:
        text: Cell body; surrounding blank lines are trimmed.
    """
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Source A — How Much Is In the Shape of an Article

**The question:** how much of a county can be recovered from the *shape* of its
Wikipedia article — how many sections, how long, which ones, in what order, how
far from the county template, and what the characters look like — without
reading a word for meaning?

**Why every number here carries three readings.** Round one reported +0.00269
mean lift for article shape and it was arithmetically correct. It was also
misleading, because the baseline it was measured against controlled for county
size *linearly, in logs*, and an information-free curve on those same size
columns scores +0.01748 through the identical protocol. The number was right;
its framing was invisible. So nothing below is reported in one framing:

- **`r2_alone`** — the block as the only predictor, no controls. How much of a
  county is recoverable from article shape, size and geography included.
- **`lift` (linear)** — over the linear size-plus-state baseline. Comparable to
  §13–§23.
- **`lift` (flexible)** — over the curvature-augmented baseline. The strict
  reading.

**Why a boost lift additionally carries `vs_boost_floor`.** The two learners
are not on equal footing here: the ridge path imputes missing predictor values
inside its pipeline, while the boosting path (HistGradientBoosting) handles
`NaN` natively and never imputes. That difference alone moves boost's own
*baseline* score, so a boost arm's raw lift is dominated by that shift, not by
what the arm adds. `vs_boost_floor` compares an arm to the boost baseline run
through the identical boost pipeline, which is the reading that isolates the
arm's own contribution under that learner.

Two arms are not findings and are labelled so wherever they appear:
`shape_v1` re-scores round one's block as a regression check, and
`size_nonlinear` is the information-free null control that prices the unit.
""")

code("""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path.cwd()
while not (REPO / "data").exists() and REPO != REPO.parent:
    REPO = REPO.parent
DATA, ANALYSIS, OUTPUTS = REPO / "data", REPO / "analysis-output", REPO / "outputs"
SOURCE_A = ANALYSIS / "source-a"

mpl.rcParams.update({"figure.figsize": (11, 5.5), "figure.dpi": 110, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

profile = pd.read_parquet(DATA / "source_a_shape_profile.parquet")
structure = pd.read_parquet(DATA / "source_a_structure_features.parquet")
feature_stats = json.loads((SOURCE_A / "source_a_shape_profile_stats.json").read_text())
stats = json.loads((SOURCE_A / "source_a_shape_profile_stats_scoring.json").read_text())
scores = pd.read_csv(OUTPUTS / "source_a_shape_profile_scores.csv")
by_pillar = pd.read_csv(OUTPUTS / "source_a_shape_profile_by_pillar.csv")

profile_cols = [c for c in profile.columns if c != "fips_code"]
v1_cols = [c for c in structure.columns if c != "fips_code"]

print(f"{feature_stats['n_counties']:,} counties")
print(f"shape_v1 {stats['n_shape_v1_features']} cols + profile {stats['n_shape_profile_features']} "
      f"= shape_v2 {stats['n_shape_v2_features']} cols, vs typed {stats['n_typed_features']}")
print(f"{stats['n_targets']} targets | modal skeleton: {len(feature_stats['modal_title_set'])} titles")
""")

md("""
## Part one — the four new families

Round one measured how many sections an article has, how long they are, and
which titles are present. These four families measure things it never looked at.
""")

code("""
POSITION_ABSENT = feature_stats["position_absent_sentinel"]
pos_cols = [c for c in profile_cols if c.startswith("pos_") and not c.startswith("pos_first_")
            and c != "pos_longest_section"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

present = profile[pos_cols].replace(POSITION_ABSENT, np.nan)
axes[0, 0].hist(present.mean(axis=1).dropna(), bins=40, color="#3b6ea5")
axes[0, 0].set_title("Mean position of a county's common sections")
axes[0, 0].set_xlabel("normalized position (0 = top of article)")

absent_share = (profile[pos_cols] == POSITION_ABSENT).mean().sort_values()
axes[0, 1].hist(absent_share, bins=30, color="#3b6ea5")
axes[0, 1].set_title("How often each common section is simply absent")
axes[0, 1].set_xlabel("share of counties lacking it")

axes[1, 0].hist(profile["template_jaccard"], bins=40, color="#3b6ea5")
axes[1, 0].set_title("Template conformity (Jaccard vs the modal skeleton)")
axes[1, 0].set_xlabel("1.0 = exactly the house template")

axes[1, 1].hist(profile["digit_density"], bins=40, color="#3b6ea5")
axes[1, 1].set_title("Digit density of the article body")
axes[1, 1].set_xlabel("digits / characters")

fig.tight_layout()
plt.show()

print(f"median template conformity {profile['template_jaccard'].median():.3f}; "
      f"median digit density {profile['digit_density'].median():.3f}")
print(f"history precedes economy in {profile['history_before_economy'].mean():.1%} of counties")
""")

code("""
merged = structure.merge(profile, on="fips_code", validate="one_to_one")
cross = pd.Series(
    {c: merged[v1_cols].corrwith(merged[c]).abs().max() for c in profile_cols}
).sort_values()

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(cross, bins=30, color="#3b6ea5")
ax.set_title("How much each new column duplicates round one's block")
ax.set_xlabel("largest |r| against any round-one column")
ax.set_ylabel("columns")
fig.tight_layout()
plt.show()

print(f"{(cross < 0.5).sum()} of {len(cross)} new columns stay under |r| = 0.5 against everything "
      "round one already had.")
print("Most duplicated:", ", ".join(cross.tail(3).index))
print("Most novel:", ", ".join(cross.head(3).index))
""")

md("""
## Part two — how much of county size is *in* the block

§23 closed on an open problem. Its per-column size audit came back clean — no
single structural column is a hidden curved size proxy — while the block as a
whole demonstrably carried size. Both are true: the dependence is **joint across
columns**, and no per-column statistic can see it.

So invert the question. Instead of asking whether each column looks like size,
ask how much of county size the whole block can reconstruct. One number, no
statistics argument, and it bounds how much of any lift below could be size in
disguise.
""")

code("""
recovery = pd.DataFrame(stats["size_recoverability"]).T

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(recovery.columns))
width = 0.38
for offset, block, color in zip((-width / 2, width / 2), recovery.index, ("#7a9cc6", "#2f5d8a")):
    ax.bar(x + offset, recovery.loc[block], width, label=block, color=color)
ax.set_xticks(x)
ax.set_xticklabels(recovery.columns, rotation=30, ha="right")
ax.set_ylabel("out-of-fold R²")
ax.set_title("How much of county size the shape block reconstructs")
ax.legend()
fig.tight_layout()
plt.show()

display(recovery.round(4))

peak = recovery.max().max()
where = recovery.stack().idxmax()
print(f"Peak: {peak:.3f} — {where[0]} reconstructing {where[1]}.")
print(f"So article shape encodes county size to R² = {peak:.3f}. Any lift below is what "
      "remains after a control has already removed size — read it with that in mind.")
""")

md("""
## Part three — the arms

Five arms, two learners, three framings, and no arm reported in fewer than all
three. `shape_v1` is a regression check on round one's block; `size_nonlinear` is
the information-free null control. Neither is a finding.

**Reading the boost rows.** `lift_linear` and `lift_flexible` are negative for
every boost arm in the table below — that is the boost baseline offset
(`boost_floor`, printed after the table), not the arms failing.
`p_linear_vs_floor` and `p_flexible_vs_floor` are the readings that actually
test an arm under the boost learner: each compares the arm to the boost
baseline run through the same pipeline, not to the ridge-baseline zero line
the raw lift implies.
""")

code("""
def _arm_row(name: str, a: dict) -> dict:
    linear, flexible = a["linear"], a["flexible"]
    row = {
        "arm": name,
        "learner": a["learner"],
        "r2_alone": a["mean_r2_alone"],
        "lift_linear": linear["mean_lift"],
        "p_linear_vs_baseline": linear["vs_baseline"]["wilcoxon_p"],
        "lift_flexible": flexible["mean_lift"],
        "p_flexible_vs_baseline": flexible["vs_baseline"]["wilcoxon_p"],
    }
    if "vs_arm" in linear:
        row["vs_arm"] = linear["vs_arm"]["compared_against"]
        row["p_linear_vs_arm"] = linear["vs_arm"]["wilcoxon_p"]
        row["p_flexible_vs_arm"] = flexible["vs_arm"]["wilcoxon_p"]
    if "vs_boost_floor" in linear:
        row["p_linear_vs_floor"] = linear["vs_boost_floor"]["wilcoxon_p"]
        row["p_flexible_vs_floor"] = flexible["vs_boost_floor"]["wilcoxon_p"]
    return row


arms = pd.DataFrame([_arm_row(name, a) for name, a in stats["arms"].items()])
display(arms.round(5))

floor = stats["boost_floor"]
print(f"boost_floor: linear {floor['linear']['mean_lift']:+.5f}, "
      f"flexible {floor['flexible']['mean_lift']:+.5f} "
      "-- every boost lift above is measured against this, not against zero.")
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, column, title in zip(
    axes,
    ("r2_alone", "lift_linear", "lift_flexible"),
    ("R² alone (no controls)", "Lift over linear baseline", "Lift over flexible baseline"),
):
    for offset, learner, color in zip((-0.2, 0.2), ("ridge", "boost"), ("#7a9cc6", "#2f5d8a")):
        subset = arms[arms["learner"] == learner]
        positions = np.arange(len(subset)) + offset
        bars = ax.bar(positions, subset[column], 0.4, label=learner, color=color)
        for bar, arm_name in zip(bars, subset["arm"]):
            if "size_nonlinear" in arm_name or "shape_v1" in arm_name:
                bar.set_hatch("//")
    if column != "r2_alone":
        floor_key = "linear" if column == "lift_linear" else "flexible"
        ax.axhline(stats["boost_floor"][floor_key]["mean_lift"], color="#2f5d8a", lw=1,
                   ls="--", label="boost floor")
    ax.set_xticks(np.arange(len(arms[arms["learner"] == "ridge"])))
    ax.set_xticklabels(
        [n.replace("_ridge", "") for n in arms[arms["learner"] == "ridge"]["arm"]],
        rotation=35, ha="right",
    )
    ax.axhline(0, color="#333", lw=1)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Hatched bars are not findings: shape_v1 is a regression check, "
             "size_nonlinear is the null control. Dashed line is the boost floor -- "
             "boost bars must be read against it, not against zero.")
fig.tight_layout()
plt.show()
""")

md("""
### Two questions `shape_v2` answers differently

`shape_v2_ridge` under the linear framing lifts `+0.00260` over the baseline,
`vs_baseline p = 0.0013` — a real, if small, lift over the baseline that never
saw article shape at all. Compared against `shape_v1` instead, the picture
changes: `vs_arm p = 0.4515`, no distinguishable difference. Those are two
different questions — "does shape add anything over no-shape?" (yes) and "does
the round's four new families add anything over round one's block alone?" (no,
not detectably) — and the cell below prints both so neither is quoted without
the other in view.
""")

code("""
v2 = stats["arms"]["shape_v2_ridge"]
v1_regression_check = stats["arms"]["shape_v1_ridge"]
print(f"shape_v2_ridge linear: mean_lift={v2['linear']['mean_lift']:+.5f}, "
      f"vs_baseline p={v2['linear']['vs_baseline']['wilcoxon_p']:.4f}, "
      f"vs_arm({v2['linear']['vs_arm']['compared_against']}) "
      f"p={v2['linear']['vs_arm']['wilcoxon_p']:.4f}")
print(f"shape_v1_ridge linear (for comparison): mean_lift={v1_regression_check['linear']['mean_lift']:+.5f}, "
      f"vs_baseline p={v1_regression_check['linear']['vs_baseline']['wilcoxon_p']:.4f}")
""")

md("""
### Per pillar, because the aggregate is a property of the basket

Twenty of the twenty-eight targets are one QCEW table, so a basket-wide mean is
71% one pillar. Reading it as a breadth claim is a mistake this project has made
before. The table below carries every arm's boost columns as raw lifts -- read
them against the printed floor beneath the table, not against zero, for the
same reason the arms section does. The chart that follows is restricted to the
ridge columns: boost has no *per-pillar* floor to read its lifts against (the
floor below was measured on the whole basket, not decomposed by pillar), so a
per-pillar boost bar would repeat exactly the framing mistake `vs_boost_floor`
exists to avoid.
""")

code("""
display(by_pillar.round(5))

floor = stats["boost_floor"]
print(f"boost_floor (whole basket, not decomposed by pillar): linear {floor['linear']['mean_lift']:+.5f}, "
      f"flexible {floor['flexible']['mean_lift']:+.5f} "
      "-- read every _boost column above against this floor, not against zero.")

fig, ax = plt.subplots(figsize=(11, 4.5))
keys = [c for c in by_pillar.columns if c not in ("pillar", "n_targets") and not c.endswith("_boost")]
x = np.arange(len(by_pillar))
width = 0.8 / len(keys)
for i, key in enumerate(keys):
    ax.bar(x + i * width - 0.4, by_pillar[key], width, label=key)
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\\n({n})" for p, n in zip(by_pillar["pillar"], by_pillar["n_targets"])])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean lift by owning pillar (ridge, linear baseline)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

md("""
## Part four — where the ceiling is

The targets article shape predicts best on its own, and what survives as the
control tightens. The gap between the first column and the last is the answer to
"how much of this is really about the article."
""")

code("""
ceiling = scores[[
    "pillar", "label", "n",
    "r2_alone_shape_v2", "r2_alone_shape_v2_boost",
    "lift_shape_v2", "lift_shape_v2_flexbase",
]].head(12)
display(ceiling.round(4))

fig, ax = plt.subplots(figsize=(12, 5))
top = scores.head(12).iloc[::-1]
ax.barh(top["label"], top["r2_alone_shape_v2"], color="#7a9cc6", label="R² alone")
ax.barh(top["label"], top["lift_shape_v2_flexbase"], color="#2f5d8a", label="lift, flexible baseline")
ax.set_title("Best targets: raw predictive power vs what survives the strict control")
ax.legend()
fig.tight_layout()
plt.show()

print(f"Mean R² alone across all {stats['n_targets']} targets: "
      f"{scores['r2_alone_shape_v2'].mean():.4f}")
print(f"Mean lift over the flexible baseline: {scores['lift_shape_v2_flexbase'].mean():+.5f}")
""")


def main() -> None:
    """Write the notebook and execute it in place."""
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(OUT))
    print(f"wrote {OUT.relative_to(REPO)} ({len(cells)} cells)")

    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace", str(OUT),
        ],
        check=True,
    )
    print(f"executed {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
