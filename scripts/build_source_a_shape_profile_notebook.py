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

**A boost lift is never shown without its floor.** Both learners are differenced
against the *same* OLS baseline -- there is no separate boost baseline anywhere
in this round -- so a boost arm's negative lift is not a baseline offset. It is
the boosting estimator's own overfitting cost on the residual target, paid where
`RidgeCV` shrinks toward zero and `HistGradientBoosting` does not. That cost
scales with block width, so every arm is priced against an information-free
Gaussian block of *its own width*, scored through the identical boost path.
`vs_boost_floor` is that comparison, and it is the reading that answers whether
an arm adds anything under this learner.

**Both flexible denominators are quoted, always.** The curvature-augmented
baseline degrades on 6 of the 28 targets, so "lift over the flexible baseline"
has two honest denominators -- all 28, and the 22 the baseline did not degrade --
and they disagree on significance. §23.6 forbids quoting either alone.

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

**Why no number here is reported in one framing.** Round one reported +0.00269
mean lift for article shape and it was arithmetically correct. It was also
misleading, because the baseline it was measured against controlled for county
size *linearly, in logs*, and an information-free curve on those same size
columns scores +0.01748 through the identical protocol. The number was right;
its framing was invisible. So:

- **`r2_alone`** — the block as the only predictor, no controls. How much of a
  county is recoverable from article shape, size and geography included.
- **`lift` (linear)** — over the linear size-plus-state baseline. Comparable to
  §13–§23.
- **`lift` (flexible)** — over the curvature-augmented baseline. The strict
  reading, and it comes with **two denominators**: the flexible baseline
  degrades on 6 of the 28 targets, so "all 28" and "the undegraded 22" are both
  honest baskets and they disagree on significance. §23.6 forbids quoting either
  alone, so both appear everywhere a flexible number does.

**Why a boost lift additionally carries `vs_boost_floor`.** Every boost lift in
this notebook is negative, and it is worth being precise about why, because the
obvious explanation is wrong. It is *not* that the two learners are measured
against different baselines: `score_target` computes one `r2_baseline` and one
`r2_flexible` per target and subtracts those same two numbers from the ridge
arms and the boost arms alike. There is no boost baseline in this round at all.
Nor is it imputation — the ridge path does impute inside its pipeline where
HistGradientBoosting handles `NaN` natively, but `shape_v1` and `shape_v2`
contain zero missing cells and carry the offset anyway.

What the negative numbers measure is the boosting estimator's **overfitting cost
on the residual target**. Fitted to what a size-plus-state model could not
explain, boosting finds structure in fold noise where `RidgeCV`'s nested
penalty search shrinks toward zero. That cost is a function of how wide the
block is, so it cannot be read off a single number: each arm is compared to a
block of pure Gaussian noise **at that arm's own width**, run through the
identical boost path under the identical folds. `vs_boost_floor` is that
comparison, and it is the only reading here that says whether an arm carries
anything a same-width block of noise does not. ("Same-width", precisely: the
noise blocks are dense, while the `typed` blocks hold 1,930 `NaN` cells and
`size_nonlinear` 259, so the floor is matched on width and not on sparsity.)

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
has_economy = profile["pos_first_economy"] != POSITION_ABSENT
precedes = profile["history_before_economy"] == 1.0
print(f"{has_economy.sum():,} of {len(profile):,} counties ({has_economy.mean():.1%}) have an "
      "economy-bucket section at all.")
print(f"Among those {has_economy.sum():,}, a narrative section precedes it in "
      f"{precedes[has_economy].mean():.1%} ({precedes[has_economy].sum():,} counties).")
print(f"Over the whole corpus that is {precedes.mean():.1%} -- which is the column's mean, and "
      "is NOT '81% of counties lead with economy': "
      f"{(~has_economy).sum():,} counties have no economy section to lead with.")
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
every boost arm in the table below. That is not a different baseline — both
learners are differenced against the same OLS baseline — and it is not
imputation. It is what boosting costs itself by overfitting the residual target
where ridge shrinks. Since that cost grows with block width, each arm is priced
against a Gaussian noise block of its own width (`floor_width`, `floor_linear`
and `floor_flexible` in the table): `d_linear_vs_floor` / `d_flexible_vs_floor`
are how far the arm sits *above* its own floor, and `p_linear_vs_floor` /
`p_flexible_vs_floor` say whether that distance is distinguishable from zero,
with `p_flexible_undeg22_vs_floor` giving the same test on the undegraded
basket. Those are the only columns here that test a boost arm.

The floor is matched on width and **not** on missingness: the noise blocks are
dense, while the `typed` blocks carry 1,930 `NaN` cells and `size_nonlinear` 259,
and HistGradientBoosting routes `NaN` as its own category.

**Reading the flexible columns.** `lift_flexible` is the all-28 reading;
`lift_flexible_undeg22` is the same quantity over the 22 targets the flexible
baseline did not degrade. §23.6 forbids quoting either without the other, so
both are in the table and neither is the headline.
""")

code("""
def _arm_row(name: str, a: dict) -> dict:
    linear, flexible = a["linear"], a["flexible"]
    undegraded = flexible["undegraded"]
    arm_key = name.rsplit("_", 1)[0]
    row = {
        "arm": name,
        "learner": a["learner"],
        "r2_alone": a["mean_r2_alone"],
        "lift_linear": linear["mean_lift"],
        "p_linear_vs_baseline": linear["vs_baseline"]["wilcoxon_p"],
        "lift_flexible": flexible["mean_lift"],
        "p_flexible_vs_baseline": flexible["vs_baseline"]["wilcoxon_p"],
        f"lift_flexible_undeg{undegraded['n_targets']}": undegraded["mean_lift"],
        f"p_flexible_undeg{undegraded['n_targets']}": undegraded["vs_baseline"]["wilcoxon_p"],
    }
    if "vs_arm" in linear:
        row["vs_arm"] = linear["vs_arm"]["compared_against"]
        row["p_linear_vs_arm"] = linear["vs_arm"]["wilcoxon_p"]
        row["p_flexible_vs_arm"] = flexible["vs_arm"]["wilcoxon_p"]
    if "vs_boost_floor" in linear:
        row["floor_width"] = stats["boost_floor"][arm_key]["width"]
        row["floor_linear"] = stats["boost_floor"][arm_key]["linear"]["mean_lift"]
        row["d_linear_vs_floor"] = linear["vs_boost_floor"]["mean_paired_difference"]
        row["p_linear_vs_floor"] = linear["vs_boost_floor"]["wilcoxon_p"]
        row["floor_flexible"] = stats["boost_floor"][arm_key]["flexible"]["mean_lift"]
        row["d_flexible_vs_floor"] = flexible["vs_boost_floor"]["mean_paired_difference"]
        row["p_flexible_vs_floor"] = flexible["vs_boost_floor"]["wilcoxon_p"]
        # The floor readings obey the same two-denominator rule as every other
        # flexible figure: they move by an order of magnitude between baskets.
        n_kept = undegraded["n_targets"]
        row[f"d_flexible_undeg{n_kept}_vs_floor"] = undegraded["vs_boost_floor"][
            "mean_paired_difference"
        ]
        row[f"p_flexible_undeg{n_kept}_vs_floor"] = undegraded["vs_boost_floor"][
            "wilcoxon_p"
        ]
    return row


arms = pd.DataFrame([_arm_row(name, a) for name, a in stats["arms"].items()])
display(arms.round(5))

print(f"The flexible baseline degraded {len(stats['flexible_degraded_targets'])} of "
      f"{stats['n_targets']} targets, so every flexible figure has two denominators:")
print("  " + ", ".join(stats["flexible_degraded_targets"]))
print()
print("Width-matched boost floor -- an information-free Gaussian block at each arm's own width,")
print("scored through the identical boost path. Read every _boost lift against ITS OWN row:")
for arm_key, entry in stats["boost_floor"].items():
    print(f"  {arm_key:22} w={entry['width']:>4}  linear {entry['linear']['mean_lift']:+.5f}"
          f"  flexible {entry['flexible']['mean_lift']:+.5f}")
""")

code("""
ridge_arms = arms[arms["learner"] == "ridge"]
arm_keys = [n.replace("_ridge", "") for n in ridge_arms["arm"]]

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
        # One floor per arm, not one line across the chart: the floor is an
        # overfitting penalty and it scales with the arm's block width.
        floor_key = "linear" if column == "lift_linear" else "flexible"
        floors = [stats["boost_floor"][k][floor_key]["mean_lift"] for k in arm_keys]
        ax.scatter(np.arange(len(arm_keys)) + 0.2, floors, marker="_", s=900,
                   color="#c1440e", lw=2, zorder=5, label="width-matched floor")
    ax.set_xticks(np.arange(len(arm_keys)))
    ax.set_xticklabels(arm_keys, rotation=35, ha="right")
    ax.axhline(0, color="#333", lw=1)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("Hatched bars are not findings: shape_v1 is a regression check, "
             "size_nonlinear is the null control. Red ticks are each arm's own "
             "width-matched boost floor -- boost bars are read against those, not against zero.")
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
print()
for name in ("shape_v1_ridge", "shape_v2_ridge", "typed_ridge", "typed_plus_shape_v2_ridge",
             "size_nonlinear_ridge"):
    flexible = stats["arms"][name]["flexible"]
    undegraded = flexible["undegraded"]
    print(f"{name:26} flexible  all-{stats['n_targets']}: {flexible['mean_lift']:+.5f} "
          f"p={flexible['vs_baseline']['wilcoxon_p']:.4f} "
          f"({flexible['vs_baseline']['n_wins']}/{stats['n_targets']})   "
          f"undegraded-{undegraded['n_targets']}: {undegraded['mean_lift']:+.5f} "
          f"p={undegraded['vs_baseline']['wilcoxon_p']:.4f} "
          f"({undegraded['vs_baseline']['n_wins']}/{undegraded['n_targets']})")
print()
print("Both denominators, side by side, because they disagree: typed_ridge is significant on")
print("all 28 and not on the undegraded 22. §23.6 forbids quoting either alone.")
""")

md("""
### Per pillar, because the aggregate is a property of the basket

Twenty of the twenty-eight targets are one QCEW table, so a basket-wide mean is
71% one pillar. Reading it as a breadth claim is a mistake this project has made
before.

Every column in the table names the baseline that produced it — `_linear` over
the size-plus-state baseline, `_flexbase` over the curvature-augmented one — so
this file cannot be opened on its own and read in a framing its reader cannot
see. That was finding I4 of the branch review: the committed CSV used to carry
the linear framing only, under column names that said so nowhere.

The boost columns are raw lifts and must be read against the per-arm floors
printed beneath the table, not against zero. The chart that follows is
restricted to the linear ridge columns: boost has no *per-pillar* floor to read
its lifts against (the floors are whole-basket, not decomposed by pillar), so a
per-pillar boost bar would repeat exactly the framing mistake `vs_boost_floor`
exists to avoid.
""")

code("""
display(by_pillar.round(5))

print("Every column above names its baseline: _linear over the size-plus-state baseline,")
print("_flexbase over the curvature-augmented one. Boost floors are per arm and whole-basket,")
print("not decomposed by pillar:")
for arm_key, entry in stats["boost_floor"].items():
    print(f"  {arm_key:22} w={entry['width']:>4}  linear {entry['linear']['mean_lift']:+.5f}"
          f"  flexible {entry['flexible']['mean_lift']:+.5f}")

fig, ax = plt.subplots(figsize=(11, 4.5))
keys = [c for c in by_pillar.columns if c.endswith("_linear")]
x = np.arange(len(by_pillar))
width = 0.8 / len(keys)
for i, key in enumerate(keys):
    ax.bar(x + i * width - 0.4, by_pillar[key], width, label=key)
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\\n({n})" for p, n in zip(by_pillar["pillar"], by_pillar["n_targets"])])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean lift by owning pillar (ridge only, linear baseline)")
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

undegraded_rows = scores["r2_baseline_flexible"] >= scores["r2_baseline"]
print(f"Mean R² alone across all {stats['n_targets']} targets: "
      f"{scores['r2_alone_shape_v2'].mean():.4f}")
print(f"Mean lift over the flexible baseline, all {stats['n_targets']}: "
      f"{scores['lift_shape_v2_flexbase'].mean():+.5f}")
print(f"Mean lift over the flexible baseline, undegraded {int(undegraded_rows.sum())}: "
      f"{scores.loc[undegraded_rows, 'lift_shape_v2_flexbase'].mean():+.5f}")
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
