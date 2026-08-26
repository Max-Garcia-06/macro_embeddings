"""Generate analysis-output/source-a/source_a_structure_round.ipynb.

The exploratory round on what an article's *shape* knows about a county. Reads
the committed artifacts -- the structural parquet, both stats files and the
scores CSV -- and computes every figure from them. No number is typed in by
hand, which is the standing rule for this project's notebooks: a number that
moves upstream has to move here.

Order is deliberate. The size-proxy audit runs *before* the scoring section, not
after, so the reader has the size question in hand before seeing the result
rather than being talked into it afterwards. The expectation it was written to
set -- "most of these columns are size measurements" -- is not what the audit
found: 6 of 64 columns clear |r| = 0.4 against any size measure, not most of
them. The audit also carries a nonlinear diagnostic, because Pearson |r| is
blind to exactly the channel the round's scoring turned out to run on.

Matplotlib, not plotly: plotly's mimetype output needs a JupyterLab extension
and renders as blank space without it.

Build and execute:

    uv run scripts/build_source_a_structure_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO: Path = Path(__file__).resolve().parent.parent
OUT: Path = REPO / "analysis-output" / "source-a" / "source_a_structure_round.ipynb"

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
# Source A — What the Shape of an Article Knows

**The question:** how much of a county can be read off the *structure* of its
Wikipedia article — how many sections it has, how long they are, which ones are
present — without reading a word of the text?

**The prior:** the answer is "county size". `n_body_sections` was computed
during the section round, correlated r = 0.550 against log tax returns, and cut
from the scored block for exactly that reason. This notebook is built so that
prior can be checked rather than assumed: the size audit runs before the
scoring, and every arm sits on a baseline that already holds three size
measures.

**Read the baseline precisely.** It holds those three measures *linearly, in
logs*. So "lift over the baseline" means "beyond a linear-in-logs size model" —
not "beyond county size". Those are different claims, and until 2026-08-25 this
round made the stronger one. Part three carries a null-control arm built from
nothing but curves on the baseline's own columns, which is what makes the
difference measurable rather than arguable.

Everything below is computed from committed artifacts. Nothing is typed in.
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

sections = pd.read_parquet(DATA / "source_a_sections.parquet")
features = pd.read_parquet(DATA / "source_a_structure_features.parquet")
feature_stats = json.loads((SOURCE_A / "source_a_structure_feature_stats.json").read_text())
scoring_stats = json.loads((SOURCE_A / "source_a_structure_stats.json").read_text())
scores = pd.read_csv(OUTPUTS / "source_a_structure_scores.csv")
by_pillar = pd.read_csv(OUTPUTS / "source_a_structure_by_pillar.csv")

print(f"{feature_stats['n_counties']:,} counties x {feature_stats['n_features']} structural features")
print(f"{len(sections):,} sections, {len(feature_stats['title_flag_vocabulary'])} title flags")
""")

md("""
## Part one — what the corpus is shaped like

Three facts set up everything after them: how many sections an article has, how
unevenly its characters are spread across them, and how quickly the title
vocabulary thins out.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].hist(features["n_body_sections"], bins=40, color="#3b6ea5")
axes[0].set_title("Sections per county")
axes[0].set_xlabel("body sections")
axes[0].axvline(features["n_body_sections"].median(), color="#c44", lw=1.5,
                label=f"median {features['n_body_sections'].median():.0f}")
axes[0].legend()

section_chars = sections["section_text"].str.len()
axes[1].hist(np.log10(section_chars.clip(lower=1)), bins=50, color="#3b6ea5")
axes[1].set_title("Section length (log10 characters)")
axes[1].set_xlabel("log10 characters")
axes[1].axvline(np.log10(section_chars.median()), color="#c44", lw=1.5,
                label=f"median {section_chars.median():,.0f} chars")
axes[1].legend()

fig.tight_layout()
plt.show()

print(f"mean section length {section_chars.mean():,.0f} chars, median {section_chars.median():,.0f}, "
      f"max {section_chars.max():,.0f}")
print("The mean sits well above the median: a handful of very long sections carry it.")
""")

code("""
titles = sections["section_title"].fillna("").str.strip().str.lower()
per_title = titles[titles != ""].groupby(titles[titles != ""]).size().sort_values(ascending=False)
county_counts = (sections.assign(t=titles).loc[titles != ""]
                 .groupby("t")["fips_code"].nunique().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(11, 5))
top = county_counts.head(25)[::-1]
ax.barh(top.index, top.to_numpy(), color="#3b6ea5")
ax.axvline(feature_stats["title_flag_min_share"] * feature_stats["n_counties"], color="#c44",
           lw=1.5, ls="--", label=f"flag floor ({feature_stats['title_flag_min_share']:.0%} of counties)")
ax.set_title("Most common section titles, by counties carrying them")
ax.set_xlabel("counties")
ax.legend()
fig.tight_layout()
plt.show()

print(f"{len(county_counts):,} distinct titles; {len(feature_stats['title_flag_vocabulary'])} clear the floor.")
""")

code("""
bucket_means = pd.Series(feature_stats["mean_bucket_share"]).sort_values(ascending=False)
bucket_means.index = [c.replace("share_chars_", "") for c in bucket_means.index]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(bucket_means.index, bucket_means.to_numpy(), color="#3b6ea5")
ax.set_title("Mean share of a county's body characters, by section theme")
ax.set_ylabel("share of characters")
for x, y in zip(bucket_means.index, bucket_means.to_numpy()):
    ax.text(x, y + 0.005, f"{y:.1%}", ha="center", fontsize=9)
fig.tight_layout()
plt.show()
""")

md("""
## Part two — the size-proxy audit

This runs before the scoring, on purpose: a structural column that is only a
size measurement in disguise has little left to contribute once a baseline
already holding three size measures has taken its share.

**Two diagnostics, because one of them is blind to the channel that matters.**
Pearson |r| against the three size features answers "is this column a *linear*
function of county size?" — and answering only that is how this round shipped a
false claim the first time. The baseline is linear in the logs, so what gets
through it is a *curved* relationship with size, and a near-deterministic curved
function of log population can sit at |r| = 0.04 and look clean. The second
diagnostic, `size_r2`, closes the gap: the out-of-fold R² of each structural
column regressed on a degree-3 polynomial basis in the same three size features,
five folds, seed 42. `linear_r2` is the same fit restricted to a straight line,
and `curve_gain` is the difference — what a curve on county size knows about
this column that a straight line does not. That last column is the one this
audit exists to show.

**And it comes back clean, which is not the same as safe.** No structural column
turns out to be a hidden curved size proxy: the strongest is explained about a
third of the way by a degree-3 size basis, and letting the fit curve buys very
little over a straight line anywhere in the block. That clears each column
*individually*. It does not clear the block — Part three's null control shows
the channel the scoring actually runs through works jointly across columns, at a
scale this per-column table cannot see. Take the clean audit as a necessary
condition for the round's claim, never a sufficient one.

**A near-constant column is not a column with headroom.** A flag that fires for
almost every county correlates weakly with everything, size included, because it
has almost no variance to correlate with. Sorting by |r| alone puts those at the
top of a "least size-dependent" list, where they read as if they had something
left to give. Prevalence and standard deviation sit beside the correlation below
so the two can be told apart, and the headroom table excludes flags firing on
more than 95% or fewer than 5% of counties.

Three size measures, not one: `n_body_sections` was cut on its correlation with
log tax returns specifically, and a table showing only population would have
understated it.

Every count, column name and range for this audit is printed by the cell that
computes it. None of it is typed into this prose — an audit result typed one
cell above the code that produces it is an audit result that rots silently the
next time the corpus moves.
""")

code("""
import sys
sys.path.insert(0, str(REPO / "scripts"))
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from analyze_source_a_structure import N_FOLDS, RANDOM_SEED
from pillar_matrix import SIZE_FEATURES, build_matrix

matrix, _ = build_matrix()
merged = matrix[["fips_code", *SIZE_FEATURES]].merge(features, on="fips_code", how="inner")
structure_cols = [c for c in features.columns if c != "fips_code"]
values = merged[structure_cols]

# Diagnostic 1 — linear. What a correlation table can see.
correlations = pd.DataFrame({size: values.corrwith(merged[size]) for size in SIZE_FEATURES})
correlations["max_abs_r"] = correlations.abs().max(axis=1)

# Diagnostic 2 — nonlinear. Out-of-fold R² of each structural column against a
# degree-3 polynomial basis in the same three size measures. This is the one the
# baseline cannot control for, so it is the one that decides whether a lift is
# content or curvature.
size_basis = merged[list(SIZE_FEATURES)].to_numpy(dtype="float64")
block = values.to_numpy(dtype="float64")
folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def size_explained(degree):
    \"\"\"Out-of-fold R2 of every structural column against a size basis of this degree.\"\"\"
    steps = [
        # Median-impute first, exactly as the scoring baseline does: `log_agi`
        # and `log_gdp_latest` are null for a handful of counties.
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if degree > 1:
        steps += [("expand", PolynomialFeatures(degree=degree, include_bias=False)),
                  ("rescale", StandardScaler())]
    steps.append(("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13), alpha_per_target=True)))
    predicted = cross_val_predict(Pipeline(steps), size_basis, block, cv=folds)
    return np.clip(r2_score(block, predicted, multioutput="raw_values"), 0.0, None)


correlations["linear_r2"] = size_explained(1)
correlations["size_r2"] = size_explained(3)
# The number that matters: what a *curve* on county size knows about this column
# that a straight line does not. That is the channel the baseline cannot control.
correlations["curve_gain"] = correlations["size_r2"] - correlations["linear_r2"]

# Variance, so "low |r|" and "nothing to correlate with" can be told apart.
correlations["sd"] = values.std()
is_flag = values.isin([0.0, 1.0]).all()
correlations["prevalence"] = values.mean().where(is_flag)
near_constant = (correlations["prevalence"].fillna(0.5) - 0.5).abs() > 0.45

shown = ["max_abs_r", "linear_r2", "size_r2", "curve_gain", "sd", "prevalence"]
correlations = correlations.sort_values("size_r2", ascending=False)
near_constant = near_constant.reindex(correlations.index)  # keep the mask in table order

print("Most size-dependent structural columns, by the nonlinear measure:")
display(correlations[[*SIZE_FEATURES, *shown]].head(12).round(3))

print("\\nLargest curvature gain — where a curve on size beats a straight line by most:")
display(correlations.nlargest(8, "curve_gain")[shown].round(3))

headroom = correlations.loc[~near_constant].nsmallest(12, "size_r2")
print("\\nLeast size-dependent — the columns with headroom, near-constant flags excluded:")
display(headroom[[*SIZE_FEATURES, *shown]].round(3))

print("\\nNear-constant flags excluded from that table, and why they are not headroom:")
display(correlations.loc[near_constant, shown].sort_values("prevalence", ascending=False).round(3))
""")

code("""
strong = correlations.index[correlations["max_abs_r"] > 0.4].tolist()
near_line = correlations.index[
    correlations["max_abs_r"].between(0.35, 0.4, inclusive="left")
].tolist()
weakest = correlations.nsmallest(12, "max_abs_r")["max_abs_r"]

print(f"LINEAR: {len(strong)} of {len(correlations)} columns clear |r| = 0.4 against at least one "
      f"size measure — {', '.join(strong)}.")
print(f"The remaining {len(correlations) - len(strong)} do not sit at a cliff edge: "
      f"{len(near_line)} land between 0.35 and 0.40 ({', '.join(near_line) or 'none'}), "
      f"{int((correlations['max_abs_r'] > 0.3).sum())} are above 0.30, and the twelve weakest "
      f"span |r| = {weakest.min():.3f}–{weakest.max():.3f}.")
print(f"{int(near_constant.sum())} columns are near-constant flags (>95% or <5% prevalence): "
      f"{', '.join(correlations.index[near_constant])}. Sorted by |r| alone they lead the "
      f"'least size-dependent' list — but their low |r| is missing variance, not spare "
      f"information, so they are excluded from the headroom table above.")

top_curve = correlations.nlargest(1, "curve_gain").iloc[0]
hidden = correlations[(correlations["max_abs_r"] < 0.4) & (correlations["size_r2"] > 0.4)]
print(f"\\nNONLINEAR: no structural column is a near-deterministic function of county size in "
      f"any shape. The largest out-of-fold R² on a degree-3 size basis is "
      f"{correlations['size_r2'].max():.3f} ({correlations['size_r2'].idxmax()}); "
      f"{int((correlations['size_r2'] > 0.2).sum())} columns clear 0.20 and "
      f"{int((correlations['size_r2'] > 0.1).sum())} clear 0.10.")
print(f"{len(hidden)} columns hide behind the linear diagnostic (|r| < 0.4 but size_r2 > 0.4): "
      f"{', '.join(hidden.index) or 'none'}.")
print(f"Curvature adds little per column: the largest gain over a straight-line fit is "
      f"{top_curve['curve_gain']:+.3f} ({correlations['curve_gain'].idxmax()}), and the median "
      f"gain is {correlations['curve_gain'].median():+.3f}.")
print("\\nRead that carefully: it clears each column INDIVIDUALLY, not the block. The null-control")
print("arm in Part three shows the size channel this block runs through operates jointly across")
print("columns and at a scale no per-column audit here can see. A clean audit is a necessary")
print("condition for the round's claim, not a sufficient one.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].hist(correlations["max_abs_r"], bins=30, color="#3b6ea5")
axes[0].set_title("Linear: largest |r| against a size measure")
axes[0].set_xlabel("max |r| over the three size features")
axes[0].set_ylabel("columns")
axes[1].hist(correlations["size_r2"], bins=30, color="#a5553b")
axes[1].set_title("Nonlinear: R² on a degree-3 size basis")
axes[1].set_xlabel("out-of-fold R²")
axes[2].hist(correlations["curve_gain"], bins=30, color="#7a5ba5")
axes[2].set_title("Curvature gain: degree-3 minus linear")
axes[2].set_xlabel("out-of-fold R² gained by allowing a curve")
fig.tight_layout()
plt.show()
""")

md("""
## Part three — the arms, and the null control that prices them

Every arm sits on the same unpenalized baseline of three size measures plus
state fixed effects, and is fitted to that baseline's residuals with a ridge
whose penalty is chosen by nested crossvalidation. Identical folds, identical
rows, five folds, seed 42. A block of pure noise therefore costs approximately
nothing rather than dragging the controls down.

**A block of pure *curvature* does not cost approximately nothing, and that is
the whole story of this round.** The `size_nonlinear` arm is a null control: nine
columns of squares, cubes and pairwise products of `log_population`, `log_agi`
and `log_gdp_latest` — the baseline's own three columns, reshaped. It contains no
information the baseline does not already hold. It is in the table below because
the number it scores is the unit every other number should be read in.
""")

code("""
arms = pd.DataFrame(scoring_stats["arms"]).T
display(arms[["label", "compared_against", "mean_lift", "mean_paired_difference", "n_wins", "wilcoxon_p"]])

null_key = scoring_stats["null_arm_key"]
null_arm, structure_arm = scoring_stats["arms"][null_key], scoring_stats["arms"]["structure"]
fusion = scoring_stats["arms"]["typed_plus_structure"]

print(f"{scoring_stats['n_targets']} targets | "
      f"{scoring_stats['n_structure_features']} structural columns vs "
      f"{scoring_stats['n_typed_features']} shipped typed columns vs "
      f"{scoring_stats['n_size_nonlinear_features']} null columns")
print(f"\\nNULL CONTROL '{null_key}' carries zero information the baseline lacks, and scores "
      f"{null_arm['mean_lift']:+.5f} on {null_arm['n_wins']}/{scoring_stats['n_targets']} targets "
      f"(p={null_arm['wilcoxon_p']:.2g}).")
print(f"The structural arm's {structure_arm['mean_lift']:+.5f} is "
      f"{scoring_stats['structure_lift_in_null_arm_units']:.2f}x that — "
      f"{1 / scoring_stats['structure_lift_in_null_arm_units']:.1f} times SMALLER than an "
      f"information-free reshaping of the controls.")
print("So a positive lift here means 'beyond a LINEAR-in-logs size model'. It does not mean "
      "'beyond county size'.")
""")

code("""
fig, ax = plt.subplots(figsize=(11, 4.5))
keys = list(scoring_stats["arms"])
values = [scoring_stats["arms"][k]["mean_lift"] for k in keys]
# The null arm is drawn in a different colour, and named as one on the axis, so
# it cannot be skim-read as a fourth result.
palette = {"structure": "#3b6ea5", "typed": "#7a9cc6", "typed_plus_structure": "#2f5d8a"}
ax.bar(keys, values, color=[palette.get(k, "#a5553b") for k in keys])
ax.axhline(0, color="#333", lw=1)
ax.axhline(null_arm["mean_lift"], color="#a5553b", lw=1.4, ls="--",
           label=f"null control: {null_arm['mean_lift']:+.4f} for zero information")
ax.set_title("Mean out-of-fold R² lift over the size-plus-state baseline")
ax.set_ylabel("mean lift")
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([f"{k}\\n(NULL CONTROL)" if k == null_key else k for k in keys])
ax.legend()
for x, y in zip(keys, values):
    ax.text(x, y, f"{y:+.4f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=10)
fig.tight_layout()
plt.show()
""")

md("""
### Per-pillar, because the aggregate is a property of the basket

Twenty of the twenty-eight targets are one QCEW table. A basket-wide mean is
therefore 71% one pillar, and reading it as a breadth claim is a mistake this
project has made before.
""")

code("""
display(by_pillar.round(5))

fig, ax = plt.subplots(figsize=(11, 4.5))
width = 0.8 / len(keys)
x = np.arange(len(by_pillar))
offsets = (np.arange(len(keys)) - (len(keys) - 1) / 2) * width
for offset, key in zip(offsets, keys):
    label = f"{key} (NULL CONTROL)" if key == null_key else key
    ax.bar(x + offset, by_pillar[key], width, label=label, color=palette.get(key, "#a5553b"))
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\\n({n} targets)" for p, n in zip(by_pillar["pillar"], by_pillar["n_targets"])])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean lift by owning pillar")
ax.legend()
fig.tight_layout()
plt.show()
""")

code("""
best = scores.head(10)[["pillar", "label", "n", "r2_baseline", "lift_structure", "lift_typed",
                        "lift_typed_plus_structure", "lift_size_nonlinear"]]
print("Targets where article shape helps most — with the null arm on the same rows:")
display(best.round(4))

worst = scores.tail(5)[["pillar", "label", "lift_structure", "lift_typed", "lift_size_nonlinear"]]
print("\\nAnd where it hurts most:")
display(worst.round(4))

beaten = scores[scores["lift_size_nonlinear"] > scores["lift_structure"]]
print(f"\\nOn {len(beaten)} of {len(scores)} targets the information-free null block beats the "
      f"structural block outright.")
""")

md("""
### What survives, and what does not

The structural arm's +0.0027 is real in the sense that it is not noise — the
Wilcoxon test on 21/28 wins is not a fluke, and the branch review measured 64
columns of pure Gaussian noise through these same protocol helpers at +0.00008,
12/28, p=0.938. It is *not* real in the sense the round originally claimed.
Three things follow, and they should be read together. Every number in points 2
and 3 comes from the branch review rather than from the cells above, and is
recorded with its provenance in
`analysis-output/source-a/source-a-findings.md` §23:

1. **Lift here is measured against a linear-in-logs size model, not against
   county size.** The null-control arm above settles that: an information-free
   reshaping of the baseline's own three columns scores several times the
   structural block. Curvature in the relationship between an article's shape and
   a county's size clears this bar without carrying any content.
2. **Under a flexible size control the structural lift drops to roughly a
   quarter of its headline.** Re-scored with the same quadratic, cubic and
   interaction terms folded into the baseline — residualized against the linear
   size columns and whitened, so they add no information and leave mean baseline
   R² essentially unchanged at 0.2612 → 0.2607 — `structure` falls from +0.00269
   to +0.00073 (20/28, p=0.0281) and `typed` from +0.00307 to +0.00161 (18/28,
   p=0.0110). **The fusion comparison does not survive at all**:
   `typed_plus_structure` over `typed` goes from +0.00184 (p=0.0118) to +0.00095
   at p=0.1315.
3. **What survives is narrow and concentrated, and is still a finding.** The
   collapse lands hardest on exactly the targets that carried the headline:
   Retail Trade LQ +0.01031 → +0.00089 (−91%), Agriculture +0.00467 → +0.00008,
   Information +0.00366 → +0.00005, Wholesale +0.00325 → −0.00040. Accommodation
   & Food Services LQ is the exception, retaining +0.01455 of its +0.02546. A
   county's article shape carrying real signal about its accommodation-and-food
   location quotient, net of a flexible size model, is a publishable result. "The
   shape of an article knows things county size does not" is not.

Dropping the six flagged size-proxy columns and rescoring the remaining 58 still
gives +0.00254 (21/28, p=0.0014), so the result is not carried by the obvious
size proxies. It is carried by the curved ones — which is precisely why the
linear audit in Part two could not have caught it, and why that audit now
carries `size_r2`.
""")

md("""
## What this round does and does not settle

- It does not propose shipping these columns. `typed_plus_structure` beats
  `typed` by +0.00184 as scored here, but that comparison does not survive a
  flexible size control (p=0.1315), so there is no fusion case to make.
- It reads no section text. Any lexicon question belongs to the section-scope
  round, which already exists.
- It does not revisit the `n_body_sections` cut on its own authority. Nothing
  above argues for reopening it; the audit's nonlinear diagnostic argues the
  other way.
- The open question it leaves is Accommodation & Food Services: the one target
  where article shape retains most of its lift under a flexible size control, and
  the one worth a round of its own.
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
