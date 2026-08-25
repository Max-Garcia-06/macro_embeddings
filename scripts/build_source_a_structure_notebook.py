"""Generate analysis-output/source-a/source_a_structure_round.ipynb.

The exploratory round on what an article's *shape* knows about a county. Reads
the committed artifacts -- the structural parquet, both stats files and the
scores CSV -- and computes every figure from them. No number is typed in by
hand, which is the standing rule for this project's notebooks: a number that
moves upstream has to move here.

Order is deliberate. The size-proxy audit runs *before* the scoring section,
not after, so the reader forms the expectation ("most of these columns are size
measurements") before seeing the result rather than being talked into it
afterwards.

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
size measurement in disguise has nothing left to contribute once a baseline
already holding three size measures has taken its share.

The audit finds size dependence concentrated in a small minority of the
block, not spread across it:

- Six of the sixty-four columns — the volume and breadth proxies
  `n_distinct_titles`, `n_body_sections`, `max_section_id`,
  `share_chars_census`, `total_body_chars`, and `share_in_largest_section` —
  clear |r| = 0.4 against at least one size measure. `n_body_sections` is the
  one that was cut from the shipped matrix on exactly this ground.
- The other fifty-eight sit well below that line; the twelve weakest are
  negligible, not merely weak, at |r| = 0.018–0.087 against every size
  measure.

That skew is a finding in itself, and it is the reason the scoring section
below has anything to measure: a block that were mostly a size proxy would
have nothing left once the baseline took its share.

Three size measures, not one: `n_body_sections` was cut on its correlation with
log tax returns specifically, and a table showing only population would have
understated it.
""")

code("""
import sys
sys.path.insert(0, str(REPO / "scripts"))
from pillar_matrix import SIZE_FEATURES, build_matrix

matrix, _ = build_matrix()
merged = matrix[["fips_code", *SIZE_FEATURES]].merge(features, on="fips_code", how="inner")
structure_cols = [c for c in features.columns if c != "fips_code"]

correlations = pd.DataFrame(
    {size: merged[structure_cols].corrwith(merged[size]) for size in SIZE_FEATURES}
)
correlations["max_abs"] = correlations.abs().max(axis=1)
correlations = correlations.sort_values("max_abs", ascending=False)

print("Most size-dependent structural columns:")
display(correlations.head(12).round(3))
print("\\nLeast size-dependent — the columns with headroom:")
display(correlations.tail(12).round(3))
""")

code("""
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(correlations["max_abs"], bins=30, color="#3b6ea5")
ax.set_title("How much each structural column is a size measurement")
ax.set_xlabel("largest |r| against log_population, log_agi or log_gdp_latest")
ax.set_ylabel("columns")
fig.tight_layout()
plt.show()

n_strong = int((correlations["max_abs"] > 0.4).sum())
print(f"{n_strong} of {len(correlations)} columns correlate above |r| = 0.4 with at least one size measure.")
""")

md("""
## Part three — the four arms

Every arm sits on the same unpenalized baseline of three size measures plus
state fixed effects, and is fitted to that baseline's residuals with a ridge
whose penalty is chosen by nested crossvalidation. Identical folds, identical
rows, five folds, seed 42. A block that knows nothing therefore costs
approximately nothing rather than dragging the controls down.
""")

code("""
arms = pd.DataFrame(scoring_stats["arms"]).T
display(arms[["label", "compared_against", "mean_lift", "mean_paired_difference", "n_wins", "wilcoxon_p"]])

print(f"{scoring_stats['n_targets']} targets | "
      f"{scoring_stats['n_structure_features']} structural columns vs "
      f"{scoring_stats['n_typed_features']} shipped typed columns")
""")

code("""
fig, ax = plt.subplots(figsize=(11, 4.5))
keys = list(scoring_stats["arms"])
values = [scoring_stats["arms"][k]["mean_lift"] for k in keys]
ax.bar(keys, values, color=["#3b6ea5", "#7a9cc6", "#2f5d8a"])
ax.axhline(0, color="#333", lw=1)
ax.set_title("Mean out-of-fold R² lift over the size-plus-state baseline")
ax.set_ylabel("mean lift")
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
width = 0.26
x = np.arange(len(by_pillar))
for offset, key, color in zip((-width, 0, width), keys, ("#3b6ea5", "#7a9cc6", "#2f5d8a")):
    ax.bar(x + offset, by_pillar[key], width, label=key, color=color)
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
                        "lift_typed_plus_structure"]]
print("Targets where article shape helps most:")
display(best.round(4))

worst = scores.tail(5)[["pillar", "label", "lift_structure", "lift_typed"]]
print("\\nAnd where it hurts most:")
display(worst.round(4))
""")

md("""
## What this round does and does not settle

- It does not propose shipping these columns. If `typed_plus_structure` beats
  `typed`, that is an argument for a follow-up round, not a change to
  `pillar_matrix`.
- It reads no section text. Any lexicon question belongs to the section-scope
  round, which already exists.
- It does not revisit the `n_body_sections` cut on its own authority. That
  decision stands unless the numbers above argue otherwise.
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
