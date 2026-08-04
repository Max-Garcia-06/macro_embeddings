"""Identify logistics hub/corridor counties from Source D trade flow data.

The proposal frames Source D as distinguishing "a logistical pass-through
corridor or industrial exporter from a pure consumer sink"
(`E_macro_extendedProposal.pdf`). `docs/plans/ingestion_recon.md (Source D)`'s Phase 1b/2 findings
ruled out raw partner degree as a signal (BTS's gravity-model
disaggregation assigns near-universal nonzero flow regardless of county
character) and found that at national scale, hub counties are *both*
higher-volume *and* more concentrated (funneling flow through a few
dominant corridors) than rural counties -- the opposite direction from the
small-sample regional spike. This script identifies hubs using exactly
those two validated signals: total tonnage (weighted degree's replacement)
and pooled-partner HHI concentration.

Output: `source_d_hubs.csv` (top hub counties ranked by total tonnage) and
`source_d_hubs.html` (interactive scatter of tonnage vs. concentration).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE_D_PARQUET_PATH: Path = REPO_ROOT / "data" / "source_d_faf.parquet"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_d_hubs.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_d_hubs.html"

TOP_N_HUBS: int = 25
TOP_LABEL_COUNT: int = 10

# Dataviz reference palette, matching the other Source D/F EDA scripts.
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
GRIDLINE_COLOR: str = "#e1e0d9"
AXIS_LINE_COLOR: str = "#c3c2b7"
POINT_COLOR: str = "#8c8b85"
HUB_MARKER_COLOR: str = "#e34948"
TREND_LINE_COLOR: str = "#52514e"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_source_d(parquet_path: Path) -> pd.DataFrame:
    """Load Source D trade-flow data.

    Args:
        parquet_path: Path to `source_d_faf.parquet`.

    Returns:
        DataFrame as written by `ingest_source_d.py`.
    """
    return pd.read_parquet(parquet_path)


def add_hub_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the two validated hub signals: total tonnage and mean partner concentration.

    Args:
        df: Source D DataFrame with `total_outbound_tons`, `total_inbound_tons`,
            `out_partner_hhi`, `in_partner_hhi` columns.

    Returns:
        Copy of df with `total_tons` (sum of both directions) and
        `mean_partner_hhi` (mean of both directions' HHI) added.
    """
    df = df.copy()
    df["total_tons"] = df["total_outbound_tons"] + df["total_inbound_tons"]
    df["mean_partner_hhi"] = df[["out_partner_hhi", "in_partner_hhi"]].mean(axis=1)
    return df


def rank_top_hubs(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Rank counties by total tonnage.

    Args:
        df: Output of add_hub_signals.
        top_n: Number of top counties to return.

    Returns:
        Top `top_n` rows by `total_tons`, descending.
    """
    columns = ["county_name", "fips_code", "total_tons", "mean_partner_hhi"]
    return df[columns].nlargest(top_n, "total_tons").reset_index(drop=True)


def build_scatter(df: pd.DataFrame, top_hubs: pd.DataFrame) -> go.Figure:
    """Build a scatter of log-tonnage vs. partner concentration, highlighting top hubs.

    Args:
        df: Output of add_hub_signals, all counties.
        top_hubs: Output of rank_top_hubs, the counties to label directly.

    Returns:
        Plotly Figure ready to export.
    """
    log_tons = np.log10(df["total_tons"].clip(lower=1))
    slope, intercept = np.polyfit(log_tons, df["mean_partner_hhi"], 1)
    trend_x = np.array([log_tons.min(), log_tons.max()])
    trend_y = slope * trend_x + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=log_tons,
            y=df["mean_partner_hhi"],
            mode="markers",
            marker=dict(color=POINT_COLOR, size=5, opacity=0.5),
            customdata=df["county_name"],
            hovertemplate="%{customdata}<br>tons=10^%{x:.2f}<br>HHI=%{y:.3f}<extra></extra>",
            name="All counties",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_x,
            y=trend_y,
            mode="lines",
            line=dict(color=TREND_LINE_COLOR, width=2, dash="dash"),
            name=f"Trend (slope={slope:.3f}/log10 ton)",
        )
    )
    labeled = top_hubs.head(TOP_LABEL_COUNT)
    fig.add_trace(
        go.Scatter(
            x=np.log10(labeled["total_tons"].clip(lower=1)),
            y=labeled["mean_partner_hhi"],
            mode="markers+text",
            marker=dict(color=HUB_MARKER_COLOR, size=10, line=dict(width=2, color=SURFACE_COLOR)),
            text=[name.split(",")[0] for name in labeled["county_name"]],
            textposition="top center",
            textfont=dict(size=10, color=PRIMARY_INK_COLOR),
            name=f"Top {TOP_LABEL_COUNT} hubs by tonnage",
        )
    )
    fig.update_layout(
        title="Source D: trade volume vs. partner concentration (national scale)",
        xaxis=dict(
            title="log10(total tons)",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        yaxis=dict(
            title="Mean partner concentration (HHI)",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main() -> None:
    """Run the Source D hub identification pipeline."""
    configure_logging()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = add_hub_signals(load_source_d(SOURCE_D_PARQUET_PATH))
    top_hubs = rank_top_hubs(df, TOP_N_HUBS)

    top_hubs.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote top %d hub counties to %s", len(top_hubs), OUTPUT_CSV_PATH)

    corr = np.log10(df["total_tons"].clip(lower=1)).corr(df["mean_partner_hhi"])
    logger.info(
        "log10(total_tons) vs. mean_partner_hhi: Pearson r=%.4f -- %s",
        corr,
        "hubs are both higher-volume and more concentrated"
        if corr > 0
        else "hubs are higher-volume but less concentrated",
    )

    logger.info("Top %d hub counties by total tonnage:", TOP_N_HUBS)
    for row in top_hubs.itertuples():
        logger.info("  %10.0f tons | HHI=%.3f | %s", row.total_tons, row.mean_partner_hhi, row.county_name)

    fig = build_scatter(df, top_hubs)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote scatter to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
