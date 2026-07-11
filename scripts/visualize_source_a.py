"""US map visualization for Source A county embeddings.

Loads `source_a_embeddings.parquet`, reduces the 1024-dim embeddings to a
single interpretable signal via PCA, and plots each county as a bubble on a
US map at its geographic centroid, colored by that signal. Centroids come
from the Census Bureau's Gazetteer counties file, cached locally after the
first download.

Output: `source_a_map.html`, an interactive Plotly map.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.decomposition import PCA

from ingest_source_a import FIPS_CROSSWALK

GAZETTEER_URL: str = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_counties_national.zip"
)
REQUEST_TIMEOUT_SECONDS: int = 30

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EMBEDDINGS_PARQUET_PATH: Path = REPO_ROOT / "data" / "source_a_embeddings.parquet"
CENTROIDS_CACHE_PATH: Path = REPO_ROOT / "data" / "county_centroids.parquet"
OUTPUT_HTML_PATH: Path = REPO_ROOT / "outputs" / "source_a_map.html"

# Diverging pair from the dataviz reference palette: blue <-> gray <-> red.
DIVERGING_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#2a78d6"),
    (0.5, "#f0efec"),
    (1.0, "#e34948"),
]

# Chart chrome from the dataviz reference palette (light mode).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"

# Number of highest/lowest-PC1 counties to direct-label on the map.
EXTREME_COUNTY_LABEL_COUNT: int = 3

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_embeddings(parquet_path: Path) -> pd.DataFrame:
    """Load Source A embeddings and fill any missing FIPS codes from the crosswalk.

    Args:
        parquet_path: Path to `source_a_embeddings.parquet`.

    Returns:
        DataFrame with `fips_code` fully populated wherever the crosswalk has
        an entry for the county name.
    """
    df = pd.read_parquet(parquet_path)
    df["fips_code"] = df["fips_code"].fillna(df["county_name"].map(FIPS_CROSSWALK))
    return df


def fetch_county_centroids(cache_path: Path) -> pd.DataFrame:
    """Load county centroid coordinates, downloading and caching on first use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with columns `fips_code`, `lat`, `lon`.
    """
    if cache_path.exists():
        logger.info("Loading cached county centroids from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading Census Gazetteer counties file...")
    response = requests.get(GAZETTEER_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        (member_name,) = archive.namelist()
        with archive.open(member_name) as f:
            gaz = pd.read_csv(f, sep="\t", encoding="latin-1", dtype={"GEOID": str})

    gaz.columns = gaz.columns.str.strip()
    centroids = gaz[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"GEOID": "fips_code", "INTPTLAT": "lat", "INTPTLONG": "lon"}
    )
    centroids["fips_code"] = centroids["fips_code"].str.strip()

    centroids.to_parquet(cache_path, engine="pyarrow", index=False)
    logger.info("Cached county centroids to %s", cache_path)
    return centroids


def compute_pc1(embeddings: pd.Series) -> tuple[np.ndarray, float]:
    """Reduce stacked embedding vectors to their first principal component.

    Args:
        embeddings: Series of 1024-dim embedding lists.

    Returns:
        Tuple of (PC1 values per row, explained variance ratio of PC1).
    """
    matrix = np.vstack(embeddings.to_numpy())
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(matrix).ravel()
    return pc1, float(pca.explained_variance_ratio_[0])


def build_map(df: pd.DataFrame, explained_variance_ratio: float) -> "px.Figure":
    """Build an interactive US bubble map colored by PC1.

    Args:
        df: DataFrame with `county_name`, `fips_code`, `lat`, `lon`, `pc1`.
        explained_variance_ratio: Fraction of embedding variance PC1 captures.

    Returns:
        Plotly Figure ready to export.
    """
    pc1_bound = float(np.abs(df["pc1"]).max())
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="pc1",
        hover_name="county_name",
        hover_data={"pc1": ":.3f", "fips_code": True, "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=DIVERGING_COLORSCALE,
        range_color=(-pc1_bound, pc1_bound),
        title=(
            "Source A county embeddings — PC1 "
            f"({explained_variance_ratio:.1%} of variance explained)"
        ),
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color=SURFACE_COLOR)))

    extremes = pd.concat(
        [
            df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "pc1"),
            df.nsmallest(EXTREME_COUNTY_LABEL_COUNT, "pc1"),
        ]
    )
    fig.add_trace(
        go.Scattergeo(
            lat=extremes["lat"],
            lon=extremes["lon"],
            mode="text",
            text=extremes["county_name"],
            textposition="top center",
            textfont=dict(size=10, color=PRIMARY_INK_COLOR),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def main() -> None:
    """Run the Source A map visualization pipeline."""
    configure_logging()

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)

    merged = embeddings_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    pc1, explained_variance_ratio = compute_pc1(merged["embedding"])
    merged = merged.assign(pc1=pc1)
    logger.info(
        "Matched %d/%d counties to centroids. PC1 explains %.1f%% of variance.",
        len(merged),
        len(embeddings_df),
        explained_variance_ratio * 100,
    )

    fig = build_map(merged, explained_variance_ratio)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
