"""Fixtures shared across the Source A representation test suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="session")
def sections_frame() -> pd.DataFrame:
    """Long-format Wikipedia section frame, loaded once per session."""
    return pd.read_parquet(REPO_ROOT / "data" / "source_a_sections.parquet")


@pytest.fixture(scope="session")
def tiered_embedding_results() -> pd.DataFrame:
    """Committed pooled results from the tiered embedding sweep."""
    return pd.read_csv(REPO_ROOT / "outputs" / "source_a_tiered_embedding.csv")


@pytest.fixture(scope="session")
def representation_stats() -> dict:
    """The shipped representation-marginal stats artifact."""
    path = (
        REPO_ROOT
        / "analysis-output"
        / "source-a"
        / "source_a_representation_marginal_stats.json"
    )
    return json.loads(path.read_text())
