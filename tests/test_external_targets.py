"""ACS external-target arithmetic and admission gates."""
from __future__ import annotations

import pandas as pd
import pytest

import ingest_external_targets as iet


def test_download_table_is_memoized(monkeypatch) -> None:
    """A second request for the same table must not hit the network."""
    calls: list[str] = []

    def fake_fetch(table: str) -> pd.DataFrame:
        calls.append(table)
        return pd.DataFrame(
            {"B00000_E001": [1.0], "B00000_M001": [0.1]},
            index=pd.Index(["01001"], name="fips_code"),
        )

    monkeypatch.setattr(iet, "_fetch_table_uncached", fake_fetch)
    iet._download_table.cache_clear()

    first = iet._download_table("b00000")
    second = iet._download_table("b00000")

    assert calls == ["b00000"], "second call should be served from cache"
    pd.testing.assert_frame_equal(first, second)
