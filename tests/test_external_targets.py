"""ACS external-target arithmetic and admission gates."""
from __future__ import annotations

import pandas as pd
import pytest

import ingest_external_targets as iet

AUTAUGA = "01001"

# Observed during planning. A mismatch means the line numbering moved.
AUTAUGA_EXPECTED = {
    "per_capita_income": 36227.0,
    "median_family_income": 83452.0,
    "median_gross_rent": 1200.0,
    "mean_household_size": 2.61,
    "owner_occupied_share": 16872 / 22523,
    "poverty_rate": 6275 / 58731,
    "bachelors_share": 6518 / 40767,
    "masters_share": 4006 / 40767,
    "labor_force_participation": 28020 / 47508,
    # Step 5 additions, observed while probing the candidate table families.
    # B25004's universe is vacant units only; the denominator comes from
    # B25024, which already reconciles against B25003/B11001 (22,523 occupied
    # + 2,208 vacant = 24,731 total housing units).
    "housing_vacancy_rate": 2208 / 24731,
    "electric_heating_share": 13553 / 22523,
    "gas_heating_share": 7026 / 22523,
    "bottled_gas_heating_share": 1737 / 22523,
    "fuel_oil_heating_share": 16 / 22523,
    "no_fuel_used_share": 75 / 22523,
    "drove_alone_share": 22907 / 26976,
    "carpooled_share": 2208 / 26976,
    "public_transit_share": 64 / 26976,
    "walked_share": 25 / 26976,
    "work_from_home_share": 1561 / 26976,
    "foreign_born_share": 1529 / 59285,
    "naturalized_share_of_foreign_born": 827 / 1529,
    "same_house_share": 52879 / 58635,
    "moved_within_county_share": 2372 / 58635,
    "moved_different_state_share": 1114 / 58635,
    "children_married_couple_share": 9474 / 11991,
    "children_female_householder_share": 1692 / 11991,
    "children_male_householder_share": 825 / 11991,
    "household_earnings_share": 15702 / 22523,
    "household_ss_income_share": 7980 / 22523,
    "mortgaged_share": 9303 / 16872,
    "computer_ownership_share": 21710 / 22523,
}


@pytest.fixture(scope="module")
def targets_frame() -> pd.DataFrame:
    return iet.fetch_external_targets().set_index("fips_code")


def test_basket_is_large_enough() -> None:
    assert len(iet.EXTERNAL_TARGETS) >= 40


def test_no_table_family_dominates() -> None:
    """The 28-target basket's defect was 71% one table. Cap this one at 6."""
    counts: dict[str, int] = {}
    for target in iet.EXTERNAL_TARGETS:
        counts[target.table] = counts.get(target.table, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1])
    assert worst[1] <= 6, f"{worst[0]} contributes {worst[1]} targets"


def test_every_target_has_a_circularity_verdict() -> None:
    verdicts = iet.TARGET_CIRCULARITY
    for target in iet.EXTERNAL_TARGETS:
        assert target.column in verdicts, f"{target.column} has no circularity verdict"
        assert verdicts[target.column] in {"clean", "ablated"}


@pytest.mark.parametrize(("column", "expected"), AUTAUGA_EXPECTED.items())
def test_autauga_reconciles(targets_frame, column: str, expected: float) -> None:
    actual = float(targets_frame.loc[AUTAUGA, column])
    assert actual == pytest.approx(expected, rel=1e-4)


def test_every_target_ships_a_standard_error(targets_frame) -> None:
    for target in iet.EXTERNAL_TARGETS:
        se = f"{target.column}_se"
        assert se in targets_frame.columns
        assert targets_frame[se].notna().sum() > 2500


def test_derive_masks_negative_numerator_sentinel(monkeypatch) -> None:
    """A suppressed numerator must not survive a proportion as a negative rate.

    `_derive`'s proportion/ratio branch used to mask only the denominator
    against ACS's negative-sentinel family (-666666666 and relatives); a
    suppressed numerator divided straight through, producing a large negative
    "rate" instead of the null the cell actually is. This locks the fix: a
    negative numerator sentinel comes out as NaN, not a negative value.
    """
    fake = pd.DataFrame(
        {
            "B00001_E001": [100.0, -666666666.0],
            "B00001_M001": [10.0, -666666666.0],
            "B00002_E001": [50.0, 40.0],
            "B00002_M001": [5.0, 4.0],
        },
        index=pd.Index(["01001", "01003"], name="fips_code"),
    )
    monkeypatch.setattr(iet, "_download_table", lambda table, year=iet.ACS_YEAR: fake)

    target = iet.ExternalTarget(
        column="fake_rate",
        table="b00001",
        numerator="B00001_E001",
        denominator="B00002_E001",
        denominator_table=None,
        kind="proportion",
        label="fake rate for the sentinel-masking test",
    )
    result = iet._derive(target).set_index("fips_code")

    assert result.loc["01001", "fake_rate"] == pytest.approx(2.0)
    assert result.loc["01001", "fake_rate_se"] >= 0
    assert not pd.isna(result.loc["01001", "fake_rate_se"])

    suppressed = result.loc["01003", "fake_rate"]
    assert pd.isna(suppressed), f"expected NaN for a suppressed numerator, got {suppressed}"
    assert pd.isna(result.loc["01003", "fake_rate_se"])


def test_derive_masks_negative_denominator_sentinel(monkeypatch) -> None:
    """A suppressed denominator (and its MOE) must not survive as a rate or SE.

    Mirrors `test_derive_masks_negative_numerator_sentinel` above, but for the
    denominator side: `safe_denominator = denominator.where(denominator > 0)`
    and `safe_denominator_se = denominator_se.where(denominator >= 0)` are
    separate masking lines from the numerator's, and only the numerator case
    was covered by an existing regression test.
    """
    fake = pd.DataFrame(
        {
            "B00001_E001": [100.0, 80.0],
            "B00001_M001": [10.0, 8.0],
            "B00002_E001": [50.0, -666666666.0],
            "B00002_M001": [5.0, -666666666.0],
        },
        index=pd.Index(["01001", "01003"], name="fips_code"),
    )
    monkeypatch.setattr(iet, "_download_table", lambda table, year=iet.ACS_YEAR: fake)

    target = iet.ExternalTarget(
        column="fake_rate",
        table="b00001",
        numerator="B00001_E001",
        denominator="B00002_E001",
        denominator_table=None,
        kind="proportion",
        label="fake rate for the denominator-se sentinel-masking test",
    )
    result = iet._derive(target).set_index("fips_code")

    assert result.loc["01001", "fake_rate"] == pytest.approx(2.0)
    assert result.loc["01001", "fake_rate_se"] >= 0
    assert not pd.isna(result.loc["01001", "fake_rate_se"])

    suppressed = result.loc["01003", "fake_rate"]
    assert pd.isna(suppressed), f"expected NaN for a suppressed denominator, got {suppressed}"
    assert pd.isna(result.loc["01003", "fake_rate_se"])


def test_download_table_is_memoized(monkeypatch) -> None:
    """A second request for the same table must not hit the network."""
    calls: list[str] = []

    def fake_fetch(table: str, year: int = iet.ACS_YEAR) -> pd.DataFrame:
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


def test_supported_vintages_span_enough_years_for_a_temporal_gap() -> None:
    """`analyze_temporal_transfer.py` needs two vintages, as far apart as possible."""
    assert len(iet.SUPPORTED_ACS_YEARS) >= 2
    assert max(iet.SUPPORTED_ACS_YEARS) - min(iet.SUPPORTED_ACS_YEARS) >= 3
    assert iet.ACS_YEAR in iet.SUPPORTED_ACS_YEARS


def test_pre_table_based_vintage_is_refused() -> None:
    """ACS 2019 publishes only sequence-based files; a silent 404 parse is worse."""
    with pytest.raises(ValueError, match="no table-based summary file"):
        iet._fetch_table_uncached("b19013", year=2019)


def test_vintage_cache_paths_do_not_collide() -> None:
    """Two vintages in one repo must not overwrite each other's parquet."""
    default = iet.vintage_cache_path(iet.ACS_YEAR)
    other = iet.vintage_cache_path(min(iet.SUPPORTED_ACS_YEARS))

    assert default == iet.EXTERNAL_TARGETS_PATH, "default vintage keeps its filename"
    assert other != default
    assert str(min(iet.SUPPORTED_ACS_YEARS)) in other.name


def test_download_table_keys_its_cache_on_the_vintage(monkeypatch) -> None:
    """Scoring two vintages in one process must not serve the second from the first."""
    calls: list[tuple[str, int]] = []

    def fake_fetch(table: str, year: int = iet.ACS_YEAR) -> pd.DataFrame:
        calls.append((table, year))
        return pd.DataFrame(
            {"B00000_E001": [float(year)], "B00000_M001": [0.1]},
            index=pd.Index(["01001"], name="fips_code"),
        )

    monkeypatch.setattr(iet, "_fetch_table_uncached", fake_fetch)
    iet._download_table.cache_clear()

    first = iet._download_table("b00000", 2021)
    second = iet._download_table("b00000", 2024)
    repeat = iet._download_table("b00000", 2021)

    assert calls == [("b00000", 2021), ("b00000", 2024)]
    assert first.iloc[0, 0] != second.iloc[0, 0], "vintages must not share a cache entry"
    pd.testing.assert_frame_equal(first, repeat)
