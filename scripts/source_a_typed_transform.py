"""The pre-registered capacity pass over Source A's typed columns.

Both representations are scored under ridge, so 29 raw columns against 384 dense
dimensions is not an equal-capacity comparison. A typed win under that setup
would be partly an artifact of the encoder's extra flexibility, and a typed loss
would be partly an artifact of the typed block's rigidity.

The two transforms here are fixed before any decision-basket target is scored,
and are chosen from how the columns are constructed rather than from what they
predict:

1. **`log1p` on count columns.** Lexicon counts are bounded below at zero and
   right-skewed; ridge fits a linear coefficient to them. This is the standard
   remedy and needs no justification from the data.
2. **`sec_n_industry_mentions` x tier.** That single column carries 97.6% of the
   section block's gain (`source_a_next_steps.md`), and tier is defined by
   article length, so the same count means something different in a stub and in a
   rich article. The interaction says so explicitly.

Nothing else is added. This is a capacity control, not a feature search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Tiers that get an interaction term. `stub` is the reference level and is
# omitted, as a dummy-coded set must be to stay full rank alongside the main
# effect.
INTERACTION_TIERS: tuple[str, ...] = ("thin", "mid", "rich")

# The column whose tier interaction is pre-registered.
INTERACTION_COLUMN: str = "sec_n_industry_mentions"


def _is_count_column(values: pd.Series) -> bool:
    """Decide whether a column is a count rather than a flag or a ratio.

    A flag is a column whose non-null values are drawn from {0, 1}; everything
    else non-negative is treated as a count. Distinct-value counting (as
    opposed to checking the value set directly) breaks on small samples --
    two rows can never have more than two distinct values regardless of what
    the column measures -- so the value set is checked directly instead.

    Args:
        values: The column.

    Returns:
        True when the column is non-negative and is not a {0, 1} flag.
    """
    finite = values.dropna()
    if finite.empty:
        return False
    is_flag = set(finite.unique()) <= {0.0, 1.0}
    return bool((finite >= 0).all() and not is_flag)


def transform_typed(
    frame: pd.DataFrame, typed_columns: list[str], tier: pd.Series
) -> tuple[np.ndarray, list[str]]:
    """Expand the typed block with its pre-registered transforms.

    Args:
        frame: Rows carrying every column in `typed_columns`.
        typed_columns: Source A's shipped column names.
        tier: Tier label per row, aligned to `frame`.

    Returns:
        The expanded design and its column names, in matching order.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []

    for column in typed_columns:
        values = frame[column].astype(float)
        columns.append(values.to_numpy())
        names.append(column)
        if _is_count_column(values):
            columns.append(np.log1p(values.clip(lower=0.0)).to_numpy())
            names.append(f"log1p_{column}")

    if INTERACTION_COLUMN in typed_columns:
        base = frame[INTERACTION_COLUMN].astype(float).to_numpy()
        tier_values = tier.to_numpy()
        for label in INTERACTION_TIERS:
            columns.append(np.where(tier_values == label, base, 0.0))
            names.append(f"{INTERACTION_COLUMN}_x_{label}")

    return np.column_stack(columns), names
