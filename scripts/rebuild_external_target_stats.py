"""Rebuild `external_target_stats.json` from the committed CSVs, without re-fitting.

Every statistic in that artifact is a pure function of four score frames the
sweep already wrote to `outputs/`. A change to how those frames are *summarized*
-- a new basket split, an added interval, a corrected sign -- therefore does not
need the 25-minute sweep that produced them.

The rebuild goes through `analyze_external_target.assemble_stats`, the same
function `main` calls, rather than a parallel copy of the assembly logic. That
is the point: a rebuild that reproduced the artifact by a different route would
be worth less than no rebuild at all, because it could agree with the sweep
today and diverge from it silently later.

**This does not substitute for re-running when the models change.** If anything
touched a design, an estimator, a fold, or the target basket, the CSVs are stale
and this script would launder stale numbers into a fresh-looking artifact. Use
it only when the change is confined to summary code.

`--check` rebuilds and diffs against the committed artifact without writing,
which is how to confirm that a full re-run would land in the same place.

Usage:
    uv run python scripts/rebuild_external_target_stats.py [--check]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from analyze_external_target import (
    DECILE_PATH,
    PLACEBO_PATH,
    SCORES_PATH,
    STATS_PATH,
    TRAINING_SIZE_PATH,
    assemble_stats,
    configure_logging,
)

logger = logging.getLogger(__name__)


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the four score frames the sweep wrote.

    Returns:
        Tuple of (scores, deciles, placebos, sizes).

    Raises:
        FileNotFoundError: If any frame is missing. `sizes` is the one that may
            legitimately be absent, on an artifact written before it had a CSV;
            the caller is told to recover it from the existing JSON rather than
            silently getting a stats file with the section missing.
    """
    missing = [path for path in (SCORES_PATH, DECILE_PATH, PLACEBO_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(p.name for p in missing)} absent -- run the sweep first"
        )

    if TRAINING_SIZE_PATH.exists():
        sizes = pd.read_csv(TRAINING_SIZE_PATH)
    elif STATS_PATH.exists():
        # Artifacts written before `TRAINING_SIZE_PATH` existed carry this frame
        # only inside the JSON. Recovering it keeps the rebuild lossless instead
        # of dropping a section nothing else can regenerate without a refit.
        previous = json.loads(STATS_PATH.read_text())
        sizes = pd.DataFrame(previous.get("by_training_size", []))
        logger.warning(
            "%s absent; recovered %d training-size rows from the previous artifact",
            TRAINING_SIZE_PATH.name,
            len(sizes),
        )
    else:
        raise FileNotFoundError(
            f"{TRAINING_SIZE_PATH.name} absent and no previous artifact to recover it from"
        )

    return (
        pd.read_csv(SCORES_PATH),
        pd.read_csv(DECILE_PATH),
        pd.read_csv(PLACEBO_PATH),
        sizes,
    )


def main() -> int:
    """Rebuild the stats artifact, or check it against what a rebuild would produce.

    Returns:
        Process exit status: 0 on success, 1 when `--check` finds a difference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="diff against the committed artifact instead of writing it",
    )
    arguments = parser.parse_args()
    configure_logging()

    scores, deciles, placebos, sizes = load_frames()
    logger.info(
        "rebuilding from %d score rows across %d targets",
        len(scores),
        scores["target"].nunique(),
    )
    stats = assemble_stats(scores, deciles, placebos, sizes)
    rendered = json.dumps(stats, indent=2)

    if arguments.check:
        if not STATS_PATH.exists():
            logger.error("%s does not exist; nothing to check against", STATS_PATH)
            return 1
        if STATS_PATH.read_text() == rendered:
            logger.info("%s matches a rebuild from the CSVs", STATS_PATH.name)
            return 0
        logger.error(
            "%s DIFFERS from a rebuild from the CSVs -- either the summary code "
            "moved since the artifact was written, or the CSVs did",
            STATS_PATH.name,
        )
        return 1

    STATS_PATH.write_text(rendered, encoding="utf-8")
    logger.info("wrote %s", STATS_PATH)
    logger.info(
        "headline basket %d targets, wide basket %d of %d scored",
        stats["headline_basket"]["n_targets"],
        stats["n_targets"],
        stats["n_targets_scored"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
