"""
CLI script fitting Correlated Component Analysis (CorrCA) of wavelet power on **one
condition at a time**: Placebo alone, then Psilocybin alone.

The time-axis join of ``run_corrca_condition_tracks.py`` reduced to a single track,
and run once per condition::

    subjects = participants        # every participant the condition has
    samples  = time × frequency    # that condition's track only
    features = channel             # ONE filter per component, per condition

Each condition gets its own filters, so ``CC 1`` of Placebo and ``CC 1`` of
Psilocybin are two different spatial combinations, not one component seen twice:
compare them by their topographies and their synchrony, not row by row. Nothing is
paired across the conditions, so each fit keeps every participant of its condition,
not only the matched ones, and the paired ISC test is not drawn. The sign anchor,
the ROI and non-ROI fits, the masked maps and the figures are the tracks run's (``--zscore_mode`` is moot with one
track).

Output::

    plots/06b-corrca-condition-comparison/IndividualConditions_<MusicType>/
        broadband/corrca_channel_individual_conditions/<Condition>/pca_<n_pca>/
        broadband/corrca_channel_individual_conditions_roi/<Condition>/pca_<n_pca>/
        broadband/corrca_channel_individual_conditions_non_roi/<Condition>/pca_<n_pca>/
    data/processed/<experiment>/iva_results/<Condition>/
        corrca_channel_individual_conditions__<MusicType>__<spectrum>__pca<n_pca>.npz

The store is a one-segment tracks store, so ``run_corrca_snr_grid.py --variant
corrca_channel_individual_conditions`` reads it as it reads the join.

Examples::

    python scripts/run_corrca_individual_conditions.py \\
        --experiment assr --n_pca 5 --reuse_wavelets --store_components

    python scripts/run_corrca_individual_conditions.py \\
        --experiment assr --conditions Psilocybin --n_pca 5 --n_channels 32 \\
        --n_times 3000 --n_pairs 5 --lenient_mask --reuse_wavelets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    add_corrca_condition_args,
)
from scripts.run_corrca_condition_tracks import (  # noqa: E402
    run_corrca_condition_tracks,
)
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
)

_logger = logging.getLogger(__name__)

#: The run's raw-power full fit; its log-power, ROI and non-ROI fits follow it.
_VARIANT = IvaVariants.CORRCA_CHANNEL_INDIVIDUAL_CONDITIONS


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CorrCA on each condition alone (its own filters, every participant "
            "it has) and draw the per-component figures and synchrony."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_corrca_condition_args(parser)
    # One track: per-condition and joint z-scoring coincide.
    parser.set_defaults(zscore_mode="per_condition")
    return parser


def run(args: argparse.Namespace) -> None:
    """Run the analysis for an already-parsed argument namespace.

    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    """
    experiment_name = ExperimentNames(args.experiment)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]
    conditions = [ConditionVariants(c) for c in dict.fromkeys(args.conditions)]
    _logger.info(
        f"CorrCA individual conditions: experiment={experiment_name.value}, "
        f"conditions={[c.value for c in conditions]}, "
        f"music_types={[m.value for m in music_types]}, n_pca={args.n_pca}, "
        f"gamma={args.gamma}"
    )
    for music_type in music_types:
        for condition in conditions:
            _logger.info(f"=== {condition.value} {music_type.value} ===")
            run_corrca_condition_tracks(
                music_type,
                args,
                conditions=[condition],
                full=_VARIANT,
            )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging and run.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
