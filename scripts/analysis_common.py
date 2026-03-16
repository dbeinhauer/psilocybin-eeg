"""
Shared helpers for analysis scripts.

Centralises the data-loading, argument-parsing and display-setup logic
that is common to every analysis workflow (ISC, mean/variance, …).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
    FrequencyBandNames,
)
from src.analysis.data_representations import AnalysisData


# ──────────────────────────────────────────────────────────────────────
# Band ISC thresholds
# ──────────────────────────────────────────────────────────────────────

#: Default broadband ISC significance thresholds for each frequency band.
BAND_ISC_THRESHOLDS: dict[str, float] = {
    FrequencyBandNames.DELTA.value: 0.1,
    FrequencyBandNames.THETA.value: 0.07,
    FrequencyBandNames.ALPHA.value: 0.035,
    FrequencyBandNames.BETA.value: 0.02,
    FrequencyBandNames.GAMMA.value: 0.01,
}


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared across all analysis scripts."""
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[cond.value for cond in ConditionVariants],
        help=f"The condition to process, should be one of {[cond.value for cond in ConditionVariants]}.",
    )
    parser.add_argument(
        "--music_type",
        type=str,
        nargs="+",
        default=[mt.value for mt in MusicTypeVariants],
        choices=[mt.value for mt in MusicTypeVariants],
        help="One or more music types to analyse. Defaults to all available types.",
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        default=False,
        help="When set, load raw files, resample, stack and save before analysis.",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=5.0,
        help="Sliding-window length in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=2.5,
        help="Sliding-window step size in seconds (default: 2.5).",
    )


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_analyzers(
    music_types: Sequence[MusicTypeVariants],
    condition: ConditionVariants,
    exclusion_categories: Sequence[ExclusionCategories],
    process_and_save: bool,
) -> dict:
    """Load (or process & save) and normalise analysers for each music type.

    Returns a dict keyed by the music-type *value* (e.g. ``"CLASSIC"``).
    """
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzers: dict = {}
    for mt in music_types:
        label = mt.value
        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[mt],
            conditions=[condition],
            exclusion_categories=list(exclusion_categories),
        )

        if process_and_save:
            analyzer.load_and_prepare_data(resample_freq=250.0, n_jobs=-1)
            print(f"[{label}] data shape: {analyzer.data.shape}")
            analyzer.save_data()
        else:
            analyzer.load_data(
                info_filename=analyzer.filtered_df[
                    SingleDataMetadata.FILENAME
                ].iloc[0],
            )
            print(f"[{label}] Loaded data shape: {analyzer.data.shape}")

        analyzer.normalize()
        analyzers[label] = analyzer

    return analyzers


def analyzers_to_datasets(analyzers: dict) -> dict[str, AnalysisData]:
    """Convert loaded analysers to AnalysisData instances."""
    datasets = {
        label: a.to_analysis_data(label=label) for label, a in analyzers.items()
    }
    for label, ad in datasets.items():
        print(f"[{label}] {ad}")
    return datasets
