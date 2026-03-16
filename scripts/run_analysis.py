"""
Unified script to run EEG analysis on preprocessed data.

Supports both **ISC** (Inter-Subject Correlation) and **mean/variance**
analyses in a single invocation.  Select which analyses to run via the
``--analysis`` flag (defaults to both).

Replicates the workflows from ``notebooks/data_analysis.ipynb`` and
``notebooks/mean_variance_analysis.ipynb`` so that they can be executed
as a stand-alone command (e.g. inside a Metacentrum PBS job).

Usage examples::

    # Run both analyses for all music types under Placebo
    python scripts/run_analysis.py

    # ISC only
    python scripts/run_analysis.py --analysis isc

    # Mean/variance only, custom sliding window
    python scripts/run_analysis.py --analysis mean_variance --window_sec 10 --step_sec 5

    # Both analyses for a single music type under Psilocybin
    python scripts/run_analysis.py --condition Psilocybin --music_type CLASSIC
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (
    add_common_arguments,
    load_analyzers,
    analyzers_to_datasets,
    run_isc_workflow,
    run_mean_variance_workflow,
)
from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
)
from src.definitions.constants import ProjectPaths

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run EEG analysis (ISC and/or mean-variance) on preprocessed data."
        ),
    )
    add_common_arguments(parser)

    args = parser.parse_args()

    # ── Configuration ─────────────────────────────────────────────
    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [ExclusionCategories.BAD_MUSIC]
    analyses = set(args.analysis)

    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    # ── Data loading (shared across both analyses) ────────────────
    analyzers = load_analyzers(
        music_types, condition, exclusion_categories, args.process_and_save
    )
    datasets = analyzers_to_datasets(analyzers)

    # ── ISC analysis ──────────────────────────────────────────────
    if "isc" in analyses:
        run_isc_workflow(
            datasets,
            save_dir=ProjectPaths.PLOTS_PATH / "OverallAnalysis",
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )

    # ── Mean / variance analysis ──────────────────────────────────
    if "mean_variance" in analyses:
        run_mean_variance_workflow(
            datasets,
            analyzers,
            save_dir=ProjectPaths.PLOTS_PATH / "MeanVarianceAnalysis",
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )

    _logger.info("All requested analyses complete.")
