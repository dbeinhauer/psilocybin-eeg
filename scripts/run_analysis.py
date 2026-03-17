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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (
    add_common_arguments,
    load_analyzers,
    analyzers_to_datasets,
    run_isc_workflow,
    run_mean_variance_workflow,
    run_wavelet_workflow,
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
    run_wavelet = "wavelet_power" in analyses

    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    wavelet_freqs = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )

    # ── Data loading (shared across both analyses) ────────────────
    analyzers = load_analyzers(
        music_types, condition, exclusion_categories, args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=not run_wavelet,
    )
    raw_datasets = analyzers_to_datasets(analyzers) if run_wavelet else None

    run_standard = "isc" in analyses or "mean_variance" in analyses
    if run_standard and run_wavelet:
        for analyzer in analyzers.values():
            analyzer.normalize()
        datasets = analyzers_to_datasets(analyzers)
    elif run_standard:
        datasets = analyzers_to_datasets(analyzers)
    else:
        datasets = {}

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

    # ── Wavelet power analysis ────────────────────────────────────
    if "wavelet_power" in analyses:
        if raw_datasets is None:
            raise RuntimeError("raw_datasets are required for wavelet power analysis")
        run_wavelet_workflow(
            raw_datasets,
            analyzers,
            representation="power",
            freqs=wavelet_freqs,
            save_dir=ProjectPaths.PLOTS_PATH / "WaveletPowerAnalysis",
            bands=args.wavelet_bands,
            include_broadband=not args.skip_wavelet_broadband,
            cache_dir=(
                Path(args.wavelet_cache_dir)
                if args.wavelet_cache_dir
                else None
            ),
            reuse_cache=args.reuse_wavelet_cache,
            keep_frequency_dim=args.wavelet_keep_frequency_dim,
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )

    _logger.info("All requested analyses complete.")
