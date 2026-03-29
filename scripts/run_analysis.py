"""
Unified script to run EEG wavelet analysis on preprocessed data.

Supports **wavelet** analyses (power and phase ISC).
Select which analyses to run via the ``--analysis`` flag.

For the ISC analysis use the dedicated script ``scripts/run_isc.py``.
For the mean-variance analysis use the dedicated script
``scripts/run_mean_variance.py``.

Usage examples::

    # Wavelet power analysis (Placebo condition)
    python scripts/run_analysis.py --analysis wavelet_power \\
        --wavelet_data_dir data/processed/psilo_music/wavelets

    # Wavelet phase analysis, single music type (Placebo condition)
    python scripts/run_analysis.py --analysis wavelet_phase \\
        --music_type CLASSIC \\
        --wavelet_data_dir data/processed/psilo_music/wavelets
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
    run_wavelet_workflow,
)
from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    AnalysisVariants,
)
from src.definitions.constants import ProjectPaths

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run EEG wavelet analysis on preprocessed data."),
    )
    add_common_arguments(parser)

    args = parser.parse_args()

    # ── Configuration ─────────────────────────────────────────────
    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    analyses = set(args.analysis)
    run_wavelet_power = AnalysisVariants.WAVELET_POWER.value in analyses
    run_wavelet_phase = AnalysisVariants.WAVELET_PHASE.value in analyses
    run_wavelet = run_wavelet_power or run_wavelet_phase

    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    wavelet_freqs = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )

    # ── Data loading ──────────────────────────────────────────────
    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=False,
    )
    raw_datasets = analyzers_to_datasets(analyzers) if run_wavelet else None

    # ── Wavelet power analysis ────────────────────────────────────
    if run_wavelet_power:
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
            wavelet_dir=Path(args.wavelet_data_dir),
            reuse_wavelets=args.reuse_wavelets,
            keep_frequency_dim=args.wavelet_keep_frequency_dim,
            reshape_frequency_dim=args.wavelet_reshape_frequency_dim,
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )

    # ── Wavelet phase analysis ────────────────────────────────────
    if run_wavelet_phase:
        if raw_datasets is None:
            raise RuntimeError("raw_datasets are required for wavelet phase analysis")
        run_wavelet_workflow(
            raw_datasets,
            analyzers,
            representation="phase",
            freqs=wavelet_freqs,
            save_dir=ProjectPaths.PLOTS_PATH / "WaveletPhaseAnalysis",
            bands=args.wavelet_bands,
            include_broadband=not args.skip_wavelet_broadband,
            wavelet_dir=Path(args.wavelet_data_dir),
            reuse_wavelets=args.reuse_wavelets,
            keep_frequency_dim=args.wavelet_keep_frequency_dim,
            reshape_frequency_dim=args.wavelet_reshape_frequency_dim,
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )

    _logger.info("All requested analyses complete.")
