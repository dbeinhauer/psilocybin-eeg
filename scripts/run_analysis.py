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
        --wavelet_data_dir data/processed

    # Wavelet phase analysis, single music type (Placebo condition)
    python scripts/run_analysis.py --analysis wavelet_phase \\
        --music_type CLASSIC \\
        --wavelet_data_dir data/processed

Note: --wavelet_data_dir is a base data directory; the
``<experiment>/wavelets`` suffix is appended automatically.
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
    precompute_pre_alignment_wavelet_cache,
    resolve_wavelet_dir,
    run_wavelet_workflow,
)
from src.definitions.fields import (
    SpectrumTypeVariants,
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    AnalysisVariants,
    ExperimentNames,
)
from src.preprocessing.stimulus_alignment import EXPERIMENT_STIMULUS_MARKERS
from src.definitions.constants import ProjectPaths

_logger = logging.getLogger(__name__)

_STAGE_DIR = "03-wavelet-analysis"

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run EEG wavelet analysis on preprocessed data."),
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Base directory for output plots. Defaults to project plots/ root.",
    )

    args = parser.parse_args()

    # Early validation: reshape requires keep
    if args.wavelet_reshape_frequency_dim and not args.wavelet_keep_frequency_dim:
        parser.error(
            "--wavelet_reshape_frequency_dim requires --wavelet_keep_frequency_dim"
        )

    # ── Configuration ─────────────────────────────────────────────
    experiment_name = ExperimentNames(args.experiment)
    condition = ConditionVariants(args.condition)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        # ASSR has no music dimension; uses a single placeholder "music type".
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    analyses = set(args.analysis)
    run_wavelet_power = AnalysisVariants.WAVELET_POWER.value in analyses
    run_wavelet_phase = AnalysisVariants.WAVELET_PHASE.value in analyses
    run_wavelet = run_wavelet_power or run_wavelet_phase

    WINDOW_SEC = args.window_sec
    WINDOW_FINE_SEC = args.window_fine_sec
    WINDOW_LARGE_SEC = args.window_large_sec
    STEP_SEC = args.step_sec
    N_CH_SUBSAMPLE = args.n_ch_subsample

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
        experiment_name=experiment_name,
    )
    raw_datasets = analyzers_to_datasets(analyzers) if run_wavelet else None

    # Canonical plot root shared by both power and phase workflows. Each
    # analysis writes under the same ``03-wavelet-analysis`` tree so the
    # Results Browser picks them up using the same notebook-based ID.
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    wavelet_plots_root = save_root / _STAGE_DIR
    # Shared wavelet cache directory — power and phase outputs coexist since
    # the representation is encoded in the cached filename. This lets the
    # power workflow reuse a pre-computed phase cache for the power-vs-phase
    # joint plot.
    wavelet_cache_root = resolve_wavelet_dir(args.wavelet_data_dir, experiment_name)

    # ── Pre-alignment wavelet cache (stimulus-based experiments) ──
    # For experiments that use stimulus-based alignment (e.g. ASSR), wavelets
    # computed on the spliced RAW_CROPPED signal suffer from edge artifacts at
    # every splice point.  When the cache is not being reused, pre-compute the
    # wavelet transform on the continuous RAW_AFTER_ICA data and trim
    # afterwards, then tell the workflow to reuse the just-saved cache.
    reuse_wavelets_for_workflow = args.reuse_wavelets
    if (
        run_wavelet
        and experiment_name in EXPERIMENT_STIMULUS_MARKERS
        and not args.reuse_wavelets
    ):
        representations = (["power"] if run_wavelet_power else []) + (
            ["phase"] if run_wavelet_phase else []
        )
        # The workflow (and all Stage-04/05 consumers) read the broadband
        # wavelet cache from ``<wavelet_dir>/broadband`` (see
        # ``_broadband_wavelet_4d``). Write the pre-aligned cache to that same
        # subdirectory so it is actually reused downstream — otherwise the
        # workflow would silently recompute wavelets from the stimulus-spliced
        # RAW_CROPPED signal, reintroducing the splice-edge artifacts this
        # pre-alignment step exists to avoid.
        precompute_pre_alignment_wavelet_cache(
            analyzers,
            freqs=wavelet_freqs,
            representations=representations,
            wavelet_dir=wavelet_cache_root / SpectrumTypeVariants.BROADBAND.value,
            resample_freq=250.0,
            n_jobs=args.n_jobs,
        )
        reuse_wavelets_for_workflow = True

    # ── Wavelet power analysis ────────────────────────────────────
    if run_wavelet_power:
        if raw_datasets is None:
            raise RuntimeError("raw_datasets are required for wavelet power analysis")
        run_wavelet_workflow(
            raw_datasets,
            analyzers,
            representation="power",
            freqs=wavelet_freqs,
            save_dir=wavelet_plots_root,
            bands=args.wavelet_bands,
            include_broadband=not args.skip_wavelet_broadband,
            wavelet_dir=wavelet_cache_root,
            reuse_wavelets=reuse_wavelets_for_workflow,
            keep_frequency_dim=args.wavelet_keep_frequency_dim,
            reshape_frequency_dim=args.wavelet_reshape_frequency_dim,
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            window_fine_sec=WINDOW_FINE_SEC,
            window_large_sec=WINDOW_LARGE_SEC,
            n_ch_subsample=N_CH_SUBSAMPLE,
            cross_representation_wavelet_dir=wavelet_cache_root,
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
            save_dir=wavelet_plots_root,
            bands=args.wavelet_bands,
            include_broadband=not args.skip_wavelet_broadband,
            wavelet_dir=wavelet_cache_root,
            reuse_wavelets=reuse_wavelets_for_workflow,
            keep_frequency_dim=args.wavelet_keep_frequency_dim,
            reshape_frequency_dim=args.wavelet_reshape_frequency_dim,
            isc_threshold=args.isc_threshold,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            window_fine_sec=WINDOW_FINE_SEC,
            window_large_sec=WINDOW_LARGE_SEC,
            n_ch_subsample=N_CH_SUBSAMPLE,
        )

    _logger.info("All requested analyses complete.")
