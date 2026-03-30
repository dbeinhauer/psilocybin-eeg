"""
Standalone script to run the ISC (Inter-Subject Correlation) analysis.

Implements the production version of the workflows demonstrated in
``notebooks/02-isc-broadband-analysis/isc_broadband.ipynb`` and
``notebooks/02-isc-broadband-analysis/isc_bands.ipynb``.

For each requested music type the script runs:

1. **Broadband analysis** (no frequency filtering)

   - LOO-ISC distribution histogram
   - Sliding-window time-resolved ISC with significant interval printout

2. **Per-band analysis** (five EEG frequency bands)

   - Per-band LOO-ISC distributions
   - Per-band mean ISC bar chart
   - Per-band sliding-window ISC (50% overlapping windows, step = window / 2)
   - Band-overlap raster

3. **Mean-field analysis** (spatial mean across all electrodes first)

   - Mean-field LOO-ISC per subject (Pearson + Spearman)
   - Mean-field pairwise ISC heatmap
   - Mean-field vs. channel-average ISC time course

All figures are saved under::

    plots/02-isc-broadband-analysis/<condition>_<music_type>/broadband/
    plots/02-isc-broadband-analysis/<condition>_<music_type>/bands/
    plots/02-isc-broadband-analysis/<condition>_<music_type>/mean_field/

Usage examples::

    # Placebo condition, both music types (default)
    python scripts/run_isc.py

    # Placebo, classical only
    python scripts/run_isc.py --music_type CLASSIC

    # Custom window / step sizes
    python scripts/run_isc.py --window_sec 10.0 --step_sec 5.0

    # Process and save .npy cache first, then analyse
    python scripts/run_isc.py --process_and_save
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    BAND_ISC_THRESHOLDS,
    analyzers_to_datasets,
    load_analyzers,
)
from src.analysis.isc import (  # noqa: E402
    FREQUENCY_BANDS,
    compute_loo_isc,
    compute_loo_isc_spearman,
    compute_mean_field_loo_isc,
    compute_mean_field_pairwise_isc,
    compute_mean_field_sliding_window_isc,
    compute_pairwise_isc,
    compute_pairwise_isc_spearman,
    compute_sliding_window_isc,
    compute_sliding_window_isc_spearman,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExclusionCategories,
    MusicTypeVariants,
)
from src.visualization.isc_plots import (  # noqa: E402
    plot_band_multiscale_sliding_window_isc,
    plot_band_overlap,
    plot_band_pairwise_isc_pearson_vs_spearman,
    plot_band_loo_isc_pearson_vs_spearman,
    plot_loo_isc_pearson_vs_spearman,
    plot_mean_field_loo_isc,
    plot_mean_field_pairwise_isc,
    plot_mean_field_vs_channel_avg_isc,
    plot_multiscale_sliding_window_isc,
    plot_pairwise_isc_pearson_vs_spearman,
    print_data_overview,
)

_logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ISC analysis on preprocessed EEG data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        choices=[c.value for c in ConditionVariants],
        default=ConditionVariants.PLACEBO.value,
        help="Experimental condition.",
    )
    parser.add_argument(
        "--music_type",
        nargs="+",
        choices=[mt.value for mt in MusicTypeVariants],
        default=[MusicTypeVariants.CLASSICAL.value, MusicTypeVariants.PSYTRANCE.value],
        help="One or more music types to analyse.",
    )
    parser.add_argument(
        "--isc_threshold",
        type=float,
        default=0.035,
        help="Broadband ISC significance threshold for significant interval detection.",
    )
    parser.add_argument(
        "--window_fine_sec",
        type=float,
        default=1.0,
        help="Fine sliding-window length in seconds (multi-scale analysis).",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=5.0,
        help="Medium sliding-window length in seconds.",
    )
    parser.add_argument(
        "--window_large_sec",
        type=float,
        default=15.0,
        help="Large sliding-window length in seconds (multi-scale analysis).",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=2.5,
        help="Sliding-window step size in seconds.",
    )
    parser.add_argument(
        "--n_ch_subsample",
        type=int,
        default=64,
        help=(
            "Number of channels randomly subsampled for Spearman ISC computation. "
            "Set to 0 to use all channels (much slower)."
        ),
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        help=(
            "Load raw .fif files, resample, stack, and save .npy caches before "
            "running the analysis.  By default cached files are used."
        ),
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for data loading.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help=(
            "Root directory for output plots. "
            "Defaults to plots/02-isc-broadband-analysis/."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _run_broadband_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    isc_threshold: float,
    window_fine_sec: float,
    window_med_sec: float,
    window_large_sec: float,
    step_sec: float,
    n_ch_subsample: int,
) -> None:
    """Run all broadband ISC sections for one dataset."""
    broadband_dir = save_dir / "broadband"
    broadband_dir.mkdir(parents=True, exist_ok=True)

    data = ad.data
    sfreq = ad.sfreq
    n_subjects, n_channels, n_times = data.shape
    _logger.info(
        f"[{label}] Data shape: {n_subjects} subjects, "
        f"{n_channels} channels, {n_times} time points.",
    )

    # Channel subsampling for Spearman
    if n_ch_subsample > 0 and n_ch_subsample < n_channels:
        rng = np.random.default_rng(42)
        ch_idx = np.sort(rng.choice(n_channels, n_ch_subsample, replace=False))
        data_sub = data[:, ch_idx, :]
        _logger.info(
            f"[{label}] Subsampling {n_ch_subsample}/{n_channels} channels for Spearman."
        )
    else:
        data_sub = data
        n_ch_subsample = n_channels

    # ── Section 1: LOO-ISC (Pearson + Spearman) ──────────────────────────
    _logger.info(f"[{label}] Section 1: LOO-ISC (Pearson + Spearman) …")
    loo_pearson, mean_pearson = compute_loo_isc(data)
    loo_spearman, mean_spearman = compute_loo_isc_spearman(data_sub)

    plot_loo_isc_pearson_vs_spearman(
        label,
        loo_pearson,
        mean_pearson,
        loo_spearman,
        mean_spearman,
        save_path_hist=broadband_dir / f"loo_isc_distribution_{label}.png",
        save_path_violin=broadband_dir / f"loo_isc_per_subject_{label}.png",
    )

    # ── Section 2: Pairwise ISC (Pearson + Spearman) ─────────────────────
    _logger.info(f"[{label}] Section 2: Pairwise ISC (Pearson + Spearman) …")
    pair_pearson = compute_pairwise_isc(data)
    pair_spearman = compute_pairwise_isc_spearman(data_sub)

    plot_pairwise_isc_pearson_vs_spearman(
        label,
        pair_pearson,
        pair_spearman,
        save_path_heatmaps=broadband_dir / f"pairwise_isc_matrix_{label}.png",
        save_path_per_subject=broadband_dir / f"pairwise_isc_per_subject_{label}.png",
        save_path_distribution=broadband_dir / f"pairwise_isc_distribution_{label}.png",
    )

    # ── Section 3: Multi-scale sliding-window ISC ─────────────────────────
    _logger.info(f"[{label}] Section 3: Multi-scale sliding-window ISC …")
    step_fine = window_fine_sec / 2
    step_med = window_med_sec / 2
    step_large = window_large_sec / 2

    _logger.info(f"  Fine   ({window_fine_sec:.0f} s / {step_fine:.1f} s step) …")
    isc_fine, times_fine = compute_sliding_window_isc(
        data_sub, window_fine_sec, step_fine, sfreq
    )
    _logger.info(f"  Medium ({window_med_sec:.0f} s / {step_med:.1f} s step) …")
    isc_med, times_med = compute_sliding_window_isc(
        data_sub, window_med_sec, step_med, sfreq
    )
    _logger.info(f"  Large  ({window_large_sec:.0f} s / {step_large:.1f} s step) …")
    isc_large, times_large = compute_sliding_window_isc(
        data_sub, window_large_sec, step_large, sfreq
    )
    _logger.info(f"  Spearman medium ({window_med_sec:.0f} s / {step_med:.1f} s) …")
    isc_spear_med, _ = compute_sliding_window_isc_spearman(
        data_sub, window_med_sec, step_med, sfreq
    )

    plot_multiscale_sliding_window_isc(
        label,
        isc_fine,
        times_fine,
        isc_med,
        times_med,
        isc_large,
        times_large,
        isc_spear_med,
        sfreq,
        n_times,
        window_fine_sec=window_fine_sec,
        window_med_sec=window_med_sec,
        window_large_sec=window_large_sec,
        isc_threshold=isc_threshold,
        n_ch_subsample=n_ch_subsample if n_ch_subsample < n_channels else None,
        save_path_bar=broadband_dir / f"sw_isc_bar_{label}.png",
        save_path_overlay=broadband_dir / f"sw_isc_overlay_{label}.png",
        save_path_comparison=broadband_dir / f"sw_isc_pearson_vs_spearman_{label}.png",
    )


def _run_band_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    isc_threshold: float,
    window_fine_sec: float,
    window_med_sec: float,
    window_large_sec: float,
    step_sec: float,
    n_ch_subsample: int,
) -> None:
    """Run all per-band ISC sections for one dataset."""
    bands_dir = save_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)

    data = ad.data
    sfreq = ad.sfreq
    _, n_channels, n_times = data.shape

    # Channel subsampling for Spearman (same seed for reproducibility)
    if n_ch_subsample > 0 and n_ch_subsample < n_channels:
        rng = np.random.default_rng(42)
        ch_idx = np.sort(rng.choice(n_channels, n_ch_subsample, replace=False))
        _logger.info(
            f"[{label}] Subsampling {n_ch_subsample}/{n_channels} channels for Spearman."
        )
    else:
        ch_idx = np.arange(n_channels)
        n_ch_subsample = n_channels

    step_fine = window_fine_sec / 2
    step_med = window_med_sec / 2
    step_large = window_large_sec / 2

    # ── Section 1: Per-band LOO-ISC (Pearson + Spearman) ─────────────────
    _logger.info(f"[{label}] Section 1: per-band LOO-ISC (Pearson + Spearman) …")
    band_iscs: dict[str, tuple] = {}
    band_iscs_spearman: dict[str, tuple] = {}
    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        _logger.info(f"  {band:6s} ({l_freq}–{h_freq} Hz) — Pearson …")
        filtered = ad.filter_to_band(l_freq, h_freq)
        loo, mean_isc = compute_loo_isc(filtered.data)
        band_iscs[band] = (loo, mean_isc)

        _logger.info(f"  {band:6s} ({l_freq}–{h_freq} Hz) — Spearman …")
        loo_sp, mean_sp = compute_loo_isc_spearman(filtered.data[:, ch_idx, :])
        band_iscs_spearman[band] = (loo_sp, mean_sp)

    plot_band_loo_isc_pearson_vs_spearman(
        label,
        band_iscs,
        band_iscs_spearman,
        bands=FREQUENCY_BANDS,
        save_path_dir=bands_dir / "loo_isc",
    )

    # ── Section 2: Per-band pairwise ISC (Pearson + Spearman) ────────────
    _logger.info(f"[{label}] Section 2: per-band pairwise ISC (Pearson + Spearman) …")
    band_pair_pearson: dict[str, np.ndarray] = {}
    band_pair_spearman: dict[str, np.ndarray] = {}
    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        _logger.info(f"  {band:6s} — Pearson pairwise …")
        filtered = ad.filter_to_band(l_freq, h_freq)
        band_pair_pearson[band] = compute_pairwise_isc(filtered.data)
        _logger.info(f"  {band:6s} — Spearman pairwise …")
        band_pair_spearman[band] = compute_pairwise_isc_spearman(
            filtered.data[:, ch_idx, :]
        )

    plot_band_pairwise_isc_pearson_vs_spearman(
        label,
        band_pair_pearson,
        band_pair_spearman,
        bands=FREQUENCY_BANDS,
        save_path_dir=bands_dir / "pairwise_isc",
    )

    # ── Section 3: Per-band multi-scale sliding-window ISC ───────────────
    _logger.info(f"[{label}] Section 3: per-band multi-scale sliding-window ISC …")
    band_sw_fine: dict[str, tuple] = {}
    band_sw_med: dict[str, tuple] = {}
    band_sw_large: dict[str, tuple] = {}
    band_sw_spearman_med: dict[str, tuple] = {}

    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        filtered = ad.filter_to_band(l_freq, h_freq)
        fdata_sub = filtered.data[:, ch_idx, :]

        _logger.info(f"  {band:6s} fine   ({window_fine_sec:.0f} s) …")
        tc_fine, t_fine = compute_sliding_window_isc(
            fdata_sub, window_fine_sec, step_fine, sfreq
        )
        band_sw_fine[band] = (tc_fine, t_fine)

        _logger.info(f"  {band:6s} medium ({window_med_sec:.0f} s) …")
        tc_med, t_med = compute_sliding_window_isc(
            fdata_sub, window_med_sec, step_med, sfreq
        )
        band_sw_med[band] = (tc_med, t_med)

        _logger.info(f"  {band:6s} large  ({window_large_sec:.0f} s) …")
        tc_large, t_large = compute_sliding_window_isc(
            fdata_sub, window_large_sec, step_large, sfreq
        )
        band_sw_large[band] = (tc_large, t_large)

        _logger.info(f"  {band:6s} Spearman medium ({window_med_sec:.0f} s) …")
        tc_sp, t_sp = compute_sliding_window_isc_spearman(
            fdata_sub, window_med_sec, step_med, sfreq
        )
        band_sw_spearman_med[band] = (tc_sp, t_sp)

    plot_band_multiscale_sliding_window_isc(
        label,
        band_sw_fine,
        band_sw_med,
        band_sw_large,
        band_sw_spearman_med,
        sfreq,
        n_times,
        window_fine_sec=window_fine_sec,
        window_med_sec=window_med_sec,
        window_large_sec=window_large_sec,
        band_thresholds=BAND_ISC_THRESHOLDS,
        bands=FREQUENCY_BANDS,
        n_ch_subsample=n_ch_subsample if n_ch_subsample < n_channels else None,
        save_path_dir=bands_dir / "sliding_window",
    )

    # ── Section 4: Band-overlap analysis ──────────────────────────────────
    _logger.info(f"[{label}] Section 4: band-overlap analysis …")
    _logger.info(f"  Broadband sliding-window ISC (medium, {window_med_sec:.0f} s) …")
    sw_isc_bb, sw_times_bb = compute_sliding_window_isc(
        data, window_med_sec, step_med, sfreq
    )
    plot_band_overlap(
        {label: band_sw_med},
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        broadband_sw={label: (sw_isc_bb, sw_times_bb)},
        broadband_threshold=isc_threshold,
        save_path=bands_dir / f"band_overlap_{label}.png",
    )


def _run_mean_field_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    isc_threshold: float,
    window_sec: float,
    step_sec: float,
) -> None:
    """Run mean-field ISC analysis for one dataset."""
    mf_dir = save_dir / "mean_field"
    mf_dir.mkdir(parents=True, exist_ok=True)

    # LOO-ISC
    _logger.info(f"[{label}] Computing mean-field LOO-ISC …")
    loo_mf_pearson, loo_mf_spearman = compute_mean_field_loo_isc(ad.data)
    _logger.info(
        f"[{label}]  Pearson mean={loo_mf_pearson.mean():.4f}  "
        f"Spearman mean={loo_mf_spearman.mean():.4f}"
    )
    plot_mean_field_loo_isc(
        {label: loo_mf_pearson},
        {label: loo_mf_spearman},
        save_path=mf_dir / "mean_field_loo_isc.png",
    )

    # Pairwise ISC
    _logger.info(f"[{label}] Computing mean-field pairwise ISC …")
    pair_mf = compute_mean_field_pairwise_isc(ad.data)
    plot_mean_field_pairwise_isc(
        {label: pair_mf},
        save_path=mf_dir / "mean_field_pairwise_isc.png",
    )

    # Sliding-window ISC
    _logger.info(f"[{label}] Computing mean-field sliding-window ISC …")
    isc_mf_tc, _ = compute_mean_field_sliding_window_isc(
        ad.data, window_sec, step_sec, ad.sfreq
    )

    # Channel-average ISC for comparison
    sw_isc_ca, _ = compute_sliding_window_isc(ad.data, window_sec, step_sec, ad.sfreq)
    channel_avg_tc = sw_isc_ca.mean(axis=1)

    plot_mean_field_vs_channel_avg_isc(
        {label: isc_mf_tc},
        {label: channel_avg_tc},
        isc_threshold=isc_threshold,
        window_sec=window_sec,
        step_sec=step_sec,
        save_path=mf_dir / "mean_field_vs_channel_avg_isc.png",
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = (
        args.save_dir
        if args.save_dir is not None
        else ProjectPaths.PLOTS_PATH / "02-isc-broadband-analysis"
    )

    _logger.info(
        f"Starting ISC analysis: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}"
    )

    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=False,
    )
    datasets = analyzers_to_datasets(analyzers)
    print_data_overview(datasets)

    for mt in music_types:
        label = mt.value  # short display label (plot titles, log messages)
        dataset_key = f"{condition.value}_{label}"  # full key used in datasets dict
        if dataset_key not in datasets:
            _logger.warning(f"No data found for music type {label!r}; skipping.")
            continue

        ad = datasets[dataset_key]
        _logger.info(f"Dataset [{label}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz")

        save_dir = save_root / dataset_key

        _run_broadband_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_fine_sec=args.window_fine_sec,
            window_med_sec=args.window_sec,
            window_large_sec=args.window_large_sec,
            step_sec=args.step_sec,
            n_ch_subsample=args.n_ch_subsample,
        )

        _run_band_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_fine_sec=args.window_fine_sec,
            window_med_sec=args.window_sec,
            window_large_sec=args.window_large_sec,
            step_sec=args.step_sec,
            n_ch_subsample=args.n_ch_subsample,
        )

        _run_mean_field_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )

    _logger.info("ISC analysis complete.")
