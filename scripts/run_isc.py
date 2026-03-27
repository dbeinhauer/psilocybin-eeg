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
   - Per-band sliding-window ISC
   - Band-overlap raster

All figures are saved under::

    plots/02-isc-broadband-analysis/<condition>_<music_type>/broadband/
    plots/02-isc-broadband-analysis/<condition>_<music_type>/bands/

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
    compute_mean_field_loo_isc,
    compute_mean_field_pairwise_isc,
    compute_mean_field_sliding_window_isc,
    compute_sliding_window_isc,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExclusionCategories,
    MusicTypeVariants,
)
from src.visualization.isc_plots import (  # noqa: E402
    plot_band_isc_distributions,
    plot_band_mean_isc_bar,
    plot_band_overlap,
    plot_band_sliding_window_isc,
    plot_loo_isc_distribution,
    plot_mean_field_loo_isc,
    plot_mean_field_pairwise_isc,
    plot_mean_field_vs_channel_avg_isc,
    plot_sliding_window_isc,
    print_band_significant_intervals,
    print_data_overview,
    print_significant_intervals,
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
        "--window_sec",
        type=float,
        default=5.0,
        help="Sliding-window length in seconds.",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=2.5,
        help="Sliding-window step size in seconds.",
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
    window_sec: float,
    step_sec: float,
) -> None:
    """Run all broadband ISC sections for one dataset."""
    broadband_dir = save_dir / "broadband"
    broadband_dir.mkdir(parents=True, exist_ok=True)

    # LOO-ISC distribution
    _logger.info(f"[{label}] Computing broadband LOO-ISC …")
    loo_isc, mean_loo_isc = compute_loo_isc(ad.data)
    _logger.info(
        f"[{label}]  loo_isc: {loo_isc.shape}  mean_loo_isc: {mean_loo_isc.mean():.4f}"
    )
    plot_loo_isc_distribution(
        {label: mean_loo_isc},
        ylabel="Number of channels",
        save_path=broadband_dir / "loo_isc_distribution.png",
    )

    # Sliding-window ISC
    _logger.info(f"[{label}] Computing broadband sliding-window ISC …")
    sw_isc, sw_times = compute_sliding_window_isc(
        ad.data,
        window_sec=window_sec,
        step_sec=step_sec,
        sfreq=ad.sfreq,
    )
    _logger.info(f"[{label}]  sw_isc: {sw_isc.shape}  sw_times: {sw_times.shape}")
    plot_sliding_window_isc(
        {label: (sw_isc, sw_times)},
        isc_threshold=isc_threshold,
        feature_axis_label="Channel index",
        save_path=broadband_dir / "sliding_window_isc.png",
    )
    print_significant_intervals(
        {label: (sw_isc, sw_times)},
        isc_threshold=isc_threshold,
    )


def _run_band_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    isc_threshold: float,
    window_sec: float,
    step_sec: float,
) -> None:
    """Run all per-band ISC sections for one dataset."""
    bands_dir = save_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)

    # Per-band LOO-ISC
    _logger.info(f"[{label}] Computing per-band LOO-ISC …")
    band_iscs: dict[str, tuple] = {}
    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        filtered = ad.filter_to_band(l_freq, h_freq)
        loo, mean_isc = compute_loo_isc(filtered.data)
        band_iscs[band] = (loo, mean_isc)
        _logger.info(f"  {band:6s}  loo_isc={loo.shape}  mean={mean_isc.mean():.4f}")

    plot_band_isc_distributions(
        {label: band_iscs},
        bands=FREQUENCY_BANDS,
        feature_axis_label="Number of channels",
        save_path=bands_dir / "band_isc_distributions.png",
    )
    plot_band_mean_isc_bar(
        {label: band_iscs},
        bands=FREQUENCY_BANDS,
        save_path=bands_dir / "band_isc_mean_bar.png",
    )

    # Per-band sliding-window ISC
    _logger.info(f"[{label}] Computing per-band sliding-window ISC …")
    band_sw: dict[str, tuple] = {}
    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        filtered = ad.filter_to_band(l_freq, h_freq)
        tc, times = compute_sliding_window_isc(
            filtered.data,
            window_sec=window_sec,
            step_sec=step_sec,
            sfreq=ad.sfreq,
        )
        band_sw[band] = (tc, times)
        _logger.info(f"  {band:6s}  isc_tc={tc.shape}  times={times.shape}")

    plot_band_sliding_window_isc(
        {label: band_sw},
        bands=FREQUENCY_BANDS,
        isc_threshold=BAND_ISC_THRESHOLDS,
        feature_axis_label="Channel",
        save_path=bands_dir / "band_sliding_window_isc.png",
    )
    print_band_significant_intervals(
        {label: band_sw},
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        default_threshold=isc_threshold,
    )

    # Band overlap (requires broadband sliding-window ISC)
    _logger.info(f"[{label}] Computing broadband ISC for band-overlap plot …")
    sw_isc, sw_times = compute_sliding_window_isc(
        ad.data,
        window_sec=window_sec,
        step_sec=step_sec,
        sfreq=ad.sfreq,
    )
    plot_band_overlap(
        {label: band_sw},
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        broadband_sw={label: (sw_isc, sw_times)},
        broadband_threshold=isc_threshold,
        save_path=bands_dir / "band_overlap.png",
    )


def _run_mean_field_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    isc_threshold: float,
    window_sec: float,
) -> None:
    """Run mean-field ISC analysis (non-overlapping windows) for one dataset."""
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

    # Sliding-window ISC (non-overlapping, step=window)
    _logger.info(f"[{label}] Computing mean-field sliding-window ISC …")
    isc_mf_tc, _ = compute_mean_field_sliding_window_isc(
        ad.data, window_sec, window_sec, ad.sfreq
    )

    # Channel-average non-overlapping ISC for comparison
    sw_isc_ca, _ = compute_sliding_window_isc(
        ad.data, window_sec, window_sec, ad.sfreq
    )
    channel_avg_tc = sw_isc_ca.mean(axis=1)

    plot_mean_field_vs_channel_avg_isc(
        {label: isc_mf_tc},
        {label: channel_avg_tc},
        isc_threshold=isc_threshold,
        window_sec=window_sec,
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
        label = mt.value
        if label not in datasets:
            _logger.warning(f"No data found for music type {label!r}; skipping.")
            continue

        ad = datasets[label]
        _logger.info(f"Dataset [{label}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz")

        save_dir = save_root / f"{condition.value}_{label}"

        _run_broadband_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )

        _run_band_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )

        _run_mean_field_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            isc_threshold=args.isc_threshold,
            window_sec=args.window_sec,
        )

    _logger.info("ISC analysis complete.")
