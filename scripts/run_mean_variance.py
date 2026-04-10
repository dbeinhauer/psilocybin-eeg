"""
Standalone script to run the intersubject mean-variance analysis.

Implements the production version of the workflows demonstrated in
``notebooks/01-raw-mean-variance-analysis/mean_variance_broadband.ipynb`` and
``notebooks/01-raw-mean-variance-analysis/mean_variance_bands.ipynb``.

For each requested music type the script runs:

1. **Raw analysis** (no frequency filtering)

   - Section 1: intersubject time-series overview
   - Section 2: intersubject variance distribution histogram
   - Section 4: windowed bar-chart and overlay figures

2. **Per-band analysis** (five EEG frequency bands)

   - Section 1: per-band intersubject time-series overview
   - Section 2: per-band variance distribution histograms
   - Section 3: pairwise ISC matrices per band
   - Section 4: per-band windowed synchrony analysis

All figures are saved under::

    plots/01-raw-mean-variance-analysis/<condition>_<music_type>/broadband/<analysis_type>/
    plots/01-raw-mean-variance-analysis/<condition>_<music_type>/bands/<analysis_type>/

where for **broadband** outputs ``<analysis_type>`` is one of ``timeseries``,
``variance``, or ``windowed``, and for **per-band** outputs ``<analysis_type>``
is one of ``timeseries``, ``variance``, ``windowed``, or ``isc_matrices``.

Usage examples::

    # Placebo condition, both music types (default)
    python scripts/run_mean_variance.py

    # Classical only, 3-second windows with 1.5 s step (50% overlap)
    python scripts/run_mean_variance.py --music_type CLASSIC --window_sec 3.0

    # Process and save .npy cache first, then analyse
    python scripts/run_mean_variance.py --process_and_save
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import load_analyzers, analyzers_to_datasets
from src.analysis.mean_variance import (
    compute_intersubject_stats,
    compute_windowed_stats,
    compute_band_intersubject_stats,
    compute_pairwise_isc_matrices,
    FREQUENCY_BANDS,
)
from src.analysis.results_store import (
    save_intersubject_timeseries,
    save_windowed_stats,
    save_pairwise_isc,
)
from src.visualization.mean_variance_plots import (
    plot_timeseries,
    plot_variance_distribution,
    plot_windowed_analysis,
    plot_band_timeseries,
    plot_band_variance_distributions,
    plot_isc_matrices,
    plot_band_windowed_analysis,
)
from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
)
from src.definitions.constants import ProjectPaths

_logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the intersubject mean-variance analysis on preprocessed EEG data."
        ),
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
        "--window_sec",
        type=float,
        default=2.0,
        help="Window length in seconds for windowed analysis.",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=None,
        help=(
            "Step between successive windows in seconds. "
            "Defaults to window_sec / 2 (50%% overlap)."
        ),
    )
    parser.add_argument(
        "--sync_percentile",
        type=float,
        default=10.0,
        help=(
            "Percentile of var_t used as the synchrony-candidate threshold. "
            "Windows whose mean variance is below this percentile are highlighted."
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
            "Defaults to plots/01-raw-mean-variance-analysis/."
        ),
    )
    parser.add_argument(
        "--results_db_dir",
        type=Path,
        default=None,
        help=(
            "Root directory for the CSV results database (Interactive Explorer). "
            "Defaults to results_db/01-raw-mean-variance-analysis/."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _run_raw_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    window_sec: float,
    sync_percentile: float,
    step_sec: float | None = None,
    *,
    condition: str = "",
    music_type: str = "",
    results_db_dir: Path | None = None,
) -> None:
    """Run all raw (broadband) mean-variance sections for one dataset."""
    broadband_timeseries_dir = save_dir / "broadband" / "timeseries"
    broadband_timeseries_dir.mkdir(parents=True, exist_ok=True)
    broadband_variance_dir = save_dir / "broadband" / "variance"
    broadband_variance_dir.mkdir(parents=True, exist_ok=True)
    broadband_windowed_dir = save_dir / "broadband" / "windowed"
    broadband_windowed_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(f"[{label}] Computing intersubject statistics …")
    stats = compute_intersubject_stats(ad.data)
    n_times = ad.data.shape[2]
    sfreq = ad.sfreq

    # Section 1 — time series
    _logger.info(f"[{label}] Section 1: intersubject time series")
    plot_timeseries(
        stats,
        sfreq,
        label,
        save_path=broadband_timeseries_dir / "timeseries.png",
    )

    # Section 2 — variance distribution
    _logger.info(f"[{label}] Section 2: variance distribution")
    plot_variance_distribution(
        stats["inter_var"],
        label,
        save_path=broadband_variance_dir / "variance_distribution.png",
    )

    # Section 4 — windowed analysis
    _logger.info(f"[{label}] Section 4: windowed analysis")
    df_wins = compute_windowed_stats(
        stats,
        n_times=n_times,
        sfreq=sfreq,
        window_sec=window_sec,
        sync_percentile=sync_percentile,
        step_sec=step_sec,
    )
    n_sync = df_wins["sync_candidate"].sum()
    _logger.info(
        f"[{label}] Synchrony candidates: {n_sync}/{len(df_wins)} windows "
        f"(threshold = {sync_percentile}th pct)"
    )
    plot_windowed_analysis(
        stats,
        df_wins,
        sfreq,
        label,
        window_sec=window_sec,
        sync_percentile=sync_percentile,
        step_sec=step_sec,
        save_path_bar=broadband_windowed_dir / "windowed_bar.png",
        save_path_overlay=broadband_windowed_dir / "windowed_overlay.png",
    )

    # ── Export CSV results for Interactive Explorer ────────────────────────
    if results_db_dir is not None:
        db_dir = results_db_dir / "broadband"
        save_intersubject_timeseries(
            stats,
            sfreq,
            db_dir,
            condition=condition,
            music_type=music_type,
        )
        save_windowed_stats(
            df_wins,
            db_dir,
            condition=condition,
            music_type=music_type,
        )


def _run_band_analysis(
    ad: "AnalysisData",  # noqa: F821
    label: str,
    save_dir: Path,
    window_sec: float,
    sync_percentile: float,
    step_sec: float | None = None,
    *,
    condition: str = "",
    music_type: str = "",
    results_db_dir: Path | None = None,
) -> None:
    """Run all per-band mean-variance sections for one dataset."""
    bands_timeseries_dir = save_dir / "bands" / "timeseries"
    bands_timeseries_dir.mkdir(parents=True, exist_ok=True)
    bands_variance_dir = save_dir / "bands" / "variance"
    bands_variance_dir.mkdir(parents=True, exist_ok=True)
    bands_isc_dir = save_dir / "bands" / "isc_matrices"
    bands_isc_dir.mkdir(parents=True, exist_ok=True)
    bands_windowed_dir = save_dir / "bands" / "windowed"
    bands_windowed_dir.mkdir(parents=True, exist_ok=True)

    sfreq = ad.sfreq
    n_subjects = ad.data.shape[0]

    _logger.info(f"[{label}] Bandpass-filtering data …")
    band_stats = compute_band_intersubject_stats(ad, FREQUENCY_BANDS)

    # Section 1 — per-band time series
    _logger.info(f"[{label}] Section 1: per-band intersubject time series")
    plot_band_timeseries(
        band_stats,
        sfreq,
        label,
        sync_percentile=sync_percentile,
        bands=FREQUENCY_BANDS,
        save_path=bands_timeseries_dir / "band_timeseries.png",
    )

    # Section 2 — per-band variance distributions
    _logger.info(f"[{label}] Section 2: per-band variance distributions")
    plot_band_variance_distributions(
        band_stats,
        label,
        bands=FREQUENCY_BANDS,
        save_path=bands_variance_dir / "band_variance_distributions.png",
    )

    # Section 3 — pairwise ISC matrices
    _logger.info(f"[{label}] Section 3: pairwise ISC matrices per band")
    band_data_dict = {
        band: ad.filter_to_band(l_freq, h_freq).data
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items()
    }
    isc_matrices = compute_pairwise_isc_matrices(band_data_dict)
    plot_isc_matrices(
        isc_matrices,
        n_subjects=n_subjects,
        label=label,
        bands=FREQUENCY_BANDS,
        save_path=bands_isc_dir / "isc_matrices.png",
    )

    # Section 4 — per-band windowed analysis
    _logger.info(f"[{label}] Section 4: per-band windowed analysis")
    plot_band_windowed_analysis(
        band_stats,
        sfreq,
        label,
        window_sec=window_sec,
        sync_percentile=sync_percentile,
        step_sec=step_sec,
        bands=FREQUENCY_BANDS,
        save_path_bar=bands_windowed_dir / "band_windowed_bar.png",
        save_path_summary=bands_windowed_dir / "band_windowed_summary.png",
        save_path_per_band_dir=bands_windowed_dir,
    )

    # ── Export CSV results for Interactive Explorer ────────────────────────
    if results_db_dir is not None:
        for band, stats in band_stats.items():
            db_dir = results_db_dir / "bands" / band
            save_intersubject_timeseries(
                stats,
                sfreq,
                db_dir,
                condition=condition,
                music_type=music_type,
                band=band,
            )
            n_band_times = len(stats["mean_t"])
            df_band_wins = compute_windowed_stats(
                stats,
                n_times=n_band_times,
                sfreq=sfreq,
                window_sec=window_sec,
                sync_percentile=sync_percentile,
                step_sec=step_sec,
            )
            save_windowed_stats(
                df_band_wins,
                db_dir,
                condition=condition,
                music_type=music_type,
                band=band,
            )
        for band, matrix in isc_matrices.items():
            db_dir = results_db_dir / "bands" / band
            save_pairwise_isc(
                matrix,
                db_dir,
                condition=condition,
                music_type=music_type,
                band=band,
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
        else ProjectPaths.PLOTS_PATH / "01-raw-mean-variance-analysis"
    )
    results_db_root = (
        args.results_db_dir
        if args.results_db_dir is not None
        else ProjectPaths.RESULTS_DB_PATH / "01-raw-mean-variance-analysis"
    )

    _logger.info(
        f"Starting mean-variance analysis: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}"
    )

    # Load and normalise data
    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=True,
    )
    datasets = analyzers_to_datasets(analyzers)

    for mt in music_types:
        label = mt.value  # short display label (plot titles, log messages)
        dataset_key = f"{condition.value}_{label}"  # full key used in datasets dict
        if dataset_key not in datasets:
            _logger.warning(f"No data found for music type {label!r}; skipping.")
            continue

        ad = datasets[dataset_key]
        _logger.info(f"Dataset [{label}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz")

        save_dir = save_root / dataset_key

        results_db_dir = results_db_root / dataset_key

        _run_raw_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            window_sec=args.window_sec,
            sync_percentile=args.sync_percentile,
            step_sec=args.step_sec,
            condition=condition.value,
            music_type=label,
            results_db_dir=results_db_dir,
        )

        _run_band_analysis(
            ad,
            label=label,
            save_dir=save_dir,
            window_sec=args.window_sec,
            sync_percentile=args.sync_percentile,
            step_sec=args.step_sec,
            condition=condition.value,
            music_type=label,
            results_db_dir=results_db_dir,
        )

    _logger.info("Mean-variance analysis complete.")
