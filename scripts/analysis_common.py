"""
Shared helpers and workflow functions for the unified analysis script.

Centralises data-loading, argument-parsing, display-setup and the actual
analysis workflows (ISC and mean/variance) so they can be driven by a single
entry-point script or from a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

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
from src.analysis.isc import (
    compute_loo_isc,
    compute_sliding_window_isc,
    compute_mean_variance,
    compute_sliding_window_mean_variance,
    FREQUENCY_BANDS,
)
from src.visualization.isc_plots import (
    plot_loo_isc_distribution,
    plot_sliding_window_isc,
    print_significant_intervals,
    plot_band_isc_distributions,
    plot_band_mean_isc_bar,
    plot_band_sliding_window_isc,
    print_band_significant_intervals,
    plot_band_overlap,
    print_data_overview,
    plot_mean_variance_distribution,
    plot_sliding_window_mean_variance,
    plot_band_mean_variance_distributions,
    plot_band_sliding_window_mean_variance,
)

if TYPE_CHECKING:
    from src.analysis.summary import EEGSummarizedAnalyzer


# ──────────────────────────────────────────────────────────────────────
# Constants
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
        "--analysis",
        type=str,
        nargs="+",
        default=["isc", "mean_variance"],
        choices=["isc", "mean_variance"],
        help=(
            "Which analyses to run. Defaults to both. "
            "Choices: isc, mean_variance."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[cond.value for cond in ConditionVariants],
        help=(
            "The condition to process, should be one of "
            f"{[cond.value for cond in ConditionVariants]}."
        ),
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
        "--isc_threshold",
        type=float,
        default=0.035,
        help="Broadband ISC significance threshold (default: 0.035).",
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
) -> dict[str, EEGSummarizedAnalyzer]:
    """Load (or process & save) and normalise analysers for each music type.

    Returns a dict keyed by the music-type *value* (e.g. ``"CLASSIC"``).
    """
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzers: dict[str, EEGSummarizedAnalyzer] = {}
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


# ──────────────────────────────────────────────────────────────────────
# ISC workflow
# ──────────────────────────────────────────────────────────────────────


def run_isc_workflow(
    datasets: dict[str, AnalysisData],
    *,
    save_dir: Path,
    isc_threshold: float = 0.035,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
) -> None:
    """Run the full ISC analysis and save figures.

    Steps:
    1. Data overview
    2. Broadband LOO-ISC — distribution plot
    3. Broadband sliding-window ISC — time-resolved plot & significant intervals
    4. Per-band LOO-ISC — distribution & bar-chart plots
    5. Per-band sliding-window ISC — time-resolved plots & significant intervals
    6. Band-overlap analysis — raster plot
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"ISC figures will be saved to: {save_dir}")

    print_data_overview(datasets)
    _first_ad = next(iter(datasets.values()))

    # ── Broadband LOO-ISC ─────────────────────────────────────────
    print("\n=== Broadband LOO-ISC ===")
    loo_iscs = {}
    mean_loo_iscs = {}
    for label, ad in datasets.items():
        loo_isc, mean_loo_isc = compute_loo_isc(ad.data)
        loo_iscs[label] = loo_isc
        mean_loo_iscs[label] = mean_loo_isc
        print(
            f"[{label}]  loo_isc: {loo_isc.shape}   "
            f"mean_loo_isc: {mean_loo_isc.shape}"
        )

    plot_loo_isc_distribution(
        mean_loo_iscs,
        ylabel=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=save_dir / "loo_isc_distribution.png",
    )

    # ── Broadband sliding-window ISC ──────────────────────────────
    print("\n=== Broadband Sliding-Window ISC ===")
    sw_results = {}
    for label, ad in datasets.items():
        sw_isc, sw_times = compute_sliding_window_isc(
            ad.data, window_sec=window_sec, step_sec=step_sec, sfreq=ad.sfreq
        )
        sw_results[label] = (sw_isc, sw_times)
        print(f"[{label}]  sw_isc: {sw_isc.shape}   sw_times: {sw_times.shape}")

    plot_sliding_window_isc(
        sw_results,
        isc_threshold=isc_threshold,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=save_dir / "sliding_window_isc.png",
    )
    print_significant_intervals(sw_results, isc_threshold=isc_threshold)

    # ── Per-band LOO-ISC ──────────────────────────────────────────
    print("\n=== Per-Band LOO-ISC ===")
    band_iscs: dict = {}
    for label, ad in datasets.items():
        print(f"Computing band ISC for {label} …")
        band_iscs[label] = {}
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
            filtered = ad.filter_to_band(l_freq, h_freq)
            loo, mean_isc = compute_loo_isc(filtered.data)
            band_iscs[label][band] = (loo, mean_isc)
            print(f"  {band:6s}  loo_isc={loo.shape}  mean={mean_isc.mean():.4f}")

    plot_band_isc_distributions(
        band_iscs,
        bands=FREQUENCY_BANDS,
        feature_axis_label=(
            f"Number of {_first_ad.feature_axis_label.lower()}s"
        ),
        save_path=save_dir / "band_isc_distributions.png",
    )
    plot_band_mean_isc_bar(
        band_iscs,
        bands=FREQUENCY_BANDS,
        save_path=save_dir / "band_isc_mean_bar.png",
    )

    # ── Per-band sliding-window ISC ───────────────────────────────
    print("\n=== Per-Band Sliding-Window ISC ===")
    band_sw: dict = {}
    for label, ad in datasets.items():
        print(f"Computing band sliding-window ISC for {label} …")
        band_sw[label] = {}
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
            filtered = ad.filter_to_band(l_freq, h_freq)
            tc, times = compute_sliding_window_isc(
                filtered.data,
                window_sec=window_sec,
                step_sec=step_sec,
                sfreq=ad.sfreq,
            )
            band_sw[label][band] = (tc, times)
            print(f"  {band:6s}  isc_tc={tc.shape}  times={times.shape}")

    plot_band_sliding_window_isc(
        band_sw,
        bands=FREQUENCY_BANDS,
        isc_threshold=BAND_ISC_THRESHOLDS,
        feature_axis_label=_first_ad.feature_axis_label,
        save_path=save_dir / "band_sliding_window_isc.png",
    )
    print_band_significant_intervals(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        default_threshold=isc_threshold,
    )

    # ── Band overlap ──────────────────────────────────────────────
    print("\n=== Band Overlap ===")
    plot_band_overlap(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        broadband_sw=sw_results,
        broadband_threshold=isc_threshold,
        save_path=save_dir / "band_overlap.png",
    )

    print("\nISC analysis complete.")


# ──────────────────────────────────────────────────────────────────────
# Mean / variance workflow
# ──────────────────────────────────────────────────────────────────────


def run_mean_variance_workflow(
    datasets: dict[str, AnalysisData],
    analyzers: dict[str, EEGSummarizedAnalyzer],
    *,
    save_dir: Path,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
) -> None:
    """Run the full mean-and-variance analysis and save figures.

    Steps:
    1. Data overview
    2. Global mean & variance — distribution plot
    3. Sliding-window mean & variance — time-resolved plot
    4. Per-band global mean & variance — distribution plots
    5. Per-band sliding-window mean & variance — time-resolved plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mean/variance figures will be saved to: {save_dir}")

    print_data_overview(datasets)
    _first_ad = next(iter(datasets.values()))

    # ── Global mean & variance ────────────────────────────────────
    print("\n=== Global Mean & Variance ===")
    mean_var_results = {}
    for label, ad in datasets.items():
        mean_f, var_f = compute_mean_variance(ad.data)
        mean_var_results[label] = (mean_f, var_f)
        print(
            f"[{label}]  mean range: [{mean_f.min():.4f}, {mean_f.max():.4f}]  "
            f"var range: [{var_f.min():.4f}, {var_f.max():.4f}]"
        )

    plot_mean_variance_distribution(
        mean_var_results,
        feature_axis_label=(
            f"Number of {_first_ad.feature_axis_label.lower()}s"
        ),
        save_path=save_dir / "mean_variance_distribution.png",
    )

    # ── Sliding-window mean & variance ────────────────────────────
    print("\n=== Sliding-Window Mean & Variance ===")
    sw_mv_results = {}
    for label, ad in datasets.items():
        mean_tc, var_tc, sw_times = compute_sliding_window_mean_variance(
            ad.data, window_sec=window_sec, step_sec=step_sec, sfreq=ad.sfreq
        )
        sw_mv_results[label] = (mean_tc, var_tc, sw_times)
        print(
            f"[{label}]  mean_tc: {mean_tc.shape}  var_tc: {var_tc.shape}  "
            f"times: {sw_times.shape}"
        )

    plot_sliding_window_mean_variance(
        sw_mv_results,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=save_dir / "sliding_window_mean_variance.png",
    )

    # ── Per-band global mean & variance ───────────────────────────
    print("\n=== Per-Band Mean & Variance ===")
    band_mv_results = {}
    for label, a in analyzers.items():
        band_mv_results[label] = a.compute_band_mean_variance()
        for band, (mf, vf) in band_mv_results[label].items():
            print(
                f"[{label}] {band:6s}  mean range: "
                f"[{mf.min():.4f}, {mf.max():.4f}]  "
                f"var range: [{vf.min():.4f}, {vf.max():.4f}]"
            )

    plot_band_mean_variance_distributions(
        band_mv_results,
        feature_axis_label=(
            f"Number of {_first_ad.feature_axis_label.lower()}s"
        ),
        save_path=save_dir / "band_mean_variance_distribution.png",
    )

    # ── Per-band sliding-window mean & variance ───────────────────
    print("\n=== Per-Band Sliding-Window Mean & Variance ===")
    band_sw_mv_results = {}
    for label, a in analyzers.items():
        band_sw_mv_results[label] = a.compute_band_sliding_window_mean_variance(
            window_sec=window_sec, step_sec=step_sec
        )
        for band, (mtc, vtc, t) in band_sw_mv_results[label].items():
            print(
                f"[{label}] {band:6s}  mean_tc: {mtc.shape}  "
                f"var_tc: {vtc.shape}  times: {t.shape}"
            )

    plot_band_sliding_window_mean_variance(
        band_sw_mv_results,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=save_dir / "band_sliding_window_mean_variance.png",
    )

    print("\nMean & variance analysis complete.")
