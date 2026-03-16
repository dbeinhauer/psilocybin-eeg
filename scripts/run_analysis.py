"""
Script to run ISC and group-level analysis on preprocessed EEG data.

Replicates the analysis workflow from ``notebooks/data_analysis.ipynb``
so that it can be executed as a stand-alone command (e.g. inside a
Metacentrum PBS job).

Workflow
--------
1. Load pre-saved concatenated data for each music type, normalise.
2. Broadband LOO-ISC – distribution plot.
3. Broadband sliding-window ISC – time-resolved plot & significant intervals.
4. Per-band LOO-ISC – distribution & bar-chart plots.
5. Per-band sliding-window ISC – time-resolved plots & significant intervals.
6. Band-overlap analysis – raster plot of consensus synchrony windows.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (
    add_common_arguments,
    load_analyzers,
    analyzers_to_datasets,
    BAND_ISC_THRESHOLDS,
)
from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
)
from src.definitions.constants import ProjectPaths
from src.analysis.isc import (
    compute_loo_isc,
    compute_sliding_window_isc,
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
)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ISC and group-level analysis on preprocessed EEG data."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--isc_threshold",
        type=float,
        default=0.035,
        help="Broadband ISC significance threshold (default: 0.035).",
    )

    args = parser.parse_args()

    # ── Configuration ─────────────────────────────────────────────────
    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [ExclusionCategories.BAD_MUSIC]

    SAVE_DIR = ProjectPaths.PLOTS_PATH / "OverallAnalysis"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {SAVE_DIR}")

    ISC_THRESHOLD = args.isc_threshold
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    # ── Data loading ──────────────────────────────────────────────────
    analyzers = load_analyzers(
        music_types, condition, exclusion_categories, args.process_and_save
    )
    datasets = analyzers_to_datasets(analyzers)

    # ── Data overview ─────────────────────────────────────────────────
    print_data_overview(datasets)

    # ── Broadband LOO-ISC ─────────────────────────────────────────────
    print("\n=== Broadband LOO-ISC ===")
    loo_iscs = {}
    mean_loo_iscs = {}
    for label, ad in datasets.items():
        loo_isc, mean_loo_isc = compute_loo_isc(ad.data)
        loo_iscs[label] = loo_isc
        mean_loo_iscs[label] = mean_loo_isc
        print(
            f"[{label}]  loo_isc: {loo_isc.shape}   mean_loo_isc: {mean_loo_isc.shape}"
        )

    _first_ad = next(iter(datasets.values()))
    plot_loo_isc_distribution(
        mean_loo_iscs,
        ylabel=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=SAVE_DIR / "loo_isc_distribution.png",
    )

    # ── Broadband sliding-window ISC ──────────────────────────────────
    print("\n=== Broadband Sliding-Window ISC ===")
    sw_results = {}
    for label, ad in datasets.items():
        sw_isc, sw_times = compute_sliding_window_isc(
            ad.data, window_sec=WINDOW_SEC, step_sec=STEP_SEC, sfreq=ad.sfreq
        )
        sw_results[label] = (sw_isc, sw_times)
        print(f"[{label}]  sw_isc: {sw_isc.shape}   sw_times: {sw_times.shape}")

    plot_sliding_window_isc(
        sw_results,
        isc_threshold=ISC_THRESHOLD,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=SAVE_DIR / "sliding_window_isc.png",
    )
    print_significant_intervals(sw_results, isc_threshold=ISC_THRESHOLD)

    # ── Per-band LOO-ISC ──────────────────────────────────────────────
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
        feature_axis_label=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=SAVE_DIR / "band_isc_distributions.png",
    )
    plot_band_mean_isc_bar(
        band_iscs,
        bands=FREQUENCY_BANDS,
        save_path=SAVE_DIR / "band_isc_mean_bar.png",
    )

    # ── Per-band sliding-window ISC ───────────────────────────────────
    print("\n=== Per-Band Sliding-Window ISC ===")
    band_sw: dict = {}
    for label, ad in datasets.items():
        print(f"Computing band sliding-window ISC for {label} …")
        band_sw[label] = {}
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
            filtered = ad.filter_to_band(l_freq, h_freq)
            tc, times = compute_sliding_window_isc(
                filtered.data,
                window_sec=WINDOW_SEC,
                step_sec=STEP_SEC,
                sfreq=ad.sfreq,
            )
            band_sw[label][band] = (tc, times)
            print(f"  {band:6s}  isc_tc={tc.shape}  times={times.shape}")

    plot_band_sliding_window_isc(
        band_sw,
        bands=FREQUENCY_BANDS,
        isc_threshold=BAND_ISC_THRESHOLDS,
        feature_axis_label=_first_ad.feature_axis_label,
        save_path=SAVE_DIR / "band_sliding_window_isc.png",
    )
    print_band_significant_intervals(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        default_threshold=ISC_THRESHOLD,
    )

    # ── Band overlap ──────────────────────────────────────────────────
    print("\n=== Band Overlap ===")
    plot_band_overlap(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        broadband_sw=sw_results,
        broadband_threshold=ISC_THRESHOLD,
        save_path=SAVE_DIR / "band_overlap.png",
    )

    print("\nISC analysis complete.")
