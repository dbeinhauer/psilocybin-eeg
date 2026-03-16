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

from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
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
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_analyzer(music_type, condition, exclusion_categories, process_and_save):
    """Create, load (or process & save) and normalise an analyser for one music type."""
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzer = EEGSummarizedAnalyzer(
        experiment_name=ExperimentNames.PSILO_MUSIC,
        coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
        music_types=[music_type],
        conditions=[condition],
        exclusion_categories=exclusion_categories,
    )

    if process_and_save:
        analyzer.load_and_prepare_data(resample_freq=250.0, n_jobs=-1)
        print(f"[{music_type.value}] data shape: {analyzer.data.shape}")
        analyzer.save_data()
    else:
        analyzer.load_data(
            info_filename=analyzer.filtered_df[SingleDataMetadata.FILENAME].iloc[0],
        )
        print(f"[{music_type.value}] Loaded data shape: {analyzer.data.shape}")

    analyzer.normalize()
    return analyzer


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ISC and group-level analysis on preprocessed EEG data."
    )
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
        help=("One or more music types to analyse. Defaults to all available types."),
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

    args = parser.parse_args()

    # ── Virtual display for headless environments ─────────────────────
    try:
        from xvfbwrapper import Xvfb

        vdisplay = Xvfb()
        vdisplay.start()
    except ImportError:
        pass

    import matplotlib

    matplotlib.use("Agg")

    # ── Configuration ─────────────────────────────────────────────────
    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [ExclusionCategories.BAD_MUSIC]

    SAVE_DIR = ProjectPaths.PLOTS_PATH / "OverallAnalysis"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {SAVE_DIR}")

    ISC_THRESHOLD = args.isc_threshold
    BAND_ISC_THRESHOLDS = {
        "delta": 0.1,
        "theta": 0.07,
        "alpha": 0.035,
        "beta": 0.02,
        "gamma": 0.01,
    }
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    # ── Data loading ──────────────────────────────────────────────────
    analyzers = {}
    for mt in music_types:
        label = mt.value
        analyzers[label] = _make_analyzer(
            mt, condition, exclusion_categories, args.process_and_save
        )

    # Convert to AnalysisData (raw time-domain representation)
    datasets = {
        label: a.to_analysis_data(label=label) for label, a in analyzers.items()
    }

    for label, ad in datasets.items():
        print(f"[{label}] {ad}")

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
