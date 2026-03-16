"""
Script to run mean-and-variance analysis on preprocessed EEG data.

Replicates the analysis workflow from
``notebooks/mean_variance_analysis.ipynb`` so that it can be executed as a
stand-alone command (e.g. inside a Metacentrum PBS job).

Workflow
--------
1. Load pre-saved concatenated data for each music type, normalise.
2. Global mean & variance per feature – distribution plot.
3. Sliding-window mean & variance – time-resolved plot.
4. Per-band global mean & variance – distribution plots.
5. Per-band sliding-window mean & variance – time-resolved plots.
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
    compute_mean_variance,
    compute_sliding_window_mean_variance,
)
from src.visualization.isc_plots import (
    plot_mean_variance_distribution,
    plot_sliding_window_mean_variance,
    plot_band_mean_variance_distributions,
    plot_band_sliding_window_mean_variance,
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
        description="Run mean-and-variance analysis on preprocessed EEG data."
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
        help=(
            "One or more music types to analyse. "
            "Defaults to all available types."
        ),
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        default=False,
        help="When set, load raw files, resample, stack and save before analysis.",
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

    SAVE_DIR = ProjectPaths.PLOTS_PATH / "MeanVarianceAnalysis"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {SAVE_DIR}")

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

    # ── Global mean & variance ────────────────────────────────────────
    print("\n=== Global Mean & Variance ===")
    mean_var_results = {}
    for label, ad in datasets.items():
        mean_f, var_f = compute_mean_variance(ad.data)
        mean_var_results[label] = (mean_f, var_f)
        print(
            f"[{label}]  mean range: [{mean_f.min():.4f}, {mean_f.max():.4f}]  "
            f"var range: [{var_f.min():.4f}, {var_f.max():.4f}]"
        )

    _first_ad = next(iter(datasets.values()))
    plot_mean_variance_distribution(
        mean_var_results,
        feature_axis_label=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=SAVE_DIR / "mean_variance_distribution.png",
    )

    # ── Sliding-window mean & variance ────────────────────────────────
    print("\n=== Sliding-Window Mean & Variance ===")
    sw_mv_results = {}
    for label, ad in datasets.items():
        mean_tc, var_tc, sw_times = compute_sliding_window_mean_variance(
            ad.data, window_sec=WINDOW_SEC, step_sec=STEP_SEC, sfreq=ad.sfreq
        )
        sw_mv_results[label] = (mean_tc, var_tc, sw_times)
        print(
            f"[{label}]  mean_tc: {mean_tc.shape}  var_tc: {var_tc.shape}  "
            f"times: {sw_times.shape}"
        )

    plot_sliding_window_mean_variance(
        sw_mv_results,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=SAVE_DIR / "sliding_window_mean_variance.png",
    )

    # ── Per-band global mean & variance ───────────────────────────────
    print("\n=== Per-Band Mean & Variance ===")
    band_mv_results = {}
    for label, a in analyzers.items():
        band_mv_results[label] = a.compute_band_mean_variance()
        for band, (mf, vf) in band_mv_results[label].items():
            print(
                f"[{label}] {band:6s}  mean range: [{mf.min():.4f}, {mf.max():.4f}]  "
                f"var range: [{vf.min():.4f}, {vf.max():.4f}]"
            )

    plot_band_mean_variance_distributions(
        band_mv_results,
        feature_axis_label=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=SAVE_DIR / "band_mean_variance_distribution.png",
    )

    # ── Per-band sliding-window mean & variance ───────────────────────
    print("\n=== Per-Band Sliding-Window Mean & Variance ===")
    band_sw_mv_results = {}
    for label, a in analyzers.items():
        band_sw_mv_results[label] = a.compute_band_sliding_window_mean_variance(
            window_sec=WINDOW_SEC, step_sec=STEP_SEC
        )
        for band, (mtc, vtc, t) in band_sw_mv_results[label].items():
            print(
                f"[{label}] {band:6s}  mean_tc: {mtc.shape}  "
                f"var_tc: {vtc.shape}  times: {t.shape}"
            )

    plot_band_sliding_window_mean_variance(
        band_sw_mv_results,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=SAVE_DIR / "band_sliding_window_mean_variance.png",
    )

    print("\nMean & variance analysis complete.")
