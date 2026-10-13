"""
Full-recording ASSR SNR / condition-contrast grid for a stored **CorrCA**
decomposition (stage 06b).

The CorrCA companion of ``run_assr_snr_grid.py`` (stage 06): the same trials, the same
three ASSR references, the same three test families and the same figures, read off a
store written by ``run_corrca_condition_comparison.py`` / ``run_corrca_condition_tracks.py``
with ``--store_components``. What changes is where the spatial operators come from:

* **The filter is the fitted one.** CorrCA stores its shared filters
  (``corrca_filters``), so they are read back rather than recovered by inverting each
  recording's pattern, which for CorrCA would give a *different* operator: the stored
  per-recording pattern is the shared component seen through that recording's
  covariance, and its pseudo-inverse is not the filter the model applied.
* **One filter for every recording**, so there is no per-recording polarity flip: the
  one sign per component was anchored at fit time, from the shared pattern against the
  ASSR electrodes, and the stored sources carry it.
* **``ASSR-mask (PCA)`` reads through the CorrCA subspace.** The IVA grid projects the
  electrode average onto each recording's PCA subspace; here the analogue is the one
  ``--n_pca``-dimensional subspace the filters were sought in (the span of the filters,
  orthogonal projector ``Q Qᵀ``). Same label, same role: the like-for-like reference
  for a component that could only see that subspace.

Signal variants (a subset of stage 06's — the ``mean_*`` rows exist there to swap
per-recording filters for one shared filter, which CorrCA already is):

  - ``zscored`` — the stored z-scored sources; no per-trial baseline
  - ``prestim`` — the raw cache through the shared filter, per-trial pre-stimulus baseline
  - ``stored_prestim`` — the stored sources, given the same per-trial baseline
  - ``masked_prestim`` — the shared filter with every weight outside the ASSR electrodes
    zeroed, on the z-scored cache

Every variant but ``zscored`` carries ``prestim``'s reference columns verbatim, so only
the learned rows move. Tests (participants as the unit, exact Wilcoxon, uncorrected):
``contrast`` (Placebo − Psilocybin per source), ``discrimination`` (does a component
separate the conditions better than a reference) and ``snr`` (is a component's response
weaker than a reference's, per condition).

Outputs::

    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv
    plots/06b-corrca-condition-comparison/<Condition>_<Music>/<spectrum>/
        corrca_snr_grid/pca_<N>/
            masked_filter_topomaps.png       # what the masked_prestim rows read
            <selection>/<window>/
                pvalue_summary__<variant>.png
                snr_by_condition__<variant>.png
                trial_course_by_source__<variant>.png

Examples::

    python scripts/run_corrca_snr_grid.py --experiment assr \\
        --variant corrca_channel_joined_separate --n_pca 10 \\
        --halfwidths 0 5 --stimulus_intervals 0:0.5 0.2:0.5

    python scripts/run_corrca_snr_grid.py --variant corrca_channel_joined_tracks \\
        --n_pca 10 --n_times 3000 --skip_plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

# The cache/onset conventions and the test bookkeeping are stage 06's, so both stages
# read the same trials and write the same CSV schema.
from scripts.run_assr_snr_grid import (  # noqa: E402
    _accumulate_tests,
    _cache_frequency_grid,
    _load_onsets,
    _stimulus_interval,
)
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.analysis.assr_variants import project_condition  # noqa: E402
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.io.iva_store import load_iva_components  # noqa: E402
from src.io.loading import (  # noqa: E402
    assr_electrode_mask,
    concatenated_dir as _concatenated_dir,
)
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.assr_filter_plots import plot_masked_filter_topomaps  # noqa: E402
from src.visualization.assr_snr_plots import (  # noqa: E402
    plot_cell,
    plot_courses,
    plot_snr,
)
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402

_logger = logging.getLogger(__name__)

_STAGE_DIR = "06b-corrca-condition-comparison"
_ANALYSIS_DIR = "corrca_snr_grid"
_MASK_LABELS = at.MASK_LABELS
_FULL_LABEL = at.FULL_MASK_LABEL
_SIGNAL_VARIANTS = ("zscored", "prestim", "stored_prestim", "masked_prestim")
_BASELINE_VARIANTS = tuple(v for v in _SIGNAL_VARIANTS if v != "zscored")
_MASKED_FILTER_FIGURE = "masked_filter_topomaps.png"
_STORE_CONDITIONS = {
    IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE: ConditionVariants.JOINED_SEPARATE,
    IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS: ConditionVariants.JOINED_TRACKS,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-recording ASSR SNR / contrast grid for a stored CorrCA run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stored = parser.add_argument_group("the stored decomposition to read")
    stored.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    stored.add_argument(
        "--variant",
        default=IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE.value,
        choices=[v.value for v in _STORE_CONDITIONS],
        help="Which stage-06b join to read: subject axis (separate) or time axis.",
    )
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument("--band", default=None)
    stored.add_argument("--n_pca", type=int, default=10)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument("--wavelet_data_dir", type=Path, default=None)
    data.add_argument(
        "--concat_dir",
        type=Path,
        default=None,
        help="Processed-data root holding the concatenated arrays' onset sidecars.",
    )
    data.add_argument(
        "--coordinate_system", default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value
    )
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Two conditions, contrast order. Default Placebo Psilocybin.",
    )
    data.add_argument(
        "--mask_sum",
        action="store_true",
        help="Sum the ASSR electrodes instead of averaging them.",
    )
    data.add_argument(
        "--lenient_mask",
        action="store_true",
        help="Accept the electrode intersection when some are missing.",
    )
    data.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Trim the time axis (a quick shape check; not for real runs).",
    )

    analysis = parser.add_argument_group("the extraction, the grid and the tests")
    analysis.add_argument("--center_freq", type=float, default=iva_quality.ASSR_FREQ)
    analysis.add_argument(
        "--halfwidths",
        type=float,
        nargs="+",
        default=[0.0, iva_quality.TF_ANCHOR_HALFWIDTH_HZ],
        help="One frequency selection per half-width in Hz (0 = single 40 Hz bin).",
    )
    analysis.add_argument(
        "--stimulus_intervals",
        type=_stimulus_interval,
        nargs="+",
        default=[
            (0.0, AssrEpoch.STIMULUS_DURATION_S),
            (0.2, AssrEpoch.STIMULUS_DURATION_S),
        ],
        help="One test window per 'lo:hi' seconds, e.g. 0:0.5 0.2:0.5.",
    )
    analysis.add_argument(
        "--signal_variants",
        nargs="+",
        default=list(_SIGNAL_VARIANTS),
        choices=list(_SIGNAL_VARIANTS),
        help="Which readings of the learned rows to test side by side.",
    )
    analysis.add_argument(
        "--contrast_alternative",
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="Direction of Placebo - Psilocybin. 'greater' = psilocybin lowers it.",
    )
    analysis.add_argument(
        "--snr_alternative",
        default="less",
        choices=["less", "greater", "two-sided"],
        help="Direction of (component SNR - reference SNR) in the SNR family.",
    )
    analysis.add_argument("--alpha", type=float, default=0.05)

    output = parser.add_argument_group("output")
    output.add_argument("--save_dir", type=Path, default=None)
    output.add_argument("--results_dir", type=Path, default=None)
    output.add_argument("--skip_plots", action="store_true")
    output.add_argument("--log_level", default="INFO")
    return parser


def corrca_operators(
    shared_filters: np.ndarray, binary: np.ndarray, electrode_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """The spatial operators of a CorrCA store, in the shapes the stage-06 grid uses.

    :param shared_filters: ``(channels, components)`` stored ``corrca_filters``.
    :param binary: ``(channels,)`` electrode-average weights.
    :param electrode_mask: Boolean ``(channels,)`` anchor mask.
    :return: ``filters`` ``(K, C)``, ``projector`` ``(C, C)`` — the orthogonal projector
        onto the filters' span — ``pca_row`` ``(C,)`` (the mask through that subspace),
        ``masked`` ``(K, C)`` and ``masked_share`` ``(K,)``.
    """
    basis, _ = np.linalg.qr(np.asarray(shared_filters, dtype=float))
    projector = basis @ basis.T
    filters = np.asarray(shared_filters, dtype=float).T
    return {
        "filters": filters,
        "projector": projector,
        "pca_row": projector @ binary,
        "masked": at.restrict_filters_to_mask(filters, electrode_mask),
        "masked_share": at.mask_weight_share(filters, electrode_mask),
    }


def run(args: argparse.Namespace) -> None:
    """Run the full grid for one stored CorrCA decomposition."""
    experiment = ExperimentNames(args.experiment)
    variant = IvaVariants(args.variant)
    store_condition = _STORE_CONDITIONS[variant]
    per_recording = variant == IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE
    music_type = MusicTypeVariants(args.music_type)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    conditions = at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))

    # ---- 1. store and operators -------------------------------------------
    results = load_iva_components(
        experiment=experiment,
        condition=store_condition,
        variant=variant,
        music_type=music_type,
        band=args.band,
        n_pca=args.n_pca,
        processed_data_dir=args.store_dir,
    )
    _logger.info(f"loaded {results.path}")
    if "corrca_filters" not in results.extras:
        raise ValueError(
            f"{results.path} holds no 'corrca_filters'; it was not written by a "
            "stage-06b CorrCA run with --store_components."
        )
    store_channels = list(results.channel_names)
    electrode_mask = assr_electrode_mask(
        store_channels, coordinate_system, strict=not args.lenient_mask
    )
    binary = at.binary_filter_weights(electrode_mask, normalize=not args.mask_sum)
    ops = corrca_operators(results.extras["corrca_filters"], binary, electrode_mask)
    n_components = ops["filters"].shape[0]
    n_rows = results.n_subjects
    n_channels = len(store_channels)
    comp_labels = [f"{COMPONENT_PREFIX} {k + 1}" for k in range(n_components)]
    labels = comp_labels + list(_MASK_LABELS)
    full_mask_column = labels.index(_FULL_LABEL)
    anchor = str(results.extras.get("polarity_anchor", np.asarray("none")))
    _logger.info(
        f"variant {variant.value}: one shared filter, {n_components} component(s), "
        f"{int(electrode_mask.sum())} ASSR electrode(s); sign anchor '{anchor}'"
    )
    for k in range(n_components):
        _logger.info(
            f"  {comp_labels[k]}: |w| kept by the ROI restriction "
            f"{ops['masked_share'][k]:.1%}"
        )

    def resolve(participant: str, condition: str) -> int:
        if per_recording:
            return results.row(participant, condition)
        return results.row(participant)

    # ---- 2. onsets, caches, paired cohort ---------------------------------
    concat_dir = _concatenated_dir(experiment, args.concat_dir)
    onsets, wavelet_sets, cached = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        wavelet_sets[condition] = find_wavelet_set(
            wavelet_label_dir(args.wavelet_data_dir, experiment, label)
        )
        cached[condition] = set(wavelet_sets[condition].participants)

    # ---- 3. frequency selections, resolved on the store's AND the cache's grid
    cache_freqs = _cache_frequency_grid(wavelet_sets, conditions)
    store_selections, cache_selections = {}, {}
    for halfwidth in args.halfwidths:
        name = at.selection_label(args.center_freq, halfwidth)
        store_selections[name] = at.frequency_selection(
            results.freqs, args.center_freq, halfwidth
        )
        cache_selections[name] = at.frequency_selection(
            cache_freqs, args.center_freq, halfwidth
        )
    union = np.unique(np.concatenate(list(cache_selections.values())))
    union_position = {int(b): i for i, b in enumerate(union)}

    def _has_row(participant: str, condition: str) -> bool:
        if per_recording:
            return bool(results.rows(participant, condition))
        return bool(results.rows(participant))

    store_participants = list(dict.fromkeys(results.participants))
    participants = [
        p
        for p in store_participants
        if all(_has_row(p, c) and p in cached[c] for c in conditions)
    ]
    if not participants:
        raise ValueError(
            "No participant is present in the store under every condition and every cache."
        )
    dropped = [p for p in store_participants if p not in participants]
    if dropped:
        _logger.warning(f"dropped (not paired / not cached): {dropped}")
    n_participants = len(participants)

    # ---- 4. project both conditions (one stream each) ---------------------
    # project_condition takes per-row operators; CorrCA has one, so every row is a view
    # of it. Its cohort-mean slots receive the shared filter too; their outputs are not
    # read here (the mean_* variants are stage 06's, and CorrCA is shared already).
    filters_rows = np.broadcast_to(ops["filters"], (n_rows, n_components, n_channels))
    projected = {
        condition: project_condition(
            wavelet_sets[condition],
            condition,
            participants,
            resolve,
            filters_rows,
            binary,
            np.broadcast_to(ops["pca_row"], (n_rows, n_channels)),
            union,
            store_channels,
            args.n_times,
            np.flatnonzero(electrode_mask),
            ops["filters"],
            np.broadcast_to(ops["masked"], (n_rows, n_components, n_channels)),
            ops["masked"],
            np.broadcast_to(ops["projector"], (n_rows, n_channels, n_channels)),
        )
        for condition in conditions
    }

    # ---- 5. epoch geometry (shared window) --------------------------------
    sfreq = results.sfreq
    geometry = {
        c: at.epoch_geometry(onsets[c], projected[c].prestim.shape[-1], sfreq)
        for c in conditions
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)
    baseline_mask = times < 0.0

    # The stored sources, per condition, on that condition's own time axis.
    if per_recording:
        stored = {
            c: [results.tf_maps[resolve(p, c)] for p in participants]
            for c in conditions
        }
    else:
        stored = {}
        for c in conditions:
            segment = results.condition_track(results.tf_maps, c)
            stored[c] = [segment[resolve(p, c)] for p in participants]
    for c in conditions:
        keep = projected[c].prestim.shape[-1]
        if any(src.shape[-1] < keep for src in stored[c]):
            raise ValueError(
                f"[{c}] a stored source is shorter than the projected cache; the store "
                "and the cache disagree on the time axis."
            )
        stored[c] = [src[..., :keep] for src in stored[c]]

    polarity_note = (
        f"shared CorrCA filter: one sign per component, anchored at fit time "
        f"('{anchor}'); no per-recording flip"
    )
    signs = np.ones((n_participants, len(labels)))

    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if args.band is None
        else SpectrumTypeVariants.BANDS.value
    )
    plots_dir = ProjectPaths.analysis_plots_dir(
        _STAGE_DIR,
        f"{store_condition.value}_{music_type.value}",
        spectrum=spectrum,
        analysis_type=_ANALYSIS_DIR,
        n_pca=args.n_pca,
        root=args.save_dir,
    )
    plot_prefix = "" if args.band is None else f"{args.band}_"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _write_filter_plot(
            plots_dir, plot_prefix, results, ops, electrode_mask, coordinate_system
        )

    # ---- 6. the grid: selection x window ----------------------------------
    rows_out: list[dict] = []
    for sel_name, bins in store_selections.items():
        take = [union_position[int(b)] for b in cache_selections[sel_name]]
        cut: dict[str, dict[str, np.ndarray]] = {name: {} for name in _SIGNAL_VARIANTS}
        roi_snr: dict[str, np.ndarray] = {}
        roi_scale: dict[str, dict[str, float]] = {}
        for condition in conditions:
            onsets_inside = geometry[condition][0]
            proj = projected[condition]
            prestim_band = proj.prestim[:, :, take, :].mean(axis=2)  # (P, K+2, T)
            zmask_band = proj.zmask[:, :, take, :].mean(axis=2)  # (P, 2, T)
            pca_z_band = zmask_band[:, 1:2]
            prestim_band = np.concatenate([prestim_band, pca_z_band], axis=1)
            reference_band = prestim_band[:, n_components:]  # (P, 3, T)
            stored_band = np.stack(
                [src[:, bins, :].mean(axis=1) for src in stored[condition]]
            )  # (P, K, T)
            masked_band = proj.masked_z[:, :, take, :].mean(axis=2)
            bands = {
                "prestim": prestim_band,
                "zscored": np.concatenate(
                    [stored_band, zmask_band, pca_z_band], axis=1
                ),
                "stored_prestim": np.concatenate([stored_band, reference_band], axis=1),
                "masked_prestim": np.concatenate([masked_band, reference_band], axis=1),
            }
            for name, band in bands.items():
                cut[name][condition], _ = at.cut_trials(band, onsets_inside, pre, post)
            roi_band = proj.roi_raw[:, :, take, :].mean(axis=2)
            roi_cut, _ = at.cut_trials(roi_band, onsets_inside, pre, post)
            roi_snr[condition], roi_scale[condition] = at.roi_channelwise_snr(
                roi_cut, baseline_mask
            )

        # One ROI scale for both conditions, so a participant's two conditions are not
        # rescaled differently (the stage-06 rule; exact, not a re-fit).
        shared_roi_scale = float(
            np.median([d["roi_baseline_sd"] for d in roi_scale.values()])
        )
        for condition in conditions:
            roi_snr[condition] = roi_snr[condition] * (
                roi_scale[condition]["roi_baseline_sd"] / shared_roi_scale
            )

        for lo, hi in args.stimulus_intervals:
            if (lo, hi) == (0.0, AssrEpoch.STIMULUS_DURATION_S):
                window = AssrEpoch.stimulus_mask(times)
            else:
                window = (times >= lo) & (times <= hi)
            if not window.any():
                raise ValueError(f"window {lo}-{hi}s selects no epoch sample.")
            window_label = f"{lo:g}-{hi:g}s"

            for signal in args.signal_variants:
                value, course = {}, {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+3, N, W)
                    if signal in _BASELINE_VARIANTS:
                        normed, _rel, _pos = at.baseline_normalise(
                            trials, baseline_mask
                        )
                        normed = normed.copy()
                        normed[:, full_mask_column] = roi_snr[condition]
                    else:
                        normed = trials
                    per_trial = normed[..., window].mean(axis=-1)
                    value[condition] = np.median(per_trial, axis=2) * signs
                    course[condition] = np.median(normed, axis=2) * signs[:, :, None]

                cell = {
                    "variant": signal,
                    "selection": sel_name,
                    "window": window_label,
                }
                _accumulate_tests(
                    rows_out,
                    cell,
                    value,
                    conditions,
                    labels,
                    comp_labels,
                    args.contrast_alternative,
                    args.snr_alternative,
                )
                if not args.skip_plots:
                    _write_cell_plots(
                        plots_dir,
                        plot_prefix,
                        cell,
                        value,
                        course,
                        times,
                        (lo, hi),
                        conditions,
                        labels,
                        comp_labels,
                        args.alpha,
                        rows_out,
                        n_participants,
                        polarity_note,
                    )

    # ---- 7. one CSV for the whole grid ------------------------------------
    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / (
        f"assr_snr_grid__{variant.value}__{spectrum}__pca{args.n_pca}.csv"
    )
    frame = pd.DataFrame(rows_out)
    frame.to_csv(csv_path, index=False)
    _logger.info(
        f"saved {csv_path} ({len(frame)} test rows, n = {n_participants}, floor p = "
        f"{at.p_floor(n_participants):.5f} two-sided)"
    )


def _write_filter_plot(
    plots_dir, plot_prefix, results, ops, electrode_mask, coordinate_system
) -> None:
    """The restricted shared filter the ``masked_prestim`` rows read, once per run."""
    try:
        info = results.topo_info(coordinate_system)
    except (ValueError, FileNotFoundError) as exc:
        _logger.warning(f"no topography layout for this store ({exc}); skipping.")
        return
    plot_masked_filter_topomaps(
        ops["masked"],
        info,
        weight_share=ops["masked_share"],
        n_mask_channels=int(np.asarray(electrode_mask).sum()),
        variant="masked_prestim",
        save_path=plots_dir / f"{plot_prefix}{_MASKED_FILTER_FIGURE}",
    )
    plt.close("all")


def _write_cell_plots(
    plots_dir,
    plot_prefix,
    cell,
    value,
    course,
    times,
    window_interval,
    conditions,
    labels,
    comp_labels,
    alpha,
    rows_out,
    n_participants,
    polarity_note,
) -> None:
    """The stage-06 three figures for one grid cell, with components named ``CC``."""
    tag = f"{cell['variant']}__{cell['selection']}__{cell['window']}"
    variant = cell["variant"]
    cell_dir = plots_dir / cell["selection"] / cell["window"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in rows_out if all(r[k] == cell[k] for k in cell)])
    cell_tables = {
        "contrast": frame[frame["family"] == "contrast"],
        **{
            f"discrimination:{m}": frame[
                (frame["family"] == "discrimination") & (frame["reference"] == m)
            ]
            for m in _MASK_LABELS
        },
        **{
            f"snr:{m}:{c}": frame[
                (frame["family"] == "snr")
                & (frame["reference"] == m)
                & (frame["condition"] == c)
            ]
            for m in _MASK_LABELS
            for c in conditions
        },
    }
    plot_cell(
        cell_tables,
        value,
        conditions,
        labels,
        comp_labels,
        alpha,
        f"Condition contrast & discrimination — {tag}, n={n_participants}",
        cell_dir / f"{plot_prefix}pvalue_summary__{variant}.png",
        component_prefix=COMPONENT_PREFIX,
    )
    plot_snr(
        cell_tables,
        value,
        conditions,
        comp_labels,
        alpha,
        f"{COMPONENT_PREFIX} SNR vs reference, per condition — {tag}, "
        f"n={n_participants}",
        cell_dir / f"{plot_prefix}snr_by_condition__{variant}.png",
        component_prefix=COMPONENT_PREFIX,
    )
    plot_courses(
        course,
        times,
        window_interval,
        cell_tables["contrast"],
        labels,
        conditions,
        alpha,
        f"Trial course per spatial filter — {tag}, n={n_participants}\n{polarity_note}",
        cell_dir / f"{plot_prefix}trial_course_by_source__{variant}.png",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
