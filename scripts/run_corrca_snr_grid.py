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

Signal variants, stage 06's two and their complement:

  - ``full`` — the stored sources (the fit's own output on z-scored data), given a
    per-trial pre-stimulus baseline
  - ``roi_masked`` — the shared filter with every weight outside the ASSR electrodes
    zeroed, on the z-scored cache, given the same baseline
  - ``non_roi_masked`` — the same filter with the ASSR electrodes zeroed instead. The
    two masked filters add up to the full one, so together they split a component's
    response into what the ROI carries and what the rest of the head carries. An ROI
    store has no electrode outside the ROI, so it skips this variant.

All carry the same reference rows (the masks on the raw cache, the full mask swapped
for the equal-weight ROI row), so only the learned rows move. Tests (participants as
the unit, exact Wilcoxon, uncorrected): ``contrast`` (Placebo − Psilocybin per source),
``discrimination`` (does a component separate the conditions better than a reference)
and ``snr`` (is a component's response weaker than a reference's, per condition).

**ROI stores.** Every run reads the join's ``_roi`` store too, after the full one: the
second fit the two stage-06b fit scripts make on the ASSR electrodes alone. The cache is
read on those electrodes only, and ``roi_masked`` coincides with ``full`` by
construction — every weight is already inside the ROI — which is a free check that the
store and the cache agree. A missing ``_roi`` store is skipped with a warning;
``--skip_roi`` skips it outright.

**Masked TF maps.** With ``--masked_tf_maps`` the masked variants also get full
trial-normalised TF maps, like the stored sources' ``tf_baseline``: the cache is then
read at every frequency of the store, not only at the 40 Hz selections, which costs
several times the read time.

Outputs::

    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__values.npz
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__courses.npz
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__tf_baseline.npz
        # tf_baseline__<Condition>; with --masked_tf_maps also
        # tf_baseline__<signal variant>__<Condition> for the masked variants
    plots/06b-corrca-condition-comparison/<Condition>_<Music>/<spectrum>/
        corrca_snr_grid/pca_<N>/                # corrca_snr_grid_roi/ for a '_roi' store
            masked_filter_topomaps.png       # what the roi_masked rows read
            non_roi_masked_filter_topomaps.png  # what the non_roi_masked rows read
            tf_baseline_maps.png             # every component, trial-normalised
            tf_baseline_maps__<masked variant>.png  # with --masked_tf_maps
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
from src.io.saving import (  # noqa: E402
    save_npz_atomic,
    save_snr_courses,
    save_snr_values,
)
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.assr_filter_plots import plot_masked_filter_topomaps  # noqa: E402
from src.visualization.assr_snr_plots import (  # noqa: E402
    plot_cell,
    plot_condition_tf_grid,
    plot_courses,
    plot_snr,
)
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402

_logger = logging.getLogger(__name__)

_STAGE_DIR = "06b-corrca-condition-comparison"
_ANALYSIS_DIR = "corrca_snr_grid"
#: The fits on the ASSR electrodes alone, whose figures go to their own directory.
_ROI_OF = {
    IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE: (
        IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE_ROI
    ),
    IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS: IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS_ROI,
}
_ROI_VARIANTS = tuple(_ROI_OF.values())
_MASK_LABELS = at.MASK_LABELS
_FULL_LABEL = at.FULL_MASK_LABEL
_SIGNAL_VARIANTS = ("full", "roi_masked", "non_roi_masked")
#: The variants read through a masked filter: signal variant -> the operator in
#: corrca_operators and the projection in ProjectedVariants at the 40 Hz selections
#: and at every store frequency.
_MASKED = {
    "roi_masked": ("masked", "masked_z", "masked_tf_z"),
    "non_roi_masked": ("outside", "outside_z", "outside_tf_z"),
}
_MASKED_FILTER_FIGURE = "masked_filter_topomaps.png"
_OUTSIDE_FILTER_FIGURE = "non_roi_masked_filter_topomaps.png"
_TF_BASELINE_FIGURE = "tf_baseline_maps.png"
_STORE_CONDITIONS = {
    IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE: ConditionVariants.JOINED_SEPARATE,
    IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS: ConditionVariants.JOINED_TRACKS,
    IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE_ROI: ConditionVariants.JOINED_SEPARATE,
    IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS_ROI: ConditionVariants.JOINED_TRACKS,
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
        choices=[v.value for v in _ROI_OF],
        help="Which stage-06b join to read: subject axis (separate) or time axis. "
        "Its '_roi' store (the fit on the ASSR electrodes alone) is read after it.",
    )
    stored.add_argument(
        "--skip_roi",
        action="store_true",
        help="Read only the full-montage store, not the join's '_roi' one.",
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
        "--masked_tf_maps",
        action="store_true",
        help="Also store and draw full trial-normalised TF maps of the masked "
        "variants. Reads the cache at every store frequency.",
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
        ``masked`` ``(K, C)`` and ``masked_share`` ``(K,)``, and their complement
        ``outside`` / ``outside_share`` (``masked + outside == filters``; all zero when
        the mask covers every channel).
    """
    basis, _ = np.linalg.qr(np.asarray(shared_filters, dtype=float))
    projector = basis @ basis.T
    filters = np.asarray(shared_filters, dtype=float).T
    outside = filters * ~np.asarray(electrode_mask, dtype=bool)
    total = np.linalg.norm(filters, axis=-1)
    return {
        "filters": filters,
        "projector": projector,
        "pca_row": projector @ binary,
        "masked": at.restrict_filters_to_mask(filters, electrode_mask),
        "masked_share": at.mask_weight_share(filters, electrode_mask),
        "outside": outside,
        "outside_share": np.divide(
            np.linalg.norm(outside, axis=-1),
            total,
            out=np.zeros_like(total),
            where=total > 0,
        ),
    }


def run(args: argparse.Namespace) -> None:
    """Run the grid on the join's full-montage store, then on its ``_roi`` store."""
    variant = IvaVariants(args.variant)
    run_store(args, variant)
    if args.skip_roi:
        return
    run_store(args, _ROI_OF[variant], optional=True)


def run_store(
    args: argparse.Namespace, variant: IvaVariants, *, optional: bool = False
) -> None:
    """Run the full grid for one stored CorrCA decomposition.

    :param optional: Skip, with a warning, when the store does not exist.
    """
    experiment = ExperimentNames(args.experiment)
    store_condition = _STORE_CONDITIONS[variant]
    per_recording = store_condition == ConditionVariants.JOINED_SEPARATE
    music_type = MusicTypeVariants(args.music_type)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    conditions = at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))

    # ---- 1. store and operators -------------------------------------------
    try:
        results = load_iva_components(
            experiment=experiment,
            condition=store_condition,
            variant=variant,
            music_type=music_type,
            band=args.band,
            n_pca=args.n_pca,
            processed_data_dir=args.store_dir,
        )
    except FileNotFoundError as exc:
        if not optional:
            raise
        _logger.warning(f"no {variant.value} store to read ({exc}); skipped.")
        return
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
    signals = list(args.signal_variants)
    if "non_roi_masked" in signals and electrode_mask.all():
        _logger.info(
            "every store channel is an ASSR electrode: no non_roi_masked rows."
        )
        signals.remove("non_roi_masked")
    masked_signals = [s for s in signals if s in _MASKED]

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
    # The cache bin of every store frequency, for the masked variants' full TF maps.
    tf_bins = None
    if args.masked_tf_maps and masked_signals:
        grid = np.asarray(cache_freqs, dtype=float)
        tf_bins = np.array([int(np.argmin(np.abs(grid - f))) for f in results.freqs])
        if not np.allclose(grid[tf_bins], results.freqs, rtol=1e-6):
            raise ValueError("The store's frequencies are not on the cache's grid.")

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
    # of it.
    projected = {
        condition: project_condition(
            wavelet_sets[condition],
            condition,
            participants,
            resolve,
            binary,
            np.broadcast_to(ops["pca_row"], (n_rows, n_channels)),
            union,
            store_channels,
            args.n_times,
            np.flatnonzero(electrode_mask),
            np.broadcast_to(ops["masked"], (n_rows, n_components, n_channels)),
            outside_filters=(
                np.broadcast_to(ops["outside"], (n_rows, n_components, n_channels))
                if "non_roi_masked" in signals
                else None
            ),
            tf_freq_indices=tf_bins,
        )
        for condition in conditions
    }

    # ---- 5. epoch geometry (shared window) --------------------------------
    sfreq = results.sfreq
    geometry = {
        c: at.epoch_geometry(onsets[c], projected[c].reference_raw.shape[-1], sfreq)
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
        keep = projected[c].reference_raw.shape[-1]
        if any(src.shape[-1] < keep for src in stored[c]):
            raise ValueError(
                f"[{c}] a stored source is shorter than the projected cache; the store "
                "and the cache disagree on the time axis."
            )
        stored[c] = [src[..., :keep] for src in stored[c]]

    # Each stored source's TF map in units of its own pre-stimulus SD, per recording:
    # the courses' reading at every frequency. condition -> (K, P, F, W).
    tf_baseline = {
        c: np.stack(
            [
                at.baseline_tf_map(src, geometry[c][0], pre, post, baseline_mask)
                for src in stored[c]
            ],
            axis=1,
        ).astype(np.float32)
        for c in conditions
    }
    # The masked variants' maps, read the same way: signal -> condition -> (K, P, F, W).
    masked_tf = {}
    if tf_bins is not None:
        for signal in masked_signals:
            projection = _MASKED[signal][2]
            masked_tf[signal] = {
                c: np.stack(
                    [
                        at.baseline_tf_map(
                            src, geometry[c][0], pre, post, baseline_mask
                        )
                        for src in getattr(projected[c], projection)
                    ],
                    axis=1,
                ).astype(np.float32)
                for c in conditions
            }

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
        analysis_type=(
            f"{_ANALYSIS_DIR}_roi" if variant in _ROI_VARIANTS else _ANALYSIS_DIR
        ),
        n_pca=args.n_pca,
        root=args.save_dir,
    )
    plot_prefix = "" if args.band is None else f"{args.band}_"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _write_filter_plot(
            plots_dir,
            plot_prefix,
            results,
            ops,
            electrode_mask,
            coordinate_system,
            masked_signals,
        )
        tf_figure = Path(_TF_BASELINE_FIGURE)
        for signal, maps in {"full": tf_baseline, **masked_tf}.items():
            what = "stored sources" if signal == "full" else f"{signal} filter"
            name = (
                tf_figure.name
                if signal == "full"
                else f"{tf_figure.stem}__{signal}{tf_figure.suffix}"
            )
            plot_condition_tf_grid(
                {c: maps[c].mean(axis=1) for c in conditions},
                {c: [""] * n_components for c in conditions},
                conditions,
                times,
                np.asarray(results.freqs),
                args.center_freq,
                f"{variant.value}: {what}, trial-normalised — group mean "
                f"(n={n_participants})",
                plots_dir / f"{plot_prefix}{name}",
                component_prefix=COMPONENT_PREFIX,
                colorbar_label="baseline SD",
            )

    # ---- 6. the grid: selection x window ----------------------------------
    rows_out: list[dict] = []
    snr_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    snr_courses: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for sel_name, bins in store_selections.items():
        take = [union_position[int(b)] for b in cache_selections[sel_name]]
        cut: dict[str, dict[str, np.ndarray]] = {name: {} for name in _SIGNAL_VARIANTS}
        roi_snr: dict[str, np.ndarray] = {}
        roi_scale: dict[str, dict[str, float]] = {}
        for condition in conditions:
            onsets_inside = geometry[condition][0]
            proj = projected[condition]
            # The reference rows, identical in every variant: the masks on the RAW
            # cache, then the PCA mask on the z-scored cache.
            reference_band = np.concatenate(
                [
                    proj.reference_raw[:, :, take, :].mean(axis=2),
                    proj.pca_z[:, :, take, :].mean(axis=2),
                ],
                axis=1,
            )  # (P, 3, T)
            bands = {
                "full": np.stack(
                    [src[:, bins, :].mean(axis=1) for src in stored[condition]]
                ),
                **{
                    s: getattr(proj, _MASKED[s][1])[:, :, take, :].mean(axis=2)
                    for s in masked_signals
                },
            }  # learned rows, each (P, K, T)
            for name, learned in bands.items():
                cut[name][condition], _ = at.cut_trials(
                    np.concatenate([learned, reference_band], axis=1),
                    onsets_inside,
                    pre,
                    post,
                )
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

            for signal in signals:
                value, course = {}, {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+3, N, W)
                    normed, _rel, _pos = at.baseline_normalise(trials, baseline_mask)
                    normed = normed.copy()
                    normed[:, full_mask_column] = roi_snr[condition]
                    per_trial = normed[..., window].mean(axis=-1)
                    value[condition] = np.median(per_trial, axis=2) * signs
                    course[condition] = np.median(normed, axis=2) * signs[:, :, None]

                cell = {
                    "variant": signal,
                    "selection": sel_name,
                    "window": window_label,
                }
                # The course does not depend on the window; every window rewrites it.
                snr_courses[(signal, sel_name)] = course
                snr_values[(signal, sel_name, window_label)] = value
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
    courses_path = csv_path.with_name(f"{csv_path.stem}__courses.npz")
    save_snr_courses(courses_path, snr_courses, participants, labels, times)
    values_path = csv_path.with_name(f"{csv_path.stem}__values.npz")
    save_snr_values(values_path, snr_values, participants, labels)
    tf_path = csv_path.with_name(f"{csv_path.stem}__tf_baseline.npz")
    save_npz_atomic(
        tf_path,
        participants=np.asarray(participants),
        conditions=np.asarray(conditions),
        sources=np.asarray(comp_labels),
        epoch_times=times,
        freqs=np.asarray(results.freqs),
        **{f"tf_baseline__{c}": tf_baseline[c] for c in conditions},
        **{
            f"tf_baseline__{signal}__{c}": maps[c]
            for signal, maps in masked_tf.items()
            for c in conditions
        },
    )
    _logger.info(
        f"saved {csv_path}, {courses_path.name}, {values_path.name} and "
        f"{tf_path.name} "
        f"({len(frame)} test rows, "
        f"n = {n_participants}, floor p = "
        f"{at.p_floor(n_participants):.5f} two-sided)"
    )


def _write_filter_plot(
    plots_dir,
    plot_prefix,
    results,
    ops,
    electrode_mask,
    coordinate_system,
    masked_signals,
) -> None:
    """The restricted shared filters the masked rows read, once per run."""
    try:
        info = results.topo_info(coordinate_system)
    except (ValueError, FileNotFoundError) as exc:
        _logger.warning(f"no topography layout for this store ({exc}); skipping.")
        return
    mask = np.asarray(electrode_mask, dtype=bool)
    figures = {
        "roi_masked": ("masked", int(mask.sum()), _MASKED_FILTER_FIGURE),
        "non_roi_masked": ("outside", int((~mask).sum()), _OUTSIDE_FILTER_FIGURE),
    }
    # roi_masked's figure is drawn whatever the variants, as before.
    for signal in dict.fromkeys(["roi_masked", *masked_signals]):
        operator, n_kept, figure = figures[signal]
        plot_masked_filter_topomaps(
            ops[operator],
            info,
            weight_share=ops[f"{operator}_share"],
            n_mask_channels=n_kept,
            variant=signal,
            save_path=plots_dir / f"{plot_prefix}{figure}",
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
