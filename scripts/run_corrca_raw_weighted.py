"""
Raw voltages weighted by the stage-06b CorrCA topographies, read two ways: their PC1
and their mean over all electrodes (stage 06b).

Read-only: every store must already exist, and nothing is fitted. For each fit of each
run (full, ``_log``, ``_roi``, ``_non_roi``, ``_log_roi``, ``_log_non_roi``) and each
spectrum (broadband, and the ``--bands`` twins), the leading ``--components`` shared
forward patterns are read from the store (``corrca_shared_patterns``):
one topography per component, the same for both conditions. Each is taken in absolute
value: the patterns were fitted on wavelet power, so their sign says where power
co-varies, not where the voltage is positive. An ROI or non-ROI fit weighs every
electrode it did not see at zero. Two fixed weightings are read beside them: the
**binary ASSR ROI** (1 on the ASSR electrodes, 0 elsewhere) and the **uniform** one over
every electrode (all ones), the reference every figure compares the others with.

Per recording (participant x condition), the raw concatenated voltage is z-scored per
channel along the whole recording, cut at every ``fam+`` onset, and every electrode is
multiplied by its weight (the topography filter). The weighted trials are then read
two ways:

* **PC1** — the PCA over channels of the weighted trial mean, by stage 00's own
  ``channel_pca_components`` (``run_assr_raw_pca.py``), gives the PC1 direction, and
  every weighted trial is projected onto it. PC1's sign is aligned across every
  recording of both conditions together (:mod:`src.analysis.pca_polarity`), so a
  condition difference is never a sign flip. The uniform PC1 direction is stage 00's.
* **all data** — the mean over all electrodes, no component separated. With uniform
  weights this is stage 00's channel mean (``run_assr_stimulus_aligned_inspection.py``)
  of the z-scored voltage. The data are average-referenced, so the electrodes sum to
  zero at every sample and a weighting spanning both poles of the evoked dipole
  cancels much of it.

Either signal's trials are normalised against the pre-stimulus window
(:func:`~src.analysis.assr_trials.baseline_normalise`, as the other onset figures
normalise a projected source: each trial minus its own pre-onset mean, divided by the
pre-onset SD pooled over the recording's trials) and averaged. The onset normalisation
acts on the read-out signal only, after the weighting, so every electrode enters on its
z-score and the weights are the only per-electrode scaling. In pre-onset SD units
neither curve depends on the weights' overall scale, so the weightings compare as
they are.

Both readouts are in pre-onset SD units, so every figure of one call shares two
y-ranges, each over both readouts and every run, fit, band, weighting, condition and
participant: the **signal** scale for the curves (the largest participant curve) and
the **difference** scale for the condition and weighting differences (the largest
mean ± SEM). ``--ylim`` / ``--ylim_diff`` pin them instead, e.g. to compare two calls.
Every panel marks the stimulus onset (dashed) and offset (dotted, :class:`AssrEpoch`).
Pass all runs in one ``--variant`` call for one scale across all of them; each
recording is then read and z-scored once and reused by every store. The uniform
reference is set apart in every figure: a dashed dark-grey line in an overlay, a grey
panel face in its own panel.

Outputs, in a ``raw_weighted/`` subdirectory of each fit's own directory, ``<r>`` the
readout (``pc1`` or ``mean``)::

    plots/06b-corrca-condition-comparison/<Join>_<Music>/broadband/<variant>/pca_<N>/
        raw_weighted/raw_weighted_<r>{,_overlay,_differences}.png
        raw_weighted/participants/raw_weighted_<r>_{cc1,cc2,roi,uniform}.png
    plots/06b-corrca-condition-comparison/<Join>_<Music>/bands/<variant>/pca_<N>/
        raw_weighted/<band>_raw_weighted_<r>{,_overlay,_differences}.png
        raw_weighted/participants/<band>_raw_weighted_<r>_*.png

The main figure has one row per condition, their group means and their paired
difference, and one column per weighting. ``_overlay`` draws the uniform weighting's
group mean against each other weighting's, one column each and one with all of them,
per condition. ``_differences`` draws, in the same columns, the paired Psilocybin -
Placebo difference of each weighting against the uniform one's, then per condition each
weighting minus uniform, participant by participant. An individual-condition fit draws
only its own condition, through its own topography, and has no condition-difference
row.

Examples::

    python scripts/run_corrca_raw_weighted.py --experiment assr \\
        --variant corrca_channel_joined_separate corrca_channel_joined_tracks

    python scripts/run_corrca_raw_weighted.py \\
        --variant corrca_channel_joined_tracks --n_subjects 2 --n_times 3000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    CORRCA_INDIVIDUAL_VARIANTS,
    CORRCA_LOG_OF,
    corrca_plots_dir,
    corrca_run_plan,
)
from scripts.run_assr_raw_pca import channel_pca_components  # noqa: E402
from scripts.run_assr_snr_grid import _load_onsets  # noqa: E402
from scripts.run_corrca_band_swap import StoreHead, read_store_head  # noqa: E402
from scripts.run_corrca_snr_grid import _STORE_CONDITIONS, _embed  # noqa: E402
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis.assr_variants import zscore_time  # noqa: E402
from src.analysis.pca_polarity import (  # noqa: E402
    align_pc1_signs,
    apply_pc1_signs,
    topography_consistency,
)
from src.definitions.constants import AssrEpoch  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import iva_results_path  # noqa: E402
from src.io.loading import assr_electrode_mask  # noqa: E402
from src.io.loading import concatenated_dir as _concatenated_dir  # noqa: E402
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.corrca_plots import (  # noqa: E402
    COMPONENT_PREFIX,
    difference_extent,
    onset_curve_extent,
    plot_difference_comparison,
    plot_participant_weighted_curves,
    plot_weighted_onset_curves,
    plot_weighting_overlay,
)

_logger = logging.getLogger(__name__)

#: Subdirectory of each fit's directory, and the figure prefix.
READING = "raw_weighted"
#: ``--bands`` token of the broadband stores, which carry no band in their name.
BROADBAND = "broadband"
UNIFORM_LABEL = "all electrodes (uniform)"
ROI_LABEL = "binary ASSR ROI"
#: The two readouts: file slug -> (figure title, y label, how the signal is read).
READOUTS = {
    "pc1": (
        "Stimulus-locked PC1 of the weighted raw voltage",
        "PC1 (pre-onset SD, polarity-aligned)",
        "PCA of the trial mean -> every trial on PC1",
    ),
    "mean": (
        "Stimulus-locked mean over all electrodes of the weighted raw voltage",
        "Weighted channel mean (pre-onset SD)",
        "mean over all electrodes",
    ),
}


def _note(readout: str) -> str:
    return (
        "voltage z-scored per channel -> electrodes x weight (|shared pattern|, binary "
        f"ROI or uniform) -> {READOUTS[readout][2]} -> every trial minus its "
        "pre-onset mean / pre-onset SD pooled over trials -> trial mean"
    )


# Metadata CSV column names (the sidecar stores enum keys as their string repr).
_PID_COL = "SingleDataMetadata.PARTICIPANT_ID"
_PIDX_COL = "SingleDataMetadata.CONCATENATED_PERSON_INDEX"


class StoreSpec(NamedTuple):
    """One store's weightings and where its figures go, before any data is read."""

    out_dir: Path
    prefix: str
    label: str
    conditions: list[str]
    participants: list[str]
    weight_labels: list[str]
    slugs: list[str]
    #: ``(weightings, C)`` electrode weights.
    weights: np.ndarray


class StoreReading(NamedTuple):
    """Everything one store's figures draw."""

    out_dir: Path
    prefix: str
    label: str
    participants: list[str]
    weight_labels: list[str]
    slugs: list[str]
    #: Readout -> condition -> ``(participants, weightings, epoch)`` curves.
    curves: dict[str, dict[str, np.ndarray]]
    #: Condition -> ``(participants, weightings)`` PC1 explained-variance ratio.
    explained: dict[str, np.ndarray]


def abs_pattern_weights(
    patterns: np.ndarray, store_channels: list[str], read_channels: list[str]
) -> np.ndarray:
    """``(K, read channels)`` electrode weights: ``|pattern|``.

    Electrodes absent from the store (an ROI or non-ROI cut) weigh zero. The overall
    scale is irrelevant: the curves are read in pre-onset SD units.

    :param patterns: ``(K, store channels)`` shared forward patterns.
    :raises ValueError: If a pattern is zero on every electrode.
    """
    weights = np.abs(_embed(patterns, store_channels, read_channels))
    if not np.all(weights.any(axis=1)):
        raise ValueError("A shared pattern is zero on every electrode.")
    return weights


def binary_roi_weights(channels: list[str]) -> np.ndarray:
    """``(C,)`` the binary ASSR ROI as a weighting: 1 on its electrodes, 0 elsewhere.

    The montage's intersection with the ASSR list, as the ROI fits take it.

    :raises ValueError: If the montage holds no ASSR electrode.
    """
    mask = assr_electrode_mask(channels, strict=False).astype(float)
    if not mask.any():
        raise ValueError("The montage holds no ASSR electrode.")
    return mask


def zscored_trials(
    recording: np.ndarray, onsets: np.ndarray, pre: int, post: int
) -> np.ndarray:
    """``(C, N, pre + post)`` trials of one recording, z-scored per channel along the
    whole recording first; windows overhanging either end are dropped."""
    trials, _kept = at.cut_trials(
        zscore_time(np.asarray(recording, dtype=np.float64)), onsets, pre, post
    )
    return trials


def _onset_mean(signal_trials: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Trial mean of ``(N, W)`` trials after :func:`~src.analysis.assr_trials.baseline_normalise`."""
    z, _rel, _positive = at.baseline_normalise(signal_trials, baseline_mask)
    return z.mean(axis=0)


def mean_onset_curve(
    trials: np.ndarray, weights: np.ndarray, baseline_mask: np.ndarray
) -> np.ndarray:
    """One recording's onset-normalised mean over all electrodes under one weighting.

    :param trials: ``(C, N, W)`` z-scored trials.
    :param weights: ``(C,)`` electrode weights, multiplied in electrode by electrode.
    :param baseline_mask: ``(W,)`` the pre-onset samples.
    :return: ``(W,)`` curve in pre-onset SD units.
    """
    signal = np.tensordot(weights, trials, axes=(0, 0)) / trials.shape[0]  # (N, W)
    return _onset_mean(signal, baseline_mask)


def pc1_onset_curve(
    trials: np.ndarray,
    weights: np.ndarray,
    baseline_mask: np.ndarray,
    evoked: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One recording's onset-normalised PC1 curve under one weighting.

    The PCA of the weighted trial mean gives the PC1 direction, and every weighted
    trial is projected on it. The projection skips the PCA's centring: that only
    shifts each trial by a constant, which the per-trial baseline subtraction removes.

    :param trials: ``(C, N, W)`` z-scored trials.
    :param weights: ``(C,)`` electrode weights, multiplied in electrode by electrode.
    :param baseline_mask: ``(W,)`` the pre-onset samples.
    :param evoked: ``trials.mean(axis=1)``, when the caller shares it across
        weightings.
    :return: ``(curve (W,), loading (C,), explained ratio)``, the sign as the PCA left
        it.
    """
    if evoked is None:
        evoked = trials.mean(axis=1)
    _, loading, ratio = channel_pca_components(weights[:, None] * evoked, 1)
    signal = np.tensordot(loading[0] * weights, trials, axes=(0, 0))  # (N, W)
    return _onset_mean(signal, baseline_mask), loading[0], float(ratio[0])


def align_recordings(
    scores: np.ndarray, loadings: np.ndarray, weights: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, np.ndarray, object, str | None]:
    """One PC1 orientation for every recording of one weighting.

    The template alignment of stage 00, on the electrodes the weighting keeps, so the
    orientation anchor falls on a weighted electrode.

    :param scores: ``(R, W)`` PC1 curves of every recording, any conditions.
    :param loadings: ``(R, C)`` their loadings.
    :param weights: ``(C,)`` the weighting.
    :param channels: Names of the ``C`` channels.
    :return: ``(scores, loadings, consistency, anchor)``, the first two sign-aligned.
    """
    used = weights > 0
    signs, anchor = align_pc1_signs(
        loadings[:, used], channel_names=[n for n, u in zip(channels, used) if u]
    )
    aligned = apply_pc1_signs(loadings, signs)
    return (
        apply_pc1_signs(scores, signs),
        aligned,
        topography_consistency(aligned[:, used]),
        anchor,
    )


def participant_rows(meta_path: Path) -> dict[str, int]:
    """Participant ID (zero-padded to three) -> row of the concatenated array."""
    meta = pd.read_csv(meta_path, index_col=0)
    return dict(
        zip(meta[_PID_COL].astype(str).str.zfill(3), meta[_PIDX_COL].astype(int))
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weight the z-scored raw voltages by each stage-06b fit's absolute "
        "shared topographies, the binary ASSR ROI and uniformly; draw the "
        "onset-normalised PC1 and mean over all electrodes of each.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stored = parser.add_argument_group("the stored decompositions")
    stored.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    stored.add_argument(
        "--variant",
        nargs="+",
        default=[IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE.value],
        choices=[v.value for v in CORRCA_LOG_OF],
        help="Which stage-06b runs: each full fit and every '_log', '_roi', "
        "'_non_roi' one are read. Several share one y-scale.",
    )
    stored.add_argument("--skip_log_power", action="store_true")
    stored.add_argument("--skip_roi", action="store_true")
    stored.add_argument("--skip_non_roi", action="store_true")
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument(
        "--bands",
        nargs="+",
        default=[BROADBAND, "30-50hz"],
        help=f"Store band tokens to read; '{BROADBAND}' is the store without one.",
    )
    stored.add_argument("--n_pca", type=int, default=5)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the raw voltages and the cohort")
    data.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help="Root of the wavelet store, read only for its channel names.",
    )
    data.add_argument("--concat_dir", type=Path, default=None)
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions to read; for the individual-condition stores, each on its "
        "own. Default Placebo Psilocybin.",
    )
    data.add_argument(
        "--n_subjects",
        type=int,
        default=None,
        help="Keep the first N participants of each run (a quick check).",
    )
    data.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Trim the time axis before the z-score (a quick check; not for real "
        "runs).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--components",
        type=int,
        default=2,
        help="Weight by this many leading components' topographies.",
    )
    output.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="Fixed symmetric y-limit of every curve panel (both readouts); default "
        "the largest participant curve drawn.",
    )
    output.add_argument(
        "--ylim_diff",
        type=float,
        default=None,
        help="The same for every difference panel; default the largest mean ± SEM "
        "drawn.",
    )
    output.add_argument("--save_dir", type=Path, default=None)
    output.add_argument("--log_level", default="INFO")
    return parser


def _store_runs(
    args: argparse.Namespace,
) -> list[tuple[ConditionVariants, list[str], list]]:
    """``(store condition, conditions, heads)`` of every run and join with a store."""
    skips = argparse.Namespace(
        skip_log_power=args.skip_log_power,
        skip_roi_fit=args.skip_roi,
        skip_non_roi_fit=args.skip_non_roi,
    )
    conditions = at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))
    runs = []
    for name in args.variant:
        variant = IvaVariants(name)
        plan = corrca_run_plan(variant, skips)
        if variant in CORRCA_INDIVIDUAL_VARIANTS:
            stores = [(ConditionVariants(c), [c]) for c in conditions]
        else:
            stores = [(_STORE_CONDITIONS[variant], conditions)]
        for store_condition, store_conditions in stores:
            heads = _heads(args, plan, store_condition)
            if heads:
                runs.append((store_condition, store_conditions, heads))
            else:
                _logger.warning(
                    f"[{variant.value} / {store_condition.value}] no store: skipped."
                )
    return runs


def _heads(
    args: argparse.Namespace,
    plan: list[tuple[IvaVariants, str, bool]],
    store_condition: ConditionVariants,
) -> list[tuple[IvaVariants, bool, str | None, StoreHead]]:
    """``(variant, log_power, band, head)`` of every store of the run that exists."""
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    heads = []
    for fit, _cut, log_power in plan:
        for token in args.bands:
            band = None if token == BROADBAND else token
            path = iva_results_path(
                experiment,
                store_condition,
                fit,
                music_type,
                band,
                args.n_pca,
                processed_data_dir=args.store_dir,
            )
            if not path.is_file():
                _logger.warning(f"no {fit.value} store at {path}: skipped.")
                continue
            heads.append((fit, log_power, band, read_store_head(path)))
    return heads


def run(args: argparse.Namespace) -> None:
    """Read every store of every run, then draw all of them on one scale."""
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    runs = _store_runs(args)
    if not runs:
        _logger.warning("No store found: nothing to do.")
        return
    sfreqs = {head.sfreq for *_, heads in runs for *_, head in heads}
    if len(sfreqs) > 1:
        raise ValueError(f"The stores differ in their sampling rate: {sorted(sfreqs)}.")
    sfreq = sfreqs.pop()

    # ---- 1. the raw arrays, their channels, rows, onsets and one epoch -------
    all_conditions = list(dict.fromkeys(c for _, conds, _ in runs for c in conds))
    concat_dir = _concatenated_dir(experiment, args.concat_dir)
    raws, rows, onsets = {}, {}, {}
    channels: list[str] = []
    for condition in all_conditions:
        label = f"{condition}_{music_type.value}"
        raws[condition] = np.load(concat_dir / f"{label}.npy", mmap_mode="r")
        rows[condition] = participant_rows(concat_dir / f"{label}.metadata.csv")
        onsets[condition] = _load_onsets(concat_dir, label)
        names = list(
            find_wavelet_set(
                wavelet_label_dir(args.wavelet_data_dir, experiment, label)
            ).channel_names
        )
        if len(names) != raws[condition].shape[1]:
            raise ValueError(
                f"{label}: the wavelet store names {len(names)} channels, the raw "
                f"array holds {raws[condition].shape[1]}."
            )
        if channels and names != channels:
            raise ValueError("The conditions' raw arrays differ in their channels.")
        channels = names
    n_times = {
        c: raws[c].shape[-1]
        if args.n_times is None
        else min(args.n_times, raws[c].shape[-1])
        for c in all_conditions
    }
    geometry = {
        c: at.epoch_geometry(onsets[c], n_times[c], sfreq) for c in all_conditions
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)

    # ---- 2. every store's weightings -----------------------------------------
    specs = []
    for store_condition, conditions, heads in runs:
        participants = [
            p
            for p in dict.fromkeys(p for *_, head in heads for p in head.participants)
            if all(p in rows[c] for c in conditions)
        ]
        if args.n_subjects is not None:
            participants = participants[: args.n_subjects]
        if not participants:
            raise ValueError(
                f"[{store_condition.value}] no store participant is in every "
                "condition's raw array."
            )
        for fit, log_power, band, head in heads:
            specs.append(
                store_spec(
                    args,
                    fit,
                    log_power,
                    band,
                    head,
                    store_condition,
                    conditions,
                    [p for p in participants if p in head.participants],
                    channels,
                )
            )

    # ---- 3. one pass over the recordings, every store's curves from each ------
    users: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, spec in enumerate(specs):
        for condition in spec.conditions:
            for row, name in enumerate(spec.participants):
                users.setdefault((condition, name), []).append((i, row))
    n_samples = pre + post
    baseline_mask = np.arange(n_samples) < pre

    def _empty(spec: StoreSpec, *shape: int) -> dict[str, np.ndarray]:
        return {
            c: np.empty((len(spec.participants), len(spec.weights), *shape))
            for c in spec.conditions
        }

    results = [
        {
            "mean": _empty(s, n_samples),
            "pc1": _empty(s, n_samples),
            "loadings": _empty(s, len(channels)),
            "explained": _empty(s),
        }
        for s in specs
    ]
    for (condition, name), uses in users.items():
        started = time.time()
        trials = zscored_trials(
            raws[condition][rows[condition][name], :, : n_times[condition]],
            geometry[condition][0],
            pre,
            post,
        )
        evoked = trials.mean(axis=1)
        for i, row in uses:
            result = results[i]
            for k, weights in enumerate(specs[i].weights):
                result["mean"][condition][row, k] = mean_onset_curve(
                    trials, weights, baseline_mask
                )
                (
                    result["pc1"][condition][row, k],
                    result["loadings"][condition][row, k],
                    result["explained"][condition][row, k],
                ) = pc1_onset_curve(trials, weights, baseline_mask, evoked)
        _logger.info(
            f"[{condition}]   {name}: {trials.shape[1]} trials, {len(uses)} store(s) "
            f"({time.time() - started:.1f}s)"
        )
        del trials
    readings = [
        finish_store(spec, result, channels) for spec, result in zip(specs, results)
    ]

    # ---- 4. one signal and one difference y-range, both readouts ------------
    ylim = args.ylim or 1.05 * max(
        onset_curve_extent(r.curves[readout]) for r in readings for readout in READOUTS
    )
    diff_ylim = args.ylim_diff or 1.05 * max(
        difference_extent(r.curves[readout], r.weight_labels.index(UNIFORM_LABEL))
        for r in readings
        for readout in READOUTS
    )
    _logger.info(f"y-range: signal ±{ylim:.3g}, differences ±{diff_ylim:.3g}")
    for reading in readings:
        draw_reading(reading, times, ylim, diff_ylim)


def store_spec(
    args: argparse.Namespace,
    fit: IvaVariants,
    log_power: bool,
    band: str | None,
    head: StoreHead,
    store_condition: ConditionVariants,
    conditions: list[str],
    participants: list[str],
    channels: list[str],
) -> StoreSpec:
    """One store's weightings, labels and output directory."""
    music_type = MusicTypeVariants(args.music_type)
    n_components = min(args.components, head.patterns.shape[0])
    weights = np.vstack(
        [
            abs_pattern_weights(head.patterns[:n_components], head.channels, channels),
            binary_roi_weights(channels),
            np.ones(len(channels)),
        ]
    )
    out_dir = (
        corrca_plots_dir(
            fit,
            store_condition,
            music_type,
            band=band,
            n_pca=args.n_pca,
            root=args.save_dir,
        )
        / READING
    )
    return StoreSpec(
        out_dir=out_dir,
        prefix=READING if band is None else f"{band}_{READING}",
        label=f"{store_condition.value}_{music_type.value}"
        + ("_log" if log_power else "")
        + f" [{band or BROADBAND}] {fit.value}",
        conditions=conditions,
        participants=participants,
        weight_labels=[
            f"|{COMPONENT_PREFIX} {k + 1}| topography" for k in range(n_components)
        ]
        + [ROI_LABEL, UNIFORM_LABEL],
        slugs=[f"cc{k + 1}" for k in range(n_components)] + ["roi", "uniform"],
        weights=weights,
    )


def finish_store(
    spec: StoreSpec, result: dict[str, dict[str, np.ndarray]], channels: list[str]
) -> StoreReading:
    """Align each weighting's PC1 over every recording of every condition at once.

    :param result: ``"mean"`` / ``"pc1"`` -> condition -> ``(P, K, W)`` curves,
        ``"loadings"`` -> ``(P, K, C)``, ``"explained"`` -> ``(P, K)``; PC1 signs as
        the PCA left them.
    """
    conditions, n_rec = spec.conditions, len(spec.participants)
    pc1 = result["pc1"]
    for k, weight_label in enumerate(spec.weight_labels):
        s, _ld, consistency, anchor = align_recordings(
            np.concatenate([pc1[c][:, k] for c in conditions]),
            np.concatenate([result["loadings"][c][:, k] for c in conditions]),
            spec.weights[k],
            channels,
        )
        for i, condition in enumerate(conditions):
            pc1[condition][:, k] = s[i * n_rec : (i + 1) * n_rec]
        _logger.info(
            f"{spec.label} {weight_label}: PC1 anchor {anchor}, "
            f"{consistency.n_agreeing}/{consistency.n_subjects} topographies agree, "
            f"median pairwise r {consistency.median_pairwise_r:+.2f}"
        )
    return StoreReading(
        out_dir=spec.out_dir,
        prefix=spec.prefix,
        label=spec.label,
        participants=spec.participants,
        weight_labels=spec.weight_labels,
        slugs=spec.slugs,
        curves={"pc1": pc1, "mean": result["mean"]},
        explained=result["explained"],
    )


def draw_reading(
    reading: StoreReading, times: np.ndarray, ylim: float, diff_ylim: float
) -> None:
    """Both readouts of one store: the group figure, the uniform-vs-weighting overlay,
    the differences and one participant grid per weighting."""
    reference = reading.weight_labels.index(UNIFORM_LABEL)
    ev_notes = {
        condition: [f"PC1 EV {100 * ev:.0f}%" for ev in ratios.mean(axis=0)]
        for condition, ratios in reading.explained.items()
    }
    out = reading.out_dir
    for readout, (title, ylabel, _how) in READOUTS.items():
        curves = reading.curves[readout]
        prefix = f"{reading.prefix}_{readout}"
        common: dict[str, Any] = {
            "label": reading.label,
            "title": title,
            "ylabel": ylabel,
            "stimulus_end": AssrEpoch.STIMULUS_DURATION_S,
        }
        plt.close(
            plot_weighted_onset_curves(
                curves,
                times,
                reading.weight_labels,
                note=_note(readout),
                panel_notes=ev_notes if readout == "pc1" else None,
                ylim=ylim,
                diff_ylim=diff_ylim,
                reference_index=reference,
                save_path=out / f"{prefix}.png",
                **common,
            )
        )
        plt.close(
            plot_weighting_overlay(
                curves,
                times,
                reading.weight_labels,
                reference,
                note=_note(readout),
                ylim=ylim,
                save_path=out / f"{prefix}_overlay.png",
                **common,
            )
        )
        plt.close(
            plot_difference_comparison(
                curves,
                times,
                reading.weight_labels,
                reference,
                note=_note(readout),
                ylim=diff_ylim,
                save_path=out / f"{prefix}_differences.png",
                **common,
            )
        )
        for k, (weight_label, slug) in enumerate(
            zip(reading.weight_labels, reading.slugs)
        ):
            plt.close(
                plot_participant_weighted_curves(
                    curves,
                    times,
                    reading.participants,
                    k,
                    weight_label=weight_label,
                    ylim=ylim,
                    reference=k == reference,
                    save_path=out / "participants" / f"{prefix}_{slug}.png",
                    **common,
                )
            )
    _logger.info(f"wrote {reading.out_dir}")


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
