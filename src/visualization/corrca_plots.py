"""
Figures specific to a CorrCA condition comparison.

The condition-mean and per-participant topography / TF grids are shared with the IVA
runs (:mod:`src.visualization.iva_condition_plots`). What CorrCA adds, and what is
drawn here, is what has no IVA counterpart:

* the **shared** forward pattern of each component — CorrCA fits one filter for every
  recording, so there is one group topography the model itself defines — and the
  shared spatial **filter** itself, the channel weights that produce the TF maps;
* the **synchrony per condition** — the ISC of each component within each condition,
  from the same filters, with every participant's leave-one-out ISC paired across
  conditions. The stage-06 IVA runs draw it too, on their sign-anchored sources;
* the **participant influence** — a leave-one-participant-out refit per participant,
  read as how much ISC and topography each component loses without them, as each
  recording's in-sample, held-out and own-pair ISC, and as the condition comparison
  redone in every leave-one-out replicate;
* the **raw voltages through a topography** — the onset-normalised raw signal with
  each electrode multiplied by a component's absolute shared pattern (or the binary
  ASSR ROI), read as its PC1 and as its mean over all electrodes, each
  onset-normalised, against the uniform reference
  (``scripts/run_corrca_raw_weighted.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.assr_snr_plots import CONDITION_COLORS
from src.visualization.iva_quality_plots import (
    _grid_shape,
    safe_vlim,
    save_fig,
    topo_info_subset,
)

#: Component label prefix ("correlated component"), used instead of IVA's "IC".
COMPONENT_PREFIX = "CC"


def plot_shared_topomaps(
    patterns: np.ndarray,
    isc: np.ndarray,
    info,
    n_channels: int,
    comp_indices: Sequence[int],
    *,
    label: str,
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
    image_interp: str = "cubic",
    contours: int = 4,
    cmap="RdBu_r",
    title: str = "Shared CorrCA topographies (pooled fit)",
):
    """One row of shared channel maps, one column per component.

    Draws the forward patterns by default; the filters (``CorrcaResult.filters.T``)
    go through the same layout with their own *title*. Each panel is on its own
    symmetric scale: CorrCA fixes a component's scale, not a common one across
    components.

    :param patterns: ``(K, C)`` shared maps — forward patterns
        (``CorrcaResult.patterns``) or filters (``CorrcaResult.filters.T``).
    :param isc: ``(K,)`` ISC of the fit, shown in each title.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels on the channel axis.
    :param comp_indices: 0-based component indices, one column each.
    :param label: Dataset label shown in the title.
    :param alignment_note: What fixed each component's sign, named in the title.
    :param save_path: Optional output path.
    :param image_interp: Interpolation between electrodes (``mne.viz.plot_topomap``);
        ``"nearest"`` keeps a two-valued map, such as an electrode mask, two-valued.
    :param contours: Number of contour lines; ``0`` for none.
    :param cmap: Colour map (a name or a ``Colormap``), on each panel's symmetric
        scale.
    :param title: Figure title, before the dataset label.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    """
    from mne.viz import plot_topomap

    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    topo_info = topo_info_subset(info, n_channels)
    fig, axes = plt.subplots(
        1,
        len(comp_indices),
        figsize=(2.4 * len(comp_indices), 3.0),
        squeeze=False,
        layout="constrained",
    )
    for j, k in enumerate(comp_indices):
        vlim = safe_vlim(patterns[k])
        plot_topomap(
            patterns[k],
            topo_info,
            axes=axes[0, j],
            show=False,
            cmap=cmap,
            vlim=(-vlim, vlim),
            contours=contours,
            image_interp=image_interp,
        )
        axes[0, j].set_title(
            f"{COMPONENT_PREFIX} {k + 1}\nISC {isc[k]:.3f}  (|max| {vlim:.3g})",
            fontsize=8,
        )
    title = f"{title} — {label}"
    if alignment_note is not None:
        title += f"\nsign: {alignment_note}"
    fig.suptitle(title, fontsize=11)
    save_fig(fig, save_path)
    return fig


def plot_condition_isc(
    group: pd.DataFrame,
    per_participant: pd.DataFrame,
    tests: Optional[pd.DataFrame],
    conditions: Sequence[str],
    comp_indices: Sequence[int],
    *,
    label: str,
    fit_isc: Optional[np.ndarray] = None,
    component_prefix: str = COMPONENT_PREFIX,
    save_path: Optional[Path] = None,
):
    """Per-component synchrony in each condition, participants paired.

    One panel per component. Each participant's leave-one-out ISC is a dot per
    condition, joined by a grey line to their other condition, so a shift carried by
    the whole group reads as parallel slopes and one carried by a few as a couple of
    steep lines. The short bar per condition is that condition's group ISC.

    :param group: First frame of :func:`~src.analysis.corrca.condition_isc`.
    :param per_participant: Second frame of the same.
    :param tests: Output of :func:`~src.analysis.corrca.paired_condition_test`, or
        ``None`` to omit the test from the titles.
    :param conditions: Conditions in x order.
    :param comp_indices: 0-based component indices, one panel each.
    :param label: Dataset label shown in the title.
    :param fit_isc: ``(K,)`` ISC of the pooled fit, shown in each title.
    :param component_prefix: Component label prefix; ``"IC"`` when the stage-06 IVA
        runs draw this figure.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    """
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    conditions = list(conditions)
    colors = {c: CONDITION_COLORS.get(c, "0.4") for c in conditions}
    xs = np.arange(len(conditions))

    n_rows, n_cols = _grid_shape(len(comp_indices))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    for ax in axes.flat[len(comp_indices) :]:
        ax.set_visible(False)

    for ax, k in zip(axes.flat, comp_indices):
        component = k + 1
        rows = per_participant[per_participant["component"] == component]
        wide = rows.pivot_table(
            index="participant", columns="condition", values="loo_isc"
        )
        wide = wide.reindex(columns=conditions)
        for _participant, values in wide.iterrows():
            ax.plot(xs, values.to_numpy(), color="0.75", lw=0.8, zorder=1)
        for x, condition in zip(xs, conditions):
            ax.scatter(
                np.full(len(wide), x),
                wide[condition],
                s=22,
                color=colors[condition],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            match = group[
                (group["component"] == component) & (group["condition"] == condition)
            ]
            if not match.empty:
                value = float(match["isc"].iloc[0])
                ax.hlines(
                    value, x - 0.25, x + 0.25, color=colors[condition], lw=2.5, zorder=2
                )
        ax.axhline(0.0, ls="--", lw=0.6, color="0.5", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(conditions, fontsize=8)
        ax.set_xlim(-0.5, len(conditions) - 0.5)

        title = f"{component_prefix} {component}"
        if fit_isc is not None:
            title += f"  (fit ISC {fit_isc[k]:.3f})"
        if tests is not None:
            test = tests[tests["component"] == component]
            if not test.empty and np.isfinite(test["p_value"].iloc[0]):
                title += (
                    f"\nΔmedian {test['median_difference'].iloc[0]:+.3f}, "
                    f"p {test['p_value'].iloc[0]:.3g}"
                )
        ax.set_title(title, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Leave-one-out ISC (r)")

    fig.suptitle(
        f"Component synchrony per condition — {label}\n"
        "dots = participants (paired by line), bar = group ISC; "
        "Wilcoxon signed-rank, uncorrected",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


def _component_grid(n_panels: int, *, width: float, height: float):
    n_rows, n_cols = _grid_shape(n_panels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width * n_cols, height * n_rows),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes


#: Colour of a participant flagged as an influence outlier.
OUTLIER_COLOR = "#c0392b"


def plot_participant_influence(
    influence: pd.DataFrame,
    comp_indices: Sequence[int],
    *,
    label: str,
    component_prefix: str = COMPONENT_PREFIX,
    save_path: Optional[Path] = None,
):
    """What each component loses when one participant is left out of the fit.

    One panel per component, one dot per participant: x is the ISC the component
    loses without them (full-fit ISC minus the matched refit's), y the |r| between the
    full and the refit topography. A component carried by the group is a tight cloud
    near x = 0, y = 1; one carried by a few has those few far right and low. Flagged
    outliers (robust z above the threshold and a large enough share of the ISC) are
    red and named, as is the participant with the largest loss.

    :param influence: Output of
        :func:`~src.analysis.corrca.channel_corrca_influence`.
    :param comp_indices: 0-based component indices, one panel each.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    """
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    per_participant = influence.drop_duplicates(["participant", "component"])
    fig, axes = _component_grid(len(comp_indices), width=2.8, height=2.8)
    for ax, k in zip(axes.flat, comp_indices):
        rows = per_participant[per_participant["component"] == k + 1]
        flagged = rows["outlier"].to_numpy(dtype=bool)
        ax.axvline(0.0, ls="--", lw=0.6, color="0.5", zorder=0)
        ax.scatter(
            rows["influence"],
            rows["pattern_r"],
            s=24,
            c=np.where(flagged, OUTLIER_COLOR, "0.35"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        named = rows[flagged]
        if len(rows) and np.isfinite(rows["influence"]).any():
            named = pd.concat([named, rows.loc[[rows["influence"].idxmax()]]])
        for row in named.drop_duplicates("participant").itertuples():
            ax.annotate(
                str(row.participant),
                (row.influence, row.pattern_r),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=OUTLIER_COLOR if row.outlier else "0.2",
            )
        ax.set_ylim(min(0.0, float(np.nanmin(rows["pattern_r"], initial=0.0))), 1.05)
        isc_full = rows["isc_full"].iloc[0] if len(rows) else np.nan
        ax.set_title(
            f"{component_prefix} {k + 1}  (fit ISC {isc_full:.3f})\n"
            f"{int(flagged.sum())} outlier(s) of {len(rows)}",
            fontsize=8,
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("ISC lost without participant", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("|r| full vs refit topography", fontsize=8)
    fig.suptitle(
        f"Leave-one-participant-out influence — {label}\n"
        "dot = participant; right and low = the component leans on them; "
        "red = flagged outlier (robust z and share of the ISC)",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


#: Columns of :func:`plot_participant_heldout_isc`, left to right.
_ISC_READINGS = {
    "insample_isc": "in-sample",
    "heldout_isc": "held-out",
    "within_participant_isc": "own pair",
}


def plot_participant_heldout_isc(
    influence: pd.DataFrame,
    comp_indices: Sequence[int],
    *,
    label: str,
    component_prefix: str = COMPONENT_PREFIX,
    save_path: Optional[Path] = None,
):
    """Each recording's synchrony with the others, through the full and the refit
    filter, and with its own participant's other recording.

    One panel per component, one line per recording across three readings: in-sample
    (the full fit's filter, which the recording helped choose), held-out (the filter
    refit without its participant) and own pair (the full filter, against the same
    participant's other recording; dropped when every participant has one row). Every
    other-recording score uses the same reference, the sum of every other
    participant, so in-sample minus held-out is the recording's own pull on the
    filter. A component the group shares keeps its level from in-sample to held-out;
    one fitted to a few drops to zero there, or lives in the own-pair column alone.

    :param influence: Output of
        :func:`~src.analysis.corrca.channel_corrca_influence`.
    :param comp_indices: 0-based component indices, one panel each.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    """
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    readings = [
        c
        for c in _ISC_READINGS
        if c != "within_participant_isc" or influence[c].notna().any()
    ]
    xs = np.arange(len(readings))
    fig, axes = _component_grid(
        len(comp_indices), width=0.9 * len(readings) + 0.6, height=3.2
    )
    shared_ylim = [np.inf, -np.inf]
    for ax, k in zip(axes.flat, comp_indices):
        rows = influence[influence["component"] == k + 1]
        for row in rows.itertuples():
            values = [getattr(row, c) for c in readings]
            line_color = OUTLIER_COLOR if row.outlier else "0.75"
            ax.plot(xs, values, color=line_color, lw=0.8, zorder=1)
            dot = CONDITION_COLORS.get(row.condition, "0.35")
            ax.scatter(xs, values, s=18, color=dot, edgecolor="white", lw=0.6, zorder=3)
            if row.outlier:
                # Named at the held-out reading, where an outlier's drop shows.
                ax.annotate(
                    str(row.participant),
                    (xs[1], values[1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6,
                    color=OUTLIER_COLOR,
                    zorder=4,
                )
        for x, column in zip(xs, readings):
            median = float(np.nanmedian(rows[column])) if len(rows) else np.nan
            ax.hlines(median, x - 0.3, x + 0.3, color="black", lw=2.0, zorder=2)
        values = rows[readings].to_numpy(dtype=float)
        if np.isfinite(values).any():
            shared_ylim[0] = min(shared_ylim[0], float(np.nanmin(values)))
            shared_ylim[1] = max(shared_ylim[1], float(np.nanmax(values)))
        ax.axhline(0.0, ls="--", lw=0.6, color="0.5", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([_ISC_READINGS[c] for c in readings], fontsize=7)
        ax.set_xlim(-0.5, len(readings) - 0.5)
        ax.set_title(f"{component_prefix} {k + 1}", fontsize=8)
        ax.tick_params(labelsize=7)
    if np.isfinite(shared_ylim).all():
        pad = 0.05 * (shared_ylim[1] - shared_ylim[0] or 1.0)
        for ax in axes.flat:
            ax.set_ylim(shared_ylim[0] - pad, shared_ylim[1] + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel("ISC (r) with the reference", fontsize=8)
    fig.suptitle(
        f"Per-recording synchrony, full vs leave-participant-out filter — {label}\n"
        "line = recording (red, named = influence outlier), bar = median; "
        "reference = sum of every other participant",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


def _onset_lines(ax, times: np.ndarray, stimulus_end: Optional[float] = None) -> None:
    ax.axvline(0.0, color="red", ls="--", lw=0.8)
    if stimulus_end is not None:
        ax.axvline(stimulus_end, color="red", ls=":", lw=0.8)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlim(times[0], times[-1])


#: How the reference weighting (uniform over every electrode) is set apart: a dashed
#: dark-grey line in an overlay, a light-grey panel face where it has its own panel.
REFERENCE_COLOR = "0.3"
REFERENCE_FACE = "#ececec"


def _mean_band(
    ax, times: np.ndarray, data: np.ndarray, color, label: str | None = None
) -> None:
    """Mean ± SEM over the first axis; the reference colour draws dashed."""
    mean, sem = data.mean(axis=0), _sem(data)
    reference = color == REFERENCE_COLOR
    ax.plot(
        times,
        mean,
        color=color,
        lw=2.0 if reference else 1.6,
        ls="--" if reference else "-",
        label=label,
        zorder=3 if reference else 2,
    )
    ax.fill_between(
        times,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.12 if reference else 0.18,
        lw=0,
    )


def _sem(data: np.ndarray) -> np.ndarray:
    if data.shape[0] < 2:
        return np.zeros(data.shape[1:])
    return data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])


def paired_difference(
    curves: Mapping[str, np.ndarray],
) -> tuple[str, np.ndarray] | None:
    """``(label, second - first)`` per participant, or ``None`` unless exactly two
    conditions with the same participant rows are given.

    :param curves: Condition -> ``(participants, ...)``, rows in the same order.
    """
    if len(curves) != 2:
        return None
    (first, a), (second, b) = curves.items()
    if a.shape != b.shape:
        return None
    return f"{second} − {first}", b - a


def onset_curve_extent(curves: Mapping[str, np.ndarray]) -> float:
    """Largest ``|value|`` of any participant's curve: the signal scale's extent."""
    return max(float(np.nanmax(np.abs(data))) for data in curves.values())


def _mean_sem_extent(data: np.ndarray) -> float:
    return float(np.nanmax(np.abs(data.mean(axis=0)) + _sem(data)))


def _set_rows_ylim(axes: np.ndarray, rows: Sequence[int], ylim: float) -> None:
    for row in rows:
        for c, ax in enumerate(axes[row]):
            ax.set_ylim(-ylim, ylim)
            if c:
                ax.tick_params(labelleft=False)


def plot_weighted_onset_curves(
    curves: Mapping[str, np.ndarray],
    times: np.ndarray,
    weight_labels: Sequence[str],
    *,
    label: str,
    title: str,
    ylabel: str,
    note: Optional[str] = None,
    panel_notes: Optional[Mapping[str, Sequence[str]]] = None,
    ylim: Optional[float] = None,
    diff_ylim: Optional[float] = None,
    reference_index: Optional[int] = None,
    stimulus_end: Optional[float] = None,
    save_path: Optional[Path] = None,
):
    """Stimulus-locked curves per weighting: one row per condition, one column per
    weighting; with two conditions, a row of their group means and one of their
    paired difference.

    A condition's row overlays its participants (thin) and their mean (black). The
    group-mean row draws each condition's mean ± SEM, the difference row the mean ±
    SEM of every participant's second-minus-first condition. The curve rows share one
    y-range, the difference row its own: each scale compares like with like.

    :param curves: Condition -> ``(participants, weightings, times)``; with two
        conditions, rows are the same participants in the same order.
    :param times: ``(times,)`` epoch time base in seconds, ``0`` at the onset.
    :param weight_labels: One title per weighting, in the curves' order.
    :param label: Dataset label shown in the title.
    :param title: What the curves are, the title's first part.
    :param ylabel: Y-axis label.
    :param note: An extra title line.
    :param panel_notes: Condition -> one suffix per weighting for its panel titles.
    :param ylim: Symmetric y-limit ``(-ylim, ylim)`` of the curve rows, so figures
        drawn with one value compare; ``None`` takes this figure's largest curve.
    :param diff_ylim: The same for the difference row; ``None`` takes its largest
        mean ± SEM.
    :param reference_index: The reference weighting, whose column is set apart.
    :param stimulus_end: Time of the stimulus offset (s), marked dotted beside
        the onset; ``None`` marks only the onset.
    :param save_path: Optional output path.
    :return: The figure.
    """
    conditions = list(curves)
    n_weights = len(weight_labels)
    diff = paired_difference(curves)
    n_rows = len(conditions) + (2 if diff is not None else 0)
    fig, axes = plt.subplots(
        n_rows,
        n_weights,
        figsize=(3.6 * n_weights, 2.6 * n_rows),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    for row, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, "C0")
        data = curves[condition]
        for w in range(n_weights):
            ax = axes[row, w]
            ax.plot(times, data[:, w].T, color=color, lw=0.7, alpha=0.35)
            ax.plot(times, data[:, w].mean(axis=0), color="black", lw=2.0)
            _onset_lines(ax, times, stimulus_end)
            extra = f", {panel_notes[condition][w]}" if panel_notes else ""
            ax.set_title(
                f"{weight_labels[w]} — {condition} (n={data.shape[0]}{extra})",
                fontsize=8,
            )
    if diff is not None:
        diff_label, d = diff
        for w in range(n_weights):
            ax = axes[-2, w]
            for condition in conditions:
                data = curves[condition][:, w]
                mean, sem = data.mean(axis=0), _sem(data)
                color = CONDITION_COLORS.get(condition, None)
                ax.plot(times, mean, color=color, lw=1.8, label=condition)
                ax.fill_between(
                    times, mean - sem, mean + sem, color=color, alpha=0.2, lw=0
                )
            _onset_lines(ax, times, stimulus_end)
            ax.set_title(f"{weight_labels[w]} — group mean ± SEM", fontsize=8)

            ax = axes[-1, w]
            mean, sem = d[:, w].mean(axis=0), _sem(d[:, w])
            ax.plot(times, mean, color="black", lw=1.8)
            ax.fill_between(times, mean - sem, mean + sem, color="0.5", alpha=0.3, lw=0)
            _onset_lines(ax, times, stimulus_end)
            ax.set_title(f"{weight_labels[w]} — {diff_label} (paired)", fontsize=8)
        axes[-2, 0].legend(fontsize=7, loc="upper right")
    if reference_index is not None:
        for ax in axes[:, reference_index]:
            ax.set_facecolor(REFERENCE_FACE)
            ax.set_title(f"{ax.get_title()} [ref]", fontsize=8)
    curve_rows = list(range(n_rows - (1 if diff is not None else 0)))
    _set_rows_ylim(axes, curve_rows, ylim or 1.05 * onset_curve_extent(curves))
    for row in curve_rows:
        axes[row, 0].set_ylabel(ylabel, fontsize=8)
    if diff is not None:
        _set_rows_ylim(axes, [n_rows - 1], diff_ylim or 1.05 * _mean_sem_extent(d))
        axes[-1, 0].set_ylabel(f"Δ {ylabel}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time relative to onset (s)")
    full_title = f"{title} — {label}"
    if note:
        full_title += f"\n{note}"
    fig.suptitle(full_title, fontsize=10)
    save_fig(fig, save_path)
    return fig


def plot_participant_weighted_curves(
    curves: Mapping[str, np.ndarray],
    times: np.ndarray,
    participants: Sequence[str],
    weight_index: int,
    *,
    weight_label: str,
    label: str,
    title: str,
    ylabel: str,
    ylim: Optional[float] = None,
    reference: bool = False,
    stimulus_end: Optional[float] = None,
    save_path: Optional[Path] = None,
):
    """One panel per participant: its curve in every condition, the conditions'
    group means dashed behind it.

    :param curves: Condition -> ``(participants, weightings, times)``, rows in
        *participants* order.
    :param times: ``(times,)`` epoch time base in seconds.
    :param participants: Participant IDs, one panel each.
    :param weight_index: Which weighting to draw.
    :param weight_label: Its name, for the title.
    :param label: Dataset label shown in the title.
    :param title: What the curves are, the title's first part.
    :param ylabel: Y-axis label.
    :param ylim: Symmetric y-limit of every panel; ``None`` leaves it to matplotlib.
    :param reference: The weighting is the reference: every panel is set apart.
    :param stimulus_end: Time of the stimulus offset (s), marked dotted beside
        the onset; ``None`` marks only the onset.
    :param save_path: Optional output path.
    :return: The figure.
    """
    n_rows, n_cols = _grid_shape(len(participants))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax in axes.flat[len(participants) :]:
        ax.set_visible(False)
    for i, (ax, participant) in enumerate(zip(axes.flat, participants)):
        for condition, data in curves.items():
            color = CONDITION_COLORS.get(condition, None)
            ax.plot(
                times,
                data[:, weight_index].mean(axis=0),
                color=color,
                lw=0.9,
                ls="--",
                alpha=0.5,
            )
            ax.plot(times, data[i, weight_index], color=color, lw=1.4, label=condition)
        _onset_lines(ax, times, stimulus_end)
        ax.set_title(f"PSI{participant}", fontsize=8)
        if reference:
            ax.set_facecolor(REFERENCE_FACE)
    if ylim is not None:
        axes[0, 0].set_ylim(-ylim, ylim)
    axes.flat[0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel(ylabel)
    fig.suptitle(
        f"{title}, {weight_label}{' [reference]' if reference else ''} — {label}\n"
        "solid = participant, dashed = group mean",
        fontsize=10,
    )
    save_fig(fig, save_path)
    return fig


def weighting_differences(
    curves: Mapping[str, np.ndarray], reference_index: int
) -> dict[str, np.ndarray]:
    """Condition -> ``(participants, others, times)``: every non-reference weighting
    minus the reference, participant by participant."""
    others = [
        k for k in range(next(iter(curves.values())).shape[1]) if k != reference_index
    ]
    return {
        condition: data[:, others] - data[:, [reference_index]]
        for condition, data in curves.items()
    }


def difference_extent(curves: Mapping[str, np.ndarray], reference_index: int) -> float:
    """Largest ``|mean| + SEM`` that :func:`plot_difference_comparison` draws."""
    rows = list(weighting_differences(curves, reference_index).values())
    diff = paired_difference(curves)
    if diff is not None:
        rows.append(diff[1])
    return max(_mean_sem_extent(d) for d in rows)


#: Line colours of the component weightings in an overlay, clear of the condition
#: colours (Okabe-Ito green, purple, yellow, sky blue).
WEIGHTING_COLORS = ("#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def plot_weighting_overlay(
    curves: Mapping[str, np.ndarray],
    times: np.ndarray,
    weight_labels: Sequence[str],
    reference_index: int,
    *,
    label: str,
    title: str,
    ylabel: str,
    note: Optional[str] = None,
    ylim: Optional[float] = None,
    stimulus_end: Optional[float] = None,
    save_path: Optional[Path] = None,
):
    """The reference weighting's group mean overlaid with each other weighting's.

    One row per condition; one column per non-reference weighting (it against the
    reference), then one with every weighting together. The differences are
    :func:`plot_difference_comparison`'s. Lines are group means, bands ± SEM; every
    panel shares one y-axis.

    :param curves: Condition -> ``(participants, weightings, times)``.
    :param times: ``(times,)`` epoch time base in seconds.
    :param weight_labels: One name per weighting, in the curves' order.
    :param reference_index: The weighting drawn in every panel (the uniform one).
    :param label: Dataset label shown in the title.
    :param title: What the curves are, the title's first part.
    :param ylabel: Y-axis label.
    :param note: An extra title line.
    :param ylim: Symmetric y-limit of every panel; ``None`` leaves it to matplotlib.
    :param stimulus_end: Time of the stimulus offset (s), marked dotted beside
        the onset; ``None`` marks only the onset.
    :param save_path: Optional output path.
    :return: The figure.
    """
    others = [k for k in range(len(weight_labels)) if k != reference_index]
    colors = {
        k: WEIGHTING_COLORS[i % len(WEIGHTING_COLORS)] for i, k in enumerate(others)
    }
    colors[reference_index] = REFERENCE_COLOR
    rows = dict(curves)
    columns = [[reference_index, k] for k in others] + [[reference_index, *others]]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.6 * len(columns), 2.6 * len(rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for r, (row_label, data) in enumerate(rows.items()):
        for c, shown in enumerate(columns):
            ax = axes[r, c]
            for k in shown:
                _mean_band(ax, times, data[:, k], colors[k], weight_labels[k])
            _onset_lines(ax, times, stimulus_end)
            what = "all" if len(shown) > 2 else weight_labels[shown[1]]
            ax.set_title(f"{row_label} — {what}", fontsize=8)
    for c, shown in enumerate(columns):
        axes[0, c].legend(fontsize=6, loc="upper right")
    if ylim is not None:
        axes[0, 0].set_ylim(-ylim, ylim)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time relative to onset (s)")
    full_title = f"{title}: {weight_labels[reference_index]} vs components — {label}"
    if note:
        full_title += f"\n{note}"
    fig.suptitle(full_title, fontsize=10)
    save_fig(fig, save_path)
    return fig


def plot_difference_comparison(
    curves: Mapping[str, np.ndarray],
    times: np.ndarray,
    weight_labels: Sequence[str],
    reference_index: int,
    *,
    label: str,
    title: str,
    ylabel: str,
    note: Optional[str] = None,
    ylim: Optional[float] = None,
    stimulus_end: Optional[float] = None,
    save_path: Optional[Path] = None,
):
    """The two differences: between the conditions, and from the reference weighting.

    With two conditions, a first row of their paired difference (second − first): each
    non-reference weighting's against the reference's. Then one row per condition of
    every non-reference weighting minus the reference, participant by participant.
    One column per non-reference weighting, then one with all of them. Lines are
    means over participants, bands ± SEM; every panel shares one y-axis.

    :param curves: Condition -> ``(participants, weightings, times)``; with two
        conditions, rows are the same participants in the same order.
    :param times: ``(times,)`` epoch time base in seconds.
    :param weight_labels: One name per weighting, in the curves' order.
    :param reference_index: The weighting the others are compared with (uniform).
    :param label: Dataset label shown in the title.
    :param title: What the curves are, the title's first part.
    :param ylabel: Y-axis label.
    :param note: An extra title line.
    :param ylim: Symmetric y-limit of every panel, the difference scale; ``None``
        leaves it to matplotlib.
    :param stimulus_end: Time of the stimulus offset (s), marked dotted beside
        the onset; ``None`` marks only the onset.
    :param save_path: Optional output path.
    :return: The figure.
    """
    n_weights = len(weight_labels)
    others = [k for k in range(n_weights) if k != reference_index]
    colors = {
        k: WEIGHTING_COLORS[i % len(WEIGHTING_COLORS)] for i, k in enumerate(others)
    }
    colors[reference_index] = REFERENCE_COLOR
    reference = weight_labels[reference_index]
    from_reference = weighting_differences(curves, reference_index)
    diff = paired_difference(curves)

    # Each row: (title with a {} for the column, data (P, n_weights or n_others, T),
    # {column: [(index, colour, legend)]}); the last column holds every line of it.
    rows = []
    if diff is not None:
        diff_label, d = diff
        lines = {
            c: [
                (reference_index, REFERENCE_COLOR, reference),
                (k, colors[k], weight_labels[k]),
            ]
            for c, k in enumerate(others)
        }
        lines[len(others)] = [(reference_index, REFERENCE_COLOR, reference)] + [
            (k, colors[k], weight_labels[k]) for k in others
        ]
        rows.append((f"{diff_label} (paired) — {{}}", d, lines))
    for condition, d in from_reference.items():
        lines = {
            c: [(c, colors[k], f"{weight_labels[k]} − uniform")]
            for c, k in enumerate(others)
        }
        lines[len(others)] = [
            (c, colors[k], f"{weight_labels[k]} − uniform")
            for c, k in enumerate(others)
        ]
        rows.append((f"{condition}: {{}} − uniform", d, lines))

    n_cols = len(others) + 1
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(3.6 * n_cols, 2.6 * len(rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for r, (row_title, data, lines) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            for index, color, legend in lines[c]:
                _mean_band(ax, times, data[:, index], color, legend)
            _onset_lines(ax, times, stimulus_end)
            what = "all" if c == len(others) else weight_labels[others[c]]
            ax.set_title(row_title.format(what), fontsize=8)
            if c == n_cols - 1 or r == 0:
                ax.legend(fontsize=6, loc="upper right")
    if ylim is not None:
        axes[0, 0].set_ylim(-ylim, ylim)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Δ {ylabel}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time relative to onset (s)")
    full_title = f"{title}: condition and weighting differences — {label}"
    if note:
        full_title += f"\n{note}"
    fig.suptitle(full_title, fontsize=10)
    save_fig(fig, save_path)
    return fig
