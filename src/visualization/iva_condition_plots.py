"""
Placebo/Psilocybin comparison figures for a subject-axis-pooled IVA decomposition.

These are the read-out of a
:attr:`~src.definitions.fields.ConditionVariants.JOINED` run: one aligned set of IVA
components, each recording carrying its own channel topography and time-frequency map,
with the subject axis holding every participant once per condition. The question the
figures answer is "does this component look different under Psilocybin", so every one of
them puts the two conditions on **one shared colour scale** — a comparison drawn on two
independent scales is not a comparison.

Two views per quantity, matching how
:mod:`src.visualization.iva_quality_plots` splits its diagnostics:

* **Condition means** — one grid, rows = conditions (plus their difference), columns =
  components. The overview: which components differ at all.
* **Per participant** — one figure per component, Placebo on the first row and
  Psilocybin on the second, one column per participant, so a participant sits directly
  above their own other-condition panel. This is what says whether a difference in the
  means is shared across the group or carried by one or two people.

Both views exist for channel topographies and for time-frequency maps.

**Sign alignment is a precondition, not a detail.** IVA fixes each component's sign only
per recording, so unaligned maps average toward zero and — because the flips fall
arbitrarily across the two condition blocks — invent condition differences. Pass maps
oriented by :func:`~src.analysis.iva_condition_comparison.align_tf_pc1_signs` and name
what did it via *alignment_note*, which every figure prints.

**Participants are always ordered by ID**, never by any score, so a participant keeps the
same column in every figure and the two condition rows stay aligned with each other.

**Every participant is put on a common scale first**
(:func:`~src.analysis.iva_quality.equalize_subject_influence`): the per-recording gain
IVA leaves behind is a nuisance, and without removing it the loudest few participants set
the colour limit and dominate both condition means, which is exactly the comparison being
made.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.iva_condition_comparison import (
    condition_component_means,
    condition_difference,
)
from src.analysis.iva_quality import equalize_subject_influence
from src.visualization.iva_quality_plots import (
    participant_sort_key,
    safe_vlim,
    save_fig,
    topo_info_subset,
)

_logger = logging.getLogger(__name__)

#: Row label for the between-condition difference row of the mean grids.
DIFFERENCE_ROW = "difference"

#: Cap on stimulus-onset markers drawn on a TF panel. Above it the lines merge into a
#: wash that hides the map they were meant to locate things on. Matches
#: :data:`src.visualization.iva_quality_plots.MAX_ONSET_MARKS`.
MAX_TIME_MARKS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_pooled(
    values: np.ndarray,
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
    *,
    expected_ndim: int,
) -> None:
    """Check a pooled array against its per-recording bookkeeping.

    :param values: ``(S, K, ...)`` pooled array.
    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions to draw.
    :param expected_ndim: Required ``values.ndim`` (3 for patterns, 4 for TF maps).
    :raises ValueError: If the shapes or the bookkeeping disagree.
    """
    if values.ndim != expected_ndim:
        raise ValueError(
            f"values must have {expected_ndim} axes (S, K, ...); got shape "
            f"{values.shape}."
        )
    n_subjects = values.shape[0]
    if len(subject_participants) != n_subjects:
        raise ValueError(
            f"subject_participants has {len(subject_participants)} entries but values "
            f"has {n_subjects} recording(s)."
        )
    if len(subject_conditions) != n_subjects:
        raise ValueError(
            f"subject_conditions has {len(subject_conditions)} entries but values has "
            f"{n_subjects} recording(s)."
        )
    if len(conditions) < 1:
        raise ValueError("At least one condition is needed.")
    missing = [c for c in conditions if c not in set(subject_conditions)]
    if missing:
        raise ValueError(
            f"Condition(s) {missing} have no recordings; present: "
            f"{sorted(set(subject_conditions))}."
        )

    # A participant contributes at most one recording per condition, so a duplicate
    # within a condition is always a mistake — and a silent one in the condition means,
    # which would simply weigh that participant twice. It is the mistake a time-axis
    # join invites: its channel patterns are shared, so they belong in a ONE-row grid
    # with one entry per participant, and handing a subject-axis 2P array to that grid
    # would land every participant in the single row twice.
    for condition in conditions:
        rows = [
            participant
            for participant, row_condition in zip(
                subject_participants, subject_conditions
            )
            if row_condition == condition
        ]
        duplicates = sorted({p for p in rows if rows.count(p) > 1})
        if duplicates:
            raise ValueError(
                f"Condition {condition!r} has more than one recording for "
                f"participant(s) {duplicates}; a participant contributes at most one "
                "recording per condition."
            )


def participant_grid(
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Lay the recordings out as a condition × participant grid of row indices.

    The column order is by participant ID (:func:`participant_sort_key`), so column *j*
    is the same participant in every condition row and stays put across figures.

    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions, in the row order wanted.
    :return: ``(participants, grid)`` where *participants* are the sorted column labels
        and ``grid[i, j]`` is the recording index for condition *i* and participant *j*,
        or ``-1`` when that participant has no recording in that condition.
    :raises ValueError: If a participant has more than one recording in a condition,
        which would make the column ambiguous.
    """
    participants = sorted(set(subject_participants), key=participant_sort_key)
    column = {participant: j for j, participant in enumerate(participants)}
    row = {condition: i for i, condition in enumerate(conditions)}

    grid = np.full((len(conditions), len(participants)), -1, dtype=int)
    for index, (participant, condition) in enumerate(
        zip(subject_participants, subject_conditions)
    ):
        if condition not in row:
            continue
        i, j = row[condition], column[participant]
        if grid[i, j] != -1:
            raise ValueError(
                f"Participant {participant} has more than one {condition} recording "
                f"(rows {grid[i, j]} and {index}); the grid column would be ambiguous."
            )
        grid[i, j] = index
    return participants, grid


def _column_limits(
    means: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    comp_indices: Sequence[int],
) -> dict[int, float]:
    """Per-component colour limit shared by every condition row.

    One limit per component, taken over all the conditions at once, is what makes the
    rows comparable. Limits are deliberately **not** shared across components: IVA fixes
    each component's scale independently, so a common limit would just render the
    weaker components flat.

    :param means: Condition name → ``(K, ...)`` mean.
    :param conditions: Conditions contributing to the limit.
    :param comp_indices: Components to measure.
    :return: Component index → symmetric half-width.
    """
    return {
        k: safe_vlim(np.stack([means[c][k] for c in conditions])) for k in comp_indices
    }


def _scale_note(n_conditions: int) -> str:
    """The suptitle clause describing the per-column colour limit.

    A single-row grid has no rows to share a limit *with*, so claiming it does would
    describe a comparison that is not being drawn — which is exactly the case a
    time-axis join hits for its channel patterns, where one mixing matrix per
    participant means one topography for both conditions.

    :param n_conditions: Number of condition rows in the grid.
    :return: The clause to put in the suptitle.
    """
    shared = " shared by the condition rows" if n_conditions > 1 else ""
    return f"one column per component, each on its own symmetric scale{shared}"


def _decorate_tf(
    ax,
    *,
    time_marks: Optional[np.ndarray],
    freq_marks: Optional[Sequence[float]],
    epoch_marks: Optional[Sequence[float]] = None,
) -> None:
    """Draw the optional time/frequency reference lines on a TF panel.

    *time_marks* and *epoch_marks* are deliberately different weights, because they
    answer different questions. Many faint lines locate events along a long recording
    without competing with the map; the one or two lines of an epoch's geometry — where
    the stimulus starts and stops — are the axis the panel is read against and have to
    be legible.

    :param ax: Axis to decorate.
    :param time_marks: Times (s) to mark with faint vertical lines, e.g. every stimulus
        onset of a whole-recording map. Dropped entirely above
        :data:`MAX_TIME_MARKS`, where they merge into a wash.
    :param freq_marks: Frequencies (Hz) to mark with horizontal lines, e.g. the ASSR
        stimulation frequency.
    :param epoch_marks: Times (s) to mark prominently, e.g. ``[0.0, 0.5]`` for the
        stimulus onset and offset of an onset-averaged epoch. Not capped: these are
        paradigm geometry, so there are only ever a couple.
    """
    if time_marks is not None and len(time_marks) <= MAX_TIME_MARKS:
        for mark in time_marks:
            ax.axvline(float(mark), color="0.2", lw=0.5, ls="--", alpha=0.3, zorder=3)
    for i, mark in enumerate(epoch_marks or ()):
        # The first mark is the onset itself (dashed); later ones are the window edges
        # it defines (dotted), matching src.visualization.iva_quality_plots.
        ax.axvline(
            float(mark),
            color="k",
            ls="--" if i == 0 else ":",
            lw=0.8,
            zorder=4,
        )
    for mark in freq_marks or ():
        ax.axhline(float(mark), color="green", ls=":", lw=0.9, zorder=3)
    ax.tick_params(labelsize=6)


def _topo_row_label(ax, label: str) -> None:
    """Label a topomap row without touching the axis.

    ``plot_topomap`` turns the axis off and sets its own limits and aspect, so a
    ``set_ylabel`` needs the frame switched back on — which distorts the head geometry
    it just laid out. Anchoring the text to the axis *transform* leaves all of that
    alone.

    :param ax: The row's leftmost topomap axis.
    :param label: Row label, drawn rotated on the left.
    """
    ax.text(
        -0.08,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=9,
        fontweight="bold",
    )


def _tf_extent(freqs: np.ndarray, times: np.ndarray) -> list[float]:
    """``imshow`` extent for a ``(F, T)`` map on the given axes."""
    return [float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])]


# ---------------------------------------------------------------------------
# Condition means — one grid over every component
# ---------------------------------------------------------------------------


def plot_condition_mean_topomaps(
    patterns: np.ndarray,
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
    info,
    n_channels: int,
    comp_indices: Sequence[int],
    *,
    label: str,
    show_difference: bool = True,
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Condition-mean channel topographies: rows = conditions, columns = components.

    The overview figure for the topography side: which components' scalp patterns differ
    between conditions at all. With exactly two conditions a **difference** row is added
    (second minus first), which is the row the eye actually needs — a small difference
    between two similar maps is much easier to see drawn directly than inferred from the
    two panels above it.

    Each **column** is drawn on its own symmetric limit, shared by the condition rows so
    they are comparable, and reported in the column title. Columns are not put on a
    common limit because IVA fixes each component's scale independently; the difference
    row gets its own limit per column for the same reason, since it is far weaker than
    the maps it comes from. There is deliberately no single colourbar: with a limit per
    column it would be meaningless.

    :param patterns: ``(S, K, C)`` sign-oriented forward channel patterns.
    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions in row order, e.g. ``["Placebo", "Psilocybin"]``.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels on the IVA channel axis.
    :param comp_indices: 0-based component indices, one column each.
    :param label: Dataset label shown in the title.
    :param show_difference: Add the difference row when exactly two conditions are
        given. Ignored otherwise.
    :param alignment_note: What resolved the per-recording sign, named in the title.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    :raises ValueError: If the shapes and the bookkeeping disagree.
    """
    from mne.viz import plot_topomap

    patterns = np.asarray(patterns, dtype=float)
    _validate_pooled(
        patterns,
        subject_participants,
        subject_conditions,
        conditions,
        expected_ndim=3,
    )
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None

    conditions = list(conditions)
    # Equal weight per recording, so the condition means are group statements rather
    # than a report on the loudest participants.
    patterns = equalize_subject_influence(patterns, comp_indices)
    means = condition_component_means(patterns, subject_conditions, conditions)
    rows = list(conditions)
    with_difference = show_difference and len(conditions) == 2
    if with_difference:
        means[DIFFERENCE_ROW] = condition_difference(means, conditions)
        rows.append(DIFFERENCE_ROW)

    vlims = _column_limits(means, conditions, comp_indices)
    diff_vlims = (
        {k: safe_vlim(means[DIFFERENCE_ROW][k]) for k in comp_indices}
        if with_difference
        else {}
    )
    topo_info = topo_info_subset(info, n_channels)

    fig, axes = plt.subplots(
        len(rows),
        len(comp_indices),
        figsize=(2.4 * len(comp_indices), 2.9 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    for i, row_name in enumerate(rows):
        is_difference = row_name == DIFFERENCE_ROW
        for j, k in enumerate(comp_indices):
            vlim = diff_vlims[k] if is_difference else vlims[k]
            plot_topomap(
                means[row_name][k],
                topo_info,
                axes=axes[i, j],
                show=False,
                cmap="RdBu_r",
                vlim=(-vlim, vlim),
                contours=4,
            )
            if i == 0:
                axes[i, j].set_title(f"IC {k + 1}\n(|max| {vlims[k]:.3g})", fontsize=8)
            elif is_difference:
                axes[i, j].set_title(f"|max| {vlim:.3g}", fontsize=6)
        row_label = (
            f"{conditions[1]} − {conditions[0]}\n(own scale)"
            if is_difference
            else row_name
        )
        _topo_row_label(axes[i, 0], row_label)

    fig.suptitle(
        f"Condition-mean channel topographies — {label}\n"
        f"{_scale_note(len(conditions))}  |  equal-weighted participants"
        f"{'' if alignment_note is None else f'  |  sign: {alignment_note}'}",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


def plot_condition_mean_tf_maps(
    sources: np.ndarray,
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
    freqs: np.ndarray,
    times: np.ndarray,
    comp_indices: Sequence[int],
    *,
    label: str,
    show_difference: bool = True,
    time_marks: Optional[np.ndarray] = None,
    freq_marks: Optional[Sequence[float]] = None,
    epoch_marks: Optional[Sequence[float]] = None,
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Condition-mean component TF maps: rows = conditions, columns = components.

    The time-frequency counterpart of :func:`plot_condition_mean_topomaps`, with the
    same scaling rules and for the same reasons: one symmetric limit per column shared
    by the condition rows, no common limit across columns, and the difference row on its
    own limit per column.

    :param sources: ``(S, K, F, T)`` sign-oriented component TF maps.
    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions in row order.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param times: ``(T,)`` time axis in seconds.
    :param comp_indices: 0-based component indices, one column each.
    :param label: Dataset label shown in the title.
    :param show_difference: Add the difference row when exactly two conditions are
        given.
    :param time_marks: Faint times (s) to mark, e.g. every stimulus onset of a
        whole-recording map.
    :param freq_marks: Frequencies (Hz) to mark, e.g. the ASSR stimulation frequency.
    :param epoch_marks: Prominent times (s) to mark, e.g. ``[0.0, 0.5]`` for the
        stimulus onset and offset of an onset-averaged epoch.
    :param alignment_note: What resolved the per-recording sign, named in the title.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    :raises ValueError: If the shapes and the bookkeeping disagree.
    """
    sources = np.asarray(sources, dtype=float)
    _validate_pooled(
        sources, subject_participants, subject_conditions, conditions, expected_ndim=4
    )
    if sources.shape[2] != len(freqs) or sources.shape[3] != len(times):
        raise ValueError(
            f"sources TF axes {sources.shape[2:]} do not match freqs/times "
            f"({len(freqs)}, {len(times)})."
        )
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None

    conditions = list(conditions)
    sources = equalize_subject_influence(sources, comp_indices)
    means = condition_component_means(sources, subject_conditions, conditions)
    rows = list(conditions)
    with_difference = show_difference and len(conditions) == 2
    if with_difference:
        means[DIFFERENCE_ROW] = condition_difference(means, conditions)
        rows.append(DIFFERENCE_ROW)

    vlims = _column_limits(means, conditions, comp_indices)
    diff_vlims = (
        {k: safe_vlim(means[DIFFERENCE_ROW][k]) for k in comp_indices}
        if with_difference
        else {}
    )
    extent = _tf_extent(freqs, times)

    fig, axes = plt.subplots(
        len(rows),
        len(comp_indices),
        figsize=(3.0 * len(comp_indices), 2.6 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    for i, row_name in enumerate(rows):
        is_difference = row_name == DIFFERENCE_ROW
        for j, k in enumerate(comp_indices):
            vlim = diff_vlims[k] if is_difference else vlims[k]
            axes[i, j].imshow(
                means[row_name][k],
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
            )
            _decorate_tf(
                axes[i, j],
                time_marks=time_marks,
                freq_marks=freq_marks,
                epoch_marks=epoch_marks,
            )
            if i == 0:
                axes[i, j].set_title(f"IC {k + 1}\n(|max| {vlims[k]:.3g})", fontsize=8)
            elif is_difference:
                axes[i, j].set_title(f"|max| {vlim:.3g}", fontsize=6)
            if i == len(rows) - 1:
                axes[i, j].set_xlabel("Time (s)", fontsize=7)
        row_label = (
            f"{conditions[1]} − {conditions[0]}\n(own scale)"
            if is_difference
            else row_name
        )
        axes[i, 0].set_ylabel(f"{row_label}\nFrequency (Hz)", fontsize=8)

    fig.suptitle(
        f"Condition-mean component TF maps — {label}\n"
        f"{_scale_note(len(conditions))}  |  equal-weighted participants"
        f"{'' if alignment_note is None else f'  |  sign: {alignment_note}'}",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per participant — one figure per component
# ---------------------------------------------------------------------------


def plot_participant_condition_topomaps(
    patterns: np.ndarray,
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
    info,
    n_channels: int,
    comp_indices: Sequence[int],
    *,
    label: str,
    root_dir: Path,
    prefix: str = "",
    show_condition_mean: bool = True,
    alignment_note: Optional[str] = None,
) -> list[Path]:
    """One figure per component: every participant's topography, condition per row.

    Rows are the conditions in the order given (Placebo first by convention) and columns
    are participants ordered by ID, so a participant's two recordings sit one directly
    above the other and the same participant occupies the same column in every figure.
    That vertical pairing is the point: it turns a condition contrast into a
    within-participant read rather than a comparison of two group averages.

    Every panel of a figure shares **one** symmetric colour limit — across participants
    *and* across conditions — so the whole grid is comparable and the shared colourbar
    means something. A trailing column holds each condition's mean on its own scale:
    averaging cancels the incoherent part of every pattern, so the means read flat under
    the participants' limit, and they are annotated with their own ``|max|``.

    Files are written flat into ``root_dir`` and numbered 1-based in **component order**,
    matching the ``IC <k+1>`` labels used elsewhere.

    :param patterns: ``(S, K, C)`` sign-oriented forward channel patterns.
    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions in row order.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels on the IVA channel axis.
    :param comp_indices: 0-based component indices to emit. Empty emits nothing.
    :param label: Dataset label used in titles and filenames.
    :param root_dir: Directory the figures are written into.
    :param prefix: Optional filename prefix (e.g. ``"alpha_"``).
    :param show_condition_mean: Append the per-condition mean column.
    :param alignment_note: What resolved the per-recording sign, named in the title.
    :return: One path per component, in *comp_indices* order.
    :raises ValueError: If the shapes and the bookkeeping disagree.
    """
    from mne.viz import plot_topomap

    patterns = np.asarray(patterns, dtype=float)
    _validate_pooled(
        patterns,
        subject_participants,
        subject_conditions,
        conditions,
        expected_ndim=3,
    )
    comp_indices = list(comp_indices)
    if not comp_indices:
        return []

    conditions = list(conditions)
    patterns = equalize_subject_influence(patterns, comp_indices)
    participants, grid = participant_grid(
        subject_participants, subject_conditions, conditions
    )
    means = (
        condition_component_means(patterns, subject_conditions, conditions)
        if show_condition_mean
        else {}
    )
    mean_vlim = (
        safe_vlim(np.stack([means[c][comp_indices] for c in conditions]))
        if show_condition_mean
        else 0.0
    )
    topo_info = topo_info_subset(info, n_channels)
    root_dir = Path(root_dir)
    written: list[Path] = []

    ncols = len(participants) + (1 if show_condition_mean else 0)
    for k in comp_indices:
        # One limit for the whole figure: rows must be comparable to each other, which
        # is the comparison the figure exists to make.
        vlim = safe_vlim(patterns[:, k, :])
        fig, axes = plt.subplots(
            len(conditions),
            ncols,
            figsize=(2.3 * ncols, 2.8 * len(conditions)),
            squeeze=False,
            layout="constrained",
        )
        im_shared = None
        for i, condition in enumerate(conditions):
            for j, participant in enumerate(participants):
                ax = axes[i, j]
                s = grid[i, j]
                if s < 0:
                    ax.axis("off")
                    ax.set_title(f"{participant}\n(no recording)", fontsize=7)
                    continue
                im_shared, _ = plot_topomap(
                    patterns[s, k],
                    topo_info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(-vlim, vlim),
                    contours=4,
                )
                if i == 0:
                    ax.set_title(participant, fontsize=8)
            if show_condition_mean:
                ax = axes[i, ncols - 1]
                plot_topomap(
                    means[condition][k],
                    topo_info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(-mean_vlim, mean_vlim),
                    contours=4,
                )
                ax.set_title(
                    f"{'mean' if i else 'condition mean'}\n"
                    f"(own scale, |max| {mean_vlim:.3g})",
                    fontsize=7,
                    fontweight="bold",
                )
            _topo_row_label(axes[i, 0], condition)

        if im_shared is not None:
            fig.colorbar(
                im_shared,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                label="pattern (a.u., per-participant scale)",
            )
        row_note = (
            "row = condition, column = participant (by ID)"
            if len(conditions) > 1
            else f"one row ({conditions[0]}), column = participant (by ID)"
        )
        fig.suptitle(
            f"IC {k + 1} — channel topography per participant — {label}\n"
            f"{row_note}  |  "
            f"equal-weighted participants, shared |pattern| ≤ {vlim:.3g}"
            f"{'' if alignment_note is None else f'  |  sign: {alignment_note}'}",
            fontsize=11,
        )
        path = (
            root_dir
            / f"{prefix}condition_topomap_ic{k + 1:02d}_participants_{label}.png"
        )
        save_fig(fig, path)
        plt.close(fig)
        written.append(path)

    _logger.debug(
        f"Wrote {len(written)} per-participant condition topomap figures to {root_dir}"
    )
    return written


def plot_participant_condition_tf_maps(
    sources: np.ndarray,
    subject_participants: Sequence[str],
    subject_conditions: Sequence[str],
    conditions: Sequence[str],
    freqs: np.ndarray,
    times: np.ndarray,
    comp_indices: Sequence[int],
    *,
    label: str,
    root_dir: Path,
    prefix: str = "",
    show_condition_mean: bool = True,
    time_marks: Optional[np.ndarray] = None,
    freq_marks: Optional[Sequence[float]] = None,
    epoch_marks: Optional[Sequence[float]] = None,
    alignment_note: Optional[str] = None,
) -> list[Path]:
    """One figure per component: every participant's TF map, condition per row.

    The time-frequency counterpart of
    :func:`plot_participant_condition_topomaps`, with the same layout and scaling: rows
    are conditions, columns are participants ordered by ID so each participant's two
    recordings are vertically paired, one shared colour limit across the whole grid, and
    a trailing per-condition mean column on its own scale.

    :param sources: ``(S, K, F, T)`` sign-oriented component TF maps.
    :param subject_participants: Participant label per recording.
    :param subject_conditions: Condition per recording.
    :param conditions: Conditions in row order.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param times: ``(T,)`` time axis in seconds.
    :param comp_indices: 0-based component indices to emit. Empty emits nothing.
    :param label: Dataset label used in titles and filenames.
    :param root_dir: Directory the figures are written into.
    :param prefix: Optional filename prefix (e.g. ``"alpha_"``).
    :param show_condition_mean: Append the per-condition mean column.
    :param time_marks: Faint times (s) to mark, e.g. every stimulus onset of a
        whole-recording map.
    :param freq_marks: Frequencies (Hz) to mark, e.g. the ASSR stimulation frequency.
    :param epoch_marks: Prominent times (s) to mark, e.g. ``[0.0, 0.5]`` for the
        stimulus onset and offset of an onset-averaged epoch.
    :param alignment_note: What resolved the per-recording sign, named in the title.
    :return: One path per component, in *comp_indices* order.
    :raises ValueError: If the shapes and the bookkeeping disagree.
    """
    sources = np.asarray(sources, dtype=float)
    _validate_pooled(
        sources, subject_participants, subject_conditions, conditions, expected_ndim=4
    )
    if sources.shape[2] != len(freqs) or sources.shape[3] != len(times):
        raise ValueError(
            f"sources TF axes {sources.shape[2:]} do not match freqs/times "
            f"({len(freqs)}, {len(times)})."
        )
    comp_indices = list(comp_indices)
    if not comp_indices:
        return []

    conditions = list(conditions)
    sources = equalize_subject_influence(sources, comp_indices)
    participants, grid = participant_grid(
        subject_participants, subject_conditions, conditions
    )
    means = (
        condition_component_means(sources, subject_conditions, conditions)
        if show_condition_mean
        else {}
    )
    mean_vlim = (
        safe_vlim(np.stack([means[c][comp_indices] for c in conditions]))
        if show_condition_mean
        else 0.0
    )
    extent = _tf_extent(freqs, times)
    root_dir = Path(root_dir)
    written: list[Path] = []

    ncols = len(participants) + (1 if show_condition_mean else 0)
    for k in comp_indices:
        vlim = safe_vlim(sources[:, k])
        fig, axes = plt.subplots(
            len(conditions),
            ncols,
            figsize=(2.9 * ncols, 2.5 * len(conditions)),
            squeeze=False,
            layout="constrained",
        )
        im_shared = None
        for i, condition in enumerate(conditions):
            for j, participant in enumerate(participants):
                ax = axes[i, j]
                s = grid[i, j]
                if s < 0:
                    ax.axis("off")
                    ax.set_title(f"{participant}\n(no recording)", fontsize=7)
                    continue
                im_shared = ax.imshow(
                    sources[s, k],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-vlim,
                    vmax=vlim,
                )
                _decorate_tf(
                    ax,
                    time_marks=time_marks,
                    freq_marks=freq_marks,
                    epoch_marks=epoch_marks,
                )
                if i == 0:
                    ax.set_title(participant, fontsize=8)
                if i == len(conditions) - 1:
                    ax.set_xlabel("Time (s)", fontsize=7)
            if show_condition_mean:
                ax = axes[i, ncols - 1]
                ax.imshow(
                    means[condition][k],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-mean_vlim,
                    vmax=mean_vlim,
                )
                _decorate_tf(
                    ax,
                    time_marks=time_marks,
                    freq_marks=freq_marks,
                    epoch_marks=epoch_marks,
                )
                ax.set_title(
                    f"{'mean' if i else 'condition mean'}\n"
                    f"(own scale, |max| {mean_vlim:.3g})",
                    fontsize=7,
                    fontweight="bold",
                )
            axes[i, 0].set_ylabel(f"{condition}\nFrequency (Hz)", fontsize=8)

        if im_shared is not None:
            fig.colorbar(
                im_shared,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                label="source (a.u., per-participant scale)",
            )
        row_note = (
            "row = condition, column = participant (by ID)"
            if len(conditions) > 1
            else f"one row ({conditions[0]}), column = participant (by ID)"
        )
        fig.suptitle(
            f"IC {k + 1} — component TF map per participant — {label}\n"
            f"{row_note}  |  "
            f"equal-weighted participants, shared |source| ≤ {vlim:.3g}"
            f"{'' if alignment_note is None else f'  |  sign: {alignment_note}'}",
            fontsize=11,
        )
        path = root_dir / f"{prefix}condition_tf_ic{k + 1:02d}_participants_{label}.png"
        save_fig(fig, path)
        plt.close(fig)
        written.append(path)

    _logger.debug(
        f"Wrote {len(written)} per-participant condition TF figures to {root_dir}"
    )
    return written
