"""
Figures for the joint-ICA (jICA) stage: components, loadings and the 40 Hz read-out.

The counterpart of :mod:`src.visualization.iva_condition_plots`, and deliberately a
separate module rather than an extension of it, because jICA produces a different
shape of result and reusing those functions would misrepresent it:

* **One TF map per component, not one per recording.** There is no subject axis to
  average over, so :func:`plot_global_tf_grid` draws the maps as they are.
  ``plot_condition_mean_tf_maps`` starts by equalising per-recording influence,
  which applied to a single map per row would rescale each row by its own amplitude
  — normalising away exactly the between-row difference a grid is drawn to show.
* **The loading is a first-class result.** A component's mixing column splits into
  per-recording blocks whose size says how strongly that recording expresses it, so
  :func:`plot_loading_bars` draws it directly rather than leaving it implicit in a
  colour limit.

The per-recording topographies *are* the same shape IVA produces, so those keep
using ``plot_condition_mean_topomaps`` / ``plot_participant_condition_topomaps``
rather than being duplicated here.

Every function takes an optional ``save_path`` and returns the figure (or ``None``
when there is nothing to draw), and none of them mutates its inputs.

**Two conventions run through all of them.** Colour limits are per **column**
(component), shared by the rows that are being compared and never across columns —
ICA fixes each component's scale independently, so a common limit would render the
weaker ones flat. And every spread is measured **across participants**, never across
trials: trials within a participant are correlated, so a band drawn from them would
look tight while saying nothing about how well an effect generalises to a new
person.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.assr_trials import BINARY_FILTER_LABEL
from src.analysis.wavelet_jica import bootstrap_median_ci, participant_spread
from src.definitions.constants import AssrEpoch
from src.visualization.iva_quality_plots import (
    participant_sort_key,
    safe_vlim,
    save_fig,
)

_logger = logging.getLogger(__name__)

#: Row label for a between-condition difference row.
DIFFERENCE_ROW = "difference"

#: Cap on faint time markers drawn on a TF panel. Above it the lines merge into a
#: wash that hides the map they were meant to locate things on.
MAX_TIME_MARKS = 100

#: Colour of a significant marker / p-value annotation.
SIGNIFICANT_COLOR = "#1B5E20"

#: Colour of a non-significant one.
NEUTRAL_COLOR = "0.45"


# ---------------------------------------------------------------------------
# The decomposition: component maps and loadings
# ---------------------------------------------------------------------------


def plot_global_tf_grid(
    maps_by_row: Mapping[str, np.ndarray],
    rows: Sequence[str],
    comp_indices: Sequence[int],
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    label: str,
    title: str,
    time_marks: Optional[np.ndarray] = None,
    freq_marks: Optional[Sequence[float]] = None,
    epoch_marks: Optional[Sequence[float]] = None,
    sign_note: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Grid of the global component TF maps: rows = views, columns = components.

    Scaling rules, matching ``plot_condition_mean_tf_maps`` so the two stages' figures
    read the same way: one symmetric limit per **column**, shared by every row that is
    not the difference row (a comparison drawn on two independent scales is not a
    comparison), no limit shared across columns, and the difference row on its own
    limit per column since it is far weaker than the maps it comes from. There is
    deliberately no colourbar: with a limit per column it would be meaningless.

    :param maps_by_row: Row label → ``(K, F, T)`` global maps.
    :param rows: Row labels in the order wanted. A row named :data:`DIFFERENCE_ROW`
        gets its own per-column limit.
    :param comp_indices: 0-based component indices, one column each.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param times: ``(T,)`` time axis in seconds.
    :param label: Dataset label shown in the title.
    :param title: Figure title prefix.
    :param time_marks: Faint times (s) to mark, e.g. every stimulus onset. Dropped
        entirely above :data:`MAX_TIME_MARKS`.
    :param freq_marks: Frequencies (Hz) to mark, e.g. the ASSR stimulation frequency.
    :param epoch_marks: Prominent times (s), e.g. ``[0.0, 0.5]`` for an epoch's onset
        and stimulus offset.
    :param sign_note: What pinned the component sign, named in the title.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    :raises ValueError: If a row is missing, or a map's axes do not match
        *freqs* / *times*.
    """
    rows = list(rows)
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    if not rows:
        raise ValueError("At least one row is needed.")
    for row in rows:
        if row not in maps_by_row:
            raise ValueError(f"No maps supplied for row {row!r}.")
        array = np.asarray(maps_by_row[row])
        if array.ndim != 3:
            raise ValueError(f"Row {row!r} must be (K, F, T); got shape {array.shape}.")
        if array.shape[1] != len(freqs) or array.shape[2] != len(times):
            raise ValueError(
                f"Row {row!r} has TF axes {array.shape[1:]} but freqs/times are "
                f"({len(freqs)}, {len(times)})."
            )

    base_rows = [row for row in rows if row != DIFFERENCE_ROW] or rows
    vlims = {
        k: safe_vlim(np.stack([np.asarray(maps_by_row[row])[k] for row in base_rows]))
        for k in comp_indices
    }
    diff_vlims = (
        {k: safe_vlim(np.asarray(maps_by_row[DIFFERENCE_ROW])[k]) for k in comp_indices}
        if DIFFERENCE_ROW in rows
        else {}
    )
    extent = [float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])]

    fig, axes = plt.subplots(
        len(rows),
        len(comp_indices),
        figsize=(3.0 * len(comp_indices), 2.6 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    for i, row in enumerate(rows):
        is_difference = row == DIFFERENCE_ROW
        for j, k in enumerate(comp_indices):
            vlim = diff_vlims[k] if is_difference else vlims[k]
            ax = axes[i, j]
            ax.imshow(
                np.asarray(maps_by_row[row])[k],
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
                ax.set_title(f"IC {k + 1}\n(|max| {vlims[k]:.3g})", fontsize=8)
            elif is_difference:
                ax.set_title(f"|max| {vlim:.3g}", fontsize=6)
            if i == len(rows) - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
        axes[i, 0].set_ylabel(f"{row}\nFrequency (Hz)", fontsize=8)

    shared = " shared by the rows" if len(base_rows) > 1 else ""
    fig.suptitle(
        f"{title} — {label}\none column per component, each on its own symmetric "
        f"scale{shared}"
        f"{'' if sign_note is None else f'  |  sign: {sign_note}'}",
        fontsize=11,
    )
    save_fig(fig, save_path)
    return fig


def plot_loading_bars(
    values: np.ndarray,
    row_participants: Sequence[str],
    row_conditions: Sequence[str],
    conditions: Sequence[str],
    comp_indices: Sequence[int],
    *,
    label: str,
    title: str,
    ylabel: str,
    colors: Optional[Mapping[str, str]] = None,
    save_path: Optional[Path] = None,
):
    """Per-recording component loading as grouped bars, one panel per component.

    Columns are ordered by participant ID, so a participant keeps the same position
    in every panel and across figures. The y axis is **shared** across panels: the
    loadings are in one unit for the whole decomposition, so the panels are directly
    comparable and the relative size of the components is part of the result.

    A bar is a *magnitude*: a recording whose topography is inverted relative to the
    group still gets a tall bar, so read this alongside the topographies.

    :param values: ``(S, K)`` loading, one number per (recording, component).
    :param row_participants: Participant label per recording.
    :param row_conditions: Condition per recording. Pass one shared label for every
        row when the loading is shared by the conditions.
    :param conditions: Bar groups within a participant, in order.
    :param comp_indices: 0-based component indices, one panel each.
    :param label: Dataset label shown in the title.
    :param title: Figure title prefix.
    :param ylabel: Y-axis label naming the loading measure.
    :param colors: Optional condition → colour mapping.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *comp_indices* is empty.
    :raises ValueError: If the bookkeeping does not match *values*.
    """
    values = np.asarray(values, dtype=float)
    conditions = list(conditions)
    comp_indices = list(comp_indices)
    if not comp_indices:
        return None
    if values.ndim != 2:
        raise ValueError(f"values must be (S, K); got shape {values.shape}.")
    if len(row_participants) != values.shape[0]:
        raise ValueError(
            f"row_participants has {len(row_participants)} entry/entries but values "
            f"has {values.shape[0]} row(s)."
        )
    if len(row_conditions) != values.shape[0]:
        raise ValueError(
            f"row_conditions has {len(row_conditions)} entry/entries but values has "
            f"{values.shape[0]} row(s)."
        )
    if not conditions:
        raise ValueError("At least one condition group is needed.")

    bar_participants = sorted(set(row_participants), key=participant_sort_key)
    row_of = {
        (participant, condition): s
        for s, (participant, condition) in enumerate(
            zip(row_participants, row_conditions)
        )
    }
    x = np.arange(len(bar_participants), dtype=float)
    width = 0.8 / len(conditions)

    ncols = min(3, len(comp_indices))
    nrows = int(np.ceil(len(comp_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            min(7.0, max(4.0, 0.5 * len(bar_participants) * len(conditions))) * ncols,
            3.0 * nrows,
        ),
        squeeze=False,
        sharey=True,
    )
    for panel, k in enumerate(comp_indices):
        ax = axes[panel // ncols][panel % ncols]
        for c, condition in enumerate(conditions):
            heights = [
                values[row_of[(participant, condition)], k]
                if (participant, condition) in row_of
                else np.nan
                for participant in bar_participants
            ]
            ax.bar(
                x + (c - (len(conditions) - 1) / 2) * width,
                heights,
                width,
                label=condition if panel == 0 else None,
                color=None if colors is None else colors.get(condition),
                edgecolor="none",
            )
        ax.set_title(f"IC {k + 1}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_participants, rotation=60, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if panel % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    for panel in range(len(comp_indices), nrows * ncols):
        axes[panel // ncols][panel % ncols].axis("off")

    if len(conditions) > 1:
        fig.legend(loc="upper right", fontsize=8, title="Condition")
    fig.suptitle(f"{title} — {label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# The 40 Hz read-out: the response over time, and the tests
# ---------------------------------------------------------------------------


def plot_response_courses(
    course_by_condition: Mapping[str, np.ndarray],
    epoch_times: np.ndarray,
    labels: Sequence[str],
    conditions: Sequence[str],
    contrast_rows: Sequence[Mapping],
    *,
    label: str,
    units: str,
    selection: str,
    condition_colors: Optional[Mapping[str, str]] = None,
    spread_mode: str = "sem",
    stimulus_duration: float = AssrEpoch.STIMULUS_DURATION_S,
    test_interval: Optional[tuple[float, float]] = None,
    reference_label: str = BINARY_FILTER_LABEL,
    alpha: float = 0.05,
    note: str = "",
    save_path: Optional[Path] = None,
):
    """The stimulus-locked response per spatial filter, both conditions overlaid.

    The stage's headline figure. One panel per filter; the line is the mean across
    participants of their own median-over-trials time course and the band is the
    spread **across participants** (see :func:`participant_spread`). The driven
    interval is shaded and each panel carries its *p* from the condition contrast, so
    the picture and the test are never read apart.

    The component panels must be passed **polarity-aligned**, the same flip the tests
    use: without it a participant whose component loads negatively on the reference
    electrodes would cancel one who loads positively, and the group mean would
    collapse toward zero for reasons that have nothing to do with the response. The
    reference panel is titled in bold — it is the assumption-free row, and the one to
    read first.

    :param course_by_condition: Condition name → ``(P, S, W)`` per-participant
        median-over-trials courses, polarity-aligned.
    :param epoch_times: ``(W,)`` epoch time in seconds, ``0`` at the onset.
    :param labels: Source label per column of the course arrays.
    :param conditions: Conditions to overlay, in legend order.
    :param contrast_rows: Records from
        :func:`~src.analysis.wavelet_jica.condition_contrast_tests`, one per source.
    :param label: Dataset label shown in the title.
    :param units: What the y axis is in, e.g. ``"pre-stimulus SD"``.
    :param selection: Frequency-selection label, e.g. ``"40hz"``.
    :param condition_colors: Optional condition → colour mapping.
    :param spread_mode: ``"sem"`` or ``"iqr"``; see :func:`participant_spread`.
    :param stimulus_duration: Length of the shaded driven interval, in seconds.
    :param test_interval: Optional custom ``(start, stop)`` window the tests read,
        shaded in gold alongside the paradigm interval.
    :param reference_label: Which source label is the fixed reference.
    :param alpha: Significance level deciding how a *p* annotation is drawn.
    :param note: Extra clause appended to the suptitle.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *labels* is empty.
    :raises ValueError: If a condition is missing or the shapes disagree.
    """
    labels = list(labels)
    conditions = list(conditions)
    if not labels:
        return None
    for condition in conditions:
        if condition not in course_by_condition:
            raise ValueError(f"No course array for condition {condition!r}.")
        array = np.asarray(course_by_condition[condition])
        if array.ndim != 3:
            raise ValueError(
                f"Condition {condition!r} must be (P, S, W); got shape {array.shape}."
            )
        if array.shape[1] != len(labels) or array.shape[2] != len(epoch_times):
            raise ValueError(
                f"Condition {condition!r} has axes {array.shape[1:]} but labels/times "
                f"are ({len(labels)}, {len(epoch_times)})."
            )
    by_source = {row["source"]: row for row in contrast_rows}

    n_src = len(labels)
    ncols = min(3, n_src)
    nrows = int(np.ceil(n_src / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 3.7 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axflat = axes.flat
    for s, source in enumerate(labels):
        ax = axflat[s]
        ax.axvspan(0.0, stimulus_duration, color="0.55", alpha=0.11, lw=0, zorder=0)
        if test_interval is not None:
            ax.axvspan(*test_interval, color="#B8860B", alpha=0.10, lw=0, zorder=0)
        ax.axhline(0.0, color="0.45", lw=0.8, zorder=1)
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
        for condition in conditions:
            colour = (
                None if condition_colors is None else condition_colors.get(condition)
            )
            centre, low, high = participant_spread(
                np.asarray(course_by_condition[condition])[:, s], spread_mode
            )
            ax.fill_between(
                epoch_times, low, high, color=colour, alpha=0.18, lw=0, zorder=2
            )
            ax.plot(
                epoch_times, centre, color=colour, lw=1.9, zorder=3, label=condition
            )

        is_reference = source == reference_label
        ax.set_title(
            f"{source}{'  (reference)' if is_reference else ''}",
            fontsize=12,
            fontweight="bold" if is_reference else "normal",
            loc="left",
        )
        row = by_source.get(source)
        if row is not None:
            significant = row["p"] <= alpha
            ax.text(
                0.985,
                0.955,
                f"{conditions[0]} > {conditions[-1]}\n"
                f"p = {row['p']:.3f}   {row['same sign']}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                family="monospace",
                color="0.25" if not significant else "black",
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    facecolor="white",
                    edgecolor="black" if significant else "0.75",
                    alpha=0.88,
                    lw=1.2 if significant else 0.8,
                ),
            )
        if s + ncols >= n_src:
            ax.set_xlabel("Time from stimulus onset (s)")
        if s % ncols == 0:
            ax.set_ylabel(f"{selection} power\n({units})")
    for j in range(n_src, nrows * ncols):
        axflat[j].axis("off")

    axflat[0].legend(loc="lower right", frameon=True, fontsize=10)
    spread_label = (
        "mean +/- SEM across participants"
        if spread_mode == "sem"
        else "median with 25-75 band across participants"
    )
    fig.suptitle(
        f"{selection} response per spatial filter — {label} — {spread_label}"
        f"{'' if not note else f' — {note}'}",
        y=1.0,
        fontsize=12.5,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_pvalue_summary(
    value_by_condition: Mapping[str, np.ndarray],
    labels: Sequence[str],
    conditions: Sequence[str],
    contrast_rows: Sequence[Mapping],
    versus_rows: Sequence[Mapping],
    *,
    label: str,
    units: str,
    reference_label: str = BINARY_FILTER_LABEL,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 42,
    note: str = "",
    save_path: Optional[Path] = None,
):
    """The two test families as forest plots, with every participant's own value.

    * **Left** — the condition contrast per source, with a bootstrap interval around
      the median and a dot per participant. Positive is the predicted direction, so
      the *p* is the one-sided one from the contrast family.
    * **Right** — each component's contrast against the reference (the interaction),
      two-sided, because nothing predicts which way it should go.

    Intervals are a percentile bootstrap over participants and are there to convey
    spread: the *p*-values come from the exact Wilcoxon test, not from the bootstrap,
    and the two can disagree slightly at small ``P``. Nothing is corrected for
    multiplicity, so read a marker against its neighbours as much as against
    *alpha*.

    :param value_by_condition: Condition name → ``(P, S)`` per-participant response,
        polarity-aligned.
    :param labels: Source label per column.
    :param conditions: Exactly two conditions, in contrast order.
    :param contrast_rows: Records from
        :func:`~src.analysis.wavelet_jica.condition_contrast_tests`.
    :param versus_rows: Records from
        :func:`~src.analysis.wavelet_jica.reference_interaction_tests`.
    :param label: Dataset label shown in the title.
    :param units: Units of the difference axis.
    :param reference_label: Which source label is the fixed reference.
    :param alpha: Significance level.
    :param n_bootstrap: Bootstrap resamples for the intervals.
    :param bootstrap_seed: Seed, so the intervals are reproducible.
    :param note: Extra clause appended to the suptitle.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when *labels* is empty.
    :raises ValueError: If two conditions are not given or a condition is missing.
    """
    labels = list(labels)
    conditions = list(conditions)
    if not labels:
        return None
    if len(conditions) != 2:
        raise ValueError(
            f"A paired contrast needs exactly two conditions; got {conditions}."
        )
    for condition in conditions:
        if condition not in value_by_condition:
            raise ValueError(f"No value array for condition {condition!r}.")
    first = np.asarray(value_by_condition[conditions[0]], dtype=float)
    second = np.asarray(value_by_condition[conditions[1]], dtype=float)
    contrast_by_source = {row["source"]: row for row in contrast_rows}
    versus_by_source = {row["source"]: row for row in versus_rows}
    ic_labels = [s for s in labels if s != reference_label]

    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )

    differences = {
        source: first[:, s] - second[:, s] for s, source in enumerate(labels)
    }
    _forest(
        axes[0],
        labels,
        differences,
        contrast_by_source,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        bootstrap=bootstrap_median_ci,
        diamond=reference_label,
        bold_ticks={reference_label},
    )
    axes[0].set_xlabel(f"{conditions[0]} - {conditions[1]}  ({units})")
    axes[0].set_title(
        f"Is {conditions[1]} lower than {conditions[0]}?\n"
        "one-sided, positive = predicted direction",
        loc="left",
        fontsize=12,
    )

    reference_index = labels.index(reference_label)
    reference_contrast = first[:, reference_index] - second[:, reference_index]
    interaction = {
        source: (first[:, labels.index(source)] - second[:, labels.index(source)])
        - reference_contrast
        for source in ic_labels
    }
    _forest(
        axes[1],
        ic_labels,
        interaction,
        versus_by_source,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        bootstrap=bootstrap_median_ci,
        zero_lw=1.6,
    )
    axes[1].set_xlabel(f"(IC contrast) - ({reference_label} contrast)   ({units})")
    axes[1].set_title(
        f"Better than {reference_label}?\ntwo-sided; 0 = as good as the reference",
        loc="left",
        fontsize=12,
    )

    fig.suptitle(
        f"Exact Wilcoxon signed-rank — {label}{'' if not note else f' — {note}'}",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_snr_vs_reference(
    value_by_variant: Mapping[str, Mapping[str, np.ndarray]],
    rows_by_variant: Mapping[str, Mapping[str, Sequence[Mapping]]],
    labels: Sequence[str],
    conditions: Sequence[str],
    *,
    label: str,
    units_by_variant: Mapping[str, str],
    reference_label: str = BINARY_FILTER_LABEL,
    condition_colors: Optional[Mapping[str, str]] = None,
    alternative: str = "less",
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 42,
    note: str = "",
    save_path: Optional[Path] = None,
):
    """Each component's response minus the reference's, per condition and variant.

    One panel per signal variant; within a panel the conditions are drawn as offset
    rows per component, coloured by condition, a **filled** marker meaning
    ``p <= alpha`` and a hollow one not. A marker left of zero means that component's
    response sits below the reference in that condition.

    :param value_by_variant: Variant → condition → ``(P, S)`` response,
        polarity-aligned.
    :param rows_by_variant: Variant → condition → records from
        :func:`~src.analysis.wavelet_jica.reference_snr_tests`.
    :param labels: Source label per column.
    :param conditions: Conditions to overlay; the first is drawn higher.
    :param label: Dataset label shown in the title.
    :param units_by_variant: Variant → units of its difference axis.
    :param reference_label: Which source label is the fixed reference.
    :param condition_colors: Optional condition → colour mapping.
    :param alternative: Named in the panel titles, e.g. ``"less"``.
    :param alpha: Significance level deciding filled vs hollow.
    :param n_bootstrap: Bootstrap resamples for the intervals.
    :param bootstrap_seed: Seed, so the intervals are reproducible.
    :param note: Extra clause appended to the suptitle.
    :param save_path: Optional output path.
    :return: The figure, or ``None`` when there is no variant or component to draw.
    :raises ValueError: If a variant or condition is missing.
    """
    labels = list(labels)
    conditions = list(conditions)
    variants = list(value_by_variant)
    ic_labels = [s for s in labels if s != reference_label]
    if not variants or not ic_labels:
        return None
    for variant in variants:
        if variant not in rows_by_variant:
            raise ValueError(f"No test records for variant {variant!r}.")
        for condition in conditions:
            if condition not in value_by_variant[variant]:
                raise ValueError(
                    f"Variant {variant!r} has no value array for {condition!r}."
                )
    reference_index = labels.index(reference_label)

    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(7.6, 1.5 + 0.82 * len(ic_labels) * len(variants)),
        squeeze=False,
    )
    offsets = np.linspace(0.2, -0.2, len(conditions))
    for r, variant in enumerate(variants):
        ax = axes[r][0]
        panel, all_d = {}, []
        for condition in conditions:
            value = np.asarray(value_by_variant[variant][condition], dtype=float)
            rows = {row["source"]: row for row in rows_by_variant[variant][condition]}
            diffs = {
                source: value[:, labels.index(source)] - value[:, reference_index]
                for source in ic_labels
            }
            cis = {
                source: bootstrap_median_ci(
                    d, n_bootstrap=n_bootstrap, seed=bootstrap_seed
                )
                for source, d in diffs.items()
            }
            panel[condition] = (rows, diffs, cis)
            all_d.append(np.concatenate(list(diffs.values())))
        edges = []
        for rows, diffs, cis in panel.values():
            edges += [bound for pair in cis.values() for bound in pair]
            edges += [rows[source]["median"] for source in ic_labels if source in rows]
        edges += list(np.nanpercentile(np.concatenate(all_d), [5, 95]))
        span = np.nanmax(edges) - np.nanmin(edges)
        lo, hi = np.nanmin(edges) - 0.16 * span, np.nanmax(edges) + 0.16 * span

        for ci_idx, condition in enumerate(conditions):
            colour = (
                NEUTRAL_COLOR
                if condition_colors is None
                else condition_colors.get(condition, NEUTRAL_COLOR)
            )
            rows, diffs, cis = panel[condition]
            for i, source in enumerate(ic_labels):
                if source not in rows:
                    continue
                y = len(ic_labels) - 1 - i + offsets[ci_idx]
                low, high = cis[source]
                row = rows[source]
                significant = row["p"] <= alpha
                ax.plot(
                    [low, high],
                    [y, y],
                    color=colour,
                    lw=2.2,
                    zorder=3,
                    solid_capstyle="round",
                    alpha=0.9,
                )
                ax.scatter(
                    [row["median"]],
                    [y],
                    s=72,
                    zorder=4,
                    marker="o",
                    color=colour if significant else "white",
                    edgecolor=colour,
                    linewidth=1.6,
                )
                ax.text(
                    0.985,
                    y,
                    f"p={row['p']:.3f}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="right",
                    fontsize=8.5,
                    family="monospace",
                    fontweight="bold" if significant else "normal",
                    color=colour,
                )
        ax.set_xlim(lo, hi)
        ax.axvline(0.0, color="0.3", lw=1.4, zorder=1)
        ax.set_yticks(range(len(ic_labels)))
        ax.set_yticklabels(list(reversed(ic_labels)))
        ax.set_xlabel(
            f"IC response - {reference_label} response   "
            f"({units_by_variant.get(variant, variant)})"
        )
        ax.set_title(
            f"{variant} — left of 0 = component below the reference "
            f"(one-sided {alternative})",
            loc="left",
            fontsize=11,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=(
                NEUTRAL_COLOR
                if condition_colors is None
                else condition_colors.get(condition, NEUTRAL_COLOR)
            ),
            lw=0,
            markersize=8,
            label=condition,
        )
        for condition in conditions
    ]
    handles += [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            markerfacecolor="0.35",
            lw=0,
            markersize=8,
            label=f"filled: p <= {alpha}",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            markerfacecolor="white",
            lw=0,
            markersize=8,
            label="hollow: n.s.",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        f"Is the component's response lower than the {reference_label}'s?  {label}"
        f"{'' if not note else f' — {note}'}",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decorate_tf(
    ax,
    *,
    time_marks: Optional[np.ndarray],
    freq_marks: Optional[Sequence[float]],
    epoch_marks: Optional[Sequence[float]] = None,
) -> None:
    """Draw the optional time/frequency reference lines on a TF panel.

    *time_marks* and *epoch_marks* are deliberately different weights, because they
    answer different questions: many faint lines locate events along a long recording
    without competing with the map, while the one or two lines of an epoch's geometry
    are the axis the panel is read against and have to be legible.
    """
    if time_marks is not None and 0 < len(time_marks) <= MAX_TIME_MARKS:
        for mark in time_marks:
            ax.axvline(float(mark), color="0.2", lw=0.5, ls="--", alpha=0.3, zorder=3)
    for i, mark in enumerate(epoch_marks or ()):
        ax.axvline(float(mark), color="k", ls="--" if i == 0 else ":", lw=0.8, zorder=4)
    for mark in freq_marks or ():
        ax.axhline(float(mark), color="green", ls=":", lw=0.9, zorder=3)
    ax.tick_params(labelsize=6)


def _forest(
    ax,
    order: Sequence[str],
    differences: Mapping[str, np.ndarray],
    rows: Mapping[str, Mapping],
    *,
    alpha: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    bootstrap,
    diamond: Optional[str] = None,
    bold_ticks: Optional[set[str]] = None,
    zero_lw: float = 1.0,
) -> None:
    """One forest panel: a dot per participant, a bootstrap interval, the median.

    Values falling outside the drawn range are pinned to the edge as arrow markers
    rather than silently widening the axis, so one outlier cannot flatten the rest.
    """
    order = list(order)
    intervals = {
        source: bootstrap(d, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        for source, d in differences.items()
    }
    edges = [bound for pair in intervals.values() for bound in pair]
    edges += [rows[source]["median"] for source in order if source in rows]
    edges += list(np.nanpercentile(np.concatenate(list(differences.values())), [8, 92]))
    span = np.nanmax(edges) - np.nanmin(edges)
    lo, hi = np.nanmin(edges) - 0.16 * span, np.nanmax(edges) + 0.16 * span

    for i, source in enumerate(order):
        y = len(order) - 1 - i
        d = np.asarray(differences[source], dtype=float)
        low, high = intervals[source]
        row = rows.get(source)
        significant = bool(row is not None and row["p"] <= alpha)
        colour = SIGNIFICANT_COLOR if significant else NEUTRAL_COLOR

        inside = (d >= lo) & (d <= hi)
        ax.scatter(
            d[inside],
            np.full(int(inside.sum()), y),
            s=13,
            color=colour,
            alpha=0.3,
            zorder=2,
            lw=0,
        )
        for off in d[(d < lo) | (d > hi)]:
            ax.scatter(
                [hi if off > hi else lo],
                [y],
                s=26,
                color=colour,
                alpha=0.55,
                marker=">" if off > hi else "<",
                zorder=2,
                lw=0,
            )
        ax.plot(
            [low, high], [y, y], color=colour, lw=2.4, zorder=3, solid_capstyle="round"
        )
        if row is not None:
            ax.scatter(
                [row["median"]],
                [y],
                s=95,
                color=colour,
                zorder=4,
                marker="D" if source == diamond else "o",
                edgecolor="white",
                linewidth=1.1,
            )
            ax.text(
                1.005,
                y,
                f"p={row['p']:.3f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9.5,
                family="monospace",
                fontweight="bold" if significant else "normal",
                color=colour,
            )

    ax.set_xlim(lo, hi)
    ax.axvline(0.0, color="0.3", lw=zero_lw, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)))
    for tick, source in zip(ax.get_yticklabels(), reversed(order)):
        if bold_ticks and source in bold_ticks:
            tick.set_fontweight("bold")
