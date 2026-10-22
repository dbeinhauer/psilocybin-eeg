"""
Figures specific to a CorrCA condition comparison.

The condition-mean and per-participant topography / TF grids are shared with the IVA
runs (:mod:`src.visualization.iva_condition_plots`). What CorrCA adds, and what is
drawn here, is what has no IVA counterpart:

* the **shared** forward pattern of each component — CorrCA fits one filter for every
  recording, so there is one group topography the model itself defines;
* the **synchrony per condition** — the ISC of each component within each condition,
  from the same filters, with every participant's leave-one-out ISC paired across
  conditions. The stage-06 IVA runs draw it too, on their sign-anchored sources.
"""

from __future__ import annotations

from collections.abc import Sequence
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
):
    """One row of shared forward patterns, one column per component.

    Each panel is on its own symmetric scale: CorrCA fixes a component's scale, not a
    common one across components.

    :param patterns: ``(K, C)`` shared forward patterns (``CorrcaResult.patterns``).
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
    title = f"Shared CorrCA topographies (pooled fit) — {label}"
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
