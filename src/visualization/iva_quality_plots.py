"""
Visualisation for the channel-IVA decomposition-quality analysis.

Renders the results of :mod:`src.analysis.iva_quality`: the reference
topography, per-component onset-locked responses, the topomap-vs-time quality
scatterplots, and per-participant channel topographies.

Every public function:
- accepts pre-computed numpy arrays (usually via an
  :class:`~src.analysis.iva_quality.IvaQualityResult`) and display parameters,
- writes one figure to disk via an optional ``save_path``,
- does **not** call ``plt.show()`` — the caller decides,
- returns the created :class:`~matplotlib.figure.Figure`.

:func:`plot_participant_topomaps` is the one exception: it writes one figure
*per component* rather than a single figure, so it takes a ``root_dir`` and
returns the paths it wrote.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.analysis.iva_quality import quality_score

_logger = logging.getLogger(__name__)

# Symmetric-window padding for the scatter axes: a fraction of |r|_max plus a
# small absolute floor, so very tight clusters never touch the axes.
AXIS_PAD_FRAC = 0.15
AXIS_PAD_MIN = 0.05
# Cap on per-component panels; the combined scatter always shows every component.
MAX_PANELS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: Figure, save_path: Optional[Path]) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _safe_vlim(arr: np.ndarray) -> float:
    """Symmetric colour limit from the 99th percentile of ``|arr|``."""
    return max(float(np.percentile(np.abs(arr), 99)), 1e-12)


def _participant_sort_key(subject_id: str) -> tuple[int, int, str]:
    """Sort key ordering ``PSI{number}`` labels numerically.

    Participant IDs are ``PSI`` followed by digits (zero-padded to three in the
    filenames, but not guaranteed to be once they reach a plot label), so the
    numeric part is compared as an integer rather than lexicographically. IDs
    without a trailing number sort last, alphabetically.

    :param subject_id: Participant label, e.g. ``"PSI007"``.
    :return: ``(has_no_number, number, subject_id)`` sort key.
    """
    match = re.search(r"(\d+)\s*$", subject_id)
    if match is None:
        return (1, 0, subject_id)
    return (0, int(match.group(1)), subject_id)


def _grid_shape(n_panels: int, max_cols: int = 6) -> tuple[int, int]:
    """Balanced ``(nrows, ncols)`` for ``n_panels`` panels.

    Filling rows greedily up to ``max_cols`` can strand a single panel on a row
    of its own (7 panels → 6 + 1); this spreads the panels evenly instead.

    :param n_panels: Number of panels to lay out.
    :param max_cols: Maximum columns per row.
    :return: ``(nrows, ncols)`` covering every panel.
    """
    ncols = min(max_cols, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, int(np.ceil(n_panels / nrows))


def topo_info_subset(info, n_channels: int):
    """EEG-only ``Info`` restricted to the first ``n_channels`` (IVA order).

    :param info: Full MNE ``Info``.
    :param n_channels: Number of channels in the IVA subset.
    :return: An ``Info`` whose channel order matches the IVA channel axis.
    """
    import mne

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(topo_info.ch_names):
        topo_info = mne.pick_info(topo_info, list(range(n_channels)))
    return topo_info


def component_colors(comp_indices: list[int]) -> dict[int, tuple]:
    """Stable colour per component (``tab20`` for ≤ 20 components, else ``hsv``).

    :param comp_indices: 0-based IVA component indices.
    :return: Mapping from component index to an RGBA colour.
    """
    if len(comp_indices) <= 20:
        base = list(plt.colormaps["tab20"].colors)
        return {k: base[i % 20] for i, k in enumerate(comp_indices)}
    cmap = plt.colormaps["hsv"]
    return {k: cmap(i / len(comp_indices)) for i, k in enumerate(comp_indices)}


def axis_limit(
    topo_corr: np.ndarray, time_corr: np.ndarray, comp_indices: list[int]
) -> float:
    """Symmetric ``[-lim, lim]`` window from the largest ``|r|`` on either axis.

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` time correlations.
    :param comp_indices: Components under consideration.
    :return: The half-width of the shared square window, capped at 1.0.
    """
    max_abs = float(
        max(
            np.abs(topo_corr[:, comp_indices]).max(),
            np.abs(time_corr[:, comp_indices]).max(),
        )
    )
    return min(1.0, max_abs * (1.0 + AXIS_PAD_FRAC) + AXIS_PAD_MIN)


def panel_indices(
    topo_corr: np.ndarray,
    time_corr: np.ndarray,
    comp_indices: list[int],
    max_panels: int = MAX_PANELS,
) -> list[int]:
    """Components to show in a per-component grid, capped at ``max_panels``.

    When capped, the highest-scoring components are kept and returned in IVA
    order so panel positions stay comparable across figures.

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` time correlations.
    :param comp_indices: Candidate components.
    :param max_panels: Maximum number of panels.
    :return: The selected component indices, sorted ascending.
    """
    if len(comp_indices) <= max_panels:
        return list(comp_indices)
    best = sorted(
        comp_indices,
        key=lambda k: quality_score(topo_corr, time_corr, k),
        reverse=True,
    )[:max_panels]
    return sorted(best)


# ---------------------------------------------------------------------------
# Reference topography
# ---------------------------------------------------------------------------


def plot_reference_topomap(
    ref_topo: np.ndarray,
    info,
    n_channels: int,
    *,
    label: str,
    save_path: Optional[Path] = None,
) -> Figure:
    """Reference (group raw-PCA evoked PC1) topomap on the IVA channel subset.

    :param ref_topo: ``(C,)`` reference topography.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels in the IVA subset.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The created figure.
    """
    from mne.viz import plot_topomap

    topo_info = topo_info_subset(info, n_channels)
    vlim = _safe_vlim(ref_topo)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    im, _ = plot_topomap(
        ref_topo,
        topo_info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(-vlim, vlim),
        contours=4,
    )
    ax.set_title(f"Reference topomap\n(group raw-PCA evoked PC1) — {label}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="PC1 loading (a.u.)")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-participant topographies
# ---------------------------------------------------------------------------


def plot_participant_topomaps(
    patterns: np.ndarray,
    ref_topo: np.ndarray,
    topo_corr: np.ndarray,
    info,
    n_channels: int,
    comp_indices: list[int],
    subject_ids: list[str],
    *,
    label: str,
    root_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """One participant-comparison figure per IVA component.

    Each component gets a single figure holding every participant's channel
    topography side by side, followed by the group mean and the raw-PCA
    reference — so the spread across participants can be read at a glance.
    Files are written flat into ``<root_dir>/`` and numbered 1-based in **IVA
    component order**, matching the ``IC <k+1>`` labels of the other quality
    figures — it is *not* the score ranking.

    Within a figure every participant panel shares one symmetric colour limit
    derived from that component's patterns across all subjects, so the panels
    are directly comparable; the shared colourbar reports it. Limits are
    deliberately **not** shared across components, whose pattern magnitudes
    differ by construction, and the reference keeps its own scale because it is
    a PC1 loading rather than a pattern.

    Panels are ordered by participant ID (``PSI{number}``, compared numerically)
    and never by ``r``, so a participant sits in the same grid position for
    every component.

    :param patterns: ``(S, K, C)`` sign-oriented forward channel patterns.
    :param ref_topo: ``(C,)`` reference topography, drawn on its own scale.
    :param topo_corr: ``(S, K)`` oriented topomap correlations, used to annotate
        each participant panel and to summarise the component.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels in the IVA subset.
    :param comp_indices: 0-based IVA component indices to emit.
    :param subject_ids: Participant labels, one per subject.
    :param label: Dataset label used in titles and filenames.
    :param root_dir: Directory the figures are written into.
    :param prefix: Optional filename prefix (e.g. ``"alpha_"``).
    :return: One path per component, in ``comp_indices`` order.
    :raises ValueError: If the array shapes and ``subject_ids`` disagree.
    """
    from mne.viz import plot_topomap

    if patterns.ndim != 3:
        raise ValueError(f"patterns must be (S, K, C); got shape {patterns.shape}.")
    n_subjects = patterns.shape[0]
    if len(subject_ids) != n_subjects:
        raise ValueError(
            f"subject_ids has {len(subject_ids)} entries but patterns has "
            f"{n_subjects} subjects."
        )
    if topo_corr.shape[0] != n_subjects:
        raise ValueError(
            f"topo_corr has {topo_corr.shape[0]} subjects but patterns has "
            f"{n_subjects}."
        )

    topo_info = topo_info_subset(info, n_channels)
    ref_vlim = _safe_vlim(ref_topo)
    root_dir = Path(root_dir)
    written: list[Path] = []
    # Panel order: participant ID, not the order subjects arrive in the arrays.
    panel_order = sorted(
        range(n_subjects), key=lambda s: _participant_sort_key(subject_ids[s])
    )

    for k in comp_indices:
        # One limit per component so subjects stay comparable within the figure.
        vlim = _safe_vlim(patterns[:, k, :])
        n_panels = n_subjects + 2
        nrows, ncols = _grid_shape(n_panels)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.6 * ncols, 3.0 * nrows),
            squeeze=False,
            layout="constrained",
        )
        flat = axes.flatten()
        for panel, s in enumerate(panel_order):
            plot_topomap(
                patterns[s, k],
                topo_info,
                axes=flat[panel],
                show=False,
                cmap="RdBu_r",
                vlim=(-vlim, vlim),
                contours=4,
            )
            flat[panel].set_title(
                f"{subject_ids[s]}  ({topo_corr[s, k]:+.2f})", fontsize=8
            )
        im_shared, _ = plot_topomap(
            patterns[:, k, :].mean(axis=0),
            topo_info,
            axes=flat[n_subjects],
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim, vlim),
            contours=4,
        )
        flat[n_subjects].set_title("group mean", fontsize=8, fontweight="bold")
        # The reference is a PC1 loading, not a pattern — it keeps its own scale.
        plot_topomap(
            ref_topo,
            topo_info,
            axes=flat[n_subjects + 1],
            show=False,
            cmap="RdBu_r",
            vlim=(-ref_vlim, ref_vlim),
            contours=4,
        )
        flat[n_subjects + 1].set_title(
            "reference\n(own scale)", fontsize=8, fontweight="bold"
        )
        for ax in flat[n_panels:]:
            ax.axis("off")
        # One bar for the shared scale; the reference panel is excluded from it.
        fig.colorbar(
            im_shared,
            ax=list(flat[: n_subjects + 1]),
            fraction=0.03,
            pad=0.02,
            label="pattern (a.u.)",
        )
        col = topo_corr[:, k]
        fig.suptitle(
            f"IC {k + 1} — all participants — {label}\n"
            f"panel label = participant (r vs reference)  |  "
            f"r: mean {col.mean():+.2f}, min {col.min():+.2f}, "
            f"max {col.max():+.2f}  |  shared |pattern| ≤ {vlim:.3g}",
            fontsize=11,
        )
        path = root_dir / f"{prefix}topomap_ic{k + 1:02d}_all_participants_{label}.png"
        _save_fig(fig, path)
        plt.close(fig)
        written.append(path)

    _logger.debug(
        f"Wrote {len(written)} participant-comparison figures "
        f"(one per component) to {root_dir}"
    )
    return written


# ---------------------------------------------------------------------------
# Onset-locked response diagnostic
# ---------------------------------------------------------------------------


def plot_onset_diagnostic(
    onset_avgs: np.ndarray,
    resp_duration_s: float,
    topo_corr: np.ndarray,
    time_corr: np.ndarray,
    epoch_times: np.ndarray,
    comp_indices: list[int],
    colors: dict,
    *,
    variant_name: str,
    label: str,
    save_path: Optional[Path] = None,
) -> Figure:
    """Per-component group-mean onset-average with the rigid response window.

    Shows, for every component, the across-subject mean onset-triggered response
    with the fixed window ``[0, resp_duration_s]`` shaded — so you can see
    whether it looks like a stimulus response and how well it fills the window
    the score is measured against.

    :param onset_avgs: ``(S, K, W)`` oriented onset averages.
    :param resp_duration_s: Length of the shaded response window, in seconds.
    :param topo_corr: ``(S, K)`` topomap correlations (for the panel scores).
    :param time_corr: ``(S, K)`` time correlations (for the panel scores).
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param comp_indices: Components to show, one panel each.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param variant_name: Name of the time-reduction variant, for the title.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The created figure.
    """
    group_avg = onset_avgs.mean(axis=0)  # (K, W)
    n_comp = len(comp_indices)
    ncols = min(5, n_comp)
    nrows = int(np.ceil(n_comp / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False, sharex=True
    )
    flat = axes.flatten()
    for panel, k in enumerate(comp_indices):
        ax = flat[panel]
        ax.plot(epoch_times, group_avg[k], color=colors[k], lw=1.4)
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.axvspan(0.0, resp_duration_s, color="0.5", alpha=0.15)
        ax.axhline(0.0, color="gray", ls=":", lw=0.5)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"IC {k + 1}  (score {quality_score(topo_corr, time_corr, k):+.2f})",
            fontsize=8.5,
            color="black",
            fontweight="bold",
        )
    for ax in flat[n_comp:]:
        ax.axis("off")
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel("Group-mean onset-averaged component response (a.u.)")
    fig.suptitle(
        f"Onset-locked response vs {resp_duration_s * 1000:.0f} ms stimulus "
        f"window — {variant_name} — {label}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Quality scatterplots
# ---------------------------------------------------------------------------


def plot_quality_scatter(
    topo_corr: np.ndarray,
    time_corr: np.ndarray,
    comp_indices: list[int],
    colors: dict,
    lim: float,
    *,
    variant_name: str,
    label: str,
    save_path: Optional[Path] = None,
) -> Figure:
    """Combined scatter: topomap correlation (x) vs time correlation (y).

    Every ``(subject, component)`` pair is one point, coloured by component.
    Both axes share one symmetric window so the ideal ``(1, 1)`` corner sits in
    a fixed place.

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` time correlations.
    :param comp_indices: Components to plot.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param lim: Half-width of the square window, from :func:`axis_limit`.
    :param variant_name: Name of the time-reduction variant.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The created figure.
    """
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    n_subjects = topo_corr.shape[0]
    for s in range(n_subjects):
        for k in comp_indices:
            ax.scatter(
                topo_corr[s, k],
                time_corr[s, k],
                color=colors[k],
                marker="o",
                s=70,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.9,
                zorder=2,
            )
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.axvline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Topomap correlation with raw-PCA reference")
    ax.set_ylabel(f"Onset-locked time correlation ({variant_name})")
    ax.set_title(
        f"IVA channel-component quality — {variant_name} — {label}\n"
        f"(colour = component; window |r| ≤ {lim:.2f})"
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[k],
            markeredgecolor="black",
            markersize=8,
            label=f"IC {k + 1}  ({quality_score(topo_corr, time_corr, k):+.2f})",
        )
        for k in comp_indices
    ]
    legend = ax.legend(
        handles=handles,
        title="Component  (score)",
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1 + (len(comp_indices) > 15),
    )
    ax.add_artist(legend)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_extra_artists keeps the out-of-axes legend inside the tight crop.
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend,)
        )
    return fig


def plot_quality_scatter_per_component(
    topo_corr: np.ndarray,
    time_corr: np.ndarray,
    comp_indices: list[int],
    colors: dict,
    subject_ids: list[str],
    lim: float,
    *,
    variant_name: str,
    label: str,
    save_path: Optional[Path] = None,
) -> Figure:
    """One small scatter per component; points are subjects labelled by ID.

    Each panel keeps that component's colour and the shared symmetric window, so
    a point can be located across this and the combined figure.

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` time correlations.
    :param comp_indices: Components to show, one panel each.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param subject_ids: Participant labels, one per subject.
    :param lim: Half-width of the square window, from :func:`axis_limit`.
    :param variant_name: Name of the time-reduction variant.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The created figure.
    """
    n_subjects = topo_corr.shape[0]
    n_comp = len(comp_indices)
    ncols = min(5, n_comp)
    nrows = int(np.ceil(n_comp / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.9 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    for panel, k in enumerate(comp_indices):
        ax = flat[panel]
        for s in range(n_subjects):
            x, y = topo_corr[s, k], time_corr[s, k]
            ax.scatter(
                x,
                y,
                color=colors[k],
                marker="o",
                s=55,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.9,
                zorder=2,
            )
            ax.annotate(
                subject_ids[s],
                (x, y),
                textcoords="offset points",
                xytext=(3.0, 2.5),
                fontsize=6,
                color="black",
                zorder=3,
            )
        ax.axhline(0.0, ls="--", lw=0.5, color="gray")
        ax.axvline(0.0, ls="--", lw=0.5, color="gray")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"IC {k + 1}  (score {quality_score(topo_corr, time_corr, k):+.2f})",
            fontsize=8.5,
            color="black",
            fontweight="bold",
        )
    for ax in flat[n_comp:]:
        ax.axis("off")
    fig.supxlabel("Topomap correlation with raw-PCA reference")
    fig.supylabel(f"Onset-locked time correlation ({variant_name})")
    fig.suptitle(
        f"Per-component quality (label = participant ID) — {variant_name} — "
        f"{label}  |  window |r| ≤ {lim:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
