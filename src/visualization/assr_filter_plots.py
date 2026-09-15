"""
Topographies of the ASSR spatial filters a trial-level run applies.

Check figures for the operators built in :mod:`src.analysis.assr_trials`, drawn so a
reader can see what each ``mean_*`` / ``masked_*`` row of an SNR grid actually read
before trusting the numbers that came out of it. Two questions, one figure each:

* :func:`plot_cohort_mean_topomaps` — **is the cohort average a real map, or a
  cancellation?** One panel per component of
  :attr:`~src.analysis.assr_trials.CohortMeanFilter.pattern`, with the fixed anchor mask
  drawn beside them for reference, and each panel annotated with its correlation to that
  mask and the median cosine of the recordings that went into it. A component whose
  median cosine sits near zero has no cohort map, so its ``mean_*`` rows are a statement
  about an average nobody resembles.
* :func:`plot_masked_filter_topomaps` — **which electrodes does the restricted operator
  listen to, and how strongly?** The shared filter after
  :func:`~src.analysis.assr_trials.restrict_filters_to_mask`, annotated with the share of
  its L2 weight that survived. Plotted as filter **weights**, not as a pattern, because
  the weights are what the ``masked_*`` variants apply.

Both follow the project's plotting contract: pre-computed arrays in, one figure out, an
optional ``save_path``, no ``plt.show()``, and the created figure returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure

from src.analysis.assr_trials import COMPONENT_LABEL
from src.visualization.iva_quality_plots import _grid_shape, save_fig

__all__ = ["plot_cohort_mean_topomaps", "plot_masked_filter_topomaps"]


def _topo_grid(n_panels: int) -> tuple[Figure, list]:
    """A square-ish grid of topomap axes, unused panels switched off.

    :param n_panels: Number of topomaps to draw.
    :return: ``(fig, panels)`` with exactly *n_panels* live axes, in order.
    """
    nrows, ncols = _grid_shape(n_panels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows), squeeze=False
    )
    panels = list(axes.flat)
    for spare in panels[n_panels:]:
        spare.axis("off")
    return fig, panels[:n_panels]


def plot_cohort_mean_topomaps(
    pattern: np.ndarray,
    info,
    electrode_mask: np.ndarray,
    *,
    mask_corr: np.ndarray,
    cosine_to_mean: np.ndarray,
    n_recordings: int,
    align_polarity: bool = True,
    normalize: bool = True,
    comp_indices: Optional[Sequence[int]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """The cohort-mean topographies, beside the fixed anchor mask.

    The reference panel is the point of the figure as much as the components are: a
    ``mean_*`` row is only worth reading against the binary reference if the map it
    averages sits somewhere near that mask, and the two are drawn on the same head so
    that is a look rather than an inference.

    Each panel is drawn on its own colour limit — IVA fixes every component's scale
    independently, so a shared limit across columns would flatten all but the largest —
    and there is deliberately no colourbar, which with a limit per panel would mean
    nothing. What the panels can be compared on is *shape*, which is what the annotated
    correlation and cosine quantify.

    :param pattern: ``(components, channels)`` cohort-average forward pattern, from
        :func:`~src.analysis.assr_trials.cohort_mean_pattern`.
    :param info: MNE ``Info`` for the topomap layout, on the same channel axis.
    :param electrode_mask: Boolean ``(channels,)`` anchor mask, drawn as the last panel.
    :param mask_corr: ``(components,)`` correlation of each mean map with that mask.
    :param cosine_to_mean: ``(recordings, components)`` cosine of each recording's
        aligned topography with the mean; its median annotates each panel.
    :param n_recordings: How many recordings the average pooled, for the title.
    :param align_polarity: Whether the signs were anchored before averaging. Named in
        the title because an unaligned average is a different quantity, not a variant.
    :param normalize: Whether each topography was unit-normed before averaging.
    :param comp_indices: 0-based components to draw; ``None`` draws every one.
    :param save_path: Destination, or ``None`` to draw without saving.
    :return: The created figure.
    :raises ValueError: If the arrays disagree on the component or channel axis.
    """
    maps = np.asarray(pattern, dtype=float)
    mask = np.asarray(electrode_mask, dtype=bool)
    if maps.ndim != 2:
        raise ValueError(f"pattern must be (components, channels); got {maps.shape}.")
    if mask.shape != (maps.shape[1],):
        raise ValueError(
            f"electrode_mask must be ({maps.shape[1]},) to match the pattern's channel "
            f"axis; got {mask.shape}."
        )
    indices = list(range(maps.shape[0])) if comp_indices is None else list(comp_indices)
    corr = np.asarray(mask_corr, dtype=float)
    cosine = np.asarray(cosine_to_mean, dtype=float)

    fig, panels = _topo_grid(len(indices) + 1)
    for panel, k in zip(panels, indices):
        mne.viz.plot_topomap(
            maps[k],
            info,
            axes=panel,
            show=False,
            cmap="RdBu_r",
            contours=4,
            sensors=False,
        )
        panel.set_title(
            f"{COMPONENT_LABEL.format(k=k + 1)}\n"
            f"corr w/ mask {corr[k]:+.2f}, median cos {np.median(cosine[:, k]):.2f}",
            fontsize=10,
        )
    # The fixed reference, on the same head: greyscale and contour-free, so it never
    # reads as one more component.
    mne.viz.plot_topomap(
        mask.astype(float),
        info,
        axes=panels[-1],
        show=False,
        cmap="Greys",
        contours=0,
        sensors=False,
    )
    panels[-1].set_title(
        f"ASSR mask (reference)\n{int(mask.sum())} electrode(s)", fontsize=10
    )
    fig.suptitle(
        "Cohort-mean topographies — the shared spatial filter of the mean_* variants "
        f"({n_recordings} recordings"
        f"{', polarity-aligned' if align_polarity else ', RAW signs'}"
        f"{', unit-norm' if normalize else ''})",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_masked_filter_topomaps(
    masked_filter: np.ndarray,
    info,
    *,
    weight_share: np.ndarray,
    n_mask_channels: int,
    variant: str,
    comp_indices: Optional[Sequence[int]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """The shared spatial filter after the anchor-electrode restriction.

    Filter **weights**, not a forward pattern: this is the operator the restricted
    variants apply, so the figure answers "which electrodes does it listen to, and how
    strongly" rather than "where would this source show up on the scalp". Contours are
    off because the field is zero almost everywhere by construction and interpolated
    contour lines over the zeroed area would suggest structure that is not there.

    The annotated share is :func:`~src.analysis.assr_trials.mask_weight_share`: near 1
    means the component was reading the anchor area anyway and its restricted row should
    track the unrestricted one; near 0 means the restriction changed *what signal is
    read*, not just how cleanly.

    :param masked_filter: ``(components, channels)`` restricted filter weights.
    :param info: MNE ``Info`` for the topomap layout, on the same channel axis.
    :param weight_share: ``(components,)`` share of L2 weight kept by the restriction.
    :param n_mask_channels: How many electrodes the mask keeps, for the title.
    :param variant: Name of the signal variant this operator belongs to, so the figure
        says which CSV rows it explains.
    :param comp_indices: 0-based components to draw; ``None`` draws every one.
    :param save_path: Destination, or ``None`` to draw without saving.
    :return: The created figure.
    :raises ValueError: If *masked_filter* is not 2-D.
    """
    weights = np.asarray(masked_filter, dtype=float)
    if weights.ndim != 2:
        raise ValueError(
            f"masked_filter must be (components, channels); got {weights.shape}."
        )
    indices = (
        list(range(weights.shape[0])) if comp_indices is None else list(comp_indices)
    )
    share = np.asarray(weight_share, dtype=float)

    fig, panels = _topo_grid(len(indices))
    for panel, k in zip(panels, indices):
        mne.viz.plot_topomap(
            weights[k],
            info,
            axes=panel,
            show=False,
            cmap="RdBu_r",
            contours=0,
            sensors=False,
        )
        panel.set_title(
            f"{COMPONENT_LABEL.format(k=k + 1)}\n{share[k]:.0%} of |w| kept",
            fontsize=10,
        )
    fig.suptitle(
        "Cohort-mean spatial FILTER restricted to the ASSR electrodes — the operator "
        f"'{variant}' applies ({n_mask_channels} electrode(s) kept)",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
