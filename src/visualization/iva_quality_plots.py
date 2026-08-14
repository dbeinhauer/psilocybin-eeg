"""
Visualisation for the channel-IVA decomposition-quality analysis.

Renders the results of :mod:`src.analysis.iva_quality`: the reference pair,
per-component group-mean topographies, onset-locked responses and time-frequency
maps, the quality scatterplots, and per-participant channel topographies.

:func:`plot_topomap_diagnostic` and :func:`plot_tf_diagnostic` are the
per-component group views — one panel per component plus the reference, on one
page — and :func:`plot_participant_topomaps` / :func:`plot_participant_tf_maps`
break the same components down per participant.

Every y-axis shares the scatter renderers — :func:`plot_quality_scatter` and
:func:`plot_quality_scatter_per_component` take whatever ``(S, K)`` pair they are
handed, with ``xlabel`` / ``ylabel`` naming the axes (they default to the
topography reference on x and the boxcar time correlation on y).

Every public function:
- accepts pre-computed numpy arrays (usually via an
  :class:`~src.analysis.iva_quality.IvaQualityResult`) and display parameters,
- writes one figure to disk via an optional ``save_path``,
- does **not** call ``plt.show()`` — the caller decides,
- returns the created :class:`~matplotlib.figure.Figure`.

:func:`plot_participant_topomaps` and :func:`plot_participant_tf_maps` are the
exceptions: each writes one figure *per component* rather than a single figure, so
they take a ``root_dir`` and return the paths they wrote. They are deliberately
parallel — same panel ordering, same per-component shared colour limit, same
own-scale reference panel, same own-scale group-mean panel — so a participant can
be traced between its scalp pattern and its time-frequency response.

**Signs are the caller's job.** IVA leaves every ``(subject, component)`` pair's
polarity free, so the map figures — the two participant-comparison sets, the two
group-mean diagnostics and :func:`plot_full_tf_maps` — must be handed maps whose
per-participant sign has been resolved
(:meth:`~src.analysis.iva_quality.IvaQualityResult.topomap_view` /
:meth:`~src.analysis.iva_quality.IvaQualityResult.tf_view`, both applying the one
sign that pair has); otherwise the panels mix polarities and their group mean
cancels. Pass ``alignment_note`` so the figure says what resolved it.

The **scores** want that same sign, here and in the scatters: a correlation is
measured on a pair's maps, so scoring it under a polarity no figure draws rates a
decomposition that was never shown (and averages to a cancelled group score). Hand
every ``(S, K)`` correlation in already aligned —
:meth:`~src.analysis.iva_quality.IvaQualityResult.variants` and
:meth:`~src.analysis.iva_quality.IvaQualityResult.wavelet_variants` with
``aligned=True``, against the ``topo_corr`` of ``topomap_view`` on x.

**Every participant weighs the same.** Anything a figure here averages over
participants first goes through
:func:`~src.analysis.iva_quality.equalize_subject_influence`, which puts each
participant's maps on a common scale; without it a group mean is dominated by the
few loudest participants and, drawn under a colour limit those same participants
set, reads flat. The group-mean panels are additionally drawn on their **own**
scale (annotated with ``|max|``), because an across-participant mean is weaker
than the individual maps by construction — the incoherent part cancels — and the
panel most worth looking at would otherwise be the only unreadable one.
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

from src.analysis.iva_quality import equalize_subject_influence, quality_score

_logger = logging.getLogger(__name__)

# Symmetric-window padding for the scatter axes: a fraction of |r|_max plus a
# small absolute floor, so very tight clusters never touch the axes.
AXIS_PAD_FRAC = 0.15
AXIS_PAD_MIN = 0.05
# Cap on per-component panels; the combined scatter always shows every component.
MAX_PANELS = 30
# Cap on stimulus-onset markers in the whole-recording TF figure. Above it the
# lines merge into a wash that hides the map they were meant to locate things on.
MAX_ONSET_MARKS = 100

# Scatter axis wording. There is one topography reference — the wavelet channel-PCA
# PC1 loading — and it is the x of every scatter; the y is either the paradigm's
# boxcar or the reference TF map (whole or band-limited).
TOPO_AXIS_LABEL = "Topomap correlation with wavelet-PCA PC1 reference"
TIME_AXIS_LABEL = "Onset-locked time correlation"
TF_AXIS_LABEL = "Onset-averaged TF-map correlation with wavelet-PCA PC1"
TF_VARIANT_NAME = "wavelet PC1 TF map"
TF_VARIANT_TAG = "wavelet_tf"


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


def _alignment_suffix(alignment_note: Optional[str]) -> str:
    """Suptitle line naming the per-participant sign alignment, or nothing.

    Every figure that compares or averages participants should say what resolved
    the per-``(subject, component)`` sign, because the maps are unreadable if it
    was not resolved and the ``r`` beside them changes sign with it.

    :param alignment_note: Anchor description, e.g. ``"Cz polarity"``. ``None``
        leaves the title unchanged (the maps arrived as they were computed).
    :return: The line to append, empty when there is nothing to say.
    """
    return "" if alignment_note is None else f"\nper-participant sign: {alignment_note}"


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
# Reference pair
# ---------------------------------------------------------------------------


def plot_wavelet_reference(
    ref_topo: np.ndarray,
    ref_tf: np.ndarray,
    info,
    n_channels: int,
    freqs: np.ndarray,
    epoch_times: np.ndarray,
    *,
    label: str,
    resp_duration_s: float,
    assr_freq: float,
    consistency=None,
    save_path: Optional[Path] = None,
) -> Figure:
    """The wavelet reference pair: PC1 channel loading beside its PC1 TF map.

    Both panels come from one channel-PCA of the trial-averaged, time-z-scored
    wavelet power (:func:`src.analysis.iva_quality.wavelet_reference`) and share
    one polarity, so they are two views of the same reference: *where* on the
    scalp the dominant stimulus-locked wavelet mode sits, and *when / at which
    frequency* it is active. It is the reference **every** score is measured
    against, so read this figure first — one with no visible onset structure makes
    a high correlation with it meaningless.

    :param ref_topo: ``(C,)`` group PC1 channel loading.
    :param ref_tf: ``(F, W)`` group PC1 score map.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels in the IVA subset.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param label: Dataset label shown in the title.
    :param resp_duration_s: Stimulus window marked on the TF panel, in seconds.
    :param assr_freq: Steady-state frequency (Hz) marked on the TF panel.
    :param consistency: Optional
        :class:`~src.analysis.pca_polarity.SignConsistency` for the topography
        panel's subtitle.
    :param save_path: Optional output path.
    :return: The created figure.
    """
    from mne.viz import plot_topomap

    topo_info = topo_info_subset(info, n_channels)
    vlim = _safe_vlim(ref_topo)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    im_topo, _ = plot_topomap(
        ref_topo,
        topo_info,
        axes=axes[0],
        show=False,
        cmap="RdBu_r",
        vlim=(-vlim, vlim),
        contours=4,
    )
    agreement = (
        ""
        if consistency is None
        else (
            f"\nagreeing {consistency.n_agreeing}/{consistency.n_subjects}, "
            f"median pairwise r {consistency.median_pairwise_r:+.2f}"
        )
    )
    axes[0].set_title(
        f"Reference topomap\n(group wavelet-PCA PC1 loading){agreement}", fontsize=9
    )
    fig.colorbar(im_topo, ax=axes[0], shrink=0.7, label="PC1 loading (a.u.)")

    tf_vmax = max(float(np.abs(ref_tf).max()), 1e-12)
    im_tf = axes[1].imshow(
        ref_tf,
        aspect="auto",
        origin="lower",
        extent=[epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]],
        cmap="RdBu_r",
        vmin=-tf_vmax,
        vmax=tf_vmax,
    )
    axes[1].axvline(0.0, color="k", ls="--", lw=0.8, label="onset")
    axes[1].axvline(resp_duration_s, color="k", ls=":", lw=0.8, label="stimulus off")
    axes[1].axhline(
        assr_freq, color="green", ls=":", lw=1.0, label=f"{assr_freq:.0f} Hz"
    )
    axes[1].set_xlabel("Time relative to onset (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Reference TF map\n(group wavelet-PCA PC1 score)", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.colorbar(im_tf, ax=axes[1], shrink=0.9, label="PC1 score (a.u.)")

    fig.suptitle(f"Wavelet reference pair — {label}", fontsize=12)
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
    alignment_note: Optional[str] = None,
) -> list[Path]:
    """One participant-comparison figure per IVA component.

    Each component gets a single figure holding every participant's channel
    topography side by side, followed by the group mean and the reference
    topography — so the spread across participants can be read at a glance.
    Files are written flat into ``<root_dir>/`` and numbered 1-based in **IVA
    component order**, matching the ``IC <k+1>`` labels of the other quality
    figures — it is *not* the score ranking.

    Patterns are first put on a common per-participant scale
    (:func:`~src.analysis.iva_quality.equalize_subject_influence`, measured over
    ``comp_indices``), so every participant panel is readable under one shared
    colour limit and every participant carries the same weight in the group mean.
    Within a figure that limit is one symmetric value derived from the rescaled
    patterns across all subjects, so the panels are directly comparable; the
    shared colourbar reports it. Limits are deliberately **not** shared across
    components, whose pattern magnitudes differ by construction, and the
    reference keeps its own scale because it is a PC1 loading rather than a
    pattern.

    The **group-mean** panel also gets its own scale, matching
    :func:`plot_participant_tf_maps`: averaging over participants cancels the
    incoherent part of every pattern, so the mean is weaker than the individual
    panels and reads flat under their limit — the panel most worth looking at
    would be the only unreadable one. It is annotated with its ``|max|`` to keep
    it from being read against the shared bar.

    Panels are ordered by participant ID (``PSI{number}``, compared numerically)
    and never by ``r``, so a participant sits in the same grid position for
    every component.

    :param patterns: ``(S, K, C)`` sign-oriented forward channel patterns.
    :param ref_topo: ``(C,)`` reference topography, drawn on its own scale.
    :param topo_corr: ``(S, K)`` oriented topomap correlations, used to annotate
        each participant panel and to summarise the component. Correlations are
        scale-invariant, so the rescaling above never contradicts them.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels in the IVA subset.
    :param comp_indices: 0-based IVA component indices to emit. Empty emits
        nothing.
    :param subject_ids: Participant labels, one per subject.
    :param label: Dataset label used in titles and filenames.
    :param root_dir: Directory the figures are written into.
    :param prefix: Optional filename prefix (e.g. ``"alpha_"``).
    :param alignment_note: What resolved the per-participant sign, named in the
        title (e.g. ``"wavelet PC1 reference topomap correlation"``). Pass the
        patterns from
        :meth:`~src.analysis.iva_quality.IvaQualityResult.topomap_view` together
        with this: unaligned patterns mix polarities across the panels and their
        group mean cancels.
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

    if len(comp_indices) == 0:
        # Nothing to draw, and the per-participant scale below has no data to
        # come from.
        return []

    topo_info = topo_info_subset(info, n_channels)
    ref_vlim = _safe_vlim(ref_topo)
    # Equal weight per participant: one scale per subject, so no participant
    # dominates the shared colour limit or the group mean.
    patterns = equalize_subject_influence(patterns, comp_indices)
    # Own limit for the group-mean panels, shared across the components in this
    # call — the mean of the rescaled patterns is far weaker than the individual
    # ones and would read flat under their limit.
    mean_vlim = _safe_vlim(patterns[:, comp_indices].mean(axis=0))
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
        im_shared = None
        for panel, s in enumerate(panel_order):
            im_shared, _ = plot_topomap(
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
        # Own scale: under the participants' shared limit the mean is flat, the
        # across-subject average having cancelled everything incoherent.
        plot_topomap(
            patterns[:, k, :].mean(axis=0),
            topo_info,
            axes=flat[n_subjects],
            show=False,
            cmap="RdBu_r",
            vlim=(-mean_vlim, mean_vlim),
            contours=4,
        )
        flat[n_subjects].set_title(
            f"group mean\n(own scale, |max| {mean_vlim:.3g})",
            fontsize=8,
            fontweight="bold",
        )
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
        # One bar for the participants' shared scale — the group-mean and
        # reference panels have their own, and say so in their titles. It spans
        # every axis rather than just the participant panels, as in
        # plot_participant_tf_maps: attached to a partial set it is placed inside
        # the grid instead of at the figure's edge.
        if im_shared is not None:
            fig.colorbar(
                im_shared,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                label="pattern (a.u., per-participant scale)",
            )
        col = topo_corr[:, k]
        fig.suptitle(
            f"IC {k + 1} — all participants — {label}\n"
            f"panel label = participant (r vs reference)  |  "
            f"r: mean {col.mean():+.2f}, min {col.min():+.2f}, "
            f"max {col.max():+.2f}  |  equal-weighted participants, "
            f"shared |pattern| ≤ {vlim:.3g}"
            f"{_alignment_suffix(alignment_note)}",
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


def plot_participant_tf_maps(
    onset_tf: np.ndarray,
    ref_tf: np.ndarray,
    tf_corr: np.ndarray,
    freqs: np.ndarray,
    epoch_times: np.ndarray,
    comp_indices: list[int],
    subject_ids: list[str],
    *,
    resp_duration_s: float,
    assr_freq: float,
    label: str,
    root_dir: Path,
    prefix: str = "",
    alignment_note: Optional[str] = None,
) -> list[Path]:
    """One participant-comparison **TF-map** figure per IVA component.

    The time-frequency counterpart of :func:`plot_participant_topomaps`, and it
    follows the same conventions throughout: one figure per component holding
    every participant's onset-averaged ``(F, W)`` map side by side, then the group
    mean and the wavelet PC1 reference; files written flat into ``<root_dir>/``
    and numbered 1-based in **IVA component order** (matching the ``IC <k+1>``
    labels elsewhere, *not* the score ranking); panels ordered by participant ID
    so a participant keeps its grid position across every component.

    Where the scatter reduces each participant to one number, this shows the map
    that number came from — so a participant dragging a component's ``tf_corr``
    down can be identified and looked at.

    Maps are first put on a common per-participant scale
    (:func:`~src.analysis.iva_quality.equalize_subject_influence`, measured over
    ``comp_indices``), so every participant panel is readable under one shared
    colour limit and every participant carries the same weight in the group mean.
    Within a figure that limit is one symmetric value derived from the rescaled
    maps across all subjects, so the panels are directly comparable; the shared
    colourbar reports it. Limits are deliberately **not** shared across
    components, and the reference keeps its own scale because it is a PC1 score
    map rather than an IVA source.

    The **group-mean** panel also gets its own scale, shared across the
    components in one call. Averaging over subjects cancels the incoherent part
    of every map, so the mean is far weaker than the individual panels and reads
    flat under their limit — the panel most worth looking at would be the only
    unreadable one. Its limit is the one :func:`plot_tf_diagnostic` uses (which
    equalises the participants the same way), so this panel matches that
    component's panel there; both are annotated with ``|max|`` to keep them from
    being read against the shared bar.

    :param onset_tf: ``(S, K, F, W)`` sign-oriented onset-averaged component TF
        maps.
    :param ref_tf: ``(F, W)`` reference TF map, drawn on its own scale.
    :param tf_corr: ``(S, K)`` oriented TF-map correlations, used to annotate each
        participant panel and to summarise the component. Correlations are
        scale-invariant, so the rescaling above never contradicts them.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param comp_indices: 0-based IVA component indices to emit.
    :param subject_ids: Participant labels, one per subject.
    :param resp_duration_s: Stimulus window marked on every panel, in seconds.
    :param assr_freq: Steady-state frequency (Hz) marked on every panel.
    :param label: Dataset label used in titles and filenames.
    :param root_dir: Directory the figures are written into.
    :param prefix: Optional filename prefix (e.g. ``"alpha_"``).
    :param alignment_note: What resolved the per-participant sign, named in the
        title. Pass the maps from
        :meth:`~src.analysis.iva_quality.IvaQualityResult.tf_view` together with
        this — unaligned maps mix polarities and their group mean cancels. That
        sign comes from the pair's *topography*, so a participant panel here can
        legitimately show a negative driven response; it is the same flip its
        topomap panel carries.
    :return: One path per component, in ``comp_indices`` order.
    :raises ValueError: If the array shapes and ``subject_ids`` disagree.
    """
    if onset_tf.ndim != 4:
        raise ValueError(f"onset_tf must be (S, K, F, W); got shape {onset_tf.shape}.")
    n_subjects = onset_tf.shape[0]
    if len(subject_ids) != n_subjects:
        raise ValueError(
            f"subject_ids has {len(subject_ids)} entries but onset_tf has "
            f"{n_subjects} subjects."
        )
    if tf_corr.shape[0] != n_subjects:
        raise ValueError(
            f"tf_corr has {tf_corr.shape[0]} subjects but onset_tf has {n_subjects}."
        )
    if onset_tf.shape[2:] != ref_tf.shape:
        raise ValueError(
            f"component TF maps {onset_tf.shape[2:]} do not match the reference "
            f"map {ref_tf.shape}."
        )

    if len(comp_indices) == 0:
        # Nothing to draw, and the shared limits below have no data to come from.
        return []

    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    ref_vmax = max(float(np.abs(ref_tf).max()), 1e-12)
    # Equal weight per participant: one scale per subject, so no participant
    # dominates the shared colour limit or the group mean.
    onset_tf = equalize_subject_influence(onset_tf, comp_indices)
    # Own limit for the group-mean panels, shared across the components in this
    # call — the same quantity plot_tf_diagnostic scales its panels by, so the
    # mean panel here and that component's panel there show the same contrast.
    mean_vmax = max(float(np.abs(onset_tf[:, comp_indices].mean(axis=0)).max()), 1e-12)
    root_dir = Path(root_dir)
    written: list[Path] = []
    # Panel order: participant ID, not the order subjects arrive in the arrays.
    panel_order = sorted(
        range(n_subjects), key=lambda s: _participant_sort_key(subject_ids[s])
    )

    def _decorate(ax) -> None:
        ax.axvline(0.0, color="k", ls="--", lw=0.7)
        ax.axvline(resp_duration_s, color="k", ls=":", lw=0.7)
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        ax.tick_params(labelsize=6)

    for k in comp_indices:
        # One limit per component so subjects stay comparable within the figure.
        vmax = max(float(np.abs(onset_tf[:, k]).max()), 1e-12)
        n_panels = n_subjects + 2
        nrows, ncols = _grid_shape(n_panels, max_cols=5)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, 2.8 * nrows),
            squeeze=False,
            layout="constrained",
        )
        flat = axes.flatten()
        im_shared = None
        for panel, s in enumerate(panel_order):
            im_shared = flat[panel].imshow(
                onset_tf[s, k],
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            _decorate(flat[panel])
            flat[panel].set_title(
                f"{subject_ids[s]}  ({tf_corr[s, k]:+.2f})", fontsize=8
            )
        # Own scale: under the participants' shared limit the mean is flat, the
        # across-subject average having cancelled everything incoherent.
        flat[n_subjects].imshow(
            onset_tf[:, k].mean(axis=0),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-mean_vmax,
            vmax=mean_vmax,
        )
        _decorate(flat[n_subjects])
        flat[n_subjects].set_title(
            f"group mean\n(own scale, |max| {mean_vmax:.3g})",
            fontsize=8,
            fontweight="bold",
        )
        # The reference is a PC1 score map, not an IVA source — its own scale.
        flat[n_subjects + 1].imshow(
            ref_tf,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-ref_vmax,
            vmax=ref_vmax,
        )
        _decorate(flat[n_subjects + 1])
        flat[n_subjects + 1].set_title(
            f"reference\n(own scale, |max| {ref_vmax:.3g})",
            fontsize=8,
            fontweight="bold",
        )
        for ax in flat[n_panels:]:
            ax.axis("off")
        # One bar for the shared scale. It spans every axis rather than just the
        # participant panels: attached to a partial set it lands inside the grid
        # and overlaps the reference panel.
        if im_shared is not None:
            fig.colorbar(
                im_shared,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                label="onset-averaged component TF (a.u., per-participant scale)",
            )
        col = tf_corr[:, k]
        fig.supxlabel("Time relative to onset (s)")
        fig.supylabel("Frequency (Hz)")
        fig.suptitle(
            f"IC {k + 1} — onset-averaged TF map, all participants — {label}\n"
            f"panel label = participant (r vs reference)  |  "
            f"r: mean {col.mean():+.2f}, min {col.min():+.2f}, "
            f"max {col.max():+.2f}  |  equal-weighted participants, "
            f"shared |TF| ≤ {vmax:.3g}"
            f"{_alignment_suffix(alignment_note)}",
            fontsize=11,
        )
        path = root_dir / f"{prefix}tf_ic{k + 1:02d}_all_participants_{label}.png"
        _save_fig(fig, path)
        plt.close(fig)
        written.append(path)

    _logger.debug(
        f"Wrote {len(written)} participant TF-map figures "
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

    The responses are put on a common per-participant scale
    (:func:`~src.analysis.iva_quality.equalize_subject_influence`, measured over
    ``comp_indices``) before the mean, so the curve is a statement about every
    participant and not about the few with the largest amplitude.

    :param onset_avgs: ``(S, K, W)`` onset averages, sign-aligned per pair
        (:meth:`~src.analysis.iva_quality.IvaQualityResult.variants` with
        ``aligned=True``) — the curve is an across-participant mean, so mixed
        polarities cancel it.
    :param resp_duration_s: Length of the shaded response window, in seconds.
    :param topo_corr: ``(S, K)`` topomap correlations under that same sign (for the
        panel scores).
    :param time_corr: ``(S, K)`` time correlations under that same sign (for the
        panel scores).
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param comp_indices: Components to show, one panel each.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param variant_name: Name of the time-reduction variant, for the title.
    :param label: Dataset label shown in the title.
    :param save_path: Optional output path.
    :return: The created figure.
    :raises ValueError: If ``comp_indices`` is empty — there is no panel to draw
        and no data to scale the participants by.
    """
    if len(comp_indices) == 0:
        raise ValueError("comp_indices must select at least one component to plot.")
    group_avg = equalize_subject_influence(onset_avgs, comp_indices).mean(axis=0)
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
    fig.supylabel(
        "Group-mean onset-averaged component response\n"
        "(a.u., equal-weighted participants)"
    )
    fig.suptitle(
        f"Onset-locked response vs {resp_duration_s * 1000:.0f} ms stimulus "
        f"window — {variant_name} — {label}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_topomap_diagnostic(
    patterns: np.ndarray,
    ref_topo: np.ndarray,
    topo_corr: np.ndarray,
    second_corr: np.ndarray,
    info,
    n_channels: int,
    comp_indices: list[int],
    *,
    variant_name: str,
    label: str,
    ref_name: str = "wavelet-PCA PC1",
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Group-mean channel topography per component, with the reference topomap.

    The spatial counterpart of :func:`plot_tf_diagnostic`, following the same
    conventions: one panel per component holding the across-participant mean
    pattern, the reference in the last panel, one shared colour limit over the
    component panels and an own scale for the reference. Where
    :func:`plot_participant_topomaps` writes one figure per component to show the
    spread *within* a component, this puts every component on one page — the view
    that answers "which components look like the reference at all" before any of
    them is opened per participant.

    The patterns are put on a common per-participant scale
    (:func:`~src.analysis.iva_quality.equalize_subject_influence`, measured over
    ``comp_indices``) before the across-subject mean, so a panel is a statement
    about every participant rather than about the loudest few. The scale is one
    number per subject, so the *relative* strength of that subject's components
    survives it.

    Component panels then share one symmetric colour limit, so they are
    comparable and a component whose patterns disagree across participants
    legitimately reads flat — the incoherent part cancels in the mean. The
    reference keeps its own scale, being a PC1 loading rather than a pattern.

    :param patterns: ``(S, K, C)`` oriented forward channel patterns, ideally
        already unit-normed per subject (see
        :func:`~src.analysis.wavelet_ica.normalize_patterns_per_subject`).
    :param ref_topo: ``(C,)`` reference topography.
    :param topo_corr: ``(S, K)`` sign-aligned topomap correlations against
        ``ref_topo`` — the panel's mean ``r`` and half of its score.
    :param second_corr: ``(S, K)`` second-axis correlations completing the score
        (a boxcar time correlation or a TF-map correlation), under that same sign.
    :param info: MNE ``Info`` for the topomap layout.
    :param n_channels: Number of channels in the IVA subset.
    :param comp_indices: Components to show, one panel each.
    :param variant_name: Name of the reference variant the score belongs to.
    :param label: Dataset label shown in the title.
    :param ref_name: What the reference panel is, for the title.
    :param alignment_note: What resolved the per-participant sign, named in the
        title. The group mean is only meaningful once it is resolved — see
        :meth:`~src.analysis.iva_quality.IvaQualityResult.topomap_view`.
    :param save_path: Optional output path.
    :return: The created figure.
    :raises ValueError: If ``comp_indices`` is empty — there is no panel to draw
        and no data to scale the participants by.
    """
    from mne.viz import plot_topomap

    if len(comp_indices) == 0:
        raise ValueError("comp_indices must select at least one component to plot.")
    topo_info = topo_info_subset(info, n_channels)
    group_patterns = equalize_subject_influence(patterns, comp_indices).mean(axis=0)
    vlim = _safe_vlim(group_patterns[comp_indices])
    ref_vlim = _safe_vlim(ref_topo)
    n_panels = len(comp_indices) + 1  # + the reference
    nrows, ncols = _grid_shape(n_panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 3.0 * nrows),
        squeeze=False,
        layout="constrained",
    )
    flat = axes.flatten()

    im_shared = None
    for panel, k in enumerate(comp_indices):
        im_shared, _ = plot_topomap(
            group_patterns[k],
            topo_info,
            axes=flat[panel],
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim, vlim),
            contours=4,
        )
        flat[panel].set_title(
            f"IC {k + 1}  (score {quality_score(topo_corr, second_corr, k):+.2f})\n"
            f"mean r {topo_corr[:, k].mean():+.2f}",
            fontsize=8.5,
            fontweight="bold",
        )
    # The reference is a PC1 loading, not a pattern — it keeps its own scale.
    ref_ax = flat[len(comp_indices)]
    plot_topomap(
        ref_topo,
        topo_info,
        axes=ref_ax,
        show=False,
        cmap="RdBu_r",
        vlim=(-ref_vlim, ref_vlim),
        contours=4,
    )
    ref_ax.set_title("reference\n(own scale)", fontsize=8.5, fontweight="bold")
    for ax in flat[n_panels:]:
        ax.axis("off")

    # One bar for the shared component scale. It spans every axis rather than
    # just the component panels: attached to a partial set it is placed inside
    # the grid instead of at the figure's edge.
    if im_shared is not None:
        fig.colorbar(
            im_shared,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            label="group-mean pattern (a.u., equal-weighted participants)",
        )
    fig.suptitle(
        f"Group-mean component topographies vs the {ref_name} reference — "
        f"{variant_name} — {label}\n"
        f"panel label = IC (score) and mean r across participants  |  "
        f"shared |pattern| ≤ {vlim:.3g}"
        f"{_alignment_suffix(alignment_note)}",
        fontsize=12,
    )
    _save_fig(fig, save_path)
    return fig


def plot_tf_diagnostic(
    onset_tf: np.ndarray,
    ref_tf: np.ndarray,
    topo_corr: np.ndarray,
    tf_corr: np.ndarray,
    freqs: np.ndarray,
    epoch_times: np.ndarray,
    comp_indices: list[int],
    *,
    resp_duration_s: float,
    assr_freq: float,
    label: str,
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Group-mean onset-averaged TF map per component, with the reference map.

    The counterpart of :func:`plot_onset_diagnostic` for the TF y-axes: it
    shows the maps the ``tf_corr`` scores are computed from, so a high or low
    correlation can be looked at rather than trusted.

    The maps are put on a common per-participant scale
    (:func:`~src.analysis.iva_quality.equalize_subject_influence`, measured over
    ``comp_indices``) before the across-subject mean, so a group-mean panel is a
    statement about every participant rather than about the loudest few. The scale
    is one number per subject, so the *relative* strength of that subject's
    components survives it.

    Component panels then share one symmetric colour limit, so they are
    comparable and a component with no onset-locked structure legitimately reads
    flat. The reference keeps its own scale, being a PC1 score map rather than an
    IVA source.

    :param onset_tf: ``(S, K, F, W)`` oriented onset-averaged component TF maps.
    :param ref_tf: ``(F, W)`` reference TF map.
    :param topo_corr: ``(S, K)`` sign-aligned wavelet topomap correlations (for
        panel scores).
    :param tf_corr: ``(S, K)`` TF-map correlations under that same sign — the ones
        measured on ``onset_tf`` (for panel scores).
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param comp_indices: Components to show, one panel each.
    :param resp_duration_s: Stimulus window marked on every panel, in seconds.
    :param assr_freq: Steady-state frequency (Hz) marked on every panel.
    :param label: Dataset label shown in the title.
    :param alignment_note: What resolved the per-participant sign, named in the
        title. The group mean is only meaningful once it is resolved — see
        :meth:`~src.analysis.iva_quality.IvaQualityResult.tf_view`.
    :param save_path: Optional output path.
    :return: The created figure.
    :raises ValueError: If ``comp_indices`` is empty — there is no panel to draw
        and no data to scale the participants by.
    """
    if len(comp_indices) == 0:
        raise ValueError("comp_indices must select at least one component to plot.")
    group_tf = equalize_subject_influence(onset_tf, comp_indices).mean(axis=0)
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    vmax = max(float(np.abs(group_tf[comp_indices]).max()), 1e-12)
    n_panels = len(comp_indices) + 1  # + the reference
    nrows, ncols = _grid_shape(n_panels, max_cols=5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        squeeze=False,
        layout="constrained",
    )
    flat = axes.flatten()

    def _decorate(ax) -> None:
        ax.axvline(0.0, color="k", ls="--", lw=0.7)
        ax.axvline(resp_duration_s, color="k", ls=":", lw=0.7)
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        ax.tick_params(labelsize=6)

    im_shared = None
    for panel, k in enumerate(comp_indices):
        ax = flat[panel]
        im_shared = ax.imshow(
            group_tf[k],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        _decorate(ax)
        ax.set_title(
            f"IC {k + 1}  (score {quality_score(topo_corr, tf_corr, k):+.2f})",
            fontsize=8.5,
            fontweight="bold",
        )
    ref_ax = flat[len(comp_indices)]
    ref_vmax = max(float(np.abs(ref_tf).max()), 1e-12)
    ref_ax.imshow(
        ref_tf,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-ref_vmax,
        vmax=ref_vmax,
    )
    _decorate(ref_ax)
    ref_ax.set_title(
        f"reference\n(own scale, |max| {ref_vmax:.3g})", fontsize=8.5, fontweight="bold"
    )
    for ax in flat[n_panels:]:
        ax.axis("off")

    # One bar at the figure's right edge for the shared component scale. It spans
    # every axis, not just the component panels: attached to a partial set it
    # would be placed inside the grid and overlap the reference panel.
    if im_shared is not None:
        fig.colorbar(
            im_shared,
            ax=axes.ravel().tolist(),
            shrink=0.6,
            label="group-mean component TF (a.u., equal-weighted participants)",
        )
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel("Frequency (Hz)")
    fig.suptitle(
        f"Onset-averaged component TF maps vs the wavelet-PCA PC1 reference "
        f"({resp_duration_s * 1000:.0f} ms stimulus) — {label}"
        f"{_alignment_suffix(alignment_note)}",
        fontsize=12,
    )
    _save_fig(fig, save_path)
    return fig


def plot_full_tf_maps(
    full_tf_mean: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    comp_indices: list[int],
    *,
    assr_freq: float,
    label: str,
    onset_times: Optional[np.ndarray] = None,
    alignment_note: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Group-mean **whole-recording** TF map per component, under the shared signs.

    The QC counterpart of :func:`plot_tf_diagnostic`: the same components and the
    same per-``(subject, component)`` signs, but over the entire recording rather
    than the onset epoch. Two things only this figure can show — what a component
    does *between* the stimuli (drift, a burst of muscle, an electrode going bad),
    and whether its onset-locked structure is present throughout or carried by one
    stretch of the recording.

    Read it against the standard IVA time-frequency figure, which averages the raw
    sources: that one leaves IVA's arbitrary per-pair polarity in place, so its
    group mean cancels whatever the participants share and a component can look
    empty there while reading clearly here. The difference between the two figures
    is the cancellation.

    The map arrives already sign-aligned and equal-weighted across participants
    (:func:`~src.analysis.iva_quality.full_tf_group_mean`), so this function only
    draws it. Component panels share one symmetric colour limit, as everywhere
    else here, so a quiet component legitimately reads flat.

    :param full_tf_mean: ``(K, F, T)`` group-mean whole-recording TF maps.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param times: ``(T,)`` recording time axis in seconds.
    :param comp_indices: Components to show, one row each.
    :param assr_freq: Steady-state frequency (Hz) marked on every panel.
    :param label: Dataset label shown in the title.
    :param onset_times: Optional stimulus onset times in seconds, drawn as thin
        markers. Onsets past the end of ``times`` are dropped — the recording was
        cropped, and a marker outside the map would only stretch the axis.
        Skipped altogether (and said so in the title) above
        :data:`MAX_ONSET_MARKS`, where the lines would cover the map rather than
        locate anything on it.
    :param alignment_note: What resolved the per-participant sign, named in the
        title.
    :param save_path: Optional output path.
    :return: The created figure.
    :raises ValueError: If ``full_tf_mean`` is not 3-D, its axes disagree with
        ``freqs`` / ``times``, or ``comp_indices`` is empty.
    """
    if full_tf_mean.ndim != 3:
        raise ValueError(
            f"full_tf_mean must be (K, F, T); got shape {full_tf_mean.shape}."
        )
    if full_tf_mean.shape[1] != len(freqs):
        raise ValueError(
            f"full_tf_mean has {full_tf_mean.shape[1]} frequency bins but "
            f"{len(freqs)} frequencies were given."
        )
    if full_tf_mean.shape[2] != len(times):
        raise ValueError(
            f"full_tf_mean has {full_tf_mean.shape[2]} samples but {len(times)} "
            f"times were given."
        )
    if len(comp_indices) == 0:
        raise ValueError("comp_indices must select at least one component to plot.")

    vmax = _safe_vlim(full_tf_mean[comp_indices])
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    n_comp = len(comp_indices)
    fig, axes = plt.subplots(
        n_comp,
        1,
        figsize=(14, 2.6 * n_comp),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    flat = axes.flatten()

    marks: np.ndarray | None = None
    n_onsets = 0
    if onset_times is not None:
        inside = np.asarray(onset_times, dtype=float)
        inside = inside[(inside >= times[0]) & (inside <= times[-1])]
        n_onsets = len(inside)
        if n_onsets <= MAX_ONSET_MARKS:
            marks = inside

    im = None
    for panel, k in enumerate(comp_indices):
        ax = flat[panel]
        im = ax.imshow(
            full_tf_mean[k],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        if marks is not None:
            for onset in marks:
                ax.axvline(onset, color="k", ls="-", lw=0.3, alpha=0.35)
        ax.set_ylabel("Freq (Hz)", fontsize=8)
        ax.set_title(f"IC {k + 1}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    flat[-1].set_xlabel("Time (s)")

    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.01,
            label="group-mean component TF (a.u., equal-weighted participants)",
        )
    onset_note = ""
    if onset_times is not None:
        onset_note = (
            f"  |  {n_onsets} stimulus onsets marked"
            if marks is not None
            else f"  |  {n_onsets} stimulus onsets, too many to mark"
        )
    fig.suptitle(
        f"Whole-recording component TF maps (QC) — {label}\n"
        f"group mean under the per-participant sign, shared |TF| ≤ {vmax:.3g}"
        f"{onset_note}{_alignment_suffix(alignment_note)}",
        fontsize=12,
    )
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
    xlabel: str = TOPO_AXIS_LABEL,
    ylabel: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Combined scatter: a topomap correlation (x) vs a second score (y).

    Every ``(subject, component)`` pair is one point, coloured by component.
    Both axes share one symmetric window so the ideal ``(1, 1)`` corner sits in
    a fixed place.

    :param topo_corr: ``(S, K)`` topomap correlations, sign-aligned per pair (see
        the module docstring) so a point scores the maps the figures draw.
    :param time_corr: ``(S, K)`` second-axis correlations under that same sign (a
        boxcar time correlation, or a TF-map correlation whole or band-limited).
    :param comp_indices: Components to plot.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param lim: Half-width of the square window, from :func:`axis_limit`.
    :param variant_name: Name of the reference variant.
    :param label: Dataset label shown in the title.
    :param xlabel: x-axis label; defaults to the topography reference.
    :param ylabel: y-axis label; defaults to
        ``f"{TIME_AXIS_LABEL} ({variant_name})"``.
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f"{TIME_AXIS_LABEL} ({variant_name})")
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
    xlabel: str = TOPO_AXIS_LABEL,
    ylabel: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """One small scatter per component; points are subjects labelled by ID.

    Each panel keeps that component's colour and the shared symmetric window, so
    a point can be located across this and the combined figure.

    :param topo_corr: ``(S, K)`` sign-aligned topomap correlations (see
        :func:`plot_quality_scatter`).
    :param time_corr: ``(S, K)`` second-axis correlations under the same sign (see
        :func:`plot_quality_scatter`).
    :param comp_indices: Components to show, one panel each.
    :param colors: Component index → colour, from :func:`component_colors`.
    :param subject_ids: Participant labels, one per subject.
    :param lim: Half-width of the square window, from :func:`axis_limit`.
    :param variant_name: Name of the reference variant.
    :param label: Dataset label shown in the title.
    :param xlabel: x-axis label; defaults to the topography reference.
    :param ylabel: y-axis label; defaults to
        ``f"{TIME_AXIS_LABEL} ({variant_name})"``.
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
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel or f"{TIME_AXIS_LABEL} ({variant_name})")
    fig.suptitle(
        f"Per-component quality (label = participant ID) — {variant_name} — "
        f"{label}  |  window |r| ≤ {lim:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
