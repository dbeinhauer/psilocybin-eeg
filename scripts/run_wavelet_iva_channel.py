"""
CLI script reproducing the analyses from
``notebooks/05-wavelet-iva-analysis/wavelet_iva_channel.ipynb``
(channel as the independent / mixing dimension: K = S subjects,
IVA mixing = channels, samples = the joint **time × frequency** axis;
per-subject PCA over channels → IVA-G).

Unlike the frequency_channel variant (which puts ``C × F`` on the mixing
axis and time on the sample axis), here **frequency moves into the sample
axis**. Each IVA component is therefore a shared-across-subjects
spectro-temporal source ``(F, T)`` plus a per-subject channel topography
``(C,)``. The aligned SCV is the whole time-frequency pattern, so a
component that aligns is aligned in time **and** spectrum jointly.

Channel topographies are the **forward (mixing) patterns** —
``pinv(W_k @ pca.components_)`` — not the unmixing rows. See
:func:`src.analysis.wavelet_ica.iva_component_patterns` for why the two differ
(and why ``iva_g``'s internal whitening makes the difference large).

Produces all the analyses from the notebook, ranked by mean off-diagonal
``Sigma_N`` correlation, with the top ``--n_top`` and bottom ``--n_bottom``
components visualised separately (bottom = noise / no-alignment contrast).

``--n_bottom 0`` drops the bottom group altogether — only the ``_top`` figures
are written, which is what you want when ``--n_pca`` is small enough that
``--n_top`` already covers every component (the standard ASSR setting: ten
components, all ten shown, no contrast group left over):

  - component-ranking bar + per-subject channel-PCA scree
  - ISC matrices (+ cluster strips) of the **temporal** and **spectral**
    signed marginals — the two synchrony read-outs
  - LOO-ISC bars for four axes (spectro-temporal, temporal, spectral,
    channel) + a grouped by-axis summary
  - topomap mean/variance of the channel patterns, with every subject weighing
    the same: each pattern rescaled to unit L2 norm, then each subject put on a
    common range, so neither map is driven by per-subject gain or focality
  - the shared time × frequency source map (direct, no rank-1 product)
  - source mean ± √variance along time and along frequency
  - pair-space heatmaps (Time × Subject, Subject × Frequency,
    Subject × Channel)
  - per-subject mean loadings

Marginals are kept **signed** (no power rectification). Note the spectral
marginal (signed mean over time) is ~0 by construction, because
``zscore_by_time`` zeroes each ``(s, c, f)`` series' time-mean — so the
spectral matrix / Subject × Frequency heatmap may look like noise. This is
intentional, matching the notebook.

Output (canonical per-condition layout)::

    plots/05-channel-wavelet-iva-analysis/<Condition>_<MusicType>/
        broadband/iva_channel/pca_<n_pca>/   # default (no --band)
            iva_component_ranking_<label>.png
            ...
        bands/iva_channel/pca_<n_pca>/        # when --band <name> is given
            <band>_iva_component_ranking_<label>.png
            ...

Pass ``--band <name>`` to slice the cached broadband wavelets down to a
single frequency band before IVA. ``--n_pca`` reduces the **channel** axis,
so it must be ≤ the number of channels.

Pass ``--quality`` (ASSR-only; reproduces
``wavelet_iva_channel_quality.ipynb``) to additionally score every
``(subject, component)`` pair against **one reference pair**: the group PC1
channel loading and PC1 TF score map of a channel-PCA of the trial-averaged,
time-z-scored wavelet power — the reduction of ``scripts/run_assr_wavelet_pca.py``
applied to the IVA's own input tensor. The reference is deliberately in the same
modality as the decomposition; see :mod:`src.analysis.iva_quality` for why a
raw-voltage evoked topography cannot serve as a reference for channel patterns
recovered from power.

The loading is the **x** of every scatter. The **y** is either the paradigm's
rigid boxcar (0 before onset, 1 for the fixed
``iva_quality.RESP_DURATION_S`` = 500 ms stimulus window, 0 after) applied to a
frequency-reduced time course, or the correlation of the component's
onset-averaged ``(F, W)`` map with the reference map — whole, or restricted to one
frequency band. Onset-locking is what makes any of this work: a plain on/off
indicator over the whole recording fails because the ASSR stimulation is
continuous. The analysis requires stimulus onsets, so it is silently skipped for
experiments without them (e.g. psilo_music).

``--quality`` also stops the standard IVA figures above from being **ordered by
the ranking**: their panels are laid out in IVA component order instead, because
every quality figure numbers its panels that way (``IC <k+1>``) and a component
must sit in the same grid position everywhere for the two sets to be read
together. The ranking still decides *which* components a partial ``--n_top`` /
``--n_bottom`` selects, and each panel title reports its rank; only the panel
order changes. Emitted figures:

  - ``*_quality_reference_wavelet`` — the reference pair: group PC1 channel
    loading beside its PC1 TF score map
  - ``*_quality_topomaps`` — per-component group-mean channel topography, with
    the reference topography on its own scale
  - ``*_quality_tf_maps_wavelet_tf`` — per-component group-mean onset-averaged
    ``(F, W)`` TF map, with the reference map on its own scale
  - ``*_quality_onset_response_{pca,40hz}`` — per-component group-mean
    onset-averaged response with the fixed response window shaded
  - ``*_quality_scatter_time_{pca,40hz}`` / ``*_per_component`` — the topography
    correlation on x, the boxcar correlation on y, for both frequency reductions
    (first PC over frequency, and the single 40 Hz row)
  - ``*_quality_scatter_wavelet_tf`` / ``*_per_component`` — the same x with the
    whole-map TF correlation on y
  - ``*_quality_scatter_tf_1_10hz`` / ``*_quality_scatter_tf_30_50hz`` (each also
    ``*_per_component``) — the same x with the TF correlation measured over a
    single frequency band (:data:`src.analysis.iva_quality.TF_BANDS`): the
    1-10 Hz evoked/onset range and the 30-50 Hz band around the ASSR steady
    state. Same maps and component signs as the whole-map score — only the
    frequency rows the y-score sees change, so a component that matches the
    reference in one band only stops being diluted by the rest of the spectrum.
    A band outside the wavelet range is skipped with a warning
  - ``*_quality_participant_topomaps_<label>/`` — one participant-comparison
    figure per IVA component (1-based, in IVA order): every participant's
    channel topography side by side plus the group mean and the reference
  - ``*_quality_participant_tf_maps_<label>/`` — the TF counterpart: one figure
    per component holding every participant's onset-averaged ``(F, W)`` map plus
    the group mean and the reference, so a participant dragging a ``tf_corr``
    down can be found
  - ``*_quality_tf_maps_full_recording`` — the same components over the **whole
    recording** rather than the onset epoch, group-averaged under the same
    per-participant signs. Read it against ``*_iva_shared_time_frequency_*``
    above, which averages the raw sources under IVA's arbitrary per-pair
    polarity: the difference between the two is what that figure cancelled, and
    this one is where anything living between the stimuli shows up

Every score carries the same per-``(subject, component)`` sign as the maps it was
measured on (see below), so any y may be read against the one x; changing one axis
at a time separates the spatial question (which topography the pattern matches)
from the temporal one (the paradigm's window, the whole TF map, or one band of
it). That gives five scatter pairs:

  x = topography reference  ×  y = boxcar-PCA / boxcar-40Hz / TF map /
  TF 1-10 Hz / TF 30-50 Hz

Every ``*_quality_scatter_*`` figure is written into a
``*_quality_scatters_<label>/`` subdirectory, so the ten scatters do not bury the
reference and diagnostic figures they are meant to be read alongside.

Every figure that **compares, averages or scores participants** — the topography
and TF-map diagnostics, the onset-response diagnostics, the two
participant-comparison families, the whole-recording QC map and every scatter —
uses the per-``(subject, component)`` **sign**-resolved variant, since IVA fixes
none: each pair's topography is correlated with the reference topography and the
pair is flipped when that correlation is negative
(``iva_quality.anchor_signs_reference``, applied through
``IvaQualityResult.topomap_view`` / ``tf_view`` and the ``aligned=True`` scores of
``variants`` / ``wavelet_variants``). Without it the panels mix polarities and
every group mean cancels.

It is **one** sign per pair, and its TF map, its onset response and the
correlations measured on them all carry the same one: a pattern and its source
belong to a single decomposition, so flipping them apart would show something the
IVA never produced — and a scatter point scored under the opposite sign to the
map it stands for cannot be read against that map at all. Each figure names what
resolved the sign. Because the flip is fixed on the topography, every scatter
point sits in the right-hand half-plane on x; what a participant disagreeing with
the reference costs is read on the **y** — its source need not follow its
topography — and in the per-component sign-agreement statistics the log reports
for the unaligned arrays.

Every figure that averages across participants — the topography, onset-response
and TF-map diagnostics, and the group-mean panel of the participant figures —
first puts each participant on a common scale
(:func:`src.analysis.iva_quality.equalize_subject_influence`), so one loud
participant cannot stand in for the group; the group-mean panels of the
participant figures are drawn on their own annotated scale, an across-participant
mean being weaker than the individual maps by construction. Correlation scores
are scale-invariant and so unchanged by any of this.

The scoring lives in :mod:`src.analysis.iva_quality` and the rendering in
:mod:`src.visualization.iva_quality_plots`; this script only orchestrates them.

Usage::

    # broadband
    python scripts/run_wavelet_iva_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 100 --n_top 10 --n_bottom 5 --reuse_wavelets

    # alpha-band only
    python scripts/run_wavelet_iva_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --band alpha --n_pca 100 --n_top 10 --n_bottom 5 --reuse_wavelets

    # ASSR: 10 components, all ten shown, no bottom group
    python scripts/run_wavelet_iva_channel.py \\
        --experiment assr --condition Placebo \\
        --n_pca 10 --n_top 10 --n_bottom 0 --reuse_wavelets

    # …the same plus the decomposition-quality analysis
    python scripts/run_wavelet_iva_channel.py \\
        --experiment assr --condition Placebo \\
        --n_pca 10 --n_top 10 --n_bottom 0 --quality --reuse_wavelets
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from independent_vector_analysis import iva_g  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from mne.viz import plot_topomap  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    FREQUENCY_BANDS,
    _broadband_wavelet_4d,
    add_wavelet_grid_args,
    analyzers_to_datasets,
    load_analyzers,
    participant_labels,
    resolve_wavelet_dir,
)
from src.analysis import iva_quality  # noqa: E402
from src.analysis.iva_quality import (  # noqa: E402
    compute_iva_quality,
    equalize_subject_influence,
)
from src.analysis.wavelet_ica import (  # noqa: E402
    align_iva_component_signs,
    iva_component_patterns,
    normalize_patterns_per_subject,
    zscore_by_time,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    SpectrumTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    FrequencyBandNames,
    MusicTypeVariants,
)
from src.visualization import iva_quality_plots as qplots  # noqa: E402

_logger = logging.getLogger(__name__)

_STAGE_DIR = "05-channel-wavelet-iva-analysis"

# ---------------------------------------------------------------------------
# Cluster-analysis constants and helpers (shared with the ICA scripts)
# ---------------------------------------------------------------------------

ISC_CLUSTER_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

_GROUP_PALETTE = list(plt.colormaps["tab10"].colors)
_CLUSTER_CMAP = ListedColormap(["#dddddd"] + _GROUP_PALETTE)
_CLUSTER_NORM = BoundaryNorm(
    np.arange(-0.5, len(_GROUP_PALETTE) + 1.5, 1.0), _CLUSTER_CMAP.N
)

# LOO-ISC axes shown side by side in the by-axis summary.
_LOO_AXIS_PALETTE = ("slategray", "steelblue", "darkorange", "seagreen")


def _cluster_grid(corr_mat: np.ndarray, n_subjects: int) -> np.ndarray:
    """(T, S) grid of within-row cluster IDs. Singletons → 0, groups → 1, 2, ...."""
    grid = np.zeros((len(ISC_CLUSTER_THRESHOLDS), n_subjects), dtype=int)
    for t_idx, thr in enumerate(ISC_CLUSTER_THRESHOLDS):
        adj = (corr_mat >= thr) & ~np.eye(n_subjects, dtype=bool)
        _, labels = connected_components(adj, directed=False)
        counts = Counter(labels.tolist())
        next_group = 1
        group_map: dict[int, int] = {}
        for s in range(n_subjects):
            lab = int(labels[s])
            if counts[lab] == 1:
                grid[t_idx, s] = 0
            else:
                if lab not in group_map:
                    group_map[lab] = next_group
                    next_group += 1
                grid[t_idx, s] = group_map[lab]
    return grid


def _annotate_cluster_grid(ax, grid: np.ndarray) -> None:
    """Overlay numeric cluster IDs (>0) on a cluster-strip imshow."""
    for ti in range(grid.shape[0]):
        for sj in range(grid.shape[1]):
            val = int(grid[ti, sj])
            if val > 0:
                ax.text(
                    sj,
                    ti,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    weight="bold",
                )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the channel-as-independent IVA decomposition on wavelet power "
            "(samples = time × frequency, mixing = channels). Reproduces the "
            "analyses from wavelet_iva_channel.ipynb for the top/bottom-ranked "
            "components."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentNames],
        default=ExperimentNames.PSILO_MUSIC.value,
        help="Which experiment dataset to analyse.",
    )
    parser.add_argument(
        "--condition",
        choices=[c.value for c in ConditionVariants],
        default=ConditionVariants.PLACEBO.value,
        help="Experimental condition.",
    )
    parser.add_argument(
        "--music_type",
        nargs="+",
        choices=[mt.value for mt in MusicTypeVariants],
        default=None,
        help=(
            "One or more music types to analyse. When omitted, defaults to "
            "CLASSIC + PSYTRANCE for the psilo_music experiment and ASSR for "
            "the assr experiment."
        ),
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=50,
        help=(
            "Per-subject PCA dim over CHANNELS before IVA-G (square mixing "
            "requirement), i.e. the number of IVA components. Must be ≤ the "
            "number of channels."
        ),
    )
    parser.add_argument(
        "--n_top",
        type=int,
        default=10,
        help=(
            "Number of top-ranked IVA components (by mean off-diagonal "
            "Sigma_N correlation) to visualise."
        ),
    )
    parser.add_argument(
        "--n_bottom",
        type=int,
        default=5,
        help=(
            "Number of bottom-ranked IVA components to visualise alongside "
            "the top components as a noise contrast. 0 skips the bottom group "
            "entirely (no '_bottom' figures), which is what you want once "
            "--n_top already covers every component."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for per-subject PCA and IVA W_init.",
    )
    parser.add_argument(
        "--iva_opt_approach",
        choices=["gradient", "newton", "quasi"],
        default="newton",
        help="IVA-G optimisation method.",
    )
    parser.add_argument(
        "--iva_max_iter",
        type=int,
        default=1024,
        help="Maximum IVA-G iterations.",
    )
    parser.add_argument(
        "--iva_w_diff_stop",
        type=float,
        default=1e-6,
        help="IVA-G convergence threshold on |ΔW|.",
    )
    add_wavelet_grid_args(parser)
    parser.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help="Directory for cached wavelet tensors.",
    )
    parser.add_argument(
        "--reuse_wavelets",
        action="store_true",
        help="Reuse cached wavelets from --wavelet_data_dir when available.",
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        help="Load raw .fif files, resample, stack, and save .npy caches.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for data loading.",
    )
    parser.add_argument(
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the cached broadband wavelet "
            "tensor is sliced to the band's frequency range before IVA, and "
            "plots are written to 'bands/iva_channel/' with a '<band>_' prefix."
        ),
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help=(
            "ASSR-only: additionally compute the decomposition-quality analysis "
            "against both reference families. Raw voltage: reference topomap, "
            "per-component onset-averaged responses and topomap-vs-time scatters, "
            "the time reference being a rigid 500 ms boxcar starting at each "
            "stimulus onset. Wavelet: the PC1 channel loading and PC1 TF score "
            "map of the trial-averaged wavelet power (as in run_assr_wavelet_pca), "
            "with a topomap-vs-TF-map scatter. Requires stimulus onsets, so it is "
            "silently skipped for experiments without them."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Base directory for output plots. Defaults to project plots/ root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_pca_scree(pca_evr: np.ndarray, *, label: str, save_path: Path) -> None:
    """Across-subject per-subject channel-PCA scree + cumulative variance."""
    pca_evr_mean = pca_evr.mean(axis=0)
    pca_evr_std = pca_evr.std(axis=0)
    pca_evr_cum_mean = np.cumsum(pca_evr_mean)
    n_subjects, n_pca = pca_evr.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(1, n_pca + 1)
    axes[0].bar(xs, pca_evr_mean, yerr=pca_evr_std, color="steelblue", capsize=2)
    axes[0].set_xlabel("PCA component (channels)")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"Channel PCA Scree (mean ± std across subjects) — {label}")

    for k in range(n_subjects):
        axes[1].plot(xs, np.cumsum(pca_evr[k]), lw=0.6, alpha=0.5, color="gray")
    axes[1].plot(xs, pca_evr_cum_mean, "o-", color="coral", label="mean cumulative")
    axes[1].axhline(0.9, ls="--", color="gray", label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance — {label}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_ranking(
    rank_score: np.ndarray,
    top_indices: list[int],
    bottom_indices: list[int],
    *,
    label: str,
    save_path: Path,
) -> None:
    """Bar chart of all per-IC ranking scores with top (blue) / bottom (red)."""
    n_pca = len(rank_score)
    colors = ["lightgray"] * n_pca
    for k in top_indices:
        colors[k] = "steelblue"
    for k in bottom_indices:
        colors[k] = "firebrick"

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_pca), 4.5))
    xs = np.arange(n_pca)
    ax.bar(xs, rank_score, color=colors)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k + 1}" for k in range(n_pca)], fontsize=7)
    ax.set_xlabel("IVA component index")
    ax.set_ylabel("Mean off-diagonal Sigma_N correlation")
    highlighted = f"top {len(top_indices)} = blue"
    if bottom_indices:
        highlighted += f", bottom {len(bottom_indices)} = red"
    ax.set_title(
        f"Component Ranking by Sigma_N Mean Off-Diagonal Correlation — "
        f"{label}  ({highlighted})"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_isc_grid(
    corr_per_comp: np.ndarray,
    *,
    labels: list[str],
    fig_title: str,
    save_path: Path,
) -> None:
    """Per-IC subject × subject correlation grid + cluster strip beneath each."""
    n_show, n_subjects, _ = corr_per_comp.shape
    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(2.6 * n_show, 5.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )
    if n_show == 1:
        axes = axes.reshape(2, 1)

    im_corr = None
    for i in range(n_show):
        corr_mat = corr_per_comp[i]
        ax_top = axes[0, i]
        im_corr = ax_top.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_top.set_xticks(range(n_subjects))
        ax_top.set_yticks(range(n_subjects))
        ax_top.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_top.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_top.set_title(labels[i], fontsize=8)

        ax_bot = axes[1, i]
        grid = _cluster_grid(corr_mat, n_subjects)
        ax_bot.imshow(grid, cmap=_CLUSTER_CMAP, norm=_CLUSTER_NORM, aspect="auto")
        _annotate_cluster_grid(ax_bot, grid)
        ax_bot.set_xticks(range(n_subjects))
        ax_bot.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_bot.set_yticks(range(len(ISC_CLUSTER_THRESHOLDS)))
        ax_bot.set_yticklabels(
            [f"r≥{thr}" for thr in ISC_CLUSTER_THRESHOLDS], fontsize=8
        )
        if i == 0:
            ax_bot.set_ylabel("Threshold")

    fig.suptitle(fig_title, fontsize=12)
    if im_corr is not None:
        fig.colorbar(im_corr, ax=axes[0, -1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _loo_isc(view: np.ndarray) -> np.ndarray:
    """LOO-ISC per component over a (K, S, L) view → (K, S)."""
    n_show, n_subjects, _ = view.shape
    loo = np.zeros((n_show, n_subjects))
    for k in range(n_show):
        vecs = view[k]  # (S, L)
        for s in range(n_subjects):
            others_mean = np.delete(vecs, s, axis=0).mean(axis=0)
            loo[k, s] = float(pearsonr(vecs[s], others_mean)[0])
    return loo


def _plot_loo_isc_bar(
    loo: np.ndarray,
    *,
    labels: list[str],
    n_top: int,
    axis_name: str,
    label: str,
    save_path: Path,
) -> None:
    """LOO-ISC bar (mean ± std across subjects) for one axis; top + bottom."""
    loo_mean = loo.mean(axis=1)
    loo_std = loo.std(axis=1)
    n_show = len(loo_mean)
    bar_colors: list[str] = []
    for i, m in enumerate(loo_mean):
        if i < n_top:
            bar_colors.append("firebrick" if m < 0 else "steelblue")
        else:
            bar_colors.append("salmon" if m < 0 else "lightsteelblue")

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * n_show), 5.0))
    xs = np.arange(n_show)
    ax.bar(xs, loo_mean, yerr=loo_std, color=bar_colors, capsize=4)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    if 0 < n_top < n_show:
        ax.axvline(
            n_top - 0.5,
            ls="--",
            lw=0.9,
            color="black",
            alpha=0.5,
            label=f"top {n_top} | bottom {n_show - n_top}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Per-IC Mean LOO-ISC — {axis_name} — {label}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_axis_summary(
    loo_means_by_axis: dict[str, np.ndarray],
    *,
    top_labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Grouped bar: mean LOO-ISC per axis, side by side, for TOP components."""
    axis_names = list(loo_means_by_axis.keys())
    n_axes = len(axis_names)
    n_top = len(top_labels)
    width = 0.8 / n_axes

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * n_top), 5.0))
    xs = np.arange(n_top)
    for j, name in enumerate(axis_names):
        offset = (j - (n_axes - 1) / 2) * width
        ax.bar(
            xs + offset,
            loo_means_by_axis[name][:n_top],
            width=width,
            color=_LOO_AXIS_PALETTE[j % len(_LOO_AXIS_PALETTE)],
            label=name,
        )
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(top_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"LOO-ISC by Axis — TOP {n_top} components — {label}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap_mean_var(
    chan_loading: np.ndarray,
    info,
    n_channels: int,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC mean (signed) and variance topomaps across subjects.

    ``chan_loading`` shape ``(S, n_show, C)`` — the channel patterns.

    Every subject enters the mean and the variance with the same weight, in two
    steps. First each pattern is rescaled to unit L2 norm: ``iva_g`` fixes the
    source scale but not the pattern scale, so unnormalised patterns differ
    across subjects by a subject-specific gain that would let the loudest
    subjects dominate the mean map and would leak amplitude differences into the
    variance map (see
    :func:`src.analysis.wavelet_ica.normalize_patterns_per_subject`). Then each
    subject's whole set of patterns is put on a common *range* by
    :func:`src.analysis.iva_quality.equalize_subject_influence` — unit L2 norm
    still leaves a focal pattern with a larger peak than a diffuse one, so a
    subject with focal topographies would otherwise steer both maps.
    """
    chan_loading = equalize_subject_influence(
        normalize_patterns_per_subject(chan_loading)
    )
    chan_mean = chan_loading.mean(axis=0)  # (n_show, C)
    chan_var = chan_loading.var(axis=0)  # (n_show, C)
    n_show = len(labels)

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(topo_info.ch_names):
        topo_info = mne.pick_info(topo_info, list(range(n_channels)))

    fig, axes = plt.subplots(2, n_show, figsize=(3.0 * n_show, 7.0))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        vlim_m = float(np.percentile(np.abs(chan_mean[i]), 99))
        if vlim_m == 0.0:
            vlim_m = 1e-12
        im_m, _ = plot_topomap(
            chan_mean[i],
            topo_info,
            axes=axes[0, i],
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim_m, vlim_m),
        )
        axes[0, i].set_title(labels[i], fontsize=8)
        fig.colorbar(im_m, ax=axes[0, i], fraction=0.046, pad=0.04)

        vlim_v = float(np.percentile(chan_var[i], 99))
        if vlim_v == 0.0:
            vlim_v = 1e-12
        im_v, _ = plot_topomap(
            chan_var[i],
            topo_info,
            axes=axes[1, i],
            show=False,
            cmap="viridis",
            vlim=(0.0, vlim_v),
        )
        fig.colorbar(im_v, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.text(
        0.01,
        0.75,
        "Mean across subjects (equal weight)",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.25,
        "Variance across subjects (equal weight)",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.suptitle(
        f"Mean and Variance Topomaps Across Subjects "
        f"(equal-weighted, unit-norm patterns) — {label}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tf_map_direct(
    mean_sources: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC subject-averaged time × frequency source map (direct, no product).

    ``mean_sources`` shape ``(K, F, T)`` — already sliced + subject-averaged.
    """
    n_show = len(labels)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = mean_sources[i]  # (F, T)
        vlim_i = max(float(np.percentile(np.abs(data_i), 99)), 1e-12)
        mesh = ax.pcolormesh(
            time,
            freqs,
            data_i,
            cmap="RdBu_r",
            vmin=-vlim_i,
            vmax=vlim_i,
            shading="auto",
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(
            f"{labels[i]} — Shared Time × Frequency source",
            fontsize=10,
        )
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Per-IC Shared Time × Frequency Source (mean across subjects) — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_profile_mean_var(
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
    x_vals: np.ndarray,
    *,
    labels: list[str],
    x_label: str,
    axis_name: str,
    label: str,
    save_path: Path,
) -> None:
    """Per-IC mean ± √variance over a 1-D axis (time or frequency).

    ``mean_arr`` / ``std_arr`` shape ``(n_show, L)`` — already sliced.
    """
    n_show = len(labels)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x_vals, mean_arr[i], lw=0.9, color="darkorange", label="mean")
        ax.fill_between(
            x_vals,
            mean_arr[i] - std_arr[i],
            mean_arr[i] + std_arr[i],
            alpha=0.25,
            color="darkorange",
            label="± √variance",
        )
        ax.set_ylabel(labels[i], fontsize=8)
        ax.set_title(f"{labels[i]} — {axis_name} Mean & Variance", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(
        f"Per-IC {axis_name} Source — Mean & Variance Across Subjects — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _safe_vlim(arr: np.ndarray) -> float:
    return max(float(np.percentile(np.abs(arr), 99)), 1e-12)


def _plot_pairmap_time_subject(
    time_subj: np.ndarray,
    time: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC Time × Subject heatmap stack (signed temporal marginal)."""
    n_show, n_subjects, _ = time_subj.shape
    vlim = _safe_vlim(time_subj)
    subject_labels = [f"S{s + 1}" for s in range(n_subjects)]

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time,
            np.arange(n_subjects),
            time_subj[i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels(subject_labels, fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(f"{labels[i]} — Time × Subject", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="source")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time × Subject per IC (temporal marginal) — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairmap_subject_x(
    data_per_comp: np.ndarray,
    y_values: np.ndarray,
    *,
    y_label: str,
    variant_name: str,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC heatmap stack with subject on x and ``y_values`` on y."""
    n_show, _, n_subjects = data_per_comp.shape
    vlim = _safe_vlim(data_per_comp)
    subjects_idx = np.arange(n_subjects)
    subject_labels = [f"S{s + 1}" for s in subjects_idx]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 5.0), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            subjects_idx,
            y_values,
            data_per_comp[i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_xticks(subjects_idx)
        ax.set_xticklabels(subject_labels, fontsize=8)
        ax.set_xlabel("Subject")
        ax.set_title(labels[i], fontsize=8)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="loading")
    axes[0].set_ylabel(y_label)
    fig.suptitle(f"{variant_name} per IC — {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_loadings(
    subject_loadings: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC horizontal bar of mean |value| over samples, one bar per subject."""
    n_subjects, n_show = subject_loadings.shape
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.barh(range(n_subjects), subject_loadings[:, i], color="darkorange")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_xlabel("|score|")
        ax.set_title(labels[i], fontsize=8)
    axes[0].set_ylabel("Subject")
    fig.suptitle(
        f"Per-Subject Mean Loading per Component — {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quality analysis (ASSR-only): topomap & time correlation scatterplots
#
# Only the orchestration lives here — it runs when ``--quality`` is passed. The
# scoring is in ``src/analysis/iva_quality.py`` (pure) and the rendering in
# ``src/visualization/iva_quality_plots.py``.
# ---------------------------------------------------------------------------


def _run_quality(
    iva_components: np.ndarray,
    iva_sources: np.ndarray,
    freqs: np.ndarray,
    info,
    n_channels: int,
    onsets: np.ndarray | None,
    sfreq: float,
    wavelet_power_z: np.ndarray,
    *,
    subject_ids: list[str] | None,
    label: str,
    out_dir: Path,
    prefix: str,
) -> None:
    """Compute + plot the quality analysis (ASSR-only).

    Scores every ``(subject, component)`` pair against the wavelet reference pair
    — its PC1 channel loading on x, the paradigm's boxcar or its PC1 TF score map
    (whole or band-limited) on y — and emits the scatters, the diagnostics and the
    per-participant breakdowns.

    Skipped with a warning when its inputs are unavailable (no stimulus onsets or
    no MNE info), so non-ASSR experiments pass through harmlessly.
    """
    if onsets is None or len(onsets) == 0:
        _logger.warning(
            f"[{label}] Quality analysis skipped: no stimulus onsets (ASSR-only)."
        )
        return
    if info is None:
        _logger.warning(
            f"[{label}] Quality analysis skipped: no MNE info for topomaps."
        )
        return

    n_subjects, n_pca, _ = iva_components.shape
    comp_indices = list(range(n_pca))
    if subject_ids is None or len(subject_ids) != n_subjects:
        # No participant metadata — fall back to the subject index, marked as
        # such so it is never mistaken for a real participant ID.
        subject_ids = [f"#{s:03d}" for s in range(n_subjects)]

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    all_ch_names = list(topo_info["ch_names"])
    iva_ch_names = (
        all_ch_names[:n_channels] if n_channels < len(all_ch_names) else all_ch_names
    )

    try:
        quality = compute_iva_quality(
            iva_components,
            iva_sources,
            freqs,
            onsets,
            sfreq,
            iva_ch_names,
            wavelet_power_z,
        )
    except Exception as exc:  # noqa: BLE001 - skip quality gracefully
        _logger.warning(f"[{label}] Quality analysis failed ({exc}); skipping.")
        return

    wav_cons = quality.wavelet_consistency
    _logger.info(
        f"[{label}] Quality: reference anchored on {quality.anchor_channel}; "
        f"{wav_cons.n_agreeing}/{wav_cons.n_subjects} PC1 topographies agree "
        f"(median pairwise r={wav_cons.median_pairwise_r:+.2f}, "
        f"weakest subject r={wav_cons.min_subject_r:+.2f})."
    )
    if wav_cons.n_agreeing < wav_cons.n_subjects:
        _logger.warning(
            f"[{label}] Quality: {wav_cons.n_subjects - wav_cons.n_agreeing} "
            f"subject(s) anti-correlate with the group PC1 topography — a "
            f"topographic outlier, not a sign problem. The reference averages them "
            f"in, so inspect it before trusting the scores."
        )
    win = quality.n_epoch_pre + quality.n_epoch_post
    _logger.info(
        f"[{label}] Quality onset epoch: {win} samples ({quality.n_epoch_pre} pre, "
        f"{quality.n_epoch_post} post); rigid response window "
        f"{quality.resp_duration_s * 1000:.0f} ms."
    )
    if not quality.have_40hz:
        _logger.warning(
            f"[{label}] Quality: {iva_quality.ASSR_FREQ:.0f} Hz outside freq range "
            f"[{freqs.min():.1f}, {freqs.max():.1f}] Hz; skipping 40 Hz variant."
        )
    # Band-limited TF variants need bins inside their interval; a band-sliced run
    # (--band alpha) reaches some of them at most, which is a skipped variant
    # rather than an error. What each available band actually covers is logged
    # too: a partly covered band is scored over the bins it has, so its figures
    # are labelled with a wider range than they were measured on.
    missing_bands = [
        band_name
        for band_name, _fmin, _fmax, tag in iva_quality.TF_BANDS
        if tag not in quality.tf_corr_bands
    ]
    if missing_bands:
        _logger.warning(
            f"[{label}] Quality: no frequency bin inside {', '.join(missing_bands)} "
            f"within [{freqs.min():.1f}, {freqs.max():.1f}] Hz; skipping "
            f"{'that' if len(missing_bands) == 1 else 'those'} band TF variant(s)."
        )
    for band_name, fmin, fmax, tag in iva_quality.TF_BANDS:
        if tag not in quality.tf_corr_bands:
            continue
        band_freqs = quality.freqs[
            iva_quality.frequency_band_mask(quality.freqs, fmin, fmax)
        ]
        _logger.info(
            f"[{label}] Quality TF band [{band_name}]: scored over "
            f"{len(band_freqs)} bin(s), {band_freqs.min():.1f}-"
            f"{band_freqs.max():.1f} Hz."
        )
    # Measured on the stored arrays, before the per-participant flip below: the
    # question here is how far the participants disagree with each other under the
    # one-flip-per-component orientation, which the aligned scores the figures use
    # have by construction resolved on x.
    sign_checks = [
        ("topomap", quality.topo_corr),
        ("time-PCA", quality.time_corr_pca),
        ("TF-map", quality.tf_corr),
    ]
    # The band variants are separate y-axes, so their sign agreement is worth the
    # same check: a band where the subjects split 50/50 has no group statement.
    sign_checks += [
        (f"TF-map {band_name}", corr)
        for band_name, corr, _tag in quality.wavelet_variants()
        if band_name is not None
    ]
    for name, arr in sign_checks:
        frac = iva_quality.sign_agreement(arr)
        _logger.info(
            f"[{label}] Quality sign agreement [{name}]: mean "
            f"{frac.mean() * 100:.0f}%, min {frac.min() * 100:.0f}%."
        )
    n_flipped = int((quality.sign_per_comp < 0).sum())
    _logger.info(
        f"[{label}] Quality: oriented {n_flipped}/{n_pca} components so the mean "
        f"topomap correlation is >= 0."
    )
    # The per-participant sign, for every figure that compares, averages or scores
    # participants. One per (subject, component) pair, worn by its topomap and its
    # TF map alike. A large flip count is expected, not a warning: IVA fixes no
    # per-pair polarity at all, so roughly half of them start out inverted.
    n_pairs = n_subjects * n_pca
    _logger.info(
        f"[{label}] Quality per-participant sign: anchored on "
        f"{quality.subject_anchor_name}; flipped "
        f"{int((quality.sign_per_subject < 0).sum())}/{n_pairs} "
        f"(subject, component) pairs — the same flip for the topomaps, the "
        f"onset-averaged TF maps, the whole-recording QC map and the scatter "
        f"scores measured on them."
    )
    agreement = quality.anchor_agreement()
    _logger.info(
        f"[{label}] Quality: under that sign the driven response "
        f"({quality.tf_anchor_name}) reads positive for {agreement * 100:.0f}% of "
        f"the (subject, component) pairs. Where it does not, that pair's "
        f"topography and its source disagree — the group-mean TF map is then "
        f"weaker than the group-mean topography for a real reason."
    )
    # How much the equal weighting matters: the loudest/quietest ratio is the
    # factor by which an unweighted group mean would have over-represented one
    # participant, and it is why an unweighted mean map reads flat.
    for name, arr in (
        ("patterns", quality.patterns),
        ("onset TF maps", quality.onset_tf),
    ):
        scales = iva_quality.subject_scales(arr, comp_indices)
        _logger.info(
            f"[{label}] Quality per-participant amplitude [{name}]: "
            f"loudest/quietest = {scales.max() / scales.min():.1f}x; equalised "
            f"before every across-participant mean."
        )

    qplots.plot_wavelet_reference(
        quality.ref_topo,
        quality.ref_tf,
        info,
        n_channels,
        quality.freqs,
        quality.epoch_times,
        label=label,
        resp_duration_s=quality.resp_duration_s,
        assr_freq=iva_quality.ASSR_FREQ,
        consistency=quality.wavelet_consistency,
        save_path=out_dir / f"{prefix}quality_reference_wavelet_{label}.png",
    )
    plt.close("all")

    # Every figure below compares or averages participants, so it gets the
    # sign-aligned maps: IVA fixes no per-(subject, component) polarity, and the
    # one-flip-per-component orientation of the stored arrays leaves participants
    # free to disagree with each other — mixed polarities in the panels, and a
    # group mean that cancels. Every view applies the *same* per-pair sign, so a
    # participant's topography, its TF map and its onset response can be read side
    # by side — and so can the scatter point scoring them, which is measured on the
    # same flipped pair (quality.variants / .wavelet_variants with aligned=True).
    topo_patterns, topo_corr_aligned = quality.topomap_view()
    onset_tf_aligned, tf_corr_aligned, _tf_bands_aligned = quality.tf_view()
    sign_note = quality.subject_anchor_name

    # The group view that comes before any per-participant breakdown: one panel
    # per component holding the across-participant mean topography, with the
    # reference beside them — which components resemble the reference at all. One
    # figure: there is one reference topography, and the maps do not depend on the
    # y-axis. The panel score uses the TF-map y, the reference pair's own, so the
    # whole figure comes from one reduction.
    qplots.plot_topomap_diagnostic(
        topo_patterns,
        quality.ref_topo,
        topo_corr_aligned,
        tf_corr_aligned,
        info,
        n_channels,
        comp_indices,
        variant_name=qplots.TF_VARIANT_NAME,
        label=label,
        alignment_note=sign_note,
        save_path=out_dir / f"{prefix}quality_topomaps_{label}.png",
    )
    plt.close("all")

    # One participant-comparison figure per IVA component. Independent of the
    # time-reduction variant, so they are emitted once.
    topo_root = out_dir / f"{prefix}quality_participant_topomaps_{label}"
    written = qplots.plot_participant_topomaps(
        topo_patterns,
        quality.ref_topo,
        topo_corr_aligned,
        info,
        n_channels,
        comp_indices,
        subject_ids,
        label=label,
        root_dir=topo_root,
        prefix=prefix,
        alignment_note=sign_note,
    )
    _logger.info(
        f"[{label}] Quality: wrote {len(written)} participant-comparison figures "
        f"(one per component, {n_pca} components) to {topo_root}"
    )

    # The same per-participant breakdown in the time-frequency domain: one figure
    # per component, every participant's onset-averaged (F, W) map side by side.
    # Also variant-independent, so also emitted once.
    tf_root = out_dir / f"{prefix}quality_participant_tf_maps_{label}"
    written_tf = qplots.plot_participant_tf_maps(
        onset_tf_aligned,
        quality.ref_tf,
        tf_corr_aligned,
        quality.freqs,
        quality.epoch_times,
        comp_indices,
        subject_ids,
        resp_duration_s=quality.resp_duration_s,
        assr_freq=iva_quality.ASSR_FREQ,
        label=label,
        root_dir=tf_root,
        prefix=prefix,
        alignment_note=sign_note,
    )
    _logger.info(
        f"[{label}] Quality: wrote {len(written_tf)} participant TF-map figures "
        f"(one per component, {n_pca} components) to {tf_root}"
    )

    colors = qplots.component_colors(comp_indices)
    # Every scored figure below — the scatters, the onset diagnostic and the panel
    # selection — is measured on the sign-aligned scores, the same per-pair flip
    # the maps above are drawn under. A score taken under the stored
    # one-flip-per-component orientation would rate a pair by a polarity no figure
    # shows, and its across-participant mean would cancel the same way a group-mean
    # map does.
    time_variants = quality.variants(aligned=True)
    tf_variants = quality.wavelet_variants(aligned=True)
    # One panel selection for every per-component grid, so a component keeps the
    # same grid position in all of them and the figures stay comparable. It only
    # bites above the MAX_PANELS cap, which the usual --n_pca never reaches.
    panels = qplots.panel_indices(topo_corr_aligned, time_variants[0][1], comp_indices)
    if len(panels) < n_pca:
        _logger.info(
            f"[{label}] Quality: {n_pca} components > {qplots.MAX_PANELS} panel "
            f"cap; per-component grid shows the top {len(panels)} by score."
        )

    # Every scatter (both families, every x/y pairing, combined and
    # per-component) goes into one subdirectory: there are ten of them against a
    # handful of reference and diagnostic figures, and mixed together in out_dir
    # the diagnostics get lost.
    scatter_dir = out_dir / f"{prefix}quality_scatters_{label}"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    # Every scatter shares one x — the reference topography — so a y can be
    # swapped without moving the other axis. The boxcar y-axes come with a
    # time-course diagnostic; the TF y-axes share the TF diagnostic below.
    for variant_name, time_corr, onset_avg, tag in time_variants:
        qplots.plot_onset_diagnostic(
            onset_avg,
            quality.resp_duration_s,
            topo_corr_aligned,
            time_corr,
            quality.epoch_times,
            comp_indices,
            colors,
            variant_name=variant_name,
            label=label,
            save_path=out_dir / f"{prefix}quality_onset_response_{tag}_{label}.png",
        )
        lim = qplots.axis_limit(topo_corr_aligned, time_corr, comp_indices)
        qplots.plot_quality_scatter(
            topo_corr_aligned,
            time_corr,
            comp_indices,
            colors,
            lim,
            variant_name=variant_name,
            label=label,
            save_path=scatter_dir / f"{prefix}quality_scatter_time_{tag}_{label}.png",
        )
        qplots.plot_quality_scatter_per_component(
            topo_corr_aligned,
            time_corr,
            panels,
            colors,
            subject_ids,
            lim,
            variant_name=variant_name,
            label=label,
            save_path=scatter_dir
            / f"{prefix}quality_scatter_time_{tag}_per_component_{label}.png",
        )
        plt.close("all")

    # The TF y-axes: the reference's own score map, whole and band-limited.
    qplots.plot_tf_diagnostic(
        onset_tf_aligned,
        quality.ref_tf,
        topo_corr_aligned,
        tf_corr_aligned,
        quality.freqs,
        quality.epoch_times,
        comp_indices,
        resp_duration_s=quality.resp_duration_s,
        assr_freq=iva_quality.ASSR_FREQ,
        label=label,
        alignment_note=sign_note,
        save_path=out_dir
        / f"{prefix}quality_tf_maps_{qplots.TF_VARIANT_TAG}_{label}.png",
    )
    plt.close("all")

    # The same components over the whole recording rather than the onset epoch,
    # under the same per-participant signs. Figure (f) above shows this without
    # them — the raw sources averaged under IVA's arbitrary per-pair polarity — so
    # the pair says how much of that figure was cancellation, and this one is
    # where anything living between the stimuli shows up.
    qplots.plot_full_tf_maps(
        quality.full_tf_mean,
        quality.freqs,
        quality.full_times,
        comp_indices,
        assr_freq=iva_quality.ASSR_FREQ,
        label=label,
        onset_times=np.asarray(onsets, dtype=float) / sfreq,
        alignment_note=sign_note,
        save_path=out_dir / f"{prefix}quality_tf_maps_full_recording_{label}.png",
    )
    plt.close("all")

    # One scatter pair per TF-map variant: the whole map, then each band-limited
    # score (iva_quality.TF_BANDS). They share the x-axis, the maps and the
    # component signs, so the only thing that moves between them is which
    # frequency rows the y-score was measured over.
    tf_name = qplots.TF_VARIANT_NAME
    for band_name, y_corr, tag in tf_variants:
        variant = tf_name if band_name is None else f"{tf_name}, {band_name}"
        y_label = (
            qplots.TF_AXIS_LABEL
            if band_name is None
            else f"{qplots.TF_AXIS_LABEL} ({band_name})"
        )
        tf_lim = qplots.axis_limit(topo_corr_aligned, y_corr, comp_indices)
        qplots.plot_quality_scatter(
            topo_corr_aligned,
            y_corr,
            comp_indices,
            colors,
            tf_lim,
            variant_name=variant,
            label=label,
            ylabel=y_label,
            save_path=scatter_dir / f"{prefix}quality_scatter_{tag}_{label}.png",
        )
        qplots.plot_quality_scatter_per_component(
            topo_corr_aligned,
            y_corr,
            panels,
            colors,
            subject_ids,
            tf_lim,
            variant_name=variant,
            label=label,
            ylabel=y_label,
            save_path=scatter_dir
            / f"{prefix}quality_scatter_{tag}_per_component_{label}.png",
        )
        plt.close("all")

    _logger.info(
        f"[{label}] Quality analysis: plots saved to {out_dir} "
        f"(scatterplots in {scatter_dir.name}/)"
    )


# ---------------------------------------------------------------------------
# Main IVA pipeline (channel as the independent / mixing axis)
# ---------------------------------------------------------------------------


def _run_iva(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    label: str,
    n_pca: int,
    n_top: int,
    n_bottom: int,
    random_state: int,
    iva_opt_approach: str,
    iva_max_iter: int,
    iva_w_diff_stop: float,
    save_dir: Path,
    band: str | None,
    quality: bool = False,
    stimulus_onsets: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
) -> None:
    """Run the channel-as-mixing IVA pipeline and save all plots."""
    pca_subdir = f"pca_{n_pca}"
    if band is None:
        out_dir = (
            save_dir / SpectrumTypeVariants.BROADBAND.value / "iva_channel" / pca_subdir
        )
        prefix = ""
    else:
        out_dir = (
            save_dir / SpectrumTypeVariants.BANDS.value / "iva_channel" / pca_subdir
        )
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] IVA-channel: {data_4d.shape}  sfreq={sfreq} Hz")

    if n_pca > n_channels:
        raise ValueError(
            f"--n_pca ({n_pca}) must be ≤ n_channels ({n_channels}); "
            f"PCA reduces the channel axis in this variant."
        )
    n_show = n_top + n_bottom
    if n_show > n_pca:
        raise ValueError(
            f"--n_top + --n_bottom ({n_show}) must be ≤ --n_pca ({n_pca}); "
            f"IVA outputs exactly N_PCA components."
        )

    # Step 1 — Z-score along time, then per-subject reshape (S, C, F*T).
    bb_z = zscore_by_time(data_4d)
    n_samples_ft = n_freqs * n_times
    X_subjects = bb_z.reshape(n_subjects, n_channels, n_samples_ft)
    _logger.info(f"[{label}] Per-subject reshape: {X_subjects.shape}  (S, C, F*T)")

    # Step 2 — Per-subject PCA over channels → IVA input (N_PCA, F*T, S).
    pcas: list[PCA] = []
    pca_scores = np.zeros((n_subjects, n_pca, n_samples_ft))
    pca_evr = np.zeros((n_subjects, n_pca))
    for k in range(n_subjects):
        subj_matrix = X_subjects[k].T  # (F*T, C)
        pca = PCA(n_components=n_pca, random_state=random_state)
        scores = pca.fit_transform(subj_matrix)
        pcas.append(pca)
        pca_scores[k] = scores.T
        pca_evr[k] = pca.explained_variance_ratio_
    X_pca = np.ascontiguousarray(pca_scores.transpose(1, 2, 0))
    _logger.info(
        f"[{label}] Channel PCA: X_pca={X_pca.shape}, "
        f"mean retained variance = {pca_evr.sum(axis=1).mean() * 100:.1f}%"
    )

    # Step 3 — Run IVA-G.
    rng = np.random.default_rng(random_state)
    W_init = rng.standard_normal((n_pca, n_pca, n_subjects))
    W, cost, Sigma_N, _isi = iva_g(
        X_pca,
        opt_approach=iva_opt_approach,
        whiten=True,
        verbose=False,
        W_init=W_init,
        max_iter=iva_max_iter,
        W_diff_stop=iva_w_diff_stop,
    )
    _logger.info(f"[{label}] IVA-G: iterations={len(cost)}  final cost={cost[-1]:.6f}")

    # Step 3b — Resolve per-subject sign ambiguity (see frequency_channel script).
    sigma_corr, W, sign_flips = align_iva_component_signs(Sigma_N, W)
    n_flipped = int((sign_flips < 0).sum())
    _logger.info(
        f"[{label}] Sign alignment: flipped {n_flipped} (component, subject) "
        f"pairs across {n_pca} components."
    )

    # Step 4 — Recover spectro-temporal sources + channel patterns + marginals.
    iva_scores_pca = np.zeros((n_subjects, n_pca, n_samples_ft))
    iva_components = np.zeros((n_subjects, n_pca, n_channels))
    for k in range(n_subjects):
        W_k = W[:, :, k]
        iva_scores_pca[k] = W_k @ X_pca[:, :, k]
        # Forward (mixing) patterns, NOT the unmixing rows: a topography is the
        # column of the mixing matrix, and after iva_g's internal whitening the
        # unmixing rows carry a Σ⁻¹ reweighting. See iva_component_patterns.
        iva_components[k] = iva_component_patterns(W_k, pcas[k].components_)
    # Reshape aligned F*T axis back to (F, T): the shared spectro-temporal source.
    iva_sources = iva_scores_pca.reshape(n_subjects, n_pca, n_freqs, n_times)
    # Signed marginals (signs kept; iva_freq is ~0 by construction — see header).
    iva_time = iva_sources.mean(axis=2)  # (S, N_PCA, T) — frequency-collapsed
    iva_freq = iva_sources.mean(axis=3)  # (S, N_PCA, F) — time-collapsed
    _logger.info(
        f"[{label}] Recover: sources={iva_sources.shape}, "
        f"components={iva_components.shape}"
    )

    # Diagnostic: how far the forward patterns sit from the unmixing rows that
    # used to be plotted. |r| near 1 → the whitening reweighting was harmless
    # here; |r| near 0 → the old topomaps were showing something else entirely.
    filter_pattern_r = np.array(
        [
            abs(
                pearsonr(
                    (W[:, :, k] @ pcas[k].components_)[c],
                    iva_components[k, c],
                )[0]
            )
            for k in range(n_subjects)
            for c in range(n_pca)
        ]
    )
    _logger.info(
        f"[{label}] Pattern vs unmixing-row |r| across "
        f"{n_subjects}x{n_pca} (subject, component) pairs: "
        f"mean {np.nanmean(filter_pattern_r):.3f}, "
        f"median {np.nanmedian(filter_pattern_r):.3f}, "
        f"max {np.nanmax(filter_pattern_r):.3f}."
    )

    # Step 5 — Rank by mean off-diagonal Sigma_N correlation, select top + bottom.
    off_diag_mask = ~np.eye(n_subjects, dtype=bool)
    rank_score = np.array([sigma_corr[k][off_diag_mask].mean() for k in range(n_pca)])
    order = np.argsort(rank_score)[::-1]
    top_indices = order[:n_top].tolist()
    bottom_indices = order[-n_bottom:][::-1].tolist() if n_bottom > 0 else []
    # 1-based rank per component, so a panel can still report where the ranking
    # put it once the panels are no longer ordered by it.
    rank_of = {int(k): i + 1 for i, k in enumerate(order)}
    if quality:
        # --quality makes these figures companions of the quality ones, which
        # number every panel in IVA component order ("IC k+1"). Ranked panel
        # positions would put the same IC in a different cell of every grid, so
        # the *selection* still comes from the ranking but the *order* does not.
        top_indices = sorted(top_indices)
        bottom_indices = sorted(bottom_indices)
    selected_indices = top_indices + bottom_indices

    def _ic_title(i: int) -> str:
        k = selected_indices[i]
        if quality:
            # No positional TOP/BOT tag: the panels sit in IVA order, so it would
            # name a grid cell rather than the component's standing.
            return f"IC {k + 1} (rank {rank_of[k]}, r={rank_score[k]:+.2f})"
        tag = f"TOP {i + 1}" if i < n_top else f"BOT {i - n_top + 1}"
        return f"{tag} (IC {k + 1}, r={rank_score[k]:+.2f})"

    selected_labels = [_ic_title(i) for i in range(len(selected_indices))]
    top_labels = selected_labels[:n_top]
    bot_labels = selected_labels[n_top:]
    _logger.info(
        f"[{label}] Top IC indices: {top_indices}  "
        f"(r={[round(rank_score[k], 3) for k in top_indices]})"
    )
    if quality:
        _logger.info(
            f"[{label}] --quality: per-component panels laid out in IVA component "
            f"order rather than by rank, so they line up with the quality figures; "
            f"each panel reports its rank."
        )

    # The bottom group is dropped entirely when --n_bottom is 0: every figure
    # below is a per-component grid, and laying one out over an empty index list
    # raises (np.stack of nothing, a 0-column subplot grid) rather than producing
    # an empty figure.
    order_note = " — IVA order" if quality else ""
    groups = [(top_indices, top_labels, f"TOP {n_top}{order_note}", "top")]
    if bottom_indices:
        groups.append(
            (bottom_indices, bot_labels, f"BOTTOM {n_bottom}{order_note}", "bottom")
        )
    else:
        _logger.info(
            f"[{label}] --n_bottom 0: skipping the bottom-component contrast figures."
        )

    # ---------- Plots ----------

    # (a) Component-ranking summary + channel-PCA scree.
    _plot_component_ranking(
        rank_score,
        top_indices,
        bottom_indices,
        label=label,
        save_path=out_dir / f"{prefix}iva_component_ranking_{label}.png",
    )
    _plot_pca_scree(
        pca_evr,
        label=label,
        save_path=out_dir / f"{prefix}pca_scree_{label}.png",
    )

    # (b) Temporal-marginal ISC matrix + cluster strip — TOP / BOTTOM.
    for indices, labels_grp, group_name, suffix in groups:
        corr = np.stack([np.corrcoef(iva_time[:, k, :]) for k in indices])
        _plot_isc_grid(
            corr,
            labels=labels_grp,
            fig_title=(f"ISC of Temporal Marginal (signed) — {group_name} — {label}"),
            save_path=out_dir / f"{prefix}iva_isc_temporal_{label}_{suffix}.png",
        )

    # (c) Spectral-marginal ISC matrix + cluster strip — TOP / BOTTOM.
    # NB: iva_freq ~0 by construction, so this matrix may look like noise.
    for indices, labels_grp, group_name, suffix in groups:
        corr = np.stack([np.corrcoef(iva_freq[:, k, :]) for k in indices])
        _plot_isc_grid(
            corr,
            labels=labels_grp,
            fig_title=(f"ISC of Spectral Marginal (signed) — {group_name} — {label}"),
            save_path=out_dir / f"{prefix}iva_isc_spectral_{label}_{suffix}.png",
        )

    # (d) LOO-ISC bars for four axes + grouped by-axis summary.
    loo_axes = (
        ("Spectro-Temporal (F·T)", iva_scores_pca, "spectrotemporal"),
        ("Temporal (T)", iva_time, "temporal"),
        ("Spectral (F)", iva_freq, "spectral"),
        ("Channel (C)", iva_components, "channel"),
    )
    loo_means_by_axis: dict[str, np.ndarray] = {}
    for axis_name, axis_data, fname in loo_axes:
        view = axis_data[:, selected_indices, :].transpose(1, 0, 2)  # (K, S, L)
        loo = _loo_isc(view)
        loo_means_by_axis[axis_name] = loo.mean(axis=1)
        _plot_loo_isc_bar(
            loo,
            labels=selected_labels,
            n_top=n_top,
            axis_name=axis_name,
            label=label,
            save_path=out_dir / f"{prefix}iva_loo_isc_{fname}_{label}.png",
        )
    _plot_loo_axis_summary(
        loo_means_by_axis,
        top_labels=top_labels,
        label=label,
        save_path=out_dir / f"{prefix}iva_loo_isc_axis_summary_{label}.png",
    )

    # (e) Mean & variance topomap of the channel patterns — TOP / BOTTOM.
    if info is not None:
        for indices, labels_grp, _group_name, suffix in groups:
            _plot_topomap_mean_var(
                iva_components[:, indices, :],
                info,
                n_channels,
                labels=labels_grp,
                label=label,
                save_path=out_dir
                / f"{prefix}iva_topomap_mean_var_{label}_{suffix}.png",
            )
    else:
        _logger.warning(f"[{label}] No info available; skipping topomap plot.")

    # (f) Shared time × frequency source map (direct) — TOP / BOTTOM.
    mean_sources_all = iva_sources.mean(axis=0)  # (N_PCA, F, T)
    for indices, labels_grp, _group_name, suffix in groups:
        _plot_tf_map_direct(
            mean_sources_all[indices],
            time,
            freqs,
            labels=labels_grp,
            label=label,
            save_path=out_dir
            / f"{prefix}iva_shared_time_frequency_{label}_{suffix}.png",
        )

    # (g) Source mean ± √variance along time and along frequency — TOP / BOTTOM.
    mean_time = iva_time.mean(axis=0)  # (N_PCA, T)
    std_time = iva_time.std(axis=0)
    mean_freq = iva_freq.mean(axis=0)  # (N_PCA, F)
    std_freq = iva_freq.std(axis=0)
    for indices, labels_grp, _group_name, suffix in groups:
        _plot_profile_mean_var(
            mean_time[indices],
            std_time[indices],
            time,
            labels=labels_grp,
            x_label="Time (s)",
            axis_name="Temporal",
            label=label,
            save_path=out_dir
            / f"{prefix}iva_mean_variance_over_time_{label}_{suffix}.png",
        )
        _plot_profile_mean_var(
            mean_freq[indices],
            std_freq[indices],
            freqs,
            labels=labels_grp,
            x_label="Frequency (Hz)",
            axis_name="Spectral",
            label=label,
            save_path=out_dir
            / f"{prefix}iva_mean_variance_over_freq_{label}_{suffix}.png",
        )

    # (h) Pair-space heatmaps — Time × Subject, Subject × Frequency/Channel.
    for indices, labels_grp, _group_name, suffix in groups:
        time_subj = iva_time.transpose(1, 0, 2)[indices]  # (K, S, T)
        _plot_pairmap_time_subject(
            time_subj,
            time,
            labels=labels_grp,
            label=label,
            save_path=out_dir
            / f"{prefix}iva_pairmap_time_subject_{label}_{suffix}.png",
        )

        subj_freq = iva_freq.transpose(1, 2, 0)[indices]  # (K, F, S)
        _plot_pairmap_subject_x(
            subj_freq,
            freqs,
            y_label="Frequency (Hz)",
            variant_name="Subject × Frequency (spectral marginal)",
            labels=labels_grp,
            label=label,
            save_path=out_dir
            / f"{prefix}iva_pairmap_subject_frequency_{label}_{suffix}.png",
        )

        subj_chan = iva_components.transpose(1, 2, 0)[indices]  # (K, C, S)
        _plot_pairmap_subject_x(
            subj_chan,
            np.arange(n_channels),
            y_label="Channel",
            variant_name="Subject × Channel (topographies)",
            labels=labels_grp,
            label=label,
            save_path=out_dir
            / f"{prefix}iva_pairmap_subject_channel_{label}_{suffix}.png",
        )

    # (i) Mean subject loading: mean |spectro-temporal score| over the F*T axis.
    subject_loadings_all = np.abs(iva_scores_pca).mean(axis=2)  # (S, N_PCA)
    for indices, labels_grp, _group_name, suffix in groups:
        _plot_subject_loadings(
            subject_loadings_all[:, indices],
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_subject_loadings_{label}_{suffix}.png",
        )

    _logger.info(f"[{label}] IVA-channel: plot files saved to {out_dir}")

    # (j) Quality analysis (ASSR-only): topomap & time/TF correlation scatterplots.
    if quality:
        _run_quality(
            iva_components,
            iva_sources,
            freqs,
            info,
            n_channels,
            stimulus_onsets,
            sfreq,
            # The IVA's own input tensor — the wavelet reference must be built
            # from exactly what was decomposed, so reuse bb_z rather than
            # z-scoring a second copy of data_4d.
            bb_z,
            subject_ids=subject_ids,
            label=label,
            out_dir=out_dir,
            prefix=prefix,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    experiment_name = ExperimentNames(args.experiment)
    condition = ConditionVariants(args.condition)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        # ASSR has no music dimension; uses a single placeholder "music type".
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    wavelet_dir = resolve_wavelet_dir(args.wavelet_data_dir, experiment_name)
    freqs_full = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )
    band_name = args.band

    _logger.info(
        f"IVA-channel: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"n_pca={args.n_pca}, n_top={args.n_top}, n_bottom={args.n_bottom}, "
        f"band={band_name or SpectrumTypeVariants.BROADBAND.value}, "
        f"iva_opt={args.iva_opt_approach}"
    )

    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=False,
        experiment_name=experiment_name,
    )
    datasets = analyzers_to_datasets(analyzers)

    for mt in music_types:
        mt_label = mt.value
        dataset_key = f"{condition.value}_{mt_label}"
        if dataset_key not in datasets:
            _logger.warning(f"No data for {dataset_key!r}; skipping.")
            continue

        ad = datasets[dataset_key]
        analyzer = analyzers.get(dataset_key)
        info = getattr(analyzer, "info", None) if analyzer is not None else None
        stimulus_onsets = (
            getattr(analyzer, "stimulus_onsets", None) if analyzer is not None else None
        )

        # Per-subject participant labels (3-digit ID from the metadata sidecar) for
        # the quality scatter point labels.
        fdf = getattr(analyzer, "filtered_df", None) if analyzer is not None else None
        try:
            subject_ids = participant_labels(fdf, ad.data.shape[0])
        except ValueError as exc:
            # Labels are optional, but never fail silently — a wrong or missing
            # participant label makes every per-subject plot unreadable.
            _logger.warning(
                f"[{dataset_key}] Participant labels unavailable ({exc}); "
                "quality plots will fall back to subject indices."
            )
            subject_ids = None

        _logger.info(
            f"Dataset [{dataset_key}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz"
        )

        wd = _broadband_wavelet_4d(
            ad,
            dataset_key,
            representation="power",
            freqs=freqs_full,
            wavelet_dir=(wavelet_dir / SpectrumTypeVariants.BROADBAND.value),
            reuse_wavelets=args.reuse_wavelets,
        )

        if band_name is None:
            data_4d = wd.data
            iva_freqs = freqs_full
        else:
            band_lo, band_hi = FREQUENCY_BANDS[band_name]
            band_mask = (freqs_full >= band_lo) & (freqs_full <= band_hi)
            if not band_mask.any():
                raise ValueError(
                    f"No broadband frequencies fall in {band_name} range "
                    f"[{band_lo}, {band_hi}] Hz; got freqs={freqs_full}."
                )
            data_4d = wd.data[:, :, band_mask, :]
            iva_freqs = freqs_full[band_mask]
            _logger.info(
                f"[{dataset_key}] band={band_name}: sliced "
                f"{wd.data.shape} -> {data_4d.shape} "
                f"(freqs {iva_freqs[0]:.1f}-{iva_freqs[-1]:.1f} Hz)"
            )

        save_dir = save_root / _STAGE_DIR / dataset_key
        _run_iva(
            data_4d,
            wd.sfreq,
            iva_freqs,
            info,
            label=dataset_key,
            n_pca=args.n_pca,
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            random_state=args.random_state,
            iva_opt_approach=args.iva_opt_approach,
            iva_max_iter=args.iva_max_iter,
            iva_w_diff_stop=args.iva_w_diff_stop,
            save_dir=save_dir,
            band=band_name,
            quality=args.quality,
            stimulus_onsets=stimulus_onsets,
            subject_ids=subject_ids,
        )

    _logger.info("IVA-channel analysis complete.")
