"""Whole-dataset replication of the ASSR raw-voltage channel-PCA notebook.

Headless, all-subjects / all-channels counterpart of
``notebooks/00-preprocessing/assr_raw_pca_analysis.ipynb``. For a given condition
it reduces the stimulus-locked **evoked voltage** to a few spatial components per
participant by collapsing the channel dimension with a **PCA over channels**, then
plots each component's time course and scalp topography.

``--n_components`` (default 3) sets how many leading components are kept. All of
them are processed **identically** — same alignment, same diagnostics, same plots —
so later components can be compared against PC1 instead of assumed uninformative:
PCA orders components by explained *channel variance* in the evoked window, which
is not the same thing as carrying the 40 Hz steady-state, so a large slow onset
deflection can own PC1 and push the ASSR into PC2 or PC3.

Pipeline (per subject):

1. **Z-score per channel** (over the whole recording) so every electrode has unit
   variance and contributes equally to the channel PCA (correlation-PCA). Optional
   (``--no_zscore_per_channel`` keeps raw µV, a covariance-PCA).
2. **Trial average.** The signal is epoch-averaged around every ``fam+`` onset. The
   epoch keeps a fixed ``pre_pad`` before onset and a post-onset length equal to the
   paradigm's post-onset span — ``0.5`` s stimulus + ``0.5`` s post-stimulus (see
   :class:`src.definitions.constants.AssrEpoch`) — capped by the shortest
   inter-onset gap so no epoch overlaps a neighbouring stimulus.
3. **Channel PCA.** Each time sample is an observation and each channel a variable;
   PCA is fit over the ``(win, n_channels)`` matrix. Every kept component's score is
   its time course ``(win,)`` and its loading is its scalp topography
   ``(n_channels,)``.
4. **Polarity alignment across participants**, run **independently per component**
   (each component's sign is its own arbitrary choice). Each subject's loading is
   aligned to a common template (iteratively-refined group mean) and the overall
   orientation anchored to a reference channel (:mod:`src.analysis.pca_polarity`).
   This maximises cross-subject topography agreement, which is what makes the group
   averages meaningful; the score is flipped with the loading so topography and time
   course stay consistent.
5. **Component comparison.** A summary table is printed with, per component, the
   mean explained variance, the topography-consistency diagnostic and the ASSR-band
   SNR of the driven interval (per-subject median and group-mean waveform) — the
   evidence for whether a later component is the better steady-state carrier.
6. **Unreduced reference** (``--no_unreduced_reference`` to skip). The same
   trial-averaged data is plotted **before** the reduction, in channel space, as a
   test reference: the component panels are only trustworthy insofar as they
   reproduce what channel space already shows. Nothing here needs polarity
   alignment — the evoked voltage has a physical sign, so the across-participant
   mean is meaningful directly, which is what makes it a reference *for* the
   alignment rather than another thing to diagnose.

Inputs (produced by stimulus alignment + the wavelet store jobs)::

    data/processed/<exp>/concatenated/<Condition>_ASSR.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.stimulus_onsets.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.metadata.csv
    data/processed/<exp>/wavelets/broadband/<Condition>_ASSR__wavelet_power__*.npz

The concatenated raw array (~1 GB) is memory-mapped, so only the accessed subjects
land in memory. The wavelet cache is opened only to read its small channel-name
member (the canonical concatenated channel order) — the huge data stream is never
touched.

Output plots::

    plots/00-preprocessing/assr_raw_pca/<Condition>_ASSR/*.png

The unreduced reference figures share that directory under the
``raw_unreduced_*`` prefix.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no interactive display on the cluster

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import numpy.lib.format as npformat  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import zscore  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pca_polarity import (  # noqa: E402
    align_pc1_signs,
    apply_pc1_signs,
    topography_consistency,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    MusicTypeVariants,
    PreprocessedDataVariants,
)
from src.preprocessing.pipeline import DatasetHandler  # noqa: E402

mne.set_log_level("ERROR")

# Metadata CSV column names (sidecar stores enum keys as their string repr).
_PID_COL = "SingleDataMetadata.PARTICIPANT_ID"
_PIDX_COL = "SingleDataMetadata.CONCATENATED_PERSON_INDEX"
_FNAME_COL = "SingleDataMetadata.FILENAME"

# Spacing of the unreduced group-mean topomap series (s). Latencies snap to the
# nearest sample, and each panel title reports the latency it actually shows.
_LATENCY_STEP_S = 0.05


# --------------------------------------------------------------------------- #
#  Core numerics                                                              #
# --------------------------------------------------------------------------- #
def epoch_average(
    arr: np.ndarray, onsets: np.ndarray, pre: int, post: int
) -> tuple[np.ndarray, int]:
    """Average fixed windows around each onset along the last (time) axis.

    Args:
        arr: Array whose last axis is time, e.g. ``(n_channels, n_times)``.
        onsets: Stimulus onset sample indices.
        pre: Samples kept before each onset.
        post: Samples kept after each onset (window length is ``pre + post``).

    Returns:
        Tuple ``(mean, n_used)`` where ``mean`` has the time axis replaced by the
        ``pre + post`` window, averaged over all onsets whose window fits inside
        the recording, and ``n_used`` is how many onsets contributed.
    """
    n_time = arr.shape[-1]
    acc = None
    n_used = 0
    for onset in onsets:
        start, end = onset - pre, onset + post
        if start < 0 or end > n_time:
            continue
        seg = arr[..., start:end]
        acc = seg.astype(np.float64) if acc is None else acc + seg
        n_used += 1
    if n_used == 0:
        raise ValueError("No onset window fits inside the recording.")
    return acc / n_used, n_used


def channel_pca_components(
    evoked: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse channels of an evoked map with PCA, keeping the leading components.

    Args:
        evoked: ``(n_channels, win)`` stimulus-locked trial average.
        n_components: Number of leading components to keep.

    Returns:
        Tuple ``(time_courses, loadings, explained)`` with shapes
        ``(n_components, win)``, ``(n_components, n_channels)`` and
        ``(n_components,)``: each component's score (time course), loading (scalp
        topography) and explained-variance ratio. Signs are left arbitrary here;
        they are aligned across participants, per component, by
        :func:`src.analysis.pca_polarity.align_pc1_signs`.
    """
    matrix = evoked.T  # (win, n_channels): observations = time, variables = channels
    pca = PCA(n_components=n_components)
    time_courses = pca.fit_transform(matrix).T  # (n_components, win)
    return time_courses, pca.components_, pca.explained_variance_ratio_


def assr_snr(
    time_course: np.ndarray,
    stim_mask: np.ndarray,
    sfreq: float,
    assr_freq: float,
    n_side: int = 3,
    n_exclude: int = 1,
) -> float:
    """Steady-state SNR of one component score over the driven interval.

    Power at *assr_freq* relative to the median of neighbouring spectral bins,
    computed on the stimulus interval only — a 1.1 s epoch around a 0.5 s stimulus
    is half silence, which dilutes a steady-state estimate. The Hann taper spreads
    the peak into its immediate neighbours, so *n_exclude* bins on each side are
    skipped when forming the noise floor.

    Args:
        time_course: ``(win,)`` component score over the full epoch.
        stim_mask: Boolean mask selecting the driven samples of the epoch.
        sfreq: Sampling rate (Hz).
        assr_freq: Steady-state frequency (Hz).
        n_side: Noise-floor bins taken on each side of the ASSR bin.
        n_exclude: Bins adjacent to the peak skipped (taper leakage).

    Returns:
        The SNR ratio, or NaN when the noise floor is degenerate.
    """
    seg = np.asarray(time_course, dtype=float)[stim_mask]
    seg = (seg - seg.mean()) * np.hanning(seg.size)
    power = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sfreq)
    peak = int(np.argmin(np.abs(freqs - assr_freq)))
    left_end = peak - n_exclude
    side = np.concatenate(
        [
            power[max(left_end - n_side, 0) : max(left_end, 0)],
            power[peak + n_exclude + 1 : peak + n_exclude + 1 + n_side],
        ]
    )
    floor = float(np.median(side)) if side.size else float("nan")
    return float(power[peak] / floor) if floor > 0 else float("nan")


def global_field_power(evoked: np.ndarray) -> np.ndarray:
    """Root mean square across channels, i.e. the global field power.

    One unsigned amplitude trace, insensitive to the dipolar sign structure that
    makes individual channel waveforms hard to compare.

    Args:
        evoked: Array whose last two axes are ``(n_channels, win)``, e.g. a single
            ``(n_channels, win)`` evoked map or a ``(n_subjects, n_channels, win)``
            stack.

    Returns:
        The input with the channel axis collapsed, e.g. ``(win,)`` or
        ``(n_subjects, win)``.
    """
    return np.sqrt((np.asarray(evoked, dtype=float) ** 2).mean(axis=-2))


def channel_assr_snr(
    evoked: np.ndarray,
    stim_mask: np.ndarray,
    sfreq: float,
    assr_freq: float,
) -> np.ndarray:
    """Steady-state SNR of every channel of an *unreduced* evoked map.

    The measure of :func:`assr_snr` applied to raw channels instead of component
    scores, so the resulting map localises the steady-state in channel space without
    assuming that any component carries it — the reference a component loading can be
    checked against.

    Args:
        evoked: ``(n_channels, win)`` trial-averaged evoked response.
        stim_mask: Boolean mask selecting the driven samples of the epoch.
        sfreq: Sampling rate (Hz).
        assr_freq: Steady-state frequency (Hz).

    Returns:
        ``(n_channels,)`` per-channel SNR.
    """
    return np.array(
        [
            assr_snr(channel, stim_mask, sfreq, assr_freq)
            for channel in np.asarray(evoked, dtype=float)
        ]
    )


def component_comparison(
    time_courses: np.ndarray,
    explained: np.ndarray,
    consistency: list,
    pc_labels: list[str],
    stim_mask: np.ndarray,
    sfreq: float,
    assr_freq: float,
    snr_threshold: float,
) -> pd.DataFrame:
    """Score every component on the same measures, for a side-by-side comparison.

    A later component is the better ASSR carrier only if it beats PC1 on the SNR
    columns *and* holds up on topography consistency: winning on SNR while failing
    consistency means subjects have 40 Hz in some second mode, but not in the same
    second mode, so its group average is not interpretable.

    Args:
        time_courses: ``(n_components, n_subjects, win)`` polarity-aligned scores.
        explained: ``(n_components, n_subjects)`` explained-variance ratios.
        consistency: Per-component
            :class:`src.analysis.pca_polarity.SignConsistency` results.
        pc_labels: Component names, e.g. ``["PC1", "PC2"]``.
        stim_mask: Boolean mask selecting the driven samples of the epoch.
        sfreq: Sampling rate (Hz).
        assr_freq: Steady-state frequency (Hz).
        snr_threshold: Per-subject SNR counted as a hit in the ``n_SNR>`` column.

    Returns:
        A DataFrame indexed by component name.
    """
    n_components = time_courses.shape[0]
    subject_snr = np.array(
        [
            [assr_snr(tc, stim_mask, sfreq, assr_freq) for tc in time_courses[c]]
            for c in range(n_components)
        ]
    )
    group_snr = np.array(
        [
            assr_snr(time_courses[c].mean(axis=0), stim_mask, sfreq, assr_freq)
            for c in range(n_components)
        ]
    )
    return pd.DataFrame(
        {
            "component": pc_labels,
            "mean_EV_%": [explained[c].mean() * 100 for c in range(n_components)],
            "agreeing": [
                f"{consistency[c].n_agreeing}/{consistency[c].n_subjects}"
                for c in range(n_components)
            ],
            "median_pairwise_r": [
                consistency[c].median_pairwise_r for c in range(n_components)
            ],
            "min_subject_r": [
                consistency[c].min_subject_r for c in range(n_components)
            ],
            f"{assr_freq:.0f}Hz_SNR_subject_median": np.nanmedian(subject_snr, axis=1),
            f"{assr_freq:.0f}Hz_SNR_group_mean": group_snr,
            f"n_SNR>{snr_threshold:.0f}": (subject_snr > snr_threshold).sum(axis=1),
        }
    ).set_index("component")


# --------------------------------------------------------------------------- #
#  IO helpers                                                                 #
# --------------------------------------------------------------------------- #
def read_channel_names(npz_path: Path) -> list[str]:
    """Recover the canonical concatenated channel order from a wavelet cache.

    Only the small ``freqs``/``feature_names`` members are decompressed; the huge
    ``data`` stream is never touched.

    Args:
        npz_path: Path to the wavelet-power ``.npz`` cache.

    Returns:
        Channel names, parsed from ``feature_names`` (layout ``channel * n_freqs``).
    """
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("freqs.npy") as fh:
            freqs = npformat.read_array(fh, allow_pickle=True)
        with zf.open("feature_names.npy") as fh:
            feature_names = npformat.read_array(fh, allow_pickle=True)
    n_freqs = len(freqs)
    return [
        str(feature_names[i * n_freqs]).split("@")[0]
        for i in range(len(feature_names) // n_freqs)
    ]


def grid_shape(n_panels: int, max_cols: int = 5) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for a roughly square panel grid."""
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def plot_timecourse_per_participant(
    time_courses: np.ndarray,
    explained: np.ndarray,
    epoch_times: np.ndarray,
    labels: list[str],
    unit: str,
    pc_label: str,
    color: str,
    stim_end: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Grid of per-participant time courses for ONE component (sorted by PID)."""
    n_subj = time_courses.shape[0]
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), sharex=True, squeeze=False
    )
    flat = axes.flatten()
    for ax, subj in zip(flat, range(n_subj)):
        ax.plot(epoch_times, time_courses[subj], lw=1.5, color=color)
        ax.axvspan(0.0, stim_end, color="grey", alpha=0.12, lw=0)
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(
            f"{labels[subj]}  ({pc_label} {explained[subj] * 100:.0f}%)", fontsize=9
        )
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel(f"{pc_label} projected signal ({unit}, polarity-aligned)")
    fig.suptitle(
        f"Per-participant channel-PCA {pc_label} time course — {title_suffix}",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        plots_dir / f"raw_pca_{pc_label.lower()}_timecourse_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_timecourse_overlay(
    time_courses: np.ndarray,
    explained: np.ndarray,
    consistency: list,
    epoch_times: np.ndarray,
    labels: list[str],
    unit: str,
    pc_labels: list[str],
    stim_end: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Participants overlaid with the group mean, one row per component.

    Each row keeps its own y-scale: component scores shrink with component order by
    construction, so a shared scale would flatten later components into lines and
    hide the waveform shape being compared.
    """
    n_components = time_courses.shape[0]
    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=(11, 4.2 * n_components),
        sharex=True,
        squeeze=False,
    )
    for comp, pc_label in enumerate(pc_labels):
        ax = axes[comp, 0]
        for subj, label in enumerate(labels):
            ax.plot(
                epoch_times, time_courses[comp, subj], lw=1.1, alpha=0.75, label=label
            )
        ax.plot(
            epoch_times,
            time_courses[comp].mean(axis=0),
            lw=2.6,
            color="black",
            label="group mean",
        )
        ax.axvspan(0.0, stim_end, color="grey", alpha=0.12, lw=0)
        ax.axvline(0.0, color="red", ls="--", lw=1, label="onset")
        ax.set_title(
            f"{pc_label} — mean EV {explained[comp].mean() * 100:.0f}%, "
            f"topographies agreeing {consistency[comp].n_agreeing}/"
            f"{consistency[comp].n_subjects}, median pairwise r "
            f"{consistency[comp].median_pairwise_r:+.2f}",
            fontsize=10,
        )
        ax.set_ylabel(f"{pc_label} ({unit})")
        if comp == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Time relative to onset (s)")
    fig.suptitle(f"Channel-PCA component time courses — {title_suffix}", y=1.0)
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_pca_timecourse_overlay.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_topomaps(
    loadings: np.ndarray,
    explained: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    labels: list[str],
    pc_label: str,
    agreement: str,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant + group scalp topographies of ONE component (nb05 convention).

    Panel titles carry each subject's explained-variance ratio for this component;
    the group panel carries the mean across subjects.
    """
    n_subj = loadings.shape[0]
    panels = [
        (
            f"{labels[s]}  ({pc_label} {explained[s] * 100:.0f}%)",
            loadings[s][info_order],
        )
        for s in range(n_subj)
    ]
    mean_ev = float(np.mean(explained)) * 100
    panels.append(
        (
            f"group average  (mean {pc_label} {mean_ev:.0f}%)",
            loadings.mean(axis=0)[info_order],
        )
    )

    # Symmetric shared scale (nb05 convention): 99th percentile of |loading|.
    vlim = float(np.percentile(np.abs(np.concatenate([v for _, v in panels])), 99))
    if vlim == 0.0:
        vlim = 1e-12

    nrows, ncols = grid_shape(len(panels))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, (label, vals) in zip(flat, panels):
        im, _ = mne.viz.plot_topomap(
            vals,
            topo_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim, vlim),
            contours=4,
        )
        ax.set_title(label, fontsize=9)
    for ax in flat[len(panels) :]:
        ax.axis("off")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.6, label=f"{pc_label} loading (a.u.)"
    )
    fig.suptitle(
        f"Per-participant channel-PCA {pc_label} topography — {title_suffix} "
        f"(agreeing {agreement})",
        y=1.0,
    )
    fig.savefig(
        plots_dir / f"raw_pca_{pc_label.lower()}_topomap_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Plotting — unreduced (no-PCA) reference                                     #
# --------------------------------------------------------------------------- #
def plot_unreduced_butterfly_per_participant(
    evoked_all: np.ndarray,
    gfp: np.ndarray,
    epoch_times: np.ndarray,
    labels: list[str],
    unit: str,
    stim_end: float,
    n_used: int,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant butterfly (all channels) with the GFP drawn on top.

    The unreduced counterpart of the per-component time-course grid: a 40 Hz ripple
    in a component's score should be visible in these channels.

    Args:
        evoked_all: ``(n_subjects, n_channels, win)`` trial averages, PID-ordered.
        gfp: ``(n_subjects, win)`` global field power of the same subjects.
        epoch_times: ``(win,)`` epoch time axis in seconds, 0 at onset.
        labels: Participant labels, aligned with *evoked_all*.
        unit: Amplitude unit for the axis label.
        stim_end: End of the driven interval (s), shaded.
        n_used: Stimuli averaged per participant (for the title).
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
    """
    n_subj, n_channels, _ = evoked_all.shape
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 2.7 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = axes.flatten()
    for ax, subj in zip(flat, range(n_subj)):
        ax.plot(epoch_times, evoked_all[subj].T, lw=0.4, color="steelblue", alpha=0.45)
        ax.plot(epoch_times, gfp[subj], lw=1.8, color="black")
        ax.axvspan(0.0, stim_end, color="grey", alpha=0.12, lw=0)
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(
            f"{labels[subj]}  (GFP max {gfp[subj].max():.2f})",
            fontsize=9,
        )
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel(f"Evoked voltage, all {n_channels} channels ({unit})")
    fig.suptitle(
        f"Unreduced per-participant evoked response, butterfly + GFP (black) — "
        f"{title_suffix}, {n_used} stimuli averaged",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_unreduced_butterfly_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_unreduced_group_timecourse(
    evoked_all: np.ndarray,
    gfp: np.ndarray,
    epoch_times: np.ndarray,
    labels: list[str],
    unit: str,
    stim_end: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant GFPs beside the group-mean evoked response.

    The two black traces answer different questions: the mean of the GFPs measures
    amplitude regardless of phase, while the GFP of the group mean only survives if
    participants are phase-consistent. Their ratio (printed by the caller) is the
    reference for whether a group average is worth reading at all.

    Args:
        evoked_all: ``(n_subjects, n_channels, win)`` trial averages, PID-ordered.
        gfp: ``(n_subjects, win)`` global field power of the same subjects.
        epoch_times: ``(win,)`` epoch time axis in seconds, 0 at onset.
        labels: Participant labels, aligned with *evoked_all*.
        unit: Amplitude unit for the axis labels.
        stim_end: End of the driven interval (s), shaded.
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
    """
    group_evoked = evoked_all.mean(axis=0)
    group_gfp = global_field_power(group_evoked)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)

    ax = axes[0]
    for subj, label in enumerate(labels):
        ax.plot(epoch_times, gfp[subj], lw=1.1, alpha=0.75, label=label)
    ax.plot(epoch_times, gfp.mean(axis=0), lw=2.6, color="black", label="mean of GFPs")
    ax.set_title("Per-participant GFP (unsigned amplitude)", fontsize=10)
    ax.set_ylabel(f"GFP ({unit})")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[1]
    ax.plot(epoch_times, group_evoked.T, lw=0.4, color="steelblue", alpha=0.45)
    ax.plot(epoch_times, group_gfp, lw=2.2, color="black", label="GFP of group mean")
    ax.set_title(
        f"Group-mean evoked response, all {group_evoked.shape[0]} channels",
        fontsize=10,
    )
    ax.set_ylabel(f"Evoked voltage ({unit})")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.axvspan(0.0, stim_end, color="grey", alpha=0.12, lw=0)
        ax.axvline(0.0, color="red", ls="--", lw=1)
        ax.set_xlabel("Time relative to onset (s)")
    fig.suptitle(f"Unreduced group-level evoked response — {title_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_unreduced_group_butterfly_gfp.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_unreduced_topomap_latencies(
    group_evoked: np.ndarray,
    epoch_times: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    unit: str,
    stim_end: float,
    title_suffix: str,
    plots_dir: Path,
    step_s: float = _LATENCY_STEP_S,
) -> None:
    """Group-mean evoked topography at a series of latencies (nb05 convention).

    What a component's single loading map summarises into one picture: a component
    whose loading resembles the topography around the latency where its time course
    peaks is describing a real spatial mode.

    Args:
        group_evoked: ``(n_channels, win)`` across-participant mean evoked map, in
            the canonical channel order.
        epoch_times: ``(win,)`` epoch time axis in seconds, 0 at onset.
        topo_info: Montage info supplying the electrode positions.
        info_order: Indices mapping the canonical channel order onto *topo_info*.
        unit: Amplitude unit for the colour-bar label.
        stim_end: End of the driven interval (s); those panels are starred.
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
        step_s: Spacing of the latency series (s); latencies snap to samples.
    """
    start = max(float(epoch_times[0]), -step_s)
    latencies = np.arange(start, float(epoch_times[-1]) + 1e-9, step_s)
    lat_idx = [int(np.argmin(np.abs(epoch_times - t))) for t in latencies]

    # Symmetric shared scale (nb05 convention): 99th percentile of |voltage|.
    vlim = float(np.percentile(np.abs(group_evoked[:, lat_idx]), 99))
    if vlim == 0.0:
        vlim = 1e-12

    nrows, ncols = grid_shape(len(lat_idx), max_cols=6)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.3 * ncols, 2.5 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, time_idx in zip(flat, lat_idx):
        im, _ = mne.viz.plot_topomap(
            group_evoked[info_order, time_idx],
            topo_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim, vlim),
            contours=4,
        )
        driven = " *" if 0.0 <= epoch_times[time_idx] <= stim_end else ""
        ax.set_title(f"{epoch_times[time_idx] * 1000:+.0f} ms{driven}", fontsize=9)
    for ax in flat[len(lat_idx) :]:
        ax.axis("off")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.6, label=f"Evoked voltage ({unit})"
    )
    fig.suptitle(
        f"Unreduced group-mean evoked topography over time (* = driven interval) — "
        f"{title_suffix}",
        y=1.0,
    )
    fig.savefig(
        plots_dir / "raw_unreduced_group_topomap_latencies.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_unreduced_summary_topomaps(
    per_subject: np.ndarray,
    group_values: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    labels: list[str],
    measure: str,
    cbar_label: str,
    filename: str,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant + group topomaps of one *unsigned* per-channel summary.

    Unsigned quantities use a sequential scale from 0 rather than the symmetric
    ``RdBu_r`` convention used for loadings, which would spend half its range on
    values that cannot occur.

    Args:
        per_subject: ``(n_subjects, n_channels)`` values, PID-ordered.
        group_values: ``(n_channels,)`` group panel, computed on the group-mean
            evoked response rather than as the mean of *per_subject* — for SNR the
            two differ, and this version is the one that says whether the response
            survives averaging.
        topo_info: Montage info supplying the electrode positions.
        info_order: Indices mapping the canonical channel order onto *topo_info*.
        labels: Participant labels, aligned with *per_subject*.
        measure: Measure name for the title.
        cbar_label: Colour-bar label.
        filename: Output file name inside *plots_dir*.
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
    """
    panels = [
        (labels[subj], per_subject[subj][info_order])
        for subj in range(per_subject.shape[0])
    ]
    panels.append(("group-mean evoked", group_values[info_order]))

    vmax = float(np.percentile(np.concatenate([v for _, v in panels]), 99))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-12

    nrows, ncols = grid_shape(len(panels))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, (label, vals) in zip(flat, panels):
        im, _ = mne.viz.plot_topomap(
            vals,
            topo_info,
            axes=ax,
            show=False,
            cmap="Reds",
            vlim=(0.0, vmax),
            contours=4,
        )
        ax.set_title(label, fontsize=9)
    for ax in flat[len(panels) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=cbar_label)
    fig.suptitle(f"Unreduced per-channel {measure} — {title_suffix}", y=1.0)
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run_pca(args: argparse.Namespace) -> None:
    """Run the full raw channel-PCA reduction for one condition, writing plots."""
    experiment = ExperimentNames(args.experiment)
    condition = ConditionVariants(args.condition)
    music_type = MusicTypeVariants.ASSR
    label = f"{condition.value}_{music_type.value}"

    processed_dir = ProjectPaths.PROCESSED_DATA_DIR / experiment.value
    concat_dir = processed_dir / PreprocessedDataVariants.CONCATENATED.value
    raw_path = concat_dir / f"{label}.npy"
    onsets_path = concat_dir / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
    meta_path = concat_dir / f"{label}.metadata.csv"
    wavelet_dir = processed_dir / "wavelets" / "broadband"
    wavelet_matches = sorted(
        wavelet_dir.glob(f"{label}__wavelet_power__*__freqdim1.npz")
    )

    # A condition may not have been generated yet — skip cleanly instead of failing.
    missing = [p for p in (raw_path, onsets_path, meta_path) if not p.exists()]
    if missing or not wavelet_matches:
        extra = [] if wavelet_matches else ["wavelet npz (channel names)"]
        print(
            f"[SKIP] {label}: missing inputs ({[p.name for p in missing] + extra}).",
            flush=True,
        )
        return

    save_root = Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH
    plots_dir = save_root / "00-preprocessing" / "assr_raw_pca" / label
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {label} ===", flush=True)
    print(f"raw   : {raw_path}", flush=True)
    print(f"plots : {plots_dir}", flush=True)

    # ---- Load shared inputs -------------------------------------------------
    raw_mm = np.load(raw_path, mmap_mode="r")  # (n_subj, n_ch, n_times)
    onsets = np.load(onsets_path)
    meta = pd.read_csv(meta_path, index_col=0)
    gaps = np.diff(onsets)
    channel_names = read_channel_names(wavelet_matches[0])
    n_subj, n_channels, n_times_raw = raw_mm.shape
    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel-name count {len(channel_names)} != raw n_channels {n_channels}"
        )

    idx_to_pid = dict(zip(meta[_PIDX_COL], meta[_PID_COL].astype(str).str.zfill(3)))

    sfreq = args.sfreq
    pre = int(round(args.pre_pad * sfreq))
    # Paradigm window, only capped by the shortest gap so epochs never overlap.
    post = min(int(round(args.post_window * sfreq)), int(gaps.min()))
    epoch_times = np.arange(-pre, post) / sfreq
    stim_mask = AssrEpoch.stimulus_mask(epoch_times)  # driven interval

    # More components than channels is not decomposable; clamp instead of failing.
    n_components = min(args.n_components, n_channels)
    if n_components < args.n_components:
        print(
            f"  NOTE: --n_components {args.n_components} exceeds the {n_channels} "
            f"available channels; using {n_components}.",
            flush=True,
        )
    pc_labels = [f"PC{c + 1}" for c in range(n_components)]

    print(
        f"n_subj={n_subj}, n_channels={n_channels}, onsets={onsets.shape[0]}, "
        f"n_components={n_components}, "
        f"epoch window={pre + post} samples ({pre} pre, {post} post) "
        f"= [{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s "
        f"(stimulus 0–{AssrEpoch.STIMULUS_DURATION_S:.2f} s), "
        f"zscore_per_channel={args.zscore_per_channel}",
        flush=True,
    )
    if pre + post > int(gaps.min()):
        print(
            f"  WARNING: epoch ({pre + post} samples) exceeds the shortest "
            f"inter-onset gap ({int(gaps.min())}) — the pre-onset baseline reaches "
            f"into the previous stimulus.",
            flush=True,
        )

    # ---- Per-subject channel PCA --------------------------------------------
    # Component-major storage: axis 0 = component, axis 1 = subject. Every
    # component is produced by the same code path, so none is special-cased.
    time_courses = np.zeros((n_components, n_subj, pre + post), dtype=np.float64)
    loadings = np.zeros((n_components, n_subj, n_channels), dtype=np.float64)
    explained = np.zeros((n_components, n_subj), dtype=np.float64)
    # Unreduced channel-space evoked responses, kept for the no-PCA reference
    # figures (small: n_subj x n_channels x win floats).
    evoked_all = np.zeros((n_subj, n_channels, pre + post), dtype=np.float64)
    for subj in range(n_subj):
        sig = np.asarray(raw_mm[subj])  # (n_ch, n_times)
        if args.zscore_per_channel:
            sig = zscore(sig, axis=1)  # equal electrode influence (correlation-PCA)
        evoked, n_used = epoch_average(sig, onsets, pre, post)  # (n_ch, win)
        evoked_all[subj] = evoked
        (
            time_courses[:, subj],
            loadings[:, subj],
            explained[:, subj],
        ) = channel_pca_components(evoked, n_components)
        ev_summary = ", ".join(
            f"{pc} {explained[c, subj] * 100:.1f}%" for c, pc in enumerate(pc_labels)
        )
        print(
            f"  reduced subject {subj + 1}/{n_subj} (PSI{idx_to_pid.get(subj, '???')},"
            f" {n_used} stimuli, {ev_summary})",
            flush=True,
        )

    # ---- Order subjects by participant ID ------------------------------------
    # Done BEFORE the polarity alignment, not just for the panels: the template is
    # seeded from the first subject, so for a weakly consistent component the
    # converged signs depend on subject order. Sorting first makes the script
    # reproduce the notebook, which works in participant-ID order throughout.
    order = sorted(range(n_subj), key=lambda s: int(idx_to_pid.get(s, "9999")))
    time_courses = time_courses[:, order]
    loadings = loadings[:, order]
    explained = explained[:, order]
    evoked_all = evoked_all[order]
    labels = [f"PSI{idx_to_pid.get(s, '???')}" for s in order]

    # ---- Align polarity across participants, per component -------------------
    # Flip every subject toward the group-mean template, which maximises
    # cross-subject topography agreement — the criterion that matters when the
    # goal is to aggregate participants. The overall orientation is then anchored
    # to a reference channel so the result is reproducible. Each component's sign
    # is independently arbitrary, so each gets its own alignment pass.
    consistency = []
    for comp, pc_label in enumerate(pc_labels):
        signs, anchor = align_pc1_signs(loadings[comp], channel_names=channel_names)
        loadings[comp] = apply_pc1_signs(loadings[comp], signs)
        time_courses[comp] = apply_pc1_signs(time_courses[comp], signs)
        cons = topography_consistency(loadings[comp])
        consistency.append(cons)
        print(
            f"{pc_label} polarity aligned across participants "
            f"({int((signs < 0).sum())}/{n_subj} subject(s) flipped; anchor {anchor}; "
            f"{cons.n_agreeing}/{cons.n_subjects} topographies agree, "
            f"median pairwise r={cons.median_pairwise_r:+.2f}, "
            f"weakest subject r={cons.min_subject_r:+.2f}).",
            flush=True,
        )
        if cons.n_agreeing < cons.n_subjects:
            print(
                f"  WARNING: {cons.n_subjects - cons.n_agreeing} subject(s) still "
                f"anti-correlate with the group topography — a topographic outlier, "
                f"not a sign problem. Inspect before trusting the {pc_label} panels.",
                flush=True,
            )

    # ---- Topomap electrode positions from a RAW_CROPPED recording -----------
    handler = DatasetHandler(experiment, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS)
    info_fname = meta.loc[meta[_PIDX_COL] == 0, _FNAME_COL].iloc[0]
    topo_info = (
        handler.load_data_file(
            info_fname,
            is_processed=True,
            processed_data_type=PreprocessedDataVariants.RAW_CROPPED,
            preload=False,
        )
        .pick("eeg")
        .info
    )
    name_pos = {name: i for i, name in enumerate(channel_names)}
    if not set(topo_info["ch_names"]) <= set(name_pos):
        raise ValueError("RAW_CROPPED channels are not a subset of channel names.")
    info_order = [name_pos[name] for name in topo_info["ch_names"]]

    # ---- Plots (one figure set per component) --------------------------------
    unit = "z-scored" if args.zscore_per_channel else "µV"
    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj})"
    stim_end = AssrEpoch.STIMULUS_DURATION_S
    for comp, pc_label in enumerate(pc_labels):
        plot_timecourse_per_participant(
            time_courses[comp],
            explained[comp],
            epoch_times,
            labels,
            unit,
            pc_label,
            f"C{comp}",
            stim_end,
            title_suffix,
            plots_dir,
        )
        plot_topomaps(
            loadings[comp],
            explained[comp],
            topo_info,
            info_order,
            labels,
            pc_label,
            f"{consistency[comp].n_agreeing}/{consistency[comp].n_subjects}",
            title_suffix,
            plots_dir,
        )
    plot_timecourse_overlay(
        time_courses,
        explained,
        consistency,
        epoch_times,
        labels,
        unit,
        pc_labels,
        stim_end,
        title_suffix,
        plots_dir,
    )

    # ---- Component comparison ------------------------------------------------
    # PCA ranks components by explained channel variance, which is not the same
    # thing as carrying the steady-state, so score every component on the same
    # measures rather than assuming PC1 wins.
    comparison = component_comparison(
        time_courses,
        explained,
        consistency,
        pc_labels,
        stim_mask,
        sfreq,
        args.assr_freq,
        args.snr_threshold,
    )
    print(
        f"Component comparison (driven interval {int(stim_mask.sum())} samples, "
        f"{sfreq / stim_mask.sum():.1f} Hz spectral resolution):",
        flush=True,
    )
    print(comparison.round(3).to_string(), flush=True)

    # ---- Unreduced reference (no PCA) ----------------------------------------
    # The same trial averages before the channel reduction. No polarity alignment
    # is involved: the evoked voltage has a physical sign, so the group mean is
    # meaningful directly, which is what makes this a reference for the alignment
    # above. Structure a component shows that is absent here comes from the
    # reduction; structure here that no component reproduces is what it discarded.
    if args.unreduced_reference:
        gfp = global_field_power(evoked_all)  # (n_subj, win)
        group_evoked = evoked_all.mean(axis=0)  # (n_channels, win)
        group_gfp = global_field_power(group_evoked)  # (win,)

        plot_unreduced_butterfly_per_participant(
            evoked_all,
            gfp,
            epoch_times,
            labels,
            unit,
            stim_end,
            n_used,
            title_suffix,
            plots_dir,
        )
        plot_unreduced_group_timecourse(
            evoked_all,
            gfp,
            epoch_times,
            labels,
            unit,
            stim_end,
            title_suffix,
            plots_dir,
        )
        plot_unreduced_topomap_latencies(
            group_evoked,
            epoch_times,
            topo_info,
            info_order,
            unit,
            stim_end,
            title_suffix,
            plots_dir,
        )

        # Per-channel summaries: overall driven amplitude, and where the
        # steady-state actually is in channel space.
        chan_rms = np.sqrt((evoked_all[:, :, stim_mask] ** 2).mean(axis=2))
        group_chan_rms = np.sqrt((group_evoked[:, stim_mask] ** 2).mean(axis=1))
        chan_snr = np.array(
            [
                channel_assr_snr(evoked_all[subj], stim_mask, sfreq, args.assr_freq)
                for subj in range(n_subj)
            ]
        )
        group_chan_snr = channel_assr_snr(
            group_evoked, stim_mask, sfreq, args.assr_freq
        )

        plot_unreduced_summary_topomaps(
            chan_rms,
            group_chan_rms,
            topo_info,
            info_order,
            labels,
            "driven-interval RMS",
            f"RMS ({unit})",
            "raw_unreduced_topomap_driven_rms.png",
            title_suffix,
            plots_dir,
        )
        plot_unreduced_summary_topomaps(
            chan_snr,
            group_chan_snr,
            topo_info,
            info_order,
            labels,
            f"{args.assr_freq:.0f} Hz SNR",
            "SNR (a.u.)",
            "raw_unreduced_topomap_assr_snr.png",
            title_suffix,
            plots_dir,
        )

        # A ratio near 1 means the response averages cleanly across participants;
        # near 1/sqrt(n) means they are essentially independent and the group mean
        # is averaging noise.
        phase_ratio = float(
            group_gfp[stim_mask].mean() / gfp.mean(axis=0)[stim_mask].mean()
        )
        top_snr = np.argsort(group_chan_snr)[::-1][:5]
        group_component_snr = comparison[f"{args.assr_freq:.0f}Hz_SNR_group_mean"]
        print(
            f"Unreduced reference: driven-interval GFP of group mean / mean of GFPs "
            f"= {phase_ratio:.2f} (1.00 = fully phase-consistent across "
            f"participants, {1 / np.sqrt(n_subj):.2f} = 1/sqrt(n), independent).",
            flush=True,
        )
        print(
            f"  top channels by {args.assr_freq:.0f} Hz SNR on the group-mean "
            f"evoked: "
            + ", ".join(f"{channel_names[i]} {group_chan_snr[i]:.1f}" for i in top_snr),
            flush=True,
        )
        print(
            f"  best-channel SNR {np.nanmax(group_chan_snr):.1f} vs components: "
            + ", ".join(f"{pc} {group_component_snr[pc]:.1f}" for pc in pc_labels),
            flush=True,
        )

    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Whole-dataset ASSR raw-voltage channel-PCA reduction: collapse the "
            "channel dimension of the stimulus-locked evoked voltage with a PCA "
            "over channels and plot each leading component's time course and scalp "
            "topography per participant (polarity aligned across participants), "
            "plus a comparison table scoring the components against each other."
        )
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help=(
            "Number of leading channel-PCA components to extract, plot and compare. "
            "All are treated identically, so later components can be compared "
            "against PC1 rather than assumed uninformative (the highest-variance "
            "mode is not necessarily the best ASSR carrier). Use 1 for PC1 only. "
            "Clamped to the channel count."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[c.value for c in ConditionVariants],
        help="Condition to reduce.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=ExperimentNames.ASSR.value,
        choices=[e.value for e in ExperimentNames],
        help="Experiment dataset (default assr).",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=250.0,
        help="Sampling rate of the concatenated data.",
    )
    parser.add_argument(
        "--pre_pad",
        type=float,
        default=AssrEpoch.PRE_ONSET_S,
        help="Baseline kept before each onset (s).",
    )
    parser.add_argument(
        "--post_window",
        type=float,
        default=AssrEpoch.POST_ONSET_S,
        help=(
            f"Post-onset epoch length (s), capped by the shortest inter-onset gap. "
            f"Default {AssrEpoch.POST_ONSET_S} s = "
            f"{AssrEpoch.STIMULUS_DURATION_S} s stimulus + "
            f"{AssrEpoch.POST_STIMULUS_S} s post-stimulus."
        ),
    )
    parser.add_argument(
        "--assr_freq",
        type=float,
        default=40.0,
        help="Expected steady-state frequency (Hz), scored by the comparison table.",
    )
    parser.add_argument(
        "--snr_threshold",
        type=float,
        default=3.0,
        help="Per-subject ASSR SNR counted as a hit in the comparison table.",
    )
    parser.add_argument(
        "--no_zscore_per_channel",
        dest="zscore_per_channel",
        action="store_false",
        help="Run PCA on raw µV (covariance-PCA) instead of z-scoring each channel "
        "against the whole recording (correlation-PCA, equal electrode "
        "influence).",
    )
    parser.set_defaults(zscore_per_channel=True)
    parser.add_argument(
        "--no_unreduced_reference",
        dest="unreduced_reference",
        action="store_false",
        help="Skip the unreduced (no-PCA) channel-space reference figures: "
        "per-participant butterfly + GFP, group-mean topography over time, and "
        "per-channel driven RMS / ASSR SNR topomaps. They are the test reference "
        "the component panels are checked against, and cost no extra data read.",
    )
    parser.set_defaults(unreduced_reference=True)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Base directory for output plots. Defaults to plots/ root.",
    )
    return parser


if __name__ == "__main__":
    run_pca(build_parser().parse_args())
