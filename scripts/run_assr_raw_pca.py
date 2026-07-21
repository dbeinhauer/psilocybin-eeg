"""Whole-dataset replication of the ASSR raw-voltage channel-PCA notebook.

Headless, all-subjects / all-channels counterpart of
``notebooks/00-preprocessing/assr_raw_pca_analysis.ipynb``. For a given condition
it reduces the stimulus-locked **evoked voltage** to a single spatial component per
participant by collapsing the channel dimension with a **PCA over channels**, then
plots the first-component time course and scalp topography.

Pipeline (per subject):

1. **Z-score per channel** (over the whole recording) so every electrode has unit
   variance and contributes equally to the channel PCA (correlation-PCA). Optional
   (``--no_zscore_per_channel`` keeps raw µV, a covariance-PCA).
2. **Trial average.** The signal is epoch-averaged around every ``fam+`` onset. The
   epoch keeps a fixed ``pre_pad`` before onset and a post-onset length equal to the
   shortest inter-onset interval, so no epoch overlaps the next stimulus.
3. **Channel PCA.** Each time sample is an observation and each channel a variable;
   PCA is fit over the ``(win, n_channels)`` matrix. PC1's score is the component's
   time course ``(win,)``; PC1's loading is its scalp topography ``(n_channels,)``.
4. **Polarity alignment across participants.** A PCA sign is arbitrary, so each
   subject's loading is aligned to a common template (iteratively-refined group
   mean) and the global orientation anchored to a reference channel (``Cz``),
   giving one consistent polarity; the score is flipped with the loading.

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

from src.definitions.constants import ProjectPaths  # noqa: E402
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


def channel_pca_component(evoked: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Collapse channels of an evoked map with PCA, keeping PC1.

    Args:
        evoked: ``(n_channels, win)`` stimulus-locked trial average.

    Returns:
        Tuple ``(time_course, loading, explained)``: PC1 score ``(win,)``, PC1
        loading ``(n_channels,)`` and PC1's explained-variance ratio. The sign is
        left arbitrary here; it is fixed across participants by ``align_polarity``.
    """
    matrix = evoked.T  # (win, n_channels): observations = time, variables = channels
    pca = PCA(n_components=1)
    time_course = pca.fit_transform(matrix)[:, 0]  # (win,)
    loading = pca.components_[0]  # (n_channels,)
    return time_course, loading, float(pca.explained_variance_ratio_[0])


def align_polarity(
    loading_mat: np.ndarray, channel_names: list[str], n_iter: int = 10
) -> tuple[np.ndarray, np.ndarray, str]:
    """Align PC1 polarity across participants to one consistent sign.

    Every subject's loading is aligned to a common template (the iteratively-
    refined group-mean loading), then the whole set's global orientation is
    anchored to a reference channel (``Cz`` if present, else the strongest template
    channel) so the sign is stable regardless of subject ordering.

    Args:
        loading_mat: ``(n_subj, n_channels)`` per-subject PC1 loadings.
        channel_names: Channel names aligned to the loading columns.
        n_iter: Template-refinement iterations.

    Returns:
        Tuple ``(aligned, signs, ref_name)``: sign-aligned loadings, the per-subject
        sign applied (``+1``/``-1``, to also flip the time courses) and the
        reference channel name.
    """
    aligned = loading_mat.copy()
    template = aligned[0].copy()
    for _ in range(n_iter):
        flips = np.sign(aligned @ template)
        flips[flips == 0] = 1.0
        aligned = aligned * flips[:, None]
        template = aligned.mean(axis=0)
    ref_name = (
        "Cz"
        if "Cz" in channel_names
        else channel_names[int(np.argmax(np.abs(template)))]
    )
    ref_idx = channel_names.index(ref_name)
    if template[ref_idx] < 0:
        aligned, template = -aligned, -template
    signs = np.sign((loading_mat * aligned).sum(axis=1))
    signs[signs == 0] = 1.0
    return aligned, signs, ref_name


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
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Grid of per-participant PC1 time courses (ordered by participant ID)."""
    n_subj = time_courses.shape[0]
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), sharex=True, squeeze=False
    )
    flat = axes.flatten()
    for ax, subj in zip(flat, range(n_subj)):
        ax.plot(epoch_times, time_courses[subj], lw=1.5, color="C0")
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(f"{labels[subj]}  (PC1 {explained[subj] * 100:.0f}%)", fontsize=9)
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.supxlabel("Time relative to onset (s)")
    fig.supylabel(f"PC1 projected signal ({unit}, polarity-aligned)")
    fig.suptitle(
        f"Per-participant channel-PCA first-component time course — {title_suffix}",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_pca_timecourse_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_timecourse_overlay(
    time_courses: np.ndarray,
    epoch_times: np.ndarray,
    labels: list[str],
    unit: str,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """All participants' PC1 time courses overlaid, with the group mean."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for subj, label in enumerate(labels):
        ax.plot(epoch_times, time_courses[subj], lw=1.1, alpha=0.75, label=label)
    ax.plot(epoch_times, time_courses.mean(axis=0), lw=2.6, color="black",
            label="group mean")
    ax.axvline(0.0, color="red", ls="--", lw=1, label="onset")
    ax.set_title(f"Channel-PCA first-component time course — {title_suffix}")
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel(f"PC1 projected signal ({unit}, polarity-aligned)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_pca_timecourse_overlay.png", dpi=150)
    plt.close(fig)


def plot_topomaps(
    loadings: np.ndarray,
    explained: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    labels: list[str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant + group PC1 scalp topographies (nb05 convention).

    Panel titles carry each subject's PC1 explained-variance ratio; the group panel
    carries the mean across subjects.
    """
    n_subj = loadings.shape[0]
    panels = [
        (f"{labels[s]}  (PC1 {explained[s] * 100:.0f}%)", loadings[s][info_order])
        for s in range(n_subj)
    ]
    mean_ev = float(np.mean(explained)) * 100
    panels.append(
        (f"group average  (mean PC1 {mean_ev:.0f}%)", loadings.mean(axis=0)[info_order])
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
            vals, topo_info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-vlim, vlim), contours=4,
        )
        ax.set_title(label, fontsize=9)
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="PC1 loading (a.u.)")
    fig.suptitle(
        f"Per-participant channel-PCA first-component topography — {title_suffix}",
        y=1.0,
    )
    fig.savefig(
        plots_dir / "raw_pca_topomap_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
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
    post = int(gaps.min())  # trim to shortest inter-onset gap: no epoch overlaps
    epoch_times = np.arange(-pre, post) / sfreq
    print(
        f"n_subj={n_subj}, n_channels={n_channels}, onsets={onsets.shape[0]}, "
        f"epoch window={pre + post} samples ({pre} pre, {post} post) "
        f"= [{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s, "
        f"zscore_per_channel={args.zscore_per_channel}",
        flush=True,
    )

    # ---- Per-subject channel PCA --------------------------------------------
    time_courses = np.zeros((n_subj, pre + post), dtype=np.float64)
    loadings = np.zeros((n_subj, n_channels), dtype=np.float64)
    explained = np.zeros(n_subj, dtype=np.float64)
    for subj in range(n_subj):
        sig = np.asarray(raw_mm[subj])  # (n_ch, n_times)
        if args.zscore_per_channel:
            sig = zscore(sig, axis=1)  # equal electrode influence (correlation-PCA)
        evoked, n_used = epoch_average(sig, onsets, pre, post)  # (n_ch, win)
        time_courses[subj], loadings[subj], explained[subj] = channel_pca_component(
            evoked
        )
        print(
            f"  reduced subject {subj + 1}/{n_subj} (PSI{idx_to_pid.get(subj, '???')},"
            f" {n_used} stimuli, PC1 {explained[subj] * 100:.1f}%)",
            flush=True,
        )

    # ---- Align PC1 polarity across participants -----------------------------
    loadings, signs, ref_name = align_polarity(loadings, channel_names)
    time_courses = time_courses * signs[:, None]
    print(
        f"Polarity aligned (reference channel {ref_name}; "
        f"{int((signs < 0).sum())} subject(s) flipped).",
        flush=True,
    )

    # ---- Order per-participant panels by participant ID ---------------------
    order = sorted(range(n_subj), key=lambda s: int(idx_to_pid.get(s, "9999")))
    time_courses = time_courses[order]
    loadings = loadings[order]
    explained = explained[order]
    labels = [f"PSI{idx_to_pid.get(s, '???')}" for s in order]

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

    # ---- Plots --------------------------------------------------------------
    unit = "z-scored" if args.zscore_per_channel else "µV"
    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj})"
    plot_timecourse_per_participant(
        time_courses, explained, epoch_times, labels, unit, title_suffix, plots_dir
    )
    plot_timecourse_overlay(
        time_courses, epoch_times, labels, unit, title_suffix, plots_dir
    )
    plot_topomaps(
        loadings, explained, topo_info, info_order, labels, title_suffix, plots_dir
    )
    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Whole-dataset ASSR raw-voltage channel-PCA reduction: collapse the "
            "channel dimension of the stimulus-locked evoked voltage with a PCA "
            "over channels and plot the first-component time course and scalp "
            "topography per participant (polarity aligned across participants)."
        )
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
    parser.add_argument("--sfreq", type=float, default=250.0,
                        help="Sampling rate of the concatenated data.")
    parser.add_argument("--pre_pad", type=float, default=0.1,
                        help="Pad kept before each onset (s); post length = "
                             "shortest inter-onset gap.")
    parser.add_argument(
        "--no_zscore_per_channel",
        dest="zscore_per_channel",
        action="store_false",
        help="Run PCA on raw µV (covariance-PCA) instead of z-scoring each channel "
             "against the whole recording (correlation-PCA, equal electrode "
             "influence).",
    )
    parser.set_defaults(zscore_per_channel=True)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Base directory for output plots. Defaults to plots/ root.")
    return parser


if __name__ == "__main__":
    run_pca(build_parser().parse_args())
