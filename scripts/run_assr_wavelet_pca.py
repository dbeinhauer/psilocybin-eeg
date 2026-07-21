"""Whole-dataset replication of the ASSR wavelet channel-PCA notebook.

This is the headless, all-subjects / all-channels counterpart of
``notebooks/00-preprocessing/assr_wavelet_pca_analysis.ipynb``. For a given
condition it reduces the stimulus-locked wavelet power to a single
**time-frequency (TF) map per participant** by collapsing the channel dimension
with a **PCA over channels**, then plots the first-component map per participant
and averaged over participants.

Pipeline (per subject):

1. **Z-score vs recording.** Each channel's per-frequency power is z-scored
   against the *whole* recording (``zscore(block, axis=1)``) before epoching,
   removing the 1/f tilt so the PCA is not dominated by absolute low-frequency
   power.
2. **Trial average.** The z-scored power is epoch-averaged around every ``fam+``
   onset. The epoch keeps a fixed ``pre_pad`` before onset and a post-onset
   length equal to the shortest inter-onset interval, so no epoch overlaps the
   next stimulus.
3. **Channel PCA.** Each ``(freq, time)`` bin is an observation and each channel
   a variable; PCA is fit over the ``(n_freqs*win, n_channels)`` matrix and the
   first component's score map, reshaped to ``(n_freqs, win)``, is kept. Each map's
   arbitrary sign is flipped so the ASSR band (``assr_freq ± 2`` Hz) during the
   post-onset interval is positive (reads red on ``RdBu_r``, comparable across
   subjects). Per-participant panels are ordered by participant ID.

Inputs (produced by stimulus alignment + the wavelet store jobs)::

    data/processed/<exp>/concatenated/<Condition>_ASSR.stimulus_onsets.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.metadata.csv
    data/processed/<exp>/wavelets/broadband/<Condition>_ASSR__wavelet_power__*__freqdim1.npz

The wavelet cache is ~tens of GB and ``savez_compressed`` stores it as a single
deflate stream, so it is read **lazily** in one sequential pass: every channel
block is decompressed once and immediately reduced to its ``(n_freqs, win)``
trial average, keeping peak memory to a single subject's channel stack.

Output plots::

    plots/00-preprocessing/assr_wavelet_pca/<Condition>_ASSR/*.png
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
    ExperimentNames,
    MusicTypeVariants,
    PreprocessedDataVariants,
)

mne.set_log_level("ERROR")

# Metadata CSV column names (sidecar stores enum keys as their string repr).
_PID_COL = "SingleDataMetadata.PARTICIPANT_ID"
_PIDX_COL = "SingleDataMetadata.CONCATENATED_PERSON_INDEX"


# --------------------------------------------------------------------------- #
#  Core numerics                                                              #
# --------------------------------------------------------------------------- #
def epoch_average(
    arr: np.ndarray, onsets: np.ndarray, pre: int, post: int
) -> tuple[np.ndarray, int]:
    """Average fixed windows around each onset along the last (time) axis.

    Args:
        arr: Array whose last axis is time, e.g. ``(n_freqs, n_times)``.
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


def channel_pca_tf_map(
    trial_avg: np.ndarray, assr_band: np.ndarray, post_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    """Collapse channels of a trial-averaged map with PCA, keeping PC1.

    Args:
        trial_avg: ``(n_channels, n_freqs, win)`` stimulus-locked trial average.
        assr_band: Boolean ``(n_freqs,)`` mask selecting the ASSR band.
        post_mask: Boolean ``(win,)`` mask selecting the post-onset interval.

    Returns:
        Tuple ``(tf_map, explained)`` where ``tf_map`` is the ``(n_freqs, win)``
        first-component score map (its arbitrary sign flipped so the ASSR-band
        post-onset response is positive — reads red on ``RdBu_r`` and comparable
        across subjects) and ``explained`` is PC1's explained-variance ratio.
    """
    n_channels, n_freqs, win = trial_avg.shape
    # Observations = (freq, time) bins, variables = channels.
    matrix = trial_avg.reshape(n_channels, n_freqs * win).T  # (n_freqs*win, n_ch)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(matrix)[:, 0]  # (n_freqs*win,)
    tf_map = scores.reshape(n_freqs, win)
    # Fix arbitrary sign: make the ASSR-band post-onset response positive (red).
    ref = tf_map[assr_band][:, post_mask].mean()
    tf_map = tf_map * np.sign(ref)
    return tf_map, float(pca.explained_variance_ratio_[0])


def stream_wavelet_pca(
    npz_path: Path,
    n_channels: int,
    n_freqs: int,
    onsets: np.ndarray,
    pre: int,
    post: int,
    assr_band: np.ndarray,
    post_mask: np.ndarray,
    *,
    zscore_vs_recording: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream the whole wavelet cache once, reducing each subject to a PC1 TF map.

    ``savez_compressed`` writes ``data.npy`` (shape ``(n_subj, n_channels*n_freqs,
    n_times)``) as one sequential deflate stream, so it cannot be sliced randomly.
    This reader decompresses one ``(n_freqs, n_times)`` channel block at a time,
    z-scores it against the whole recording (optional) and epoch-averages it, then
    once a subject's channel stack is complete collapses it to a single PC1 TF map
    and frees the stack. Peak memory is one subject's ``(n_channels, n_freqs,
    win)`` array, not the full multi-GB tensor.

    Args:
        npz_path: Path to the ``*__wavelet_power__*__freqdim1.npz`` cache.
        n_channels: Number of EEG channels in the cache.
        n_freqs: Number of wavelet frequencies in the cache.
        onsets: Stimulus onset sample indices.
        pre: Samples before each onset.
        post: Samples after each onset.
        assr_band: Boolean ``(n_freqs,)`` mask selecting the ASSR band (sign ref).
        post_mask: Boolean ``(win,)`` mask selecting the post-onset interval.
        zscore_vs_recording: Z-score each channel's per-frequency power against the
            whole recording before epoching (removes the 1/f tilt).

    Returns:
        Tuple ``(tf_maps, explained, n_times)`` where ``tf_maps`` is
        ``(n_subj, n_freqs, win)``, ``explained`` is ``(n_subj,)`` PC1 variance
        ratios and ``n_times`` is the wavelet time length.
    """
    win = pre + post
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("data.npy") as fh:
            version = npformat.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = npformat.read_array_header_1_0(fh)
            else:
                shape, _, dtype = npformat.read_array_header_2_0(fh)
            n_subj, n_feat_flat, n_times = shape
            if n_feat_flat != n_channels * n_freqs:
                raise ValueError(
                    f"feature axis {n_feat_flat} != n_channels*n_freqs "
                    f"{n_channels * n_freqs}"
                )

            block_bytes = n_freqs * n_times * dtype.itemsize
            tf_maps = np.zeros((n_subj, n_freqs, win), dtype=np.float64)
            explained = np.zeros(n_subj, dtype=np.float64)
            for subj in range(n_subj):
                stack = np.empty((n_channels, n_freqs, win), dtype=np.float64)
                for chan in range(n_channels):
                    buf = fh.read(block_bytes)
                    block = np.frombuffer(
                        buf, dtype=dtype, count=n_freqs * n_times
                    ).reshape(n_freqs, n_times)
                    if zscore_vs_recording:
                        block = zscore(block, axis=1)
                    ev, _ = epoch_average(block, onsets, pre, post)  # (n_freqs, win)
                    stack[chan] = ev
                tf_maps[subj], explained[subj] = channel_pca_tf_map(
                    stack, assr_band, post_mask
                )
                del stack
                print(
                    f"  reduced subject {subj + 1}/{n_subj} "
                    f"(PC1 {explained[subj] * 100:.1f}%)",
                    flush=True,
                )
    return tf_maps, explained, n_times


# --------------------------------------------------------------------------- #
#  IO helpers                                                                 #
# --------------------------------------------------------------------------- #
def read_wavelet_metadata(npz_path: Path) -> tuple[np.ndarray, list[str], int]:
    """Read the small metadata members and channel names from a wavelet cache.

    Args:
        npz_path: Path to the wavelet-power ``.npz`` cache.

    Returns:
        Tuple ``(freqs, channel_names, n_freqs)``. ``channel_names`` is recovered
        from ``feature_names`` (layout ``channel * n_freqs + freq``).
    """
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("freqs.npy") as fh:
            freqs = npformat.read_array(fh, allow_pickle=True)
        with zf.open("feature_names.npy") as fh:
            feature_names = npformat.read_array(fh, allow_pickle=True)
    n_freqs = len(freqs)
    channel_names = [
        str(feature_names[i * n_freqs]).split("@")[0]
        for i in range(len(feature_names) // n_freqs)
    ]
    return np.asarray(freqs, dtype=float), channel_names, n_freqs


def grid_shape(n_panels: int, max_cols: int = 5) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for a roughly square panel grid."""
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def plot_pca_per_participant(
    tf_maps: np.ndarray,
    explained: np.ndarray,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    labels: list[str],
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Grid of per-participant channel-PCA first-component TF maps.

    ``tf_maps``/``explained``/``labels`` are aligned row-for-row and expected to be
    ordered by participant ID (matching the notebook).
    """
    n_subj = tf_maps.shape[0]
    vmax = float(np.abs(tf_maps).max())
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, subj in zip(flat, range(n_subj)):
        im = ax.imshow(
            tf_maps[subj], aspect="auto", origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.axvline(0.0, color="k", ls="--", lw=0.7)
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        ax.set_title(f"{labels[subj]}  (PC1 {explained[subj] * 100:.0f}%)", fontsize=9)
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="PC1 score (a.u.)")
    fig.suptitle(
        f"Per-participant channel-PCA first-component TF map — {title_suffix}",
        y=1.0,
    )
    fig.savefig(
        plots_dir / "wavelet_pca_tf_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pca_group(
    tf_maps: np.ndarray,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Group-average channel-PCA first-component TF map.

    Each subject's PCA sign is fixed independently (ASSR-band post-onset response
    positive), so the average is only meaningful where the dominant spatial mode
    is consistent across participants.
    """
    group_map = tf_maps.mean(axis=0)
    gvmax = float(np.abs(group_map).max())
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(
        group_map, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r",
        vmin=-gvmax, vmax=gvmax,
    )
    ax.axvline(0.0, color="k", ls="--", lw=0.8, label="onset")
    ax.axhline(assr_freq, color="green", ls=":", lw=1.2, label=f"{assr_freq:.0f} Hz")
    ax.set_title(
        f"Channel-PCA first-component TF map, averaged over participants\n"
        f"{title_suffix}"
    )
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="PC1 score (a.u.)")
    fig.tight_layout()
    fig.savefig(plots_dir / "wavelet_pca_tf_group_average.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run_pca(args: argparse.Namespace) -> None:
    """Run the full channel-PCA reduction for one condition, writing all plots."""
    experiment = ExperimentNames(args.experiment)
    condition = ConditionVariants(args.condition)
    music_type = MusicTypeVariants.ASSR
    label = f"{condition.value}_{music_type.value}"

    processed_dir = ProjectPaths.PROCESSED_DATA_DIR / experiment.value
    concat_dir = processed_dir / PreprocessedDataVariants.CONCATENATED.value
    onsets_path = concat_dir / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
    meta_path = concat_dir / f"{label}.metadata.csv"
    wavelet_dir = processed_dir / "wavelets" / "broadband"
    wavelet_matches = sorted(
        wavelet_dir.glob(f"{label}__wavelet_power__*__freqdim1.npz")
    )

    # A condition may not have been generated yet — skip cleanly instead of failing.
    missing = [p for p in (onsets_path, meta_path) if not p.exists()]
    if missing or not wavelet_matches:
        print(
            f"[SKIP] {label}: missing inputs "
            f"({[p.name for p in missing] + ([] if wavelet_matches else ['wavelet npz'])}).",
            flush=True,
        )
        return
    wavelet_path = wavelet_matches[0]

    save_root = Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH
    plots_dir = save_root / "00-preprocessing" / "assr_wavelet_pca" / label
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {label} ===", flush=True)
    print(f"wavelet : {wavelet_path}", flush=True)
    print(f"plots   : {plots_dir}", flush=True)

    # ---- Load shared inputs -------------------------------------------------
    onsets = np.load(onsets_path)
    meta = pd.read_csv(meta_path, index_col=0)
    gaps = np.diff(onsets)
    freqs, channel_names, n_freqs = read_wavelet_metadata(wavelet_path)
    n_channels = len(channel_names)

    # Subject index -> participant label (ordered by concatenated person index).
    idx_to_pid = dict(zip(meta[_PIDX_COL], meta[_PID_COL].astype(str).str.zfill(3)))

    sfreq = args.sfreq
    pre = int(round(args.pre_pad * sfreq))
    post = int(gaps.min())  # trim to shortest inter-onset gap: no epoch overlaps
    epoch_times = np.arange(-pre, post) / sfreq
    # Sign-reference region: ASSR band (assr_freq ± 2 Hz) in the post-onset interval.
    assr_band = (freqs >= args.assr_freq - 2) & (freqs <= args.assr_freq + 2)
    post_mask = epoch_times >= 0.0
    print(
        f"n_channels={n_channels}, n_freqs={n_freqs}, onsets={onsets.shape[0]}, "
        f"epoch window={pre + post} samples ({pre} pre, {post} post) "
        f"= [{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s, "
        f"zscore_vs_recording={args.zscore_vs_recording}",
        flush=True,
    )

    # ---- Wavelet: single streaming pass, PCA per subject --------------------
    print(f"Streaming wavelet cache ({wavelet_path.name}) ...", flush=True)
    tf_maps, explained, n_times_wav = stream_wavelet_pca(
        wavelet_path,
        n_channels,
        n_freqs,
        onsets,
        pre,
        post,
        assr_band,
        post_mask,
        zscore_vs_recording=args.zscore_vs_recording,
    )
    n_subj = tf_maps.shape[0]
    print(
        f"Wavelet reduced: tf_maps={tf_maps.shape} (wavelet n_times={n_times_wav}).",
        flush=True,
    )

    # Order per-participant panels by participant ID (matches the notebook).
    order = sorted(range(n_subj), key=lambda s: int(idx_to_pid.get(s, "9999")))
    tf_maps = tf_maps[order]
    explained = explained[order]
    labels = [f"PSI{idx_to_pid.get(s, '???')}" for s in order]

    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj})"
    plot_pca_per_participant(
        tf_maps, explained, epoch_times, freqs, labels, args.assr_freq,
        title_suffix, plots_dir,
    )
    plot_pca_group(
        tf_maps, epoch_times, freqs, args.assr_freq, title_suffix, plots_dir
    )
    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Whole-dataset ASSR wavelet channel-PCA reduction: collapse the "
            "channel dimension of the stimulus-locked wavelet power with a PCA "
            "over channels and plot the first-component TF map per participant "
            "and averaged over participants."
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
                        help="Sampling rate of the concatenated / wavelet data.")
    parser.add_argument("--pre_pad", type=float, default=0.1,
                        help="Pad kept before each onset (s); post length = "
                             "shortest inter-onset gap.")
    parser.add_argument("--assr_freq", type=float, default=40.0,
                        help="Expected steady-state frequency (Hz).")
    parser.add_argument(
        "--no_zscore_vs_recording",
        dest="zscore_vs_recording",
        action="store_false",
        help="Run PCA on raw power instead of z-scoring each channel's "
             "per-frequency power against the whole recording.",
    )
    parser.set_defaults(zscore_vs_recording=True)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Base directory for output plots. Defaults to plots/ root.")
    return parser


if __name__ == "__main__":
    run_pca(build_parser().parse_args())
