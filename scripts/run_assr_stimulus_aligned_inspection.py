"""Whole-dataset replication of the ASSR stimulus-aligned inspection notebook.

This is the headless, all-subjects / all-channels counterpart of
``notebooks/00-preprocessing/assr_stimulus_aligned_inspection.ipynb``. For a given
condition it inspects the **stimulus-aligned** ASSR data — both the raw
concatenated signal and the precomputed wavelet power — averaged over every
``fam+`` stimulus onset, and writes the diagnostic plots used to spot
per-participant outliers:

* raw stimulus-locked broadband GFP (global field power of the evoked response;
  overlay + one panel per participant) and channel-mean evoked response (signed
  mean across channels; overlay + one panel per participant);
* wavelet time-frequency power maps (per participant + group, z-scored per freq
  against the whole recording);
* 40 Hz ASSR-band power time course (per participant + group);
* stimulus-locked z-scored wavelet-power topomaps (per participant + group).

Inputs (produced by stimulus alignment + the wavelet store jobs)::

    data/processed/<exp>/concatenated/<Condition>_ASSR.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.stimulus_onsets.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.metadata.csv
    data/processed/<exp>/wavelets/broadband/<Condition>_ASSR__wavelet_power__*__freqdim1.npz

The wavelet cache is ~tens of GB and ``savez_compressed`` stores it as a single
deflate stream, so it is read **lazily** in one sequential pass: every channel
block is decompressed once and immediately reduced (channel-mean TF map + a signed
per-channel z-scored-power topomap value), keeping peak memory at one block.

Output plots::

    plots/00-preprocessing/assr_stimulus_aligned_inspection/<Condition>_ASSR/*.png
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
        arr: Array whose last axis is time, e.g. ``(n_channels, n_times)`` or
            ``(n_freqs, n_times)``.
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


def stream_wavelet_reduce(
    npz_path: Path,
    n_channels: int,
    n_freqs: int,
    onsets: np.ndarray,
    pre: int,
    post: int,
    topo_freq_mask: np.ndarray,
    topo_post_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream the whole wavelet-power cache once, reducing every channel block.

    ``savez_compressed`` writes ``data.npy`` (shape ``(n_subj, n_channels*n_freqs,
    n_times)``) as one sequential deflate stream, so it cannot be sliced randomly.
    This reader decompresses one ``(n_freqs, n_times)`` channel block at a time and
    immediately reduces it, so peak memory is a single block (~tens of MB) rather
    than the full multi-GB tensor.

    For each subject it accumulates two reductions, both built from the power
    **z-scored per frequency against the whole recording** (``zscore(block,
    axis=1)``) — which removes the 1/f tilt and references every frequency to the
    most stable baseline available:

    * the **channel-mean stimulus-locked TF map** ``(n_freqs, win)`` — the
      z-scored power epoch-averaged over onsets, then averaged across channels
      (for the time-frequency plots);
    * a **per-channel topomap scalar** — the same z-scored, epoch-averaged map
      mean-collapsed over ``topo_freq_mask`` and ``topo_post_mask`` (signed).

    Args:
        npz_path: Path to the ``*__wavelet_power__*__freqdim1.npz`` cache.
        n_channels: Number of EEG channels in the cache.
        n_freqs: Number of wavelet frequencies in the cache.
        onsets: Stimulus onset sample indices.
        pre: Samples before each onset.
        post: Samples after each onset.
        topo_freq_mask: Boolean mask over frequencies to collapse for the topomap.
        topo_post_mask: Boolean mask over the epoch window to collapse (post-onset).

    Returns:
        Tuple ``(power_maps, topo, n_times)`` where ``power_maps`` is
        ``(n_subj, n_freqs, win)``, ``topo`` is ``(n_subj, n_channels)`` and
        ``n_times`` is the wavelet time length.
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
            power_maps = np.zeros((n_subj, n_freqs, win), dtype=np.float64)
            topo = np.full((n_subj, n_channels), np.nan, dtype=np.float64)
            for subj in range(n_subj):
                tf_sum = np.zeros((n_freqs, win), dtype=np.float64)
                for chan in range(n_channels):
                    buf = fh.read(block_bytes)
                    block = np.frombuffer(
                        buf, dtype=dtype, count=n_freqs * n_times
                    ).reshape(n_freqs, n_times)
                    # Z-score each frequency against the whole recording, then
                    # epoch-average: the per-channel stimulus-locked TF map in
                    # units of SD relative to the recording baseline.
                    ev_z, _ = epoch_average(zscore(block, axis=1), onsets, pre, post)
                    tf_sum += ev_z  # channel-mean TF map
                    # Topomap scalar: collapse the same z-scored map over freq/time.
                    topo[subj, chan] = ev_z[topo_freq_mask][:, topo_post_mask].mean()
                power_maps[subj] = tf_sum / n_channels
                print(f"  reduced subject {subj + 1}/{n_subj}", flush=True)
    return power_maps, topo, n_times


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
def plot_gfp_overlay(
    gfp_curves: dict[int, np.ndarray],
    group_curve: np.ndarray,
    epoch_times: np.ndarray,
    labels: dict[int, str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Overlay of per-participant stimulus-locked broadband GFP + group average."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for subj, curve in gfp_curves.items():
        ax.plot(epoch_times, curve, lw=1.0, alpha=0.7, label=labels[subj])
    ax.plot(epoch_times, group_curve, lw=2.8, color="black", label="group average")
    ax.axvline(0.0, color="red", ls="--", lw=1, label="onset")
    ax.set_title(f"Stimulus-locked broadband GFP (evoked) — {title_suffix}")
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Global field power (spatial SD across channels)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_stimulus_locked_gfp.png", dpi=150)
    plt.close(fig)


def plot_gfp_per_participant(
    gfp_curves: dict[int, np.ndarray],
    group_curve: np.ndarray,
    epoch_times: np.ndarray,
    labels: dict[int, str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """One panel per participant: its broadband GFP vs the group average."""
    subjects = list(gfp_curves)
    nrows, ncols = grid_shape(len(subjects))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharey=True, squeeze=False
    )
    flat = axes.flatten()
    for ax, subj in zip(flat, subjects):
        ax.plot(epoch_times, group_curve, lw=1.0, color="0.6", ls="--")
        ax.plot(epoch_times, gfp_curves[subj], lw=1.6, color="C0")
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(labels[subj], fontsize=9)
    for ax in flat[len(subjects) :]:
        ax.axis("off")
    fig.suptitle(
        f"Per-participant stimulus-locked broadband GFP (evoked) — {title_suffix}",
        y=1.0,
    )
    fig.supxlabel("Time relative to onset (s)")
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_stimulus_locked_gfp_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_butterfly_overlay(
    evoked_curves: dict[int, np.ndarray],
    group_evoked: np.ndarray,
    epoch_times: np.ndarray,
    labels: dict[int, str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Overlay of per-participant channel-mean signed evoked response + group mean."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for subj, evoked in evoked_curves.items():
        ax.plot(epoch_times, evoked.mean(axis=0), lw=1.0, alpha=0.7, label=labels[subj])
    ax.plot(
        epoch_times,
        group_evoked.mean(axis=0),
        lw=2.8,
        color="black",
        label="group mean",
    )
    ax.axvline(0.0, color="red", ls="--", lw=1, label="onset")
    ax.set_title(f"Stimulus-locked channel-mean evoked response — {title_suffix}")
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_stimulus_locked_evoked_mean.png", dpi=150)
    plt.close(fig)


def plot_butterfly_per_participant(
    evoked_curves: dict[int, np.ndarray],
    group_evoked: np.ndarray,
    epoch_times: np.ndarray,
    labels: dict[int, str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """One panel per participant: channel-mean evoked vs group mean."""
    subjects = list(evoked_curves)
    nrows, ncols = grid_shape(len(subjects))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharey=True, squeeze=False
    )
    flat = axes.flatten()
    for ax, subj in zip(flat, subjects):
        ax.plot(epoch_times, group_evoked.mean(axis=0), lw=1.0, color="0.6", ls="--")
        ax.plot(epoch_times, evoked_curves[subj].mean(axis=0), lw=1.6, color="C0")
        ax.axvline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(labels[subj], fontsize=9)
    for ax in flat[len(subjects) :]:
        ax.axis("off")
    fig.suptitle(
        f"Per-participant channel-mean evoked response — {title_suffix}",
        y=1.0,
    )
    fig.supxlabel("Time relative to onset (s)")
    fig.tight_layout()
    fig.savefig(
        plots_dir / "raw_stimulus_locked_evoked_mean_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tf_per_participant(
    power_maps: np.ndarray,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    labels: dict[int, str],
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Grid of per-participant time-frequency maps (z-scored vs whole recording)."""
    n_subj = power_maps.shape[0]
    vmax = float(np.abs(power_maps).max())
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, subj in zip(flat, range(n_subj)):
        im = ax.imshow(
            power_maps[subj],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0.0, color="k", ls="--", lw=0.7)
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        ax.set_title(labels[subj], fontsize=9)
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="z-scored power")
    fig.suptitle(
        f"Per-participant stimulus-locked wavelet power (z-scored per freq) — "
        f"{title_suffix}",
        y=1.0,
    )
    fig.savefig(
        plots_dir / "wavelet_tf_per_participant.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_tf_group(
    power_maps: np.ndarray,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Group-average time-frequency map (z-scored vs whole recording)."""
    group_tf = power_maps.mean(axis=0)
    vmax = float(np.abs(group_tf).max())
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(
        group_tf,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0.0, color="k", ls="--", lw=0.8, label="onset")
    ax.axhline(assr_freq, color="green", ls=":", lw=1.2, label=f"{assr_freq:.0f} Hz")
    ax.set_title(
        f"Stimulus-locked wavelet power, averaged over participants "
        f"(z-scored per freq)\n{title_suffix}"
    )
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="z-scored power")
    fig.tight_layout()
    fig.savefig(plots_dir / "wavelet_tf_group_average.png", dpi=150)
    plt.close(fig)


def plot_assr_band(
    power_maps: np.ndarray,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    labels: dict[int, str],
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant + group ASSR-band power time course (z-scored vs recording)."""
    band = (freqs >= assr_freq - 2) & (freqs <= assr_freq + 2)
    n_subj = power_maps.shape[0]
    # power_maps is already z-scored vs the whole recording; just average the band.
    curves = np.stack([power_maps[s, band].mean(axis=0) for s in range(n_subj)])
    fig, ax = plt.subplots(figsize=(12, 6))
    for subj in range(n_subj):
        ax.plot(epoch_times, curves[subj], lw=1.0, alpha=0.7, label=labels[subj])
    ax.plot(
        epoch_times, curves.mean(axis=0), lw=2.8, color="black", label="group average"
    )
    ax.axvline(0.0, color="red", ls="--", lw=1, label="onset")
    ax.set_title(
        f"Stimulus-locked {assr_freq:.0f} Hz power (z-scored vs recording) — "
        f"{title_suffix}"
    )
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("z-scored ASSR-band power")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "wavelet_assr_band_timecourse.png", dpi=150)
    plt.close(fig)


def plot_topomaps(
    topo: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    labels: dict[int, str],
    freq_range: tuple[float, float],
    window: tuple[float, float],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Per-participant + group signed z-scored-power topomaps (nb05 convention)."""
    n_subj = topo.shape[0]
    panels = [(labels[s], topo[s][info_order]) for s in range(n_subj)]
    panels.append(("group average", topo.mean(axis=0)[info_order]))

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
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="z-scored power")
    fig.suptitle(
        f"Stimulus-locked z-scored wavelet power topography "
        f"({freq_range[0]:.0f}-{freq_range[1]:.0f} Hz, "
        f"[{window[0]}, {window[1]}] s) — {title_suffix}",
        y=1.0,
    )
    fig.savefig(
        plots_dir / "wavelet_stimulus_locked_topomaps.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run_inspection(args: argparse.Namespace) -> None:
    """Run the full inspection for one condition, writing all plots."""
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

    # The Psilocybin (or any) condition may not have been generated yet — skip
    # cleanly instead of failing the whole job.
    missing = [p for p in (raw_path, onsets_path, meta_path) if not p.exists()]
    if missing or not wavelet_matches:
        print(
            f"[SKIP] {label}: missing inputs "
            f"({[p.name for p in missing] + ([] if wavelet_matches else ['wavelet npz'])}).",
            flush=True,
        )
        return
    wavelet_path = wavelet_matches[0]

    save_root = Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH
    plots_dir = (
        save_root / "00-preprocessing" / "assr_stimulus_aligned_inspection" / label
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {label} ===", flush=True)
    print(f"raw     : {raw_path}", flush=True)
    print(f"wavelet : {wavelet_path}", flush=True)
    print(f"plots   : {plots_dir}", flush=True)

    # ---- Load shared inputs -------------------------------------------------
    raw_mm = np.load(raw_path, mmap_mode="r")  # (n_subj, n_ch, n_times)
    onsets = np.load(onsets_path)
    meta = pd.read_csv(meta_path, index_col=0)
    n_subj, n_channels, n_times_raw = raw_mm.shape

    freqs, channel_names, n_freqs = read_wavelet_metadata(wavelet_path)
    if len(channel_names) != n_channels:
        raise ValueError(
            f"wavelet channel count {len(channel_names)} != raw n_channels {n_channels}"
        )

    # Subject index -> participant label (ordered by concatenated person index).
    idx_to_pid = dict(zip(meta[_PIDX_COL], meta[_PID_COL].astype(str).str.zfill(3)))
    labels = {s: f"PSI{idx_to_pid.get(s, '???')}" for s in range(n_subj)}

    sfreq = args.sfreq
    pre = int(round(-args.epoch_tmin * sfreq))
    post = int(round(args.epoch_tmax * sfreq))
    epoch_times = np.arange(-pre, post) / sfreq
    topo_tmax = args.topo_tmax if args.topo_tmax is not None else args.epoch_tmax
    topo_post_mask = (epoch_times >= args.topo_tmin) & (epoch_times <= topo_tmax)
    topo_freq_mask = (freqs >= args.topo_freq_min) & (freqs <= args.topo_freq_max)
    print(
        f"n_subjects={n_subj}, n_channels={n_channels}, n_freqs={n_freqs}, "
        f"epoch=[{args.epoch_tmin}, {args.epoch_tmax}] s ({pre + post} samples)",
        flush=True,
    )

    # ---- Raw stimulus-locked broadband GFP (all channels) -------------------
    # ASSR is phase-locked, so average the signed epochs first (the evoked
    # response), then collapse channels with the spatial std (GFP). GFP is
    # sign-invariant; a signed channel mean would cancel under the average
    # reference and is not informative here.
    gfp_curves: dict[int, np.ndarray] = {}
    evoked_curves: dict[int, np.ndarray] = {}
    for subj in range(n_subj):
        evoked, n_used = epoch_average(
            np.asarray(raw_mm[subj]), onsets, pre, post
        )  # (n_ch, win)
        gfp_curves[subj] = evoked.std(axis=0)  # (win,)
        evoked_curves[subj] = evoked
    group_curve = np.mean([gfp_curves[s] for s in range(n_subj)], axis=0)
    group_evoked = np.mean([evoked_curves[s] for s in range(n_subj)], axis=0)
    print(f"Raw GFP: averaged {n_used} stimuli per participant.", flush=True)

    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj})"
    plot_gfp_overlay(
        gfp_curves, group_curve, epoch_times, labels, title_suffix, plots_dir
    )
    plot_gfp_per_participant(
        gfp_curves, group_curve, epoch_times, labels, title_suffix, plots_dir
    )
    plot_butterfly_overlay(
        evoked_curves, group_evoked, epoch_times, labels, title_suffix, plots_dir
    )
    plot_butterfly_per_participant(
        evoked_curves, group_evoked, epoch_times, labels, title_suffix, plots_dir
    )

    # ---- Wavelet: single streaming pass over the whole cache ----------------
    print(f"Streaming wavelet cache ({wavelet_path.name}) ...", flush=True)
    power_maps, topo, n_times_wav = stream_wavelet_reduce(
        wavelet_path,
        n_channels,
        n_freqs,
        onsets,
        pre,
        post,
        topo_freq_mask,
        topo_post_mask,
    )
    print(
        f"Wavelet reduced: power_maps={power_maps.shape}, topo={topo.shape} "
        f"(raw n_times={n_times_raw}, wavelet n_times={n_times_wav}).",
        flush=True,
    )

    plot_tf_per_participant(
        power_maps, epoch_times, freqs, labels, args.assr_freq, title_suffix, plots_dir
    )
    plot_tf_group(
        power_maps, epoch_times, freqs, args.assr_freq, title_suffix, plots_dir
    )
    plot_assr_band(
        power_maps, epoch_times, freqs, labels, args.assr_freq, title_suffix, plots_dir
    )

    # ---- Topomaps: electrode positions from a RAW_CROPPED recording ---------
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
        raise ValueError("RAW_CROPPED channels are not a subset of wavelet channels.")
    info_order = [name_pos[name] for name in topo_info["ch_names"]]

    plot_topomaps(
        topo,
        topo_info,
        info_order,
        labels,
        (args.topo_freq_min, args.topo_freq_max),
        (args.topo_tmin, topo_tmax),
        title_suffix,
        plots_dir,
    )
    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Whole-dataset ASSR stimulus-aligned inspection (raw + wavelet), "
            "averaged per stimulus onset. Writes diagnostic plots for "
            "per-participant outlier detection."
        )
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[c.value for c in ConditionVariants],
        help="Condition to inspect.",
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
        help="Sampling rate of the concatenated / wavelet data.",
    )
    parser.add_argument(
        "--epoch_tmin",
        type=float,
        default=-0.1,
        help="Epoch start relative to onset (s).",
    )
    parser.add_argument(
        "--epoch_tmax", type=float, default=1.0, help="Epoch end relative to onset (s)."
    )
    parser.add_argument(
        "--assr_freq",
        type=float,
        default=40.0,
        help="Expected steady-state frequency (Hz).",
    )
    parser.add_argument(
        "--topo_tmin",
        type=float,
        default=0.0,
        help="Topomap post-onset window start (s).",
    )
    parser.add_argument(
        "--topo_tmax",
        type=float,
        default=None,
        help="Topomap post-onset window end (s); default = epoch_tmax.",
    )
    parser.add_argument(
        "--topo_freq_min",
        type=float,
        default=1.0,
        help="Lowest frequency collapsed into the topomap (Hz).",
    )
    parser.add_argument(
        "--topo_freq_max",
        type=float,
        default=50.0,
        help="Highest frequency collapsed into the topomap (Hz).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Base directory for output plots. Defaults to plots/ root.",
    )
    return parser


if __name__ == "__main__":
    run_inspection(build_parser().parse_args())
