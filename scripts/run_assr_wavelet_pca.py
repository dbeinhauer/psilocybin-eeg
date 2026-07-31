"""Whole-dataset replication of the ASSR wavelet channel-PCA notebook.

This is the headless, all-subjects / all-channels counterpart of
``notebooks/00-preprocessing/assr_wavelet_pca_analysis.ipynb``. For a given
condition it reduces the stimulus-locked wavelet power to a few
**time-frequency (TF) maps per participant** by collapsing the channel dimension
with a **PCA over channels**, then plots each component's map per participant and
averaged over participants.

``--n_components`` (default 3) sets how many leading components are kept. All of
them are processed **identically** — same alignment, same diagnostics, same plots —
so later components can be compared against PC1 instead of assumed uninformative:
PCA orders components by explained *channel variance* over the whole TF plane,
which is mostly low-frequency bins, so a broad onset response can own PC1 while the
narrow 40 Hz steady-state lands in PC2 or PC3. Extra components are nearly free —
the single streaming pass over the cache dominates the runtime.

Pipeline (per subject):

1. **Z-score vs recording.** Each channel's per-frequency power is z-scored
   against the *whole* recording (``zscore(block, axis=1)``) before epoching,
   removing the 1/f tilt so the PCA is not dominated by absolute low-frequency
   power.
2. **Trial average.** The z-scored power is epoch-averaged around every ``fam+``
   onset. The epoch keeps a fixed ``pre_pad`` before onset and the paradigm's
   post-onset span — ``0.5`` s stimulus + ``0.5`` s post-stimulus (see
   :class:`src.definitions.constants.AssrEpoch`) — capped by the shortest
   inter-onset gap so no epoch overlaps a neighbouring stimulus.
3. **Channel PCA.** Each ``(freq, time)`` bin is an observation and each channel
   a variable; PCA is fit over the ``(n_freqs*win, n_channels)`` matrix and each
   kept component's score map, reshaped to ``(n_freqs, win)``, is stored.
4. **Polarity alignment across participants**, run **independently per component**
   (each component's sign is its own arbitrary choice). Each subject's channel
   loading is aligned to a common template (:mod:`src.analysis.pca_polarity`) and
   the map flipped with it. Aligning on the loading rather than on an ASSR-band
   reference keeps the convention well defined even when no 40 Hz response is
   present — and it has to be, because for later components a 40 Hz reference would
   be noise by construction. Subjects are ordered by participant ID.
5. **Scalp topographies.** Each component's loading is plotted as a topomap, per
   participant and averaged — the record of what the channel reduction kept, since
   every spatial pattern orthogonal to these topographies is discarded. The
   retained / discarded channel-variance split is printed alongside.
6. **Component comparison.** A summary table is printed with, per component, the
   mean explained variance, the topography-consistency diagnostic and the
   driven-minus-baseline contrast in the ASSR band and across all frequencies —
   the evidence for whether a later component is the better steady-state carrier.

Inputs (produced by stimulus alignment + the wavelet store jobs)::

    data/processed/<exp>/concatenated/<Condition>_ASSR.stimulus_onsets.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.metadata.csv
    data/processed/<exp>/wavelets/broadband/<Condition>_ASSR__wavelet_power__*__freqdim1.npz

One ``RAW_CROPPED`` recording header is also opened (``preload=False``, no sample
data) for the topomap electrode positions.

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


def channel_pca_tf_maps(
    trial_avg: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse channels of a trial-averaged map with PCA, keeping leading components.

    Args:
        trial_avg: ``(n_channels, n_freqs, win)`` stimulus-locked trial average.
        n_components: Number of leading components to keep.

    Returns:
        Tuple ``(tf_maps, loadings, explained)`` with shapes
        ``(n_components, n_freqs, win)``, ``(n_components, n_channels)`` and
        ``(n_components,)``: each component's score map, channel loading (its scalp
        topography) and explained-variance ratio. Signs are left arbitrary here;
        they are aligned across participants, per component, from the loadings by
        :func:`src.analysis.pca_polarity.align_pc1_signs`.
    """
    n_channels, n_freqs, win = trial_avg.shape
    # Observations = (freq, time) bins, variables = channels.
    matrix = trial_avg.reshape(n_channels, n_freqs * win).T  # (n_freqs*win, n_ch)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(matrix)  # (n_freqs*win, n_components)
    tf_maps = scores.T.reshape(n_components, n_freqs, win)
    return tf_maps, pca.components_, pca.explained_variance_ratio_


def driven_contrast(
    tf_map: np.ndarray,
    freq_mask: np.ndarray,
    stim_mask: np.ndarray,
    base_mask: np.ndarray,
) -> float:
    """Driven-minus-baseline score contrast, in units of the map's own SD.

    The SD normalisation is what makes components with different absolute score
    magnitudes comparable; the baseline subtraction is what stops a component that
    simply sits high in the band all epoch from scoring.

    Args:
        tf_map: ``(n_freqs, win)`` component score map.
        freq_mask: Boolean mask over frequencies to average within.
        stim_mask: Boolean mask selecting the driven samples of the epoch.
        base_mask: Boolean mask selecting the pre-onset baseline samples.

    Returns:
        The dimensionless contrast, or NaN for a constant map.
    """
    sd = float(tf_map.std())
    if sd == 0.0:
        return float("nan")
    band = tf_map[freq_mask]
    return float(band[:, stim_mask].mean() - band[:, base_mask].mean()) / sd


def component_comparison(
    tf_maps: np.ndarray,
    explained: np.ndarray,
    consistency: list,
    pc_labels: list[str],
    freqs: np.ndarray,
    stim_mask: np.ndarray,
    base_mask: np.ndarray,
    assr_freq: float,
    assr_halfwidth: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score every component on the same measures, for a side-by-side comparison.

    A later component is the better ASSR carrier only if it beats PC1 on the
    ASSR-band contrast *and* holds up on topography consistency: winning on contrast
    while failing consistency means subjects have 40 Hz in some second mode, but not
    in the same second mode, so its group map is not interpretable. Reading the
    broadband column next to the ASSR one separates a frequency-specific response
    from a broad onset/arousal response.

    Args:
        tf_maps: ``(n_components, n_subjects, n_freqs, win)`` aligned score maps.
        explained: ``(n_components, n_subjects)`` explained-variance ratios.
        consistency: Per-component
            :class:`src.analysis.pca_polarity.SignConsistency` results.
        pc_labels: Component names, e.g. ``["PC1", "PC2"]``.
        freqs: ``(n_freqs,)`` wavelet frequencies.
        stim_mask: Boolean mask selecting the driven samples of the epoch.
        base_mask: Boolean mask selecting the pre-onset baseline samples.
        assr_freq: Steady-state frequency (Hz).
        assr_halfwidth: Half-width of the ASSR band (Hz).

    Returns:
        Tuple ``(table, band_mask)``: the comparison DataFrame indexed by component
        name, and the frequency mask that defined the ASSR band.
    """
    n_components = tf_maps.shape[0]
    band_mask = np.abs(freqs - assr_freq) <= assr_halfwidth
    all_freqs = np.ones_like(band_mask, dtype=bool)

    def per_subject(freq_mask: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    driven_contrast(tf_map, freq_mask, stim_mask, base_mask)
                    for tf_map in tf_maps[c]
                ]
                for c in range(n_components)
            ]
        )

    def per_group(freq_mask: np.ndarray) -> np.ndarray:
        return np.array(
            [
                driven_contrast(
                    tf_maps[c].mean(axis=0), freq_mask, stim_mask, base_mask
                )
                for c in range(n_components)
            ]
        )

    table = pd.DataFrame(
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
            "assr_contrast_subject_median": np.nanmedian(
                per_subject(band_mask), axis=1
            ),
            "assr_contrast_group_map": per_group(band_mask),
            "broadband_contrast_subject_median": np.nanmedian(
                per_subject(all_freqs), axis=1
            ),
            "broadband_contrast_group_map": per_group(all_freqs),
        }
    ).set_index("component")
    return table, band_mask


def stream_wavelet_pca(
    npz_path: Path,
    n_channels: int,
    n_freqs: int,
    onsets: np.ndarray,
    pre: int,
    post: int,
    n_components: int,
    *,
    zscore_vs_recording: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stream the whole wavelet cache once, reducing each subject to its TF maps.

    ``savez_compressed`` writes ``data.npy`` (shape ``(n_subj, n_channels*n_freqs,
    n_times)``) as one sequential deflate stream, so it cannot be sliced randomly.
    This reader decompresses one ``(n_freqs, n_times)`` channel block at a time,
    z-scores it against the whole recording (optional) and epoch-averages it, then
    once a subject's channel stack is complete collapses it to its leading component
    TF maps and frees the stack. Peak memory is one subject's ``(n_channels,
    n_freqs, win)`` array, not the full multi-GB tensor. Keeping more components
    costs one PCA fit per subject either way — the streaming read dominates.

    Args:
        npz_path: Path to the ``*__wavelet_power__*__freqdim1.npz`` cache.
        n_channels: Number of EEG channels in the cache.
        n_freqs: Number of wavelet frequencies in the cache.
        onsets: Stimulus onset sample indices.
        pre: Samples before each onset.
        post: Samples after each onset.
        n_components: Number of leading components to keep per subject.
        zscore_vs_recording: Z-score each channel's per-frequency power against the
            whole recording before epoching (removes the 1/f tilt).

    Returns:
        Tuple ``(tf_maps, loadings, explained, n_times)``, component-major:
        ``tf_maps`` is ``(n_components, n_subj, n_freqs, win)``, ``loadings`` is
        ``(n_components, n_subj, n_channels)`` channel topographies, ``explained``
        is ``(n_components, n_subj)`` variance ratios and ``n_times`` is the wavelet
        time length. Signs are left arbitrary here and aligned across participants,
        per component, by the caller.
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
            # Component-major storage: axis 0 = component, axis 1 = subject. Every
            # component is produced by the same code path, so none is special-cased.
            tf_maps = np.zeros((n_components, n_subj, n_freqs, win), dtype=np.float64)
            loadings = np.zeros((n_components, n_subj, n_channels), dtype=np.float64)
            explained = np.zeros((n_components, n_subj), dtype=np.float64)
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
                (
                    tf_maps[:, subj],
                    loadings[:, subj],
                    explained[:, subj],
                ) = channel_pca_tf_maps(stack, n_components)
                del stack
                ev_summary = ", ".join(
                    f"PC{c + 1} {explained[c, subj] * 100:.1f}%"
                    for c in range(n_components)
                )
                print(
                    f"  reduced subject {subj + 1}/{n_subj} ({ev_summary})",
                    flush=True,
                )
    return tf_maps, loadings, explained, n_times


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


def build_topomap_montage(
    experiment: ExperimentNames,
    meta: pd.DataFrame,
    channel_names: list[str],
) -> tuple[mne.Info, list[int]]:
    """Electrode positions for the topomaps, remapped onto the cache's channel order.

    Positions come from one ``RAW_CROPPED`` recording's ``Info``; the header is read
    with ``preload=False``, so no sample data is loaded.

    Args:
        experiment: Experiment whose recordings hold the montage.
        meta: Concatenated-metadata sidecar (needs the filename column).
        channel_names: Canonical channel order of the wavelet cache.

    Returns:
        Tuple ``(topo_info, info_order)`` where ``info_order`` indexes a loading
        vector in *channel_names* order into the montage's channel order.

    Raises:
        ValueError: If the montage's channels are not a subset of *channel_names*.
    """
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
    return topo_info, [name_pos[name] for name in topo_info["ch_names"]]


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
    pc_label: str,
    agreement: str,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Grid of per-participant TF maps for ONE component, written to its own figure.

    ``tf_maps``/``explained``/``labels`` are aligned row-for-row and expected to be
    ordered by participant ID (matching the notebook). The colour scale is shared by
    this component's panels only: component scores shrink with component order by
    construction, so a scale shared across components would flatten the later ones.
    """
    n_subj = tf_maps.shape[0]
    vmax = float(np.abs(tf_maps).max()) or 1e-12
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    nrows, ncols = grid_shape(n_subj)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False
    )
    flat = axes.flatten()
    im = None
    for ax, subj in zip(flat, range(n_subj)):
        im = ax.imshow(
            tf_maps[subj],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0.0, color="k", ls="--", lw=0.7)
        ax.axhline(assr_freq, color="green", ls=":", lw=0.9)
        ax.set_title(
            f"{labels[subj]}  ({pc_label} {explained[subj] * 100:.0f}%)", fontsize=9
        )
    for ax in flat[n_subj:]:
        ax.axis("off")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.6, label=f"{pc_label} score (a.u.)"
    )
    fig.suptitle(
        f"Per-participant channel-PCA {pc_label} TF map — {title_suffix} "
        f"(agreeing {agreement})",
        y=1.0,
    )
    fig.savefig(
        plots_dir / f"wavelet_pca_{pc_label.lower()}_tf_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_component_topomaps(
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

    The TF maps show *when and at which frequency* a component is active; the loading
    shows *where on the scalp* it comes from — the record of what the channel
    reduction kept. A plausible evoked dipole is signal; a map concentrated on one
    electrode or the rim is a bad channel or edge artefact the PCA has isolated (and
    is therefore not contaminating the other components); a smooth full-scalp
    gradient is usually a reference or drift mode.
    """
    n_subj = loadings.shape[0]
    panels = [
        (
            f"{labels[s]}  ({pc_label} {explained[s] * 100:.0f}%)",
            loadings[s][info_order],
        )
        for s in range(n_subj)
    ]
    panels.append(
        (
            f"group average  (mean {pc_label} {float(np.mean(explained)) * 100:.0f}%)",
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
        plots_dir / f"wavelet_pca_{pc_label.lower()}_topomap_per_participant.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pca_group(
    tf_maps: np.ndarray,
    explained: np.ndarray,
    consistency: list,
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    assr_freq: float,
    pc_labels: list[str],
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Group-average TF map per component, one panel each.

    Signs are aligned across participants within each component, so an average is
    only meaningful where that component's spatial mode is consistent across
    participants — the panel titles carry the agreement counts printed by the
    polarity step. Each panel is scaled to its own maximum, so compare structure
    across panels, not intensity.
    """
    n_components = tf_maps.shape[0]
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]
    fig, axes = plt.subplots(
        1, n_components, figsize=(7 * n_components, 4.6), squeeze=False
    )
    for comp, pc_label in enumerate(pc_labels):
        ax = axes[0, comp]
        group_map = tf_maps[comp].mean(axis=0)
        gvmax = float(np.abs(group_map).max()) or 1e-12
        im = ax.imshow(
            group_map,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-gvmax,
            vmax=gvmax,
        )
        ax.axvline(0.0, color="k", ls="--", lw=0.8, label="onset")
        ax.axhline(
            assr_freq, color="green", ls=":", lw=1.2, label=f"{assr_freq:.0f} Hz"
        )
        ax.set_title(
            f"{pc_label} — mean EV {explained[comp].mean() * 100:.0f}%, agreeing "
            f"{consistency[comp].n_agreeing}/{consistency[comp].n_subjects}, "
            f"median r {consistency[comp].median_pairwise_r:+.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Time relative to onset (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.9, label=f"{pc_label} score (a.u.)")
    fig.suptitle(
        f"Channel-PCA component TF maps, averaged over participants — {title_suffix}",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(
        plots_dir / "wavelet_pca_tf_group_average.png", dpi=150, bbox_inches="tight"
    )
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
    # Paradigm window, only capped by the shortest gap so epochs never overlap.
    post = min(int(round(args.post_window * sfreq)), int(gaps.min()))
    epoch_times = np.arange(-pre, post) / sfreq
    stim_mask = AssrEpoch.stimulus_mask(epoch_times)  # driven interval
    base_mask = epoch_times < 0.0  # pre-onset baseline

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
        f"n_channels={n_channels}, n_freqs={n_freqs}, onsets={onsets.shape[0]}, "
        f"n_components={n_components}, "
        f"epoch window={pre + post} samples ({pre} pre, {post} post) "
        f"= [{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s "
        f"(stimulus 0–{AssrEpoch.STIMULUS_DURATION_S:.2f} s), "
        f"zscore_vs_recording={args.zscore_vs_recording}",
        flush=True,
    )
    if pre + post > int(gaps.min()):
        print(
            f"  WARNING: epoch ({pre + post} samples) exceeds the shortest "
            f"inter-onset gap ({int(gaps.min())}) — the pre-onset baseline reaches "
            f"into the previous stimulus.",
            flush=True,
        )

    # ---- Wavelet: single streaming pass, PCA per subject --------------------
    print(f"Streaming wavelet cache ({wavelet_path.name}) ...", flush=True)
    tf_maps, loadings, explained, n_times_wav = stream_wavelet_pca(
        wavelet_path,
        n_channels,
        n_freqs,
        onsets,
        pre,
        post,
        n_components,
        zscore_vs_recording=args.zscore_vs_recording,
    )
    n_subj = tf_maps.shape[1]
    print(
        f"Wavelet reduced: tf_maps={tf_maps.shape} (wavelet n_times={n_times_wav}).",
        flush=True,
    )

    # ---- Order subjects by participant ID ------------------------------------
    # Done BEFORE the polarity alignment, not just for the panels: the template is
    # seeded from the first subject, so for a weakly consistent component the
    # converged signs depend on subject order. Sorting first makes the script
    # reproduce the notebook, which works in participant-ID order throughout.
    order = sorted(range(n_subj), key=lambda s: int(idx_to_pid.get(s, "9999")))
    tf_maps = tf_maps[:, order]
    loadings = loadings[:, order]
    explained = explained[:, order]
    labels = [f"PSI{idx_to_pid.get(s, '???')}" for s in order]

    # ---- Align polarity across participants, per component -------------------
    # Flip every subject toward the group-mean template, maximising cross-subject
    # topography agreement so each group map averages consistently signed maps.
    # This replaces an ASSR-band sign reference, which silently degrades to a coin
    # flip whenever the 40 Hz response is absent — and would be noise by
    # construction for the later components. Each component's sign is independently
    # arbitrary, so each gets its own alignment pass.
    consistency = []
    for comp, pc_label in enumerate(pc_labels):
        signs, anchor = align_pc1_signs(loadings[comp], channel_names=channel_names)
        loadings[comp] = apply_pc1_signs(loadings[comp], signs)
        tf_maps[comp] = apply_pc1_signs(tf_maps[comp], signs)
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
                f"not a sign problem. Inspect before trusting the {pc_label} map.",
                flush=True,
            )

    # ---- How much the channel reduction kept ---------------------------------
    # The blunt version of "what was filtered out": everything orthogonal to the
    # kept components' topographies is gone. A large discarded fraction is not
    # automatically a problem (much of it is noise spread across electrodes), but
    # it bounds what any later analysis on these maps can see.
    retained = explained.sum(axis=0)  # (n_subj,) summed over kept components
    print(
        f"Channel variance retained by the {n_components} kept component(s): "
        f"mean {retained.mean() * 100:.1f}% "
        f"(range {retained.min() * 100:.1f}–{retained.max() * 100:.1f}%); "
        f"discarded mean {(1 - retained.mean()) * 100:.1f}%.",
        flush=True,
    )

    # ---- Topomap electrode positions from a RAW_CROPPED recording ------------
    topo_info, info_order = build_topomap_montage(experiment, meta, channel_names)
    print(
        f"Topomap montage: {len(topo_info['ch_names'])} electrodes "
        f"(reordered onto the wavelet channel order).",
        flush=True,
    )

    # ---- Plots (one per-participant figure per component) --------------------
    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj})"
    for comp, pc_label in enumerate(pc_labels):
        plot_pca_per_participant(
            tf_maps[comp],
            explained[comp],
            epoch_times,
            freqs,
            labels,
            args.assr_freq,
            pc_label,
            f"{consistency[comp].n_agreeing}/{consistency[comp].n_subjects}",
            title_suffix,
            plots_dir,
        )
        plot_component_topomaps(
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
    plot_pca_group(
        tf_maps,
        explained,
        consistency,
        epoch_times,
        freqs,
        args.assr_freq,
        pc_labels,
        title_suffix,
        plots_dir,
    )

    # ---- Component comparison ------------------------------------------------
    # PCA ranks components by explained channel variance over the whole TF plane,
    # which is not the same thing as carrying the steady-state, so score every
    # component on the same measures rather than assuming PC1 wins.
    comparison, band_mask = component_comparison(
        tf_maps,
        explained,
        consistency,
        pc_labels,
        freqs,
        stim_mask,
        base_mask,
        args.assr_freq,
        args.assr_halfwidth,
    )
    print(
        f"Component comparison (ASSR band {freqs[band_mask].min():.0f}–"
        f"{freqs[band_mask].max():.0f} Hz, {int(band_mask.sum())} bins; driven "
        f"{int(stim_mask.sum())} samples vs baseline {int(base_mask.sum())} "
        f"samples):",
        flush=True,
    )
    print(comparison.round(3).to_string(), flush=True)
    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Whole-dataset ASSR wavelet channel-PCA reduction: collapse the "
            "channel dimension of the stimulus-locked wavelet power with a PCA "
            "over channels and plot each leading component's TF map per participant "
            "and averaged over participants, plus a comparison table scoring the "
            "components against each other."
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
            "Clamped to the channel count. Costs no extra IO — the single streaming "
            "pass over the wavelet cache dominates the runtime."
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
        help="Sampling rate of the concatenated / wavelet data.",
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
        help="Expected steady-state frequency (Hz).",
    )
    parser.add_argument(
        "--assr_halfwidth",
        type=float,
        default=2.0,
        help=(
            "Half-width (Hz) of the ASSR band averaged by the comparison table's "
            "driven-minus-baseline contrast."
        ),
    )
    parser.add_argument(
        "--no_zscore_vs_recording",
        dest="zscore_vs_recording",
        action="store_false",
        help="Run PCA on raw power instead of z-scoring each channel's "
        "per-frequency power against the whole recording.",
    )
    parser.set_defaults(zscore_vs_recording=True)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Base directory for output plots. Defaults to plots/ root.",
    )
    return parser


if __name__ == "__main__":
    run_pca(build_parser().parse_args())
