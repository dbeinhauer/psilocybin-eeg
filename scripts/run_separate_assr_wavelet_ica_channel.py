"""Per-participant ("separate") channel-ICA of the stimulus-locked ASSR wavelet power.

Whole-cohort counterpart of
``notebooks/04-wavelet-ica-analysis/separate_assr_wavelet_ica_channel.ipynb``: the
notebook reads a subject subset so it stays interactive, this script streams the
cache once and decomposes **every** participant, writing the same figures.

Sibling of the *joint* channel-IVA analysis
(``scripts/run_wavelet_iva_channel.py`` /
``notebooks/05-wavelet-iva-analysis/wavelet_iva_channel.ipynb``): the mixing
dimension is again **channels** and the sample axis is again the joint
**frequency × time** plane, but the decomposition is run **independently for every
participant** instead of tying the subjects together with IVA. That is the point —
IVA forces one component index to mean the same thing in every subject, so a
subject whose 40 Hz response is spread over two modes has that fact absorbed into
the alignment, whereas here it stays visible per subject.

Pipeline (per subject, all of it inside one streaming pass over the cache):

1. **Z-score vs recording.** Each channel's per-frequency power is z-scored against
   the *whole* recording before epoching, removing the 1/f tilt so the channel
   decomposition is not dominated by absolute low-frequency power. Disable with
   ``--no_zscore_vs_recording``.
2. **Trial average.** The z-scored power is epoch-averaged around every ``fam+``
   onset, keeping ``--pre_pad`` before onset and the paradigm's post-onset span
   (0.5 s stimulus + 0.5 s post-stimulus, see
   :class:`src.definitions.constants.AssrEpoch`), capped by the shortest
   inter-onset gap so no epoch overlaps a neighbouring stimulus.

   The decomposition is fit on the trial average, not on single trials: the target
   is the stimulus-locked (evoked) pattern, and averaging first is what removes the
   induced, non-phase-locked background from the covariance the ICA sees.
3. **FastICA over channels.** Each ``(freq, time)`` bin is one sample and each
   channel one mixing variable; FastICA unmixes the ``(n_freqs*win, n_channels)``
   matrix directly.

   **There is no PCA step and none is needed.** ``FastICA(n_components=K)`` whitens
   by SVD and keeps its leading ``K`` directions — the same subspace a ``PCA(K)``
   would produce. Measured on this data, an explicit PCA first gives an identical
   subspace, identical retained variance, and identical components wherever
   FastICA converges (|r| = 1.000 at ``--n_ica 3``, 0.999 at 6). The reduction did
   not vanish; it lives inside FastICA, controlled by ``--n_ica``.

   Each IC yields a spectro-temporal score map ``(n_freqs, win)`` — the
   stimulus-locked TF map — and a **forward (mixing)** channel pattern
   ``ica.mixing_.T``. The latter is what belongs on a topomap; the unmixing rows
   are spatial *filters*, and plotting them instead is the classic
   filter-vs-pattern error (Haufe et al., 2014, NeuroImage 87:96-110). MNE follows
   the same convention: ``ica.get_components()`` returns the mixing matrix.
4. **Display sign and order.** ICA fixes neither, so both are imposed — and neither
   is a claim about the data. Sign: flip each IC so the largest absolute excursion
   of its TF map is positive, so each panel's dominant event reads as an increase.
   Order: by ``ic_variance`` descending, i.e. by how much of the subject's
   channel-space energy the component accounts for. Ranks are **subject-local**
   labels; with independent per-subject decompositions there is no reason
   subject A's ``IC1`` is subject B's ``IC1``.
5. **Stimulus correlation.** For every IC, the absolute Pearson correlation between
   its trial-averaged time course at ``--assr_freq`` and a binary stimulus
   regressor (1 while the stimulus is on, 0 otherwise) over the epoch window.

   The **absolute** value is what makes this a property of the component: flipping
   an IC flips its time course and hence the sign of *r*, so a signed value would
   report the sign convention from step 4 — which is set by the largest TF
   excursion and has nothing to do with the ASSR frequency. Read the magnitudes
   *relative to the other ICs of the same subject*: the trial average is smooth by
   construction, so a box regressor correlates appreciably with anything carrying a
   broad post-onset bump, and even a perfect steady state cannot reach 1 because
   the box has edges the wavelet's temporal smoothing cannot follow.

Results are written under a ``pca_<n_ica>`` subdirectory — the same layout the IVA
scripts use (``scripts/run_wavelet_iva_*.py``), so runs at different component
counts sit side by side instead of overwriting each other, and a count sweep reads
the same way everywhere in the project. In this script the number is the IC count,
which is also how many channel directions FastICA's whitening retains; there is no
separate PCA step, and the ``pca_`` prefix is kept only for cross-analysis
consistency.

Outputs (canonical stage-04 layout)::

    plots/04-wavelet-ica-analysis/<Condition>_ASSR/broadband/separate_assr_wavelet_ica_channel/
        pca_<n_ica>/
            separate_ica_<PSInnn>_components.png    # per participant: TF map + topomap per IC
            separate_ica_stimulus_correlation.png   # |r| bar panels, one per participant
            separate_ica_stimulus_correlation.csv   # long table, one row per (subject, IC)

Inputs (produced by stimulus alignment + the wavelet store jobs)::

    data/processed/<exp>/concatenated/<Condition>_ASSR.stimulus_onsets.npy
    data/processed/<exp>/concatenated/<Condition>_ASSR.metadata.csv
    data/processed/<exp>/wavelets/broadband/<Condition>_ASSR__wavelet_power__*__freqdim1.npz

One ``RAW_CROPPED`` recording header is also opened (``preload=False``, no sample
data) for the topomap electrode positions.

The wavelet cache is ~tens of GB and ``savez_compressed`` stores it as a single
deflate stream, so it is read **lazily** in one sequential pass: every channel
block is decompressed once and immediately reduced to its ``(n_freqs, win)`` trial
average, and each subject's channel stack is decomposed and freed before the next
subject is read. Peak memory is one subject's channel stack.

A note on ``--n_ica``: convergence is **subject-dependent** and degrades as the
count grows — measured on this data, 3 converges for every subject while 6 already
fails for some and 10 fails for all — and raising ``--ica_max_iter`` makes the
result drift further rather than settle. A non-converged unmixing is wherever the
solver stopped rather than a fixed point, which makes the component identities
unstable between runs, so prefer a count that converges and read the per-subject
convergence report. Component identities also shift with the count, so IC numbers
are only comparable within one ``pca_<n_ica>`` directory.

Usage::

    python scripts/run_separate_assr_wavelet_ica_channel.py --condition Placebo
    python scripts/run_separate_assr_wavelet_ica_channel.py \\
        --condition Psilocybin --n_ica 6
"""

import argparse
import sys
import warnings
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
from sklearn.decomposition import FastICA  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reused verbatim from the stage-00 channel-PCA script: the epoching, the lazy
# cache reader's metadata helpers and the montage lookup are the same objects
# here, and a second copy would be a second place for the ASSR epoch convention
# to drift.
from scripts.run_assr_wavelet_pca import (  # noqa: E402
    build_topomap_montage,
    epoch_average,
    read_wavelet_metadata,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    MusicTypeVariants,
    PreprocessedDataVariants,
    SingleDataMetadata,
)

mne.set_log_level("ERROR")

# Metadata CSV column names (sidecar stores enum keys as their string repr).
_PID_COL = str(SingleDataMetadata.PARTICIPANT_ID)
_PIDX_COL = str(SingleDataMetadata.CONCATENATED_PERSON_INDEX)

_STAGE_DIR = "04-wavelet-ica-analysis"
_ANALYSIS_DIR = "separate_assr_wavelet_ica_channel"

# Panels per row in the per-participant component figure and the |r| bar figure.
_MAX_COMPONENT_COLS = 5
_MAX_BAR_COLS = 5

# The figure renderers force these: a caller running under seaborn's whitegrid
# theme would otherwise get grid lines across every heatmap, and MNE draws its
# head outline with rcParams["axes.edgecolor"], so a light theme edge colour
# makes the outline nearly invisible.
_PLOT_RC = {"axes.grid": False, "axes.edgecolor": "black"}


# --------------------------------------------------------------------------- #
#  Core numerics                                                              #
# --------------------------------------------------------------------------- #
def fit_channel_ica(
    matrix: np.ndarray,
    n_ica: int,
    n_freqs: int,
    win: int,
    random_state: int,
    *,
    algorithm: str = "parallel",
    fun: str = "logcosh",
    max_iter: int = 2000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """FastICA over the channel axis of one subject's trial average.

    No PCA is applied first: ``FastICA(n_components=n_ica)`` whitens by SVD and
    keeps its leading ``n_ica`` directions, which is the same subspace a
    ``PCA(n_ica)`` would produce, so an explicit reduction would be redundant.

    Args:
        matrix: ``(n_freqs*win, n_channels)`` samples-by-channels matrix.
        n_ica: Independent components to extract. Also sets the whitening
            truncation, since FastICA reduces the channel axis itself.
        n_freqs: Wavelet frequency count, for folding sources back into a map.
        win: Epoch length in samples, likewise.
        random_state: Seed for FastICA's random initial unmixing.
        algorithm: ``"parallel"`` (symmetric) or ``"deflation"`` (one at a time,
            often converging where the symmetric update oscillates).
        fun: Contrast function — ``"logcosh"``, ``"exp"`` or ``"cube"``.
        max_iter: Iteration cap.
        tol: Convergence tolerance.

    Returns:
        Tuple ``(tf_maps, patterns, ic_variance, retained, converged)``:

        - ``tf_maps`` ``(n_ica, n_freqs, win)`` component score maps. FastICA's
          ``unit-variance`` whitening fixes every source to unit variance, so
          these share a scale within a subject and a common colour limit is fair
          — the amplitude lives in ``patterns`` instead.
        - ``patterns`` ``(n_ica, n_channels)`` forward (mixing) channel
          topographies, ``ica.mixing_.T``. NOT the unmixing rows.
        - ``ic_variance`` ``(n_ica,)`` fraction of the total channel-space sum of
          squares each IC's rank-one back-projection accounts for. They sum to
          ``retained`` insofar as the sources are uncorrelated; read them as
          per-component weights, not as a partition.
        - ``retained`` fraction of channel variance the whitening truncation kept
          — the hard ceiling on everything the ICs can carry.
        - ``converged`` False if FastICA hit *max_iter* without converging.
    """
    ica = FastICA(
        n_components=n_ica,
        algorithm=algorithm,
        fun=fun,
        whiten="unit-variance",
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
    )
    # ConvergenceWarning is informative here, not noise: a non-converged unmixing
    # is wherever the solver stopped rather than a fixed point, so it is captured
    # and reported per subject rather than printed once and lost.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        sources = ica.fit_transform(matrix)  # (n_freqs*win, n_ica)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    patterns = ica.mixing_.T  # (n_ica, n_channels) — forward model
    tf_maps = sources.T.reshape(n_ica, n_freqs, win)

    centred = matrix - matrix.mean(axis=0)
    total_ss = float((centred**2).sum())
    if total_ss <= 0.0:
        return tf_maps, patterns, np.full(n_ica, np.nan), float("nan"), converged
    retained = 1.0 - float(((centred - sources @ patterns) ** 2).sum()) / total_ss
    ic_variance = (sources**2).sum(axis=0) * (patterns**2).sum(axis=1) / total_ss
    return tf_maps, patterns, ic_variance, retained, converged


def back_projection(tf_maps: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    """Sum over components of source ⊗ pattern — invariant to sign and order.

    Args:
        tf_maps: ``(S, K, F, win)`` component score maps.
        patterns: ``(S, K, C)`` forward channel patterns.

    Returns:
        ``(S, F*win, C)`` reconstruction of the retained channel data.
    """
    flat = tf_maps.reshape(tf_maps.shape[0], tf_maps.shape[1], -1)
    return np.einsum("skn,skc->snc", flat, patterns)


def orient_and_order_components(
    tf_maps: np.ndarray, patterns: np.ndarray, ic_variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pin the ICA sign ambiguity and order components by variance, per subject.

    Both operations are unavoidable and neither is a claim about the data:
    FastICA returns components in an arbitrary order with an arbitrary sign, and
    ``(map, pattern)`` and ``(-map, -pattern)`` are the same component.

    - *Sign* is fixed by making each IC's largest absolute TF excursion positive,
      so each panel's dominant event reads as a power increase. Deliberately
      hypothesis-free: it says nothing about the ASSR frequency, so it stays valid
      whatever is measured afterwards. An IC whose two largest excursions are
      near-equal has an effectively arbitrary sign, which is exactly why the
      stimulus correlation below is taken in absolute value.
    - *Order* is by ``ic_variance`` descending — a statement about size, not
      relevance: a 40 Hz steady state can easily sit below a broad onset response.

    Args:
        tf_maps: ``(S, K, F, win)`` score maps.
        patterns: ``(S, K, C)`` forward channel patterns.
        ic_variance: ``(S, K)`` per-IC variance fractions.

    Returns:
        Tuple ``(tf_maps, patterns, ic_variance, signs)`` — sign-corrected and
        reordered copies plus the ``(S, K)`` signs that were applied.
    """
    n_subj, n_ica = ic_variance.shape
    flat = tf_maps.reshape(n_subj, n_ica, -1)
    peak = np.take_along_axis(
        flat, np.abs(flat).argmax(axis=2)[..., np.newaxis], axis=2
    )[..., 0]
    signs = np.where(peak < 0.0, -1.0, 1.0)  # (S, K)
    tf_maps = tf_maps * signs[..., np.newaxis, np.newaxis]
    patterns = patterns * signs[..., np.newaxis]

    order = np.argsort(-ic_variance, axis=1, kind="stable")
    rows = np.arange(n_subj)[:, np.newaxis]
    return (
        tf_maps[rows, order],
        patterns[rows, order],
        ic_variance[rows, order],
        signs,
    )


def stimulus_abs_correlation(
    tf_maps: np.ndarray, freq_bin: int, regressor: np.ndarray
) -> np.ndarray:
    """|Pearson r| between each IC's time course at one frequency and a regressor.

    Absolute because flipping an IC flips its time course and therefore the sign
    of *r*: a signed value would report the display convention rather than a
    property of the component. Taking the magnitude makes the measure invariant to
    the ICA sign ambiguity, which is what lets the ranking mean anything. Nothing
    real is lost — "40 Hz power rises while the stimulus is on" and "falls while it
    is on" are the same component seen two ways, and which one a given IC is can be
    read off its TF map.

    Args:
        tf_maps: ``(..., n_ica, n_freqs, win)`` trial-averaged score maps.
        freq_bin: Index of the frequency row to correlate.
        regressor: ``(win,)`` regressor, here the binary stimulus box.

    Returns:
        ``(..., n_ica)`` absolute correlations, NaN where a component's time
        course is constant.
    """
    course = tf_maps[..., freq_bin, :]
    centred = course - course.mean(axis=-1, keepdims=True)
    reg = regressor - regressor.mean()
    denom = np.sqrt((centred**2).sum(axis=-1) * (reg**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denom > 0.0, np.abs((centred * reg).sum(axis=-1) / denom), np.nan
        )


def stream_wavelet_ica(
    npz_path: Path,
    n_channels: int,
    n_freqs: int,
    onsets: np.ndarray,
    pre: int,
    post: int,
    n_ica: int,
    random_state: int,
    *,
    ica_kwargs: dict | None = None,
    zscore_vs_recording: bool = True,
    max_subjects: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    """Stream the whole wavelet cache once, decomposing each subject as it arrives.

    ``savez_compressed`` writes ``data.npy`` as one sequential deflate stream, so
    it cannot be sliced randomly. This reader decompresses one
    ``(n_freqs, n_times)`` channel block at a time, z-scores it against the whole
    recording (optional) and epoch-averages it; once a subject's channel stack is
    complete it is decomposed and freed. Peak memory is one subject's
    ``(n_channels, n_freqs, win)`` array, not the multi-GB tensor.

    Args:
        npz_path: Path to the ``*__wavelet_power__*__freqdim1.npz`` cache.
        n_channels: Number of EEG channels in the cache.
        n_freqs: Number of wavelet frequencies in the cache.
        onsets: Stimulus onset sample indices.
        pre: Samples before each onset.
        post: Samples after each onset.
        n_ica: Independent components extracted per subject.
        random_state: Seed shared by every subject's FastICA.
        ica_kwargs: Extra keyword arguments forwarded to
            :func:`fit_channel_ica` (``algorithm``, ``fun``, ``max_iter``, ``tol``).
        zscore_vs_recording: Z-score each channel's per-frequency power against
            the whole recording before epoching (removes the 1/f tilt).
        max_subjects: Stop after this many subjects (debugging aid). The read is
            sequential, so this genuinely shortens the pass.

    Returns:
        Tuple ``(tf_maps, patterns, ic_variance, retained, non_converged,
        n_times)``: ``(S, n_ica, n_freqs, win)`` score maps, ``(S, n_ica,
        n_channels)`` forward patterns, ``(S, n_ica)`` variance fractions,
        ``(S,)`` retained fractions, the cache-order indices of subjects whose
        FastICA did not converge, and the cache's time length. Signs and order are
        arbitrary here — see :func:`orient_and_order_components`.
    """
    ica_kwargs = ica_kwargs or {}
    win = pre + post
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("data.npy") as fh:
            version = npformat.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = npformat.read_array_header_1_0(fh)
            else:
                shape, _, dtype = npformat.read_array_header_2_0(fh)
            n_subj_cache, n_feat_flat, n_times = shape
            if n_feat_flat != n_channels * n_freqs:
                raise ValueError(
                    f"feature axis {n_feat_flat} != n_channels*n_freqs "
                    f"{n_channels * n_freqs}"
                )
            n_subj = (
                n_subj_cache
                if max_subjects is None
                else min(n_subj_cache, max_subjects)
            )

            block_bytes = n_freqs * n_times * dtype.itemsize
            tf_maps = np.zeros((n_subj, n_ica, n_freqs, win), dtype=np.float64)
            patterns = np.zeros((n_subj, n_ica, n_channels), dtype=np.float64)
            ic_variance = np.zeros((n_subj, n_ica), dtype=np.float64)
            retained = np.zeros(n_subj, dtype=np.float64)
            non_converged: list[int] = []
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
                # Samples = (freq, time) bins, mixing variables = channels.
                matrix = stack.reshape(n_channels, n_freqs * win).T
                del stack
                (
                    tf_maps[subj],
                    patterns[subj],
                    ic_variance[subj],
                    retained[subj],
                    converged,
                ) = fit_channel_ica(
                    matrix, n_ica, n_freqs, win, random_state, **ica_kwargs
                )
                del matrix
                if not converged:
                    non_converged.append(subj)
                print(
                    f"  decomposed subject {subj + 1}/{n_subj}: {n_ica} ICs, "
                    f"retained {retained[subj] * 100:.1f}% of channel variance, "
                    f"top ic_variance {ic_variance[subj].max() * 100:.1f}%"
                    f"{'' if converged else '  <-- FastICA DID NOT CONVERGE'}",
                    flush=True,
                )
    return tf_maps, patterns, ic_variance, retained, non_converged, n_times


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def plot_subject_components(
    tf_maps: np.ndarray,
    patterns: np.ndarray,
    ic_variance: np.ndarray,
    topo_info: mne.Info,
    info_order: list[int],
    epoch_times: np.ndarray,
    freqs: np.ndarray,
    label: str,
    retained: float,
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """TF map over scalp topography for every IC of one subject.

    The two rows answer different halves of "what is this component": the TF map
    says *when and at which frequency* it is active, the topomap says *where on
    the scalp* it comes from. A band-limited stripe over a single electrode or the
    rim is an artefact the ICA has isolated, and the TF map alone would not
    distinguish it from a real response.

    Colour scales are shared across a subject's TF maps but **per IC** for the
    topomaps, and that asymmetry is forced by the decomposition rather than
    chosen: ``whiten="unit-variance"`` puts every source on one scale, which is
    exactly why the amplitude ends up in the pattern, so one shared topomap limit
    would render every low-variance component as a blank disc. Compare topomap
    *shape* across panels and use the ``var`` percentage for weight.

    Args:
        tf_maps: ``(K, F, win)`` score maps, sign- and order-corrected.
        patterns: ``(K, C)`` forward channel patterns, in cache channel order.
        ic_variance: ``(K,)`` per-IC share of channel-space energy.
        topo_info: Montage info supplying the electrode positions.
        info_order: Indices mapping the cache's channel order onto *topo_info*.
        epoch_times: ``(win,)`` epoch time axis in seconds, 0 at onset.
        freqs: ``(F,)`` wavelet frequencies.
        label: Participant label, used in the title and the file name.
        retained: Channel variance the whitening truncation kept.
        assr_freq: Steady-state frequency (Hz), marked on every TF map.
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
    """
    n_k = tf_maps.shape[0]
    ncols = min(_MAX_COMPONENT_COLS, n_k)
    n_blocks = int(np.ceil(n_k / ncols))
    vmax = float(np.abs(tf_maps).max()) or 1e-12
    extent = [epoch_times[0], epoch_times[-1], freqs[0], freqs[-1]]

    with plt.rc_context(_PLOT_RC):
        # Constrained layout: each TF panel carries axis labels directly above a
        # topomap with its own title, and a fixed grid collides the two.
        fig, axes = plt.subplots(
            2 * n_blocks,
            ncols,
            figsize=(3.4 * ncols, 6.4 * n_blocks),
            squeeze=False,
            layout="constrained",
        )
        tf_im = None
        for comp in range(n_k):
            block, col = divmod(comp, ncols)
            ax_tf, ax_topo = axes[2 * block, col], axes[2 * block + 1, col]

            tf_im = ax_tf.imshow(
                tf_maps[comp],
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax_tf.axvline(0.0, color="k", ls="--", lw=0.8)  # onset
            ax_tf.axvline(AssrEpoch.STIMULUS_DURATION_S, color="k", ls="--", lw=0.8)
            ax_tf.axhline(assr_freq, color="lime", ls=":", lw=1.2)
            ax_tf.set_title(
                f"IC{comp + 1}   var {ic_variance[comp] * 100:.1f}%", fontsize=9
            )
            ax_tf.set_xlabel("Time rel. onset (s)", fontsize=8)
            ax_tf.set_ylabel("Frequency (Hz)", fontsize=8)

            vlim = float(np.percentile(np.abs(patterns[comp]), 99)) or 1e-12
            mne.viz.plot_topomap(
                patterns[comp][info_order],
                topo_info,
                axes=ax_topo,
                show=False,
                cmap="RdBu_r",
                vlim=(-vlim, vlim),
                contours=4,
            )
            ax_topo.set_title(f"IC{comp + 1} pattern (±{vlim:.2g})", fontsize=8)

        for comp in range(n_k, n_blocks * ncols):
            block, col = divmod(comp, ncols)
            axes[2 * block, col].axis("off")
            axes[2 * block + 1, col].axis("off")

        fig.colorbar(
            tf_im,
            ax=axes.ravel().tolist(),
            shrink=0.4,
            label="IC score (unit variance)",
        )
        fig.suptitle(
            f"{label} — per-IC stimulus-locked TF map and scalp topography "
            f"({title_suffix})\n{n_k} ICs unmixed over {patterns.shape[1]} "
            f"channels; whitening retained {retained * 100:.1f}% of channel "
            f"variance; ICs ordered by variance, topomap scales are per-IC",
            fontsize=11,
        )
    fig.savefig(plots_dir / f"separate_ica_{label}_components.png", dpi=150)
    plt.close(fig)


def plot_stimulus_correlation(
    abs_r: np.ndarray,
    labels: list[str],
    n_used: list[int],
    assr_freq: float,
    title_suffix: str,
    plots_dir: Path,
) -> None:
    """Horizontal |r| bars per participant, components sorted descending.

    The panels are the cohort read-out this analysis exists for: a subject whose
    top bar stands clear of the rest has one component carrying the response,
    while two or three comparable bars are what a split looks like.

    Read magnitudes *within* a panel. A box regressor correlates appreciably with
    anything carrying a broad post-onset bump, and the trial average is smooth by
    construction, so mid-list values are close to what an unrelated component
    scores; only the separations at the top of a panel carry information.

    Args:
        abs_r: ``(S, K)`` absolute correlations, in the components' plotted order.
        labels: Participant labels, aligned with axis 0 of *abs_r*.
        n_used: Stimuli averaged per participant, for the panel titles.
        assr_freq: The frequency the correlation was taken at (Hz).
        title_suffix: Group description appended to the title.
        plots_dir: Directory the figure is written to.
    """
    n_subj, n_ica = abs_r.shape
    ncols = min(_MAX_BAR_COLS, n_subj)
    nrows = int(np.ceil(n_subj / ncols))
    xmax = max(1.05 * float(np.nanmax(abs_r)), 0.05)

    with plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.4 * ncols, (0.42 * n_ica + 2.4) * nrows),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        flat = axes.flatten()
        for subj, ax in enumerate(flat[:n_subj]):
            rank = np.argsort(-abs_r[subj])  # descending |r|
            y = np.arange(n_ica)
            ax.barh(y, abs_r[subj][rank], color="#4c72b0")
            ax.set_yticks(y, [f"IC{i + 1}" for i in rank], fontsize=8)
            ax.invert_yaxis()  # largest at the top
            ax.set_xlim(0.0, xmax)
            ax.set_title(
                f"{labels[subj]}  (trial average, n={n_used[subj]} stimuli)\n"
                f"max |r| = {np.nanmax(abs_r[subj]):.3f}",
                fontsize=10,
            )
            for yi, val in zip(y, abs_r[subj][rank]):
                ax.text(val + 0.008, yi, f"{val:.3f}", va="center", fontsize=7)
        for ax in flat[n_subj:]:
            ax.axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("|Pearson r| with stimulus box")
        for ax in axes[:, 0]:
            ax.set_ylabel("component (ordered by |r|)")
        fig.suptitle(
            f"Trial-averaged {assr_freq:.0f} Hz time course vs the stimulus box "
            f"— {title_suffix}\nabsolute correlation, so the ranking does not "
            f"depend on each IC's arbitrary sign",
            fontsize=11,
        )
    fig.savefig(plots_dir / "separate_ica_stimulus_correlation.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run_separate_ica(args: argparse.Namespace) -> None:
    """Run the per-participant channel ICA for one condition, writing all outputs."""
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
        absent = [p.name for p in missing] + (
            [] if wavelet_matches else ["wavelet npz"]
        )
        print(f"[SKIP] {label}: missing inputs ({absent}).", flush=True)
        return
    wavelet_path = wavelet_matches[0]

    save_root = Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH
    analysis_dir = save_root / _STAGE_DIR / label / "broadband" / _ANALYSIS_DIR

    print(f"=== {label} ===", flush=True)
    print(f"wavelet : {wavelet_path}", flush=True)

    # ---- Load shared inputs -------------------------------------------------
    onsets = np.load(onsets_path)
    meta = pd.read_csv(meta_path, index_col=0)
    gaps = np.diff(onsets)
    freqs, channel_names, n_freqs = read_wavelet_metadata(wavelet_path)
    n_channels = len(channel_names)

    idx_to_pid = dict(zip(meta[_PIDX_COL], meta[_PID_COL].astype(str).str.zfill(3)))

    sfreq = args.sfreq
    pre = int(round(args.pre_pad * sfreq))
    # Paradigm window, only capped by the shortest gap so epochs never overlap.
    post = min(int(round(args.post_window * sfreq)), int(gaps.min()))
    win = pre + post
    epoch_times = np.arange(-pre, post) / sfreq
    stim_mask = AssrEpoch.stimulus_mask(epoch_times)
    box = stim_mask.astype(float)
    assr_bin = int(np.argmin(np.abs(freqs - args.assr_freq)))

    # More components than channels is not decomposable; clamp instead of failing.
    n_ica = min(args.n_ica, n_channels)
    if n_ica < args.n_ica:
        print(
            f"  NOTE: --n_ica {args.n_ica} exceeds the {n_channels} available "
            f"channels; using {n_ica}.",
            flush=True,
        )

    # `pca_<n>` matches the IVA family's layout (run_wavelet_iva_*.py), so runs at
    # different component counts sit side by side instead of overwriting each
    # other. Built from the CLAMPED count, so the directory names what was
    # actually fit. Here the number is the IC count, which is also how many
    # channel directions FastICA's whitening retains — there is no separate PCA.
    plots_dir = analysis_dir / f"pca_{n_ica}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"plots   : {plots_dir}", flush=True)

    print(
        f"n_channels={n_channels}, n_freqs={n_freqs}, onsets={onsets.shape[0]}, "
        f"n_ica={n_ica} ({args.ica_algorithm}/{args.ica_fun}), "
        f"epoch window={win} samples ({pre} pre, {post} post) "
        f"= [{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s "
        f"(stimulus 0–{AssrEpoch.STIMULUS_DURATION_S:.2f} s), "
        f"samples per decomposition={n_freqs * win}, "
        f"correlation frequency={freqs[assr_bin]:.0f} Hz (bin {assr_bin}), "
        f"zscore_vs_recording={args.zscore_vs_recording}",
        flush=True,
    )
    if win > int(gaps.min()):
        print(
            f"  WARNING: epoch ({win} samples) exceeds the shortest inter-onset "
            f"gap ({int(gaps.min())}) — the pre-onset baseline reaches into the "
            f"previous stimulus.",
            flush=True,
        )

    # ---- Wavelet: single streaming pass, FastICA per subject ----------------
    print(f"Streaming wavelet cache ({wavelet_path.name}) ...", flush=True)
    (
        tf_maps,
        patterns,
        ic_variance,
        retained,
        non_converged,
        n_times_wav,
    ) = stream_wavelet_ica(
        wavelet_path,
        n_channels,
        n_freqs,
        onsets,
        pre,
        post,
        n_ica,
        args.random_state,
        ica_kwargs={
            "algorithm": args.ica_algorithm,
            "fun": args.ica_fun,
            "max_iter": args.ica_max_iter,
            "tol": args.ica_tol,
        },
        zscore_vs_recording=args.zscore_vs_recording,
        max_subjects=args.max_subjects,
    )
    n_subj = tf_maps.shape[0]
    print(
        f"Decomposed {n_subj} subject(s): tf_maps={tf_maps.shape}, "
        f"patterns={patterns.shape} (wavelet n_times={n_times_wav}).",
        flush=True,
    )

    # ---- Order subjects by participant ID -----------------------------------
    order = sorted(range(n_subj), key=lambda s: int(idx_to_pid.get(s, "9999")))
    tf_maps = tf_maps[order]
    patterns = patterns[order]
    ic_variance = ic_variance[order]
    retained = retained[order]
    labels = [f"PSI{idx_to_pid.get(s, '???')}" for s in order]

    if non_converged:
        print(
            f"  WARNING: FastICA did not converge for "
            f"{sorted(labels[order.index(i)] for i in non_converged)} within "
            f"{args.ica_max_iter} iterations at tol={args.ica_tol:g}, so their "
            f"unmixing is wherever the solver stopped rather than a fixed point — "
            f"component identities will not be reproducible between runs. "
            f"Convergence is subject-dependent and degrades as --n_ica grows; "
            f"raising --ica_max_iter makes the result drift further rather than "
            f"settle, so prefer a lower --n_ica, or try --ica_algorithm deflation.",
            flush=True,
        )

    # ---- Display sign & order ------------------------------------------------
    recon_before = back_projection(tf_maps, patterns)
    tf_maps, patterns, ic_variance, signs = orient_and_order_components(
        tf_maps, patterns, ic_variance
    )
    # Neither the flip nor the reordering may change what the components add up to.
    np.testing.assert_allclose(
        back_projection(tf_maps, patterns),
        recon_before,
        atol=1e-9,
        err_msg=(
            "Re-orienting or reordering changed the back-projection — the sign "
            "was not applied to the map and the pattern together."
        ),
    )
    del recon_before
    print(
        f"Flipped {int((signs < 0).sum())}/{signs.size} (subject, IC) pairs for "
        f"display; reordered by ic_variance. Back-projection unchanged.",
        flush=True,
    )

    # ---- Stimulus correlation ------------------------------------------------
    abs_r = stimulus_abs_correlation(tf_maps, assr_bin, box)  # (S, K)
    print(
        f"\n|r| of each IC's {freqs[assr_bin]:.0f} Hz trial-averaged time course "
        f"with the stimulus box (on for {int(box.sum())}/{box.size} samples):",
        flush=True,
    )
    for subj, subj_label in enumerate(labels):
        rank = np.argsort(-abs_r[subj])
        top = ", ".join(
            f"IC{i + 1} {abs_r[subj][i]:.3f}" for i in rank[: min(3, n_ica)]
        )
        print(
            f"  {subj_label}: {top}   (retained {retained[subj] * 100:.1f}%)",
            flush=True,
        )

    summary = pd.DataFrame(
        {
            "participant": np.repeat(labels, n_ica),
            "n_ica": n_ica,
            "ic_rank": np.tile(np.arange(1, n_ica + 1), n_subj),
            "abs_r_stimulus": abs_r.ravel(),
            "ic_variance_%": ic_variance.ravel() * 100.0,
            "retained_%": np.repeat(retained * 100.0, n_ica),
        }
    )
    csv_path = plots_dir / "separate_ica_stimulus_correlation.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Per-component table written to {csv_path}", flush=True)

    # ---- Topomap electrode positions from a RAW_CROPPED recording -----------
    topo_info, info_order = build_topomap_montage(experiment, meta, channel_names)
    print(
        f"Topomap montage: {len(topo_info['ch_names'])} electrodes "
        f"(reordered onto the wavelet channel order).",
        flush=True,
    )

    # ---- Plots ---------------------------------------------------------------
    title_suffix = f"{condition.value}/{music_type.value} (n={n_subj}), n_ica={n_ica}"
    for subj, subj_label in enumerate(labels):
        plot_subject_components(
            tf_maps[subj],
            patterns[subj],
            ic_variance[subj],
            topo_info,
            info_order,
            epoch_times,
            freqs,
            subj_label,
            float(retained[subj]),
            args.assr_freq,
            title_suffix,
            plots_dir,
        )
        print(f"  wrote components figure for {subj_label}", flush=True)

    # Every epoch that fits contributes to the trial average, and the window is
    # the same for every subject, so the count is shared.
    n_fitting = int((((onsets - pre) >= 0) & ((onsets + post) <= n_times_wav)).sum())
    n_used = [n_fitting] * n_subj
    plot_stimulus_correlation(
        abs_r, labels, n_used, args.assr_freq, title_suffix, plots_dir
    )
    print(f"[DONE] {label}: plots written to {plots_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-participant ('separate') channel ICA of the stimulus-locked ASSR "
            "wavelet power: FastICA over the channel axis of each subject's trial "
            "average, with every component's stimulus-locked TF map and scalp "
            "topography plotted per participant, plus the absolute correlation of "
            "each component's ASSR-frequency time course with a binary stimulus "
            "regressor. No PCA step — FastICA's whitening performs the channel "
            "reduction itself."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=ExperimentNames.ASSR.value,
        choices=[e.value for e in ExperimentNames],
        help="Experiment dataset (default assr).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[c.value for c in ConditionVariants],
        help="Condition to decompose.",
    )
    parser.add_argument(
        "--n_ica",
        type=int,
        default=10,
        help=(
            "Independent components per subject. Results go to a "
            "'pca_<n_ica>' subdirectory, the same layout the IVA scripts use, so "
            "runs at different counts sit side by side instead of overwriting "
            "each other. Clamped to the channel count. "
            "This is the ONLY dimensionality knob: FastICA whitens the channel "
            "matrix by SVD and keeps its leading --n_ica directions, so no "
            "separate PCA is needed. It is the dimensionality the independence "
            "assumption has to work in — too high and the ICA splits one response "
            "across several noisy components, too low and it cannot separate the "
            "steady state from an overlapping onset response. Convergence is "
            "subject-dependent and degrades as this grows, and a non-converged "
            "unmixing makes component identities irreproducible, so prefer a "
            "count that converges for every subject."
        ),
    )
    parser.add_argument(
        "--ica_algorithm",
        type=str,
        default="parallel",
        choices=["parallel", "deflation"],
        help=(
            "FastICA algorithm. 'deflation' extracts components one at a time and "
            "often converges where the symmetric 'parallel' update oscillates."
        ),
    )
    parser.add_argument(
        "--ica_fun",
        type=str,
        default="logcosh",
        choices=["logcosh", "exp", "cube"],
        help=(
            "FastICA contrast function. 'cube' (kurtosis) favours spiky sources, "
            "'exp' heavier tails."
        ),
    )
    parser.add_argument(
        "--ica_max_iter",
        type=int,
        default=2000,
        help=(
            "FastICA iteration cap. Non-convergence is reported per subject. "
            "Raising this does not help when the cause is too high --n_ica."
        ),
    )
    parser.add_argument(
        "--ica_tol",
        type=float,
        default=1e-4,
        help="FastICA convergence tolerance.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help=(
            "Seed for FastICA's random initial unmixing. FastICA is not "
            "deterministic without it."
        ),
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
        help=(
            "Steady-state frequency (Hz). Marked on every TF map, and the nearest "
            "wavelet bin is the row the stimulus correlation uses."
        ),
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help=(
            "Stop after this many subjects (debugging aid). The cache is read "
            "sequentially, so this genuinely shortens the pass."
        ),
    )
    parser.add_argument(
        "--no_zscore_vs_recording",
        dest="zscore_vs_recording",
        action="store_false",
        help=(
            "Decompose raw power instead of z-scoring each channel's per-frequency "
            "power against the whole recording. Without the z-scoring the 1/f tilt "
            "dominates the channel covariance and the leading components are "
            "low-frequency power, not the steady state."
        ),
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
    run_separate_ica(build_parser().parse_args())
