"""
Decomposition-quality scoring for the channel-as-independent IVA workflow.

Scores every ``(subject, component)`` pair of a channel-IVA decomposition
against an independent reference, so a component can be judged on whether it
actually captures the stimulus response rather than on cross-subject
consistency alone.

**The reference pair** (:func:`wavelet_reference`) is *one* channel-PCA of the
trial-averaged, time-z-scored wavelet power — the reduction of
``notebooks/00-preprocessing/assr_wavelet_pca_analysis.ipynb`` and
``scripts/run_assr_wavelet_pca.py``, applied to the IVA's own input tensor. PC1's
channel loading is the reference *topography*, its score map the reference
*time-frequency map*, and one per-subject sign aligns both, so the two are two
views of the same thing.

The reference is deliberately in the **same modality as the decomposition**.
Wavelet power is a non-negative, slowly varying envelope; an evoked *voltage*
topography is a signed, phase-locked average of the raw signal. A channel pattern
recovered from power is not comparable with one recovered from voltage — the
polarity of a voltage topography has no counterpart in power — so a raw-voltage
reference cannot say whether a component reproduces the response, only whether two
unrelated quantities happen to co-vary across channels.

Scored against that reference:

- **x — topomap correlation.** Pearson correlation between the component's
  forward channel pattern and the group PC1 **channel loading**. Both sides are
  *patterns*, never unmixing filters; see
  :func:`src.analysis.wavelet_ica.iva_component_patterns`.
- **y — TF-map correlation.** The component's *onset-averaged* ``(F, W)``
  time-frequency map against the group PC1 **score map**, as one flattened
  Pearson correlation, so frequency is never collapsed. Also computed **per
  frequency band** (:data:`TF_BANDS`: 1-10 Hz and 30-50 Hz), which asks the same
  question where the response is expected rather than over the whole spectrum —
  see :func:`tf_map_correlation`.
- **y — onset-locked time correlation.** The component's ``(F, T)`` source is
  reduced to a single time course, averaged around every stimulus onset, and
  correlated (zero-lag Pearson) with a **rigid boxcar**: 0 before onset, 1 for
  :data:`RESP_DURATION_S` after it, 0 afterwards. A cruder y than the TF map — it
  collapses frequency — but it tests the *timing* against the paradigm itself
  rather than against a data-derived reference.

Every y shares the one x, and every score carries one shared per-component sign,
so any y may be read against it: changing one axis at a time separates the spatial
question (does the pattern match the reference topography?) from the temporal one
(the paradigm's window, the whole TF map, or one band of it).

Onset-locking is what makes the time axis work. ASSR stimulation is
*continuous*, so a plain on/off indicator over the whole recording fails — a
sustained response is flat once z-scored. Averaging around onsets and scoring
against a window fixed at the stimulus length means every component is measured
against one and the same reference instead of a per-component fitted one. Every
reference and every component map is cut on the **same** epoch
(:func:`onset_window`), so no two quality axes are scored on different windows.

The analysis needs stimulus onsets, so it applies to the ASSR experiment only.

**Signs.** IVA fixes each component's polarity only up to a per-``(subject,
component)`` flip — the sources across subjects are *dependent*, not correlated,
and flipping a pattern together with its source changes nothing about the data.
The stored score arrays are oriented with one flip per component, which leaves
participants free to disagree with each other; anything that *compares, averages
or plots participants* needs more than that, or participants of opposite polarity
cancel: :func:`anchor_signs_reference` resolves the flip per ``(subject,
component)`` pair by correlating that pair's **topography** with the reference
topography and flipping when the correlation comes out negative.

That is **one** sign per pair, and every view applies it — the same array orients
:meth:`IvaQualityResult.topomap_view`, :meth:`~IvaQualityResult.tf_view` and the
``aligned=True`` scores of :meth:`~IvaQualityResult.variants` /
:meth:`~IvaQualityResult.wavelet_variants`. A pattern and its source are only
interpretable together: flipping one without the other describes a decomposition
the IVA never produced, so a pair whose topomap and TF map carry different flips
cannot be read pair by pair at all — and neither can a scatter point that scores
a pair the figures draw under the opposite sign. The topography decides it
because the reference topography is a single fixed target every participant can
be compared against, and because that is the axis every score is measured on
(:attr:`~IvaQualityResult.topo_corr` then reads non-negative in the views, which
is what makes the annotated ``r`` describe the map beside it and puts every
scatter point in the right-hand half-plane, with the y saying whether that pair's
source follows its topography). :func:`anchor_signs_tf` still computes the
driven-response polarity, but
only as a **diagnostic**: :meth:`IvaQualityResult.anchor_agreement` reports how
often the shared sign also leaves a pair's :data:`ASSR_FREQ` stimulus-window
response positive, which is a statement about that pair's physiology rather than
about the alignment.

:attr:`IvaQualityResult.full_tf_mean` carries those same shared signs over the
**whole recording** instead of the onset epoch. It is the QC counterpart of the
standard IVA time-frequency figure, which averages the raw IVA sources under
their arbitrary per-``(subject, component)`` polarity and so is not the same
picture — under mixed signs a group mean cancels, and the two figures side by
side say how much of it was cancellation.

Everything a figure averages across participants is first put on a common
per-participant scale by :func:`equalize_subject_influence`, so no participant
carries more weight in a group mean than any other — see that function for why
the raw amplitudes are a nuisance here.

All functions here are **pure**: no I/O, no plotting, no mutation of inputs.
Rendering lives in :mod:`src.visualization.iva_quality_plots`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from src.analysis.pca_polarity import (
    SignConsistency,
    align_pc1_signs,
    apply_pc1_signs,
    topography_consistency,
)
from src.analysis.wavelet_ica import normalize_patterns_per_subject
from src.definitions.constants import AssrEpoch

# ---------------------------------------------------------------------------
# Reference-topomap settings
# ---------------------------------------------------------------------------

REF_POLARITY_CHANNEL = "Cz"  # anchors the reference topomap's global sign

# ---------------------------------------------------------------------------
# Onset-locked time-reference settings
# ---------------------------------------------------------------------------

ASSR_FREQ = 40.0  # frequency (Hz) used by the "40 Hz-only" time variant
EPOCH_PRE_S = AssrEpoch.PRE_ONSET_S  # pre-onset baseline in the averaged epoch
EPOCH_POST_S = AssrEpoch.POST_ONSET_S  # post-onset span (capped at shortest gap)
RESP_DURATION_S = AssrEpoch.STIMULUS_DURATION_S  # rigid expected-response window

# ---------------------------------------------------------------------------
# Band-limited TF-map variants
# ---------------------------------------------------------------------------

# The TF-map score restricted to a frequency interval, as ``(name, fmin, fmax,
# tag)``. The whole-map score asks "does this component match the reference
# anywhere in the spectrum"; these ask it *where the response is expected*, and
# the two ranges answer different questions:
#
# - 1-10 Hz — the onset transient and the slow evoked shape. A component can
#   score well here while carrying no steady-state response at all.
# - 30-50 Hz — the band bracketing the :data:`ASSR_FREQ` steady state, wide
#   enough to survive wavelet smearing and a few Hz of stimulator drift.
#
# Read alongside the whole-map score: high on 30-50 Hz but low over the whole map
# means the component is the ASSR and little else, which the whole-map score
# dilutes with every frequency the reference is quiet in.
TF_BANDS: tuple[tuple[str, float, float, str], ...] = (
    ("1-10 Hz", 1.0, 10.0, "tf_1_10hz"),
    ("30-50 Hz", 30.0, 50.0, "tf_30_50hz"),
)

# ---------------------------------------------------------------------------
# Per-participant sign anchoring (display alignment)
# ---------------------------------------------------------------------------

# What resolves a ``(subject, component)`` pair's polarity, for the figures that
# name their alignment. One sign per pair, from its topography against the
# reference (:func:`anchor_signs_reference`), shared by every view of that pair.
SUBJECT_SIGN_ANCHOR = "wavelet PC1 reference topomap correlation"

# Half-width (Hz) of the band around :data:`ASSR_FREQ` whose stimulus-window
# power anchors the reference pair's global polarity and backs the driven-response
# diagnostic (see :func:`anchor_signs_tf`). Wide enough that a coarse wavelet grid
# still has a bin inside it, and that a few Hz of wavelet smearing or stimulator
# drift cannot move the response out of it.
TF_ANCHOR_HALFWIDTH_HZ = 5.0

# ---------------------------------------------------------------------------
# Cross-participant weighting
# ---------------------------------------------------------------------------

# Percentile of |value| defining a subject's amplitude scale in
# :func:`subject_scales`. Robust to a single hot bin, unlike the maximum.
SUBJECT_SCALE_PERCENTILE = 99.0


# ---------------------------------------------------------------------------
# Onset epoching
# ---------------------------------------------------------------------------


def epoch_average(
    arr: np.ndarray, onsets: np.ndarray, pre: int, post: int
) -> tuple[np.ndarray, int]:
    """Average fixed windows around each onset along the **last** axis.

    Windows that would run off either end of the recording are dropped.

    :param arr: Array whose last axis is time; leading axes are preserved.
    :param onsets: Onset sample indices.
    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :return: ``(average, n_used)`` — the averaged window and how many onsets
        contributed.
    :raises ValueError: If no onset window fits inside the recording.
    """
    n_time = arr.shape[-1]
    acc, n_used = None, 0
    for o in onsets:
        start, end = int(o) - pre, int(o) + post
        if start < 0 or end > n_time:
            continue
        seg = arr[..., start:end]
        acc = seg.astype(np.float64) if acc is None else acc + seg
        n_used += 1
    if n_used == 0:
        raise ValueError("No onset window fits inside the recording.")
    return acc / n_used, n_used


def subtract_epoch_baseline(
    averaged: np.ndarray, baseline_mask: np.ndarray
) -> np.ndarray:
    """Reference an epoch-averaged map to its own pre-onset window, **per frequency**.

    Only the mean is removed; nothing is divided. On a TF map built from data that was
    already z-scored per (channel, frequency) before the decomposition, the 1/f falloff
    is gone before the map exists, so a per-frequency *divisor* buys little — and costs
    a lot at the bottom of the map, where a 100 ms baseline spans under one cycle and
    its SD is mostly wavelet phase rather than noise. Dividing by that would manufacture
    texture in the delta/theta corner. Subtracting the mean is the part that is sound at
    every frequency.

    **Why this is equivalent to per-trial subtraction.**
    :func:`epoch_average` is a plain mean over trials, and the mean is linear, so
    ``mean_n(x_n - b_n) == mean_n(x_n) - mean_n(b_n)`` and ``mean_n(b_n)`` *is* the
    baseline of the averaged map. Referencing each trial to its own baseline before
    averaging therefore gives exactly this — so the correction is applied here, once, on
    the averaged map, instead of re-epoching. (The equality needs the mean; it would not
    hold for a median over trials.)

    The reduction is over the last axis only, so every leading axis — recordings,
    components and **frequency** — keeps its own baseline.

    :param averaged: ``(..., frequencies, epoch samples)`` epoch-averaged map, time last.
    :param baseline_mask: Boolean ``(epoch samples,)`` selecting the pre-onset window.
    :return: A **new** array of the same shape, each frequency referenced to its own
        pre-onset mean.
    :raises ValueError: If *baseline_mask* does not match the epoch axis or selects
        nothing.
    """
    data = np.asarray(averaged, dtype=float)
    mask = np.asarray(baseline_mask, dtype=bool)
    if mask.ndim != 1 or mask.size != data.shape[-1]:
        raise ValueError(
            f"baseline_mask must be ({data.shape[-1]},) to match the epoch axis; got "
            f"{mask.shape}."
        )
    if not mask.any():
        raise ValueError("baseline_mask selects no pre-onset sample.")
    return data - data[..., mask].mean(axis=-1, keepdims=True)


def onset_window(onsets: np.ndarray, n_times: int, sfreq: float) -> tuple[int, int]:
    """``(pre, post)`` epoch length in samples; ``post`` never overlaps the next onset.

    :param onsets: Onset sample indices.
    :param n_times: Length of the time axis the epochs are cut from.
    :param sfreq: Sampling frequency in Hz.
    :return: ``(pre, post)`` sample counts around each onset.
    """
    onsets_in = onsets[onsets < n_times]
    min_gap = int(np.diff(onsets_in).min()) if len(onsets_in) > 1 else None
    return (
        AssrEpoch.pre_onset_samples(sfreq),
        AssrEpoch.post_onset_samples(sfreq, min_gap=min_gap),
    )


def onset_average(
    x: np.ndarray, onsets: np.ndarray, pre: int, post: int, n_times: int
) -> np.ndarray:
    """Onset-triggered average of a 1-D time course.

    :param x: ``(T,)`` time course.
    :param onsets: Onset sample indices.
    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :param n_times: Length of the usable time axis.
    :return: ``(pre + post,)`` onset-averaged response.
    :raises ValueError: If no onset window fits inside the time axis.
    """
    acc, n_used = None, 0
    for o in onsets:
        start, end = int(o) - pre, int(o) + post
        if start < 0 or end > n_times:
            continue
        seg = x[start:end]
        acc = seg.astype(np.float64) if acc is None else acc + seg
        n_used += 1
    if n_used == 0:
        raise ValueError("No onset window fits inside the IVA time window.")
    return acc / n_used


# ---------------------------------------------------------------------------
# The rigid boxcar reference
# ---------------------------------------------------------------------------


def boxcar(pre: int, win: int, d_samples: int) -> np.ndarray:
    """Expected response over the epoch: 0 baseline, 1 for ``d_samples`` post-onset.

    :param pre: Baseline length in samples (the onset sits at index ``pre``).
    :param win: Total epoch length in samples.
    :param d_samples: Length of the response window after the onset.
    :return: ``(win,)`` boxcar reference.
    """
    b = np.zeros(win)
    b[pre : pre + int(d_samples)] = 1.0
    return b


def response_duration_samples(post: int, sfreq: float) -> int:
    """The rigid response-window length in samples, clamped to the epoch's ``post``.

    :param post: Post-onset epoch length in samples.
    :param sfreq: Sampling frequency in Hz.
    :return: Response-window length in samples (at least 1).
    """
    return max(1, min(post, int(round(RESP_DURATION_S * sfreq))))


def boxcar_correlation(
    onset_avgs: np.ndarray, pre: int, win: int, d_samples: int
) -> np.ndarray:
    """Zero-lag Pearson correlation with the rigid onset-locked boxcar.

    The reference is identical for every subject and component, so the scores
    are directly comparable across both.

    :param onset_avgs: ``(S, K, W)`` onset-averaged component responses.
    :param pre: Baseline length in samples.
    :param win: Total epoch length in samples.
    :param d_samples: Response-window length in samples.
    :return: ``(S, K)`` correlations, with non-finite values mapped to 0.
    """
    n_subjects, n_comp, _ = onset_avgs.shape
    box = boxcar(pre, win, d_samples)
    time_corr = np.zeros((n_subjects, n_comp))
    for k in range(n_comp):
        time_corr[:, k] = np.nan_to_num(
            np.array([pearsonr(onset_avgs[s, k], box)[0] for s in range(n_subjects)])
        )
    return time_corr


# ---------------------------------------------------------------------------
# Wavelet reference (channel-PCA of the trial-averaged wavelet power)
# ---------------------------------------------------------------------------


def wavelet_reference(
    wavelet_power_z: np.ndarray,
    onsets: np.ndarray,
    pre: int,
    post: int,
    iva_ch_names: list[str],
    *,
    polarity_channel: str = REF_POLARITY_CHANNEL,
) -> tuple[np.ndarray, np.ndarray, str | None, SignConsistency]:
    """Group PC1 channel loading **and** TF score map of the wavelet response.

    The reduction of
    ``notebooks/00-preprocessing/assr_wavelet_pca_analysis.ipynb`` /
    ``scripts/run_assr_wavelet_pca.py``, applied to the IVA's own input tensor:
    per subject the time-z-scored power is trial-averaged around every onset to
    ``(C, F, W)``, then a **PCA over channels** (each ``(freq, time)`` bin is an
    observation, each channel a variable) collapses the channel axis. PC1's
    channel loading is the reference *topography*; its score map, reshaped to
    ``(F, W)``, is the reference *time-frequency map*.

    Loading and score map carry the **same** arbitrary PCA sign, so one
    per-subject sign (:func:`src.analysis.pca_polarity.align_pc1_signs`, anchored
    on *polarity_channel*) aligns both and keeps the two returned references
    mutually consistent. Group means are taken after that alignment.

    A PC1 loading's *global* sign is arbitrary. It is settled by
    *polarity_channel* when that channel is in the subset, and otherwise by "make
    the template's strongest channel positive" — either way one fixed rule, which
    is all that is needed now that this is the only reference: it sets the sign
    convention of every score computed against it, and the per-participant
    anchoring (:func:`anchor_signs_reference`) then follows the reference rather
    than imposing a second convention.

    The epoch is passed in rather than derived, so this reference is cut on the
    identical window as the component maps it is compared against.

    :param wavelet_power_z: ``(S, C, F, T)`` wavelet power already z-scored along
        time — the tensor the IVA consumed (see
        :func:`src.analysis.wavelet_ica.zscore_by_time`). Z-scoring is what
        removes the 1/f tilt, so the PCA is not dominated by absolute
        low-frequency power. **Not** mutated.
    :param onsets: Stimulus onset sample indices.
    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :param iva_ch_names: Channel names matching the tensor's channel axis, used
        to anchor the group's overall polarity.
    :param polarity_channel: Channel anchoring the global sign; falls back to the
        template's largest-magnitude channel when absent.
    :return: ``(ref_topo, ref_tf, anchor, consistency)`` with shapes ``(C,)`` and
        ``(F, pre + post)``. *consistency* scores how well the per-subject PC1
        topographies agree once aligned — a group reference built from
        disagreeing subjects is not a meaningful target.
    :raises ValueError: If *wavelet_power_z* is not 4-D, its channel axis does not
        match *iva_ch_names*, or no onset window fits inside the recording.
    """
    if wavelet_power_z.ndim != 4:
        raise ValueError(
            f"wavelet_power_z must be (S, C, F, T); got shape {wavelet_power_z.shape}."
        )
    n_subjects, n_channels, n_freqs, _ = wavelet_power_z.shape
    if n_channels != len(iva_ch_names):
        raise ValueError(
            f"wavelet_power_z has {n_channels} channels but {len(iva_ch_names)} "
            f"channel names were given."
        )
    win = pre + post

    loadings = np.zeros((n_subjects, n_channels))
    tf_maps = np.zeros((n_subjects, n_freqs, win))
    for s in range(n_subjects):
        # (C, F, W) trial average — one subject at a time, so the epoched copy
        # never exists for the whole cohort at once.
        trial_avg, _ = epoch_average(wavelet_power_z[s], onsets, pre, post)
        matrix = trial_avg.reshape(n_channels, n_freqs * win).T  # (F*W, C)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(matrix)  # (F*W, 1)
        tf_maps[s] = scores[:, 0].reshape(n_freqs, win)
        loadings[s] = pca.components_[0]

    signs, anchor = align_pc1_signs(
        loadings, channel_names=iva_ch_names, reference=polarity_channel
    )
    # The same signs on both: a subject's topography and its score map must not
    # end up mutually flipped, or the two reference axes would disagree.
    loadings = apply_pc1_signs(loadings, signs)
    tf_maps = apply_pc1_signs(tf_maps, signs)

    return (
        loadings.mean(axis=0),
        tf_maps.mean(axis=0),
        anchor,
        topography_consistency(loadings),
    )


def frequency_band_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Boolean ``(F,)`` mask of the frequency bins inside ``[fmin, fmax]``.

    Inclusive on both ends. An empty mask is a legitimate answer (the band lies
    outside the wavelet range), so the caller decides whether to skip the band —
    the same way :attr:`IvaQualityResult.have_40hz` guards the 40 Hz variant.

    :param freqs: ``(F,)`` frequency axis in Hz.
    :param fmin: Lower edge of the band in Hz, inclusive.
    :param fmax: Upper edge of the band in Hz, inclusive.
    :return: ``(F,)`` boolean mask, all ``False`` when no bin falls inside.
    :raises ValueError: If ``fmin`` exceeds ``fmax``.
    """
    if fmin > fmax:
        raise ValueError(f"fmin ({fmin}) must not exceed fmax ({fmax}).")
    axis = np.asarray(freqs, dtype=float)
    return (axis >= fmin) & (axis <= fmax)


def tf_map_correlation(
    onset_tf: np.ndarray,
    ref_tf: np.ndarray,
    freq_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Flattened Pearson correlation of component TF maps against one reference.

    The whole ``(F, W)`` map is one observation vector, so the score rewards a
    component that matches the reference in *both* frequency and time — a
    component with the right time course at the wrong frequency scores low.

    Pass ``freq_mask`` (see :func:`frequency_band_mask`) to score a **frequency
    band** instead: both sides are restricted to the selected rows before
    flattening, so the correlation only sees that band. This is what the
    :data:`TF_BANDS` variants are, and it changes the question from "does the
    component match the reference overall" to "does it match it where the
    response is expected" — a component matching the reference in one band only
    is diluted by every other frequency in the whole-map score.

    :param onset_tf: ``(S, K, F, W)`` onset-averaged component TF maps.
    :param ref_tf: ``(F, W)`` reference TF map.
    :param freq_mask: Optional ``(F,)`` boolean mask selecting the frequency rows
        to score. ``None`` uses the whole map.
    :return: ``(S, K)`` correlations, with non-finite values mapped to 0.
    :raises ValueError: If the map shapes disagree, or the mask does not match the
        frequency axis or selects nothing.
    """
    if onset_tf.ndim != 4:
        raise ValueError(f"onset_tf must be (S, K, F, W); got shape {onset_tf.shape}.")
    if onset_tf.shape[2:] != ref_tf.shape:
        raise ValueError(
            f"component TF maps {onset_tf.shape[2:]} do not match the reference "
            f"map {ref_tf.shape}."
        )
    if freq_mask is not None:
        freq_mask = np.asarray(freq_mask, dtype=bool)
        if freq_mask.shape != (ref_tf.shape[0],):
            raise ValueError(
                f"freq_mask shape {freq_mask.shape} does not match the "
                f"{ref_tf.shape[0]}-bin frequency axis."
            )
        if not freq_mask.any():
            raise ValueError("freq_mask selects no frequency bin.")
        onset_tf = onset_tf[:, :, freq_mask]
        ref_tf = ref_tf[freq_mask]
    n_subjects, n_comp = onset_tf.shape[:2]
    flat_ref = ref_tf.ravel()
    corr = np.zeros((n_subjects, n_comp))
    for s in range(n_subjects):
        for k in range(n_comp):
            corr[s, k] = pearsonr(onset_tf[s, k].ravel(), flat_ref)[0]
    return np.nan_to_num(corr)


# ---------------------------------------------------------------------------
# Per-participant sign anchoring
# ---------------------------------------------------------------------------
#
# IVA fixes each component's *sign* only up to a per-``(subject, component)``
# flip: the sources are statistically **dependent** across subjects, not
# correlated, and ``iva_g`` has no reason to prefer one polarity over the other
# (flipping a pattern and its source together leaves the data unchanged). The
# module's one-flip-per-component orientation (:attr:`~IvaQualityResult.
# sign_per_comp`) therefore leaves participants free to disagree *with each
# other* — and when they do, a figure comparing them is unreadable and their
# group mean cancels.
#
# :func:`anchor_signs_reference` resolves that flip **once per pair**, from the
# pair's own topography against the reference topography, and everything the pair
# owns — its channel pattern, its onset-averaged TF map, its whole-recording TF
# map, and the correlations annotating them — is flipped by that one sign. A
# pattern and its source belong to a single decomposition, so flipping them apart
# would show something the IVA never produced.
#
# :func:`anchor_signs_tf` derives the same kind of flip from the *driven
# response* instead — the sign of the stimulus-window power at the steady-state
# frequency. It is what anchors the reference pair's own global polarity, and
# beside the shared sign it is a **diagnostic**:
# :meth:`IvaQualityResult.anchor_agreement` reports how often a pair whose
# topography agrees with the reference also has a positive driven response. Where
# the two disagree, that pair's pattern and its source tell different stories —
# which is a property of the pair, not a sign left unresolved.


def anchor_signs_reference(patterns: np.ndarray, ref_topo: np.ndarray) -> np.ndarray:
    """Per-``(subject, component)`` flips agreeing with the reference topography.

    A pair is flipped when its channel pattern correlates *negatively* with
    *ref_topo*, so after this every participant's topography agrees with the
    reference — and therefore with every other participant, which is what makes a
    participant-comparison figure readable and stops a group mean from
    cancelling. The correlation is the same quantity
    :attr:`IvaQualityResult.topo_corr` reports, so applying these signs makes that
    score read non-negative and a panel's annotated ``r`` describes the map drawn
    beside it.

    The whole topography decides it rather than one electrode: the reference is a
    dipolar PC1 loading whose sign at any single channel is a weak criterion, and
    a run on a channel subset need not even contain that electrode. This is the
    **one** sign a pair gets — see the module docstring on why its TF map must
    carry the same one.

    :param patterns: ``(S, K, C)`` forward channel patterns. Any positive
        per-pair scaling is irrelevant: a correlation's sign survives it.
    :param ref_topo: ``(C,)`` reference topography.
    :return: ``(S, K)`` of ``+1`` / ``-1``; a pair with no defined correlation
        (a constant pattern) keeps ``+1``.
    :raises ValueError: If *patterns* is not 3-D or its channel axis does not
        match *ref_topo*.
    """
    data = np.asarray(patterns, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"patterns must be (S, K, C); got shape {data.shape}.")
    ref = np.asarray(ref_topo, dtype=float)
    if ref.shape != (data.shape[2],):
        raise ValueError(
            f"ref_topo shape {ref.shape} does not match the "
            f"{data.shape[2]}-channel axis."
        )

    signs = np.zeros(data.shape[:2])
    for s in range(data.shape[0]):
        for k in range(data.shape[1]):
            signs[s, k] = np.sign(np.nan_to_num(pearsonr(data[s, k], ref)[0]))
    signs[signs == 0] = 1.0
    return signs


def anchor_signs_tf(
    onset_tf: np.ndarray,
    freqs: np.ndarray,
    epoch_times: np.ndarray,
    *,
    assr_freq: float = ASSR_FREQ,
    resp_duration_s: float = RESP_DURATION_S,
    halfwidth_hz: float = TF_ANCHOR_HALFWIDTH_HZ,
) -> tuple[np.ndarray, bool]:
    """Per-``(subject, component)`` flips making the driven response positive.

    The anchor is the mean of the onset-averaged TF map over the band around
    *assr_freq* (± *halfwidth_hz*) and the stimulus window ``[0,
    resp_duration_s]`` — the 40 Hz response in the first 500 ms. A map that is
    negative there would be flipped, so every steady-state response reads
    positive.

    Two callers, one rule. It fixes the **reference pair's** own global polarity
    in :func:`compute_iva_quality`, where the arbitrary PCA sign has to be settled
    against the paradigm rather than against the group. Applied to the component
    maps it is a **diagnostic** instead: the pairs are aligned by
    :func:`anchor_signs_reference`, and comparing the two says how often a pair's
    topography and its driven response agree
    (:meth:`IvaQualityResult.anchor_agreement`). It is anchored on the paradigm,
    so neither use inherits the group mean's own polarity.

    :param onset_tf: ``(S, K, F, W)`` onset-averaged component TF maps.
    :param freqs: ``(F,)`` frequency axis in Hz.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param assr_freq: Steady-state frequency the anchor band is centred on.
    :param resp_duration_s: Length of the anchoring window after onset, in
        seconds.
    :param halfwidth_hz: Half-width of the anchor band around ``assr_freq``.
    :return: ``(signs, have_anchor)`` — ``(S, K)`` of ``+1`` / ``-1``, and whether
        the anchor cell existed. When it does not (no frequency bin in the band,
        or no sample in the window) the signs are all ``+1`` and the caller should
        fall back to something else.
    :raises ValueError: If ``onset_tf`` is not 4-D or its axes disagree with
        ``freqs`` / ``epoch_times``.
    """
    data = np.asarray(onset_tf, dtype=float)
    if data.ndim != 4:
        raise ValueError(f"onset_tf must be (S, K, F, W); got shape {data.shape}.")
    if data.shape[2] != len(freqs):
        raise ValueError(
            f"onset_tf has {data.shape[2]} frequency bins but {len(freqs)} "
            f"frequencies were given."
        )
    if data.shape[3] != len(epoch_times):
        raise ValueError(
            f"onset_tf has {data.shape[3]} samples but {len(epoch_times)} epoch "
            f"times were given."
        )

    band = frequency_band_mask(
        freqs, assr_freq - halfwidth_hz, assr_freq + halfwidth_hz
    )
    times = np.asarray(epoch_times, dtype=float)
    window = (times >= 0.0) & (times <= resp_duration_s)
    if not band.any() or not window.any():
        return np.ones(data.shape[:2]), False

    driven = data[:, :, band][:, :, :, window].mean(axis=(2, 3))
    signs = np.sign(driven)
    signs[signs == 0] = 1.0
    return signs, True


# ---------------------------------------------------------------------------
# Per-participant weighting (equal influence in across-subject means)
# ---------------------------------------------------------------------------


def subject_scales(
    arr: np.ndarray,
    comp_indices: Sequence[int] | None = None,
    *,
    percentile: float = SUBJECT_SCALE_PERCENTILE,
) -> np.ndarray:
    """Per-subject amplitude scale of a ``(S, K, ...)`` stack of maps.

    A robust stand-in for "how loud is this subject": the *percentile*-th
    percentile of ``|arr|`` over every axis except the subject one, taken across
    the selected components together. The percentile rather than the maximum, so
    one hot bin (a single channel, a single time-frequency cell) cannot set a
    subject's scale.

    :param arr: ``(S, K, ...)`` array whose first axis is the subject and second
        the component; any number of trailing axes is allowed.
    :param comp_indices: Components the scale is measured over. ``None`` uses all
        of them. Restrict it to the components actually plotted, so a subject's
        scale is not set by components no figure shows.
    :param percentile: Percentile of ``|arr|`` defining the scale.
    :return: ``(S,)`` strictly positive scales; a subject whose selected maps are
        all zero (or non-finite) gets ``1.0`` rather than a division by zero.
    :raises ValueError: If *arr* has fewer than 3 dimensions, has no subjects, or
        *comp_indices* is empty or out of range.
    """
    data = np.asarray(arr, dtype=float)
    if data.ndim < 3:
        raise ValueError(
            f"arr must be at least 3-D (n_subjects, n_components, ...); "
            f"got shape {data.shape} (ndim={data.ndim})."
        )
    if data.shape[0] < 1:
        raise ValueError("arr must contain at least one subject.")

    n_comp = data.shape[1]
    idx = (
        list(range(n_comp)) if comp_indices is None else [int(k) for k in comp_indices]
    )
    if not idx:
        raise ValueError("comp_indices must select at least one component.")
    if min(idx) < 0 or max(idx) >= n_comp:
        raise ValueError(
            f"comp_indices {sorted(set(idx))} out of range for {n_comp} components."
        )

    scales = np.percentile(
        np.abs(data[:, idx]), percentile, axis=tuple(range(1, data.ndim))
    )
    return np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)


def equalize_subject_influence(
    arr: np.ndarray,
    comp_indices: Sequence[int] | None = None,
    *,
    percentile: float = SUBJECT_SCALE_PERCENTILE,
) -> np.ndarray:
    """Rescale each subject's maps so every participant spans the same range.

    Divides subject *s*'s whole ``(K, ...)`` slab by its :func:`subject_scales`
    value, so after this every participant contributes the same amplitude range
    — the same weight — to any across-participant mean, and every participant
    panel of a figure is readable under one shared colour limit.

    Why it is needed: the per-subject gain is a nuisance, not signal. ``iva_g``
    fixes the *source* variance per ``(subject, component)`` but the onset
    *average* of a source, and the forward pattern that goes with it, still carry
    a subject-specific amplitude (channel-PCA eigenvalue spread on the pattern
    side, how much of the source is onset-locked on the map side). Averaging
    unequal-range maps across participants is dominated by the loudest few: the
    group mean stops being a group statement, and drawn under a colour limit set
    by those same loud participants it reads flat — "invisibly differentiated".

    The scale is per **subject**, deliberately *not* per
    ``(subject, component)``: a subject's loudness is one number, while the
    differences *between* that subject's components are exactly what the
    diagnostics are asked to show. Normalising each component separately would
    make a component with no stimulus-locked response look as strong as the best
    one.

    Correlation-based scores (:attr:`IvaQualityResult.topo_corr`,
    :attr:`~IvaQualityResult.tf_corr`, …) are scale-invariant, so they stay valid
    beside the rescaled maps and never need recomputing.

    :param arr: ``(S, K, ...)`` maps — patterns ``(S, K, C)``, onset averages
        ``(S, K, W)``, onset TF maps ``(S, K, F, W)``. **Not** mutated.
    :param comp_indices: Components the per-subject scale is measured over;
        ``None`` uses all of them. Every component is rescaled either way — this
        only chooses which ones define the scale.
    :param percentile: Percentile of ``|arr|`` defining each subject's scale.
    :return: A **new** array of the same shape, equally weighted across subjects.
    :raises ValueError: As :func:`subject_scales`.
    """
    data = np.asarray(arr, dtype=float)
    scales = subject_scales(data, comp_indices, percentile=percentile)
    return data / scales.reshape((-1,) + (1,) * (data.ndim - 1))


def full_tf_group_mean(
    sources: np.ndarray,
    signs: np.ndarray,
    *,
    percentile: float = SUBJECT_SCALE_PERCENTILE,
) -> np.ndarray:
    """Equal-weighted group mean of the **whole-recording** IVA sources.

    The quality analysis scores the onset epoch, but the recording either side of
    it is where a component's artefacts live, so this reduces the full
    ``(S, K, F, T)`` sources to one ``(K, F, T)`` map per component the same way
    every other group mean here is taken: each pair flipped by its own *signs*
    entry, each subject divided by its own amplitude scale, then averaged.

    It is what makes the figure differ from the standard IVA time-frequency map,
    which averages the raw sources: under IVA's arbitrary per-``(subject,
    component)`` polarity that mean cancels whatever the participants share, so a
    component can look empty there and structured here. Reading the two together
    is the point — see the module docstring on signs.

    Subject scales are measured one subject at a time and over **all**
    components, matching :func:`subject_scales` with ``comp_indices=None`` while
    never materialising ``|sources|`` for the whole cohort — the full sources are
    the largest array in the pipeline after the wavelet tensor itself.

    :param sources: ``(S, K, F, T)`` IVA sources over the whole recording, as the
        decomposition produced them. **Not** mutated.
    :param signs: ``(S, K)`` total per-pair flip to apply — the component
        orientation times the per-participant sign
        (:func:`anchor_signs_reference`).
    :param percentile: Percentile of ``|sources|`` defining each subject's scale.
    :return: ``(K, F, T)`` group-mean map.
    :raises ValueError: If *sources* is not 4-D, has no subjects, or *signs* does
        not match its ``(S, K)`` axes.
    """
    data = np.asarray(sources, dtype=float)
    if data.ndim != 4:
        raise ValueError(f"sources must be (S, K, F, T); got shape {data.shape}.")
    if data.shape[0] < 1:
        raise ValueError("sources must contain at least one subject.")
    sgn = np.asarray(signs, dtype=float)
    if sgn.shape != data.shape[:2]:
        raise ValueError(
            f"signs shape {sgn.shape} does not match the sources' "
            f"{data.shape[:2]} (subject, component) axes."
        )

    n_subjects = data.shape[0]
    out = np.zeros(data.shape[1:], dtype=float)
    for s in range(n_subjects):
        scale = float(np.percentile(np.abs(data[s]), percentile))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        out += (sgn[s] / scale)[:, np.newaxis, np.newaxis] * data[s]
    return out / n_subjects


# ---------------------------------------------------------------------------
# Scoring and orientation
# ---------------------------------------------------------------------------


def sign_agreement(corr: np.ndarray) -> np.ndarray:
    """Per-component fraction of subjects agreeing with the mean sign.

    A diagnostic for residual sign ambiguity: a component whose subjects split
    near 50/50 has not been consistently oriented, and its group mean is
    meaningless regardless of how the individual subjects score.

    :param corr: ``(S, K)`` per-subject correlations.
    :return: ``(K,)`` fractions in ``[0, 1]``.
    """
    frac = np.zeros(corr.shape[1])
    for k in range(corr.shape[1]):
        col = corr[:, k]
        ref = np.sign(col.mean()) or 1.0
        frac[k] = float(np.mean(np.sign(col) == ref))
    return frac


def quality_score(topo_corr: np.ndarray, time_corr: np.ndarray, k: int) -> float:
    """Mean over subjects of ``(topomap + time) / 2`` for component ``k``.

    Equals 1.0 exactly when every subject sits at the ideal ``(1, 1)`` corner,
    so it measures how close a component's points are to that corner.

    Both arrays must carry the *same* sign convention, and the figures pass the
    per-``(subject, component)`` aligned scores: it is a mean over subjects, so a
    component scored under mixed per-pair polarities cancels itself.

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` onset-locked time correlations (or any second
        axis — a TF-map correlation, whole or band-limited).
    :param k: Component index.
    :return: The component's score.
    """
    return float(np.mean((topo_corr[:, k] + time_corr[:, k]) / 2.0))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IvaQualityResult:
    """Everything :func:`compute_iva_quality` produces, ready for plotting.

    All correlation arrays and ``onset_avg_*`` / ``onset_tf`` / ``patterns`` are
    already oriented by :attr:`sign_per_comp`, so a component's maps, its onset
    response and its reported scores always share one sign convention.

    :attr:`ref_topo` and :attr:`ref_tf` are the two views of the single reference
    pair (:func:`wavelet_reference`): the group PC1 channel loading and its PC1 TF
    score map, sharing one polarity.

    :param ref_topo: ``(C,)`` group PC1 channel loading of the reference, on the
        IVA channel subset — the x-axis of every score here.
    :param anchor_channel: Channel that anchored the reference's global sign
        (``None`` when no channel names were usable).
    :param patterns: ``(S, K, C)`` oriented forward channel patterns, each
        rescaled to unit L2 norm per ``(subject, component)`` so participants
        stay comparable under a shared colour limit (see
        :func:`src.analysis.wavelet_ica.normalize_patterns_per_subject`).
    :param topo_corr: ``(S, K)`` oriented correlations of the component patterns
        against :attr:`ref_topo`.
    :param time_corr_pca: ``(S, K)`` oriented time correlations, PCA variant.
    :param time_corr_40: ``(S, K)`` oriented time correlations, 40 Hz variant —
        all zeros when :attr:`have_40hz` is ``False``.
    :param onset_avg_pca: ``(S, K, W)`` oriented onset averages, PCA variant.
    :param onset_avg_40: ``(S, K, W)`` oriented onset averages, 40 Hz variant.
    :param ref_tf: ``(F, W)`` group PC1 score map of the reference.
    :param wavelet_consistency: Cross-subject agreement of the per-subject PC1
        topographies the reference was averaged from — a group reference built
        from disagreeing subjects is not a meaningful target.
    :param tf_corr: ``(S, K)`` oriented correlations of :attr:`onset_tf` against
        :attr:`ref_tf` (whole map flattened).
    :param tf_corr_bands: The same score restricted to each :data:`TF_BANDS`
        interval, keyed by the band's tag and oriented like :attr:`tf_corr`. A
        band with no frequency bin inside the wavelet range is absent rather than
        zero-filled, so ``in`` is the availability check (as :attr:`have_40hz` is
        for the 40 Hz time variant). Use :meth:`wavelet_variants` to iterate the
        whole-map score and the bands together.
    :param onset_tf: ``(S, K, F, W)`` oriented onset-averaged component TF maps.
    :param full_tf_mean: ``(K, F, T)`` group-mean TF map over the **whole
        recording**, with :attr:`sign_per_subject` and :attr:`sign_per_comp`
        already applied and every participant equally weighted
        (:func:`full_tf_group_mean`). Unlike everything else here it is a group
        mean rather than a per-participant stack — the full sources are too large
        to carry — and it is a QC view rather than a scored one: it says what the
        components look like outside the onset epoch under the same signs the
        scored figures use.
    :param full_times: ``(T,)`` recording time axis in seconds, for
        :attr:`full_tf_mean`.
    :param freqs: ``(F,)`` wavelet frequency axis in Hz, for the TF figures.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param resp_duration_s: The rigid response window actually used, in seconds.
    :param sign_per_comp: ``(K,)`` of ``+1`` / ``-1`` orientation flips applied.
    :param sign_per_subject: ``(S, K)`` of ``+1`` / ``-1`` per-participant flips
        putting every pair's topography at the reference's polarity
        (:func:`anchor_signs_reference`). **Not** applied to the stored
        per-participant arrays — :meth:`topomap_view`, :meth:`tf_view` and the
        ``aligned=True`` scores of :meth:`variants` / :meth:`wavelet_variants`
        apply it, all the same one, so a pair's pattern, its TF map, its time
        course and the scores measured on them are never flipped apart. It *is*
        already applied to :attr:`full_tf_mean`.
    :param sign_tf_per_subject: ``(S, K)`` of ``+1`` / ``-1`` flips that would make
        every driven response positive (:func:`anchor_signs_tf`). A
        **diagnostic**, never applied: :meth:`anchor_agreement` compares it with
        :attr:`sign_per_subject`.
    :param subject_anchor_name: What resolved :attr:`sign_per_subject`, for the
        figures to name.
    :param tf_anchor_name: What the driven-response diagnostic measured: the
        frequency band and window, or the reference-map fallback.
    :param have_40hz: Whether :data:`ASSR_FREQ` fell inside the frequency range.
    :param n_epoch_pre: Baseline samples in the epoch (onset index).
    :param n_epoch_post: Post-onset samples in the epoch.
    """

    ref_topo: np.ndarray
    anchor_channel: str | None
    patterns: np.ndarray
    topo_corr: np.ndarray
    time_corr_pca: np.ndarray
    time_corr_40: np.ndarray
    onset_avg_pca: np.ndarray
    onset_avg_40: np.ndarray
    ref_tf: np.ndarray
    wavelet_consistency: SignConsistency
    tf_corr: np.ndarray
    tf_corr_bands: dict[str, np.ndarray]
    onset_tf: np.ndarray
    full_tf_mean: np.ndarray
    full_times: np.ndarray
    freqs: np.ndarray
    epoch_times: np.ndarray
    resp_duration_s: float
    sign_per_comp: np.ndarray
    sign_per_subject: np.ndarray
    sign_tf_per_subject: np.ndarray
    subject_anchor_name: str
    tf_anchor_name: str
    have_40hz: bool
    n_epoch_pre: int
    n_epoch_post: int

    @property
    def n_subjects(self) -> int:
        """Number of subjects ``S``."""
        return int(self.patterns.shape[0])

    @property
    def n_components(self) -> int:
        """Number of IVA components ``K``."""
        return int(self.patterns.shape[1])

    def variants(
        self, *, aligned: bool = False
    ) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
        """Boxcar time variants as ``(name, time_corr, onset_avg, tag)``.

        Each entry is one way of reducing the source to a time course before it is
        scored against the paradigm's boxcar, and brings a ``(S, K, W)``
        time-course diagnostic with it. The TF-map y-axes have maps rather than
        time courses behind them, so they live in :meth:`wavelet_variants`; both
        share the topomap correlation as their x-axis.

        The 40 Hz variant is omitted when it could not be computed.

        :param aligned: Apply :attr:`sign_per_subject`, the per-pair flip
            :meth:`topomap_view` and :meth:`tf_view` draw their maps under. A
            source flips with its pattern, so the time course — and the boxcar
            correlation measured on it — flips by that same sign. Pair it with the
            aligned x from :meth:`topomap_view` so a scatter and the maps behind it
            describe one and the same decomposition.
        :return: One entry per available boxcar variant.
        """
        signs = self.sign_per_subject

        def _orient(
            corr: np.ndarray, onset_avg: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if not aligned:
                return corr, onset_avg
            return corr * signs, onset_avg * signs[:, :, np.newaxis]

        corr_pca, avg_pca = _orient(self.time_corr_pca, self.onset_avg_pca)
        out = [("PCA frequency-reduction", corr_pca, avg_pca, "pca")]
        if self.have_40hz:
            corr_40, avg_40 = _orient(self.time_corr_40, self.onset_avg_40)
            out.append(("40 Hz band only", corr_40, avg_40, "40hz"))
        return out

    def topomap_view(self) -> tuple[np.ndarray, np.ndarray]:
        """Sign-aligned ``(patterns, topo_corr)`` for the topography figures.

        The stored arrays with :attr:`sign_per_subject` applied, so every
        participant's topography agrees with the reference and hence with every
        other participant. Use this for anything that *shows* or *averages* the
        patterns: without it a participant-comparison figure mixes polarities and
        a group mean cancels, because IVA leaves the sign of each
        ``(subject, component)`` pair free.

        The correlation comes along so a panel's annotated ``r`` describes the map
        drawn beside it — and since the sign *is* that correlation's sign, the
        returned ``r`` is non-negative throughout. It is also the **x of every
        scatter**: a scatter point and the maps it stands for have to describe the
        same decomposition, so the scores are measured on the flipped pair rather
        than under the one-flip-per-component convention. What a participant
        disagreeing with the reference costs is therefore read on the y (the pair
        is flipped to agree on x, and its source follows that flip) and in the
        per-component sign statistics, not on the x.

        :return: ``(patterns, topo_corr)``, both sign-aligned.
        """
        signs = self.sign_per_subject
        return (self.patterns * signs[:, :, np.newaxis], self.topo_corr * signs)

    def tf_view(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Sign-aligned ``(onset_tf, tf_corr, tf_corr_bands)`` for the TF maps.

        The TF counterpart of :meth:`topomap_view`, and deliberately the **same**
        :attr:`sign_per_subject`: a pair's TF map is the source belonging to the
        pattern that view draws, so the two must be flipped together or neither
        panel means what it appears to. As there, the returned correlations both
        annotate the maps and are the y of the TF scatters — see
        :meth:`wavelet_variants` with ``aligned=True`` for the same score together
        with its band-limited variants.

        Because the sign comes from the topography, a participant's driven
        response is *not* forced positive here; :meth:`anchor_agreement` reports
        how often it comes out that way anyway.

        :return: ``(onset_tf, tf_corr, tf_corr_bands)``, all sign-aligned.
        """
        signs = self.sign_per_subject
        return (
            self.onset_tf * signs[:, :, np.newaxis, np.newaxis],
            self.tf_corr * signs,
            {tag: corr * signs for tag, corr in self.tf_corr_bands.items()},
        )

    def anchor_agreement(self) -> float:
        """Fraction of pairs whose driven response is positive under the shared sign.

        :attr:`sign_per_subject` is fixed by the topography; the driven-response
        criterion (:func:`anchor_signs_tf`) is an independent physical statement
        about the same pair. Near 1 means a pair's pattern and its source tell the
        same story, so the two views can be read together pair by pair. Well below
        1 means they often do not: a component whose topographies agree may still
        carry steady-state responses of opposite sign across participants, and its
        group-mean TF map is then weaker than its group-mean topography for a real
        reason rather than through a sign left unresolved.

        :return: Agreement fraction in ``[0, 1]``.
        """
        return float(np.mean(self.sign_per_subject == self.sign_tf_per_subject))

    def wavelet_variants(
        self, *, aligned: bool = False
    ) -> list[tuple[str | None, np.ndarray, str]]:
        """TF-map y-axes as ``(band_name, tf_corr, tag)``.

        Every entry shares the topomap correlation as its x-axis and
        :attr:`onset_tf` as the maps behind it; they differ only in which
        frequency rows the correlation is measured over. The first entry is the
        whole-map score (``band_name`` ``None``, tag ``"wavelet_tf"``), followed by
        each :data:`TF_BANDS` band that fell inside the frequency range, in
        ``TF_BANDS`` order.

        :param aligned: Apply :attr:`sign_per_subject`, so every entry is measured
            on the maps :meth:`tf_view` draws — the whole-map score then matches
            that view's ``tf_corr`` exactly, and the bands are the same score
            restricted in frequency. Pair it with the aligned x from
            :meth:`topomap_view`.
        :return: One entry per available TF-map variant.
        """
        signs = self.sign_per_subject
        tf_corr = self.tf_corr * signs if aligned else self.tf_corr
        out: list[tuple[str | None, np.ndarray, str]] = [(None, tf_corr, "wavelet_tf")]
        for name, _fmin, _fmax, tag in TF_BANDS:
            if tag in self.tf_corr_bands:
                band = self.tf_corr_bands[tag]
                out.append((name, band * signs if aligned else band, tag))
        return out


# ---------------------------------------------------------------------------
# Orchestration (still pure)
# ---------------------------------------------------------------------------


def compute_iva_quality(
    iva_components: np.ndarray,
    iva_sources: np.ndarray,
    freqs: np.ndarray,
    onsets: np.ndarray,
    sfreq: float,
    iva_ch_names: list[str],
    wavelet_power_z: np.ndarray,
) -> IvaQualityResult:
    """Score every ``(subject, component)`` pair against the reference pair.

    :param iva_components: ``(S, K, C)`` forward channel patterns (**not**
        unmixing rows — see
        :func:`src.analysis.wavelet_ica.iva_component_patterns`). Any
        per-subject scaling is irrelevant here: the scores are correlations, and
        :attr:`IvaQualityResult.patterns` is unit-normed per subject.
    :param iva_sources: ``(S, K, F, T)`` spectro-temporal IVA sources.
    :param freqs: ``(F,)`` wavelet frequency axis in Hz.
    :param onsets: Stimulus onset sample indices.
    :param sfreq: Sampling frequency in Hz.
    :param iva_ch_names: Channel names of the IVA subset, in IVA order.
    :param wavelet_power_z: ``(S, C, F, T)`` time-z-scored wavelet power the
        IVA was run on — the source of the reference pair
        (:func:`wavelet_reference`). Its channel, frequency and time axes must
        match the IVA's.
    :return: A fully populated :class:`IvaQualityResult`.
    :raises ValueError: If the component/source shapes disagree, the channel
        axis does not match ``iva_ch_names``, or ``wavelet_power_z`` does not
        match the IVA's ``(C, F, T)`` extent.
    """
    if iva_components.ndim != 3:
        raise ValueError(
            f"iva_components must be (S, K, C); got shape {iva_components.shape}."
        )
    if iva_sources.ndim != 4:
        raise ValueError(
            f"iva_sources must be (S, K, F, T); got shape {iva_sources.shape}."
        )
    n_subjects, n_comp, n_channels = iva_components.shape
    if iva_sources.shape[:2] != (n_subjects, n_comp):
        raise ValueError(
            f"iva_sources leading axes {iva_sources.shape[:2]} do not match "
            f"iva_components {(n_subjects, n_comp)}."
        )
    if n_channels != len(iva_ch_names):
        raise ValueError(
            f"iva_components has {n_channels} channels but {len(iva_ch_names)} "
            f"channel names were given."
        )
    if iva_sources.shape[2] != len(freqs):
        raise ValueError(
            f"iva_sources has {iva_sources.shape[2]} frequency bins but "
            f"{len(freqs)} frequencies were given."
        )
    n_times = int(iva_sources.shape[3])
    if wavelet_power_z.ndim != 4:
        raise ValueError(
            f"wavelet_power_z must be (S, C, F, T); got shape {wavelet_power_z.shape}."
        )
    # The wavelet reference and the component TF maps are compared bin-for-bin,
    # so the tensor the reference is built from must be the IVA's own input.
    if wavelet_power_z.shape != (n_subjects, n_channels, len(freqs), n_times):
        raise ValueError(
            f"wavelet_power_z shape {wavelet_power_z.shape} does not match the IVA "
            f"extent {(n_subjects, n_channels, len(freqs), n_times)}."
        )

    pre, post = onset_window(onsets, n_times, sfreq)
    win = pre + post
    epoch_times = np.arange(-pre, post) / sfreq
    resp_samples = response_duration_samples(post, sfreq)

    # The reference pair, cut on the same epoch as everything below and built from
    # the IVA's own input tensor, so the topography and the TF map are compared
    # with the component maps bin for bin and in the same modality.
    ref_topo, ref_tf, anchor, wav_consistency = wavelet_reference(
        wavelet_power_z, onsets, pre, post, iva_ch_names
    )
    # The pair's *global* sign is a PCA sign — arbitrary — and it propagates into
    # every score through sign_per_comp below. The boxcar y-axis is **not**
    # arbitrary (it is positive during the stimulus), so leaving the reference's
    # polarity to the channel rule would let a component that follows the stimulus
    # score -1 on it. Anchor the pair on the physics instead: the driven response
    # in the reference's own TF map reads positive, by the same rule that aligns
    # the participants. Both halves flip together, so they stay consistent.
    ref_sign, have_ref_anchor = anchor_signs_tf(
        ref_tf[np.newaxis, np.newaxis],
        freqs,
        epoch_times,
        resp_duration_s=resp_samples / sfreq,
    )
    if have_ref_anchor and ref_sign[0, 0] < 0:
        ref_topo, ref_tf = -ref_topo, -ref_tf

    # x-axis: pattern-vs-pattern correlation with the reference topography.
    topo_corr = np.zeros((n_subjects, n_comp))
    for s in range(n_subjects):
        for k in range(n_comp):
            topo_corr[s, k] = pearsonr(iva_components[s, k], ref_topo)[0]
    topo_corr = np.nan_to_num(topo_corr)

    # TF y-axis: the component's own onset-averaged TF map, scored against the
    # reference map as a whole rather than after collapsing frequency.
    onset_tf, _ = epoch_average(iva_sources, onsets, pre, post)  # (S, K, F, W)
    tf_corr = tf_map_correlation(onset_tf, ref_tf)
    # The same score over one frequency band at a time. A band the frequency axis
    # does not reach is simply absent (an alpha-band run has nothing in 30-50 Hz),
    # and a partly covered band is scored over the bins it does have.
    tf_corr_bands: dict[str, np.ndarray] = {}
    for _name, fmin, fmax, tag in TF_BANDS:
        mask = frequency_band_mask(freqs, fmin, fmax)
        if mask.any():
            tf_corr_bands[tag] = tf_map_correlation(onset_tf, ref_tf, freq_mask=mask)

    # y-axis: onset-average of each component's reduced time course. Variant A
    # takes the first PC over frequency; variant B the single ASSR_FREQ row.
    have_40 = bool(freqs.min() <= ASSR_FREQ <= freqs.max())
    f40 = int(np.argmin(np.abs(freqs - ASSR_FREQ)))
    onset_avg_pca = np.zeros((n_subjects, n_comp, win))
    onset_avg_40 = np.zeros((n_subjects, n_comp, win))
    for s in range(n_subjects):
        for k in range(n_comp):
            src = iva_sources[s, k]  # (F, T)
            pc1 = PCA(n_components=1).fit_transform(src.T)[:, 0]
            # Anchor the frequency-PCA sign to the signed temporal marginal so it
            # is consistent across subjects (IVA never fixed this extra flip).
            if pearsonr(pc1, src.mean(axis=0))[0] < 0:
                pc1 = -pc1
            onset_avg_pca[s, k] = onset_average(pc1, onsets, pre, post, n_times)
            if have_40:
                onset_avg_40[s, k] = onset_average(src[f40], onsets, pre, post, n_times)

    time_corr_pca = boxcar_correlation(onset_avg_pca, pre, win, resp_samples)
    time_corr_40 = (
        boxcar_correlation(onset_avg_40, pre, win, resp_samples)
        if have_40
        else np.zeros((n_subjects, n_comp))
    )

    # One flip per component so its mean topomap correlation reads non-negative;
    # every other score and every map follow that same shared component sign, a
    # component having a single orientation. A y-score that still reads negative
    # is then a genuine disagreement between the spatial and temporal axes.
    sign_per_comp = np.sign(topo_corr.mean(axis=0))
    sign_per_comp[sign_per_comp == 0] = 1.0

    # Unit-norm per subject: the participant-comparison figures share one colour
    # limit across subjects, so a per-subject gain (which IVA leaves on the
    # patterns) would saturate the loud subjects and flatten the quiet ones.
    # topo_corr is scale-invariant and so unaffected.
    patterns_oriented = normalize_patterns_per_subject(
        iva_components * sign_per_comp[np.newaxis, :, np.newaxis]
    )
    onset_tf_oriented = onset_tf * sign_per_comp[np.newaxis, :, np.newaxis, np.newaxis]

    # One flip per component leaves the participants free to disagree with each
    # other, IVA having fixed no per-(subject, component) sign. This resolves that
    # for the figures that compare or average participants
    # (IvaQualityResult.topomap_view / .tf_view) — one sign per pair, from its
    # topography, worn by that pair's TF map too.
    sign_per_subject = anchor_signs_reference(patterns_oriented, ref_topo)

    # The same flip on the whole recording, so the QC map outside the onset epoch
    # is the one the scored figures imply. Built from the raw sources, hence the
    # component orientation has to be folded in as well.
    full_tf_mean = full_tf_group_mean(
        iva_sources, sign_per_subject * sign_per_comp[np.newaxis, :]
    )

    # Diagnostic only: whether a pair aligned on its topography also has a
    # positive driven response. Where it does not, the pattern and the source
    # disagree — a property of that pair, reported by anchor_agreement().
    sign_tf_subj, have_tf_anchor = anchor_signs_tf(
        onset_tf_oriented, freqs, epoch_times, resp_duration_s=resp_samples / sfreq
    )
    if have_tf_anchor:
        tf_anchor_name = (
            f"{ASSR_FREQ - TF_ANCHOR_HALFWIDTH_HZ:.0f}-"
            f"{ASSR_FREQ + TF_ANCHOR_HALFWIDTH_HZ:.0f} Hz, "
            f"0-{resp_samples / sfreq * 1000:.0f} ms"
        )
    else:
        # No bin in the driven band (a band-sliced run): fall back to agreeing with
        # the reference map, which says nothing about the steady state but still
        # gives the shared sign something independent to be compared against.
        sign_tf_subj = np.sign(tf_map_correlation(onset_tf_oriented, ref_tf))
        sign_tf_subj[sign_tf_subj == 0] = 1.0
        tf_anchor_name = "reference TF-map correlation"

    return IvaQualityResult(
        ref_topo=ref_topo,
        anchor_channel=anchor,
        patterns=patterns_oriented,
        topo_corr=topo_corr * sign_per_comp,
        time_corr_pca=time_corr_pca * sign_per_comp,
        time_corr_40=time_corr_40 * sign_per_comp,
        onset_avg_pca=onset_avg_pca * sign_per_comp[np.newaxis, :, np.newaxis],
        onset_avg_40=onset_avg_40 * sign_per_comp[np.newaxis, :, np.newaxis],
        ref_tf=ref_tf,
        wavelet_consistency=wav_consistency,
        tf_corr=tf_corr * sign_per_comp,
        tf_corr_bands={
            tag: corr * sign_per_comp for tag, corr in tf_corr_bands.items()
        },
        onset_tf=onset_tf_oriented,
        full_tf_mean=full_tf_mean,
        full_times=np.arange(n_times) / sfreq,
        freqs=np.asarray(freqs, dtype=float),
        epoch_times=epoch_times,
        resp_duration_s=resp_samples / sfreq,
        sign_per_comp=sign_per_comp,
        sign_per_subject=sign_per_subject,
        sign_tf_per_subject=sign_tf_subj,
        subject_anchor_name=SUBJECT_SIGN_ANCHOR,
        tf_anchor_name=tf_anchor_name,
        have_40hz=have_40,
        n_epoch_pre=pre,
        n_epoch_post=post,
    )
