"""
Decomposition-quality scoring for the channel-as-independent IVA workflow.

Scores every ``(subject, component)`` pair of a channel-IVA decomposition
against two independent references, so a component can be judged on whether it
actually captures the stimulus response rather than on cross-subject
consistency alone:

- **x — topomap correlation.** Pearson correlation between the component's
  forward channel pattern and a group reference topography (the first
  channel-PCA loading of the raw stimulus-evoked response). Both sides are
  *patterns*, never unmixing filters; see
  :func:`src.analysis.wavelet_ica.iva_component_patterns`.
- **y — onset-locked time correlation.** The component's ``(F, T)`` source is
  reduced to a single time course, averaged around every stimulus onset, and
  correlated (zero-lag Pearson) with a **rigid boxcar**: 0 before onset, 1 for
  :data:`RESP_DURATION_S` after it, 0 afterwards.

Onset-locking is what makes the time axis work. ASSR stimulation is
*continuous*, so a plain on/off indicator over the whole recording fails — a
sustained response is flat once z-scored. Averaging around onsets and scoring
against a window fixed at the stimulus length means every component is measured
against one and the same reference instead of a per-component fitted one.

The analysis needs stimulus onsets, so it applies to the ASSR experiment only.

All functions here are **pure**: no I/O, no plotting, no mutation of inputs.
Rendering lives in :mod:`src.visualization.iva_quality_plots`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA

from src.analysis.wavelet_ica import normalize_patterns_per_subject
from src.definitions.constants import AssrEpoch

# ---------------------------------------------------------------------------
# Reference-topomap settings (raw evoked channel-PCA)
# ---------------------------------------------------------------------------

# Epoch geometry comes from the paradigm definition so every onset-locked ASSR
# analysis cuts the same window; these names are kept as the module's public API.
REF_PRE_PAD_S = AssrEpoch.PRE_ONSET_S  # pre-onset pad (s), reference evoked epoch
REF_POLARITY_CHANNEL = "Cz"  # anchors the reference topomap's global sign

# ---------------------------------------------------------------------------
# Onset-locked time-reference settings
# ---------------------------------------------------------------------------

ASSR_FREQ = 40.0  # frequency (Hz) used by the "40 Hz-only" time variant
EPOCH_PRE_S = AssrEpoch.PRE_ONSET_S  # pre-onset baseline in the averaged epoch
EPOCH_POST_S = AssrEpoch.POST_ONSET_S  # post-onset span (capped at shortest gap)
RESP_DURATION_S = AssrEpoch.STIMULUS_DURATION_S  # rigid expected-response window


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
# Reference topography
# ---------------------------------------------------------------------------


def reference_topomap(
    raw_voltage: np.ndarray,
    onsets: np.ndarray,
    ref_ch_names: list[str],
    iva_ch_names: list[str],
    sfreq: float,
    *,
    pre_pad_s: float = REF_PRE_PAD_S,
    polarity_channel: str = REF_POLARITY_CHANNEL,
) -> tuple[np.ndarray, str]:
    """Group-averaged first channel-PCA loading of the raw evoked response.

    Per subject: z-score each channel, trial-average around every onset, take the
    first channel-PCA loading (the evoked topography). For a single PC of an
    orthogonal decomposition the filter and the pattern coincide, so this is a
    genuine topography and is directly comparable with IVA forward patterns.
    Loadings are polarity-aligned across subjects and anchored to
    ``polarity_channel``, then averaged and restricted to ``iva_ch_names``
    **by name** (never by position).

    :param raw_voltage: ``(S, C_full, T_full)`` preprocessed voltage.
    :param onsets: Stimulus onset sample indices.
    :param ref_ch_names: Channel names matching ``raw_voltage``'s channel axis.
    :param iva_ch_names: Channel names of the IVA subset, in IVA order.
    :param sfreq: Sampling frequency in Hz.
    :param pre_pad_s: Pre-onset pad kept in the evoked epoch, in seconds.
    :param polarity_channel: Channel anchoring the global sign; falls back to the
        largest-magnitude channel when absent.
    :return: ``(ref_topo, anchor_channel)`` with ``ref_topo`` of shape
        ``(len(iva_ch_names),)``.
    :raises ValueError: If ``raw_voltage``'s channel axis and ``ref_ch_names``
        disagree, or a requested IVA channel is missing from the reference.
    """
    if raw_voltage.shape[1] != len(ref_ch_names):
        raise ValueError(
            f"raw voltage channels {raw_voltage.shape[1]} != names {len(ref_ch_names)}"
        )
    missing = set(iva_ch_names) - set(ref_ch_names)
    if missing:
        raise ValueError(f"IVA channels missing from the reference: {sorted(missing)}")

    gaps = np.diff(onsets)
    pre = int(round(pre_pad_s * sfreq))
    # Paradigm window (capped by the shortest gap), not the gap itself — so this
    # reference epoch matches every other onset-locked ASSR analysis.
    post = AssrEpoch.post_onset_samples(sfreq, min_gap=int(gaps.min()))
    loadings = np.zeros((raw_voltage.shape[0], raw_voltage.shape[1]))
    for si in range(raw_voltage.shape[0]):
        sig = zscore(raw_voltage[si], axis=1)
        evoked, _ = epoch_average(sig, onsets, pre, post)
        pca = PCA(n_components=1)
        pca.fit(evoked.T)
        loadings[si] = pca.components_[0]

    # Iteratively align each subject's PC1 polarity to the running group mean.
    template = loadings[0].copy()
    for _ in range(10):
        flips = np.sign(loadings @ template)
        flips[flips == 0] = 1.0
        loadings = loadings * flips[:, None]
        template = loadings.mean(axis=0)

    anchor = (
        polarity_channel
        if polarity_channel in ref_ch_names
        else ref_ch_names[int(np.argmax(np.abs(template)))]
    )
    if template[ref_ch_names.index(anchor)] < 0:
        loadings, template = -loadings, -template
    by_name = dict(zip(ref_ch_names, loadings.mean(axis=0)))
    return np.array([by_name[ch] for ch in iva_ch_names]), anchor


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

    :param topo_corr: ``(S, K)`` topomap correlations.
    :param time_corr: ``(S, K)`` onset-locked time correlations.
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

    All correlation arrays and ``onset_avg_*`` / ``patterns`` are already
    oriented by :attr:`sign_per_comp`, so a component's maps, its onset response
    and its reported scores always share one sign convention.

    :param ref_topo: ``(C,)`` reference topography on the IVA channel subset.
    :param anchor_channel: Channel that anchored the reference's global sign.
    :param patterns: ``(S, K, C)`` oriented forward channel patterns, each
        rescaled to unit L2 norm per ``(subject, component)`` so participants
        stay comparable under a shared colour limit (see
        :func:`src.analysis.wavelet_ica.normalize_patterns_per_subject`).
    :param topo_corr: ``(S, K)`` oriented topomap correlations.
    :param time_corr_pca: ``(S, K)`` oriented time correlations, PCA variant.
    :param time_corr_40: ``(S, K)`` oriented time correlations, 40 Hz variant —
        all zeros when :attr:`have_40hz` is ``False``.
    :param onset_avg_pca: ``(S, K, W)`` oriented onset averages, PCA variant.
    :param onset_avg_40: ``(S, K, W)`` oriented onset averages, 40 Hz variant.
    :param epoch_times: ``(W,)`` epoch time axis in seconds, 0 at onset.
    :param resp_duration_s: The rigid response window actually used, in seconds.
    :param sign_per_comp: ``(K,)`` of ``+1`` / ``-1`` orientation flips applied.
    :param have_40hz: Whether :data:`ASSR_FREQ` fell inside the frequency range.
    :param n_epoch_pre: Baseline samples in the epoch (onset index).
    :param n_epoch_post: Post-onset samples in the epoch.
    """

    ref_topo: np.ndarray
    anchor_channel: str
    patterns: np.ndarray
    topo_corr: np.ndarray
    time_corr_pca: np.ndarray
    time_corr_40: np.ndarray
    onset_avg_pca: np.ndarray
    onset_avg_40: np.ndarray
    epoch_times: np.ndarray
    resp_duration_s: float
    sign_per_comp: np.ndarray
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

    def variants(self) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
        """Time-reduction variants as ``(name, time_corr, onset_avg, tag)``.

        The 40 Hz variant is omitted when it could not be computed.
        """
        out = [
            (
                "PCA frequency-reduction",
                self.time_corr_pca,
                self.onset_avg_pca,
                "pca",
            )
        ]
        if self.have_40hz:
            out.append(
                ("40 Hz band only", self.time_corr_40, self.onset_avg_40, "40hz")
            )
        return out


# ---------------------------------------------------------------------------
# Orchestration (still pure)
# ---------------------------------------------------------------------------


def compute_iva_quality(
    iva_components: np.ndarray,
    iva_sources: np.ndarray,
    freqs: np.ndarray,
    raw_voltage: np.ndarray,
    onsets: np.ndarray,
    sfreq: float,
    ref_ch_names: list[str],
    iva_ch_names: list[str],
) -> IvaQualityResult:
    """Score every ``(subject, component)`` pair against both references.

    :param iva_components: ``(S, K, C)`` forward channel patterns (**not**
        unmixing rows — see
        :func:`src.analysis.wavelet_ica.iva_component_patterns`). Any
        per-subject scaling is irrelevant here: the scores are correlations, and
        :attr:`IvaQualityResult.patterns` is unit-normed per subject.
    :param iva_sources: ``(S, K, F, T)`` spectro-temporal IVA sources.
    :param freqs: ``(F,)`` wavelet frequency axis in Hz.
    :param raw_voltage: ``(S, C_full, T_full)`` preprocessed voltage for the
        reference topography.
    :param onsets: Stimulus onset sample indices.
    :param sfreq: Sampling frequency in Hz.
    :param ref_ch_names: Channel names matching ``raw_voltage``.
    :param iva_ch_names: Channel names of the IVA subset, in IVA order.
    :return: A fully populated :class:`IvaQualityResult`.
    :raises ValueError: If the component/source shapes disagree, or the channel
        axis does not match ``iva_ch_names``.
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

    ref_topo, anchor = reference_topomap(
        raw_voltage, onsets, ref_ch_names, iva_ch_names, sfreq
    )

    pre, post = onset_window(onsets, n_times, sfreq)
    win = pre + post
    epoch_times = np.arange(-pre, post) / sfreq
    resp_samples = response_duration_samples(post, sfreq)

    # x-axis: pattern-vs-pattern correlation with the reference topography.
    topo_corr = np.zeros((n_subjects, n_comp))
    for s in range(n_subjects):
        for k in range(n_comp):
            topo_corr[s, k] = pearsonr(iva_components[s, k], ref_topo)[0]

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
    # the time scores and the patterns follow the same shared component sign.
    sign_per_comp = np.sign(topo_corr.mean(axis=0))
    sign_per_comp[sign_per_comp == 0] = 1.0

    return IvaQualityResult(
        ref_topo=ref_topo,
        anchor_channel=anchor,
        # Unit-norm per subject: the participant-comparison figures share one
        # colour limit across subjects, so a per-subject gain (which IVA leaves
        # on the patterns) would saturate the loud subjects and flatten the
        # quiet ones. topo_corr is scale-invariant and so unaffected.
        patterns=normalize_patterns_per_subject(
            iva_components * sign_per_comp[np.newaxis, :, np.newaxis]
        ),
        topo_corr=topo_corr * sign_per_comp,
        time_corr_pca=time_corr_pca * sign_per_comp,
        time_corr_40=time_corr_40 * sign_per_comp,
        onset_avg_pca=onset_avg_pca * sign_per_comp[np.newaxis, :, np.newaxis],
        onset_avg_40=onset_avg_40 * sign_per_comp[np.newaxis, :, np.newaxis],
        epoch_times=epoch_times,
        resp_duration_s=resp_samples / sfreq,
        sign_per_comp=sign_per_comp,
        have_40hz=have_40,
        n_epoch_pre=pre,
        n_epoch_post=post,
    )
