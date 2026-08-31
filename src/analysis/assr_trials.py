"""
Per-trial 40 Hz time courses of the ASSR, read through a set of spatial filters.

The step between a stored IVA decomposition and a trial-level test. A stored run gives
*spatial filters* — one ``(components, channels)`` operator per participant — and the
paradigm gives *onsets*. This module turns the pair into the array a trial-level
analysis wants::

    (participants, sources, trials, epoch samples)

for one frequency selection, per condition, plus the paired tests over it.

**Every source is a spatial filter**, and that is the point of the layout. The IVA
components and the checked-in binary ASSR-electrode selection
(:func:`~src.io.loading.assr_electrode_mask`) are the same *kind* of operator — a
weighting over channels that contracts the channel axis and leaves frequency and time
untouched — so they are stacked into one ``(sources, channels)`` matrix and projected in
one contraction. They differ only in provenance: the IVA rows are learned per
participant, the mask row is fixed by the montage and the paradigm and is identical for
everyone. Keeping them on one axis is what makes "the same trial under each filter" a
slice rather than a join.

Four things this module deliberately does **not** do, each because doing them would
change what a between-condition difference means:

* **No standardisation of the input.** Trials come out in the wavelet cache's own power
  units. The *filters* were estimated on z-scored data — that is baked into the stored
  decomposition and cannot be undone — but the signal they are applied to is left alone,
  so an overall power difference between conditions survives into the result.
* **No baseline subtraction at extraction.** The epoch keeps its pre-onset samples
  (:attr:`~src.definitions.constants.AssrEpoch.PRE_ONSET_S`), so the reference stays a
  choice :func:`baseline_normalise` makes explicitly rather than one already applied.
* **No averaging over trials.** :func:`~src.analysis.iva_quality.epoch_average` is the
  averaging counterpart of :func:`cut_trials`; this keeps every trial, which is the
  whole reason to run the extraction.
* **No ratio anywhere.** A projected IVA source is a *signed* combination of channel
  powers, so it runs negative on a large fraction of trials and any ratio against a
  baseline flips sign or explodes. Every normalisation here is subtractive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.definitions.constants import AssrEpoch
from src.definitions.fields import ConditionVariants

#: Label of the source row produced by the binary ASSR-electrode filter, as opposed to
#: the learned ``IC <k>`` rows. Kept in the label rather than in a separate array so a
#: trial-level analysis can group by source without knowing which rows were learned.
BINARY_FILTER_LABEL = "ASSR-mask"

#: Format of a learned component's source label, 1-based to match the ``IC <k>`` naming
#: every figure in the project uses.
COMPONENT_LABEL = "IC {k}"

#: Prefixes of the per-condition arrays inside a stored trial file. Per-condition rather
#: than one stacked array because the conditions may keep different trial counts, and
#: padding them to a common length would invent data.
TRIALS_KEY = "trials"
TRIALS_Z_KEY = "trials_z"
TRIALS_REL_KEY = "trials_rel"
BASELINE_POSITIVE_KEY = "baseline_positive"
ONSETS_KEY = "onsets"

#: How large a component's pattern weight over the anchor electrodes must be, relative
#: to its weight over all channels, before its sign is trustworthy. Below this the
#: component barely projects onto the area and the flip is a coin toss on noise.
#:
#: Note this is deliberately NOT a measure of how much participants agree on the sign.
#: They are free to disagree — that is precisely what the flip exists to reconcile, and
#: on a real cohort every component splits. What has to hold is that each participant's
#: own weight there is big enough for its sign to mean something.
POLARITY_STRENGTH_FLOOR = 0.2


def source_labels(n_components: int) -> list[str]:
    """``["IC 1", ..., "IC K", "ASSR-mask"]`` for a run with *n_components* components.

    :param n_components: Number of learned components.
    :return: Source labels in stacking order, the fixed binary filter last.
    """
    return [COMPONENT_LABEL.format(k=k + 1) for k in range(n_components)] + [
        BINARY_FILTER_LABEL
    ]


# ---------------------------------------------------------------------------
# Spatial filters
# ---------------------------------------------------------------------------


def recover_spatial_filters(
    channel_patterns: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Invert stored forward patterns back into the backward spatial filters.

    The IVA store keeps the **forward patterns** (the topographies), not the filters,
    and the two are not interchangeable. For participant *p* the filter ``U_p``
    (``components x channels``) *extracts* a source from the channels, which is what
    projecting a new signal needs; the stored pattern ``A_p = pinv(U_p)``
    (``channels x components``) says how the source projects *onto* the channels, which
    is what belongs on a topomap. After a whitened IVA the two are nearly unrelated,
    because the filter carries a ``Sigma^-1`` reweighting that up-weights the
    lowest-variance retained PCA directions (Haufe et al., 2014). Projecting with a
    pattern is the classic filter-vs-pattern error.

    The inversion is exact rather than approximate: ``U_p`` has full row rank, so
    ``pinv(pinv(U_p)) = U_p``. That is checked here rather than assumed. Any
    per-participant sign flips the run applied are diagonal +-1 scalings on the
    component index, so they propagate through the inverse and the recovered filters
    stay consistent with the stored sources.

    :param channel_patterns: ``(participants, components, channels)`` as the store holds
        them, i.e. ``A_p`` transposed on the last two axes.
    :param tolerance: Largest tolerated ``|U_p A_p - I|``. The default suits the store's
        ``float32`` patterns, whose inversion inherits ``float32`` precision (expect
        ~1e-7, not ~1e-15).
    :return: ``(participants, components, channels)`` spatial filters.
    :raises ValueError: If *channel_patterns* is not 3-D, or the round trip fails for
        any participant — which means the stored patterns are rank-deficient and the
        filter is not recoverable from them.
    """
    patterns = np.asarray(channel_patterns)
    if patterns.ndim != 3:
        raise ValueError(
            f"channel_patterns must be (participants, components, channels); got "
            f"{patterns.shape}."
        )
    n_participants, n_components, _ = patterns.shape

    filters = np.stack([np.linalg.pinv(patterns[p].T) for p in range(n_participants)])
    residuals = np.array(
        [
            np.abs(filters[p] @ patterns[p].T - np.eye(n_components)).max()
            for p in range(n_participants)
        ]
    )
    if residuals.max() > tolerance:
        raise ValueError(
            f"The recovered filters do not invert the stored patterns (max "
            f"|U A - I| = {residuals.max():.2e} at participant index "
            f"{int(residuals.argmax())}, tolerance {tolerance:.1e}). The stored "
            "patterns are probably rank-deficient, so the filter that produced the "
            "stored sources cannot be recovered from them."
        )
    return filters


def binary_filter_weights(mask: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Turn a 0/1 electrode mask into the ``(channels,)`` weighting used as a filter.

    :param mask: Boolean mask over the channel axis.
    :param normalize: Average over the selected electrodes rather than summing them.
        The mean is the default because it keeps the output in the input's power units
        and does not scale with how many electrodes the list happens to contain.
    :return: ``(channels,)`` weights.
    :raises ValueError: If *mask* selects nothing.
    """
    weights = np.asarray(mask, dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("The electrode mask selects no channel.")
    return weights / total if normalize else weights


def stack_filters(
    spatial_filters: np.ndarray,
    binary_filter: np.ndarray,
) -> np.ndarray:
    """Append the fixed binary filter as one more row of every participant's filter.

    Turns the learned ``(participants, components, channels)`` operator and the single
    fixed ``(channels,)`` weighting into one ``(participants, components + 1, channels)``
    matrix, so both are applied by the same contraction and land on the same source
    axis. The binary row is identical for every participant, which is exactly what makes
    it the reference the learned rows are read against.

    :param spatial_filters: ``(participants, components, channels)`` learned filters.
    :param binary_filter: ``(channels,)`` fixed channel weighting.
    :return: ``(participants, components + 1, channels)``, the binary row last.
    :raises ValueError: If the channel axes disagree.
    """
    filters = np.asarray(spatial_filters, dtype=float)
    binary = np.asarray(binary_filter, dtype=float)
    if binary.ndim != 1 or binary.size != filters.shape[-1]:
        raise ValueError(
            f"binary_filter must be ({filters.shape[-1]},) to match the learned "
            f"filters' channel axis; got {binary.shape}."
        )
    rows = np.broadcast_to(binary, (filters.shape[0], 1, binary.size))
    return np.concatenate([filters, rows], axis=1)


def project_channels(filters: np.ndarray, track: np.ndarray) -> np.ndarray:
    """Apply a ``(sources, channels)`` filter to a ``(channels, ...)`` track.

    The contraction every spatial filter in this project performs: only the channel axis
    is consumed, so frequency and time pass through untouched and the track keeps its
    own length.

    :param filters: ``(sources, channels)`` weighting.
    :param track: ``(channels, ...)`` signal.
    :return: ``(sources, ...)``.
    :raises ValueError: If the channel axes disagree.
    """
    filters = np.asarray(filters, dtype=float)
    track = np.asarray(track)
    if filters.shape[-1] != track.shape[0]:
        raise ValueError(
            f"Filter spans {filters.shape[-1]} channel(s) but the track has "
            f"{track.shape[0]}. A spatial filter cannot be restricted to a subset of "
            "its channels by dropping columns; that is a different operator."
        )
    return np.tensordot(filters, track, axes=([1], [0]))


# ---------------------------------------------------------------------------
# Polarity anchoring
# ---------------------------------------------------------------------------


def polarity_flip(
    channel_patterns: np.ndarray,
    electrode_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-participant sign that orients each component to the ASSR electrodes.

    A component's sign is fixed only up to whatever the run's own alignment chose,
    which is not the same as "positive means more power over the electrodes of
    interest". Measured on a real cohort, the sign of a component's forward-pattern
    weight over those electrodes splits across participants — so left alone, a positive
    value means *more* power there for some participants and *less* for others, no
    direction can be predicted, and a group mean partly cancels.

    Flipping by that sign fixes it. The anchor is legitimate for a paired contrast
    precisely because the pattern is a property of the decomposition, shared by both
    conditions when one mixing matrix per participant covers the whole recording — so
    it is symmetric in the conditions and cannot favour either. An anchor read off the
    tested quantity instead (say, "flip so condition A is positive") forces that
    condition's value to be ``|v| >= 0`` while leaving the other centred, and
    manufactures a difference under the null.

    :param channel_patterns: ``(participants, components, channels)`` forward patterns.
    :param electrode_mask: Boolean mask over the channel axis selecting the electrodes
        the components are anchored to.
    :return: ``(flip, strength)`` — ``flip`` is ``(participants, components)`` of +-1,
        and ``strength`` is ``(participants, components)`` holding
        ``|mean weight on the mask| / |mean weight overall|``, which says whether the
        anchor is well determined or a coin toss on noise.
    :raises ValueError: If the mask does not match the pattern's channel axis.
    """
    patterns = np.asarray(channel_patterns, dtype=float)
    mask = np.asarray(electrode_mask, dtype=bool)
    if mask.ndim != 1 or mask.size != patterns.shape[-1]:
        raise ValueError(
            f"electrode_mask must be ({patterns.shape[-1]},) to match the patterns' "
            f"channel axis; got {mask.shape}."
        )
    on_mask = patterns[:, :, mask].mean(axis=2)
    overall = np.abs(patterns).mean(axis=2)
    strength = np.divide(
        np.abs(on_mask), overall, out=np.zeros_like(on_mask), where=overall > 0
    )
    return np.where(on_mask >= 0, 1.0, -1.0), strength


def polarity_is_determined(
    strength: np.ndarray,
    *,
    floor: float = POLARITY_STRENGTH_FLOOR,
) -> np.ndarray:
    """Whether each component projects onto the anchor electrodes strongly enough.

    Takes the *strength* from :func:`polarity_flip`, not the flips themselves: a
    component whose participants split evenly on sign is perfectly well anchored — that
    split is the reason to flip at all. What would make an anchor untrustworthy is a
    component that hardly loads on those electrodes, leaving its sign there set by
    noise.

    :param strength: ``(participants, components)`` of
        ``|mean weight on the mask| / |mean weight overall|``.
    :param floor: Minimum median strength across participants.
    :return: Boolean ``(components,)``.
    """
    return np.median(np.asarray(strength, dtype=float), axis=0) >= floor


# ---------------------------------------------------------------------------
# Frequency selection
# ---------------------------------------------------------------------------


def frequency_selection(
    freqs: np.ndarray,
    center: float,
    halfwidth: float = 0.0,
) -> np.ndarray:
    """Bin indices covering ``center +- halfwidth`` in *freqs*.

    A zero *halfwidth* selects the single nearest bin, which is what "the 40 Hz row"
    means on a 1 Hz grid. A positive one selects every bin inside the closed interval,
    and never returns nothing: if the interval falls between bins the nearest one is
    used, because an empty selection would silently produce a NaN track.

    :param freqs: Frequency grid in Hz, ascending.
    :param center: Centre frequency in Hz.
    :param halfwidth: Half-width in Hz; ``0`` selects the nearest single bin.
    :return: Ascending bin indices.
    :raises ValueError: If *freqs* is empty or *halfwidth* is negative.
    """
    grid = np.asarray(freqs, dtype=float)
    if grid.size == 0:
        raise ValueError("freqs is empty; there is no bin to select.")
    if halfwidth < 0:
        raise ValueError(f"halfwidth must be >= 0; got {halfwidth}.")
    if halfwidth == 0:
        return np.array([int(np.argmin(np.abs(grid - center)))])
    inside = np.flatnonzero((grid >= center - halfwidth) & (grid <= center + halfwidth))
    return inside if inside.size else np.array([int(np.argmin(np.abs(grid - center)))])


def selection_label(center: float, halfwidth: float = 0.0) -> str:
    """Filename-safe name of a frequency selection, e.g. ``40hz`` or ``35-45hz``.

    :param center: Centre frequency in Hz.
    :param halfwidth: Half-width in Hz; ``0`` names the single bin.
    :return: The selection's label.
    """
    if halfwidth == 0:
        return f"{center:g}hz"
    return f"{center - halfwidth:g}-{center + halfwidth:g}hz"


# ---------------------------------------------------------------------------
# Trial cutting
# ---------------------------------------------------------------------------


def epoch_geometry(
    onsets: np.ndarray,
    n_times: int,
    sfreq: float,
) -> tuple[np.ndarray, int, int]:
    """``(usable onsets, pre, post)`` for one recording's time axis.

    The paradigm decides the window (:class:`~src.definitions.constants.AssrEpoch`) and
    the recording only *caps* it: ``post`` is shortened by the shortest inter-onset gap
    so an epoch can never reach the next stimulus. Deriving the length from the observed
    gap alone would make the window a property of whatever jitter the recording happened
    to have rather than of the paradigm.

    :param onsets: Onset sample indices, local to this time axis.
    :param n_times: Length of the time axis.
    :param sfreq: Sampling frequency in Hz.
    :return: The onsets falling inside the axis, and the pre/post sample counts.
    :raises ValueError: If no onset falls inside the axis.
    """
    inside = np.asarray(onsets, dtype=int)
    inside = inside[(inside >= 0) & (inside < n_times)]
    if inside.size == 0:
        raise ValueError(
            f"No stimulus onset falls inside the {n_times}-sample time axis."
        )
    min_gap = int(np.diff(inside).min()) if inside.size > 1 else None
    return (
        inside,
        AssrEpoch.pre_onset_samples(sfreq),
        AssrEpoch.post_onset_samples(sfreq, min_gap=min_gap),
    )


def common_epoch_window(
    geometries: Mapping[str, tuple[np.ndarray, int, int]],
) -> tuple[int, int]:
    """One ``(pre, post)`` window every condition can supply.

    Each condition caps its own ``post`` by its own shortest inter-onset gap, so they
    can differ by a sample or two. A contrast needs one window, and the shorter one is
    the only choice that does not fabricate samples, so the minimum is taken across
    conditions.

    :param geometries: Per condition, the triple :func:`epoch_geometry` returned.
    :return: The shared ``(pre, post)``.
    :raises ValueError: If *geometries* is empty, or the conditions disagree on ``pre``
        — which would mean they were cut at different sampling rates.
    """
    if not geometries:
        raise ValueError("No condition geometries given.")
    pres = {pre for _onsets, pre, _post in geometries.values()}
    if len(pres) > 1:
        raise ValueError(
            f"Conditions disagree on the pre-onset baseline ({sorted(pres)} samples); "
            "they cannot share an epoch window."
        )
    return pres.pop(), min(post for _onsets, _pre, post in geometries.values())


def cut_trials(
    array: np.ndarray,
    onsets: np.ndarray,
    pre: int,
    post: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut one fixed window per onset from the **last** axis, keeping every trial.

    The per-trial counterpart of :func:`~src.analysis.iva_quality.epoch_average`, which
    collapses the same windows to their mean. Windows overhanging either end are
    dropped, and the onsets that survived come back with the data so a trial index is
    never guessed.

    :param array: Any array whose last axis is time; leading axes are preserved.
    :param onsets: Onset sample indices.
    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :return: ``(trials, kept_onsets)`` with *trials* shaped ``(..., n_kept, pre + post)``
        — the trial axis is inserted just before time, so ``trials[..., i, :]`` is trial
        *i* of every leading index.
    :raises ValueError: If no window fits inside the recording.
    """
    array = np.asarray(array)
    n_times = array.shape[-1]
    kept = np.asarray(
        [
            int(onset)
            for onset in onsets
            if int(onset) - pre >= 0 and int(onset) + post <= n_times
        ],
        dtype=int,
    )
    if kept.size == 0:
        raise ValueError(
            f"No {pre + post}-sample onset window fits inside the {n_times}-sample "
            "recording."
        )
    windows = [array[..., onset - pre : onset + post] for onset in kept]
    return np.stack(windows, axis=-2), kept


def epoch_time_base(pre: int, post: int, sfreq: float) -> np.ndarray:
    """Epoch time base in seconds, ``0`` at the onset.

    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :param sfreq: Sampling frequency in Hz.
    :return: ``(pre + post,)`` times in seconds.
    """
    return np.arange(-pre, post) / float(sfreq)


# ---------------------------------------------------------------------------
# Per-trial normalisation
# ---------------------------------------------------------------------------


def baseline_normalise(
    trials: np.ndarray,
    baseline_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference every trial to its own pre-stimulus window.

    Per-trial rather than per-condition because it removes two things at once: the drift
    in level between trials, and the difference in absolute scale between participants,
    which spans orders of magnitude and otherwise lets two high-power participants own
    any group mean.

    Two forms come back, and only one of them is safe on every row:

    * ``z = (x - baseline) / baseline SD`` — defined for **every** trial of every
      source whatever its sign, dimensionless, and comparable across participants and
      sources. **This is the canonical form**, and the one a comparison spanning both
      learned and fixed filters must use.
    * ``rel = x / baseline - 1`` — the relative change, as a readable percentage, but
      written only where the baseline is positive; every other trial is ``NaN``,
      deliberately, so a downstream mean cannot average a sign-flipped value. That makes
      it complete for a non-negative row like an electrode average and a *biased subset*
      on a signed component row.

    :param trials: ``(..., trials, samples)`` with time last.
    :param baseline_mask: Boolean ``(samples,)`` selecting the pre-stimulus window.
    :return: ``(z, rel, baseline_positive)``, the last being the ``(..., trials)`` mask
        of trials ``rel`` is valid for.
    :raises ValueError: If *baseline_mask* selects fewer than two samples, which leaves
        the standard deviation undefined.
    """
    array = np.asarray(trials, dtype=float)
    mask = np.asarray(baseline_mask, dtype=bool)
    if int(mask.sum()) < 2:
        raise ValueError(
            f"The baseline window holds {int(mask.sum())} sample(s); at least 2 are "
            "needed for a standard deviation."
        )

    baseline = array[..., mask].mean(axis=-1, keepdims=True)
    spread = array[..., mask].std(axis=-1, ddof=1, keepdims=True)

    positive = baseline > 0
    rel = np.where(positive, array / np.where(positive, baseline, 1.0) - 1.0, np.nan)

    # A zero-variance baseline never occurs on real wavelet power, but guard rather
    # than emit a silent inf.
    usable = spread > 0
    z = np.where(usable, (array - baseline) / np.where(usable, spread, 1.0), np.nan)
    return z, rel, positive[..., 0]


# ---------------------------------------------------------------------------
# Paired tests over participants
# ---------------------------------------------------------------------------


def paired_test(differences: np.ndarray, alternative: str = "two-sided") -> dict:
    """Exact Wilcoxon signed-rank test on per-participant differences.

    **Participants are the unit, never trials.** Trials within a participant are
    correlated, so a test that treats them as independent observations is
    anticonservative. Aggregating first costs nothing: for a balanced design the paired
    test on participant summaries and a random-intercept mixed model on all trials give
    the same standard error, because ``Var = sigma_b^2/P + sigma_w^2/(P*m)`` and the
    degrees of freedom are set by the number of participants either way.

    Wilcoxon enumerates all ``2**P`` sign assignments over the **ranks** of ``|d|``,
    which makes the p-value exact and free of any distributional assumption. Ranking is
    why it is preferred over the same enumeration on the raw differences: it bounds how
    far a single participant can move the result, which matters when a per-trial
    normalisation divides by a baseline SD estimated from few samples.

    Note the ceiling this implies: the smallest attainable two-sided p is ``2/2**P``,
    reached only when every participant points the same way.

    Deliberately not offered: a test of ``|d|``. It would remove the need for a
    direction and would also destroy the test — ``|d| >= 0`` by construction, so "is it
    above zero?" is true whenever there is any noise at all.

    :param differences: One value per participant.
    :param alternative: ``"two-sided"``, or a directional alternative where the
        direction was fixed in advance *and* the rows' polarity makes it meaningful.
    :return: The median difference, a robust effect size (median over its own MAD), how
        many participants point positive, the alternative used, and the p-value.
    :raises ValueError: If fewer than two participants have finite values.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        raise ValueError(f"Need at least 2 participants; got {n}.")

    # method="exact" enumerates the sign lattice rather than approximating it; scipy
    # would otherwise switch to the normal approximation as n grows.
    result = wilcoxon(d, alternative=alternative, method="exact")

    spread = 1.4826 * np.median(np.abs(d - np.median(d)))
    return {
        "median": float(np.median(d)),
        "effect": float(np.median(d) / spread) if spread > 0 else np.nan,
        "same sign": f"{int((d > 0).sum())}/{n}",
        "alt": alternative,
        "p": float(result.pvalue),
    }


def p_floor(n_participants: int, *, one_sided: bool = False) -> float:
    """Smallest attainable p for a paired sign test over *n_participants*.

    :param n_participants: Number of paired observations.
    :param one_sided: Whether the alternative is directional.
    :return: The floor.
    """
    return (1.0 if one_sided else 2.0) / 2**n_participants


def condition_contrast(
    values: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    source_index: int,
) -> np.ndarray:
    """Per-participant ``conditions[0] - conditions[1]`` for one source.

    :param values: Per condition, a ``(participants, sources)`` array.
    :param conditions: The two conditions, in subtraction order.
    :param source_index: Row of the source axis.
    :return: ``(participants,)`` differences.
    """
    first, second = conditions[0], conditions[1]
    return values[first][:, source_index] - values[second][:, source_index]


def discrimination_gain(
    values: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    source_index: int,
    reference_index: int,
) -> np.ndarray:
    """How much better one source separates the conditions than a reference source.

    The **interaction**, per participant::

        (cond0_source - cond1_source) - (cond0_reference - cond1_reference)

    Zero means the source separates the conditions exactly as well as the reference;
    positive means better. This is the right contrast when the reference is there to be
    a *discriminator* rather than another detector.

    Comparing source against reference **within** a condition instead splits this into
    two halves that only mean something together — their difference is algebraically
    this same quantity — and each half alone answers "which filter picks up more
    signal", which is not what the conditions are being compared for.

    :param values: Per condition, a ``(participants, sources)`` array.
    :param conditions: The two conditions, in subtraction order.
    :param source_index: Row of the source being judged.
    :param reference_index: Row of the reference source.
    :return: ``(participants,)`` interaction values.
    """
    return condition_contrast(values, conditions, source_index) - condition_contrast(
        values, conditions, reference_index
    )


# ---------------------------------------------------------------------------
# The extracted product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssrTrialSet:
    """Per-trial time courses of one frequency selection, for every condition.

    :param trials: Per condition, ``(participants, sources, trials, samples)`` of raw
        wavelet power. Trial counts may differ between conditions; nothing else may.
    :param trials_z: Per condition, the same array referenced to each trial's own
        pre-stimulus window in units of its SD. The canonical form for any comparison.
    :param trials_rel: Per condition, ``x / baseline - 1``, ``NaN`` where the baseline
        was not positive.
    :param baseline_positive: Per condition, ``(participants, sources, trials)`` marking
        the trials ``trials_rel`` is valid for.
    :param onsets: Per condition, the sample index of each kept trial's onset on that
        condition's own time axis, aligned with the trial axis.
    :param participants: Participant label per row of the first axis.
    :param labels: Label per row of the source axis — ``IC <k>`` rows then
        :data:`BINARY_FILTER_LABEL`.
    :param times: Epoch time base in seconds, ``0`` at the onset.
    :param sfreq: Sampling frequency in Hz.
    :param freqs: The frequency bins averaged into the time course.
    :param selection: Filename-safe name of that selection (:func:`selection_label`).
    :param binary_channels: Channel names the binary filter selected.
    :param metadata: Free-form provenance — which store the filters came from, the
        decomposition's settings, what was and was not done to the signal.
    """

    trials: Mapping[str, np.ndarray]
    trials_z: Mapping[str, np.ndarray]
    trials_rel: Mapping[str, np.ndarray]
    baseline_positive: Mapping[str, np.ndarray]
    onsets: Mapping[str, np.ndarray]
    participants: tuple[str, ...]
    labels: tuple[str, ...]
    times: np.ndarray
    sfreq: float
    freqs: np.ndarray
    selection: str
    binary_channels: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Conditions this set holds, in insertion order."""
        return tuple(self.trials)

    @property
    def n_participants(self) -> int:
        """Length of the participant axis."""
        return len(self.participants)

    @property
    def n_sources(self) -> int:
        """Length of the source axis (components plus the binary filter)."""
        return len(self.labels)

    @property
    def reference_index(self) -> int:
        """Row of the source axis holding the fixed binary filter."""
        return self.source_index(BINARY_FILTER_LABEL)

    def source_index(self, label: str) -> int:
        """Row of the source axis carrying *label*.

        :param label: A source label, e.g. ``"IC 1"`` or :data:`BINARY_FILTER_LABEL`.
        :return: Its index on the source axis.
        :raises KeyError: If no source carries that label.
        """
        if label not in self.labels:
            raise KeyError(
                f"No source labelled {label!r}; this set holds {list(self.labels)}."
            )
        return self.labels.index(label)

    def stimulus_mask(self) -> np.ndarray:
        """Boolean mask over :attr:`times` selecting the driven interval."""
        return AssrEpoch.stimulus_mask(self.times)

    def baseline_mask(self) -> np.ndarray:
        """Boolean mask over :attr:`times` selecting the pre-onset window."""
        return self.times < 0.0

    def participant_frame(self) -> pd.DataFrame:
        """One row per ``(condition, participant)`` with that cell's trial count.

        :return: The row bookkeeping in tabular form.
        """
        records = [
            {
                "condition": condition,
                "participant": participant,
                "participant_index": index,
                "n_trials": int(array.shape[2]),
            }
            for condition, array in self.trials.items()
            for index, participant in enumerate(self.participants)
        ]
        return pd.DataFrame.from_records(records)


def trials_filename(
    variant: str,
    music_type: str,
    selection: str,
    n_pca: int,
) -> str:
    """Canonical filename of an extracted trial set.

    Mirrors :func:`~src.io.iva_store.iva_results_filename`: the settings that decide
    what is in the file are in its name, so an entry can be found without opening it.
    The frequency selection is part of the name because two selections of the same run
    are two different products.

    :param variant: The IVA variant the filters came from.
    :param music_type: Music type (a placeholder for ASSR).
    :param selection: Frequency-selection label (:func:`selection_label`).
    :param n_pca: The run's component count.
    :return: The filename, with extension.
    """
    return f"assr_trials__{variant}__{music_type}__{selection}__pca{n_pca}.npz"


def save_assr_trials(path: Path, trial_set: AssrTrialSet) -> Path:
    """Write a trial set to a compressed ``.npz``.

    Per-condition arrays are keyed ``<name>__<condition>`` rather than stacked, because
    the conditions may keep different trial counts and padding them to a common length
    would invent data.

    :param path: Destination file; parent directories are created.
    :param trial_set: The set to write.
    :return: *path*.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "participants": np.asarray(trial_set.participants, dtype=object),
        "source_labels": np.asarray(trial_set.labels, dtype=object),
        "conditions": np.asarray(trial_set.conditions, dtype=object),
        "times": np.asarray(trial_set.times, dtype=float),
        "sfreq": np.asarray(trial_set.sfreq, dtype=float),
        "freqs": np.asarray(trial_set.freqs, dtype=float),
        "selection": np.asarray(trial_set.selection),
        "binary_channels": np.asarray(trial_set.binary_channels, dtype=object),
    }
    for condition in trial_set.conditions:
        payload[f"{TRIALS_KEY}__{condition}"] = np.asarray(trial_set.trials[condition])
        payload[f"{TRIALS_Z_KEY}__{condition}"] = np.asarray(
            trial_set.trials_z[condition]
        )
        payload[f"{TRIALS_REL_KEY}__{condition}"] = np.asarray(
            trial_set.trials_rel[condition]
        )
        payload[f"{BASELINE_POSITIVE_KEY}__{condition}"] = np.asarray(
            trial_set.baseline_positive[condition]
        )
        payload[f"{ONSETS_KEY}__{condition}"] = np.asarray(
            trial_set.onsets[condition], dtype=int
        )
    for key, value in trial_set.metadata.items():
        payload[f"meta__{key}"] = np.asarray(str(value))

    np.savez_compressed(path, **payload)
    return path


def load_assr_trials(path: Path) -> AssrTrialSet:
    """Read back a trial set written by :func:`save_assr_trials`.

    :param path: The ``.npz`` to read.
    :return: The trial set.
    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If a declared condition has no trial array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trial file at {path}.")

    with np.load(path, allow_pickle=True) as stored:
        conditions = [str(name) for name in stored["conditions"]]
        blocks: dict[str, dict[str, np.ndarray]] = {
            TRIALS_KEY: {},
            TRIALS_Z_KEY: {},
            TRIALS_REL_KEY: {},
            BASELINE_POSITIVE_KEY: {},
            ONSETS_KEY: {},
        }
        for condition in conditions:
            for name in blocks:
                key = f"{name}__{condition}"
                if key not in stored:
                    raise ValueError(
                        f"{path.name} declares condition {condition!r} but holds no "
                        f"'{key}' array."
                    )
                blocks[name][condition] = stored[key]
        metadata = {
            name.removeprefix("meta__"): str(stored[name])
            for name in stored.files
            if name.startswith("meta__")
        }
        return AssrTrialSet(
            trials=blocks[TRIALS_KEY],
            trials_z=blocks[TRIALS_Z_KEY],
            trials_rel=blocks[TRIALS_REL_KEY],
            baseline_positive=blocks[BASELINE_POSITIVE_KEY],
            onsets=blocks[ONSETS_KEY],
            participants=tuple(str(p) for p in stored["participants"]),
            labels=tuple(str(s) for s in stored["source_labels"]),
            times=np.asarray(stored["times"], dtype=float),
            sfreq=float(stored["sfreq"]),
            freqs=np.asarray(stored["freqs"], dtype=float),
            selection=str(stored["selection"]),
            binary_channels=tuple(str(c) for c in stored["binary_channels"]),
            metadata=metadata,
        )


def resolve_conditions(
    conditions: Sequence[ConditionVariants | str],
) -> list[str]:
    """Condition names as the plain strings the trial arrays are keyed by.

    :param conditions: Conditions, as enum members or their values.
    :return: Their string values, in order.
    """
    return [
        condition.value if isinstance(condition, ConditionVariants) else str(condition)
        for condition in conditions
    ]
