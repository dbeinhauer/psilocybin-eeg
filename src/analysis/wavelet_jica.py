"""
Joint ICA (jICA) of wavelet power: assembling a join, fitting it, and its filters.

The ICA counterpart of :mod:`src.analysis.iva_condition_comparison`. Both force
independence over the **same** axis — the joint ``(frequency, time)`` plane, with
channels as the mixing dimension — and both run on the same two participant-matched
joins of the two conditions. They differ in how the recordings are tied together:

* **IVA** gives every recording its own unmixing matrix and couples them through
  the source-component vector, so a component has one source *per recording*.
* **jICA** lays every recording's channels side by side into a single feature axis
  and fits **one** FastICA to the lot, so a component has one **shared** source
  and a mixing column that splits into per-recording blocks.

**Scope.** This module stops where the decomposition stops. Everything downstream —
the frequency selection, the projection, the stimulus-locked trials, the per-trial
normalisation and the paired tests — is decomposition-agnostic and already lives in
:mod:`src.analysis.assr_trials`; jICA reaches it through
:meth:`JointIcaResult.recording_filters` and :func:`cohort_mean_filter`, both of
which hand over the same ``(components, channels)`` operator a stored IVA run does.
Nothing here is reimplemented from there.

The **polarity anchoring** is the one exception, and deliberately so: it is the
one downstream step whose right granularity differs between the two
decompositions, so jICA has its own :func:`component_polarity` rather than reusing
:func:`~src.analysis.assr_trials.polarity_flip` directly. See below.

Two consequences of the construction are worth stating before any number computed
here is read.

**There is no per-recording sign ambiguity.** FastICA's ``E[G(w.x)]`` with an even
contrast function is invariant under flipping an *entire* unmixing row and nothing
else, so a component's sign flips its map and its whole mixing column together and
a recording whose block comes out negative is genuinely inverted relative to the
rest — a result, not an ambiguity. jICA therefore needs none of the alignment
passes the IVA path must spend (IVA-G's ``sum_k log det Sigma_k`` *is* invariant
under per-dataset flips, since ``det(D Sigma D) = det(Sigma) det(D)**2``, which is
exactly why those signs are unidentifiable there), and carries none of the risk
that arbitrary flips manufacture a condition difference.

Two functions pin the one sign that *is* free, and they are the only ones that may:
:func:`orient_components` settles it hypothesis-free at fit time, and
:func:`component_polarity` re-pins the same single sign against a fixed electrode
selection when a directional test needs "higher = more power there" to be true.
Neither works per recording. Applying
:func:`~src.analysis.assr_trials.polarity_flip` per ``(recording, component)`` here
— the IVA convention — would overwrite a determined quantity, and under a
recording-axis join could flip one condition of a pair and not the other.

**A per-block filter is a contribution, not a component.** For component *k*::

    source_k(f, t) = sum_b  u_{b,k} . x_b(f, t)

so ``u_{b,k}`` — row *k* of the unmixing matrix restricted to block *b*'s channels
— is block *b*'s **contribution** to the shared source rather than its own copy of
the component. It is the only per-recording read-out jICA offers and it is what
:func:`recording_filters` returns, but a block with a near-zero loading contributes
almost nothing, so its extracted time course is near-noise while still carrying
full weight in any group statistic.

Every function here is **pure**: no file or plot output, and no mutation of inputs.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from src.analysis.wavelet_ica import zscore_by_time
from src.definitions.fields import ConditionVariants

_logger = logging.getLogger(__name__)

#: Read-out variants the stage-07 workflows run side by side, keyed by a **(spatial
#: filter, signal)** pair — the same organising idea the stage-06 IVA notebooks use.
#: Every default variant applies a per-trial pre-stimulus baseline, so all four land in
#: the same units (that trial's own pre-stimulus SD) and are directly comparable:
#:
#: .. code-block:: text
#:
#:                          | raw wavelet power | z-scored (what the fit saw)
#:     ---------------------+-------------------+----------------------------
#:     that block's filter  | "prestim"         | "zscored_prestim"
#:     cohort-mean filter   | "mean_prestim"    | "mean_zscored_prestim"
#:
#: Down the **signal** axis:
#:
#: * raw power — applying the filter to un-z-scored power drops the per-channel
#:   ``1/sd`` weighting the fit folded in, so the result *approximates* the component.
#: * z-scored — the filter is applied to exactly the signal the decomposition saw, so
#:   the per-block terms ``u_{b,k} . x_b^z`` **are** the component; they sum over
#:   blocks to the global TF map. Referenced to each trial's own pre-stimulus window
#:   afterwards so it shares an axis with the fixed reference. The counterpart of the
#:   IVA stage's ``stored_prestim``.
#:
#: Down the **filter** axis:
#:
#: * ``"own"`` — that block's own row of the unmixing matrix
#:   (:meth:`JointIcaResult.recording_filters`), i.e. that recording's contribution to
#:   the shared source. One operator per recording, ``C`` free parameters each.
#: * ``"mean"`` — one cohort-mean operator shared by everybody
#:   (:func:`cohort_mean_pattern` then :func:`cohort_mean_filter`). The point of
#:   carrying it: a per-recording filter spends ``C`` parameters rediscovering a
#:   stereotyped topography, and the variance of that estimate can exceed the bias it
#:   removes. The cohort filter is the same comparison the fixed electrode mask
#:   already makes — a single group-level spatial weighting — but learned rather than
#:   drawn.
#:
#: ``"zscored"`` (z-scored signal, **no** baseline) stays a legal choice for
#: back-compatibility, but it is not part of the default grid: without a per-trial
#: baseline it is not in the reference's units, which is exactly what the grid is for.
#:
#: **The reference row is held fixed across variants** — see
#: :data:`VARIANT_REFERENCE_FROM`. Unlike the IVA stage there is no trial-count
#: asymmetry between variants: every route reads the same cached extent, so all of
#: them carry the same trials and the same participants.
#: * ``"masked"`` / ``"mean_masked"`` — either of the two above with every weight
#:   **outside the ASSR electrodes zeroed** (:func:`~src.analysis.assr_trials.
#:   restrict_filters_to_mask`). These sit between the two things the grid already has:
#:   the binary reference reads the anchor electrodes with *equal* weight, an
#:   unrestricted component reads the *whole head* with learned weights, and these read
#:   only those electrodes with learned weights — which is what separates a better
#:   weighting *inside* the anchor area from access to signal *outside* it, a
#:   distinction no other row can make. Both read the **z-scored** signal, because the
#:   weights were estimated on it: a weighted average of raw power carries each
#:   electrode's own power level on top, the very artefact
#:   :func:`~src.analysis.assr_trials.roi_channelwise_snr` removes from the binary row.
#:   The IVA stage's ``masked_prestim`` / ``mean_masked_prestim``.
SIGNAL_VARIANTS = (
    "prestim",
    "zscored_prestim",
    "mean_prestim",
    "mean_zscored_prestim",
    "masked_prestim",
    "mean_masked_prestim",
    "zscored",
)

#: The grid the stage-07 workflows run unless told otherwise: the six
#: (filter x signal) cells, all per-trial baselined, matching the IVA stage's grid
#: cell for cell. ``"zscored"`` is deliberately excluded — see :data:`SIGNAL_VARIANTS`.
DEFAULT_SIGNAL_VARIANTS = (
    "prestim",
    "zscored_prestim",
    "mean_prestim",
    "mean_zscored_prestim",
    "masked_prestim",
    "mean_masked_prestim",
)

#: What each variant name means, as a ``filter`` / ``zscore`` / ``baseline`` triple.
#: Read by both the notebook and ``scripts/run_wavelet_jica.py`` so they cannot
#: disagree about it.
VARIANT_RECIPE = {
    "prestim": {"filter": "own", "zscore": False, "baseline": True},
    "zscored_prestim": {"filter": "own", "zscore": True, "baseline": True},
    "mean_prestim": {"filter": "mean", "zscore": False, "baseline": True},
    "mean_zscored_prestim": {"filter": "mean", "zscore": True, "baseline": True},
    "masked_prestim": {"filter": "masked", "zscore": True, "baseline": True},
    "mean_masked_prestim": {"filter": "mean_masked", "zscore": True, "baseline": True},
    "zscored": {"filter": "own", "zscore": True, "baseline": False},
}

#: Which variants read ONE operator shared by the whole cohort rather than each
#: recording's own. A shared operator carries a single sign, resolved once when it was
#: built, so a per-recording flip must never be applied to these rows. (jICA resolves
#: one sign per component for every row, so this is bookkeeping the figures read rather
#: than a second convention — see :func:`component_polarity`.)
SHARED_FILTER_VARIANTS = ("mean_prestim", "mean_zscored_prestim", "mean_masked_prestim")

#: Variants whose fixed-reference rows are taken from another variant rather than
#: recomputed, mapped to the variant they come from.
#:
#: **Every per-trial-baselined variant borrows from** ``"prestim"``, which is the
#: invariant the whole grid rests on: the reference is what a component is *judged
#: against*, so holding it still is what makes a difference between two cells a
#: difference of spatial filters and nothing else. The same rule the IVA stage applies
#: (``run_assr_snr_grid.py`` carries ``prestim``'s reference columns verbatim into every
#: other variant).
#:
#: For a variant that reads raw power the borrow is a no-op — it would recompute the
#: identical numbers — and it is listed anyway so the invariant is stated once rather
#: than inferred per variant. For a z-scored one it is load-bearing: the mask on
#: z-scored data is a different quantity.
#:
#: ``"zscored"`` is absent on purpose: with no per-trial baseline there is nothing to
#: put it in the borrowed rows' units, so it keeps the mask read on exactly the signal
#: its components were read on.
VARIANT_REFERENCE_FROM = {
    "zscored_prestim": "prestim",
    "mean_prestim": "prestim",
    "mean_zscored_prestim": "prestim",
    "masked_prestim": "prestim",
    "mean_masked_prestim": "prestim",
}

#: Units each variant's response is in, for figure axes.
VARIANT_UNITS = {
    "prestim": "pre-stimulus SD",
    "zscored_prestim": "pre-stimulus SD, exact component",
    "mean_prestim": "pre-stimulus SD, cohort filter",
    "mean_zscored_prestim": "pre-stimulus SD, cohort filter, exact component",
    "masked_prestim": "pre-stimulus SD, ASSR electrodes only",
    "mean_masked_prestim": "pre-stimulus SD, cohort filter, ASSR electrodes only",
    "zscored": "z-scored over time",
}

#: Residual tolerance on ``components_ @ mixing_ - I``. Expect 1e-7 to 1e-5, not
#: 1e-15: the wavelet caches are float32, so sklearn computes ``mixing_`` as a
#: pseudo-inverse in that precision over ``n_blocks * n_channels`` features, and an
#: ill-conditioned whitening inflates it further. Orders of magnitude above this
#: mean the unmixing is genuinely rank-deficient and the per-block filters are not
#: the operator the fit used.
IDENTITY_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Assembling a join
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinLayout:
    """The channel-block bookkeeping of one join, plus its FastICA input.

    The whole of the difference between the two joins lives in :attr:`blocks`::

        JOINED         [(participant, condition), ...]   B = 2P
        JOINED_TRACKS  [(participant, None),      ...]   B = P

    and :meth:`block_of` hides it, so a caller can ask for "this recording's filter"
    and get the right one under either join.

    :param matrix: ``(n_freqs * n_times, n_blocks * n_channels)`` z-scored samples by
        stacked-channels matrix — the FastICA input. Samples run frequency-slow /
        time-fast and features block-slow / channel-fast, so both axes fold back
        exactly after the fit.
    :param blocks: Identity of each channel block, in feature-axis order.
    :param participants: Participant label per row of the per-condition arrays the
        layout was assembled from.
    :param conditions: Conditions in the order they were joined.
    :param join: Which join this is.
    :param n_channels: Channels per block.
    :param n_freqs: Frequency bins on the sample axis.
    :param n_times: Time samples on the sample axis (both tracks for a time join).
    """

    matrix: np.ndarray
    blocks: tuple[tuple[str, Optional[str]], ...]
    participants: tuple[str, ...]
    conditions: tuple[str, ...]
    join: ConditionVariants
    n_channels: int
    n_freqs: int
    n_times: int

    @property
    def n_blocks(self) -> int:
        """Number of channel blocks on the feature axis."""
        return len(self.blocks)

    def block_of(self, participant: str, condition: str) -> int:
        """Feature-block index carrying *participant*'s filter for *condition*.

        :param participant: Participant label.
        :param condition: Condition name.
        :return: Index into the block axis.
        :raises KeyError: If the pair is not part of this join.
        """
        key = (
            participant,
            condition if self.join is ConditionVariants.JOINED else None,
        )
        index = {block: b for b, block in enumerate(self.blocks)}
        if key not in index:
            raise KeyError(
                f"No channel block for {key}; this join holds participants "
                f"{sorted({b[0] for b in self.blocks})} under conditions "
                f"{sorted(str(b[1]) for b in self.blocks)}."
            )
        return index[key]

    def block_rows(self, condition: str) -> list[int]:
        """Block index per participant, in :attr:`participants` order.

        :param condition: Condition whose recordings are wanted.
        :return: One block index per participant.
        """
        return [self.block_of(p, condition) for p in self.participants]


def assemble_join(
    raw_by_condition: dict[str, np.ndarray],
    participants: Sequence[str],
    conditions: Sequence[str],
    join: ConditionVariants,
) -> JoinLayout:
    """Standardise and stack one join's recordings into a FastICA input matrix.

    The standardisation is the decomposition's own: each ``(block, channel,
    frequency)`` series is z-scored along time. Under
    :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` that means
    z-scoring **each condition's track on its own and then concatenating**, which is
    the ``per_condition`` mode of
    :mod:`src.analysis.condition_tracks` — so an overall power difference between
    the conditions is normalised away and what the components describe is temporal
    and spectral *structure*. The inputs are not modified, so raw power stays
    available for the trial extraction.

    Because every series is standardised independently, every **column** of the
    returned matrix has unit variance by construction, which is what makes the
    stacking fair: no recording can dominate the joint whitening by amplitude.

    :param raw_by_condition: Condition name → ``(P, C, F, T)`` un-z-scored wavelet
        power, participant-matched row for row.
    :param participants: Participant label per row, in row order.
    :param conditions: Conditions in the order they should be joined.
    :param join: :attr:`~src.definitions.fields.ConditionVariants.JOINED` (one block
        per recording) or
        :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` (one block
        per participant, sample axis spanning both tracks).
    :return: The assembled :class:`JoinLayout`.
    :raises ValueError: If the join is not one of the two, fewer than two conditions
        are given, a condition is missing, an array is not 4-D, the non-time axes
        disagree between conditions, or a row count does not match *participants*.
    """
    conditions = [str(c) for c in conditions]
    participants = [str(p) for p in participants]
    if join not in (ConditionVariants.JOINED, ConditionVariants.JOINED_TRACKS):
        raise ValueError(
            f"join must be JOINED or JOINED_TRACKS; got {join}. Those are the two "
            "ways this module knows how to tie the recordings together."
        )
    if len(conditions) < 2:
        raise ValueError(
            f"At least two conditions are needed to join; got {conditions}."
        )
    for condition in conditions:
        if condition not in raw_by_condition:
            raise ValueError(f"No array supplied for condition {condition!r}.")

    reference = np.asarray(raw_by_condition[conditions[0]])
    if reference.ndim != 4:
        raise ValueError(
            f"Arrays must be (P, C, F, T); got shape {reference.shape} for "
            f"{conditions[0]!r}."
        )
    for condition in conditions:
        array = np.asarray(raw_by_condition[condition])
        if array.shape[:-1] != reference.shape[:-1]:
            raise ValueError(
                f"Condition {condition!r} has non-time axes {array.shape[:-1]}, "
                f"expected {reference.shape[:-1]} to match {conditions[0]!r}."
            )
        if array.shape[0] != len(participants):
            raise ValueError(
                f"Condition {condition!r} has {array.shape[0]} row(s) but "
                f"{len(participants)} participant label(s)."
            )

    _n_participants, n_channels, n_freqs, _ = reference.shape
    standardised = [zscore_by_time(np.asarray(raw_by_condition[c])) for c in conditions]

    if join is ConditionVariants.JOINED:
        # One block per recording: stack on the block axis, condition-major, so the
        # feature order matches `blocks` below.
        block_data = np.concatenate(standardised, axis=0)
        blocks = [(p, c) for c in conditions for p in participants]
    else:
        # One block per participant: lay the standardised tracks end to end.
        block_data = np.concatenate(standardised, axis=-1)
        blocks = [(p, None) for p in participants]

    n_blocks = block_data.shape[0]
    n_times = block_data.shape[-1]
    matrix = np.ascontiguousarray(
        block_data.reshape(n_blocks * n_channels, n_freqs * n_times).T
    )
    _logger.info(
        f"[{join.value}] FastICA input {matrix.shape} "
        f"(F*T samples, B*C mixing variables), {matrix.nbytes / 1e9:.2f} GB"
    )
    return JoinLayout(
        matrix=matrix,
        blocks=tuple(blocks),
        participants=tuple(participants),
        conditions=tuple(conditions),
        join=join,
        n_channels=n_channels,
        n_freqs=n_freqs,
        n_times=n_times,
    )


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointIcaResult:
    """One joint-ICA decomposition, sign-oriented and ordered by size.

    :param tf_maps: ``(K, F, T)`` **global** component time-frequency maps — one per
        component, shared by every recording.
    :param filters: ``(K, B, C)`` per-block spatial filters: the unmixing rows split
        by channel block. The **backward** model, and the only one that may be
        applied to a new signal.
    :param patterns: ``(B, K, C)`` per-block forward patterns — the topographies, and
        what a polarity anchor must read. Not interchangeable with :attr:`filters`
        (Haufe et al., 2014, NeuroImage 87:96-110).
    :param sources: ``(F*T, K)`` unit-variance independent sources, on the unfolded
        sample axis.
    :param ic_variance: ``(K,)`` share of the joint channel-space energy each
        component's rank-one back-projection accounts for.
    :param block_ic_energy: ``(B, K)`` that share split by channel block — the
        per-recording loading. Because the sources are shared and unit-variance, it
        is exactly proportional to the squared L2 norm of that block's slice of the
        mixing column, i.e. the loading in the ordinary sense.
    :param retained: Fraction of the joint channel-space variance the whitening
        truncation kept: the hard ceiling on everything the components can carry.
    :param converged: Whether FastICA met its tolerance. ``False`` means the unmixing
        is wherever the solver stopped rather than a fixed point, and nothing
        downstream is worth reading from it.
    :param signs: ``(K,)`` global sign :func:`orient_components` applied, in the
        final component order.
    :param order: ``(K,)`` permutation applied to sort by :attr:`ic_variance`.
    """

    tf_maps: np.ndarray
    filters: np.ndarray
    patterns: np.ndarray
    sources: np.ndarray
    ic_variance: np.ndarray
    block_ic_energy: np.ndarray
    retained: float
    converged: bool
    signs: np.ndarray
    order: np.ndarray

    @property
    def n_components(self) -> int:
        """Number of components ``K``."""
        return self.tf_maps.shape[0]

    def identity_error(self) -> float:
        """``max |U A - I|`` over the recovered filter/pattern pair.

        ``mixing_`` is ``pinv(components_)``, so the composite must be the ``K x K``
        identity. See :data:`IDENTITY_TOLERANCE` for the magnitude to expect and what
        exceeding it means.

        :return: The largest absolute deviation from the identity.
        """
        k = self.n_components
        unmixing = self.filters.reshape(k, -1)
        mixing = self.patterns.transpose(1, 0, 2).reshape(k, -1)
        return float(np.abs(unmixing @ mixing.T - np.eye(k)).max())

    def filter_share(self) -> np.ndarray:
        """``(K, B)`` share of each component's filter energy carried by each block.

        A block with a near-zero row contributes almost nothing to that component, so
        the time course projected for it is near-noise. Read alongside
        :attr:`block_ic_energy`, which says the same thing on the forward side.

        :return: Rows summing to one.
        """
        share = (self.filters**2).sum(axis=2)
        return share / share.sum(axis=1, keepdims=True)

    def recording_filters(self, layout: JoinLayout, condition: str) -> np.ndarray:
        """``(P, K, C)`` filters, one per participant, for one condition.

        The bridge to :mod:`src.analysis.assr_trials`: the shape a stored IVA run's
        :func:`~src.analysis.assr_trials.recover_spatial_filters` produces, so every
        step from :func:`~src.analysis.assr_trials.stack_filters` onwards is shared
        between the two decompositions rather than reimplemented here.

        Under :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` the
        same block serves both conditions, so both calls return the identical
        operator — which is the point of that join: a condition difference cannot
        come from the filter having changed.

        :param layout: The join this result was fitted on.
        :param condition: Condition whose recordings are wanted.
        :return: ``(participants, components, channels)``, rows in
            :attr:`JoinLayout.participants` order.
        :raises ValueError: If *layout* does not match this result.
        """
        if layout.n_blocks != self.filters.shape[1]:
            raise ValueError(
                f"layout has {layout.n_blocks} block(s) but the result was fitted on "
                f"{self.filters.shape[1]}; they do not belong together."
            )
        rows = layout.block_rows(condition)
        return np.stack([self.filters[:, b, :] for b in rows])

    def recording_patterns(self, layout: JoinLayout, condition: str) -> np.ndarray:
        """``(P, K, C)`` forward patterns, one per participant, for one condition.

        The counterpart of :meth:`recording_filters` on the forward side: the
        topographies, and what belongs on a topomap. A polarity anchor must read the
        **pattern**, never the filter — but in jICA it must also read it at
        :func:`component_polarity`'s granularity, one sign for the whole component.
        Handing these per-recording patterns to
        :func:`~src.analysis.assr_trials.polarity_flip` reproduces the IVA convention,
        which is wrong here: see the module docstring.

        :param layout: The join this result was fitted on.
        :param condition: Condition whose recordings are wanted.
        :return: ``(participants, components, channels)``.
        :raises ValueError: If *layout* does not match this result.
        """
        if layout.n_blocks != self.patterns.shape[0]:
            raise ValueError(
                f"layout has {layout.n_blocks} block(s) but the result was fitted on "
                f"{self.patterns.shape[0]}; they do not belong together."
            )
        return np.stack([self.patterns[b] for b in layout.block_rows(condition)])


def fit_joint_ica(
    layout: JoinLayout,
    *,
    n_ica: int,
    random_state: int = 42,
    algorithm: str = "parallel",
    fun: str = "logcosh",
    max_iter: int = 2000,
    tol: float = 1e-4,
    chunk: int = 20_000,
) -> JointIcaResult:
    """Reduce the stacked channel axis, fit FastICA on the scores, fold both axes back.

    **The reduction is done here rather than inside FastICA, and it has to be.**
    ``FastICA(n_components=K)`` would whiten by calling ``scipy.linalg.svd`` on the
    transposed input, and LAPACK indexes with 32-bit integers — so a matrix with more
    than ``2**31 - 1`` elements is refused outright::

        ValueError: Indexing a matrix of 10993320000 elements would incur an in
        integer overflow in LAPACK.

    At 2340 features that ceiling is ~918k samples, i.e. ~73 s of both ASSR tracks at
    a 50-bin frequency grid — far short of the full recording. This is a hard limit,
    not a memory budget: no node size makes that call work.

    So the leading ``K`` directions are obtained from the ``(n_features,
    n_features)`` covariance instead, which is 2340 x 2340 here and trivial. The two
    routes span the **same** subspace — ``u`` and ``d`` of ``svd(X.T)`` are exactly
    the eigenvectors and square-root eigenvalues of ``X.T X`` — so this is the same
    decomposition, reached by the arithmetic that fits. It is also far cheaper: no
    LAPACK workspace proportional to the data, and the covariance accumulates in
    blocks of samples so the centred matrix is never formed whole.

    FastICA then runs on the ``(n_samples, K)`` scores, and the operators are composed
    back into the original feature space. Because the PCA loadings ``P`` have
    orthonormal rows::

        unmixing = ica.components_ @ P                 # (K, n_features)
        mixing   = P.T @ ica.mixing_ = pinv(unmixing)  # (n_features, K)

    which is the same composition :func:`~src.analysis.wavelet_ica._run_pca_ica` and
    :func:`~src.analysis.wavelet_ica.iva_component_patterns` use, so
    :meth:`JointIcaResult.identity_error` still holds exactly.

    :param layout: The assembled join, carrying both the matrix and the axis sizes.
    :param n_ica: Components to extract, and the dimension the channel axis is
        reduced to. Not capped by the channel count — the feature axis is ``B*C`` —
        but capped in practice by convergence.
    :param random_state: Seed for FastICA's initial unmixing.
    :param algorithm: ``"parallel"`` or ``"deflation"``; the latter extracts one
        component at a time and often converges where the symmetric update
        oscillates.
    :param fun: Contrast function: ``"logcosh"``, ``"exp"`` or ``"cube"``.
    :param max_iter: Maximum FastICA iterations.
    :param tol: FastICA convergence tolerance.
    :param chunk: Sample rows per block for the covariance and the projection. Each
        block's Gram product is formed in the input's own dtype and accumulated in
        float64, so a 4.7M-sample pass does not lose precision to float32 summation.
    :return: The sign-oriented, size-ordered :class:`JointIcaResult`.
    :raises ValueError: If *n_ica* is not between 1 and the feature count, or the
        input has no variance to decompose.
    """
    matrix = np.asarray(layout.matrix)
    n_features = layout.n_blocks * layout.n_channels
    if matrix.shape[1] != n_features:
        raise ValueError(
            f"layout.matrix has {matrix.shape[1]} feature(s) but the layout has "
            f"{layout.n_blocks} block(s) x {layout.n_channels} channel(s) = "
            f"{n_features}."
        )
    if not 1 <= n_ica <= n_features:
        raise ValueError(
            f"n_ica must be between 1 and the feature count ({n_features}); got "
            f"{n_ica}."
        )

    # ---- the leading K directions, from the covariance ---------------------
    n_samples = matrix.shape[0]
    feature_mean = matrix.mean(axis=0, dtype=np.float64)
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    for start in range(0, n_samples, chunk):
        block = matrix[start : start + chunk] - feature_mean.astype(matrix.dtype)
        gram += (block.T @ block).astype(np.float64)
    total_ss = float(np.trace(gram))
    if total_ss <= 0.0:
        raise ValueError(
            "The input matrix has zero total variance; there is nothing to decompose."
        )
    # eigh returns ascending eigenvalues, so the leading directions are the last ones.
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    top = np.argsort(eigenvalues)[::-1][:n_ica]
    loadings = np.ascontiguousarray(eigenvectors[:, top].T)  # (K, n_features)
    # The hard ceiling on everything the components can carry: whatever is orthogonal
    # to the retained subspace is gone. Exactly a PCA's explained_variance_ratio_ sum.
    retained = float(np.clip(eigenvalues[top].sum() / total_ss, 0.0, 1.0))

    scores = np.empty((n_samples, n_ica), dtype=np.float64)
    for start in range(0, n_samples, chunk):
        block = matrix[start : start + chunk] - feature_mean.astype(matrix.dtype)
        scores[start : start + chunk] = block @ loadings.T
    _logger.info(
        f"[{layout.join.value}] reduced {n_features} -> {n_ica} channel direction(s) "
        f"via the {n_features}x{n_features} covariance; scores {scores.shape}, "
        f"retained {retained * 100:.1f}%"
    )

    # ---- FastICA on the scores -------------------------------------------
    ica = FastICA(
        n_components=n_ica,
        algorithm=algorithm,
        fun=fun,
        whiten="unit-variance",
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
    )
    # Capture ConvergenceWarning instead of letting it print once and vanish: a
    # non-converged unmixing is an arbitrary point on the solver's path.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        sources = ica.fit_transform(scores)  # (F*T, K)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not converged:
        _logger.warning(
            f"FastICA did not converge within max_iter={max_iter} at tol={tol:g}; the "
            "unmixing is wherever the solver stopped, not a fixed point. Lower n_ica "
            "or try algorithm='deflation'."
        )

    # The two operators, composed back into the original feature space. They are NOT
    # interchangeable: unmixing extracts a source from the channels (backward), mixing
    # says how a source projects onto them (forward).
    unmixing = np.asarray(ica.components_, dtype=float) @ loadings  # (K, B*C)
    mixing = loadings.T @ np.asarray(ica.mixing_, dtype=float)  # (B*C, K)

    source_ss = (sources**2).sum(axis=0)  # (K,)
    # Rank-one energy per feature. Summing over a block's channels gives that block's
    # share; summing over everything gives ic_variance.
    feature_shares = source_ss[:, np.newaxis] * mixing.T**2 / total_ss
    ic_variance = feature_shares.sum(axis=1)
    block_ic_energy = (
        feature_shares.reshape(n_ica, layout.n_blocks, layout.n_channels).sum(axis=2).T
    )  # (B, K)

    _logger.info(
        f"[{layout.join.value}] {n_ica} IC(s), retained {retained * 100:.1f}% of the "
        f"joint channel space ({n_ica} of {n_features} directions), "
        f"converged={converged}"
    )
    return orient_components(
        JointIcaResult(
            tf_maps=sources.T.reshape(n_ica, layout.n_freqs, layout.n_times),
            filters=unmixing.reshape(n_ica, layout.n_blocks, layout.n_channels),
            patterns=mixing.T.reshape(
                n_ica, layout.n_blocks, layout.n_channels
            ).transpose(1, 0, 2),
            sources=sources,
            ic_variance=ic_variance,
            block_ic_energy=block_ic_energy,
            retained=retained,
            converged=converged,
            signs=np.ones(n_ica),
            order=np.arange(n_ica),
        )
    )


def orient_components(result: JointIcaResult) -> JointIcaResult:
    """Pin each component's global sign and sort the components by size.

    FastICA fixes neither, so both have to be settled before anything is plotted or
    tested — otherwise the same data refitted with a different seed looks like a
    different result. Neither choice is a claim about the data:

    * **Sign** — ``(map, pattern)`` and ``(-map, -pattern)`` are the *same*
      component, so each is flipped to make the **largest absolute excursion of its
      TF map positive**, i.e. its dominant event reads as a power increase. That is
      deliberately hypothesis-free (it says nothing about 40 Hz), so it stays valid
      whatever is measured next. It is *not* the anchoring a directional test needs —
      that is :func:`~src.analysis.assr_trials.polarity_flip`, which orients a
      component to a fixed electrode selection.
    * **Order** — descending :attr:`~JointIcaResult.ic_variance`, so ``IC 1``
      accounts for the largest share of the joint channel-space energy. A statement
      about *size*, not relevance: a 40 Hz steady state can easily sit below a broad
      onset response.

    **One sign for the whole component, and that is the point.** It multiplies the
    map, the filter row *and* the pattern together, so a block that comes out
    negative is genuinely inverted rather than arbitrarily flipped, and the
    back-projection ``sum_k source_k (x) pattern_k`` is untouched by either
    operation.

    :param result: A freshly fitted decomposition.
    :return: A **new** result with the flips and the permutation applied and recorded
        in :attr:`~JointIcaResult.signs` / :attr:`~JointIcaResult.order`.
    """
    n_ica = result.n_components
    flat = result.tf_maps.reshape(n_ica, -1)
    peak = np.take_along_axis(flat, np.abs(flat).argmax(axis=1)[:, np.newaxis], axis=1)
    signs = np.where(peak[:, 0] < 0.0, -1.0, 1.0)
    order = np.argsort(-result.ic_variance, kind="stable")

    return JointIcaResult(
        tf_maps=(result.tf_maps * signs[:, np.newaxis, np.newaxis])[order],
        filters=(result.filters * signs[:, np.newaxis, np.newaxis])[order],
        patterns=(result.patterns * signs[np.newaxis, :, np.newaxis])[:, order],
        sources=result.sources[:, order] * signs[order][np.newaxis, :],
        ic_variance=result.ic_variance[order],
        block_ic_energy=result.block_ic_energy[:, order],
        retained=result.retained,
        converged=result.converged,
        signs=signs[order],
        order=order,
    )


# ---------------------------------------------------------------------------
# The cohort-mean spatial filter, and the one sign per component
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortPattern:
    """One forward topography per component, averaged over every channel block.

    :param pattern: ``(K, C)`` cohort-mean forward model.
    :param cosine: ``(B, K)`` cosine between each block's own pattern and the mean.
        **This is the number that says whether the mean is a consensus or a
        cancellation.** Values near 1 mean the cohort agrees about where component *k*
        sits; values scattered around 0, or negative for a substantial share of blocks,
        mean it does not — and then the mean topography is a weak operator whatever its
        norm says.
    :param normalized: Whether each block's pattern was scaled to unit L2 norm before
        averaging.
    :param aligned: Whether block signs were flipped toward the anchor before
        averaging. ``False`` is the jICA default; see :func:`cohort_mean_pattern`.
    """

    pattern: np.ndarray
    cosine: np.ndarray
    normalized: bool
    aligned: bool

    @property
    def n_components(self) -> int:
        """Number of components ``K``."""
        return self.pattern.shape[0]

    def disagreeing_blocks(self) -> np.ndarray:
        """``(K,)`` count of blocks whose pattern points away from the cohort mean."""
        return (self.cosine < 0.0).sum(axis=0)


def cohort_mean_pattern(
    patterns: np.ndarray,
    *,
    normalize: bool = True,
    align_to: Optional[np.ndarray] = None,
) -> CohortPattern:
    """Average the per-block forward patterns into one topography per component.

    The operator behind the ``mean_*`` variants (:data:`SIGNAL_VARIANTS`), and the
    thing :func:`component_polarity` anchors the component sign to.

    Averaged over **every** block, both conditions pooled. Never per condition: a
    condition-specific filter would break the exchangeability the paired contrast rests
    on, because a difference could then come from the operator having changed rather
    than from the signal. (Under
    :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` the patterns are
    shared anyway, so this only bites under ``JOINED``.)

    Each pattern is scaled to **unit L2 norm** first. That matters more here than it
    does for IVA: jICA's per-block loadings are unconstrained and can be wildly uneven,
    so without it a single high-loading block would set the cohort topography by itself.

    **Signs are left as the fit produced them by default**, and that is the one
    deliberate departure from the stage-06 IVA construction. IVA flips each
    ``(recording, component)`` pattern toward the anchor before averaging, and must:
    IVA-G's objective is invariant under per-dataset sign flips, so those signs are
    arbitrary. FastICA's is invariant only under flipping an *entire* unmixing row, so
    a jICA block's sign is a **result** — a block that comes out negative genuinely
    points away from the rest. Aligning first would manufacture a coherent cohort
    topography where the cohort has none. :attr:`CohortPattern.cosine` is what reports
    whether that happened, and *align_to* is kept so the two can be compared.

    :param patterns: ``(B, K, C)`` per-block forward patterns, e.g.
        :attr:`JointIcaResult.patterns`. **Not** mutated.
    :param normalize: Scale each block's pattern to unit L2 norm before averaging.
    :param align_to: Boolean ``(C,)`` anchor mask. When given, each block's pattern is
        first flipped so its correlation with the mask is non-negative — the stage-06
        convention. ``None`` (the default) leaves the fit's own signs alone.
    :return: The cohort pattern and its per-block cosine diagnostic.
    :raises ValueError: If *patterns* is not 3-D, is empty on any axis, or *align_to*
        does not match its channel axis.
    """
    array = np.asarray(patterns, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"patterns must be (B, K, C); got shape {array.shape}.")
    if min(array.shape) == 0:
        raise ValueError(
            f"patterns must be non-empty on every axis; got shape {array.shape}."
        )

    scaled = array
    if normalize:
        norms = np.linalg.norm(scaled, axis=2, keepdims=True)
        # A block with an all-zero pattern contributes nothing either way; leave it at
        # zero rather than dividing by it.
        scaled = np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms > 0.0)

    aligned = align_to is not None
    if aligned:
        mask = np.asarray(align_to, dtype=bool)
        if mask.ndim != 1 or mask.size != array.shape[-1]:
            raise ValueError(
                f"align_to must be ({array.shape[-1]},) to match the patterns' channel "
                f"axis; got {mask.shape}."
            )
        from src.analysis import assr_trials as _at  # local: avoids an import cycle

        flip, _strength = _at.polarity_flip(scaled, mask)
        scaled = scaled * flip[:, :, np.newaxis]

    mean = scaled.mean(axis=0)  # (K, C)

    # Cosine of every block against the mean, per component. Computed on the same
    # (possibly flipped, possibly normalised) patterns the mean was built from, so it
    # describes that mean rather than some other quantity.
    mean_norm = np.linalg.norm(mean, axis=1)  # (K,)
    block_norm = np.linalg.norm(scaled, axis=2)  # (B, K)
    denominator = block_norm * mean_norm[np.newaxis, :]
    numerator = np.einsum("bkc,kc->bk", scaled, mean)
    cosine = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0
    )

    return CohortPattern(
        pattern=mean,
        cosine=np.clip(cosine, -1.0, 1.0),
        normalized=normalize,
        aligned=aligned,
    )


def cohort_mean_filter(
    mean_pattern: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Invert a cohort forward model into the backward operator that extracts it.

    The forward pattern says how a source projects onto the channels; the **filter** is
    what recovers it from them, and the two are not interchangeable (Haufe et al., 2014,
    NeuroImage 87:96-110). This is the same ``pinv`` composition the stage-06 IVA
    notebooks use for their cohort filter, so the two stages' ``mean_*`` rows are the
    same kind of operator.

    The round trip is checked rather than trusted: averaging patterns can leave a
    rank-deficient forward model — most easily when two components' cohort topographies
    collapse onto each other — and the pseudo-inverse of that is not the operator it
    claims to be.

    :param mean_pattern: ``(K, C)`` cohort forward model, from
        :func:`cohort_mean_pattern`.
    :param tolerance: Largest tolerated ``max |U A - I|``.
    :return: ``(K, C)`` backward operator.
    :raises ValueError: If *mean_pattern* is not 2-D, or the round trip exceeds
        *tolerance*.
    """
    forward = np.asarray(mean_pattern, dtype=float)
    if forward.ndim != 2:
        raise ValueError(f"mean_pattern must be (K, C); got shape {forward.shape}.")
    n_components = forward.shape[0]

    backward = np.linalg.pinv(forward.T)  # (K, C)
    residual = float(np.abs(backward @ forward.T - np.eye(n_components)).max())
    if residual > tolerance:
        raise ValueError(
            f"The cohort filter does not invert the cohort pattern (max residual "
            f"{residual:.2e} > {tolerance:.0e}). The averaged forward model is "
            "rank-deficient — most likely two components' cohort topographies have "
            "collapsed onto each other — so the pseudo-inverse is not the operator it "
            "claims to be. Lower n_ica, or read the per-block cosine diagnostic."
        )
    return backward


def component_polarity(
    mean_pattern: np.ndarray,
    electrode_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """**One** sign per component, anchored to a fixed electrode selection.

    The whole of the sign freedom jICA leaves. :func:`orient_components` has already
    pinned it once, hypothesis-free (largest ``|TF|`` excursion positive), which says
    nothing about the electrodes a directional test cares about. This re-pins the same
    single sign so that "higher means more power over the anchor area" is true of the
    **component**, which is what a one-sided prior needs in order to be stated at all.

    **Why one sign and not one per recording.** FastICA's ``E[G(w.x)]`` with even ``G``
    is invariant under flipping an entire unmixing row and nothing else, so after
    :func:`orient_components` there is no per-recording sign left to resolve — a block
    that comes out negative is genuinely inverted relative to the rest, a result rather
    than an ambiguity. Flipping per recording, as the IVA stage must
    (:func:`~src.analysis.assr_trials.polarity_flip`, where IVA-G's invariance makes
    those signs arbitrary), would overwrite that result. Worse, under a
    recording-axis join it is resolved per ``(participant, condition)``, so one
    condition of a pair can flip and the other not — which turns the paired difference
    into a paired *sum*.

    Being one vector over components, this sign is identical for every participant and
    both conditions, so it cannot manufacture a condition difference under any join.
    And it reads only a spatial property, never the tested response.

    :param mean_pattern: ``(K, C)`` cohort forward model, from
        :func:`cohort_mean_pattern`. The anchor is deliberately the same object the
        ``mean_*`` variants filter with, so the sign convention and that filter cannot
        disagree about which way a component points.
    :param electrode_mask: Boolean ``(C,)`` anchor selection.
    :return: ``(flip, strength)`` — ``flip`` is ``(K,)`` of +-1, ``strength`` is
        ``(K,)`` of ``|corr(pattern, mask)|``, to be read against
        :data:`~src.analysis.assr_trials.POLARITY_CORR_FLOOR`.
    :raises ValueError: If *mean_pattern* is not 2-D, or whatever
        :func:`~src.analysis.assr_trials.polarity_flip` rejects.
    """
    from src.analysis import assr_trials as _at  # local: avoids an import cycle

    forward = np.asarray(mean_pattern, dtype=float)
    if forward.ndim != 2:
        raise ValueError(f"mean_pattern must be (K, C); got shape {forward.shape}.")
    # polarity_flip works per (recording, component); one "recording" here is the
    # cohort, so the same criterion and the same floor apply unchanged.
    flip, strength = _at.polarity_flip(forward[np.newaxis], electrode_mask)
    return flip[0], strength[0]


# ---------------------------------------------------------------------------
# The retained-subspace reference rows
# ---------------------------------------------------------------------------


def subspace_mask_rows(
    result: JointIcaResult,
    mask_weights: np.ndarray,
) -> np.ndarray:
    """Each block's electrode mask, carried through the joint channel reduction.

    The operator behind the ``ASSR-mask (PCA)`` and ``ASSR-mask (PCA, z)`` reference
    rows (:data:`~src.analysis.assr_trials.MASK_LABELS`): the same fronto-central
    average the binary row takes, but reading **only what the reduction left
    available** — so a component is compared against a reference that lives in the
    same subspace it does, rather than against one with access to directions the fit
    threw away.

    With orthonormal-row PCA loadings ``P``, the part of the data the fit could use is
    ``P^T P x`` — a rank-``K`` orthogonal projector — and it is recoverable from the
    operators already in hand, because ``A W = I``
    (:meth:`JointIcaResult.identity_error` is the check)::

        mixing @ unmixing = (P^T A) (W P) = P^T P

    So the row for block *b* is ``m_b^T P^T P``, and it factors through the components
    without ever forming the ``(B*C, B*C)`` matrix::

        coeffs = patterns @ mask          # (B, K)   = m_b^T P^T A
        rows   = coeffs @ unmixing        # (B, B*C) = m_b^T P^T P

    **A row spans every block, not just its own, and that is the point.** jICA reduces
    the *stacked* channel axis, so the subspace couples the recordings: what the fit
    was handed for recording *b* at one sample genuinely depends on the other
    recordings at that sample. This is the one place the stage-07 reference differs in
    kind from the stage-06 one, whose PCA is fitted per recording
    (:func:`~src.analysis.assr_trials.pca_mask_rows`) and so has no cross-block term.
    Read that as a property of the decomposition rather than of the reference: a
    per-block restriction would be cheaper and look more like the IVA row, but it is
    not the operator this fit applied.

    :param result: The fitted decomposition.
    :param mask_weights: ``(C,)`` weights over one block's channels, e.g.
        :func:`~src.analysis.assr_trials.binary_filter_weights` of the ASSR mask.
    :return: ``(B, B*C)`` rows, one per channel block, in feature-axis order.
    :raises ValueError: If *mask_weights* does not match the channel axis.
    """
    patterns = np.asarray(result.patterns, dtype=float)  # (B, K, C)
    weights = np.asarray(mask_weights, dtype=float)
    if weights.ndim != 1 or weights.size != patterns.shape[2]:
        raise ValueError(
            f"mask_weights must be ({patterns.shape[2]},) to match the channel axis; "
            f"got {weights.shape}."
        )
    unmixing = np.asarray(result.filters, dtype=float).reshape(
        result.n_components, -1
    )  # (K, B*C)
    return (patterns @ weights) @ unmixing


def subspace_reference_rows(
    raw_by_condition: Mapping[str, np.ndarray],
    layout: JoinLayout,
    mask_rows: np.ndarray,
    bins: Sequence[int],
    *,
    zscore: bool,
) -> dict[str, np.ndarray]:
    """Apply :func:`subspace_mask_rows` to the data, per condition and participant.

    The rows span the whole stacked feature axis, so the data has to be stacked the
    way the fit stacked it before they can be applied — which is what this does, on
    the frequency selection rather than the whole grid. Averaging over the selection
    commutes with the rows (they act on channels alone), so the band mean is taken
    first and the stacked array is ``(B*C, T)`` rather than ``(B*C, F_sel, T)``.

    *zscore* selects which of the two PCA references is being built: ``False`` reads
    RAW power (``ASSR-mask (PCA)``), ``True`` reads the z-scored signal the fit was
    handed (``ASSR-mask (PCA, z)``). The standardisation is per ``(block, channel,
    frequency)`` over that condition's own time axis, exactly as :func:`assemble_join`
    does it, so restricting to the selection's bins first changes nothing.

    **Expect the raw row to be the weaker of the two here**, which is a property of the
    joint reduction rather than a fault. The retained directions were chosen on
    *z-scored* data over ``B*C`` features, so raw power — whose variance is dominated by
    each channel's own scale — is close to orthogonal to them, and ``K`` of a few
    thousand directions leaves little of the mask standing either way
    (:attr:`JointIcaResult.retained` is the honest summary). The stage-06 row of the
    same name is reduced per recording, ``C -> K``, so far more of the mask survives
    there; the two are the same construction on decompositions that kept different
    amounts, not two different constructions. The per-trial baseline downstream puts
    both in pre-stimulus-SD units regardless, which is what makes them comparable to
    the equal-weight row at all.

    :param raw_by_condition: Condition → ``(P, C, F, T)`` **un-z-scored** power, the
        same arrays the layout was assembled from.
    :param layout: The join the decomposition was fitted on.
    :param mask_rows: ``(B, B*C)`` rows from :func:`subspace_mask_rows`.
    :param bins: Frequency bins of this selection, indexing the ``F`` axis.
    :param zscore: Read the z-scored signal instead of raw power.
    :return: Condition → ``(P, T)`` reference row per participant, rows in
        :attr:`JoinLayout.participants` order.
    :raises ValueError: If *mask_rows* does not match the layout, or a condition's
        array is missing.
    """
    rows = np.asarray(mask_rows, dtype=float)
    n_features = layout.n_blocks * layout.n_channels
    if rows.shape != (layout.n_blocks, n_features):
        raise ValueError(
            f"mask_rows must be ({layout.n_blocks}, {n_features}) to match the "
            f"layout's stacked feature axis; got {rows.shape}."
        )
    for condition in layout.conditions:
        if condition not in raw_by_condition:
            raise ValueError(f"No array supplied for condition {condition!r}.")

    selection = np.asarray(bins, dtype=int)
    # Under JOINED every block already names its own condition, so one stacked array
    # serves both; under JOINED_TRACKS a block is a participant and the array is that
    # condition's own track. Keyed accordingly so the shared case is built once.
    stacked_by_key: dict[Optional[str], np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for condition in layout.conditions:
        key = None if layout.join is ConditionVariants.JOINED else condition
        if key not in stacked_by_key:
            stacked_by_key[key] = _stacked_band(
                raw_by_condition, layout, condition, selection, zscore=zscore
            )
        projected = rows @ stacked_by_key[key]  # (B, T)
        out[condition] = projected[
            [layout.block_of(p, condition) for p in layout.participants]
        ]
    return out


def _stacked_band(
    raw_by_condition: Mapping[str, np.ndarray],
    layout: JoinLayout,
    condition: str,
    bins: np.ndarray,
    *,
    zscore: bool,
) -> np.ndarray:
    """``(B*C, T)`` band-averaged data, stacked block-slow / channel-fast.

    The same feature order :func:`assemble_join` builds, which is what makes the rows
    of :func:`subspace_mask_rows` applicable to it.

    :param raw_by_condition: Condition → ``(P, C, F, T)`` un-z-scored power.
    :param layout: The join being read.
    :param condition: Condition whose time axis is being built. Blocks that name their
        own condition (a recording-axis join) read theirs; blocks that do not (a time
        join) read this one.
    :param bins: Frequency bins to average over.
    :param zscore: Standardise each ``(channel, frequency)`` series over time first.
    :return: The stacked array.
    """
    n_channels = layout.n_channels
    n_times = np.asarray(raw_by_condition[condition]).shape[-1]
    stacked = np.empty((layout.n_blocks * n_channels, n_times), dtype=float)
    row_of = {participant: r for r, participant in enumerate(layout.participants)}
    for block, (participant, block_condition) in enumerate(layout.blocks):
        source = np.asarray(
            raw_by_condition[condition if block_condition is None else block_condition]
        )
        band = np.asarray(source[row_of[participant]][:, bins, :], dtype=float)
        if zscore:
            mean = band.mean(axis=-1, keepdims=True)
            deviation = band.std(axis=-1, keepdims=True)
            band = (band - mean) / np.where(deviation == 0.0, 1.0, deviation)
        start = block * n_channels
        stacked[start : start + n_channels] = band.mean(axis=1)
    return stacked


# ---------------------------------------------------------------------------
# Small shared utilities the stage's figures need
# ---------------------------------------------------------------------------


def bootstrap_median_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the median, resampling **participants**.

    Shown alongside the tests to convey spread only: the p-values come from the exact
    Wilcoxon test in :func:`~src.analysis.assr_trials.paired_test`, not from this, and
    the two can disagree slightly at small ``P``.

    :param differences: One value per participant; non-finite entries are dropped.
    :param n_bootstrap: Resamples to draw.
    :param seed: Seed, so the interval is reproducible.
    :param alpha: Two-sided coverage, e.g. ``0.05`` for a 95% interval.
    :return: ``(low, high)``, or ``(nan, nan)`` when fewer than two values remain.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.median(d[rng.integers(0, d.size, size=(n_bootstrap, d.size))], axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def participant_spread(
    values: np.ndarray, mode: str = "sem"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre line and spread across the **participant** axis of a ``(P, W)`` array.

    Across participants, never across trials: trials within a participant are
    correlated, so a band drawn from them would look tight while saying nothing about
    how well the effect generalises to a new person.

    :param values: ``(P, W)`` per-participant courses.
    :param mode: ``"sem"`` for mean +/- standard error, ``"iqr"`` for the median with
        a 25-75 band.
    :return: ``(centre, low, high)``.
    :raises ValueError: On a non-2-D input or an unknown *mode*.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be (P, W); got shape {values.shape}.")
    if mode == "sem":
        centre = np.nanmean(values, axis=0)
        n = np.sum(~np.isnan(values), axis=0)
        half = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        return centre, centre - half, centre + half
    if mode == "iqr":
        return (
            np.nanmedian(values, axis=0),
            np.nanpercentile(values, 25, axis=0),
            np.nanpercentile(values, 75, axis=0),
        )
    raise ValueError(f"mode must be 'sem' or 'iqr'; got {mode!r}.")
