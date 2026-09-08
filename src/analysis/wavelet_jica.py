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
R
**Scope.** This module stops where the decomposition stops. Everything downstream —
the frequency selection, the projection, the stimulus-locked trials, the per-trial
normalisation, the polarity anchoring and the paired tests — is
decomposition-agnostic and already lives in :mod:`src.analysis.assr_trials`; jICA
reaches it through :func:`recording_filters`, which hands over the same
``(participants, components, channels)`` operator a stored IVA run does. Nothing
here is reimplemented from there.

Two consequences of the construction are worth stating before any number computed
here is read.

**There is no per-recording sign ambiguity.** A component's sign flips its map and
its *whole* mixing column together, so a recording whose block comes out negative
is genuinely inverted relative to the rest — a result, not an ambiguity. jICA
therefore needs none of the two alignment passes the IVA path spends on
``Sigma_N`` and on PC1 of the TF maps, and carries none of the risk that arbitrary
flips manufacture a condition difference. :func:`orient_components` pins the one
remaining global sign per component.

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from src.analysis.wavelet_ica import zscore_by_time
from src.definitions.fields import ConditionVariants

_logger = logging.getLogger(__name__)

#: Signal-normalisation variants the stage-07 workflows run side by side. They differ
#: in two independent ways — which signal the filter is applied to, and whether each
#: trial is referenced to its own pre-stimulus window — and the three of them fill the
#: reachable cells of that 2x2:
#:
#: .. code-block:: text
#:
#:                        | no per-trial baseline | per-trial pre-stimulus baseline
#:     -------------------+-----------------------+--------------------------------
#:     raw wavelet power  |          --           | "prestim"
#:     z-scored (the fit) | "zscored"             | "zscored_prestim"
#:
#: * ``"prestim"`` — the **raw** wavelet power is projected and every trial is
#:   referenced to its own pre-stimulus interval
#:   (:func:`~src.analysis.assr_trials.baseline_normalise`), so the value is a
#:   signal-to-noise ratio in units of that trial's pre-stimulus SD. Its one flaw is
#:   that applying the filter to un-z-scored power drops the per-channel ``1/sd``
#:   weighting the fit folded in, so the result approximates the component rather
#:   than reproducing it.
#: * ``"zscored"`` — the power is z-scored along time before projection, i.e. the
#:   filter is applied to exactly the signal the decomposition saw. **This is the
#:   component itself, not an approximation**: the per-block terms
#:   ``u_{b,k} . x_b^z`` sum over blocks to the global TF map. Already
#:   dimensionless, so no per-trial baseline is applied.
#: * ``"zscored_prestim"`` — the exact component from ``"zscored"``, then referenced
#:   to each trial's own pre-stimulus window, which puts it in the same
#:   pre-stimulus-SD units as the fixed reference so the two can be read on one
#:   axis. The counterpart of the IVA stage's ``stored_prestim``.
#:
#: **The reference row is held fixed across variants.** In ``"zscored_prestim"`` the
#: fixed-electrode row is ``"prestim"``'s verbatim — raw power through the mask,
#: per-trial baselined — because it is the quantity a component is judged against, so
#: holding it still is what makes the ``prestim`` / ``zscored_prestim`` comparison mean
#: one thing: the component is derived a different way, and nothing else changed.
#:
#: Unlike the IVA stage there is no trial-count asymmetry between the variants: every
#: route reads the same cached extent, so all of them carry the same trials and the
#: same participants.
SIGNAL_VARIANTS = ("prestim", "zscored", "zscored_prestim")

#: Which variants z-score along time before projecting, and which apply a per-trial
#: pre-stimulus baseline afterwards. Read by the stage-07 workflows so the notebook
#: and the CLI cannot disagree about what a variant name means.
VARIANT_RECIPE = {
    "prestim": {"zscore": False, "baseline": True},
    "zscored": {"zscore": True, "baseline": False},
    "zscored_prestim": {"zscore": True, "baseline": True},
}

#: Variants whose fixed-reference row is taken from another variant rather than
#: recomputed, mapped to the variant it comes from. See :data:`SIGNAL_VARIANTS`.
VARIANT_REFERENCE_FROM = {"zscored_prestim": "prestim"}

#: Units each variant's response is in, for figure axes.
VARIANT_UNITS = {
    "prestim": "pre-stimulus SD",
    "zscored": "z-scored over time",
    "zscored_prestim": "pre-stimulus SD, exact component",
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

        The counterpart of :meth:`recording_filters` on the forward side, and the
        input :func:`~src.analysis.assr_trials.polarity_flip` needs: a polarity anchor
        must read the **pattern**, never the filter.

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
