"""
Sign alignment and per-condition aggregation for a subject-axis-pooled IVA run.

When Placebo and Psilocybin are pooled on the **subject** axis
(:attr:`~src.definitions.fields.ConditionVariants.JOINED`, assembled by
:func:`~src.analysis.condition_tracks.pool_condition_subjects`), IVA returns one
aligned set of sources but a **per-recording** sign for each of them. Two things
follow, and this module handles both.

**The sign has to be resolved before anything is averaged or compared.** IVA recovers
each component only up to a per-dataset sign, so recording *i*'s copy of component *k*
may be the negation of recording *j*'s. Averaging unresolved maps drives the group mean
toward zero — with *S* recordings it shrinks to roughly ``1/sqrt(S)`` of the aligned one
— and on a *pooled* dataset it is worse than a lost mean: the flips fall arbitrarily
across the two condition blocks, so an unresolved sign manufactures a condition
difference out of nothing.

:func:`align_tf_pc1_signs` resolves it from the **time-frequency maps themselves**. For
each component it takes PC1 of the per-recording ``(F, T)`` maps — the single direction
in time-frequency space that best accounts for the ensemble — and flips every recording
whose projection onto it is negative, so afterwards every recording projects positively
and PC1 is what they agree on. Both the source maps and the channel patterns are flipped
together (:func:`apply_component_signs`), so a component's map and its topography never
disagree about which way is up.

Why PC1 of the maps rather than the ``Sigma_N`` criterion
(:func:`~src.analysis.wavelet_ica.align_iva_component_signs`): they answer the same
question with different evidence, and this one uses the quantity actually being plotted.
``Sigma_N`` is IVA's internal cross-dataset covariance over the whole flattened sample
axis; PC1 here is measured on the reshaped ``(F, T)`` maps that the figures show, so the
maps a figure draws are aligned by their own content. Composing the two is well defined
— both are per-``(component, recording)`` sign flips — so this can be applied on top of
an already ``Sigma_N``-aligned decomposition, and it decides the final orientation.

**PCA is not centred across recordings here.** Centring would subtract the group-mean
map, i.e. remove exactly the shared component whose orientation is being resolved, and
leave PC1 describing deviations from it. The leading direction of the *uncentred*
ensemble is the one that makes "everybody projects positively" mean "everybody agrees
with the group".

The group's overall orientation is still free — PC1 and ``-PC1`` describe the same
ensemble — so it is anchored by making PC1's largest-magnitude time-frequency bin
positive. That is reproducible, independent of recording order, and the same convention
:func:`~src.analysis.wavelet_ica.align_iva_component_signs` uses for its eigenvector.

All functions here are **pure**: no file or plot output, and no mutation of inputs.
(:func:`decompose_channel_iva` reports progress through the module logger, which is
the only side effect anywhere here.)
"""

from __future__ import annotations

import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from independent_vector_analysis import iva_g
from sklearn.decomposition import PCA

from src.analysis.wavelet_ica import (
    align_iva_component_signs,
    iva_component_patterns,
    zscore_by_time,
)
from src.definitions.frequency import FREQUENCY_BANDS

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfPc1Polarity:
    """Per-``(recording, component)`` signs derived from PC1 of the TF maps.

    :param signs: ``(S, K)`` array of ``+1`` / ``-1``. Entry ``[s, k]`` is what
        recording *s*'s component *k* must be multiplied by. Apply it with
        :func:`apply_component_signs` to every quantity carrying that sign — the source
        maps *and* the channel patterns.
    :param pc1: ``(K, F, T)`` oriented PC1 map per component: what the recordings agree
        on once aligned, and the template every sign was decided against.
    :param loadings: ``(S, K)`` projection of each recording's map onto its component's
        oriented PC1, **before** the flip. Its sign is what :attr:`signs` inverts; its
        magnitude says how strongly that recording expresses the shared map, so a
        near-zero entry marks a recording whose sign was decided on almost no evidence.
    :param explained_variance_ratio: ``(K,)`` fraction of the ensemble's total sum of
        squares captured by PC1. Uncentred (see the module docstring), so it is a
        share of raw power rather than of variance about the mean. Low values mean the
        recordings have no single dominant shared map, which makes the alignment — and
        any group mean built on it — weak evidence.
    """

    signs: np.ndarray
    pc1: np.ndarray
    loadings: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def n_flipped(self) -> int:
        """Number of ``(recording, component)`` pairs the alignment inverts."""
        return int((self.signs < 0).sum())

    def flipped_per_component(self) -> np.ndarray:
        """``(K,)`` count of recordings inverted for each component."""
        return (self.signs < 0).sum(axis=0)


def align_tf_pc1_signs(sources: np.ndarray) -> TfPc1Polarity:
    """Resolve each component's per-recording sign from PC1 of its TF maps.

    See the module docstring for why PC1 of the uncentred ensemble is the criterion and
    how the group's overall orientation is anchored.

    The leading direction is obtained from the ``S × S`` Gram matrix rather than an SVD
    of the ``S × (F·T)`` matrix: they give the same answer, and with ``F·T`` in the
    hundreds of thousands the Gram route is the one that fits comfortably in memory.

    :param sources: ``(S, K, F, T)`` per-recording component time-frequency maps, e.g.
        the ``iva_sources`` of a channel-IVA run. **Not** mutated.
    :return: The signs, the oriented PC1 map per component, the pre-flip loadings and
        PC1's share of the ensemble power.
    :raises ValueError: If *sources* is not 4-D, or has no recordings, components,
        frequencies or time samples.
    """
    data = np.asarray(sources, dtype=float)
    if data.ndim != 4:
        raise ValueError(f"sources must be (S, K, F, T); got shape {data.shape}.")
    n_subjects, n_components, n_freqs, n_times = data.shape
    if min(data.shape) == 0:
        raise ValueError(
            f"sources must be non-empty on every axis; got shape {data.shape}."
        )

    signs = np.ones((n_subjects, n_components))
    loadings = np.zeros((n_subjects, n_components))
    pc1 = np.zeros((n_components, n_freqs, n_times))
    evr = np.zeros(n_components)

    for k in range(n_components):
        maps = data[:, k].reshape(n_subjects, n_freqs * n_times)  # (S, F*T)
        # Uncentred Gram matrix: eigenvector 1 gives the recording weights of the
        # leading direction, its eigenvalue that direction's share of the power.
        gram = maps @ maps.T  # (S, S)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors[:, -1]  # leading eigenvector, unit norm
        total = float(np.trace(gram))
        evr[k] = float(eigenvalues[-1]) / total if total > 0 else 0.0

        direction = maps.T @ weights  # (F*T,) unoriented PC1
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            # An all-zero component: no direction to align to, so nothing is flipped.
            continue
        direction = direction / norm
        # Anchor the group's overall orientation: strongest bin positive.
        if direction[int(np.argmax(np.abs(direction)))] < 0:
            direction = -direction
            weights = -weights

        pc1[k] = direction.reshape(n_freqs, n_times)
        loadings[:, k] = maps @ direction
        # A recording that projects negatively is inverted relative to the rest. Zero
        # is left unflipped: there is no evidence either way.
        signs[:, k] = np.where(loadings[:, k] < 0, -1.0, 1.0)

    return TfPc1Polarity(
        signs=signs, pc1=pc1, loadings=loadings, explained_variance_ratio=evr
    )


def apply_component_signs(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply per-``(recording, component)`` signs to any array carrying them.

    Deliberately shape-agnostic past the first two axes, because the same signs must be
    applied to every quantity a component owns or its map and its topography end up
    disagreeing: source maps ``(S, K, F, T)``, channel patterns ``(S, K, C)``, temporal
    or spectral marginals ``(S, K, T)`` / ``(S, K, F)``.

    :param values: ``(S, K, ...)`` array to orient. **Not** mutated.
    :param signs: ``(S, K)`` signs, e.g. :attr:`TfPc1Polarity.signs`.
    :return: A **new** array of the same shape, sign-oriented.
    :raises ValueError: If *values* has fewer than two axes, or its leading two axes do
        not match *signs*.
    """
    data = np.asarray(values, dtype=float)
    signs = np.asarray(signs, dtype=float)
    if signs.ndim != 2:
        raise ValueError(f"signs must be (S, K); got shape {signs.shape}.")
    if data.ndim < 2:
        raise ValueError(f"values must be (S, K, ...); got shape {data.shape}.")
    if data.shape[:2] != signs.shape:
        raise ValueError(
            f"values has leading axes {data.shape[:2]} but signs is {signs.shape}."
        )
    return data * signs.reshape(signs.shape + (1,) * (data.ndim - 2))


def condition_component_means(
    values: np.ndarray,
    subject_conditions: Sequence[str],
    conditions: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Average a pooled per-recording array within each condition.

    The group statistic every condition-comparison figure is built on. Sign-align
    *values* first (:func:`align_tf_pc1_signs`, :func:`apply_component_signs`): an
    unresolved sign cancels each condition's mean by a different amount, which reads as
    a condition difference.

    :param values: ``(S, K, ...)`` pooled array — sources, patterns, marginals.
    :param subject_conditions: Condition of each recording, in recording order.
    :param conditions: Conditions to average, in the order wanted. ``None`` uses every
        condition present, in first-appearance order.
    :return: Condition name → ``(K, ...)`` mean over that condition's recordings.
    :raises ValueError: If *subject_conditions* does not match the recording axis, or a
        requested condition has no recordings.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim < 2:
        raise ValueError(f"values must be (S, K, ...); got shape {data.shape}.")
    labels = list(subject_conditions)
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"subject_conditions has {len(labels)} entries but values has "
            f"{data.shape[0]} recording(s)."
        )
    if conditions is None:
        conditions = list(dict.fromkeys(labels))

    means: dict[str, np.ndarray] = {}
    for condition in conditions:
        mask = np.asarray([label == condition for label in labels], dtype=bool)
        if not mask.any():
            raise ValueError(
                f"Condition {condition!r} has no recordings; present: "
                f"{sorted(set(labels))}."
            )
        means[condition] = data[mask].mean(axis=0)
    return means


def condition_difference(
    means: Mapping[str, np.ndarray],
    conditions: Sequence[str],
) -> np.ndarray:
    """Difference between two conditions' means, in the given order.

    :param means: Output of :func:`condition_component_means`.
    :param conditions: Exactly two condition names; the result is
        ``means[conditions[1]] - means[conditions[0]]``.
    :return: The difference array, shaped like one condition's mean.
    :raises ValueError: If two conditions are not given, one is missing, or their means
        have different shapes.
    """
    if len(conditions) != 2:
        raise ValueError(
            f"A difference needs exactly two conditions; got {list(conditions)}."
        )
    first, second = conditions
    for condition in (first, second):
        if condition not in means:
            raise ValueError(
                f"No mean for condition {condition!r}; have {sorted(means)}."
            )
    if means[first].shape != means[second].shape:
        raise ValueError(
            f"Means disagree in shape: {first} is {means[first].shape}, "
            f"{second} is {means[second].shape}."
        )
    return means[second] - means[first]


def stack_conditions_on_subject_axis(
    per_condition: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    participants: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Lay a time-axis join's per-condition split out on the subject axis.

    The bridge between the two ways of joining the conditions. A
    :class:`~src.analysis.condition_tracks.PairedConditionTracks` run puts each
    participant on the subject axis once and both conditions end to end along **time**,
    so splitting its results with
    :meth:`~src.analysis.condition_tracks.PairedConditionTracks.condition_track` gives
    one ``(P, K, ...)`` array *per condition*. The comparison figures in
    :mod:`src.visualization.iva_condition_plots` instead index a single ``(S, K, ...)``
    array by per-recording bookkeeping, the layout a subject-axis
    (:attr:`~src.definitions.fields.ConditionVariants.JOINED`) run produces natively.
    This restacks the former into the latter, so **one set of figures serves both
    variants**.

    Only the *sources* should come through here. A time-axis run estimates one mixing
    matrix per participant over the whole concatenated recording, so the channel
    patterns are shared by the conditions by construction — there is no per-condition
    pattern to stack, and pretending otherwise would draw two identical rows and a zero
    difference.

    The conditions keep their own time bases in a time-axis join and may differ in
    length. Stacking needs one common axis, so the last axis is trimmed to the shortest
    condition's length; a caller that needs the full length of each should read them
    per condition instead.

    :param per_condition: Condition name → ``(P, K, ..., T_c)`` array, e.g. the output
        of ``condition_track`` for each condition. Every entry must agree on all axes
        but the last.
    :param conditions: Conditions in the block order wanted on the subject axis.
    :param participants: Participant label per row of each per-condition array, in row
        order — e.g. :attr:`PairedConditionTracks.participants`.
    :return: ``(stacked, subject_participants, subject_conditions)`` — the
        ``(len(conditions) * P, K, ..., T_min)`` array plus the per-row participant and
        condition labels the figures index by.
    :raises ValueError: If a condition is missing, the non-time axes disagree, or a
        participant list does not match the row count.
    """
    conditions = list(conditions)
    participants = list(participants)
    if not conditions:
        raise ValueError("At least one condition is needed.")
    for condition in conditions:
        if condition not in per_condition:
            raise ValueError(f"No array supplied for condition {condition!r}.")

    reference = np.asarray(per_condition[conditions[0]], dtype=float)
    if reference.ndim < 2:
        raise ValueError(
            f"Arrays must be (P, K, ..., T); got shape {reference.shape} for "
            f"{conditions[0]!r}."
        )
    for condition in conditions:
        array = np.asarray(per_condition[condition], dtype=float)
        if array.shape[:-1] != reference.shape[:-1]:
            raise ValueError(
                f"Condition {condition!r} has non-time axes {array.shape[:-1]}, "
                f"expected {reference.shape[:-1]} to match {conditions[0]!r}."
            )
        if len(participants) != array.shape[0]:
            raise ValueError(
                f"Condition {condition!r} has {array.shape[0]} row(s) but "
                f"{len(participants)} participant label(s)."
            )

    n_times = min(int(np.asarray(per_condition[c]).shape[-1]) for c in conditions)
    blocks = [
        np.asarray(per_condition[c], dtype=float)[..., :n_times] for c in conditions
    ]
    stacked = np.concatenate(blocks, axis=0)
    subject_participants = [p for _ in conditions for p in participants]
    subject_conditions = [c for c in conditions for _ in participants]
    return stacked, subject_participants, subject_conditions


def slice_to_band(
    data_4d: np.ndarray, freqs: np.ndarray, band: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the pooled tensor's frequency axis to one band.

    :param data_4d: ``(S, C, F, T)`` pooled wavelet power.
    :param freqs: ``(F,)`` frequency axis of *data_4d*.
    :param band: Band name, or ``None`` to keep the whole grid.
    :return: ``(sliced, band_freqs)``.
    :raises ValueError: If the band lies outside the wavelet grid.
    """
    if band is None:
        return data_4d, freqs
    band_lo, band_hi = FREQUENCY_BANDS[band]
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    if not mask.any():
        raise ValueError(
            f"Band {band!r} ({band_lo}-{band_hi} Hz) has no frequency inside the "
            f"wavelet grid {freqs[0]:.1f}-{freqs[-1]:.1f} Hz."
        )
    return data_4d[:, :, mask, :], freqs[mask]


def decompose_channel_iva(
    data_4d: np.ndarray,
    *,
    label: str,
    n_pca: int,
    random_state: int,
    iva_opt_approach: str,
    iva_max_iter: int,
    iva_w_diff_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the channel-as-mixing IVA-G pipeline on a pooled tensor.

    Z-score along time, reshape each recording to ``(C, F·T)``, reduce the channel
    axis with a per-recording PCA, run ``iva_g``, resolve the ``Sigma_N`` sign
    ambiguity, then recover the spectro-temporal sources and the forward channel
    patterns. Identical to the single-condition variant; only the subject axis differs.

    :param data_4d: ``(S, C, F, T)`` pooled wavelet power.
    :param label: Dataset label, for logging.
    :param n_pca: Per-recording channel-PCA dimension (= number of components).
    :param random_state: Seed for the PCA and ``W_init``.
    :param iva_opt_approach: ``iva_g`` optimisation method.
    :param iva_max_iter: Maximum ``iva_g`` iterations.
    :param iva_w_diff_stop: ``iva_g`` convergence threshold.
    :return: ``(sources, patterns)`` shaped ``(S, K, F, T)`` and ``(S, K, C)``.
    :raises ValueError: If *n_pca* exceeds the channel count.
    """
    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    if n_pca > n_channels:
        raise ValueError(
            f"--n_pca ({n_pca}) must be ≤ n_channels ({n_channels}); PCA reduces "
            "the channel axis in this variant."
        )

    # Z-score along time, then flatten frequency-slow / time-fast so the sample axis
    # reshapes back to (F, T) exactly after IVA.
    n_samples_ft = n_freqs * n_times
    x_subjects = zscore_by_time(data_4d).reshape(n_subjects, n_channels, n_samples_ft)

    pcas: list[PCA] = []
    pca_scores = np.zeros((n_subjects, n_pca, n_samples_ft))
    for k in range(n_subjects):
        pca = PCA(n_components=n_pca, random_state=random_state)
        # sklearn wants (n_samples, n_features): (F*T, C).
        pca_scores[k] = pca.fit_transform(x_subjects[k].T).T
        pcas.append(pca)
        _logger.debug(
            f"[{label}] recording {k + 1}/{n_subjects}: channel PCA explains "
            f"{pca.explained_variance_ratio_.sum() * 100:.1f}%"
        )

    x_pca = np.ascontiguousarray(pca_scores.transpose(1, 2, 0))  # (N, T, K)
    _logger.info(f"[{label}] IVA input {x_pca.shape}  (N_PCA, F*T, K=datasets)")

    rng = np.random.default_rng(random_state)
    w_init = rng.standard_normal((n_pca, n_pca, n_subjects))
    demix, cost, sigma_n, _isi = iva_g(
        x_pca,
        opt_approach=iva_opt_approach,
        whiten=True,
        verbose=False,
        W_init=w_init,
        max_iter=iva_max_iter,
        W_diff_stop=iva_w_diff_stop,
    )
    _logger.info(
        f"[{label}] IVA-G: {len(cost)} iteration(s), final cost {cost[-1]:.6f}"
    )
    if len(cost) >= iva_max_iter:
        _logger.warning(
            f"[{label}] hit --iva_max_iter {iva_max_iter}; W may not have converged "
            f"(--iva_w_diff_stop {iva_w_diff_stop})."
        )

    _sigma_corr, demix, _flips = align_iva_component_signs(sigma_n, demix)

    scores = np.zeros((n_subjects, n_pca, n_samples_ft))
    patterns = np.zeros((n_subjects, n_pca, n_channels))
    for k in range(n_subjects):
        scores[k] = demix[:, :, k] @ x_pca[:, :, k]
        # Forward (mixing) patterns, NOT the unmixing rows: iva_g folds its whitening
        # into W, so the filter and the pattern of one component can be uncorrelated.
        patterns[k] = iva_component_patterns(demix[:, :, k], pcas[k].components_)

    sources = scores.reshape(n_subjects, n_pca, n_freqs, n_times)
    return sources, patterns
