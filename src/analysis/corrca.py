"""
Correlated Component Analysis (CorrCA) across subjects.

CorrCA (Dmochowski et al., 2012; Parra et al., 2018, arXiv:1801.08881) finds
spatial filters ``w`` that are **shared by every subject** and maximise the
inter-subject correlation (ISC) of the projected signals::

    ISC(w) = (wᵀ R_b w) / (wᵀ R_w w)

    R_w = Σ_k R_kk                  pooled within-subject covariance
    R_b = Σ_{k≠l} R_kl / (S - 1)    pooled between-subject cross-covariance

The subjects play the role of repetitions, so a single presentation per subject
is enough — only the stimulus has to be time-aligned across them. With this
normalisation the generalised eigenvalues *are* the ISCs, ``R_b`` is computed as
``(R_t - R_w) / (S - 1)`` with ``R_t`` the covariance of the subject sum, and the
problem is solved in the whitened subspace of ``R_w``:

1. optional shrinkage, ``R_w ← (1-γ) R_w + γ · mean(eig R_w) · I``;
2. truncation to the leading ``n_pca`` eigenvectors of ``R_w`` (or to its
   numerical rank). After average referencing and ICA cleaning the channel
   covariance is rank-deficient, so this step is not optional in practice;
3. ``eigh`` of the whitened ``R_b``, components sorted by descending ISC.

Unlike IVA, the filter is one for all subjects, so a component's polarity is
the same in every subject by construction. Only one global sign per component
is free; it is fixed so the shared forward pattern's largest-magnitude entry is
positive.

Topographies are **forward** patterns (Haufe et al., 2014),
``A = R W (Wᵀ R W)⁻¹``, never the filters — see
:func:`src.analysis.wavelet_ica.iva_component_patterns` for why the two differ.

All functions are pure (no I/O, no plotting) and never mutate their inputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.wavelet_ica import zscore_by_time

_logger = logging.getLogger(__name__)

#: Eigenvalues of ``R_w`` below this fraction of the largest are treated as zero
#: when no explicit ``n_pca`` is given.
RANK_RTOL: float = 1e-8

#: Below this :func:`anchor_component_signs` strength the shared pattern's mean over the
#: anchor sits within a tenth of its RMS of zero, so the sign is close to a coin toss.
ANCHOR_STRENGTH_FLOOR: float = 0.1


@dataclass(frozen=True)
class CorrcaCovariances:
    """Sufficient statistics of a CorrCA fit.

    One pass over the data yields everything the fit, the forward patterns and
    the ISC evaluation need, so the (large) sample axis is read once.

    :param subject: ``(S, D, D)`` per-subject covariance ``R_kk`` (sums of
        products of the centred data, not divided by the sample count).
    :param total: ``(D, D)`` covariance of the subject sum, ``R_t``.
    :param n_samples: Length of the sample axis.
    """

    subject: np.ndarray
    total: np.ndarray
    n_samples: int

    @property
    def n_subjects(self) -> int:
        return self.subject.shape[0]

    @property
    def within(self) -> np.ndarray:
        return self.subject.sum(axis=0)

    @property
    def between(self) -> np.ndarray:
        return (self.total - self.within) / (self.n_subjects - 1)


@dataclass(frozen=True)
class CorrcaResult:
    """A fitted CorrCA model.

    :param filters: ``(D, K)`` shared spatial filters, one column per component.
    :param patterns: ``(K, D)`` shared forward patterns from the pooled ``R_w``,
        indexed ``[component, feature]`` like the IVA patterns.
    :param isc: ``(K,)`` training-set ISC per component, descending. Inflated by
        construction — evaluate held-out data with :func:`component_isc`.
    :param n_retained: Dimension of the ``R_w`` subspace the problem was solved in.
    :param gamma: Shrinkage applied to ``R_w``.
    """

    filters: np.ndarray
    patterns: np.ndarray
    isc: np.ndarray
    n_retained: int
    gamma: float

    @property
    def n_components(self) -> int:
        return self.filters.shape[1]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project ``(S, D, N)`` data onto the components → ``(S, K, N)``."""
        _validate_3d(x)
        if x.shape[1] != self.filters.shape[0]:
            raise ValueError(
                f"x has {x.shape[1]} features but the filters expect "
                f"{self.filters.shape[0]}."
            )
        return np.einsum("dk,sdn->skn", self.filters, x)

    def oriented(self, signs: np.ndarray) -> CorrcaResult:
        """Copy with component *k*'s filter and pattern multiplied by ``signs[k]``."""
        signs = _validate_signs(signs, self.n_components)
        return replace(
            self, filters=self.filters * signs, patterns=self.patterns * signs[:, None]
        )


@dataclass(frozen=True)
class CorrcaDecomposition:
    """CorrCA of wavelet power, laid out like the channel-IVA outputs.

    :param model: The fitted :class:`CorrcaResult` (channels as features).
    :param sources: ``(S, K, F, T)`` per-subject component time-frequency maps.
    :param patterns: ``(S, K, C)`` per-subject forward channel patterns.
    """

    model: CorrcaResult
    sources: np.ndarray
    patterns: np.ndarray

    def oriented(self, signs: np.ndarray) -> CorrcaDecomposition:
        """Copy with every quantity of component *k* multiplied by ``signs[k]``.

        One sign per component, for every subject alike: the filter is shared, so a
        per-subject flip would invent a polarity difference the model does not have.
        """
        signs = _validate_signs(signs, self.model.n_components)
        per_subject = signs.reshape(1, -1, *([1] * (self.sources.ndim - 2)))
        return CorrcaDecomposition(
            model=self.model.oriented(signs),
            sources=self.sources * per_subject,
            patterns=self.patterns * signs[None, :, None],
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_3d(x: np.ndarray) -> None:
    if x.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array (n_subjects, n_features, n_samples), got shape "
            f"{x.shape} (ndim={x.ndim})."
        )


def _validate_signs(signs: np.ndarray, n_components: int) -> np.ndarray:
    signs = np.asarray(signs, dtype=float)
    if signs.shape != (n_components,) or not np.all(np.abs(signs) == 1.0):
        raise ValueError(
            f"signs must be ({n_components},) of +-1; got shape {signs.shape}."
        )
    return signs


# ---------------------------------------------------------------------------
# Covariances
# ---------------------------------------------------------------------------


def corrca_covariances(x: np.ndarray) -> CorrcaCovariances:
    """Accumulate the per-subject and total covariances of ``(S, D, N)`` data.

    Each subject's features are centred over the sample axis. Subjects are read
    one at a time and promoted to float64 individually, so a memory-mapped
    float32 stack is never copied whole.

    :param x: ``(S, D, N)`` time-aligned data, **not** mutated.
    :return: The :class:`CorrcaCovariances` of *x*.
    :raises ValueError: If *x* is not 3-D or has fewer than two subjects.
    """
    _validate_3d(x)
    n_subjects, n_features, n_samples = x.shape
    if n_subjects < 2:
        raise ValueError(f"CorrCA needs at least two subjects; got {n_subjects}.")

    subject = np.zeros((n_subjects, n_features, n_features))
    subject_sum = np.zeros((n_features, n_samples))
    for k in range(n_subjects):
        xk = np.asarray(x[k], dtype=np.float64)
        xk = xk - xk.mean(axis=1, keepdims=True)
        subject[k] = xk @ xk.T
        subject_sum += xk
    return CorrcaCovariances(
        subject=subject, total=subject_sum @ subject_sum.T, n_samples=n_samples
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def forward_patterns(cov: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Forward patterns ``A = R W (Wᵀ R W)⁻¹`` of *filters* under covariance *cov*.

    :param cov: ``(D, D)`` data covariance — pooled for the shared patterns, one
        subject's for that subject's patterns.
    :param filters: ``(D, K)`` spatial filters.
    :return: ``(K, D)`` patterns, one row per component.
    """
    cw = cov @ filters
    # pinv: a subject may carry (almost) no variance along some filter.
    return (cw @ np.linalg.pinv(filters.T @ cw)).T


def fit_corrca_from_covariances(
    cov: CorrcaCovariances,
    *,
    n_pca: int | None = None,
    gamma: float = 0.0,
) -> CorrcaResult:
    """Fit CorrCA from precomputed covariances.

    :param cov: Output of :func:`corrca_covariances`.
    :param n_pca: Leading ``R_w`` eigen-directions to keep, which is also the
        number of components returned. ``None`` keeps the numerical rank
        (eigenvalues above :data:`RANK_RTOL` of the largest).
    :param gamma: Shrinkage of ``R_w`` toward a scaled identity, in ``[0, 1]``.
        With ``gamma > 0`` the full rank is kept unless *n_pca* says otherwise.
    :return: The fitted :class:`CorrcaResult`, components sorted by descending ISC.
    :raises ValueError: If *gamma* or *n_pca* is out of range.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1]; got {gamma}.")
    within = cov.within
    n_features = within.shape[0]
    if n_pca is not None and not 1 <= n_pca <= n_features:
        raise ValueError(f"n_pca must lie in [1, {n_features}]; got {n_pca}.")

    shrunk = within
    if gamma > 0.0:
        scale = np.trace(within) / n_features
        shrunk = (1.0 - gamma) * within + gamma * scale * np.eye(n_features)

    evals, evecs = np.linalg.eigh(shrunk)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    rank = int((evals > RANK_RTOL * evals[0]).sum())
    n_keep = rank if n_pca is None else n_pca
    if n_keep > rank:
        _logger.warning(
            f"n_pca={n_pca} exceeds the numerical rank {rank} of R_w; the trailing "
            "directions carry no variance and their ISC is meaningless. Lower "
            "n_pca or set gamma > 0."
        )
        n_keep = min(n_keep, int((evals > 0).sum()))

    # Whitening basis of the retained subspace: Mᵀ R_w M = I.
    whitener = evecs[:, :n_keep] / np.sqrt(evals[:n_keep])
    between_white = whitener.T @ cov.between @ whitener
    isc, rotation = np.linalg.eigh((between_white + between_white.T) / 2.0)
    order = np.argsort(isc)[::-1]
    isc, filters = isc[order], whitener @ rotation[:, order]

    patterns = forward_patterns(within, filters)
    # Fix the one free sign per component: largest-magnitude pattern entry > 0.
    peak = patterns[np.arange(n_keep), np.abs(patterns).argmax(axis=1)]
    signs = np.where(peak < 0.0, -1.0, 1.0)
    return CorrcaResult(
        filters=filters * signs,
        patterns=patterns * signs[:, None],
        isc=isc,
        n_retained=n_keep,
        gamma=gamma,
    )


def fit_corrca(
    x: np.ndarray,
    *,
    n_pca: int | None = None,
    gamma: float = 0.0,
) -> CorrcaResult:
    """Fit CorrCA to ``(S, D, N)`` time-aligned data.

    Convenience wrapper over :func:`corrca_covariances` and
    :func:`fit_corrca_from_covariances`; see the latter for *n_pca* and *gamma*.
    """
    return fit_corrca_from_covariances(corrca_covariances(x), n_pca=n_pca, gamma=gamma)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def isc_from_covariances(cov: CorrcaCovariances, filters: np.ndarray) -> np.ndarray:
    """ISC of fixed *filters* on the data *cov* was computed from → ``(K,)``.

    The same ratio the fit maximises, so on the training data it reproduces
    :attr:`CorrcaResult.isc`; on held-out subjects or another condition it is the
    unbiased read-out.
    """
    num = np.einsum("dk,de,ek->k", filters, cov.between, filters)
    den = np.einsum("dk,de,ek->k", filters, cov.within, filters)
    return num / np.where(den == 0.0, 1.0, den)


def component_isc(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """ISC of fixed ``(D, K)`` *filters* on ``(S, D, N)`` data → ``(K,)``."""
    return isc_from_covariances(corrca_covariances(x), filters)


def _as_projected(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim < 3:
        raise ValueError(
            f"Expected projected components (S, K, ...samples); got shape {y.shape}."
        )
    if y.shape[0] < 2:
        raise ValueError(f"ISC needs at least two subjects; got {y.shape[0]}.")
    return y.reshape(y.shape[0], y.shape[1], -1)


def _centred(ys: np.ndarray) -> np.ndarray:
    ys = np.asarray(ys, dtype=np.float64)
    return ys - ys.mean(axis=1, keepdims=True)


def projected_isc(y: np.ndarray) -> np.ndarray:
    """Group ISC of already-projected components, ``(S, K, ...)`` → ``(K,)``.

    The CorrCA ratio evaluated on the component signals themselves, every trailing
    axis counting as samples. It equals :func:`component_isc` on the data the
    components were projected from, so a subset of subjects — one condition of a
    pooled run — can be scored from the sources alone. Subjects are read one at a
    time, so no centred copy of the whole stack is made.
    """
    y = _as_projected(y)
    n_subjects = y.shape[0]
    within = np.zeros(y.shape[1])
    total = np.zeros(y.shape[1:])
    for s in range(n_subjects):
        ys = _centred(y[s])
        within += np.einsum("kn,kn->k", ys, ys)
        total += ys
    between = (np.einsum("kn,kn->k", total, total) - within) / (n_subjects - 1)
    return between / np.where(within == 0.0, 1.0, within)


def leave_one_out_isc(y: np.ndarray) -> np.ndarray:
    """Per-subject ISC: each subject against the mean of the others → ``(S, K)``.

    Pearson correlation of subject *s*'s component signal with the average of every
    other subject's, the per-subject read-out a paired condition test needs. The
    filters are not refitted, so a subject that took part in the fit still helped
    choose them; compare conditions scored the same way, not against a held-out ISC.

    :param y: ``(S, K, ...)`` projected components, trailing axes = samples.
    :return: ``(S, K)`` correlations.
    """
    y = _as_projected(y)
    n_subjects = y.shape[0]
    total = np.zeros(y.shape[1:])
    for s in range(n_subjects):
        total += _centred(y[s])
    loo = np.zeros(y.shape[:2])
    for s in range(n_subjects):
        ys = _centred(y[s])
        others = total - ys
        num = np.einsum("kn,kn->k", ys, others)
        den = np.sqrt(
            np.einsum("kn,kn->k", ys, ys) * np.einsum("kn,kn->k", others, others)
        )
        loo[s] = num / np.where(den == 0.0, 1.0, den)
    return loo


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------


def anchor_component_signs(
    decomposition: CorrcaDecomposition, electrode_mask: np.ndarray
) -> tuple[CorrcaDecomposition, np.ndarray, np.ndarray]:
    """Orient each component so its **shared** pattern is positive over the anchor.

    Flips on the sign of the shared pattern's mean over the anchor electrodes, so the
    topomap reads positive there and more power over the ASSR area is a positive
    source. Read on the one shared pattern and applied to every recording alike: the
    filter is shared, and a recording-level flip would invert a component the model
    fitted as one.

    This is deliberately not the IVA anchor
    (:func:`~src.analysis.assr_trials.polarity_flip`), which asks whether the area is
    more positive *than the rest of the head* and so leaves a component on a global
    negative offset negative over the anchor. That relative criterion protects IVA's
    per-recording flips from each recording's own offset; with one shared sign there is
    no such risk to protect against.

    :param decomposition: Output of :func:`decompose_channel_corrca`.
    :param electrode_mask: Boolean mask over the channel axis.
    :return: ``(oriented, flip, strength)`` — the oriented decomposition, the ``(K,)``
        signs applied and ``|mean over the mask| / RMS over all channels`` of the shared
        pattern, to be read against :data:`ANCHOR_STRENGTH_FLOOR`.
    :raises ValueError: If the mask does not match the channel axis or selects no
        channel.
    """
    patterns = decomposition.model.patterns
    mask = np.asarray(electrode_mask, dtype=bool)
    if mask.shape != (patterns.shape[1],) or not mask.any():
        raise ValueError(
            f"electrode_mask must be ({patterns.shape[1]},) and select at least one "
            f"channel; got shape {mask.shape} selecting {int(mask.sum())}."
        )
    inside = patterns[:, mask].mean(axis=1)
    rms = np.sqrt((patterns**2).mean(axis=1))
    strength = np.divide(np.abs(inside), rms, out=np.zeros_like(inside), where=rms > 0)
    # A zero mean keeps +1 rather than inventing a sign; the zero strength marks it.
    flip = np.where(inside < 0.0, -1.0, 1.0)
    return decomposition.oriented(flip), flip, strength


def condition_isc(
    sources_by_condition: Mapping[str, np.ndarray],
    participants_by_condition: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synchrony of every component within each condition, from fixed filters.

    The inter-condition read-out CorrCA adds to the IVA comparison. Each condition's
    sources come from the **same** filters, so a difference in their ISC is a
    difference in how synchronised the one component is, not a difference between
    two decompositions.

    :param sources_by_condition: Condition → ``(P_c, K, ...)`` projected sources of
        that condition's recordings (or time-axis segments).
    :param participants_by_condition: Condition → participant label per row.
    :return: ``(group, per_participant)`` long-form frames. ``group`` has columns
        ``component`` (1-based, matching the figure labels), ``condition``, ``isc``,
        ``n_subjects``; ``per_participant`` has ``component``, ``condition``,
        ``participant``, ``loo_isc``.
    :raises ValueError: If a condition's labels do not match its row count, or the
        conditions disagree on the component count.
    """
    n_components = {np.shape(s)[1] for s in sources_by_condition.values()}
    if len(n_components) != 1:
        raise ValueError(f"Conditions disagree on the component count: {n_components}.")

    group_rows: list[dict] = []
    subject_rows: list[dict] = []
    for condition, sources in sources_by_condition.items():
        participants = list(participants_by_condition[condition])
        if len(participants) != np.shape(sources)[0]:
            raise ValueError(
                f"{condition!r} has {np.shape(sources)[0]} row(s) but "
                f"{len(participants)} participant label(s)."
            )
        isc = projected_isc(sources)
        loo = leave_one_out_isc(sources)
        for k, value in enumerate(isc):
            group_rows.append(
                {
                    "component": k + 1,
                    "condition": condition,
                    "isc": float(value),
                    "n_subjects": len(participants),
                }
            )
            for participant, r in zip(participants, loo[:, k]):
                subject_rows.append(
                    {
                        "component": k + 1,
                        "condition": condition,
                        "participant": participant,
                        "loo_isc": float(r),
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(subject_rows)


def paired_condition_test(
    per_participant: pd.DataFrame, conditions: Sequence[str]
) -> pd.DataFrame:
    """Wilcoxon signed-rank test of per-participant ISC between two conditions.

    Pairs are formed by participant label, so only participants present in both
    conditions count. P-values are per component and **uncorrected**.

    :param per_participant: The second frame of :func:`condition_isc`.
    :param conditions: Exactly two conditions; differences are second − first.
    :return: One row per component: ``component``, ``n_pairs``,
        ``mean_difference``, ``median_difference``, ``statistic``, ``p_value``
        (``NaN`` when fewer than two non-zero differences leave nothing to test).
    :raises ValueError: If two conditions are not given.
    """
    if len(conditions) != 2:
        raise ValueError(f"A paired test needs two conditions; got {conditions}.")
    first, second = conditions
    wide = per_participant.pivot_table(
        index=["component", "participant"], columns="condition", values="loo_isc"
    )
    rows: list[dict] = []
    for component, block in wide.groupby(level="component"):
        if first not in block or second not in block:
            diff = np.array([])
        else:
            diff = (block[second] - block[first]).dropna().to_numpy()
        statistic, p_value = np.nan, np.nan
        if np.count_nonzero(diff) >= 2:
            result = wilcoxon(diff)
            statistic, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "component": int(component),
                "n_pairs": int(diff.size),
                "mean_difference": float(diff.mean()) if diff.size else np.nan,
                "median_difference": float(np.median(diff)) if diff.size else np.nan,
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wavelet-power decomposition
# ---------------------------------------------------------------------------


def decompose_channel_corrca(
    data_4d: np.ndarray,
    *,
    label: str,
    n_pca: int | None,
    gamma: float = 0.0,
) -> CorrcaDecomposition:
    """CorrCA of wavelet power with channels as features — the IVA-channel analogue.

    Z-scores along time (as the IVA pipeline does), flattens each subject to
    ``(C, F·T)`` so a component is a shared channel filter whose output is a
    spectro-temporal map, fits CorrCA, and returns the sources and per-subject
    forward patterns in the shapes :func:`~src.analysis.iva_condition_comparison
    .decompose_channel_iva` returns, so the IVA figures and store take them as is.

    The filter is shared, but the per-subject pattern ``A_s = R_ss W (Wᵀ R_ss W)⁻¹``
    is not: it is how the common component appears on that subject's scalp.

    :param data_4d: ``(S, C, F, T)`` wavelet power, time-aligned across subjects.
    :param label: Dataset label, for logging.
    :param n_pca: ``R_w`` subspace dimension (= number of components); ``None``
        keeps the numerical rank.
    :param gamma: ``R_w`` shrinkage.
    :return: The :class:`CorrcaDecomposition`.
    :raises ValueError: If *data_4d* is not 4-D or *n_pca* exceeds the channel count.
    """
    z = zscore_by_time(data_4d)
    n_subjects, n_channels, n_freqs, n_times = z.shape
    x = z.reshape(n_subjects, n_channels, n_freqs * n_times)

    cov = corrca_covariances(x)
    model = fit_corrca_from_covariances(cov, n_pca=n_pca, gamma=gamma)
    _logger.info(
        f"[{label}] CorrCA: {model.n_components} component(s) in a "
        f"{model.n_retained}-dim R_w subspace (gamma={gamma}); top ISC "
        f"{np.round(model.isc[:5], 3).tolist()}"
    )

    sources = model.transform(x).reshape(
        n_subjects, model.n_components, n_freqs, n_times
    )
    patterns = np.stack([forward_patterns(r, model.filters) for r in cov.subject])
    return CorrcaDecomposition(model=model, sources=sources, patterns=patterns)
