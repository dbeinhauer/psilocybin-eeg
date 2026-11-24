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

import itertools
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
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


def _centred_subject(xk: np.ndarray) -> np.ndarray:
    xk = np.asarray(xk, dtype=np.float64)
    return xk - xk.mean(axis=1, keepdims=True)


def corrca_covariances(
    x: np.ndarray, *, rows: Sequence[int] | None = None
) -> CorrcaCovariances:
    """Accumulate the per-subject and total covariances of ``(S, D, N)`` data.

    Each subject's features are centred over the sample axis. Subjects are read
    one at a time and promoted to float64 individually, so a memory-mapped
    float32 stack is never copied whole.

    :param x: ``(S, D, N)`` time-aligned data, **not** mutated.
    :param rows: Subjects to accumulate, in this order; ``None`` takes every one.
        Read by index, so a subset of *x* is never copied.
    :return: The :class:`CorrcaCovariances` of *x* (of its *rows*).
    :raises ValueError: If *x* is not 3-D, fewer than two subjects are taken, or
        *rows* repeats or leaves the subject axis.
    """
    _validate_3d(x)
    n_total, n_features, n_samples = x.shape
    rows = list(range(n_total)) if rows is None else [int(r) for r in rows]
    if len(set(rows)) != len(rows) or not all(0 <= r < n_total for r in rows):
        raise ValueError(
            f"rows must be distinct subjects in [0, {n_total}); got {rows}."
        )
    n_subjects = len(rows)
    if n_subjects < 2:
        raise ValueError(f"CorrCA needs at least two subjects; got {n_subjects}.")

    subject = np.zeros((n_subjects, n_features, n_features))
    subject_sum = np.zeros((n_features, n_samples))
    for i, k in enumerate(rows):
        xk = _centred_subject(x[k])
        subject[i] = xk @ xk.T
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


def matched_component_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r over channels between component *k* of *a* and component *k* of *b*.

    Components are matched by index only, never re-paired. The sign is kept, so a
    component the two fits oriented oppositely reads negative.

    :param a: ``(K, C)`` patterns (or filters transposed) of one fit.
    :param b: ``(K', C)`` the same of another fit on the same channels.
    :return: ``(min(K, K'),)`` correlations; NaN for a constant component.
    :raises ValueError: If the channel axes differ.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"a and b must be (K, C) on the same channels; got {a.shape}, {b.shape}."
        )
    k = min(a.shape[0], b.shape[0])
    a = a[:k] - a[:k].mean(axis=1, keepdims=True)
    b = b[:k] - b[:k].mean(axis=1, keepdims=True)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (a * b).sum(axis=1) / norm, np.nan)


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
    fit_rows: Sequence[int] | None = None,
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
    :param fit_rows: Subjects the filters are fitted on; ``None`` fits on every one.
        The sources and patterns are still returned for every subject, the others
        read through filters they did not shape. The model's ``isc`` and shared
        ``patterns`` are those of the *fit_rows* alone.
    :return: The :class:`CorrcaDecomposition`.
    :raises ValueError: If *data_4d* is not 4-D, *n_pca* exceeds the channel count
        or *fit_rows* is invalid (see :func:`corrca_covariances`).
    """
    z = zscore_by_time(data_4d)
    n_subjects, n_channels, n_freqs, n_times = z.shape
    x = z.reshape(n_subjects, n_channels, n_freqs * n_times)

    rows = list(range(n_subjects)) if fit_rows is None else [int(r) for r in fit_rows]
    cov = corrca_covariances(x, rows=rows)
    model = fit_corrca_from_covariances(cov, n_pca=n_pca, gamma=gamma)
    _logger.info(
        f"[{label}] CorrCA: {model.n_components} component(s) in a "
        f"{model.n_retained}-dim R_w subspace (gamma={gamma}), fitted on "
        f"{len(rows)}/{n_subjects} subject(s); top ISC "
        f"{np.round(model.isc[:5], 3).tolist()}"
    )

    sources = model.transform(x).reshape(
        n_subjects, model.n_components, n_freqs, n_times
    )
    fitted = dict(zip(rows, cov.subject))
    patterns = np.empty((n_subjects, model.n_components, n_channels))
    for s in range(n_subjects):
        if s in fitted:
            subject_cov = fitted[s]
        else:
            xs = _centred_subject(x[s])
            subject_cov = xs @ xs.T
        patterns[s] = forward_patterns(subject_cov, model.filters)
    return CorrcaDecomposition(model=model, sources=sources, patterns=patterns)


# ---------------------------------------------------------------------------
# Leave-one-participant-out influence
# ---------------------------------------------------------------------------

#: Robust z of a participant's influence above which it is flagged as an outlier
#: (the modified z-score cut of Iglewicz & Hoaglin, 1993).
INFLUENCE_Z_THRESHOLD: float = 3.5

#: Share of a component's ISC a participant's removal must cost before a high robust
#: z flags them. Where every loss is near zero the MAD is tiny and the z alone would
#: flag a negligible difference.
MIN_RELATIVE_INFLUENCE: float = 0.1

#: Float64 values of one sample chunk of the jackknife pass, bounding its memory.
_JACKKNIFE_CHUNK_VALUES: int = 2**25


@dataclass(frozen=True)
class JackknifeStatistics:
    """Sufficient statistics of every leave-one-group-out CorrCA refit.

    With ``Σ = Σ_k x_k`` the subject sum and ``y_g = Σ_{r∈g} x_r`` the sum of group
    *g*, dropping *g* leaves ``(Σ - y_g)(Σ - y_g)ᵀ = R_t - (G + Gᵀ) + H`` with
    ``G = Σ_{r∈g} x_r Σᵀ`` and ``H = Σ_{r∈g} x_r y_gᵀ``. Both come from the two
    per-row terms kept here, so one pass over the data serves every refit.

    Dropping several groups at once needs ``H = y_A y_Aᵀ`` of their joint sum ``y_A``,
    i.e. the cross terms between the dropped groups: ``group_cross`` keeps
    ``y_g y_hᵀ`` for every pair of groups.

    :param covariances: The full-data :class:`CorrcaCovariances`.
    :param groups: Row indices of each group, e.g. one participant's recordings.
    :param cross_total: ``(S, D, D)`` ``x_r Σᵀ`` per row.
    :param cross_group: ``(S, D, D)`` ``x_r y_gᵀ`` per row, *g* the row's group.
    :param group_cross: ``(G, G, D, D)`` ``y_g y_hᵀ`` per pair of groups, or ``None``
        when only single groups are ever dropped.
    """

    covariances: CorrcaCovariances
    groups: tuple[tuple[int, ...], ...]
    cross_total: np.ndarray
    cross_group: np.ndarray
    group_cross: np.ndarray | None = None

    def without(self, g: int | Sequence[int]) -> CorrcaCovariances:
        """The covariances of the data with group *g* — or every group in *g* —
        removed.

        :raises ValueError: If several groups are dropped without ``group_cross``,
            or fewer than two groups would remain.
        """
        dropped = (int(g),) if np.ndim(g) == 0 else tuple(int(x) for x in g)
        if len(self.groups) - len(set(dropped)) < 2:
            raise ValueError(
                f"Dropping {len(set(dropped))} of {len(self.groups)} groups leaves "
                "fewer than two to fit on."
            )
        rows = [r for x in dropped for r in self.groups[x]]
        if len(dropped) == 1:
            inner = self.cross_group[rows].sum(axis=0)
        elif self.group_cross is None:
            raise ValueError(
                "Dropping several groups needs group_cross; accumulate the statistics "
                "with jackknife_statistics(..., group_cross=True)."
            )
        else:
            inner = self.group_cross[np.ix_(dropped, dropped)].sum(axis=(0, 1))
        cov = self.covariances
        drop_sum = self.cross_total[rows].sum(axis=0)
        return CorrcaCovariances(
            subject=np.delete(cov.subject, rows, axis=0),
            total=cov.total - drop_sum - drop_sum.T + inner,
            n_samples=cov.n_samples,
        )

    def to_reference(self, row: int) -> np.ndarray:
        """``x_r (Σ - y_g)ᵀ``: row *r* against the sum of every other group."""
        return self.cross_total[row] - self.cross_group[row]


def participant_groups(participants: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Row indices per participant label, in order of first appearance."""
    groups: dict[str, list[int]] = {}
    for row, participant in enumerate(participants):
        groups.setdefault(participant, []).append(row)
    return tuple(tuple(rows) for rows in groups.values())


def _validate_groups(groups: Sequence[Sequence[int]], n_subjects: int) -> None:
    rows = sorted(r for g in groups for r in g)
    if rows != list(range(n_subjects)):
        raise ValueError(
            f"groups must partition the {n_subjects} rows exactly once; got {groups}."
        )
    if len(groups) < 3:
        raise ValueError(
            "Leaving a group out must leave at least two groups to fit on; got "
            f"{len(groups)} group(s)."
        )


def jackknife_statistics(
    chunks: Iterable[np.ndarray],
    groups: Sequence[Sequence[int]],
    *,
    group_cross: bool = False,
) -> JackknifeStatistics:
    """Accumulate :class:`JackknifeStatistics` over sample chunks.

    Every term is a sum over samples of an outer product, so the sample axis can be
    read in any number of ``(S, D, n)`` chunks, in any order. Raw products are
    accumulated with the per-row sums and centred at the end
    (``Σ x yᵀ - N μ_x μ_yᵀ``), which matches :func:`corrca_covariances`, centring
    each row's features over all samples.

    :param chunks: ``(S, D, n_i)`` pieces of the sample axis, in any order.
    :param groups: Row indices per group; must partition the rows, at least three.
    :param group_cross: Also keep the cross terms of every pair of groups, so several
        groups can be dropped at once. One symmetric ``(G·D, G·D)`` product per
        chunk, about ``G² / (6 S)`` times the work of the other terms together.
    :return: The statistics of the concatenated data.
    :raises ValueError: If the chunks disagree in shape or the groups are invalid.
    """
    groups = tuple(tuple(int(r) for r in g) for g in groups)
    group_of: np.ndarray | None = None
    raw_subject = raw_cross_total = raw_cross_group = row_sum = None
    raw_group_cross = None
    n_samples = 0
    for chunk in chunks:
        xc = np.asarray(chunk, dtype=np.float64)
        _validate_3d(xc)
        if group_of is None:
            n_subjects, n_features = xc.shape[:2]
            _validate_groups(groups, n_subjects)
            group_of = np.empty(n_subjects, dtype=int)
            for g, rows in enumerate(groups):
                group_of[list(rows)] = g
            raw_subject = np.zeros((n_subjects, n_features, n_features))
            raw_cross_total = np.zeros_like(raw_subject)
            raw_cross_group = np.zeros_like(raw_subject)
            row_sum = np.zeros((n_subjects, n_features))
            if group_cross:
                size = len(groups) * n_features
                raw_group_cross = np.zeros((size, size))
        elif xc.shape[:2] != raw_subject.shape[:2]:
            raise ValueError(
                f"Chunk shape {xc.shape} disagrees with the first chunk's "
                f"{raw_subject.shape[:2]} rows × features."
            )
        total_chunk = xc.sum(axis=0)
        group_chunk = np.stack([xc[list(rows)].sum(axis=0) for rows in groups])
        raw_subject += xc @ xc.transpose(0, 2, 1)
        raw_cross_total += xc @ total_chunk.T
        raw_cross_group += xc @ group_chunk[group_of].transpose(0, 2, 1)
        if raw_group_cross is not None:
            flat = group_chunk.reshape(-1, xc.shape[2])
            # np.dot of a matrix with its own transpose runs as one symmetric BLAS
            # product (syrk), half the work of a general one.
            raw_group_cross += np.dot(flat, flat.T)
        row_sum += xc.sum(axis=2)
        n_samples += xc.shape[2]
    if group_of is None:
        raise ValueError("jackknife_statistics got no data chunks.")

    mean = row_sum / n_samples
    total_mean = mean.sum(axis=0)
    per_group_mean = np.stack([mean[list(rows)].sum(axis=0) for rows in groups])
    group_mean = per_group_mean[group_of]
    subject = raw_subject - n_samples * np.einsum("sd,se->sde", mean, mean)
    cross_total = raw_cross_total - n_samples * np.einsum("sd,e->sde", mean, total_mean)
    cross_group = raw_cross_group - n_samples * np.einsum(
        "sd,se->sde", mean, group_mean
    )
    pairs = None
    if raw_group_cross is not None:
        n_groups = len(groups)
        # (G·D, G·D) -> (G, G, D, D), indexed [g, h, d, e].
        pairs = raw_group_cross.reshape(
            n_groups, n_features, n_groups, n_features
        ).transpose(0, 2, 1, 3) - n_samples * np.einsum(
            "gd,he->ghde", per_group_mean, per_group_mean
        )
    total = cross_total.sum(axis=0)
    return JackknifeStatistics(
        covariances=CorrcaCovariances(
            subject=subject, total=(total + total.T) / 2.0, n_samples=n_samples
        ),
        groups=groups,
        cross_total=cross_total,
        cross_group=cross_group,
        group_cross=pairs,
    )


def _quadratic(filters: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("dk,de,ek->k", filters, matrix, filters)


def _reference_isc(
    stats: JackknifeStatistics, g: int, filters: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """``(len(group), K)`` correlation of each row of group *g* with the sum of the
    other groups, every signal projected through *filters*."""
    other = _quadratic(filters, reference)
    out = np.empty((len(stats.groups[g]), filters.shape[1]))
    for i, row in enumerate(stats.groups[g]):
        num = _quadratic(filters, stats.to_reference(row))
        own = _quadratic(filters, stats.covariances.subject[row])
        den = np.sqrt(np.clip(own * other, 0.0, None))
        out[i] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def _own_group_isc(
    stats: JackknifeStatistics, g: int, filters: np.ndarray
) -> np.ndarray:
    """``(len(group), K)`` correlation of each row of group *g* with the sum of the
    group's **other** rows (NaN for a one-row group)."""
    members = stats.groups[g]
    out = np.full((len(members), filters.shape[1]), np.nan)
    if len(members) < 2:
        return out
    group_auto = stats.cross_group[list(members)].sum(axis=0)
    for i, row in enumerate(members):
        own_cov = stats.covariances.subject[row]
        partner = stats.cross_group[row] - own_cov
        num = _quadratic(filters, partner)
        rest = _quadratic(filters, group_auto - partner - partner.T - own_cov)
        den = np.sqrt(np.clip(_quadratic(filters, own_cov) * rest, 0.0, None))
        out[i] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def _abs_pattern_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(K_a, K_b)`` |Pearson r| over features between every pair of patterns."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    norm = np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.where(norm > 0, (a @ b.T) / norm, 0.0))


def _matched_refit(
    stats: JackknifeStatistics,
    g: int | Sequence[int],
    model: CorrcaResult,
    *,
    n_pca: int | None,
    gamma: float,
) -> tuple[CorrcaCovariances, CorrcaResult, np.ndarray, np.ndarray]:
    """Refit without group *g* (or every group in *g*); ``matched[k]`` is the refit
    component paired with *model*'s component *k* (Hungarian on |r| of the shared
    patterns), ``-1`` if none.

    :return: ``(reduced covariances, refit, matched, |r| similarity matrix)``.
    """
    from scipy.optimize import linear_sum_assignment

    reduced = stats.without(g)
    refit = fit_corrca_from_covariances(reduced, n_pca=n_pca, gamma=gamma)
    similarity = _abs_pattern_correlation(model.patterns, refit.patterns)
    full_idx, refit_idx = linear_sum_assignment(similarity, maximize=True)
    matched = np.full(model.n_components, -1)
    matched[full_idx] = refit_idx
    return reduced, refit, matched, similarity


def _matched_filters(
    model: CorrcaResult, refit: CorrcaResult, matched: np.ndarray
) -> np.ndarray:
    """The refit's filters in *model*'s component order; zero where unmatched."""
    filters = np.zeros_like(model.filters)
    has_match = matched >= 0
    filters[:, has_match] = refit.filters[:, matched[has_match]]
    return filters


def leave_group_out_influence(
    stats: JackknifeStatistics,
    model: CorrcaResult,
    *,
    n_pca: int | None,
    gamma: float,
) -> pd.DataFrame:
    """Refit CorrCA without each group and read how much the fit depended on it.

    For every group *g* and component *k* of *model*:

    - ``isc_without`` / ``pattern_r`` — the refit's component best matching *k*
      (Hungarian assignment on |r| of the shared forward patterns), its ISC on the
      remaining data and its |r| with *k*'s pattern. A component that exists only
      because of *g* loses its ISC and its topography when *g* is left out.
    - ``insample_isc`` — each row of *g* projected through *model*'s filter,
      correlated with the sum of every **other** group's projections.
    - ``heldout_isc`` — the same through the refit's matched filter, which never saw
      *g*: the unbiased version. A row that pulled the filter towards itself has a
      high in-sample and a low held-out score.
    - ``within_participant_isc`` — each row of *g* through *model*'s filter,
      correlated with the group's other rows (a participant's other condition). In a
      subject-axis join the between-recording term rewards this pair too, so a
      component can rest on participants resembling **themselves** across conditions
      while both other-group scores stay near zero. NaN for a one-row group.

    The reference excludes the whole group, so a participant's other recording never
    serves as its own partner.

    :param stats: Output of :func:`jackknife_statistics` on the data *model* was
        fitted to.
    :param model: The full-data fit; its orientation is irrelevant (|r|, ISC are
        sign-invariant).
    :param n_pca: Subspace dimension of the refits, as for the full fit.
    :param gamma: Shrinkage of the refits, as for the full fit.
    :return: One row per (row of the data, component): ``group``, ``row``,
        ``component`` (1-based), ``isc_full``, ``isc_without``, ``influence``
        (``isc_full - isc_without``), ``pattern_r``, ``insample_isc``,
        ``heldout_isc``, ``within_participant_isc``. ``isc_full`` is the fit's
        training ISC; refit columns are
        NaN when the refit has fewer components than *model*.
    """
    isc_full = isc_from_covariances(stats.covariances, model.filters)
    n_components = model.n_components
    rows: list[dict] = []
    for g, members in enumerate(stats.groups):
        reduced, refit, matched, similarity = _matched_refit(
            stats, g, model, n_pca=n_pca, gamma=gamma
        )
        has_match = matched >= 0
        filters = _matched_filters(model, refit, matched)
        insample = _reference_isc(stats, g, model.filters, reduced.total)
        own_group = _own_group_isc(stats, g, model.filters)
        heldout = _reference_isc(stats, g, filters, reduced.total)
        isc_without = np.where(has_match, refit.isc[np.clip(matched, 0, None)], np.nan)
        pattern_r = np.where(
            has_match,
            similarity[np.arange(n_components), np.clip(matched, 0, None)],
            np.nan,
        )
        heldout[:, ~has_match] = np.nan
        for i, row in enumerate(members):
            for k in range(n_components):
                rows.append(
                    {
                        "group": g,
                        "row": row,
                        "component": k + 1,
                        "isc_full": float(isc_full[k]),
                        "isc_without": float(isc_without[k]),
                        "influence": float(isc_full[k] - isc_without[k]),
                        "pattern_r": float(pattern_r[k]),
                        "insample_isc": float(insample[i, k]),
                        "heldout_isc": float(heldout[i, k]),
                        "within_participant_isc": float(own_group[i, k]),
                    }
                )
    return pd.DataFrame(rows)


def robust_influence_z(influence: pd.DataFrame) -> pd.Series:
    """Modified z-score of each group's influence within its component.

    ``0.6745 (Δ - median) / MAD`` over the groups (one value per group, not per row),
    broadcast back to every row. NaN where the MAD is zero.
    """
    per_group = influence.drop_duplicates(["group", "component"])

    def _z(values: pd.Series) -> pd.Series:
        median = values.median()
        mad = (values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            return pd.Series(np.nan, index=values.index)
        return 0.6745 * (values - median) / mad

    z = per_group.groupby("component")["influence"].transform(_z)
    lookup = dict(zip(zip(per_group["group"], per_group["component"]), z))
    return pd.Series(
        [lookup[(g, c)] for g, c in zip(influence["group"], influence["component"])],
        index=influence.index,
        name="influence_z",
    )


def _channel_chunks(
    data_4d: np.ndarray, edges: Sequence[int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """The z-scored ``(S, C, F·T)`` samples of :func:`decompose_channel_corrca` in
    ``(segment, chunk)`` pieces.

    Z-scored over the whole time axis one block of whole frequency rows at a time (so
    no z-scored copy of the whole tensor is held), then cut along time to bound the
    float64 chunk, never across a segment edge.

    :param edges: Time-axis segment edges ``(0, …, T)``; ``None`` is one segment.
    """
    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    edges = (0, n_times) if edges is None else tuple(int(e) for e in edges)
    per_time = n_subjects * n_channels
    t_step = max(1, _JACKKNIFE_CHUNK_VALUES // per_time)
    f_step = max(1, _JACKKNIFE_CHUNK_VALUES // (per_time * n_times))
    for f0 in range(0, n_freqs, f_step):
        block = zscore_by_time(data_4d[:, :, f0 : f0 + f_step])
        for f in range(block.shape[2]):
            for segment, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
                for t0 in range(start, stop, t_step):
                    yield segment, block[:, :, f, t0 : min(t0 + t_step, stop)]
        del block


def channel_jackknife_statistics(
    data_4d: np.ndarray, participants: Sequence[str], *, group_cross: bool = False
) -> JackknifeStatistics:
    """:func:`jackknife_statistics` of a :func:`decompose_channel_corrca` input, with
    every recording of a participant in one group."""
    if np.ndim(data_4d) != 4:
        raise ValueError(f"data_4d must be (S, C, F, T); got {np.shape(data_4d)}.")
    if len(participants) != data_4d.shape[0]:
        raise ValueError(
            f"{data_4d.shape[0]} row(s) but {len(participants)} participant label(s)."
        )
    return jackknife_statistics(
        (chunk for _, chunk in _channel_chunks(data_4d)),
        participant_groups(participants),
        group_cross=group_cross,
    )


def channel_corrca_influence(
    data_4d: np.ndarray,
    model: CorrcaResult,
    participants: Sequence[str],
    conditions: Sequence[str] | None = None,
    *,
    n_pca: int | None,
    gamma: float,
    stats: JackknifeStatistics | None = None,
) -> pd.DataFrame:
    """Leave-one-participant-out influence of a :func:`decompose_channel_corrca` fit.

    Every recording of a participant is left out together, so in a subject-axis join
    (Placebo and Psilocybin as separate rows) the participant's own cross-condition
    pair — which the between-recording term rewards — goes with them.

    :param data_4d: ``(S, C, F, T)`` the wavelet power *model* was fitted to.
    :param model: The fitted (possibly sign-anchored, possibly stored) model.
    :param participants: Participant label per row.
    :param conditions: Condition per row, or ``None`` when a row is one participant.
    :param n_pca: The full fit's ``n_pca``.
    :param gamma: The full fit's shrinkage.
    :param stats: :func:`channel_jackknife_statistics` of *data_4d*, when already
        computed; ``None`` computes it.
    :return: :func:`leave_group_out_influence` with ``participant``, ``condition``,
        ``relative_influence`` (``influence / |isc_full|``), ``influence_z``
        (:func:`robust_influence_z`) and ``outlier`` (``influence_z >
        INFLUENCE_Z_THRESHOLD`` and ``relative_influence >= MIN_RELATIVE_INFLUENCE``)
        added.
    :raises ValueError: If the labels do not match the rows, or *data_4d* is not 4-D.
    """
    if np.ndim(data_4d) != 4:
        raise ValueError(f"data_4d must be (S, C, F, T); got {np.shape(data_4d)}.")
    if len(participants) != data_4d.shape[0] or (
        conditions is not None and len(conditions) != data_4d.shape[0]
    ):
        raise ValueError(
            f"{data_4d.shape[0]} row(s) but {len(participants)} participant and "
            f"{None if conditions is None else len(conditions)} condition label(s)."
        )
    if stats is None:
        stats = channel_jackknife_statistics(data_4d, participants)
    influence = leave_group_out_influence(stats, model, n_pca=n_pca, gamma=gamma)
    return annotate_influence(influence, participants, conditions)


def annotate_influence(
    influence: pd.DataFrame,
    participants: Sequence[str],
    conditions: Sequence[str] | None,
) -> pd.DataFrame:
    """Add the labels and the outlier flag to :func:`leave_group_out_influence`-style
    rows.

    :param participants: Participant per data row (the ``row`` column).
    :param conditions: Condition per data row, or ``None``.
    :return: A copy with ``participant``, ``condition``, ``relative_influence``
        (``influence / |isc_full|``), ``influence_z`` (:func:`robust_influence_z`) and
        ``outlier`` (``influence_z > INFLUENCE_Z_THRESHOLD`` and ``relative_influence
        >= MIN_RELATIVE_INFLUENCE``).
    """
    influence = influence.copy()
    influence.insert(0, "participant", [participants[r] for r in influence["row"]])
    influence.insert(
        1,
        "condition",
        None if conditions is None else [conditions[r] for r in influence["row"]],
    )
    scale = influence["isc_full"].abs()
    influence["relative_influence"] = influence["influence"] / scale.where(scale > 0)
    influence["influence_z"] = robust_influence_z(influence)
    influence["outlier"] = (influence["influence_z"] > INFLUENCE_Z_THRESHOLD) & (
        influence["relative_influence"] >= MIN_RELATIVE_INFLUENCE
    )
    return influence


def _correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(K,)`` Pearson r of two centred ``(K, n)`` signals, NaN where one is flat."""
    num = np.einsum("kn,kn->k", a, b)
    den = np.sqrt(np.einsum("kn,kn->k", a, a) * np.einsum("kn,kn->k", b, b))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def map_influence(
    full_maps: np.ndarray,
    refit_maps: Mapping[str, np.ndarray],
    participants: Sequence[str],
    pattern_r: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """:func:`leave_group_out_influence`'s read-outs on maps the refits were read
    through, rather than on the data they were fitted to.

    For a read-out that is not the fit's own samples — e.g. each recording's
    onset-locked, trial-averaged TF map — the covariances cannot score it, so every
    refit's filters are applied and the map made first. ISCs are those of
    :func:`projected_isc` (group) and Pearson r against the sum of the other
    participants' maps (per recording), every trailing axis counting as samples.

    :param full_maps: ``(R, K, ...)`` per recording, through the full fit's filters.
    :param refit_maps: Participant → ``(R, K, ...)`` every recording through the
        filters refitted without that participant (matched to the full fit's
        components).
    :param participants: Participant per recording; their recordings are left out
        together.
    :param pattern_r: Participant → ``(K,)`` |r| of the refit's and the full fit's
        topographies; NaN when not given.
    :return: The columns of :func:`leave_group_out_influence`, ``isc_full`` being the
        full maps' group ISC.
    :raises ValueError: If shapes or labels disagree, or a participant has no refit.
    """
    full = _as_projected(full_maps)
    n_rows, n_components = full.shape[:2]
    if len(participants) != n_rows:
        raise ValueError(f"{n_rows} map(s) but {len(participants)} participant(s).")
    groups = participant_groups(participants)
    missing = [
        participants[g[0]] for g in groups if participants[g[0]] not in refit_maps
    ]
    if missing:
        raise ValueError(f"No refit maps for {missing}.")
    if len(groups) < 3:
        raise ValueError("Leaving a participant out must leave at least two.")
    centred = np.stack([_centred(full[r]) for r in range(n_rows)])
    total = centred.sum(axis=0)
    isc_full = projected_isc(full)
    out: list[dict] = []
    for g, members in enumerate(groups):
        name = participants[members[0]]
        refit = _as_projected(refit_maps[name])
        if refit.shape != full.shape:
            raise ValueError(
                f"Refit maps of {name} are {refit.shape}; the full maps {full.shape}."
            )
        others = [r for r in range(n_rows) if r not in members]
        refit_centred = np.stack([_centred(refit[r]) for r in range(n_rows)])
        refit_others = refit_centred[others].sum(axis=0)
        isc_without = projected_isc(refit[others])
        own_total = centred[list(members)].sum(axis=0)
        r_pattern = (
            np.full(n_components, np.nan)
            if pattern_r is None
            else np.asarray(pattern_r[name], dtype=float)
        )
        for row in members:
            insample = _correlation(centred[row], total - own_total)
            heldout = _correlation(refit_centred[row], refit_others)
            own = (
                _correlation(centred[row], own_total - centred[row])
                if len(members) > 1
                else np.full(n_components, np.nan)
            )
            for k in range(n_components):
                out.append(
                    {
                        "group": g,
                        "row": row,
                        "component": k + 1,
                        "isc_full": float(isc_full[k]),
                        "isc_without": float(isc_without[k]),
                        "influence": float(isc_full[k] - isc_without[k]),
                        "pattern_r": float(r_pattern[k]),
                        "insample_isc": float(insample[k]),
                        "heldout_isc": float(heldout[k]),
                        "within_participant_isc": float(own[k]),
                    }
                )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Leave-subset-out refit filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetRefits:
    """CorrCA refitted without each subset of groups, in the full fit's components.

    :param subsets: ``R`` tuples of left-out group indices.
    :param filters: ``(R, D, K)`` each refit's filters matched to the full fit's
        components (Hungarian on |r| of the shared patterns) and signed so the matched
        pattern correlates positively with the full one, which keeps a sign-sensitive
        read-out (a baseline-referenced response) comparable; zero where a component
        found no match.
    :param pattern_r: ``(R, K)`` |r| of each matched pattern with the full one; NaN
        where unmatched.
    :param filter_r: ``(R, K)`` Pearson r of each matched filter with the full one,
        after the sign alignment; NaN where unmatched.
    """

    subsets: tuple[tuple[int, ...], ...]
    filters: np.ndarray
    pattern_r: np.ndarray
    filter_r: np.ndarray


def leave_subsets_out(
    n_groups: int, sizes: Iterable[int]
) -> tuple[tuple[int, ...], ...]:
    """Every subset of ``range(n_groups)`` of each size in *sizes*, smallest first.

    :raises ValueError: If a size would leave fewer than two groups, or is below one.
    """
    subsets: list[tuple[int, ...]] = []
    for size in sorted({int(s) for s in sizes}):
        if not 1 <= size <= n_groups - 2:
            raise ValueError(
                f"Leaving {size} of {n_groups} groups out must leave at least two."
            )
        subsets.extend(itertools.combinations(range(n_groups), size))
    return tuple(subsets)


def subset_refit_filters(
    stats: JackknifeStatistics,
    model: CorrcaResult,
    subsets: Sequence[Sequence[int]],
    *,
    n_pca: int | None,
    gamma: float,
) -> SubsetRefits:
    """Refit CorrCA without each subset of groups, all from one set of statistics.

    :param stats: :func:`jackknife_statistics` of the data *model* was fitted to, with
        ``group_cross`` when any subset holds more than one group.
    :param model: The full (sign-anchored) fit the refits are matched and signed to.
    :param subsets: Group indices left out together, e.g. :func:`leave_subsets_out`.
    :param n_pca: The full fit's ``n_pca``.
    :param gamma: The full fit's shrinkage.
    :return: A :class:`SubsetRefits`.
    """
    subsets = tuple(tuple(int(g) for g in subset) for subset in subsets)
    n_components = model.n_components
    filters = np.zeros((len(subsets), *model.filters.shape))
    pattern_r = np.full((len(subsets), n_components), np.nan)
    filter_r = np.full_like(pattern_r, np.nan)
    for i, subset in enumerate(subsets):
        _, refit, matched, _ = _matched_refit(
            stats, subset, model, n_pca=n_pca, gamma=gamma
        )
        has = matched >= 0
        if not has.any():
            continue
        r = matched_component_correlation(
            model.patterns[has], refit.patterns[matched[has]]
        )
        aligned = refit.filters[:, matched[has]] * np.where(r < 0, -1.0, 1.0)
        filters[i][:, has] = aligned
        pattern_r[i, has] = np.abs(r)
        filter_r[i, has] = matched_component_correlation(
            model.filters[:, has].T, aligned.T
        )
    return SubsetRefits(
        subsets=subsets, filters=filters, pattern_r=pattern_r, filter_r=filter_r
    )


#: A refit topography is flagged as an outlier when its |r| with the full fit's is a
#: robust low outlier among the refits of its size AND sits at least this far below
#: their median, so a negligible dip among near-identical refits is never flagged.
MIN_TOPOGRAPHY_DROP: float = 0.05


def low_topography_flags(pattern_r: np.ndarray) -> np.ndarray:
    """Which refit topographies moved unusually far from the full fit's.

    ``0.6745 (r - median) / MAD < -INFLUENCE_Z_THRESHOLD`` (the influence check's
    robust cut, on the low side) and ``median - r >= MIN_TOPOGRAPHY_DROP``, over the
    values given (one subset size of one component). NaN is never flagged.

    :param pattern_r: ``(R,)`` |r| of each refit's topography with the full one.
    :return: ``(R,)`` boolean.
    """
    r = np.asarray(pattern_r, dtype=float)
    finite = np.isfinite(r)
    flags = np.zeros(r.shape, dtype=bool)
    if finite.sum() < 3:
        return flags
    median = np.median(r[finite])
    drop = median - r
    mad = np.median(np.abs(r[finite] - median))
    if mad > 0:
        z = 0.6745 * (r - median) / mad
        flags[finite] = (z[finite] < -INFLUENCE_Z_THRESHOLD) & (
            drop[finite] >= MIN_TOPOGRAPHY_DROP
        )
    else:
        flags[finite] = drop[finite] >= MIN_TOPOGRAPHY_DROP
    return flags
