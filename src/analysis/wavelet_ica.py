"""
PCA / ICA decomposition of 4-D wavelet-power tensors.

Five reshape strategies project the ``(n_subjects, n_channels, n_freqs,
n_times)`` wavelet-power array into a 2-D matrix suitable for sklearn
``PCA`` → ``FastICA``.  Each strategy highlights a different aspect of the
data (spatial vs. spectral vs. temporal structure).

Before reshaping, the tensor is **z-scored along the time axis** so that
every ``(subject, channel, frequency)`` slice has zero mean and unit
variance.  This removes overall amplitude differences across
subjects/channels/frequencies and ensures PCA/ICA operates on
standardised activations.

+---+------------------------+-------------------+------------------+
| # | Name                   | Observation axis  | Feature axis     |
+---+------------------------+-------------------+------------------+
| 1 | Super-Brain            | time  ``(T,)``    | ``S × C × F``   |
| 2 | Inter-Subject          | ``S × T``         | ``C × F``        |
| 3 | Temporal               | ``S × C``         | ``F × T``        |
| 4 | Inverted Super-Brain   | ``S × C × F``     | time ``(T,)``    |
| 5 | Subject-Frequency      | ``S × F``         | ``C × T``        |
+---+------------------------+-------------------+------------------+

All functions are **pure** (no I/O, no plotting) and never mutate input
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, FastICA

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_4d(data: np.ndarray) -> None:
    """Raise ``ValueError`` when *data* is not 4-D."""
    if data.ndim != 4:
        raise ValueError(
            f"Expected a 4-D array (n_subjects, n_channels, n_freqs, n_times), "
            f"got shape {data.shape} (ndim={data.ndim})."
        )


def zscore_by_time(data: np.ndarray) -> np.ndarray:
    """Z-score a 4-D wavelet-power tensor along the time axis.

    For each ``(subject, channel, frequency)`` slice the time series is
    normalised to zero mean and unit variance.  This removes overall
    amplitude differences across subjects/channels/frequencies so that
    PCA/ICA operate on standardised activations.

    :param data: ``(S, C, F, T)`` wavelet-power tensor (**not** mutated).
    :return: A **new** array of the same shape with each
        ``(subject, channel, frequency)`` slice having mean ≈ 0 and
        std ≈ 1 along the time axis.
    """
    _validate_4d(data)
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    # Guard against zero-variance slices (constant time series).
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std


# ---------------------------------------------------------------------------
# Result containers (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuperBrainResult:
    """Results of Super-Brain PCA/ICA decomposition.

    Observation axis: time ``(T,)``; feature axis: ``S × C × F``.

    :param pca_scores: ``(T, n_pca)`` — PCA-transformed time courses.
    :param pca_components: ``(n_pca, S*C*F)`` — PCA loading vectors.
    :param pca_explained_variance_ratio: ``(n_pca,)`` — fraction of variance.
    :param ica_sources: ``(T, n_ica)`` — independent component time courses.
    :param ica_mixing: ``(S*C*F, n_ica)`` — ICA mixing matrix.
    :param n_subjects: Number of subjects ``S``.
    :param n_channels: Number of channels ``C``.
    :param n_freqs: Number of frequency bins ``F``.
    :param n_times: Number of time points ``T``.
    """

    pca_scores: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    ica_sources: np.ndarray
    ica_mixing: np.ndarray
    n_subjects: int
    n_channels: int
    n_freqs: int
    n_times: int


@dataclass(frozen=True)
class InterSubjectResult:
    """Results of Inter-Subject PCA/ICA decomposition.

    Observation axis: ``S × T``; feature axis: ``C × F``.

    :param pca_scores: ``(S*T, n_pca)`` — PCA-transformed activations.
    :param pca_components: ``(n_pca, C*F)`` — PCA loading vectors.
    :param pca_explained_variance_ratio: ``(n_pca,)`` — fraction of variance.
    :param ica_sources: ``(S*T, n_ica)`` — independent component activations.
    :param ica_mixing: ``(C*F, n_ica)`` — ICA mixing matrix.
    :param n_subjects: ``S``.
    :param n_channels: ``C``.
    :param n_freqs: ``F``.
    :param n_times: ``T``.
    """

    pca_scores: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    ica_sources: np.ndarray
    ica_mixing: np.ndarray
    n_subjects: int
    n_channels: int
    n_freqs: int
    n_times: int


@dataclass(frozen=True)
class TemporalResult:
    """Results of Temporal PCA/ICA decomposition.

    Observation axis: ``S × C``; feature axis: ``F × T``.

    :param pca_scores: ``(S*C, n_pca)`` — PCA-transformed spectra.
    :param pca_components: ``(n_pca, F*T)`` — PCA loading vectors.
    :param pca_explained_variance_ratio: ``(n_pca,)`` — fraction of variance.
    :param ica_sources: ``(S*C, n_ica)`` — independent component spectra.
    :param ica_mixing: ``(F*T, n_ica)`` — ICA mixing matrix.
    :param n_subjects: ``S``.
    :param n_channels: ``C``.
    :param n_freqs: ``F``.
    :param n_times: ``T``.
    """

    pca_scores: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    ica_sources: np.ndarray
    ica_mixing: np.ndarray
    n_subjects: int
    n_channels: int
    n_freqs: int
    n_times: int


@dataclass(frozen=True)
class InvertedSuperBrainResult:
    """Results of Inverted Super-Brain PCA/ICA decomposition.

    Observation axis: ``S × C × F``; feature axis: time ``(T,)``.

    :param pca_scores: ``(S*C*F, n_pca)`` — PCA-transformed features.
    :param pca_components: ``(n_pca, T)`` — PCA loading vectors (temporal).
    :param pca_explained_variance_ratio: ``(n_pca,)`` — fraction of variance.
    :param ica_sources: ``(S*C*F, n_ica)`` — independent component features.
    :param ica_mixing: ``(T, n_ica)`` — ICA mixing matrix (temporal).
    :param n_subjects: ``S``.
    :param n_channels: ``C``.
    :param n_freqs: ``F``.
    :param n_times: ``T``.
    """

    pca_scores: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    ica_sources: np.ndarray
    ica_mixing: np.ndarray
    n_subjects: int
    n_channels: int
    n_freqs: int
    n_times: int


@dataclass(frozen=True)
class SubjectFrequencyResult:
    """Results of Subject-Frequency PCA/ICA decomposition.

    Observation axis: ``S × F``; feature axis: ``C × T``.

    :param pca_scores: ``(S*F, n_pca)`` — PCA-transformed activations.
    :param pca_components: ``(n_pca, C*T)`` — PCA loading vectors.
    :param pca_explained_variance_ratio: ``(n_pca,)`` — fraction of variance.
    :param ica_sources: ``(S*F, n_ica)`` — independent component activations.
    :param ica_mixing: ``(C*T, n_ica)`` — ICA mixing matrix.
    :param n_subjects: ``S``.
    :param n_channels: ``C``.
    :param n_freqs: ``F``.
    :param n_times: ``T``.
    """

    pca_scores: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    ica_sources: np.ndarray
    ica_mixing: np.ndarray
    n_subjects: int
    n_channels: int
    n_freqs: int
    n_times: int


# ---------------------------------------------------------------------------
# Reshape functions
# ---------------------------------------------------------------------------


def reshape_superbrain(data: np.ndarray) -> np.ndarray:
    """Reshape for Super-Brain: observations = time, features = S × C × F.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :return: ``(T, S*C*F)`` 2-D matrix.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    # Transpose to (T, S, C, F) then flatten the last three axes.
    return np.ascontiguousarray(data.transpose(3, 0, 1, 2).reshape(T, S * C * F))


def reshape_intersubject(data: np.ndarray) -> np.ndarray:
    """Reshape for Inter-Subject: observations = S × T, features = C × F.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :return: ``(S*T, C*F)`` 2-D matrix.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    # Transpose to (S, T, C, F) then flatten appropriately.
    return np.ascontiguousarray(data.transpose(0, 3, 1, 2).reshape(S * T, C * F))


def reshape_temporal(data: np.ndarray) -> np.ndarray:
    """Reshape for Temporal: observations = S × C, features = F × T.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :return: ``(S*C, F*T)`` 2-D matrix.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    # Flatten first two axes as observations, last two as features.
    return np.ascontiguousarray(data.reshape(S * C, F * T))


def reshape_inverted_superbrain(data: np.ndarray) -> np.ndarray:
    """Reshape for Inverted Super-Brain: observations = S × C × F, features = T.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :return: ``(S*C*F, T)`` 2-D matrix.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    return np.ascontiguousarray(data.reshape(S * C * F, T))


def reshape_subject_frequency(data: np.ndarray) -> np.ndarray:
    """Reshape for Subject-Frequency: observations = S × F, features = C × T.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :return: ``(S*F, C*T)`` 2-D matrix.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    # Transpose to (S, F, C, T) then flatten appropriately.
    return np.ascontiguousarray(data.transpose(0, 2, 1, 3).reshape(S * F, C * T))


# ---------------------------------------------------------------------------
# Decomposition functions
# ---------------------------------------------------------------------------


def _run_pca_ica(
    matrix: np.ndarray,
    n_pca: int,
    n_ica: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run PCA dimensionality reduction followed by FastICA.

    :returns: ``(pca_scores, pca_components, explained_var_ratio,
        ica_sources, ica_mixing)``
    """
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_scores = pca.fit_transform(matrix)

    ica = FastICA(n_components=n_ica, random_state=random_state, max_iter=1000)
    ica_sources = ica.fit_transform(pca_scores)
    # Mixing matrix maps from ICA sources back to the PCA space, then to
    # the original feature space: A_orig = pca.components_.T @ ica.mixing_
    ica_mixing = pca.components_.T @ ica.mixing_

    return (
        pca_scores,
        pca.components_,
        pca.explained_variance_ratio_,
        ica_sources,
        ica_mixing,
    )


def decompose_superbrain(
    data: np.ndarray,
    n_pca: int = 20,
    n_ica: int = 10,
    random_state: int = 42,
) -> SuperBrainResult:
    """PCA/ICA decomposition using the Super-Brain reshape.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :param n_pca: Number of PCA components to retain.
    :param n_ica: Number of ICA components to extract.
    :param random_state: Seed for reproducibility.
    :return: :class:`SuperBrainResult` with all scores, loadings, and shapes.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    matrix = reshape_superbrain(zscore_by_time(data))
    pca_scores, pca_comp, evr, ica_src, ica_mix = _run_pca_ica(
        matrix, n_pca, n_ica, random_state
    )
    return SuperBrainResult(
        pca_scores=pca_scores,
        pca_components=pca_comp,
        pca_explained_variance_ratio=evr,
        ica_sources=ica_src,
        ica_mixing=ica_mix,
        n_subjects=S,
        n_channels=C,
        n_freqs=F,
        n_times=T,
    )


def decompose_intersubject(
    data: np.ndarray,
    n_pca: int = 20,
    n_ica: int = 10,
    random_state: int = 42,
) -> InterSubjectResult:
    """PCA/ICA decomposition using the Inter-Subject reshape.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :param n_pca: Number of PCA components to retain.
    :param n_ica: Number of ICA components to extract.
    :param random_state: Seed for reproducibility.
    :return: :class:`InterSubjectResult`.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    matrix = reshape_intersubject(zscore_by_time(data))
    pca_scores, pca_comp, evr, ica_src, ica_mix = _run_pca_ica(
        matrix, n_pca, n_ica, random_state
    )
    return InterSubjectResult(
        pca_scores=pca_scores,
        pca_components=pca_comp,
        pca_explained_variance_ratio=evr,
        ica_sources=ica_src,
        ica_mixing=ica_mix,
        n_subjects=S,
        n_channels=C,
        n_freqs=F,
        n_times=T,
    )


def decompose_temporal(
    data: np.ndarray,
    n_pca: int = 20,
    n_ica: int = 10,
    random_state: int = 42,
) -> TemporalResult:
    """PCA/ICA decomposition using the Temporal reshape.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :param n_pca: Number of PCA components to retain.
    :param n_ica: Number of ICA components to extract.
    :param random_state: Seed for reproducibility.
    :return: :class:`TemporalResult`.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    matrix = reshape_temporal(zscore_by_time(data))
    pca_scores, pca_comp, evr, ica_src, ica_mix = _run_pca_ica(
        matrix, n_pca, n_ica, random_state
    )
    return TemporalResult(
        pca_scores=pca_scores,
        pca_components=pca_comp,
        pca_explained_variance_ratio=evr,
        ica_sources=ica_src,
        ica_mixing=ica_mix,
        n_subjects=S,
        n_channels=C,
        n_freqs=F,
        n_times=T,
    )


def decompose_inverted_superbrain(
    data: np.ndarray,
    n_pca: int = 20,
    n_ica: int = 10,
    random_state: int = 42,
) -> InvertedSuperBrainResult:
    """PCA/ICA decomposition using the Inverted Super-Brain reshape.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :param n_pca: Number of PCA components to retain.
    :param n_ica: Number of ICA components to extract.
    :param random_state: Seed for reproducibility.
    :return: :class:`InvertedSuperBrainResult`.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    matrix = reshape_inverted_superbrain(zscore_by_time(data))
    pca_scores, pca_comp, evr, ica_src, ica_mix = _run_pca_ica(
        matrix, n_pca, n_ica, random_state
    )
    return InvertedSuperBrainResult(
        pca_scores=pca_scores,
        pca_components=pca_comp,
        pca_explained_variance_ratio=evr,
        ica_sources=ica_src,
        ica_mixing=ica_mix,
        n_subjects=S,
        n_channels=C,
        n_freqs=F,
        n_times=T,
    )


def decompose_subject_frequency(
    data: np.ndarray,
    n_pca: int = 20,
    n_ica: int = 10,
    random_state: int = 42,
) -> SubjectFrequencyResult:
    """PCA/ICA decomposition using the Subject-Frequency reshape.

    :param data: ``(S, C, F, T)`` wavelet-power tensor.
    :param n_pca: Number of PCA components to retain.
    :param n_ica: Number of ICA components to extract.
    :param random_state: Seed for reproducibility.
    :return: :class:`SubjectFrequencyResult`.
    """
    _validate_4d(data)
    S, C, F, T = data.shape
    matrix = reshape_subject_frequency(zscore_by_time(data))
    pca_scores, pca_comp, evr, ica_src, ica_mix = _run_pca_ica(
        matrix, n_pca, n_ica, random_state
    )
    return SubjectFrequencyResult(
        pca_scores=pca_scores,
        pca_components=pca_comp,
        pca_explained_variance_ratio=evr,
        ica_sources=ica_src,
        ica_mixing=ica_mix,
        n_subjects=S,
        n_channels=C,
        n_freqs=F,
        n_times=T,
    )


# ---------------------------------------------------------------------------
# IVA per-subject sign alignment
# ---------------------------------------------------------------------------


def align_iva_component_signs(
    sigma_n: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve per-subject sign ambiguity of IVA components.

    Independent Vector Analysis (e.g. ``iva_g``) recovers each source-component
    vector (SCV) only up to a per-subject sign: subject *i*'s copy of component
    *k* may be the negative of subject *j*'s. Left unresolved, this corrupts any
    cross-subject comparison (correlations between mismatched subjects come out
    negative even when the underlying activity is shared).

    For each component ``k`` this routine normalises ``Sigma_N[:, :, k]`` to a
    subject × subject correlation matrix, takes its leading eigenvector (largest
    eigenvalue) as the dominant cross-subject direction, orients that
    eigenvector so its largest-magnitude entry is positive (eigenvectors are
    only defined up to sign), and flips every subject whose loading on it is
    negative. Flipping subject *i* on component *k* means negating row ``i`` and
    column ``i`` of that component's correlation matrix and negating component
    ``k``'s row of subject *i*'s unmixing matrix.

    The returned ``signs`` array lets callers propagate the same flips to
    already-recovered sources / component patterns; equivalently, recovering
    sources with the returned ``W_aligned`` yields sign-aligned sources directly.

    :param sigma_n: ``(S, S, K)`` source covariance from the IVA model (``S``
        subjects/datasets, ``K`` components), as returned by ``iva_g``.
    :param W: ``(K, K, S)`` per-subject IVA unmixing matrices.
    :return: ``(sigma_corr, W_aligned, signs)`` where ``sigma_corr`` is the
        sign-aligned ``(K, S, S)`` correlation stack, ``W_aligned`` is a
        sign-aligned copy of ``W`` (``(K, K, S)``), and ``signs`` is the
        ``(K, S)`` array of ``+1`` / ``-1`` flips applied per component &
        subject.
    """
    if sigma_n.ndim != 3 or sigma_n.shape[0] != sigma_n.shape[1]:
        raise ValueError(
            f"sigma_n must be (S, S, K) with a square subject axis; "
            f"got shape {sigma_n.shape}."
        )
    n_subjects, _, n_comp = sigma_n.shape
    if W.shape != (n_comp, n_comp, n_subjects):
        raise ValueError(
            f"W must be (K, K, S) = ({n_comp}, {n_comp}, {n_subjects}) to match "
            f"sigma_n; got shape {W.shape}."
        )

    sigma_corr = np.zeros((n_comp, n_subjects, n_subjects))
    signs = np.ones((n_comp, n_subjects))
    W_aligned = W.copy()

    for k in range(n_comp):
        cov = sigma_n[:, :, k]
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(d, d)

        # Leading eigenvector = dominant cross-subject mode. ``eigh`` returns
        # eigenvalues in ascending order, so the last column is the largest.
        _eigvals, eigvecs = np.linalg.eigh(corr)
        v = eigvecs[:, -1]
        # Eigenvectors are defined up to sign; orient so the largest-magnitude
        # entry is positive (robust, matches sklearn's svd_flip convention).
        if v[np.argmax(np.abs(v))] < 0.0:
            v = -v

        s = np.where(v < 0.0, -1.0, 1.0)
        signs[k] = s
        # Flip mismatched subjects: corr[i, j] *= s_i * s_j (diagonal unchanged).
        sigma_corr[k] = corr * np.outer(s, s)
        # Negate component k's unmixing row for each flipped subject.
        W_aligned[k, :, :] *= s[np.newaxis, :]

    return sigma_corr, W_aligned, signs
