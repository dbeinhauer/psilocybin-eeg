"""
Shared preparation and IVA component helpers for 4-D wavelet-power tensors.

Two concerns live here: z-scoring a ``(n_subjects, n_channels, n_freqs,
n_times)`` tensor along the time axis, and resolving the free polarity that
IVA leaves on its components.

All functions are pure (no I/O, no plotting) and never mutate their inputs.
"""

from __future__ import annotations

import numpy as np

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


def iva_component_patterns(
    w_subject: np.ndarray, pca_components: np.ndarray
) -> np.ndarray:
    """Forward (mixing) patterns of one subject's IVA components.

    ``W_k @ pca_components`` is the **backward** model — the spatial *filter*
    that extracts each source from the features. What belongs on a topomap (or
    any "what does this component look like" plot) is the **forward** model: the
    mixing pattern that says how the source projects *onto* the features. The
    two are not interchangeable, and conflating them is the classic filter-vs-
    pattern error (Haufe et al., 2014, NeuroImage 87:96-110). MNE follows the
    same convention: ``ica.get_components()`` returns the mixing matrix, not
    ``unmixing_matrix_``.

    The distinction matters especially after ``iva_g(..., whiten=True)``, which
    folds the whitening matrix ``V_k = Λ^{-1/2} Eᵀ`` into the returned ``W_k``.
    Writing ``W_k = W_white,k V_k`` with ``W_white,k`` (near-)orthogonal::

        pattern ∝ P_kᵀ V_k⁻¹ W_white,kᵀ = P_kᵀ E Λ^{+1/2} c
        filter  ∝ P_kᵀ V_kᵀ  W_white,kᵀ = P_kᵀ E Λ^{-1/2} c

    so the filter carries a ``Σ⁻¹`` weighting that up-weights the *lowest*-
    variance retained PCA directions — the noisy tail of the per-subject PCA.
    With a realistic eigenvalue spread the filter and the pattern of the same
    component are nearly unrelated.

    Because ``pca_components`` has orthonormal rows (sklearn's ``PCA``), the
    pattern matrix is exactly the pseudo-inverse of the composite unmixing
    operator::

        A_k = pinv(W_k @ P_k) = P_kᵀ W_k⁻¹        (shape ``(n_features, K)``)

    and this function returns ``A_kᵀ``, so the result is indexed
    ``[component, feature]`` — a drop-in replacement for
    ``W_k @ pca_components``.

    Per-subject sign alignment propagates correctly: pre-multiplying ``W_k`` by
    ``diag(s)`` with ``s = ±1`` scales column ``k`` of the inverse by ``s_k``,
    so a component's pattern flips together with its source. Call this **after**
    :func:`align_iva_component_signs`.

    :param w_subject: ``(K, K)`` unmixing matrix for one subject — a
        ``W[:, :, k]`` slice of the ``iva_g`` output (whitening already folded
        in by ``iva_g``).
    :param pca_components: ``(K, n_features)`` per-subject PCA loadings
        (``pca.components_``), assumed to have orthonormal rows.
    :return: ``(K, n_features)`` forward patterns, one row per component.
    :raises ValueError: If ``w_subject`` is not square or its size does not
        match ``pca_components``' first axis.
    """
    if w_subject.ndim != 2 or w_subject.shape[0] != w_subject.shape[1]:
        raise ValueError(
            f"w_subject must be a square (K, K) matrix; got shape {w_subject.shape}."
        )
    if pca_components.ndim != 2 or pca_components.shape[0] != w_subject.shape[0]:
        raise ValueError(
            f"pca_components must be (K, n_features) with K = "
            f"{w_subject.shape[0]} to match w_subject; got shape "
            f"{pca_components.shape}."
        )
    # (W⁻¹)ᵀ P — via pinv so a near-singular W degrades gracefully instead of
    # raising. Equivalent to pinv(W @ P).T for orthonormal-row P.
    return np.linalg.pinv(w_subject).T @ pca_components


# ---------------------------------------------------------------------------
# Per-subject pattern scaling
# ---------------------------------------------------------------------------


def normalize_patterns_per_subject(patterns: np.ndarray) -> np.ndarray:
    """Rescale each subject's component pattern to unit L2 norm.

    ``iva_g`` fixes the *source* scale (its unmixing rows are unit-norm in the
    whitened space) but not the *pattern* scale: ``A_k = Pᵀ W_k⁻¹`` still
    carries subject *k*'s channel-PCA eigenvalue spread, so patterns differ
    across subjects by a subject-specific gain. Averaging them unnormalised
    lets the largest-gain subjects dominate the group mean, and makes an
    across-subject variance map a picture of amplitude differences rather than
    of topographic disagreement. Call this before any across-subject mean /
    variance / shared-colour-limit plot of the patterns.

    Correlation-based quantities (e.g. topography correlations against a
    reference) are scale-invariant and therefore unaffected by this rescaling.

    :param patterns: ``(S, K, C)`` per-subject component patterns (channel
        topographies, or any feature axis) — **not** mutated.
    :return: A **new** array of the same shape, each ``(subject, component)``
        pattern having unit L2 norm. All-zero patterns are left at zero
        instead of dividing by zero.
    :raises ValueError: If *patterns* is not 3-D or has no subjects.
    """
    if patterns.ndim != 3:
        raise ValueError(
            f"patterns must be 3-D (n_subjects, n_components, n_channels); "
            f"got shape {patterns.shape} (ndim={patterns.ndim})."
        )
    if patterns.shape[0] < 1:
        raise ValueError("patterns must contain at least one subject.")

    norms = np.linalg.norm(patterns, axis=2)  # (S, K)
    return patterns / np.where(norms == 0.0, 1.0, norms)[:, :, np.newaxis]
