"""
Tests for src/analysis/wavelet_ica.py — PCA/ICA decomposition of wavelet tensors.

All functions operate on 4D numpy arrays: (n_subjects, n_channels, n_freqs, n_times).
"""

import pytest
import numpy as np

from src.analysis.wavelet_ica import (
    SuperBrainResult,
    InterSubjectResult,
    TemporalResult,
    InvertedSuperBrainResult,
    SubjectFrequencyResult,
    reshape_superbrain,
    reshape_intersubject,
    reshape_temporal,
    reshape_inverted_superbrain,
    reshape_subject_frequency,
    decompose_superbrain,
    decompose_intersubject,
    decompose_temporal,
    decompose_inverted_superbrain,
    decompose_subject_frequency,
    zscore_by_time,
    align_iva_component_signs,
    iva_component_patterns,
    normalize_patterns_per_subject,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SUBJECTS = 4
N_CHANNELS = 8
N_FREQS = 6
N_TIMES = 200


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """Synthetic 4-D wavelet-power array (float32)."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(N_SUBJECTS, N_CHANNELS, N_FREQS, N_TIMES)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Z-score tests
# ---------------------------------------------------------------------------


class TestZscoreByTime:
    """zscore_by_time normalises each (S, C, F) slice along T."""

    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = zscore_by_time(synthetic_data)
        assert result.shape == synthetic_data.shape

    def test_zero_mean(self, synthetic_data: np.ndarray) -> None:
        result = zscore_by_time(synthetic_data)
        means = result.mean(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-6)

    def test_unit_variance(self, synthetic_data: np.ndarray) -> None:
        result = zscore_by_time(synthetic_data)
        stds = result.std(axis=-1)
        np.testing.assert_allclose(stds, 1, atol=1e-6)

    def test_input_not_mutated(self, synthetic_data: np.ndarray) -> None:
        original = synthetic_data.copy()
        zscore_by_time(synthetic_data)
        np.testing.assert_array_equal(synthetic_data, original)

    def test_constant_timeseries_no_nan(self) -> None:
        """Constant time series should produce zeros, not NaN."""
        data = np.ones((2, 3, 4, 50))
        result = zscore_by_time(data)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, 0)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            zscore_by_time(np.zeros((3, 4, 5)))


# ---------------------------------------------------------------------------
# Reshape tests
# ---------------------------------------------------------------------------


class TestReshapeSuperbrain:
    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = reshape_superbrain(synthetic_data)
        assert result.shape == (N_TIMES, N_SUBJECTS * N_CHANNELS * N_FREQS)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            reshape_superbrain(np.zeros((3, 4, 5)))


class TestReshapeIntersubject:
    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = reshape_intersubject(synthetic_data)
        assert result.shape == (N_SUBJECTS * N_TIMES, N_CHANNELS * N_FREQS)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            reshape_intersubject(np.zeros((3, 4, 5)))


class TestReshapeTemporal:
    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = reshape_temporal(synthetic_data)
        assert result.shape == (N_SUBJECTS * N_CHANNELS, N_FREQS * N_TIMES)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            reshape_temporal(np.zeros((3, 4, 5)))


class TestReshapeInvertedSuperbrain:
    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = reshape_inverted_superbrain(synthetic_data)
        assert result.shape == (N_SUBJECTS * N_CHANNELS * N_FREQS, N_TIMES)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            reshape_inverted_superbrain(np.zeros((3, 4, 5)))


class TestReshapeSubjectFrequency:
    def test_output_shape(self, synthetic_data: np.ndarray) -> None:
        result = reshape_subject_frequency(synthetic_data)
        assert result.shape == (N_SUBJECTS * N_FREQS, N_CHANNELS * N_TIMES)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            reshape_subject_frequency(np.zeros((3, 4, 5)))


# ---------------------------------------------------------------------------
# Decomposition tests
# ---------------------------------------------------------------------------

N_PCA = 10
N_ICA = 5


class TestDecomposeSuperbrain:
    def test_result_type(self, synthetic_data: np.ndarray) -> None:
        result = decompose_superbrain(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert isinstance(result, SuperBrainResult)

    def test_shapes(self, synthetic_data: np.ndarray) -> None:
        r = decompose_superbrain(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.pca_scores.shape == (N_TIMES, N_PCA)
        assert r.pca_components.shape == (N_PCA, N_SUBJECTS * N_CHANNELS * N_FREQS)
        assert r.pca_explained_variance_ratio.shape == (N_PCA,)
        assert r.ica_sources.shape == (N_TIMES, N_ICA)
        assert r.ica_mixing.shape == (N_SUBJECTS * N_CHANNELS * N_FREQS, N_ICA)

    def test_dimension_attrs(self, synthetic_data: np.ndarray) -> None:
        r = decompose_superbrain(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.n_subjects == N_SUBJECTS
        assert r.n_channels == N_CHANNELS
        assert r.n_freqs == N_FREQS
        assert r.n_times == N_TIMES

    def test_reproducibility(self, synthetic_data: np.ndarray) -> None:
        r1 = decompose_superbrain(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        r2 = decompose_superbrain(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        np.testing.assert_allclose(r1.pca_scores, r2.pca_scores, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(r1.ica_sources, r2.ica_sources, rtol=1e-5, atol=1e-8)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_superbrain(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


class TestDecomposeIntersubject:
    def test_result_type(self, synthetic_data: np.ndarray) -> None:
        result = decompose_intersubject(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert isinstance(result, InterSubjectResult)

    def test_shapes(self, synthetic_data: np.ndarray) -> None:
        r = decompose_intersubject(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.pca_scores.shape == (N_SUBJECTS * N_TIMES, N_PCA)
        assert r.pca_components.shape == (N_PCA, N_CHANNELS * N_FREQS)
        assert r.pca_explained_variance_ratio.shape == (N_PCA,)
        assert r.ica_sources.shape == (N_SUBJECTS * N_TIMES, N_ICA)
        assert r.ica_mixing.shape == (N_CHANNELS * N_FREQS, N_ICA)

    def test_reproducibility(self, synthetic_data: np.ndarray) -> None:
        r1 = decompose_intersubject(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        r2 = decompose_intersubject(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        np.testing.assert_allclose(r1.pca_scores, r2.pca_scores, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(r1.ica_sources, r2.ica_sources, rtol=1e-5, atol=1e-8)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_intersubject(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


class TestDecomposeTemporal:
    def test_result_type(self, synthetic_data: np.ndarray) -> None:
        result = decompose_temporal(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert isinstance(result, TemporalResult)

    def test_shapes(self, synthetic_data: np.ndarray) -> None:
        r = decompose_temporal(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.pca_scores.shape == (N_SUBJECTS * N_CHANNELS, N_PCA)
        assert r.pca_components.shape == (N_PCA, N_FREQS * N_TIMES)
        assert r.pca_explained_variance_ratio.shape == (N_PCA,)
        assert r.ica_sources.shape == (N_SUBJECTS * N_CHANNELS, N_ICA)
        assert r.ica_mixing.shape == (N_FREQS * N_TIMES, N_ICA)

    def test_reproducibility(self, synthetic_data: np.ndarray) -> None:
        r1 = decompose_temporal(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        r2 = decompose_temporal(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        np.testing.assert_allclose(r1.pca_scores, r2.pca_scores, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(r1.ica_sources, r2.ica_sources, rtol=1e-5, atol=1e-8)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_temporal(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


class TestDecomposeInvertedSuperbrain:
    def test_result_type(self, synthetic_data: np.ndarray) -> None:
        result = decompose_inverted_superbrain(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert isinstance(result, InvertedSuperBrainResult)

    def test_shapes(self, synthetic_data: np.ndarray) -> None:
        r = decompose_inverted_superbrain(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.pca_scores.shape == (N_SUBJECTS * N_CHANNELS * N_FREQS, N_PCA)
        assert r.pca_components.shape == (N_PCA, N_TIMES)
        assert r.pca_explained_variance_ratio.shape == (N_PCA,)
        assert r.ica_sources.shape == (N_SUBJECTS * N_CHANNELS * N_FREQS, N_ICA)
        assert r.ica_mixing.shape == (N_TIMES, N_ICA)

    def test_reproducibility(self, synthetic_data: np.ndarray) -> None:
        r1 = decompose_inverted_superbrain(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        r2 = decompose_inverted_superbrain(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        np.testing.assert_allclose(r1.pca_scores, r2.pca_scores, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(r1.ica_sources, r2.ica_sources, rtol=1e-5, atol=1e-8)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_inverted_superbrain(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


class TestDecomposeSubjectFrequency:
    def test_result_type(self, synthetic_data: np.ndarray) -> None:
        result = decompose_subject_frequency(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert isinstance(result, SubjectFrequencyResult)

    def test_shapes(self, synthetic_data: np.ndarray) -> None:
        r = decompose_subject_frequency(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.pca_scores.shape == (N_SUBJECTS * N_FREQS, N_PCA)
        assert r.pca_components.shape == (N_PCA, N_CHANNELS * N_TIMES)
        assert r.pca_explained_variance_ratio.shape == (N_PCA,)
        assert r.ica_sources.shape == (N_SUBJECTS * N_FREQS, N_ICA)
        assert r.ica_mixing.shape == (N_CHANNELS * N_TIMES, N_ICA)

    def test_dimension_attrs(self, synthetic_data: np.ndarray) -> None:
        r = decompose_subject_frequency(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert r.n_subjects == N_SUBJECTS
        assert r.n_channels == N_CHANNELS
        assert r.n_freqs == N_FREQS
        assert r.n_times == N_TIMES

    def test_reproducibility(self, synthetic_data: np.ndarray) -> None:
        r1 = decompose_subject_frequency(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        r2 = decompose_subject_frequency(
            synthetic_data, n_pca=N_PCA, n_ica=N_ICA, random_state=0
        )
        np.testing.assert_allclose(r1.pca_scores, r2.pca_scores, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(r1.ica_sources, r2.ica_sources, rtol=1e-5, atol=1e-8)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_subject_frequency(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestExplainedVariance:
    """Explained variance ratios should sum to ≤ 1 and be non-negative."""

    @pytest.mark.parametrize(
        "decompose_fn",
        [
            decompose_superbrain,
            decompose_intersubject,
            decompose_temporal,
            decompose_inverted_superbrain,
            decompose_subject_frequency,
        ],
    )
    def test_variance_ratio_valid(
        self, synthetic_data: np.ndarray, decompose_fn
    ) -> None:
        r = decompose_fn(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        assert np.all(r.pca_explained_variance_ratio >= 0)
        assert r.pca_explained_variance_ratio.sum() <= 1.0 + 1e-6


class TestInputNotMutated:
    """Decomposition functions must not modify the input array."""

    @pytest.mark.parametrize(
        "decompose_fn",
        [
            decompose_superbrain,
            decompose_intersubject,
            decompose_temporal,
            decompose_inverted_superbrain,
            decompose_subject_frequency,
        ],
    )
    def test_input_unchanged(self, synthetic_data: np.ndarray, decompose_fn) -> None:
        original = synthetic_data.copy()
        decompose_fn(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        np.testing.assert_array_equal(synthetic_data, original)


# ---------------------------------------------------------------------------
# IVA sign-alignment tests
# ---------------------------------------------------------------------------


def _make_iva_inputs(true_signs: np.ndarray, n_comp: int = 3):
    """Build synthetic (sigma_n, W) where component 0 has known sign flips.

    Component 0's correlation matrix is a rank-1-plus-identity structure with a
    strictly positive consensus eigenvector, then corrupted by ``true_signs``
    (so mismatched subjects show negative cross-correlations). Remaining
    components are clean (all-positive) so they should incur no flips.
    """
    n_subjects = true_signs.shape[0]
    sigma_n = np.zeros((n_subjects, n_subjects, n_comp))
    # Distinct positive loadings → a unique largest-magnitude eigenvector entry.
    a = np.linspace(0.9, 0.3, n_subjects)
    base = np.outer(a, a) + np.eye(n_subjects) * 0.1
    d = np.sqrt(np.diag(base))
    clean_corr = base / np.outer(d, d)
    sigma_n[:, :, 0] = clean_corr * np.outer(true_signs, true_signs)
    for k in range(1, n_comp):
        sigma_n[:, :, k] = clean_corr
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_comp, n_comp, n_subjects))
    return sigma_n, W, clean_corr


class TestAlignIvaComponentSigns:
    """align_iva_component_signs must undo per-subject sign ambiguity."""

    def test_output_shapes(self) -> None:
        true_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs)
        n_subjects, _, n_comp = sigma_n.shape
        sigma_corr, W_aligned, signs = align_iva_component_signs(sigma_n, W)
        assert sigma_corr.shape == (n_comp, n_subjects, n_subjects)
        assert W_aligned.shape == W.shape
        assert signs.shape == (n_comp, n_subjects)

    def test_recovers_known_flips(self) -> None:
        # First entry positive so there is no global-sign ambiguity to chase.
        true_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs)
        _sigma_corr, _W_aligned, signs = align_iva_component_signs(sigma_n, W)
        np.testing.assert_array_equal(signs[0], true_signs)

    def test_aligned_correlation_is_all_positive(self) -> None:
        true_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        sigma_n, W, clean_corr = _make_iva_inputs(true_signs)
        sigma_corr, _W_aligned, _signs = align_iva_component_signs(sigma_n, W)
        # The flips should restore the original all-positive correlations.
        np.testing.assert_allclose(sigma_corr[0], clean_corr, atol=1e-10)
        assert (sigma_corr[0] > 0).all()

    def test_clean_components_unchanged(self) -> None:
        true_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs)
        _sigma_corr, _W_aligned, signs = align_iva_component_signs(sigma_n, W)
        # Components 1..K already aligned → no subject flipped.
        np.testing.assert_array_equal(signs[1:], np.ones_like(signs[1:]))

    def test_w_rows_flipped_by_signs(self) -> None:
        true_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs)
        _sigma_corr, W_aligned, signs = align_iva_component_signs(sigma_n, W)
        for k in range(W.shape[0]):
            expected = W[k] * signs[k][np.newaxis, :]
            np.testing.assert_allclose(W_aligned[k], expected)

    def test_source_recovery_consistency(self) -> None:
        # Recovering sources with W_aligned == flipping recovered sources by sign.
        true_signs = np.array([1.0, -1.0, 1.0, -1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs, n_comp=2)
        n_subjects, _, n_comp = sigma_n.shape
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n_comp, 50, n_subjects))  # (K, T, S)
        _sigma_corr, W_aligned, signs = align_iva_component_signs(sigma_n, W)
        for subj in range(n_subjects):
            orig = W[:, :, subj] @ X[:, :, subj]
            aligned = W_aligned[:, :, subj] @ X[:, :, subj]
            np.testing.assert_allclose(
                aligned, orig * signs[:, subj][:, np.newaxis], atol=1e-10
            )

    def test_input_not_mutated(self) -> None:
        true_signs = np.array([1.0, -1.0, 1.0])
        sigma_n, W, _ = _make_iva_inputs(true_signs)
        sigma_n_copy, W_copy = sigma_n.copy(), W.copy()
        align_iva_component_signs(sigma_n, W)
        np.testing.assert_array_equal(sigma_n, sigma_n_copy)
        np.testing.assert_array_equal(W, W_copy)

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            align_iva_component_signs(np.zeros((3, 4, 2)), np.zeros((2, 2, 3)))
        with pytest.raises(ValueError):
            # W shape inconsistent with sigma_n.
            align_iva_component_signs(np.zeros((4, 4, 3)), np.zeros((3, 3, 5)))


class TestIvaComponentPatterns:
    """iva_component_patterns must return forward patterns, not unmixing rows."""

    @staticmethod
    def _inputs(n_comp: int = 5, n_features: int = 12, seed: int = 3):
        """Random invertible W plus an orthonormal-row PCA loading matrix."""
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n_comp, n_comp))
        # Orthonormal rows, as produced by sklearn's PCA.
        q, _ = np.linalg.qr(rng.standard_normal((n_features, n_comp)))
        return w, q.T

    def test_output_shape(self) -> None:
        w, p = self._inputs()
        assert iva_component_patterns(w, p).shape == p.shape

    def test_equals_pseudo_inverse_of_unmixing(self) -> None:
        # A = pinv(W @ P); the function returns A.T.
        w, p = self._inputs()
        patterns = iva_component_patterns(w, p)
        np.testing.assert_allclose(patterns, np.linalg.pinv(w @ p).T, atol=1e-10)

    def test_differs_from_unmixing_rows(self) -> None:
        # Guards the regression this function exists to fix.
        w, p = self._inputs()
        assert not np.allclose(iva_component_patterns(w, p), w @ p, atol=1e-6)

    def test_recovers_known_mixing(self) -> None:
        # With a known mixing A and W = pinv(A) (up to the PCA subspace), the
        # returned patterns must match A's columns.
        rng = np.random.default_rng(11)
        n_comp, n_features = 4, 10
        a_true = rng.standard_normal((n_features, n_comp))
        q, _ = np.linalg.qr(a_true)  # PCA basis spanning A's column space
        p = q.T  # (K, n_features), orthonormal rows
        w = np.linalg.pinv(p @ a_true)  # unmixing within the PCA subspace
        np.testing.assert_allclose(iva_component_patterns(w, p), a_true.T, atol=1e-8)

    def test_unmixing_composition_is_identity(self) -> None:
        # patterns.T is a right inverse of the composite unmixing operator.
        w, p = self._inputs()
        patterns = iva_component_patterns(w, p)
        np.testing.assert_allclose((w @ p) @ patterns.T, np.eye(w.shape[0]), atol=1e-10)

    def test_sign_flip_propagates_per_component(self) -> None:
        # align_iva_component_signs left-multiplies W by diag(s); each pattern
        # row must flip with its own component and nothing else.
        w, p = self._inputs()
        signs = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
        base = iva_component_patterns(w, p)
        flipped = iva_component_patterns(signs[:, np.newaxis] * w, p)
        np.testing.assert_allclose(flipped, base * signs[:, np.newaxis], atol=1e-10)

    def test_input_not_mutated(self) -> None:
        w, p = self._inputs()
        w_copy, p_copy = w.copy(), p.copy()
        iva_component_patterns(w, p)
        np.testing.assert_array_equal(w, w_copy)
        np.testing.assert_array_equal(p, p_copy)

    def test_rejects_bad_shapes(self) -> None:
        w, p = self._inputs()
        with pytest.raises(ValueError):
            iva_component_patterns(np.zeros((3, 4)), p)  # not square
        with pytest.raises(ValueError):
            iva_component_patterns(w, np.zeros((w.shape[0] + 1, 12)))  # K mismatch
        with pytest.raises(ValueError):
            iva_component_patterns(np.zeros((2, 2, 2)), p)  # not 2-D


class TestNormalizePatternsPerSubject:
    """normalize_patterns_per_subject must strip per-subject gain, nothing else."""

    @staticmethod
    def _shared_patterns(n_subjects: int = 8, n_comp: int = 3, n_chan: int = 12):
        """Patterns sharing one topography per component, times a subject gain."""
        rng = np.random.default_rng(7)
        shared = rng.standard_normal((n_comp, n_chan))
        gains = np.linspace(0.5, 4.0, n_subjects)
        patterns = np.empty((n_subjects, n_comp, n_chan))
        for s in range(n_subjects):
            patterns[s] = gains[s] * (shared + 0.05 * rng.standard_normal(shared.shape))
        return patterns, shared, gains

    def test_shape_preserved(self) -> None:
        patterns, _, _ = self._shared_patterns()
        assert normalize_patterns_per_subject(patterns).shape == patterns.shape

    def test_patterns_are_unit_norm(self) -> None:
        patterns, _, _ = self._shared_patterns()
        norms = np.linalg.norm(normalize_patterns_per_subject(patterns), axis=2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_invariant_to_per_subject_gain(self) -> None:
        # The whole point: rescaling one subject must not move the group maps.
        patterns, _, _ = self._shared_patterns()
        rescaled = patterns.copy()
        rescaled[0] *= 17.0
        base = normalize_patterns_per_subject(patterns)
        scaled = normalize_patterns_per_subject(rescaled)
        np.testing.assert_allclose(base.mean(axis=0), scaled.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(base.var(axis=0), scaled.var(axis=0), atol=1e-10)

    def test_gain_dominates_the_unnormalised_mean(self) -> None:
        # Guards the regression this function exists to fix: with one loud
        # subject the raw mean map follows that subject, the normalised one does
        # not.
        patterns, _, _ = self._shared_patterns()
        loud = patterns.copy()
        loud[0] *= 50.0
        raw_shift = np.abs(loud.mean(axis=0) - patterns.mean(axis=0)).max()
        norm_shift = np.abs(
            normalize_patterns_per_subject(loud).mean(axis=0)
            - normalize_patterns_per_subject(patterns).mean(axis=0)
        ).max()
        assert norm_shift < 1e-10 < raw_shift

    def test_direction_unchanged(self) -> None:
        # Only the scale may change: each pattern stays collinear with itself.
        patterns, _, _ = self._shared_patterns()
        normalized = normalize_patterns_per_subject(patterns)
        for s in range(patterns.shape[0]):
            for k in range(patterns.shape[1]):
                np.testing.assert_allclose(
                    normalized[s, k] * np.linalg.norm(patterns[s, k]),
                    patterns[s, k],
                    atol=1e-10,
                )

    def test_zero_pattern_does_not_divide_by_zero(self) -> None:
        patterns, _, _ = self._shared_patterns()
        patterns[2, 1, :] = 0.0
        normalized = normalize_patterns_per_subject(patterns)
        assert np.isfinite(normalized).all()
        np.testing.assert_array_equal(normalized[2, 1], 0.0)

    def test_single_subject_is_just_rescaled(self) -> None:
        patterns, _, _ = self._shared_patterns(n_subjects=1)
        norms = np.linalg.norm(normalize_patterns_per_subject(patterns), axis=2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_input_not_mutated(self) -> None:
        patterns, _, _ = self._shared_patterns()
        original = patterns.copy()
        normalize_patterns_per_subject(patterns)
        np.testing.assert_array_equal(patterns, original)

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            normalize_patterns_per_subject(np.zeros((4, 5)))  # not 3-D
        with pytest.raises(ValueError):
            normalize_patterns_per_subject(np.zeros((3, 4, 5, 6)))  # not 3-D
        with pytest.raises(ValueError):
            normalize_patterns_per_subject(np.zeros((0, 4, 5)))  # no subjects


# ---------------------------------------------------------------------------
# Cross-band visualisation tests
# ---------------------------------------------------------------------------


class TestCrossBandScreeComparison:
    """Smoke test: plot_cross_band_scree_comparison returns a Figure."""

    def test_returns_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from src.visualization.wavelet_ica_plots import (
            plot_cross_band_scree_comparison,
        )

        band_var = {
            "delta": np.array([0.3, 0.2, 0.1]),
            "theta": np.array([0.25, 0.15, 0.12]),
        }
        fig = plot_cross_band_scree_comparison(
            band_var, approach="Super-Brain", label="test"
        )
        assert isinstance(fig, Figure)


class TestCrossBandVarianceSummary:
    """Smoke test: plot_cross_band_variance_summary returns a Figure."""

    def test_returns_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from src.visualization.wavelet_ica_plots import (
            plot_cross_band_variance_summary,
        )

        band_var = {
            "Super-Brain": {
                "delta": np.array([0.3, 0.2]),
                "theta": np.array([0.25, 0.15]),
            },
            "Temporal": {
                "delta": np.array([0.4, 0.1]),
                "theta": np.array([0.2, 0.1]),
            },
        }
        fig = plot_cross_band_variance_summary(band_var, label="test")
        assert isinstance(fig, Figure)
