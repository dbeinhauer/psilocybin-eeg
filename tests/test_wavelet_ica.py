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
    reshape_superbrain,
    reshape_intersubject,
    reshape_temporal,
    reshape_inverted_superbrain,
    decompose_superbrain,
    decompose_intersubject,
    decompose_temporal,
    decompose_inverted_superbrain,
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
        np.testing.assert_array_equal(r1.pca_scores, r2.pca_scores)
        np.testing.assert_array_equal(r1.ica_sources, r2.ica_sources)

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
        np.testing.assert_array_equal(r1.pca_scores, r2.pca_scores)
        np.testing.assert_array_equal(r1.ica_sources, r2.ica_sources)

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
        np.testing.assert_array_equal(r1.pca_scores, r2.pca_scores)
        np.testing.assert_array_equal(r1.ica_sources, r2.ica_sources)

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
        np.testing.assert_array_equal(r1.pca_scores, r2.pca_scores)
        np.testing.assert_array_equal(r1.ica_sources, r2.ica_sources)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            decompose_inverted_superbrain(np.zeros((3, 4, 5)), n_pca=5, n_ica=3)


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
        ],
    )
    def test_input_unchanged(self, synthetic_data: np.ndarray, decompose_fn) -> None:
        original = synthetic_data.copy()
        decompose_fn(synthetic_data, n_pca=N_PCA, n_ica=N_ICA)
        np.testing.assert_array_equal(synthetic_data, original)
