"""
Tests for src/analysis/isc.py — Inter-Subject Correlation functions.

All ISC functions operate on 3D numpy arrays: (n_items, n_features, n_samples).
"""

import pytest
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.analysis.isc import (
    FREQUENCY_BANDS,
    compute_loo_isc,
    compute_loo_isc_spearman,
    compute_mean_field_loo_isc,
    compute_mean_field_pairwise_isc,
    compute_mean_field_sliding_window_isc,
    compute_pairwise_isc,
    compute_pairwise_isc_per_feature,
    compute_pairwise_isc_spearman,
    compute_sliding_window_isc,
    compute_sliding_window_isc_spearman,
)


class TestFrequencyBands:
    """Test FREQUENCY_BANDS constant."""

    def test_all_bands_present(self):
        expected = {"delta", "theta", "alpha", "beta", "gamma"}
        assert set(FREQUENCY_BANDS.keys()) == expected

    def test_bands_are_tuples(self):
        for band, (lo, hi) in FREQUENCY_BANDS.items():
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            assert lo < hi

    def test_delta(self):
        assert FREQUENCY_BANDS["delta"] == (1.0, 4.0)

    def test_gamma(self):
        assert FREQUENCY_BANDS["gamma"] == (30.0, 70.0)


class TestComputeLooISC:
    """Test leave-one-out ISC computation."""

    def test_identical_items_perfect_correlation(self):
        """Identical items should yield ISC ~1.0."""
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(1, 3, 100))
        data = np.repeat(signal, 5, axis=0)
        loo_isc, mean_isc = compute_loo_isc(data)
        assert loo_isc.shape == (5, 3)
        assert mean_isc.shape == (3,)
        np.testing.assert_allclose(mean_isc, 1.0, atol=1e-10)

    def test_output_shapes(self):
        """Check output shapes for arbitrary input."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 2, 50))
        loo_isc, mean_isc = compute_loo_isc(data)
        assert loo_isc.shape == (4, 2)
        assert mean_isc.shape == (2,)

    def test_random_data_isc_near_zero(self):
        """Independent random data should have ISC near 0."""
        rng = np.random.default_rng(123)
        data = rng.normal(size=(10, 2, 500))
        _, mean_isc = compute_loo_isc(data)
        # With 10 subjects and 500 samples, ISC should be small
        assert all(abs(r) < 0.3 for r in mean_isc)

    def test_two_items(self):
        """With two items, LOO-ISC for each should equal pairwise r."""
        rng = np.random.default_rng(7)
        data = rng.normal(size=(2, 1, 100))
        loo_isc, _ = compute_loo_isc(data)
        expected_r = float(pearsonr(data[0, 0], data[1, 0])[0])
        np.testing.assert_allclose(loo_isc[0, 0], expected_r, atol=1e-10)
        np.testing.assert_allclose(loo_isc[1, 0], expected_r, atol=1e-10)


class TestComputePairwiseISC:
    """Test pairwise ISC (mean across features)."""

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(5, 3, 100))
        result = compute_pairwise_isc(data)
        assert result.shape == (5, 5)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(4, 2, 80))
        result = compute_pairwise_isc(data)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(4, 2, 80))
        result = compute_pairwise_isc(data)
        np.testing.assert_allclose(result, result.T)

    def test_identical_items(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(size=(1, 2, 100))
        data = np.repeat(signal, 3, axis=0)
        result = compute_pairwise_isc(data)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestComputePairwiseISCPerFeature:
    """Test pairwise ISC per feature."""

    def test_output_shape(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(4, 3, 80))
        result = compute_pairwise_isc_per_feature(data)
        assert result.shape == (3, 4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(4, 3, 80))
        result = compute_pairwise_isc_per_feature(data)
        for f in range(3):
            np.testing.assert_allclose(np.diag(result[f]), 1.0)

    def test_symmetric_per_feature(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(4, 2, 80))
        result = compute_pairwise_isc_per_feature(data)
        for f in range(2):
            np.testing.assert_allclose(result[f], result[f].T)


class TestComputeSlidingWindowISC:
    """Test time-resolved sliding-window ISC."""

    def test_output_shapes(self):
        rng = np.random.default_rng(8)
        data = rng.normal(size=(3, 2, 1000))
        isc_tc, times = compute_sliding_window_isc(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        # 1000 samples at 100 Hz = 10 s; window 1s, step 0.5s -> 19 windows
        assert isc_tc.shape[1] == 2
        assert len(times) == isc_tc.shape[0]

    def test_window_times_are_monotonic(self):
        rng = np.random.default_rng(9)
        data = rng.normal(size=(3, 1, 500))
        _, times = compute_sliding_window_isc(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_identical_signals_high_isc(self):
        """Identical items in each window → ISC near 1."""
        rng = np.random.default_rng(10)
        signal = rng.normal(size=(1, 2, 200))
        data = np.repeat(signal, 4, axis=0)
        isc_tc, _ = compute_sliding_window_isc(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        np.testing.assert_allclose(isc_tc, 1.0, atol=1e-10)


class TestComputeLooISCSpearman:
    """Test leave-one-out ISC using Spearman correlation."""

    def test_output_shapes(self):
        rng = np.random.default_rng(20)
        data = rng.normal(size=(4, 3, 100))
        loo_isc, mean_isc = compute_loo_isc_spearman(data)
        assert loo_isc.shape == (4, 3)
        assert mean_isc.shape == (3,)

    def test_identical_items_perfect_correlation(self):
        """Identical items should yield Spearman ISC ~1.0."""
        rng = np.random.default_rng(21)
        signal = rng.normal(size=(1, 2, 100))
        data = np.repeat(signal, 5, axis=0)
        loo_isc, mean_isc = compute_loo_isc_spearman(data)
        np.testing.assert_allclose(mean_isc, 1.0, atol=1e-10)

    def test_two_items_matches_spearmanr(self):
        """With two items, LOO-ISC should equal pairwise spearmanr."""
        rng = np.random.default_rng(22)
        data = rng.normal(size=(2, 1, 80))
        loo_isc, _ = compute_loo_isc_spearman(data)
        expected_r = float(spearmanr(data[0, 0], data[1, 0])[0])
        np.testing.assert_allclose(loo_isc[0, 0], expected_r, atol=1e-10)
        np.testing.assert_allclose(loo_isc[1, 0], expected_r, atol=1e-10)

    def test_mean_isc_is_mean_of_loo(self):
        rng = np.random.default_rng(23)
        data = rng.normal(size=(4, 3, 80))
        loo_isc, mean_isc = compute_loo_isc_spearman(data)
        np.testing.assert_allclose(mean_isc, loo_isc.mean(axis=0))


class TestComputePairwiseISCSpearman:
    """Test pairwise ISC using Spearman correlation."""

    def test_output_shape(self):
        rng = np.random.default_rng(30)
        data = rng.normal(size=(5, 2, 80))
        result = compute_pairwise_isc_spearman(data)
        assert result.shape == (5, 5)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(31)
        data = rng.normal(size=(4, 2, 80))
        result = compute_pairwise_isc_spearman(data)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(32)
        data = rng.normal(size=(4, 2, 80))
        result = compute_pairwise_isc_spearman(data)
        np.testing.assert_allclose(result, result.T)

    def test_identical_items(self):
        rng = np.random.default_rng(33)
        signal = rng.normal(size=(1, 2, 80))
        data = np.repeat(signal, 3, axis=0)
        result = compute_pairwise_isc_spearman(data)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestComputeSlidingWindowISCSpearman:
    """Test time-resolved sliding-window ISC using Spearman correlation."""

    def test_output_shapes(self):
        rng = np.random.default_rng(40)
        data = rng.normal(size=(3, 2, 500))
        isc_tc, times = compute_sliding_window_isc_spearman(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        assert isc_tc.shape[1] == 2
        assert len(times) == isc_tc.shape[0]

    def test_window_times_are_monotonic(self):
        rng = np.random.default_rng(41)
        data = rng.normal(size=(3, 1, 300))
        _, times = compute_sliding_window_isc_spearman(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_identical_signals_high_isc(self):
        """Identical items in each window → Spearman ISC near 1."""
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(1, 1, 200))
        data = np.repeat(signal, 4, axis=0)
        isc_tc, _ = compute_sliding_window_isc_spearman(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        np.testing.assert_allclose(isc_tc, 1.0, atol=1e-10)

    def test_same_window_count_as_pearson(self):
        """Spearman and Pearson sliding-window should produce same number of windows."""
        rng = np.random.default_rng(43)
        data = rng.normal(size=(3, 2, 400))
        _, times_p = compute_sliding_window_isc(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        _, times_s = compute_sliding_window_isc_spearman(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        assert len(times_p) == len(times_s)


class TestComputeMeanFieldLooISC:
    """Test mean-field LOO-ISC (Pearson + Spearman)."""

    def test_output_shapes(self):
        rng = np.random.default_rng(50)
        data = rng.normal(size=(5, 10, 200))
        loo_p, loo_s = compute_mean_field_loo_isc(data)
        assert loo_p.shape == (5,)
        assert loo_s.shape == (5,)

    def test_identical_items_perfect_correlation(self):
        """Identical items → mean-field LOO-ISC ~1.0."""
        rng = np.random.default_rng(51)
        signal = rng.normal(size=(1, 3, 100))
        data = np.repeat(signal, 5, axis=0)
        loo_p, loo_s = compute_mean_field_loo_isc(data)
        np.testing.assert_allclose(loo_p, 1.0, atol=1e-10)
        np.testing.assert_allclose(loo_s, 1.0, atol=1e-10)

    def test_two_items_matches_pearsonr(self):
        """With two items, Pearson mean-field LOO-ISC should equal pearsonr of mean-fields."""
        rng = np.random.default_rng(52)
        data = rng.normal(size=(2, 3, 100))
        mf = data.mean(axis=1)
        from scipy.stats import pearsonr

        expected = float(pearsonr(mf[0], mf[1])[0])
        loo_p, _ = compute_mean_field_loo_isc(data)
        np.testing.assert_allclose(loo_p[0], expected, atol=1e-10)
        np.testing.assert_allclose(loo_p[1], expected, atol=1e-10)


class TestComputeMeanFieldPairwiseISC:
    """Test mean-field pairwise ISC."""

    def test_output_shape(self):
        rng = np.random.default_rng(60)
        data = rng.normal(size=(4, 5, 100))
        result = compute_mean_field_pairwise_isc(data)
        assert result.shape == (4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(61)
        data = rng.normal(size=(4, 5, 100))
        result = compute_mean_field_pairwise_isc(data)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(62)
        data = rng.normal(size=(4, 5, 100))
        result = compute_mean_field_pairwise_isc(data)
        np.testing.assert_allclose(result, result.T)

    def test_identical_items(self):
        rng = np.random.default_rng(63)
        signal = rng.normal(size=(1, 3, 100))
        data = np.repeat(signal, 4, axis=0)
        result = compute_mean_field_pairwise_isc(data)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestComputeMeanFieldSlidingWindowISC:
    """Test mean-field sliding-window ISC."""

    def test_output_shapes(self):
        rng = np.random.default_rng(70)
        data = rng.normal(size=(3, 5, 500))
        isc_tc, times = compute_mean_field_sliding_window_isc(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        assert isc_tc.ndim == 1
        assert len(times) == len(isc_tc)

    def test_window_times_monotonic(self):
        rng = np.random.default_rng(71)
        data = rng.normal(size=(3, 5, 400))
        _, times = compute_mean_field_sliding_window_isc(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_identical_signals_high_isc(self):
        """Identical items → mean-field SW ISC near 1.0."""
        rng = np.random.default_rng(72)
        signal = rng.normal(size=(1, 3, 200))
        data = np.repeat(signal, 4, axis=0)
        isc_tc, _ = compute_mean_field_sliding_window_isc(
            data, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        np.testing.assert_allclose(isc_tc, 1.0, atol=1e-10)

    def test_scalar_timecourse(self):
        """Mean-field ISC returns 1D array (scalar per window)."""
        rng = np.random.default_rng(73)
        data = rng.normal(size=(3, 5, 500))
        isc_tc, _ = compute_mean_field_sliding_window_isc(
            data, window_sec=1.0, step_sec=0.5, sfreq=100.0
        )
        assert isc_tc.ndim == 1
