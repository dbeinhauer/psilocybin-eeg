"""
Tests for src/analysis/isc.py — Inter-Subject Correlation functions.

All ISC functions operate on 3D numpy arrays: (n_items, n_features, n_samples).
"""

import pytest
import numpy as np
from scipy.stats import pearsonr

from src.analysis.isc import (
    FREQUENCY_BANDS,
    compute_loo_isc,
    compute_pairwise_isc,
    compute_pairwise_isc_per_feature,
    compute_sliding_window_isc,
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
