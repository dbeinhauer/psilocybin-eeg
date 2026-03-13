"""
Tests for the mean & variance computation functions in ``src.analysis.isc``.
"""

import numpy as np
import pytest

from src.analysis.isc import (
    compute_mean_variance,
    compute_sliding_window_mean_variance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """3 items × 2 features × 100 samples of known values."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 2, 100))


@pytest.fixture
def constant_data():
    """Data where every item/feature/sample is constant (=5)."""
    return np.full((4, 3, 200), 5.0)


# ---------------------------------------------------------------------------
# compute_mean_variance
# ---------------------------------------------------------------------------


class TestComputeMeanVariance:
    def test_output_shapes(self, simple_data):
        mean_f, var_f = compute_mean_variance(simple_data)
        n_features = simple_data.shape[1]
        assert mean_f.shape == (n_features,)
        assert var_f.shape == (n_features,)

    def test_constant_data(self, constant_data):
        """Constant signal ⇒ variance is 0 and mean is the constant."""
        mean_f, var_f = compute_mean_variance(constant_data)
        np.testing.assert_allclose(mean_f, 5.0)
        np.testing.assert_allclose(var_f, 0.0)

    def test_known_values(self):
        """Manually verify with a small hand-crafted array."""
        # 2 items × 1 feature × 4 samples
        data = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0, 6.0]],
            ]
        )
        mean_f, var_f = compute_mean_variance(data)
        # mean across items: [2, 3, 4, 5]
        # temporal mean = 3.5, var = 1.25
        np.testing.assert_allclose(mean_f, [3.5])
        np.testing.assert_allclose(var_f, [1.25])


# ---------------------------------------------------------------------------
# compute_sliding_window_mean_variance
# ---------------------------------------------------------------------------


class TestComputeSlidingWindowMeanVariance:
    def test_output_shapes(self, simple_data):
        mean_tc, var_tc, times = compute_sliding_window_mean_variance(
            simple_data, window_sec=0.2, step_sec=0.1, sfreq=100.0
        )
        n_features = simple_data.shape[1]
        assert mean_tc.ndim == 2
        assert var_tc.ndim == 2
        assert times.ndim == 1
        assert mean_tc.shape[0] == times.shape[0]
        assert var_tc.shape[0] == times.shape[0]
        assert mean_tc.shape[1] == n_features
        assert var_tc.shape[1] == n_features

    def test_constant_data(self, constant_data):
        """Constant signal: every window mean=5, var=0."""
        mean_tc, var_tc, times = compute_sliding_window_mean_variance(
            constant_data, window_sec=0.1, step_sec=0.05, sfreq=100.0
        )
        np.testing.assert_allclose(mean_tc, 5.0)
        np.testing.assert_allclose(var_tc, 0.0)
        assert len(times) > 0

    def test_window_times_are_centred(self, simple_data):
        """Window centre times should sit at the expected positions."""
        sfreq = 100.0
        window_sec = 0.2
        step_sec = 0.1
        win_samples = int(round(window_sec * sfreq))
        step_samples = int(round(step_sec * sfreq))
        n_samples = simple_data.shape[2]
        starts = np.arange(0, n_samples - win_samples + 1, step_samples)
        expected_times = (starts + win_samples / 2) / sfreq

        _, _, times = compute_sliding_window_mean_variance(
            simple_data, window_sec=window_sec, step_sec=step_sec, sfreq=sfreq
        )
        np.testing.assert_allclose(times, expected_times)

    def test_single_window_matches_global(self):
        """When window covers the entire signal, result matches global."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((5, 4, 50))
        sfreq = 50.0  # 1 second of data

        mean_tc, var_tc, times = compute_sliding_window_mean_variance(
            data, window_sec=1.0, step_sec=1.0, sfreq=sfreq
        )
        mean_f, var_f = compute_mean_variance(data)

        assert mean_tc.shape[0] == 1
        np.testing.assert_allclose(mean_tc[0], mean_f, atol=1e-12)
        np.testing.assert_allclose(var_tc[0], var_f, atol=1e-12)
