"""
Tests for the mean & variance computation functions in ``src.analysis.isc``
and band-specific workflows.
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


# ---------------------------------------------------------------------------
# Band mean & variance — per-band application of compute_mean_variance
# ---------------------------------------------------------------------------


class TestBandMeanVarianceWorkflow:
    """Test that compute_mean_variance applied to multiple band-filtered
    arrays gives the expected structure, without importing the analyzer.

    This mirrors how TestBandIsc works in test_isc.py: we apply the pure
    function independently to each 'band' and verify the aggregate dict.
    """

    def _simulate_band_results(
        self,
        data: np.ndarray,
        bands: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run compute_mean_variance on identical data for each band (no MNE)."""
        return {name: compute_mean_variance(data) for name in bands}

    def test_all_bands_present(self):
        """Result has one entry per band."""
        from src.analysis.isc import FREQUENCY_BANDS

        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 8, 200))
        result = self._simulate_band_results(data, FREQUENCY_BANDS)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())

    def test_per_band_shapes(self):
        """Each band entry contains (n_features,) arrays."""
        rng = np.random.default_rng(1)
        n_features = 6
        data = rng.standard_normal((3, n_features, 150))
        bands = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0), "gamma": (30.0, 70.0)}
        result = self._simulate_band_results(data, bands)
        for band, (mean_f, var_f) in result.items():
            assert mean_f.shape == (n_features,), f"mean shape wrong for {band}"
            assert var_f.shape == (n_features,), f"var shape wrong for {band}"

    def test_constant_data_per_band(self):
        """Constant signal gives var=0, mean=constant for every band."""
        bands = {"theta": (4.0, 8.0), "beta": (13.0, 30.0)}
        data = np.full((2, 4, 100), 3.0)
        result = self._simulate_band_results(data, bands)
        for band, (mean_f, var_f) in result.items():
            np.testing.assert_allclose(mean_f, 3.0, err_msg=f"mean wrong for {band}")
            np.testing.assert_allclose(var_f, 0.0, err_msg=f"var wrong for {band}")

    def test_independent_bands(self):
        """Different synthetic band arrays produce different results."""
        rng = np.random.default_rng(7)
        n_features = 4
        # Simulate two different 'filtered' signals for two bands
        data_alpha = rng.standard_normal((3, n_features, 200))
        data_beta = rng.standard_normal((3, n_features, 200)) * 10
        result = {
            "alpha": compute_mean_variance(data_alpha),
            "beta": compute_mean_variance(data_beta),
        }
        # Beta has 10x larger scale so its variance should be much larger
        assert result["beta"][1].mean() > result["alpha"][1].mean()


# ---------------------------------------------------------------------------
# Band sliding-window mean & variance workflow
# ---------------------------------------------------------------------------


class TestBandSlidingWindowMeanVarianceWorkflow:
    """Test compute_sliding_window_mean_variance applied per band."""

    def _simulate_band_sw_results(
        self,
        data: np.ndarray,
        bands: dict[str, tuple[float, float]],
        window_sec: float = 0.2,
        step_sec: float = 0.1,
        sfreq: float = 100.0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {
            name: compute_sliding_window_mean_variance(
                data, window_sec=window_sec, step_sec=step_sec, sfreq=sfreq
            )
            for name in bands
        }

    def test_all_bands_present(self):
        from src.analysis.isc import FREQUENCY_BANDS

        rng = np.random.default_rng(5)
        data = rng.standard_normal((3, 4, 200))
        result = self._simulate_band_sw_results(data, FREQUENCY_BANDS)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())

    def test_per_band_shapes(self):
        """Each band produces (n_windows, n_features) arrays and (n_windows,) times."""
        rng = np.random.default_rng(6)
        n_features = 5
        data = rng.standard_normal((3, n_features, 300))
        bands = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0)}
        result = self._simulate_band_sw_results(
            data, bands, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        for band, (mean_tc, var_tc, times) in result.items():
            assert mean_tc.ndim == 2, f"mean_tc ndim wrong for {band}"
            assert var_tc.ndim == 2, f"var_tc ndim wrong for {band}"
            assert times.ndim == 1, f"times ndim wrong for {band}"
            assert mean_tc.shape == var_tc.shape
            assert mean_tc.shape[0] == times.shape[0]
            assert mean_tc.shape[1] == n_features

    def test_constant_data_per_band(self):
        """Constant signal: every window, every band → mean=const, var=0."""
        bands = {"theta": (4.0, 8.0), "gamma": (30.0, 70.0)}
        data = np.full((2, 3, 200), 7.0)
        result = self._simulate_band_sw_results(data, bands)
        for band, (mean_tc, var_tc, _) in result.items():
            np.testing.assert_allclose(
                mean_tc, 7.0, err_msg=f"mean_tc wrong for {band}"
            )
            np.testing.assert_allclose(
                var_tc, 0.0, err_msg=f"var_tc wrong for {band}"
            )

    def test_consistent_times_across_bands(self):
        """All bands computed from the same data → identical window_times."""
        rng = np.random.default_rng(8)
        data = rng.standard_normal((4, 6, 400))
        bands = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
        result = self._simulate_band_sw_results(
            data, bands, window_sec=0.5, step_sec=0.25, sfreq=100.0
        )
        band_list = list(bands.keys())
        times_ref = result[band_list[0]][2]
        for band in band_list[1:]:
            np.testing.assert_array_equal(
                result[band][2], times_ref, err_msg=f"times differ for {band}"
            )
