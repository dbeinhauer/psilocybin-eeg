"""
Tests for the intersubject mean-variance analysis functions in
``src.analysis.mean_variance``.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.mean_variance import (
    compute_intersubject_stats,
    compute_windowed_stats,
    compute_band_intersubject_stats,
    compute_pairwise_isc_matrices,
    FREQUENCY_BANDS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """3 subjects × 2 channels × 100 time points of random values."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 2, 100))


@pytest.fixture
def constant_data():
    """All values equal 5 — variance across subjects must be 0."""
    return np.full((4, 3, 200), 5.0)


# ---------------------------------------------------------------------------
# compute_intersubject_stats
# ---------------------------------------------------------------------------


class TestComputeIntersubjectStats:
    def test_output_keys(self, simple_data):
        stats = compute_intersubject_stats(simple_data)
        assert set(stats.keys()) == {
            "inter_var",
            "inter_mean",
            "mean_t",
            "var_t",
            "std_t",
            "mean_over_ch",
        }

    def test_output_shapes(self, simple_data):
        n_subjects, n_channels, n_times = simple_data.shape
        stats = compute_intersubject_stats(simple_data)
        assert stats["inter_var"].shape == (n_channels, n_times)
        assert stats["inter_mean"].shape == (n_channels, n_times)
        assert stats["mean_t"].shape == (n_times,)
        assert stats["var_t"].shape == (n_times,)
        assert stats["std_t"].shape == (n_times,)
        assert stats["mean_over_ch"].shape == (n_subjects, n_times)

    def test_constant_data_zero_variance(self, constant_data):
        """Constant signal → intersubject variance is 0 everywhere."""
        stats = compute_intersubject_stats(constant_data)
        np.testing.assert_allclose(stats["inter_var"], 0.0)
        np.testing.assert_allclose(stats["var_t"], 0.0)
        np.testing.assert_allclose(stats["std_t"], 0.0)

    def test_constant_data_correct_mean(self, constant_data):
        stats = compute_intersubject_stats(constant_data)
        np.testing.assert_allclose(stats["inter_mean"], 5.0)
        np.testing.assert_allclose(stats["mean_t"], 5.0)

    def test_known_values(self):
        """Hand-crafted 2×1×4 array — verify against manual calculation."""
        # 2 subjects × 1 channel × 4 time points
        data = np.array(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0, 6.0]],
            ]
        )
        stats = compute_intersubject_stats(data)

        # inter_mean = [[2, 3, 4, 5]]  →  mean_t = [2, 3, 4, 5] (channel avg = itself)
        np.testing.assert_allclose(stats["inter_mean"], [[2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_allclose(stats["mean_t"], [2.0, 3.0, 4.0, 5.0])

        # inter_var at each time = var([1,3], [2,4], [3,5], [4,6]) = 1.0 each
        np.testing.assert_allclose(stats["inter_var"], [[1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_allclose(stats["var_t"], [1.0, 1.0, 1.0, 1.0])

    def test_std_t_is_sqrt_var_t(self, simple_data):
        stats = compute_intersubject_stats(simple_data)
        np.testing.assert_allclose(stats["std_t"], np.sqrt(stats["var_t"]))

    def test_raises_on_2d_input(self):
        with pytest.raises(ValueError, match="3D"):
            compute_intersubject_stats(np.ones((3, 100)))

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="3D"):
            compute_intersubject_stats(np.ones(100))

    def test_mean_over_ch_is_channel_mean(self, simple_data):
        """mean_over_ch[s] should equal simple_data[s].mean(axis=0)."""
        stats = compute_intersubject_stats(simple_data)
        for s in range(simple_data.shape[0]):
            np.testing.assert_allclose(
                stats["mean_over_ch"][s], simple_data[s].mean(axis=0)
            )


# ---------------------------------------------------------------------------
# compute_windowed_stats
# ---------------------------------------------------------------------------


class TestComputeWindowedStats:
    def _make_stats(self, data: np.ndarray) -> dict[str, np.ndarray]:
        return compute_intersubject_stats(data)

    def test_output_is_dataframe(self, simple_data):
        stats = self._make_stats(simple_data)
        n_times = simple_data.shape[2]
        df = compute_windowed_stats(stats, n_times=n_times, sfreq=100.0, window_sec=0.2)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, simple_data):
        stats = self._make_stats(simple_data)
        n_times = simple_data.shape[2]
        df = compute_windowed_stats(stats, n_times=n_times, sfreq=100.0, window_sec=0.2)
        expected = {
            "window",
            "center",
            "t_start",
            "t_end",
            "mean_signal",
            "var_signal",
            "mean_variance",
            "sync_candidate",
        }
        assert set(df.columns) >= expected

    def test_n_windows(self, simple_data):
        n_times = simple_data.shape[2]  # 100
        sfreq = 100.0
        window_sec = 0.2  # 20 samples
        stats = self._make_stats(simple_data)
        # Non-overlapping (step = window): 100 // 20 = 5 windows
        df = compute_windowed_stats(
            stats,
            n_times=n_times,
            sfreq=sfreq,
            window_sec=window_sec,
            step_sec=window_sec,
        )
        expected_n = n_times // int(window_sec * sfreq)
        assert len(df) == expected_n

    def test_n_windows_overlapping(self, simple_data):
        """50% overlap doubles the window count compared to non-overlapping."""
        n_times = simple_data.shape[2]  # 100
        sfreq = 100.0
        window_sec = 0.2  # 20 samples, step = 10 samples → 9 windows
        stats = self._make_stats(simple_data)
        df = compute_windowed_stats(
            stats, n_times=n_times, sfreq=sfreq, window_sec=window_sec
        )
        # starts: 0, 10, 20, 30, 40, 50, 60, 70, 80 → 9 windows (start+20 ≤ 100)
        step_samples = int(round(window_sec / 2 * sfreq))
        win_samples = int(round(window_sec * sfreq))
        expected_starts = np.arange(0, n_times - win_samples + 1, step_samples)
        assert len(df) == len(expected_starts)

    def test_sync_candidate_is_bool(self, simple_data):
        stats = self._make_stats(simple_data)
        n_times = simple_data.shape[2]
        df = compute_windowed_stats(stats, n_times=n_times, sfreq=100.0, window_sec=0.5)
        assert df["sync_candidate"].dtype == bool

    def test_sync_percentile_threshold(self, simple_data):
        """Exactly sync_percentile % of windows below threshold on average."""
        rng = np.random.default_rng(7)
        large_data = rng.standard_normal((4, 3, 1000))
        stats = self._make_stats(large_data)
        for pct in (10.0, 25.0, 50.0):
            df = compute_windowed_stats(
                stats,
                n_times=1000,
                sfreq=100.0,
                window_sec=0.2,
                sync_percentile=pct,
            )
            threshold = np.percentile(stats["var_t"], pct)
            # All sync_candidate windows must be strictly below threshold
            assert (df.loc[df["sync_candidate"], "mean_variance"] < threshold).all()

    def test_raises_on_zero_samples(self, simple_data):
        stats = self._make_stats(simple_data)
        with pytest.raises(ValueError, match="samples"):
            compute_windowed_stats(stats, n_times=100, sfreq=1.0, window_sec=0.001)

    def test_raises_on_nonpositive_window_sec(self, simple_data):
        """window_sec <= 0 must raise ValueError."""
        stats = self._make_stats(simple_data)
        with pytest.raises(ValueError, match="positive"):
            compute_windowed_stats(stats, n_times=100, sfreq=100.0, window_sec=0.0)
        with pytest.raises(ValueError, match="positive"):
            compute_windowed_stats(stats, n_times=100, sfreq=100.0, window_sec=-1.0)

    def test_raises_on_nonpositive_step_sec(self, simple_data):
        """step_sec <= 0 must raise ValueError."""
        stats = self._make_stats(simple_data)
        with pytest.raises(ValueError, match="positive"):
            compute_windowed_stats(
                stats, n_times=100, sfreq=100.0, window_sec=0.2, step_sec=0.0
            )
        with pytest.raises(ValueError, match="positive"):
            compute_windowed_stats(
                stats, n_times=100, sfreq=100.0, window_sec=0.2, step_sec=-0.1
            )

    def test_raises_when_window_longer_than_recording(self, simple_data):
        """window_sec longer than recording duration must raise ValueError."""
        stats = self._make_stats(simple_data)
        with pytest.raises(ValueError, match="longer than"):
            # 100 samples at 100 Hz = 1 s recording; window of 5 s is too long
            compute_windowed_stats(stats, n_times=100, sfreq=100.0, window_sec=5.0)

    def test_window_indices_non_overlapping(self, simple_data):
        """t_start should advance by window_sec when step_sec == window_sec."""
        stats = self._make_stats(simple_data)
        n_times = simple_data.shape[2]
        sfreq = 100.0
        window_sec = 0.2
        df = compute_windowed_stats(
            stats,
            n_times=n_times,
            sfreq=sfreq,
            window_sec=window_sec,
            step_sec=window_sec,
        )
        t_starts = df["t_start"].values
        for i in range(1, len(t_starts)):
            np.testing.assert_allclose(
                t_starts[i], t_starts[i - 1] + window_sec, atol=1e-9
            )

    def test_window_indices_overlapping(self, simple_data):
        """t_start should advance by step_sec for 50% overlapping windows."""
        stats = self._make_stats(simple_data)
        n_times = simple_data.shape[2]
        sfreq = 100.0
        window_sec = 0.2
        step_sec = 0.1  # 50% overlap
        df = compute_windowed_stats(
            stats,
            n_times=n_times,
            sfreq=sfreq,
            window_sec=window_sec,
            step_sec=step_sec,
        )
        t_starts = df["t_start"].values
        for i in range(1, len(t_starts)):
            np.testing.assert_allclose(
                t_starts[i], t_starts[i - 1] + step_sec, atol=1e-9
            )

    def test_constant_data_all_zero_mean_variance(self, constant_data):
        """Constant signal → every window has mean_variance = 0."""
        stats = self._make_stats(constant_data)
        df = compute_windowed_stats(
            stats, n_times=constant_data.shape[2], sfreq=100.0, window_sec=0.2
        )
        np.testing.assert_allclose(df["mean_variance"].values, 0.0)


# ---------------------------------------------------------------------------
# compute_band_intersubject_stats — pure function simulation without MNE
# ---------------------------------------------------------------------------


class _StubFilteredData:
    """Minimal stub that mimics AnalysisData.filter_to_band() return value."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _StubAnalysisData:
    """Stub AnalysisData object whose filter_to_band ignores frequency arguments."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def filter_to_band(self, l_freq: float, h_freq: float) -> _StubFilteredData:  # noqa: ARG002
        return _StubFilteredData(self._data)


class TestComputeBandIntersubjectStats:
    """Tests for compute_band_intersubject_stats using a lightweight stub that
    avoids real MNE bandpass filtering.
    """

    def test_all_bands_present(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 8, 200))
        ad = _StubAnalysisData(data)
        result = compute_band_intersubject_stats(ad, FREQUENCY_BANDS)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())

    def test_per_band_shapes(self):
        rng = np.random.default_rng(1)
        n_subjects, n_channels, n_times = 3, 6, 150
        data = rng.standard_normal((n_subjects, n_channels, n_times))
        bands = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0)}
        ad = _StubAnalysisData(data)
        result = compute_band_intersubject_stats(ad, bands)
        for band, stats in result.items():
            assert stats["inter_var"].shape == (n_channels, n_times), band
            assert stats["var_t"].shape == (n_times,), band
            assert stats["mean_over_ch"].shape == (n_subjects, n_times), band

    def test_constant_data_zero_variance(self):
        bands = {"theta": (4.0, 8.0), "beta": (13.0, 30.0)}
        data = np.full((2, 4, 100), 3.0)
        ad = _StubAnalysisData(data)
        result = compute_band_intersubject_stats(ad, bands)
        for band, stats in result.items():
            np.testing.assert_allclose(
                stats["var_t"], 0.0, err_msg=f"var_t non-zero for {band}"
            )

    def test_default_bands_used_when_none(self):
        """When bands=None, the function should use FREQUENCY_BANDS."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((3, 4, 100))
        ad = _StubAnalysisData(data)
        result = compute_band_intersubject_stats(ad, None)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())

    def test_stats_dict_keys(self):
        rng = np.random.default_rng(3)
        data = rng.standard_normal((2, 4, 80))
        ad = _StubAnalysisData(data)
        result = compute_band_intersubject_stats(ad, {"delta": (1.0, 4.0)})
        expected_keys = {
            "inter_var",
            "inter_mean",
            "mean_t",
            "var_t",
            "std_t",
            "mean_over_ch",
        }
        assert set(result["delta"].keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_pairwise_isc_matrices
# ---------------------------------------------------------------------------


class TestComputePairwiseIscMatrices:
    def test_output_shape(self):
        rng = np.random.default_rng(10)
        n_subjects, n_channels, n_times = 5, 4, 200
        data = rng.standard_normal((n_subjects, n_channels, n_times))
        band_data = {"delta": data, "alpha": data}
        result = compute_pairwise_isc_matrices(band_data)
        for band, mat in result.items():
            assert mat.shape == (n_subjects, n_subjects), band

    def test_keys_match_input(self):
        rng = np.random.default_rng(11)
        data = rng.standard_normal((3, 4, 100))
        band_data = {"delta": data, "gamma": data}
        result = compute_pairwise_isc_matrices(band_data)
        assert set(result.keys()) == {"delta", "gamma"}

    def test_symmetric(self):
        rng = np.random.default_rng(12)
        data = rng.standard_normal((4, 8, 300))
        # z-score data so the mean-product equals Pearson r
        from scipy.stats import zscore

        data = zscore(data, axis=2)
        result = compute_pairwise_isc_matrices({"alpha": data})
        mat = result["alpha"]
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)

    def test_identical_subjects_max_value(self):
        """Two identical subjects → ISC = 1.0 (z-scored data)."""
        rng = np.random.default_rng(13)
        signal = rng.standard_normal((1, 4, 200))
        from scipy.stats import zscore

        signal = zscore(signal, axis=2)
        # Stack the same signal for both subjects
        data = np.concatenate([signal, signal], axis=0)
        result = compute_pairwise_isc_matrices({"beta": data})
        # Both diagonal and off-diagonal should be 1 (identical signals)
        np.testing.assert_allclose(result["beta"][0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["beta"][0, 1], 1.0, atol=1e-10)

    def test_all_bands_from_frequency_bands(self):
        rng = np.random.default_rng(14)
        n_subjects = 3
        data = rng.standard_normal((n_subjects, 4, 100))
        band_data = {band: data for band in FREQUENCY_BANDS}
        result = compute_pairwise_isc_matrices(band_data)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())
        for mat in result.values():
            assert mat.shape == (n_subjects, n_subjects)
