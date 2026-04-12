"""
Tests for src/visualization/ — Plotting utility functions (non-visual).

We test the computation logic and data handling without requiring an active
display. matplotlib is set to the 'Agg' backend so all tests run headless.
"""

import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.preprocessing_plots import DatasetPlotter
from src.visualization.isc_plots import (
    plot_loo_isc_pearson_vs_spearman,
    plot_pairwise_isc_pearson_vs_spearman,
    plot_multiscale_sliding_window_isc,
    plot_band_loo_isc_pearson_vs_spearman,
    plot_band_pairwise_isc_pearson_vs_spearman,
    plot_band_multiscale_sliding_window_isc,
)
from src.visualization.wavelet_plots import compute_itpc, compute_phase_band_loo_iscs
from src.analysis.isc import FREQUENCY_BANDS
from src.definitions.constants import ProjectPaths
from pathlib import Path


class TestComputeTagSignalCorrelationMatrix:
    """Test the correlation matrix computation (no plot output needed)."""

    def test_identical_signals_perfect_correlation(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 200))
        signals = [sig.copy(), sig.copy(), sig.copy()]
        corr = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        np.testing.assert_allclose(corr, 1.0, atol=1e-10)

    def test_shape(self):
        rng = np.random.default_rng(42)
        signals = [rng.normal(size=100) for _ in range(4)]
        corr = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(43)
        signals = [rng.normal(size=100) for _ in range(3)]
        corr = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_symmetric(self):
        rng = np.random.default_rng(44)
        signals = [rng.normal(size=100) for _ in range(3)]
        corr = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        np.testing.assert_allclose(corr, corr.T)


class TestGetPlotPath:
    """Test plot path construction."""

    def test_custom_full_path(self):
        path = DatasetPlotter.get_plot_path(
            "test.png", "power_spectrum", "after_ica", "/custom/path/plot.png"
        )
        assert path == Path("/custom/path/plot.png")

    def test_default_path_construction(self):
        path = DatasetPlotter.get_plot_path(
            "test.png", "power_spectrum", "after_ica", ""
        )
        expected = ProjectPaths.PLOTS_PATH / "power_spectrum" / "after_ica" / "test.png"
        assert path == expected


class TestPlotSignalOverlap:
    """Test that signal overlap plotting produces a figure."""

    def test_creates_figure(self, tmp_path):
        rng = np.random.default_rng(45)
        signals = [rng.normal(size=1000) for _ in range(3)]
        # Just verify it doesn't error; we won't check visual output
        DatasetPlotter.plot_signal_overlap(
            signals,
            sfreq=250.0,
            t_start=0,
            time_duration=2,
            save_fig=str(tmp_path / "test_overlap.png"),
        )
        plt.close("all")


# ---------------------------------------------------------------------------
# Helpers for ISC plot tests
# ---------------------------------------------------------------------------

_N_SUBJECTS = 4
_N_CHANNELS = 8
_N_TIMES = 500
_SFREQ = 50.0


@pytest.fixture()
def isc_rng():
    return np.random.default_rng(0)


@pytest.fixture()
def fake_loo(isc_rng):
    """(n_subjects, n_channels) LOO-ISC array."""
    return isc_rng.uniform(-0.1, 0.2, (_N_SUBJECTS, _N_CHANNELS)).astype(np.float32)


@pytest.fixture()
def fake_mean_isc(isc_rng):
    """(n_channels,) mean LOO-ISC array."""
    return isc_rng.uniform(-0.1, 0.2, (_N_CHANNELS,)).astype(np.float32)


@pytest.fixture()
def fake_pair(isc_rng):
    """(n_subjects, n_subjects) pairwise ISC matrix (symmetric)."""
    mat = isc_rng.uniform(0.0, 0.1, (_N_SUBJECTS, _N_SUBJECTS)).astype(np.float32)
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    return mat


@pytest.fixture()
def fake_sw(isc_rng):
    """Sliding-window ISC: (n_windows, n_channels) + times (n_windows,)."""
    n_windows = 20
    isc_tc = isc_rng.uniform(-0.05, 0.1, (n_windows, _N_CHANNELS)).astype(np.float32)
    times = np.arange(n_windows, dtype=float) * (_N_TIMES / _SFREQ / n_windows)
    return isc_tc, times


# ---------------------------------------------------------------------------
# Tests for new ISC plot functions
# ---------------------------------------------------------------------------


class TestPlotLooIscPearsonVsSpearman:
    """plot_loo_isc_pearson_vs_spearman returns two Figure objects."""

    def test_returns_two_figures(self, fake_loo, fake_mean_isc):
        result = plot_loo_isc_pearson_vs_spearman(
            "TEST",
            fake_loo,
            fake_mean_isc,
            fake_loo[:, :4],  # subsampled Spearman
            fake_mean_isc[:4],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        for fig in result:
            assert fig is not None
        plt.close("all")

    def test_saves_to_path(self, tmp_path, fake_loo, fake_mean_isc):
        hist_path = tmp_path / "hist.png"
        violin_path = tmp_path / "violin.png"
        plot_loo_isc_pearson_vs_spearman(
            "TEST",
            fake_loo,
            fake_mean_isc,
            fake_loo,
            fake_mean_isc,
            save_path_hist=hist_path,
            save_path_violin=violin_path,
        )
        assert hist_path.exists()
        assert violin_path.exists()
        plt.close("all")


class TestPlotPairwiseIscPearsonVsSpearman:
    """plot_pairwise_isc_pearson_vs_spearman returns three Figure objects."""

    def test_returns_three_figures(self, fake_pair):
        result = plot_pairwise_isc_pearson_vs_spearman("TEST", fake_pair, fake_pair)
        assert isinstance(result, tuple)
        assert len(result) == 3
        plt.close("all")

    def test_saves_to_paths(self, tmp_path, fake_pair):
        p_hm = tmp_path / "heatmaps.png"
        p_subj = tmp_path / "per_subject.png"
        p_dist = tmp_path / "distribution.png"
        plot_pairwise_isc_pearson_vs_spearman(
            "TEST",
            fake_pair,
            fake_pair,
            save_path_heatmaps=p_hm,
            save_path_per_subject=p_subj,
            save_path_distribution=p_dist,
        )
        assert p_hm.exists()
        assert p_subj.exists()
        assert p_dist.exists()
        plt.close("all")


class TestPlotMultiscaleSlidingWindowIsc:
    """plot_multiscale_sliding_window_isc returns three Figure objects."""

    def test_returns_three_figures(self, fake_sw):
        isc_fine, times_fine = fake_sw
        result = plot_multiscale_sliding_window_isc(
            "TEST",
            isc_fine,
            times_fine,
            isc_fine,
            times_fine,
            isc_fine,
            times_fine,
            isc_fine,
            _SFREQ,
            _N_TIMES,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        plt.close("all")

    def test_saves_bar_figure(self, tmp_path, fake_sw):
        isc_tc, times = fake_sw
        bar_path = tmp_path / "bar.png"
        plot_multiscale_sliding_window_isc(
            "TEST",
            isc_tc,
            times,
            isc_tc,
            times,
            isc_tc,
            times,
            isc_tc,
            _SFREQ,
            _N_TIMES,
            save_path_bar=bar_path,
        )
        assert bar_path.exists()
        plt.close("all")


class TestPlotBandLooIscPearsonVsSpearman:
    """plot_band_loo_isc_pearson_vs_spearman returns a dict keyed by band."""

    def test_returns_dict_with_all_bands(self, fake_loo, fake_mean_isc):
        band_iscs = {band: (fake_loo, fake_mean_isc) for band in FREQUENCY_BANDS}
        band_iscs_sp = {
            band: (fake_loo[:, :4], fake_mean_isc[:4]) for band in FREQUENCY_BANDS
        }
        result = plot_band_loo_isc_pearson_vs_spearman("TEST", band_iscs, band_iscs_sp)
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())
        for band, (fig_hist, fig_violin) in result.items():
            assert fig_hist is not None
            assert fig_violin is not None
        plt.close("all")

    def test_saves_to_directory(self, tmp_path, fake_loo, fake_mean_isc):
        band_iscs = {band: (fake_loo, fake_mean_isc) for band in FREQUENCY_BANDS}
        band_iscs_sp = {band: (fake_loo, fake_mean_isc) for band in FREQUENCY_BANDS}
        plot_band_loo_isc_pearson_vs_spearman(
            "TEST", band_iscs, band_iscs_sp, save_path_dir=tmp_path
        )
        # Each band should produce two files
        for band in FREQUENCY_BANDS:
            assert (tmp_path / f"loo_isc_distribution_{band}_TEST.png").exists()
            assert (tmp_path / f"loo_isc_per_subject_{band}_TEST.png").exists()
        plt.close("all")


class TestPlotBandPairwiseIscPearsonVsSpearman:
    """plot_band_pairwise_isc_pearson_vs_spearman returns a dict keyed by band."""

    def test_returns_dict_with_all_bands(self, fake_pair):
        band_pair = {band: fake_pair for band in FREQUENCY_BANDS}
        result = plot_band_pairwise_isc_pearson_vs_spearman(
            "TEST", band_pair, band_pair
        )
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())
        for band, (fig_hm, fig_subj, fig_dist) in result.items():
            assert fig_hm is not None
            assert fig_subj is not None
            assert fig_dist is not None
        plt.close("all")

    def test_saves_to_directory(self, tmp_path, fake_pair):
        band_pair = {band: fake_pair for band in FREQUENCY_BANDS}
        plot_band_pairwise_isc_pearson_vs_spearman(
            "TEST", band_pair, band_pair, save_path_dir=tmp_path
        )
        for band in FREQUENCY_BANDS:
            assert (tmp_path / f"pairwise_isc_matrix_{band}_TEST.png").exists()
            assert (tmp_path / f"pairwise_isc_per_subject_{band}_TEST.png").exists()
            assert (tmp_path / f"pairwise_isc_distribution_{band}_TEST.png").exists()
        plt.close("all")


class TestPlotBandMultiscaleSlidingWindowIsc:
    """plot_band_multiscale_sliding_window_isc returns a dict keyed by band."""

    def test_returns_dict_with_all_bands(self, fake_sw):
        isc_tc, times = fake_sw
        band_sw = {band: (isc_tc, times) for band in FREQUENCY_BANDS}
        result = plot_band_multiscale_sliding_window_isc(
            "TEST",
            band_sw,
            band_sw,
            band_sw,
            band_sw,
            _SFREQ,
            _N_TIMES,
        )
        assert set(result.keys()) == set(FREQUENCY_BANDS.keys())
        for band, (fig_bar, fig_ov, fig_cmp) in result.items():
            assert fig_bar is not None
            assert fig_ov is not None
            assert fig_cmp is not None
        plt.close("all")

    def test_saves_to_directory(self, tmp_path, fake_sw):
        isc_tc, times = fake_sw
        band_sw = {band: (isc_tc, times) for band in FREQUENCY_BANDS}
        plot_band_multiscale_sliding_window_isc(
            "TEST",
            band_sw,
            band_sw,
            band_sw,
            band_sw,
            _SFREQ,
            _N_TIMES,
            save_path_dir=tmp_path,
        )
        for band in FREQUENCY_BANDS:
            assert (tmp_path / f"sw_isc_bar_{band}_TEST.png").exists()
            assert (tmp_path / f"sw_isc_overlay_{band}_TEST.png").exists()
            assert (tmp_path / f"sw_isc_pearson_vs_spearman_{band}_TEST.png").exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests for wavelet_plots computation helpers
# ---------------------------------------------------------------------------

_N_SUBJECTS_W = 4
_N_CHANNELS_W = 5
_N_FREQS_W = 8
_N_TIMES_W = 50


class TestComputeItpc:
    """compute_itpc: shape, range and deterministic edge cases."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        phase = rng.uniform(
            -np.pi, np.pi, (_N_SUBJECTS_W, _N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W)
        )
        itpc = compute_itpc(phase)
        assert itpc.shape == (_N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W)

    def test_values_in_range(self):
        rng = np.random.default_rng(1)
        phase = rng.uniform(
            -np.pi, np.pi, (_N_SUBJECTS_W, _N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W)
        )
        itpc = compute_itpc(phase)
        assert float(itpc.min()) >= 0.0
        assert float(itpc.max()) <= 1.0 + 1e-9

    def test_perfect_phase_lock_gives_one(self):
        # All subjects have exactly the same phase → ITPC should be 1
        phase = np.full((_N_SUBJECTS_W, _N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W), 0.5)
        itpc = compute_itpc(phase)
        np.testing.assert_allclose(itpc, 1.0, atol=1e-10)

    def test_raises_on_wrong_ndim(self):
        with pytest.raises(ValueError, match="4D"):
            compute_itpc(np.zeros((3, 4, 5)))

    def test_small_deterministic_example(self):
        # With 2 subjects and phases 0 and π, the mean unit vector is 0 → ITPC ≈ 0
        phase = np.array([0.0, np.pi]).reshape(2, 1, 1, 1) * np.ones(
            (2, _N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W)
        )
        itpc = compute_itpc(phase)
        np.testing.assert_allclose(itpc, 0.0, atol=1e-10)


class TestComputePhaseBandLooIscs:
    """compute_phase_band_loo_iscs: shape, band coverage and edge cases."""

    @pytest.fixture()
    def fake_phase_4d(self):
        rng = np.random.default_rng(42)
        return rng.uniform(
            -np.pi, np.pi, (_N_SUBJECTS_W, _N_CHANNELS_W, _N_FREQS_W, _N_TIMES_W)
        ).astype(np.float32)

    @pytest.fixture()
    def freqs(self):
        # 8 evenly-spaced freqs spanning 1–30 Hz (covers delta, theta, alpha, beta)
        return np.linspace(1.0, 30.0, _N_FREQS_W)

    def test_output_bands_present(self, fake_phase_4d, freqs):
        result = compute_phase_band_loo_iscs(fake_phase_4d, freqs)
        # At least some bands should be returned
        assert len(result) > 0
        for mean_loo in result.values():
            assert mean_loo.shape == (_N_CHANNELS_W,)

    def test_all_standard_bands_when_freqs_cover_them(self, fake_phase_4d):
        # Use freqs spanning all standard bands (1–70 Hz)
        freqs_full = np.linspace(1.0, 70.0, 40)
        phase_full = np.random.default_rng(7).uniform(
            -np.pi, np.pi, (_N_SUBJECTS_W, _N_CHANNELS_W, 40, _N_TIMES_W)
        )
        result = compute_phase_band_loo_iscs(phase_full, freqs_full)
        for band in FREQUENCY_BANDS:
            assert band in result, f"Band {band!r} missing from result"
            assert result[band].shape == (_N_CHANNELS_W,)

    def test_raises_on_wrong_ndim(self, freqs):
        with pytest.raises(ValueError, match="4D"):
            compute_phase_band_loo_iscs(np.zeros((3, 4, 5)), freqs)

    def test_empty_when_no_freqs_in_band(self, fake_phase_4d):
        # Very narrow frequency range that matches no standard band
        freqs_narrow = np.array([50.0, 51.0])
        phase_narrow = np.random.default_rng(9).uniform(
            -np.pi, np.pi, (_N_SUBJECTS_W, _N_CHANNELS_W, 2, _N_TIMES_W)
        )
        result = compute_phase_band_loo_iscs(
            phase_narrow,
            freqs_narrow,
            bands={"delta": (1.0, 4.0)},  # no match for 50–51 Hz
        )
        assert result == {}

    def test_custom_band(self, fake_phase_4d, freqs):
        custom_bands = {"my_band": (1.0, 30.0)}
        result = compute_phase_band_loo_iscs(fake_phase_4d, freqs, bands=custom_bands)
        assert "my_band" in result
        assert result["my_band"].shape == (_N_CHANNELS_W,)
