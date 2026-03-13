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

    def test_creates_figure(self):
        rng = np.random.default_rng(45)
        signals = [rng.normal(size=1000) for _ in range(3)]
        # Just verify it doesn't error; we won't check visual output
        DatasetPlotter.plot_signal_overlap(
            signals,
            sfreq=250.0,
            t_start=0,
            time_duration=2,
            save_fig="/tmp/test_overlap.png",
        )
        plt.close("all")
