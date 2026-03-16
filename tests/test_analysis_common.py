"""
Tests for scripts/analysis_common.py — argument parsing and workflow helpers.
"""

import argparse

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import add_common_arguments, run_wavelet_workflow
from src.analysis.data_representations import AnalysisData, DataRepresentation


class TestAddCommonArguments:
    """Test that add_common_arguments registers all expected CLI parameters."""

    @pytest.fixture
    def parser(self):
        p = argparse.ArgumentParser()
        add_common_arguments(p)
        return p

    # ── --analysis ────────────────────────────────────────────────

    def test_analysis_defaults_to_isc_and_mean_variance(self, parser):
        args = parser.parse_args([])
        assert set(args.analysis) == {"isc", "mean_variance"}

    def test_analysis_accepts_wavelet_amplitude(self, parser):
        args = parser.parse_args(["--analysis", "wavelet_amplitude"])
        assert "wavelet_amplitude" in args.analysis

    def test_analysis_accepts_wavelet_power(self, parser):
        args = parser.parse_args(["--analysis", "wavelet_power"])
        assert "wavelet_power" in args.analysis

    def test_analysis_accepts_all_choices(self, parser):
        args = parser.parse_args(
            [
                "--analysis",
                "isc",
                "mean_variance",
                "wavelet_amplitude",
                "wavelet_power",
            ]
        )
        assert set(args.analysis) == {
            "isc",
            "mean_variance",
            "wavelet_amplitude",
            "wavelet_power",
        }

    def test_analysis_rejects_invalid_choice(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--analysis", "invalid_choice"])

    # ── --n_jobs ──────────────────────────────────────────────────

    def test_n_jobs_default_is_one(self, parser):
        args = parser.parse_args([])
        assert args.n_jobs == 1

    def test_n_jobs_accepts_minus_one(self, parser):
        args = parser.parse_args(["--n_jobs", "-1"])
        assert args.n_jobs == -1

    def test_n_jobs_accepts_positive_int(self, parser):
        args = parser.parse_args(["--n_jobs", "4"])
        assert args.n_jobs == 4

    # ── wavelet frequency parameters ─────────────────────────────

    def test_wavelet_freq_min_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_freq_min == pytest.approx(1.0)

    def test_wavelet_freq_max_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_freq_max == pytest.approx(40.0)

    def test_wavelet_n_freqs_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_n_freqs == 20

    def test_wavelet_freq_min_custom(self, parser):
        args = parser.parse_args(["--wavelet_freq_min", "4.0"])
        assert args.wavelet_freq_min == pytest.approx(4.0)

    def test_wavelet_freq_max_custom(self, parser):
        args = parser.parse_args(["--wavelet_freq_max", "100.0"])
        assert args.wavelet_freq_max == pytest.approx(100.0)

    def test_wavelet_n_freqs_custom(self, parser):
        args = parser.parse_args(["--wavelet_n_freqs", "50"])
        assert args.wavelet_n_freqs == 50

    def test_wavelet_freqs_linspace_construction(self, parser):
        """Check that wavelet_freq_{min,max,n_freqs} produce a valid linspace."""
        args = parser.parse_args(
            [
                "--wavelet_freq_min", "1.0",
                "--wavelet_freq_max", "40.0",
                "--wavelet_n_freqs", "20",
            ]
        )
        freqs = np.linspace(args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs)
        assert freqs.shape == (20,)
        assert freqs[0] == pytest.approx(1.0)
        assert freqs[-1] == pytest.approx(40.0)

    # ── existing parameters still present ────────────────────────

    def test_window_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.window_sec == pytest.approx(5.0)

    def test_step_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.step_sec == pytest.approx(2.5)

    def test_isc_threshold_default(self, parser):
        args = parser.parse_args([])
        assert args.isc_threshold == pytest.approx(0.035)


class TestRunWaveletWorkflowValidation:
    """Test input validation in run_wavelet_workflow."""

    @pytest.fixture
    def sample_datasets(self, tmp_path):
        rng = np.random.default_rng(0)
        return {
            "TEST": AnalysisData(
                data=rng.normal(size=(3, 4, 500)),
                sfreq=250.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="TEST",
            )
        }

    def test_invalid_representation_raises(self, sample_datasets, tmp_path):
        freqs = np.linspace(4.0, 30.0, 5)
        with pytest.raises(ValueError, match="representation must be"):
            run_wavelet_workflow(
                sample_datasets,
                analyzers={},
                representation="invalid",
                freqs=freqs,
                save_dir=tmp_path / "out",
            )
