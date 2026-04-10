"""
Tests for the results-store module (``src.analysis.results_store``).

Verifies CSV round-trip (save → load), directory scanning, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.results_store import (
    save_intersubject_timeseries,
    save_windowed_stats,
    save_loo_isc,
    save_pairwise_isc,
    load_intersubject_timeseries,
    load_windowed_stats,
    load_loo_isc,
    load_pairwise_isc,
    scan_results_db,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_stats():
    """Minimal intersubject stats dict (3 channels × 100 time points)."""
    rng = np.random.default_rng(0)
    n_times = 100
    return {
        "mean_t": rng.standard_normal(n_times),
        "var_t": np.abs(rng.standard_normal(n_times)),
        "std_t": np.abs(rng.standard_normal(n_times)),
    }


@pytest.fixture
def sample_windowed_df():
    """Minimal windowed-stats DataFrame."""
    return pd.DataFrame(
        {
            "window": [1, 2, 3],
            "center": [0.5, 1.5, 2.5],
            "t_start": [0.0, 1.0, 2.0],
            "t_end": [1.0, 2.0, 3.0],
            "mean_signal": [0.1, 0.2, 0.3],
            "var_signal": [0.01, 0.02, 0.03],
            "mean_variance": [0.5, 0.6, 0.4],
            "sync_candidate": [False, False, True],
        }
    )


@pytest.fixture
def sample_loo():
    """LOO-ISC: 4 subjects × 8 channels."""
    rng = np.random.default_rng(1)
    loo = rng.uniform(-0.1, 0.3, (4, 8))
    mean_isc = loo.mean(axis=0)
    return loo, mean_isc


@pytest.fixture
def sample_pairwise():
    """Symmetric 4×4 pairwise ISC matrix."""
    rng = np.random.default_rng(2)
    mat = rng.uniform(0.0, 0.2, (4, 4))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    return mat


# ---------------------------------------------------------------------------
# save / load round-trip tests
# ---------------------------------------------------------------------------


class TestSaveLoadIntersubjectTimeseries:
    def test_round_trip(self, tmp_path, sample_stats):
        dest = save_intersubject_timeseries(
            sample_stats,
            sfreq=100.0,
            out_dir=tmp_path,
            condition="Placebo",
            music_type="CLASSIC",
            band="broadband",
        )
        assert dest.exists()

        df = load_intersubject_timeseries(dest)
        assert len(df) == sample_stats["mean_t"].shape[0]
        assert set(df.columns) >= {
            "time",
            "mean_signal",
            "variance",
            "std",
            "condition",
            "music_type",
            "band",
        }
        np.testing.assert_allclose(
            df["mean_signal"].values, sample_stats["mean_t"], atol=1e-6
        )

    def test_time_column_uses_sfreq(self, tmp_path, sample_stats):
        save_intersubject_timeseries(sample_stats, sfreq=250.0, out_dir=tmp_path)
        df = load_intersubject_timeseries(tmp_path / "intersubject_timeseries.csv")
        expected_time = np.arange(sample_stats["mean_t"].shape[0]) / 250.0
        np.testing.assert_allclose(df["time"].values, expected_time, atol=1e-8)

    def test_metadata_columns(self, tmp_path, sample_stats):
        save_intersubject_timeseries(
            sample_stats,
            sfreq=100.0,
            out_dir=tmp_path,
            condition="Psilocybin",
            music_type="PSYTRANCE",
            band="alpha",
        )
        df = load_intersubject_timeseries(tmp_path / "intersubject_timeseries.csv")
        assert (df["condition"] == "Psilocybin").all()
        assert (df["music_type"] == "PSYTRANCE").all()
        assert (df["band"] == "alpha").all()

    def test_rejects_non_positive_sfreq(self, tmp_path, sample_stats):
        with pytest.raises(ValueError, match="sfreq must be a positive"):
            save_intersubject_timeseries(sample_stats, sfreq=0.0, out_dir=tmp_path)
        with pytest.raises(ValueError, match="sfreq must be a positive"):
            save_intersubject_timeseries(sample_stats, sfreq=-10.0, out_dir=tmp_path)


class TestSaveLoadWindowedStats:
    def test_round_trip(self, tmp_path, sample_windowed_df):
        dest = save_windowed_stats(
            sample_windowed_df,
            out_dir=tmp_path,
            condition="Placebo",
            music_type="CLASSIC",
        )
        assert dest.exists()

        df = load_windowed_stats(dest)
        assert len(df) == len(sample_windowed_df)
        assert set(df.columns) >= set(sample_windowed_df.columns) | {
            "condition",
            "music_type",
            "band",
        }

    def test_preserves_values(self, tmp_path, sample_windowed_df):
        save_windowed_stats(sample_windowed_df, out_dir=tmp_path)
        df = load_windowed_stats(tmp_path / "windowed_stats.csv")
        np.testing.assert_allclose(
            df["mean_variance"].values,
            sample_windowed_df["mean_variance"].values,
            atol=1e-6,
        )


class TestSaveLoadLooIsc:
    def test_round_trip(self, tmp_path, sample_loo):
        loo, mean_isc = sample_loo
        dest = save_loo_isc(
            loo,
            mean_isc,
            out_dir=tmp_path,
            condition="Placebo",
            music_type="CLASSIC",
            method="pearson",
        )
        assert dest.exists()

        df = load_loo_isc(dest)
        n_subjects, n_channels = loo.shape
        # Per-subject rows + per-channel mean rows
        assert len(df) == n_subjects * n_channels + n_channels

    def test_subject_values(self, tmp_path, sample_loo):
        loo, mean_isc = sample_loo
        save_loo_isc(loo, mean_isc, out_dir=tmp_path)
        df = load_loo_isc(tmp_path / "loo_isc.csv")
        # Filter out summary rows (subject == -1)
        per_subject = df[df["subject"] >= 0]
        for s in range(loo.shape[0]):
            for ch in range(loo.shape[1]):
                row = per_subject[
                    (per_subject["subject"] == s) & (per_subject["channel"] == ch)
                ]
                np.testing.assert_allclose(row["isc"].values[0], loo[s, ch], atol=1e-6)

    def test_mean_rows(self, tmp_path, sample_loo):
        loo, mean_isc = sample_loo
        save_loo_isc(loo, mean_isc, out_dir=tmp_path)
        df = load_loo_isc(tmp_path / "loo_isc.csv")
        means = df[df["subject"] == -1].sort_values("channel")
        np.testing.assert_allclose(means["isc"].values, mean_isc, atol=1e-6)

    def test_rejects_mismatched_mean_isc_shape(self, tmp_path, sample_loo):
        loo, _ = sample_loo
        wrong_mean = np.zeros(loo.shape[1] + 3)
        with pytest.raises(ValueError, match="mean_isc shape"):
            save_loo_isc(loo, wrong_mean, out_dir=tmp_path)


class TestSaveLoadPairwiseIsc:
    def test_round_trip(self, tmp_path, sample_pairwise):
        dest = save_pairwise_isc(sample_pairwise, out_dir=tmp_path)
        assert dest.exists()

        df = load_pairwise_isc(dest)
        n = sample_pairwise.shape[0]
        # Upper-triangle including diagonal: n*(n+1)/2
        expected_rows = n * (n + 1) // 2
        assert len(df) == expected_rows

    def test_values(self, tmp_path, sample_pairwise):
        save_pairwise_isc(sample_pairwise, out_dir=tmp_path)
        df = load_pairwise_isc(tmp_path / "pairwise_isc.csv")
        for _, row in df.iterrows():
            i, j = int(row["subject_i"]), int(row["subject_j"])
            np.testing.assert_allclose(row["isc"], sample_pairwise[i, j], atol=1e-6)

    def test_rejects_non_square_matrix(self, tmp_path):
        rect = np.ones((3, 4))
        with pytest.raises(ValueError, match="square 2-D array"):
            save_pairwise_isc(rect, out_dir=tmp_path)

    def test_rejects_1d_array(self, tmp_path):
        vec = np.ones(5)
        with pytest.raises(ValueError, match="square 2-D array"):
            save_pairwise_isc(vec, out_dir=tmp_path)


# ---------------------------------------------------------------------------
# scan_results_db
# ---------------------------------------------------------------------------


class TestScanResultsDb:
    def _populate(self, root, condition_music, spectrum, band=None):
        """Create a dummy CSV in the expected directory layout."""
        if spectrum == "broadband":
            d = root / condition_music / "broadband"
        else:
            assert band is not None
            d = root / condition_music / "bands" / band
        d.mkdir(parents=True, exist_ok=True)
        csv = d / "loo_isc.csv"
        csv.write_text("subject,channel,isc\n0,0,0.1\n")
        return csv

    def test_finds_broadband(self, tmp_path):
        self._populate(tmp_path, "Placebo_CLASSIC", "broadband")
        records = scan_results_db(tmp_path)
        assert len(records) == 1
        rec = records[0]
        assert rec["condition"] == "Placebo"
        assert rec["music_type"] == "CLASSIC"
        assert rec["spectrum_type"] == "broadband"
        assert rec["band"] == "broadband"
        assert rec["analysis_type"] == "loo_isc"

    def test_finds_band(self, tmp_path):
        self._populate(tmp_path, "Placebo_CLASSIC", "bands", band="alpha")
        records = scan_results_db(tmp_path)
        assert len(records) == 1
        assert records[0]["band"] == "alpha"
        assert records[0]["spectrum_type"] == "bands"

    def test_ignores_unknown_csv(self, tmp_path):
        d = tmp_path / "Placebo_CLASSIC" / "broadband"
        d.mkdir(parents=True)
        (d / "random_file.csv").write_text("col\n1\n")
        records = scan_results_db(tmp_path)
        assert len(records) == 0

    def test_ignores_bad_layout(self, tmp_path):
        # No condition_music separator
        d = tmp_path / "Placebo" / "broadband"
        d.mkdir(parents=True)
        (d / "loo_isc.csv").write_text("x\n1\n")
        records = scan_results_db(tmp_path)
        assert len(records) == 0

    def test_multiple_entries(self, tmp_path):
        self._populate(tmp_path, "Placebo_CLASSIC", "broadband")
        self._populate(tmp_path, "Placebo_CLASSIC", "bands", band="delta")
        self._populate(tmp_path, "Placebo_PSYTRANCE", "broadband")
        records = scan_results_db(tmp_path)
        assert len(records) == 3

    def test_empty_directory(self, tmp_path):
        records = scan_results_db(tmp_path)
        assert records == []

    def test_nonexistent_directory(self, tmp_path):
        records = scan_results_db(tmp_path / "does_not_exist")
        assert records == []
