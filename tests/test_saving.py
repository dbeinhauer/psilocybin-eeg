"""
Tests for src/io/saving.py — Data saving functions.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.definitions.fields import PreprocessedDataVariants
from src.io.saving import save_data_file


class TestSaveDataFile:
    """Test save_data_file dispatches correctly based on data type."""

    def test_save_raw_data_calls_save(self, tmp_path):
        """Test that Raw-type data variants invoke data.save()."""
        mock_data = MagicMock()
        save_data_file(
            data=mock_data,
            processed_data_dir=tmp_path,
            filename="PSI001",
            data_type=PreprocessedDataVariants.RAW_AFTER_ICA,
        )
        mock_data.save.assert_called_once()
        call_args = mock_data.save.call_args
        assert str(call_args[0][0]).endswith(".fif")

    def test_save_ica_components_calls_save(self, tmp_path):
        """Test that ICA components invoke data.save()."""
        mock_data = MagicMock()
        save_data_file(
            data=mock_data,
            processed_data_dir=tmp_path,
            filename="PSI001",
            data_type=PreprocessedDataVariants.ICA_COMPONENTS,
            interim_data_dir=tmp_path / "interim",
        )
        mock_data.save.assert_called_once()

    @patch("src.io.saving.np.save")
    def test_save_ic_probabilities_calls_np_save(self, mock_np_save, tmp_path):
        """Test that IC probabilities invoke np.save()."""
        data = np.array([0.1, 0.2, 0.3])
        save_data_file(
            data=data,
            processed_data_dir=tmp_path,
            filename="PSI001",
            data_type=PreprocessedDataVariants.IC_PROBABILITIES,
            interim_data_dir=tmp_path / "interim",
        )
        mock_np_save.assert_called_once()
        call_args = mock_np_save.call_args
        assert str(call_args[0][0]).endswith(".npy")

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        mock_data = MagicMock()
        nested = tmp_path / "deep" / "nested"
        save_data_file(
            data=mock_data,
            processed_data_dir=nested,
            filename="PSI001",
            data_type=PreprocessedDataVariants.RAW_AFTER_ICA,
        )
        # The parent of the file should exist
        expected_parent = nested / "after_ica"
        assert expected_parent.exists()

    def test_save_unknown_type_warns(self, tmp_path, caplog):
        """Test that unsupported data type logs a warning."""
        mock_data = MagicMock()
        save_data_file(
            data=mock_data,
            processed_data_dir=tmp_path,
            filename="PSI001",
            data_type=PreprocessedDataVariants.CONCATENATED,
        )
        # Should not call save since CONCATENATED isn't in RAW_DATA_VARIANTS or IC types
        mock_data.save.assert_not_called()


class TestSnrCourses:
    """save_snr_courses / load_snr_courses round trip."""

    def test_round_trip(self, tmp_path):
        from src.io.loading import load_snr_courses
        from src.io.saving import save_snr_courses

        rng = np.random.default_rng(0)
        courses = {
            ("2_roi_zscore_pca", "40hz"): {
                "Placebo": rng.standard_normal((3, 2, 5)).astype(np.float32),
                "Psilocybin": rng.standard_normal((3, 2, 5)).astype(np.float32),
            },
            ("full", "35-45hz"): {
                "Placebo": rng.standard_normal((3, 2, 5)).astype(np.float32)
            },
        }
        times = np.linspace(-0.1, 0.3, 5)
        path = tmp_path / "snr_courses.npz"
        save_snr_courses(path, courses, ["019", "023", "026"], ["PC 1", "ROI"], times)

        participants, sources, loaded_times, loaded = load_snr_courses(path)
        assert participants == ["019", "023", "026"]
        assert sources == ["PC 1", "ROI"]
        np.testing.assert_array_equal(loaded_times, times)
        assert loaded.keys() == courses.keys()
        for key, by_condition in courses.items():
            for condition, course in by_condition.items():
                np.testing.assert_array_equal(loaded[key][condition], course)
        assert [p.name for p in tmp_path.iterdir()] == ["snr_courses.npz"]


class TestSnrValues:
    """save_snr_values / load_snr_values round trip."""

    def test_round_trip(self, tmp_path):
        from src.io.loading import load_snr_values
        from src.io.saving import save_snr_values

        rng = np.random.default_rng(0)
        values = {
            ("2_roi_zscore_pca", "40hz", "0-0.5s"): {
                "Placebo": rng.standard_normal((3, 2)),
                "Psilocybin": rng.standard_normal((3, 2)),
            },
            ("full", "35-45hz", "0-0.500 s (paradigm default)"): {
                "Placebo": rng.standard_normal((3, 2))
            },
        }
        path = tmp_path / "snr_values.npz"
        save_snr_values(path, values, ["019", "023", "026"], ["PC 1", "ROI"])

        participants, sources, loaded = load_snr_values(path)
        assert participants == ["019", "023", "026"]
        assert sources == ["PC 1", "ROI"]
        assert loaded.keys() == values.keys()
        for key, by_condition in values.items():
            for condition, value in by_condition.items():
                np.testing.assert_array_equal(loaded[key][condition], value)
        assert [p.name for p in tmp_path.iterdir()] == ["snr_values.npz"]
