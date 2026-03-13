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
