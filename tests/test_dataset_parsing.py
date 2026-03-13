"""
This module tests the DatasetParser class functionality.
"""

import pytest  # noqa: F401  — needed for @pytest.fixture decorators
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from src.io.parsing import DatasetParser
from src.definitions.fields import (
    SingleDataMetadata,
    ConditionVariants,
    MusicTypeVariants,
)


class TestDatasetParser:
    """Test class for DatasetParser functionality."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_participant_map):
        """Set up test fixtures before each test method."""
        self.parser = DatasetParser(sample_participant_map)

    def test_parse_filename_valid_classical(self):
        filename = "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf"

        result = self.parser.parse_filename(filename)

        assert result is not None
        assert result[SingleDataMetadata.PARTICIPANT_ID] == "018"
        assert result[SingleDataMetadata.CONDITION] == ConditionVariants.PLACEBO
        assert result[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.CLASSICAL
        assert result[SingleDataMetadata.FILENAME] == filename

    def test_parse_filename_valid_psytrance(self):
        filename = "PSI019_EEGB_MUSIC_PSYTRANCE_EC_20171125_014218.edf"

        result = self.parser.parse_filename(filename)

        assert result is not None
        assert result[SingleDataMetadata.PARTICIPANT_ID] == "019"
        assert result[SingleDataMetadata.CONDITION] == ConditionVariants.PSILOCYBIN
        assert result[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.PSYTRANCE
        assert result[SingleDataMetadata.FILENAME] == filename

    def test_parse_filename_invalid_pattern(self):
        filename = "invalid_filename.edf"

        result = self.parser.parse_filename(filename)

        assert result is None

    def test_parse_filename_missing_date_part(self):
        filename = "PSI018_EEGA_MUSIC_CLASSIC_EC.edf"

        result = self.parser.parse_filename(filename)

        assert result is None

    def test_parse_filename_wrong_extension(self):
        filename = "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.txt"

        result = self.parser.parse_filename(filename)

        assert result is None

    def test_parse_filename_invalid_condition(self):
        filename = "PSI018_EEGX_MUSIC_CLASSIC_EC_20171124_014218.edf"

        result = self.parser.parse_filename(filename)

        assert result is None

    def test_parse_filename_invalid_music_type(self):
        filename = "PSI018_EEGA_MUSIC_ROCK_EC_20171124_014218.edf"

        result = self.parser.parse_filename(filename)

        assert result is None

    def test_parse_filename_case_insensitive_condition(self):
        filename = "PSI018_EEGa_MUSIC_CLASSIC_EC_20171124_014218.edf"

        result = self.parser.parse_filename(filename)

        # The regex matches lowercase 'a' but EEGConditions only has 'A' / 'B',
        # so the condition ID validation fails and None is returned.
        assert result is None

    def test_parse_filename_case_insensitive_music_type(self):
        filename = "PSI018_EEGA_MUSIC_claSSic_EC_20171124_014218.edf"

        result = self.parser.parse_filename(filename)

        assert result is not None
        assert result[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.CLASSICAL

    @patch("pathlib.Path.glob")
    def test_parse_dataset_filenames_empty_directory(self, mock_glob):
        """Test parsing filenames from an empty directory."""
        mock_glob.return_value = []
        dataset_path = Path("/fake/dataset/path")

        result = self.parser.parse_dataset_filenames(dataset_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == []

    @patch("pathlib.Path.glob")
    def test_parse_dataset_filenames_valid_files(self, mock_glob):
        valid_files = [
            Mock(),
            Mock(),
        ]
        valid_files[0].name = "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf"
        valid_files[1].name = "PSI019_EEGB_MUSIC_PSYTRANCE_EC_20171125_014218.edf"
        mock_glob.return_value = valid_files

        dataset_path = Path("/fake/dataset/path")
        result = self.parser.parse_dataset_filenames(dataset_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Check first row
        row1 = result.iloc[0]
        assert row1[SingleDataMetadata.PARTICIPANT_ID] == "018"
        assert row1[SingleDataMetadata.CONDITION] == ConditionVariants.PLACEBO
        assert row1[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.CLASSICAL
        assert (
            row1[SingleDataMetadata.FILENAME]
            == "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf"
        )

        # Check second row
        row2 = result.iloc[1]
        assert row2[SingleDataMetadata.PARTICIPANT_ID] == "019"
        assert row2[SingleDataMetadata.CONDITION] == ConditionVariants.PSILOCYBIN
        assert row2[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.PSYTRANCE
        assert (
            row2[SingleDataMetadata.FILENAME]
            == "PSI019_EEGB_MUSIC_PSYTRANCE_EC_20171125_014218.edf"
        )

    @patch("pathlib.Path.glob")
    def test_parse_dataset_filenames_mixed_valid_invalid(self, mock_glob):
        mixed_files = [
            Mock(name="valid_0"),
            Mock(name="invalid_format"),
            Mock(name="valid_1"),
            Mock(name="invalid_condition"),
        ]
        mixed_files[0].name = "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf"
        mixed_files[1].name = "invalid_filename.edf"
        mixed_files[2].name = "PSI019_EEGB_MUSIC_PSYTRANCE_EC_20171125_014218.edf"
        mixed_files[3].name = "PSI018_EEGX_MUSIC_CLASSIC_EC_20171124_014218.edf"
        mock_glob.return_value = mixed_files

        dataset_path = Path("/fake/dataset/path")
        result = self.parser.parse_dataset_filenames(dataset_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Only valid files should be included

        # Check that only valid files are included
        filenames = result[SingleDataMetadata.FILENAME].tolist()
        assert "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf" in filenames
        assert "PSI019_EEGB_MUSIC_PSYTRANCE_EC_20171125_014218.edf" in filenames
        # Invalid filenames
        assert "invalid_filename.edf" not in filenames
        assert "PSI018_EEGX_MUSIC_CLASSIC_EC_20171124_014218.edf" not in filenames

    def test_parse_filename_edge_cases(self):
        # Test with minimal valid filename
        filename = "PSI000_EEGA_MUSIC_CLASSIC_EC_20170101_000000.edf"
        result = self.parser.parse_filename(filename)
        assert result is not None
        assert result[SingleDataMetadata.PARTICIPANT_ID] == "000"

        # Test with maximum participant ID
        filename = "PSI999_EEGA_MUSIC_CLASSIC_EC_20170101_000000.edf"
        result = self.parser.parse_filename(filename)
        assert result is not None
        assert result[SingleDataMetadata.PARTICIPANT_ID] == "999"


class TestDatasetParserIntegration:
    """Integration tests for DatasetParser."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_participant_map):
        """Set up test fixtures before each test method."""
        self.parser = DatasetParser(sample_participant_map)

    def test_full_workflow(self):
        """Test the complete workflow from filename to DataFrame."""
        # Test individual filename parsing
        filename = "PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf"

        # Test DataFrame creation with mock data
        with patch("pathlib.Path.glob") as mock_glob:
            mock_files = [Mock()]
            mock_files[0].name = filename
            mock_glob.return_value = mock_files

            dataset_path = Path("/fake/dataset/path")
            df = self.parser.parse_dataset_filenames(dataset_path)

            assert len(df) == 1
            row = df.iloc[0]
            assert row[SingleDataMetadata.PARTICIPANT_ID] == "018"
            assert row[SingleDataMetadata.CONDITION] == ConditionVariants.PLACEBO
            assert row[SingleDataMetadata.MUSIC_TYPE] == MusicTypeVariants.CLASSICAL
            assert row[SingleDataMetadata.FILENAME] == filename
