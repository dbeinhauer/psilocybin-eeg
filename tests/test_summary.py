"""
Tests for src/analysis/summary.py -- summarized analyzer loading behavior.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.analysis.summary import EEGSummarizedAnalyzer
from src.definitions.fields import (
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    MusicTypeVariants,
    SingleDataMetadata,
)


class TestLoadAndPrepareData:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_adds_data_axis0_index_mapping_to_filtered_df(
        self, mock_dataset_handler_cls, mock_filter
    ):
        filtered_df = pd.DataFrame(
            {
                SingleDataMetadata.FILENAME: ["first_raw.fif", "second_raw.fif"],
            },
            index=[10, 20],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()

        mock_raw = MagicMock()
        mock_raw.pick.return_value = mock_raw
        mock_raw.resample.return_value = mock_raw
        mock_raw.info = {"sfreq": 200.0, "ch_names": ["Cz", "Pz"]}
        mock_raw.get_data.side_effect = [
            np.zeros((2, 3)),
            np.ones((2, 3)),
        ]
        mock_dataset_handler.load_data_file.return_value = mock_raw
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        data, _ = analyzer.load_and_prepare_data(resample_freq=200.0, n_jobs=1)

        assert data.shape == (2, 2, 3)
        np.testing.assert_array_equal(data[0], np.zeros((2, 3)))
        np.testing.assert_array_equal(data[1], np.ones((2, 3)))
        assert (
            SingleDataMetadata.CONCATENATED_PERSON_INDEX in analyzer.filtered_df.columns
        )
        assert analyzer.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0, 1]
        assert analyzer.filtered_df.index.tolist() == [10, 20]


class TestSaveLoadDataWithMetadata:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_metadata_persists_through_save_load_cycle(
        self, mock_dataset_handler_cls, mock_filter, tmp_path
    ):
        filtered_df = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["first_raw.fif", "second_raw.fif"]},
            index=[10, 20],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        analyzer.data = np.arange(12).reshape(2, 2, 3)
        analyzer.filtered_df = filtered_df.copy()
        analyzer.filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = [0, 1]
        save_path = tmp_path / "concatenated.npy"
        metadata_path = tmp_path / "concatenated.metadata.csv"
        analyzer.save_data(save_path=save_path, metadata_path=metadata_path)

        restored = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )
        restored.load_data(load_path=save_path, metadata_path=metadata_path)

        np.testing.assert_array_equal(restored.data, analyzer.data)
        assert (
            SingleDataMetadata.CONCATENATED_PERSON_INDEX in restored.filtered_df.columns
        )
        assert restored.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0, 1]
        assert restored.filtered_df[SingleDataMetadata.FILENAME].tolist() == [
            "first_raw.fif",
            "second_raw.fif",
        ]
        assert restored.filtered_df.index.tolist() == [10, 20]
