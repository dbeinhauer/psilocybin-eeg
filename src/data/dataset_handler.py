"""
This module contains implementation of the main dataset handler class.
"""

from pathlib import Path

import mne

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
)
from src.data.dataset_parsing import DatasetParser
from src.data.dataset_preprocessing import DatasetPreprocessor
from src.utils.logging_config import LoggerMixin


class DatasetHandler(LoggerMixin):
    def __init__(
        self, experiment_name: ExperimentNames, coordinate_system: CoordinateSystems
    ):
        (
            self.raw_data_dir,  # Raw data directory
            self.participant_map_path,  # Path to CSV mapping for each experiment (Placebo/Psilocybin)
            self.processed_data_dir,  # Output directory for the processed data
            self.coordinates_path,  # Path to electrode coordinates file
        ) = self._init_dataset_paths(experiment_name)

        self.dataset_parser = DatasetParser(
            self.participant_map_path, self.coordinates_path
        )
        self.dataset_metadata = self.dataset_parser.parse_dataset_filenames(
            self.raw_data_dir
        )
        self.dataset_preprocessor = DatasetPreprocessor(coordinate_system)

    def _init_dataset_paths(
        self, experiment_name: ExperimentNames, coordinate_system: CoordinateSystems
    ) -> tuple[Path, Path, Path, Path]:
        """
        Initializes paths to working dataset.

        :param experiment_name: Variant of the experiment to process.
        :param coordinate_system: Variant of the coordinate system file we want to load.
        :return: Returns tuple of path to raw dataset directory, path to each participant
        experiment CSV mapping, path to directory where the processed results should be stored
        and path to the file where the coordinates of the electrodes are stored.
        """
        raw_data_dir, participant_map_path = ProjectPaths.get_experiment_data_dir(
            experiment_name, raw=True
        )
        processed_data_dir, _ = ProjectPaths.get_experiment_data_dir(
            experiment_name, raw=False
        )
        coordinates_path = ProjectPaths.get_coordinates_file_path(coordinate_system)

        return raw_data_dir, participant_map_path, processed_data_dir, coordinates_path

    def load_data_file(
        self, data_filename: str, is_unprocessed: bool = False
    ) -> mne.io.Raw:
        """
        Loads one EEG sequence (one data example).

        :param data_filename: Name of the file containing the wanted data.
        :param is_unprocessed: Flag whether the data to load is already processed or not
        (from where we want to load the data).
        :return: Returns loaded data in the Raw data type.
        """
        base_dir = self.raw_data_dir if is_unprocessed else self.processed_data_dir
        return mne.io.read_raw_edf(
            base_dir / data_filename,
            preload=True,
        )

    def preprocess_all_dataset(self):
        for i, row in self.dataset_metadata.iterrows():
            preprocessed_data = self.dataset_preprocessor.preprocess_one_raw_data(
                self.load_data_file(
                    row[SingleDataMetadata.FILENAME], is_unprocessed=True
                )
            )
