"""
This module contains implementation of the main dataset handler class.
"""

from pathlib import Path

import numpy as np
import mne

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
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
            self.excluded_electrodes_path,  # Path to CSV list of excluded electrodes.
        ) = self._init_dataset_paths(experiment_name, coordinate_system)

        self.dataset_parser = DatasetParser(self.participant_map_path)
        self.dataset_metadata = self.dataset_parser.parse_dataset_filenames(
            self.raw_data_dir
        )
        self.dataset_preprocessor = DatasetPreprocessor(
            self.coordinates_path, self.excluded_electrodes_path
        )

    def _init_dataset_paths(
        self, experiment_name: ExperimentNames, coordinate_system: CoordinateSystems
    ) -> tuple[Path, Path, Path, Path, Path]:
        """
        Initializes paths to working dataset.

        :param experiment_name: Variant of the experiment to process.
        :param coordinate_system: Variant of the coordinate system file we want to load.
        :return: Returns tuple of path to raw dataset directory, path to each participant
        experiment CSV mapping, path to directory where the processed results should be stored,
        path to the file where the coordinates of the electrodes are stored, and path where the
        electrodes selected for exclusion are stored (due problematic position in head).
        """
        raw_data_dir, participant_map_path = ProjectPaths.get_experiment_data_dir(
            experiment_name, is_processed=False
        )
        processed_data_dir, _ = ProjectPaths.get_experiment_data_dir(
            experiment_name, is_processed=True
        )
        coordinates_path, excluded_electrodes_path = (
            ProjectPaths.get_coordinates_file_path(coordinate_system)
        )

        return (
            raw_data_dir,
            participant_map_path,
            processed_data_dir,
            coordinates_path,
            excluded_electrodes_path,
        )

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

    def save_data_file(
        self,
        data: mne.io.Raw | mne.preprocessing.ICA | np.ndarray,
        filename: str,
        data_type: PreprocessedDataVariants,
    ):
        """
        Stores the selected data after preprocessing.

        :param data: Data to be stored.
        :param filename: Name of the file where the data should be stored (the path is computed
        based on the data type and default parameters.)
        :param data_type: Type of the data to be processed.
        """
        self.logger.info(
            f"Saving the '{data_type.value}' data into the file {filename}."
        )
        # Path to the directory where to store the data.
        data_dir = self.processed_data_dir / data_type.value
        data_dir.mkdir(parents=True, exist_ok=True)

        if data_type in RAW_DATA_VARIANTS:
            data.save(data_dir / (filename + ".fif"), overwrite=True)
        elif data_type == PreprocessedDataVariants.ICA_COMPONENTS:
            data.save(str(data_dir / (filename + ".fif")), overwrite=True)
        elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
            np.save(data_dir / (filename + ".npy"), data)
        else:
            self.logger.warning(
                f"Wrong datatype: '{data_type.value}' to store. Skipping!"
            )
        self.logger.info("Data saved successfully!")

    def process_one_file(self, filename: str, save_processing_info: bool):
        """
        Processes one raw EEG data series file.

        First names channels, removes not wanted channels, trims start and end time intervals,
        filters line noise, interpolates bad channels (while using Ransac for detection), and runs
        ICA labeling and filters components marked as (eye, heart, or muscle).

        :param filename: Name of the file to process.
        :param save_processing_info: Flag whether we want to store intermediate processing results
        (for analysis of the preprocessing performance).
        """

        self.logger.info(f"Processing file: {filename}")
        # Get preprocessed data with interpolated bad channels.
        interpolated_data = self.dataset_preprocessor.initial_preprocessing_and_bad_channel_interpolation(
            self.load_data_file(filename, is_unprocessed=True)
        )

        # Filename base without the suffix.
        filename_base = filename.split(".")[0]
        if save_processing_info:
            # Save data before running IC labeling.
            self.save_data_file(
                interpolated_data,
                filename_base,
                data_type=PreprocessedDataVariants.RAW_BEFORE_ICA,
            )

        # ICA component labeling and filtering.
        ic_reduced_data, ica_components, ic_probabilites = (
            self.dataset_preprocessor.apply_ica_component_filtering(interpolated_data)
        )

        # Store final processed data.
        self.save_data_file(
            ic_reduced_data, filename_base, PreprocessedDataVariants.RAW_AFTER_ICA
        )
        if save_processing_info:
            # Save found ICA components and its probabilites to belong to selected classes.
            self.save_data_file(
                ica_components,
                filename_base,
                data_type=PreprocessedDataVariants.ICA_COMPONENTS,
            )
            self.save_data_file(
                ic_probabilites,
                filename_base,
                data_type=PreprocessedDataVariants.IC_PROBABILITIES,
            )

    def preprocess_all_dataset(self, save_processing_info: bool = True):
        for i, row in self.dataset_metadata.iterrows():
            filename = row[SingleDataMetadata.FILENAME]
            self.process_one_file(filename, save_processing_info)
