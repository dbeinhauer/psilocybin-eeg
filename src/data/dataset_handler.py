"""
This module contains implementation of the main dataset handler class.
"""

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
    ICLabelComponentsClasses,
    ExcludedICsMetadata,
)
from src.data.dataset_parsing import DatasetParser
from src.data.dataset_preprocessing import DatasetPreprocessor
from src.data.dataset_plotting import DatasetPlotter
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
        self.dataset_excluded_ics_metadata = self._init_excluded_ics_metadata()

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

    def _get_excluded_ics_mapping_path(self) -> Path:
        """
        :return: Returns path to excluded ICs mapping CSV file.
        """
        return (
            self.processed_data_dir
            / PreprocessedDataVariants.RAW_EXCLUDED_IC.value
            / ProjectPaths.EXCLUDED_ICS_FILENAME_MAPPING
        )

    def _init_excluded_ics_metadata(self) -> pd.DataFrame:
        """
        Initializes metadata dataframe where info about the excluded ICs should be stored.

        :return: Returns already extracted ICs metadata dataframe from csv file (if already processed.),
        otherwise prepared dataframe for filling in the information.
        """
        excluded_ics_path = self._get_excluded_ics_mapping_path()
        if excluded_ics_path.exists():
            return pd.read_csv(excluded_ics_path)
        return pd.DataFrame(
            columns=[column_name.value for column_name in ExcludedICsMetadata]
        )

    @staticmethod
    def get_excluded_ic_filename(
        filename: str, ic_id: int, ic_category: ICLabelComponentsClasses
    ) -> str:
        """
        Creates filename for excluded ICs. New filename is in format:
            `{old_filename_prefix}_IC-{ic_id}_{ic_category.value()}.{old_suffix}`

        NOTE: The suffix stays the same as original due to compatibility reasons.
        The suffix will be appropriately changed while storing the object.

        :param filename: Filename of original data file (original raw data filename from metadata).
        :param ic_id: ID of the IC.
        :param ic_category: Category where the IC was put after IC labelling.
        :return: Returns new prepared filename.
        """
        filename_parts = filename.split(".")
        ic_postfix = f"_IC-{ic_id}_{ic_category.value}"
        return filename_parts[0] + ic_postfix

    def load_data_file(
        self,
        data_filename: str,
        is_processed: bool = False,
        processed_data_type: PreprocessedDataVariants = PreprocessedDataVariants.RAW_AFTER_ICA,
    ) -> mne.io.Raw | np.ndarray:
        """
        Loads one EEG sequence (one data example).

        :param data_filename: Name of the file containing the wanted data.
        :param is_processed: Flag whether the data to load is already processed or not
        (from where we want to load the data).
        :param processed_data_type: Type of the processed file to load
        :return: Returns loaded data in the Raw data type.
        """
        data_path = self.raw_data_dir / data_filename
        if is_processed:
            # Load processed data file
            data_path = self.get_preprocessing_results_path(
                data_filename.split(".")[0], data_type=processed_data_type
            )
            if processed_data_type == PreprocessedDataVariants.IC_PROBABILITIES:
                # Load IC Probabilities
                return np.load(data_path)
            elif processed_data_type == PreprocessedDataVariants.ICA_COMPONENTS:
                # Load ICA components
                return mne.preprocessing.read_ica(
                    data_path,
                )
            else:
                # We need this else for Raw dataseries are in '.fif' format.
                return mne.io.read_raw_fif(
                    data_path,
                    preload=True,
                )

        # Load unprocessed raw data are in '.edf' format.
        return mne.io.read_raw_edf(
            data_path,
            preload=True,
        )

    def load_excluded_ic_dataseries(
        self, original_filename: str, ic_id: int
    ) -> mne.io.Raw:
        """
        Loads selected excluded IC dataseries from selected original file.

        :param original_filename: File of the original dataseries.
        :param ic_id: ID of the excluded IC.
        :return: Returns dataseries of the excluded IC.
        """
        excluded_filename = self.dataset_excluded_ics_metadata[
            (
                self.dataset_excluded_ics_metadata[
                    ExcludedICsMetadata.ORIGINAL_FILENAME.value
                ]
                == original_filename
            )
            & (
                self.dataset_excluded_ics_metadata[ExcludedICsMetadata.IC_ID.value]
                == ic_id
            )
        ][ExcludedICsMetadata.TIMESERIES_FILENAME.value].iloc[0]
        if not excluded_filename:
            self.logger.error(
                f"Cannot load dataseries. Possibly wrong filename: {original_filename}, IC ID: {ic_id} or extracted IC does not exist."
            )
            return None
        return self.load_data_file(
            excluded_filename,
            is_processed=True,
            processed_data_type=PreprocessedDataVariants.RAW_EXCLUDED_IC,
        )

    def get_preprocessing_results_path(
        self, filename, data_type: PreprocessedDataVariants
    ) -> Path:
        """
        Gets the path to a specified processing results.

        :param filename: Name of the experiment file.
        :param data_type: Type of the processed data.
        :return: Returns path to the specified processing results.
        """
        suffix = ""
        if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
            suffix = ".fif"
        elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
            suffix = ".npy"

        return self.processed_data_dir / data_type.value / (filename + suffix)

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
        # Path to results file.
        data_path = self.get_preprocessing_results_path(filename, data_type)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
            data.save(data_path, overwrite=True)
        elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
            np.save(data_path, data)
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
        interpolated_data, aux_data = (
            self.dataset_preprocessor.initial_preprocessing_and_bad_channel_interpolation(
                self.load_data_file(filename, is_processed=False)
            )
        )

        # Filename base without the suffix.
        filename_base = filename.split(".")[0]
        if save_processing_info:
            # Save data before running IC labeling.
            self.save_data_file(
                interpolated_data.copy().add_channels(
                    [aux_data.copy()], force_update_info=True
                ),
                filename_base,
                data_type=PreprocessedDataVariants.RAW_BEFORE_ICA,
            )

        # ICA component labeling and filtering.
        ic_reduced_data, ica_components, ic_probabilites = (
            self.dataset_preprocessor.apply_ica_component_filtering(interpolated_data)
        )

        ic_reduced_data.add_channels([aux_data], force_update_info=True)

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
        """
        Runs preprocessing steps for all edf files in dataset.

        :param save_processing_info: Flag whether we want to store intermediate processing results
        (for analysis of the preprocessing performance).
        """
        for i, row in self.dataset_metadata.iterrows():
            # if i > 0:
            #     continue
            filename = row[SingleDataMetadata.FILENAME]
            self.process_one_file(filename, save_processing_info)

    def get_one_original_filename_ic_metadata(
        self, filename: str
    ) -> list[dict[str, Any]]:
        """
        From the ICAs retrieves all excluded ICs, its labels and component numbers.

        :param filename: Filename of the original raw data we are interested in excluded ICs.
        :return: Returns list of rows containing metadata regarding ICs for one original data file.
        """
        self.logger.info(
            f"Start generating time series of excluded ICs from file: {filename}"
        )

        # ICAs of the dataseries.
        data_icas = self.load_data_file(
            filename,
            is_processed=True,
            processed_data_type=PreprocessedDataVariants.ICA_COMPONENTS,
        )
        ics_probabilities = DatasetPreprocessor.get_ic_labeling_probabilities(
            self.load_data_file(
                filename,
                is_processed=True,
                processed_data_type=PreprocessedDataVariants.IC_PROBABILITIES,
            )
        )

        # List of rows for excluded ICs metadata pd.Dataframe.
        excluded_ics_per_data = []

        # Get ID of the IC and its category, exclude all ICs from the signal and store the series.
        ic_exclusion_map = self.dataset_parser.get_ic_exclusion_map(data_icas)
        for ic_id, ic_category in ic_exclusion_map.items():
            # Exclude all ICs with exception of `ic_id`.
            excluded_ics_per_data.append(
                {
                    ExcludedICsMetadata.ORIGINAL_FILENAME.value: filename,
                    ExcludedICsMetadata.IC_ID.value: ic_id,
                    ExcludedICsMetadata.IC_CATEGORY.value: ic_category.value,
                    ExcludedICsMetadata.TOTAL_ICS.value: data_icas.n_components_,
                    ExcludedICsMetadata.MAIN_PROBABILITY.value: ics_probabilities[
                        ic_category
                    ][ic_id],
                }
            )

        return excluded_ics_per_data

    def extract_all_excluded_ic_metadata(
        self,
    ):
        """
        Extracts all metadata for extracted ICs and stores them
        to `self.dataset_excluded_ics_metadata` DataFrame.
        """
        self.logger.info("Generating excluded ICs time series.")
        excluded_ics_rows = []
        for i, row in self.dataset_metadata.iterrows():
            # if i > 0:
            #     continue
            excluded_ics_rows += self.get_one_original_filename_ic_metadata(
                row[SingleDataMetadata.FILENAME]
            )
        self.logger.info("All excluded ICs time series generated.")

        # Create excluded ICs metadata pandas dataframe and store them into CSV file.
        self.logger.info("Storing excluded ICs metadata to CSV file")
        self.dataset_excluded_ics_metadata = pd.DataFrame(excluded_ics_rows)
        self.dataset_excluded_ics_metadata.to_csv(self._get_excluded_ics_mapping_path())

    def plot_all_one_variant(self, plot_variant: str = "power_spectrum"):
        """
        Plot all dataset Raw dataseries object using MNE plotting functionalities from `compute_psd` base.

        :param plot_variant: Which plot we want to create. Either "power_spectrum", or "topomap".
        """
        self.logger.info(f"Plotting {plot_variant} for all dataset.")
        for i, row in self.dataset_metadata.iterrows():
            # if i > 0:
            #     continue
            # Iterate through all data from the provided dataset.
            original_filename = row[SingleDataMetadata.FILENAME]
            self.logger.info(f"Plotting original file: {original_filename}")

            for data_variant in RAW_DATA_VARIANTS:
                # Plot all Raw data variants.
                self.logger.info(f"Plotting variant: {data_variant.value}")
                if data_variant != PreprocessedDataVariants.RAW_EXCLUDED_IC:
                    # Plot Raw data that are not belonging to excluded ICs.
                    raw_data = self.load_data_file(
                        original_filename,
                        is_processed=True,
                        processed_data_type=data_variant,
                    )
                    DatasetPlotter.plot_raw_dataseries(
                        raw_data,
                        save_fig=original_filename.split(".")[0],
                        plot_variant=plot_variant,
                        variant_name=data_variant,
                    )
                else:
                    if plot_variant == "power_spectrum":
                        continue
                    # Plot all excluded ICs raw data.
                    excluded_metadata = self.dataset_excluded_ics_metadata[
                        self.dataset_excluded_ics_metadata[
                            ExcludedICsMetadata.ORIGINAL_FILENAME.value
                        ]
                        == original_filename
                    ]
                    raw_data = self.load_data_file(
                        original_filename,
                        is_processed=True,
                        processed_data_type=PreprocessedDataVariants.ICA_COMPONENTS,
                    )
                    for j, excluded_row in excluded_metadata.iterrows():
                        # Plot each excluded IC dataseries.
                        self.logger.info(
                            f"Plotting IC excluded component: {excluded_row[ExcludedICsMetadata.IC_ID.value]}"
                        )
                        # raw_data = self.load_excluded_ic_dataseries(
                        #     original_filename,
                        #     excluded_row[ExcludedICsMetadata.IC_ID.value],
                        # )
                        DatasetPlotter.plot_raw_dataseries(
                            raw_data,
                            save_fig=excluded_row[
                                ExcludedICsMetadata.ORIGINAL_FILENAME.value
                            ].split(".")[0],
                            is_excluded=True,
                            excluded_ic_id=j,
                            plot_variant=plot_variant,
                            variant_name=data_variant,
                            title=f"IC - {excluded_row[ExcludedICsMetadata.IC_ID.value]}, {excluded_row[ExcludedICsMetadata.IC_CATEGORY.value]}, p: {excluded_row[ExcludedICsMetadata.MAIN_PROBABILITY.value]:.2f}",
                        )

        self.logger.info("Plotting successfully finished")
