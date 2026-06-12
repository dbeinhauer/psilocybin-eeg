"""
This module contains the top-level orchestration for the EEG preprocessing
pipeline (current DatasetHandler preprocessing).
"""

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ChannelTypes,
    ConditionVariants,
    ExperimentNames,
    CoordinateSystems,
    MusicTypeVariants,
    SingleDataMetadata,
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
    ICLabelComponentsClasses,
    ExcludedICsMetadata,
    ExclusionCategories,
)
from src.definitions.mappings import RAW_CHANNEL_NAMES
from src.io.parsing import DatasetParser
from src.io.loading import load_data_file, get_preprocessing_results_path
from src.io.saving import save_data_file
from src.preprocessing.channel_prep import (
    load_coordinates_file,
    add_coordinates_montage,
    exclude_selected_channels,
    crop_start_and_end_of_dataseries,
)
from src.preprocessing.filtering import (
    apply_filters_to_data,
    apply_ransac_filter,
    interpolate_bad_channels,
    detect_bad_epochs,
)
from src.preprocessing.ica import (
    get_ic_labeling_probabilities,
    apply_ica_component_filtering,
)
from src.preprocessing.time_alignment import TimeAligner, TAGObject
from src.filtering.dataset_filter import DatasetFilter
from src.visualization.preprocessing_plots import DatasetPlotter
from src.utils.logging_config import LoggerMixin


class DatasetPreprocessor(LoggerMixin):
    """
    Handles EEG data preprocessing including channel renaming, montage application,
    filtering, bad channel interpolation, ICA decomposition, and artifact component removal.

    Delegates to focused modules in src.preprocessing (channel_prep, filtering, ica).
    """

    def __init__(self, coordinates_file_path: Path, excluded_coordinates_path: Path):
        """
        Initialize the preprocessor with electrode coordinate and exclusion information.

        :param coordinates_file_path: Path to the SFP montage file with electrode positions.
        :param excluded_coordinates_path: Path to CSV listing electrodes to exclude (e.g. boundary electrodes).
        """
        self.montage = load_coordinates_file(coordinates_file_path)
        # List of all electrodes that we want to exclude.
        self.electrodes_to_exclude: list[str] = (
            pd.read_csv(excluded_coordinates_path)["electrode_name"]
            .str.strip()
            .tolist()
        )

    def _data_preparation(self, data: mne.io.Raw) -> tuple[mne.io.Raw, mne.io.Raw]:
        """
        Do first preparation of the raw data.

        Namely it adds the electrode coordinates, excludes the electrodes
        that has problematic placement, crops first and last 10 seconds of
        the data series (typically noisy).

        :param data: Data to be processed.
        :return: Returns tuple of prepared data splitted on EEG and rest of channels.
        """
        prepared_data = crop_start_and_end_of_dataseries(
            exclude_selected_channels(
                add_coordinates_montage(data, self.montage, logger=self.logger),
                self.electrodes_to_exclude,
                logger=self.logger,
            ),
            logger=self.logger,
        )

        eeg_data = prepared_data.copy().pick_types(eeg=True)
        aux_data = prepared_data.copy().pick_types(eeg=False, ecg=True, stim=True)

        return eeg_data, aux_data

    def _filter_data(self, eeg_data: mne.io.Raw) -> mne.io.Raw:
        """
        Applies notch and FIR filters to remove the line noise, and
        applies Ransac algorithm to detect badchannels.

        :param eeg_data: EEG data to do the preprocessing on.
        :return: Returns filtered EEG data.
        """
        return apply_ransac_filter(
            apply_filters_to_data(eeg_data, logger=self.logger),
            logger=self.logger,
        )

    def initial_preprocessing_and_bad_channel_interpolation(
        self, data: mne.io.Raw
    ) -> tuple[mne.io.Raw, mne.io.Raw]:
        """
        Run initial preprocessing steps (electrode naming,
        time trimming, electrodes exclusion, line noise filtering, bad
        channel interpolation).

        :param data: Raw EEG data series from one measurement.
        :return: Tuple of preprocessed data with interpolated bad channels and rest of channels (other than EEG).
        """
        self.logger.info(
            "Starting initial data preprocessing and interpolation of the bad channels."
        )
        # Prepare the data for preprocessing.
        eeg_data, aux_data = self._data_preparation(data)
        # Filter, interpolate the bad channels by using average reference and annotate bad epochs.
        eeg_data = detect_bad_epochs(
            interpolate_bad_channels(self._filter_data(eeg_data), logger=self.logger)
        )
        return eeg_data, aux_data

    def apply_ica_component_filtering(
        self,
        interpolated_data: mne.io.Raw,
    ) -> tuple[mne.io.Raw, mne.preprocessing.ICA, np.ndarray]:
        """
        Runs ICA and then executes ICLabel tool to predict probabilities
        of the several artifacts in each component, excludes the artifact
        components and returns the filtered signal.

        :param interpolated_data: Already preprocessed and interpolated data without artifacts.
        :return: Returns tuple of processed data series by applying ICLabeling, ICAs with
        marked artifact ICs and array of probabilities of each IC artifact class.
        """
        return apply_ica_component_filtering(interpolated_data, logger=self.logger)


class DatasetHandler(LoggerMixin):
    """
    Main orchestrator for EEG dataset operations including parsing, preprocessing,
    filtering, time alignment, and plotting.

    This class ties together all sub-components (DatasetParser, DatasetPreprocessor,
    DatasetFilter, TimeAligner, DatasetPlotter) and provides high-level methods
    for the full preprocessing workflow.
    """

    def __init__(
        self, experiment_name: ExperimentNames, coordinate_system: CoordinateSystems
    ):
        """
        Initialize the DatasetHandler with experiment configuration.

        :param experiment_name: Which experiment dataset to use (e.g. ExperimentNames.PSILO_MUSIC).
        :param coordinate_system: Electrode coordinate system for montage loading.
        """
        self.experiment_name = experiment_name
        (
            self.raw_data_dir,  # Raw data directory
            self.participant_map_path,  # Path to CSV mapping for each experiment (Placebo/Psilocybin)
            self.processed_data_dir,  # Output directory for the processed data
            self.interim_data_dir,  # Output directory for intermediate products
            self.coordinates_path,  # Path to electrode coordinates file
            self.excluded_electrodes_path,  # Path to CSV list of excluded electrodes.
            self.excluded_participants_path,  # Path to CSV list of excluded participants.
        ) = self._init_dataset_paths(experiment_name, coordinate_system)

        self.dataset_parser = DatasetParser(
            experiment_name, self.participant_map_path
        )
        self.dataset_metadata = self.dataset_parser.parse_dataset_filenames(
            self.raw_data_dir
        )
        self.dataset_preprocessor = DatasetPreprocessor(
            self.coordinates_path, self.excluded_electrodes_path
        )
        self.dataset_excluded_ics_metadata = self._init_excluded_ics_metadata()
        self.excluded_participants_metadata = pd.read_csv(
            self.excluded_participants_path,
            sep=";",
            dtype=str,
        )

    def _init_dataset_paths(
        self, experiment_name: ExperimentNames, coordinate_system: CoordinateSystems
    ) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
        """
        Initializes paths to working dataset.

        :param experiment_name: Variant of the experiment to process.
        :param coordinate_system: Variant of the coordinate system file we want to load.
        :return: Returns tuple of path to raw dataset directory, path to each participant
        experiment CSV mapping, path to directory where the processed results should be stored,
        path to the interim data directory, path to the file where the coordinates of the
        electrodes are stored, and path where the electrodes selected for exclusion are stored
        (due problematic position in head).
        """
        raw_data_dir, participant_map_path = ProjectPaths.get_experiment_data_dir(
            experiment_name, is_processed=False
        )
        processed_data_dir, _ = ProjectPaths.get_experiment_data_dir(
            experiment_name, is_processed=True
        )
        interim_data_dir = ProjectPaths.get_experiment_interim_dir(experiment_name)
        coordinates_path, excluded_electrodes_path = (
            ProjectPaths.get_coordinates_file_path(coordinate_system)
        )
        excluded_participants_path = (
            ProjectPaths.EXCLUDED_PARTICIPANTS_DIR / f"{experiment_name.value}.csv"
        )

        return (
            raw_data_dir,
            participant_map_path,
            processed_data_dir,
            interim_data_dir,
            coordinates_path,
            excluded_electrodes_path,
            excluded_participants_path,
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
        preload=True,
    ) -> mne.io.Raw | np.ndarray:
        """
        Loads one EEG sequence (one data example).

        :param data_filename: Name of the file containing the wanted data.
        :param is_processed: Flag whether the data to load is already processed or not
        (from where we want to load the data).
        :param processed_data_type: Type of the processed file to load
        :param preload: Whether to preload data into memory.
        :return: Returns loaded data in the Raw data type.
        """
        return load_data_file(
            self.raw_data_dir,
            self.processed_data_dir,
            data_filename,
            is_processed=is_processed,
            processed_data_type=processed_data_type,
            preload=preload,
            interim_data_dir=self.interim_data_dir,
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
        return get_preprocessing_results_path(
            self.processed_data_dir,
            filename,
            data_type,
            interim_data_dir=self.interim_data_dir,
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
        save_data_file(
            data,
            self.processed_data_dir,
            filename,
            data_type,
            logger=self.logger,
            interim_data_dir=self.interim_data_dir,
        )

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
        ic_reduced_data, ica_components, ic_probabilities = (
            self.dataset_preprocessor.apply_ica_component_filtering(interpolated_data)
        )

        ic_reduced_data.add_channels([aux_data], force_update_info=True)

        # Store final processed data.
        self.save_data_file(
            ic_reduced_data, filename_base, PreprocessedDataVariants.RAW_AFTER_ICA
        )
        if save_processing_info:
            # Save found ICA components and its probabilities to belong to selected classes.
            self.save_data_file(
                ica_components,
                filename_base,
                data_type=PreprocessedDataVariants.ICA_COMPONENTS,
            )
            self.save_data_file(
                ic_probabilities,
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
        ics_probabilities = get_ic_labeling_probabilities(
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
            excluded_ics_rows += self.get_one_original_filename_ic_metadata(
                row[SingleDataMetadata.FILENAME]
            )
        self.logger.info("All excluded ICs time series generated.")

        # Create excluded ICs metadata pandas dataframe and store them into CSV file.
        self.logger.info("Storing excluded ICs metadata to CSV file")
        self.dataset_excluded_ics_metadata = pd.DataFrame(excluded_ics_rows)
        csv_path = self._get_excluded_ics_mapping_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_excluded_ics_metadata.to_csv(csv_path)

    def plot_all_one_variant(self, plot_variant: str = "power_spectrum"):
        """
        Plot all dataset Raw dataseries object using MNE plotting functionalities from `compute_psd` base.

        :param plot_variant: Which plot we want to create. Either "power_spectrum", or "topomap".
        """
        self.logger.info(f"Plotting {plot_variant} for all dataset.")
        for i, row in self.dataset_metadata.iterrows():
            # Iterate through all data from the provided dataset.
            original_filename = row[SingleDataMetadata.FILENAME]
            self.logger.info(f"Plotting original file: {original_filename}")

            for data_variant in RAW_DATA_VARIANTS:
                if data_variant == PreprocessedDataVariants.RAW_CROPPED:
                    # Do not plot cropped raw data, because it usually does not exist
                    # and does not differ that much from after ICA variant.
                    continue
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
                        experiment_name=self.experiment_name.value,
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
                    for j, excluded_row in excluded_metadata.iterrows():
                        # Plot each excluded IC dataseries.
                        self.logger.info(
                            f"Plotting IC excluded component: {excluded_row[ExcludedICsMetadata.IC_ID.value]}"
                        )
                        raw_data = self.load_data_file(
                            original_filename,
                            is_processed=True,
                            processed_data_type=PreprocessedDataVariants.ICA_COMPONENTS,
                        )
                        self.logger.debug(excluded_row[ExcludedICsMetadata.IC_ID.value])
                        DatasetPlotter.plot_raw_dataseries(
                            raw_data,
                            save_fig=f"{excluded_row[ExcludedICsMetadata.ORIGINAL_FILENAME.value].split('.')[0]}-{excluded_row[ExcludedICsMetadata.IC_ID.value]}-{excluded_row[ExcludedICsMetadata.IC_CATEGORY.value]}",
                            is_excluded=True,
                            excluded_ic_id=excluded_row[
                                ExcludedICsMetadata.IC_ID.value
                            ],
                            plot_variant=plot_variant,
                            variant_name=data_variant,
                            title=f"IC - {excluded_row[ExcludedICsMetadata.IC_ID.value]}, {excluded_row[ExcludedICsMetadata.IC_CATEGORY.value]}, p: {excluded_row[ExcludedICsMetadata.MAIN_PROBABILITY.value]:.2f}",
                            experiment_name=self.experiment_name.value,
                        )

        self.logger.info("Plotting successfully finished")

    def _load_all_tags(
        self,
        filtered_df: pd.DataFrame,
        data_type_to_load: PreprocessedDataVariants,
    ) -> tuple[list[TAGObject], float]:
        """
        Loads all TAG signals from the dataset based on the provided filtered metadata DataFrame.

        :param filtered_df: The filtered metadata DataFrame containing the files to load TAG signals from.
        :param data_type_to_load: The type of the processed data to load for extracting the TAG signal.
        :return: A tuple containing a list of TAGObject instances and the sample frequency of the TAG signals.
        """
        all_tags = []
        sfreqs = []

        for i, row in filtered_df.iterrows():
            filename = row[SingleDataMetadata.FILENAME]
            raw = self.load_data_file(
                filename,
                is_processed=True,
                processed_data_type=data_type_to_load,
                preload=False,
            )
            # Pick only the TAG channel — cheap to load
            raw.pick([RAW_CHANNEL_NAMES[ChannelTypes.TAG]])
            raw.load_data()
            tag_signal = raw.get_data(picks=RAW_CHANNEL_NAMES[ChannelTypes.TAG])[0]
            all_tags.append(TAGObject(filename, tag_signal))
            sfreqs.append(raw.info["sfreq"])

        assert len(set(sfreqs)) == 1, "Sample rates differ across files!"
        sfreq = sfreqs[0]

        return all_tags, sfreq

    def plot_aligned_tags(
        self,
        aligned_tags,
        time_aligner: TimeAligner,
        t_start: float = 50.0,
        time_duration: float = 50.0,
    ):
        """
        Plots the aligned TAG signals.

        :param aligned_tags: List of aligned TAG signals to plot.
        :param time_aligner: TimeAligner object containing all alignment info.
        :param t_start: Start time in seconds for the signal overlap plot.
        :param time_duration: Duration in seconds for the signal overlap plot and cross-correlation plot.
        """
        DatasetPlotter.print_correlation_statistics(aligned_tags)
        DatasetPlotter.plot_alignment_correlation_heatmap(aligned_tags)
        DatasetPlotter.plot_signal_overlap(
            aligned_tags,
            time_aligner.sfreq,
            t_start=t_start,
            time_duration=time_duration,
        )
        ref = aligned_tags[time_aligner.reference_idx]
        for i, (sig, tag) in enumerate(zip(aligned_tags, time_aligner.all_tags)):
            if i == time_aligner.reference_idx:
                continue
            DatasetPlotter.plot_crosscorr_vs_shift(
                ref,
                sig,
                time_aligner.sfreq,
                max_lag_sec=time_duration,
                label1=time_aligner.all_tags[time_aligner.reference_idx].filename,
                label2=tag.filename,
            )

    def align_time_series(
        self,
        music_type: MusicTypeVariants,
        condition_type: ConditionVariants,
        exclusion_categories: list[ExclusionCategories],
        plot_alignment_results: bool = False,
        data_type_to_load: PreprocessedDataVariants = PreprocessedDataVariants.RAW_AFTER_ICA,
    ) -> tuple[list[np.ndarray], TimeAligner]:
        """
        Aligns the signals of the participants based on the TAG signal in time.

        :param music_type: Music type to include.
        :param condition_type: Condition type to include (ConditionVariants.PLACEBO or ConditionVariants.PSILOCYBIN).
        :param exclusion_categories: Which categories of participants to exclude.
        :param plot_alignment_results: Whether to plot the alignment statistics.
        :param data_type_to_load: The type of the processed data to load for extracting the TAG signal.
        :return: Returns list of aligned TAG signals and the TimeAligner object
        containing all alignment info (useful for future signal alignment of the EEG data).
        """
        filtered_df = DatasetFilter.filter_dataset_by_all_categories(
            self.dataset_metadata,
            self.excluded_participants_metadata,
            [music_type],
            [condition_type],
            exclusion_categories,
        )
        all_tag_signals, sfreq = self._load_all_tags(filtered_df, data_type_to_load)
        time_aligner = TimeAligner(all_tag_signals, sfreq)
        aligned_tags = time_aligner.crop_to_overlap()
        if plot_alignment_results:
            self.plot_aligned_tags(aligned_tags, time_aligner)
        return aligned_tags, time_aligner

    def crop_all_raw_to_alignment(self, time_aligner: TimeAligner):
        """
        Crops all raw dataseries to the precomputed alignment.

        :param time_aligner: TimeAligner object containing all alignment info (shifts and common time window).
        """
        shifts = time_aligner.shifts
        for tag_object, shift in zip(time_aligner.all_tags, shifts):
            filename = tag_object.filename
            raw = self.load_data_file(
                filename,
                is_processed=True,
                processed_data_type=PreprocessedDataVariants.RAW_AFTER_ICA,
                preload=True,
            )
            crop_start, crop_end = time_aligner.get_crop_indices_for_signal(shift)
            cropped = raw.crop(
                tmin=crop_start / time_aligner.sfreq, tmax=crop_end / time_aligner.sfreq
            )
            assert cropped.n_times == time_aligner.end - time_aligner.start + 1, (
                f"Cropped signal of file {filename} has length {cropped.n_times}, expected {time_aligner.end - time_aligner.start + 1}!"
            )

            self.save_data_file(
                cropped, filename.split(".")[0], PreprocessedDataVariants.RAW_CROPPED
            )
