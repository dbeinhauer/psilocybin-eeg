"""
This script contains classes and functions to parse dataset filenames and extract metadata from them.
"""

import re
from pathlib import Path
from enum import Enum
from typing import Type

import mne
import pandas as pd

from src.utils.logging_config import LoggerMixin
from src.definitions.fields import (
    ExperimentNames,
    SingleDataMetadata,
    ConditionVariants,
    MusicTypeVariants,
    SingleDataMetadataTypes,
    EEGConditions,
    ICLabelComponentsClasses,
)


def check_enum_value_in_variants(enum_class: Type[Enum], value: str) -> Enum | None:
    """
    Check if a value is a valid member of an enum class.

    :param enum_class: The enum class to check against.
    :param value: The value to check.
    :return: The matching enum member if found, otherwise None.
    """
    for variant in enum_class:
        if variant.value == value:
            return variant
    return None


class DatasetParser(LoggerMixin):
    """
    A class to parse dataset filenames and extract metadata from it. The filenames are expected to follow a specific format.

    The exact format depends on the experiment (see ``_FILENAME_PATTERNS``):

    - PSILO_MUSIC: PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{rest}.edf
    - ASSR:        PSI{participant_id}_EEG{condition_id}_ASSR_{rest}.edf

    The experimental condition (Placebo/Psilocybin) is always derived from the
    participant mapping. Experiments without a music dimension (e.g. ASSR) use a
    fixed default music type instead of parsing it from the filename (see
    ``_DEFAULT_MUSIC_TYPE``).
    """

    # Per-experiment filename regex. Group 1 = participant id, group 2 = EEG
    # condition id ('A'/'B'). Experiments with a music dimension capture the music
    # type in group 3.
    _FILENAME_PATTERNS = {
        ExperimentNames.PSILO_MUSIC: r"PSI(\d{3})_EEG([A-Za-z])_MUSIC_(\w+)_EC_(.+)\.edf",
        ExperimentNames.ASSR: r"PSI(\d{3})_EEG([A-Za-z])_ASSR_(.+)\.edf",
    }

    # Music type assigned for experiments that have no music dimension encoded in
    # their filenames. Experiments not listed here parse the music type from the
    # filename instead.
    _DEFAULT_MUSIC_TYPE = {
        ExperimentNames.ASSR: MusicTypeVariants.ASSR,
    }

    def __init__(
        self,
        experiment_name: ExperimentNames,
        participant_map_path: Path,
    ):
        """
        :param experiment_name: Which experiment dataset is being parsed. Determines
            the expected filename pattern and how the music type is assigned.
        :param participant_map_path: Path to CSV file containing participant mapping
            information (used to derive the Placebo/Psilocybin condition). Required:
            parsing a dataset without the condition mapping is not supported.
        :raises FileNotFoundError: If the participant mapping file does not exist.
        """
        if participant_map_path is None or not participant_map_path.exists():
            raise FileNotFoundError(
                f"Participant mapping file is required but was not found: {participant_map_path}"
            )
        self.experiment_name = experiment_name
        self.participant_map = pd.read_csv(participant_map_path, sep=";")

    def _assign_condition_variant(
        self,
        condition_value: str,
        participant_id: str,
    ) -> ConditionVariants | None:
        """
        Assigns experiment condition (placebo/psilocybin) based on provided metadata.

        :param condition_value: One letter code of the condition (either 'A' or 'B')
        :param participant_id: Numerical 3 digit part of the participant ID (e.g. '001', '023', etc.)
        :return: Returns corresponding ConditionVariants enum if valid, None otherwise.
        """

        match_participant_id = "PSI" + participant_id
        match_condition_value = "EEG" + condition_value

        # Find matching row in participant map
        matching_rows = self.participant_map[
            (self.participant_map["participant"] == match_participant_id)
            & (self.participant_map["eeg"] == match_condition_value)
        ]

        condition_string = ""
        if not matching_rows.empty:
            condition_string = matching_rows.iloc[0]["condition"]
        else:
            self.logger.warning(
                f"No matching condition variant found for participant {participant_id} and condition {condition_value}"
            )
            return None

        return check_enum_value_in_variants(ConditionVariants, condition_string)

    def _assign_music_type(self, match: re.Match) -> MusicTypeVariants | None:
        """
        Assigns the music type for the current experiment.

        For experiments without a music dimension (see ``_DEFAULT_MUSIC_TYPE``) a
        fixed default is returned. Otherwise the music type is parsed from the
        third regex group of the filename.

        :param match: Regex match of the filename against the experiment pattern.
        :return: Corresponding MusicTypeVariants enum if valid, None otherwise.
        """
        if self.experiment_name in self._DEFAULT_MUSIC_TYPE:
            return self._DEFAULT_MUSIC_TYPE[self.experiment_name]
        return check_enum_value_in_variants(MusicTypeVariants, match.group(3).upper())

    def parse_filename(
        self, filename: str
    ) -> dict[SingleDataMetadata, SingleDataMetadataTypes] | None:
        """
        Parse EDF filename to extract metadata.

        The expected filename format depends on ``self.experiment_name`` (see
        ``_FILENAME_PATTERNS``).

        :param filename: Name of the EDF file to parse
        :return: Dictionary with keys corresponding to SingleDataMetadata.
                 Returns None if filename doesn't match expected pattern or contains invalid metadata.
        """
        pattern = self._FILENAME_PATTERNS[self.experiment_name]

        match = re.match(pattern, filename)

        if not match:
            self.logger.error(f"Filename does not match expected pattern: {filename}")
            return None

        # Extract groups
        participant_id = match.group(1)
        condition_value = match.group(2)

        results = {
            SingleDataMetadata.PARTICIPANT_ID: participant_id,
            SingleDataMetadata.FILENAME: filename,
        }

        # Parse EEG condition id.
        eeg_condition_id = check_enum_value_in_variants(EEGConditions, condition_value)
        if eeg_condition_id is None:
            self.logger.error(
                f"Invalid EEG condition ID value: '{condition_value}' in filename: {filename}"
            )
            return None
        results[SingleDataMetadata.EEG_CONDITION_ID] = eeg_condition_id

        # Parse music type.
        music_type_value = self._assign_music_type(match)
        if music_type_value is None:
            self.logger.error(f"Invalid music type value in filename: {filename}")
            return None
        results[SingleDataMetadata.MUSIC_TYPE] = music_type_value

        # Parse condition variant.
        condition_variant = self._assign_condition_variant(
            condition_value, participant_id
        )
        if condition_variant is None:
            self.logger.error(
                f"Invalid condition value: '{condition_value}' for participant ID: '{participant_id}' in filename: {filename}"
            )
            return None
        results[SingleDataMetadata.CONDITION] = condition_variant

        return results

    def parse_dataset_filenames(self, dataset_path: Path) -> pd.DataFrame:
        """
        Parse all EDF filenames in a dataset directory and compile metadata into a DataFrame.

        :param dataset_path: Path to the dataset directory containing EDF files.
        :return: DataFrame with metadata for all successfully parsed files.
        """
        metadata_records = []

        for edf_file in dataset_path.glob("*.edf"):
            metadata = self.parse_filename(edf_file.name)
            if metadata:
                metadata_records.append(metadata)

        # Convert list of metadata dicts to DataFrame
        metadata_df = pd.DataFrame(metadata_records)
        return metadata_df

    @staticmethod
    def get_ic_exclusion_map(
        ica_components: mne.preprocessing.ICA,
    ) -> dict[int, ICLabelComponentsClasses]:
        """
        Retrieves all excluded channels and marks them with their components.

        :param ica_components: ICA components with marked channels to exclude.
        :return: Returns dictionary of channel to exclude and its labeled reason for exclusion.
        """
        # Get all ICs to excludes and IC labels to all ICA components.
        ic_ids_exclude = ica_components.exclude
        ic_labels = ica_components.labels_

        # Mark each IC index for exclusion to its label.
        return {
            int(idx): check_enum_value_in_variants(ICLabelComponentsClasses, label)
            for idx in ic_ids_exclude
            for label, ids_list in ic_labels.items()
            if idx in ids_list
        }
