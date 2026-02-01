"""
This script contains classes and functions to parse dataset filenames and extract metadata from them.
"""

import re
from pathlib import Path
from enum import Enum
from typing import Type

import mne
import pandas as pd

from src.utils.logging_config import LoggerMixin, LogLevel
from src.definitions.fields import (
    SingleDataMetadata,
    ConditionVariants,
    MusicTypeVariants,
    SingleDataMetadataTypes,
    ChannelTypes,
    EEGConditions,
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

    Format: PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{rest_of_filename_date}.edf
    """

    def __init__(self, participant_map_path: Path, coordinates_file_path: Path):
        """
        :param participant_map_path: Path to CSV file containing participant mapping information.
        """
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

    def parse_filename(
        self, filename: str
    ) -> dict[SingleDataMetadata, SingleDataMetadataTypes] | None:
        """
        Parse EDF filename to extract metadata.

        Format: PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{rest_of_filename_date}.edf

        :param filename: Name of the EDF file to parse
        :return: Dictionary with keys corresponding to SingleDataMetadata.
                 Returns None if filename doesn't match expected pattern or contains invalid metadata.
        """
        pattern = r"PSI(\d{3})_EEG([A-Za-z])_MUSIC_(\w+)_EC_(.+)\.edf"

        match = re.match(pattern, filename)

        if not match:
            self.logger.error(f"Filename does not match expected pattern: {filename}")
            return None

        # Extract groups
        participant_id = match.group(1)
        condition_value = match.group(2)
        music_type_value = match.group(3)

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
        music_type_value = check_enum_value_in_variants(
            MusicTypeVariants, music_type_value.upper()
        )
        if music_type_value is None:
            self.logger.error(
                f"Invalid music type value: '{music_type_value}' in filename: {filename}"
            )
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
