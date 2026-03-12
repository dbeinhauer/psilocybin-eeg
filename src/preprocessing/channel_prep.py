"""
This module contains functions for EEG channel preparation, including
renaming, type setting, montage application, and channel exclusion.
"""

import logging
import re
from pathlib import Path

import mne

from src.definitions.fields import ChannelTypes
from src.definitions.mappings import (
    RAW_CHANNEL_NAMES,
    MONTAGE_CHANNEL_NAMES,
    CHANNEL_TYPES_TO_MNE_TYPES_MAPPING,
)

_logger = logging.getLogger(__name__)


def load_coordinates_file(coordinates_file_path: Path) -> mne.channels.DigMontage:
    """
    Loads coordinates (montage) from the provided SFP file.

    :param coordinates_file_path: Path to the coordinates file.
    :return: Returns the montage object loaded from the provided file.
    """
    return mne.channels.read_custom_montage(coordinates_file_path)


def rename_channels(data: mne.io.Raw, logger=None) -> mne.io.Raw:
    """
    Rename raw data channels to match the montage file naming conventions.

    :param data: MNE-Python Raw object containing the EEG data.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: MNE-Python Raw object with renamed channels to match the montage naming conventions.
    """
    log = logger or _logger

    log.info("Renaming channels to match montage file...")
    channel_name_mapping = {}
    for ch_name in data.ch_names:
        # Handle "EEG VREF" -> "Cz"
        if ch_name == RAW_CHANNEL_NAMES[ChannelTypes.EEG_REF]:
            channel_name_mapping[ch_name] = MONTAGE_CHANNEL_NAMES[
                ChannelTypes.EEG_REF
            ]
            # Handle "EEG {num}" -> "E{num}"
        elif ch_name.startswith(RAW_CHANNEL_NAMES[ChannelTypes.EEG]):
            # Extract numbers using regex (handles multiple spaces, etc.)
            match = re.search(
                rf"{RAW_CHANNEL_NAMES[ChannelTypes.EEG]}\s+(\d+)", ch_name
            )
            if match:
                num = match.group(1)
                channel_name_mapping[ch_name] = (
                    f"{MONTAGE_CHANNEL_NAMES[ChannelTypes.EEG]}{num}"
                )

    # Rename channels to match montage:
    data.rename_channels(channel_name_mapping)
    log.info("Channels renamed successfully.")
    return data


def set_channel_types(data: mne.io.Raw, logger=None) -> mne.io.Raw:
    """
    Set channel types in the raw data according to predefined mappings.

    Note: This function assumes that the channel names have already been renamed to match the
    montage naming conventions e.g. by running function `rename_channels`.

    :param data: MNE-Python Raw object containing the EEG data.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: MNE-Python Raw object with updated channel types.
    """
    log = logger or _logger

    log.info("Setting channel types...")
    channel_types_mapping = {
        ch: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.EEG]
        for ch in data.ch_names
        if ch in MONTAGE_CHANNEL_NAMES.values()
    }  # All channels in montage file are EEG channels (either normal or reference).
    channel_types_mapping.update(
        {
            RAW_CHANNEL_NAMES[ChannelTypes.ECG]: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[
                ChannelTypes.ECG
            ],
            RAW_CHANNEL_NAMES[ChannelTypes.TAG]: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[
                ChannelTypes.TAG
            ],
        }
    )
    data.set_channel_types(channel_types_mapping)
    log.info("Channel types set successfully.")

    return data


def add_coordinates_montage(
    data: mne.io.Raw, montage: mne.channels.DigMontage, logger=None
) -> mne.io.Raw:
    """
    Add channel coordinates montage to the raw data and rename channel based on the template.

    :param data: MNE-Python Raw object containing the EEG data.
    :param montage: MNE-Python DigMontage object containing the channel coordinates.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: MNE-Python Raw object with added montage.
    """
    log = logger or _logger

    log.info("Adding channel coordinates montage...")
    data = set_channel_types(rename_channels(data, logger=log), logger=log).set_montage(
        montage, on_missing="warn"
    )
    log.info("Montage added successfully.")

    return data


def exclude_selected_channels(
    data: mne.io.Raw, electrodes_to_exclude: list[str], logger=None
) -> mne.io.Raw:
    """
    Exclude the selected channels from the raw data.

    We want to exclude the selected electrodes with the problematic placement
    (the electrodes are typically located at the cheeks or other problematic
    positions in the head.)

    :param data: Data from which we want to exclude the electrodes.
    :param electrodes_to_exclude: List of electrode names to exclude.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns provided data without the selected electrodes.
    """
    log = logger or _logger

    log.info("Dropping the selected channels (from problematic positions).")
    return data.drop_channels(electrodes_to_exclude)


def crop_start_and_end_of_dataseries(
    data: mne.io.Raw, start_offset: float = 10.0, end_offset: float = 10.0, logger=None
) -> mne.io.Raw:
    """
    Crops start and end time blocks from the data (they are always noisy).

    :param data: Data to be cropped.
    :param start_offset: Start offset in seconds.
    :param end_offset: End offset in seconds.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns cropped raw data series.
    """
    log = logger or _logger

    log.info(
        f"Cropping the first {start_offset} and last {end_offset} seconds from the dataseries."
    )
    start_time = data.times[0]
    end_time = data.times[-1]

    return data.crop(tmin=start_time + start_offset, tmax=end_time - end_offset)
