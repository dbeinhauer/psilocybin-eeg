"""
This source contains definitions of data field mappings used in the project.
"""

from src.definitions.fields import ChannelTypes

# Raw channels names prefixes used in the EDF files.
RAW_CHANNEL_NAMES = {
    ChannelTypes.EEG: "EEG",  # Standard EEG channel prefix.
    ChannelTypes.EEG_REF: "EEG VREF",
    ChannelTypes.ECG: "ECG",
    ChannelTypes.TAG: "TAG",
}

# Montage channel names used in the custom montage file.
MONTAGE_CHANNEL_NAMES = {
    ChannelTypes.EEG: "E",  # Standard EEG channel prefix.
    ChannelTypes.EEG_REF: "Cz",
}

# Mapping from our channel types to MNE-Python channel types.
CHANNEL_TYPES_TO_MNE_TYPES_MAPPING = {
    ChannelTypes.EEG: "eeg",
    ChannelTypes.EEG_REF: "eeg",
    ChannelTypes.ECG: "ecg",
    ChannelTypes.TAG: "stim",  # We consider music tags as stimulus channels.
}
