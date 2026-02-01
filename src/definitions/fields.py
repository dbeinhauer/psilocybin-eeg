"""
This source contains definitions of data fields and their variants across the project.
"""

from enum import Enum


class ExperimentNames(Enum):
    """
    All names of the processed experiments (also name of the dataset directories).
    """

    # Experiment with placebo and psilocybin, with music listening.
    PSILO_MUSIC = "psilo_music"


class CoordinateSystems(Enum):
    """
    All coordinate systems used in the project.
    """

    HYDROGEL_257 = "GSN-HydroCel-257"
    HYDROGEL_257_NO_FIDUCIALS = "GSN-HydroCel-257_no-fiducials"


class SingleDataMetadata(Enum):
    """
    All metadata fields per single data
    """

    PARTICIPANT_ID = "participant_id"
    EEG_CONDITION_ID = (
        "eeg_condition_id"  # Raw condition id from filename (e.g., 'A' or 'B')
    )
    CONDITION = "condition"  # Placebo or Psilocybin
    MUSIC_TYPE = "music_type"
    FILENAME = "filename"  # Exact filename of the data file (to know which file contains the data).


class EEGConditions(Enum):
    """
    All EEG condition ids from filenames.
    """

    CONDITION_A = "A"
    CONDITION_B = "B"


class ConditionVariants(Enum):
    """
    All variants of experimental conditions, values are ids from filenames.
    """

    PLACEBO = "Placebo"
    PSILOCYBIN = "Psilocybin"


class MusicTypeVariants(Enum):
    """
    All variants of music used in the experiment, values are ids from filenames in lowercase.
    """

    CLASSICAL = "CLASSIC"
    PSYTRANCE = "PSYTRANCE"


# All data types of the SingleDataMetadata values.
SingleDataMetadataTypes = ConditionVariants | MusicTypeVariants | str


class ChannelTypes(Enum):
    """
    All channel types used in the project.
    """

    EEG = "eeg"
    EEG_REF = "eeg_ref"  # Reference EEG channel
    ECG = "ecg"
    TAG = "tag"  # Music event channel.


class ICLabelComponentsClasses(Enum):
    """
    All components detected by the `iclabel_label_components` function.
    """

    BRAIN = "brain"
    MUSCLE = "muscle artifact"
    EYE = "eye blink"
    HEART = "heart beat"
    LINE = "line noise"
    CHANNEL = "channel noise"
    OTHER = "other"
