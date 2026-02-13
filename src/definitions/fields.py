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


class ExcludedICsMetadata(Enum):
    """
    All metadata of the excluded ICs.
    """

    ORIGINAL_FILENAME = (
        "original_filename"  # Filename of the original Raw data (for mapping).
    )
    TIMESERIES_FILENAME = (
        "timeseries_filename"  # Filename where the timeseries of the IC is stored.
    )
    IC_ID = "ic_id"  # ID of the excluded IC
    IC_CATEGORY = "ic_category"  # Category where the IC was put after IC labelling.
    TOTAL_ICS = "total_ics"  # Total number of ICs for the give data.
    MAIN_PROBABILITY = "main_probability"  # Probability of the selected class


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
    MUSCLE = "muscle"
    EYE = "eog"
    HEART = "ecg"
    LINE = "line_noise"
    CHANNEL = "ch_noise"
    OTHER = "other"


class PreprocessedDataVariants(Enum):
    """
    All variants of possible data stored during preprocessing (for quality of the preprocessing analysis).
    """

    RAW_BEFORE_ICA = "before_ica"  # Raw dataseries before ICA component reduction.
    RAW_AFTER_ICA = (
        "after_ica"  # Raw dataseries after the application of ICA component reduction.
    )
    ICA_COMPONENTS = "ica_components"  # All found ICA components
    IC_PROBABILITIES = "ic_probabilities"  # Probability distribution of the ICA components across different component classes (muscle, eye, brain etc.)

    RAW_EXCLUDED_IC = (
        "raw_excluded_ic"  # Raw dataseries of excluded component selected by ICA.
    )


# All data variants that are `mne.io.Raw` types.
RAW_DATA_VARIANTS = [
    PreprocessedDataVariants.RAW_BEFORE_ICA,
    PreprocessedDataVariants.RAW_AFTER_ICA,
    PreprocessedDataVariants.RAW_EXCLUDED_IC,
]
