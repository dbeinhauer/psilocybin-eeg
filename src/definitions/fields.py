from enum import Enum


class SingleDataMetadata(Enum):
    """
    All metadata fields per single data
    """

    PARTICIPANT_ID = "participant_id"
    CONDITION = "condition"  # Placebo or Psilocybin
    MUSIC_TYPE = "music_type"
    FILENAME = "filename"  # Exact filename of the data file (to know which file contains the data).


class ConditionVariants(Enum):
    """
    All variants of experimental conditions, values are ids from filenames.
    """

    PLACEBO = "A"
    PSILOCYBIN = "B"


class MusicTypeVariants(Enum):
    """
    All variants of music used in the experiment, values are ids from filenames in lowercase.
    """

    CLASSICAL = "CLASSIC"
    PSYTRANCE = "PSYTRANCE"


# All data types of the SingleDataMetadata values.
SingleDataMetadataTypes = ConditionVariants | MusicTypeVariants | str
