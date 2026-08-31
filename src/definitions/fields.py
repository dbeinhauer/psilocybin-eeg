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
    # Auditory steady-state response experiment (no music, both conditions).
    ASSR = "assr"


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
    EXCLUSION_EXPLANATION = (
        "explanation"  # Explanation for exclusion of the data (if applicable).
    )
    CONCATENATED_PERSON_INDEX = (
        "concatenated_person_index"  # Index on axis 0 of the concatenated NumPy array.
    )


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

    :attr:`JOINED` and :attr:`JOINED_TRACKS` are *virtual* conditions: no recording
    ever carries either. Both select both real conditions at once, restricted to
    participants that contribute a recording to both (see
    :meth:`~src.filtering.dataset_filter.DatasetFilter.filter_dataset_by_all_categories`),
    so both are balanced within-subject designs. Use them wherever a single condition
    is expected.

    Neither requires any extra preprocessing. The alignment is fitted **once** over
    every recording of both conditions, so all conditions already share one time base
    and carry the same stimuli; selecting a condition — or any custom participant
    subset — is pure filtering on top of that. The two differ only in *which axis* the
    conditions are pooled along:

    * :attr:`JOINED` pools on the **subject** axis: every recording is one subject, so
      a participant appears twice. Products are named ``Joined_<MusicType>``; recover a
      condition with a subject-axis mask
      (:func:`~scripts.analysis_common.condition_index_mask`).
    * :attr:`JOINED_TRACKS` pools on the **time** axis: each participant is one subject
      whose recording is their Placebo track followed by their Psilocybin track.
      Products are named ``JoinedTracks_<MusicType>``; recover a condition with
      :meth:`~src.analysis.condition_tracks.PairedConditionTracks.condition_track`.
    """

    PLACEBO = "Placebo"
    PSILOCYBIN = "Psilocybin"
    JOINED = "Joined"
    JOINED_TRACKS = "JoinedTracks"


# The conditions a recording can actually carry. The virtual conditions are excluded:
# they are selectors over these, never values found in dataset metadata.
REAL_CONDITIONS = (ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN)

# Conditions that select both real conditions and restrict to complete participant
# pairs. They differ only in the axis the two conditions are pooled along.
JOINED_CONDITIONS = (ConditionVariants.JOINED, ConditionVariants.JOINED_TRACKS)


class MusicTypeVariants(Enum):
    """
    All variants of music used in the experiment, values are ids from filenames in lowercase.
    """

    CLASSICAL = "CLASSIC"
    PSYTRANCE = "PSYTRANCE"
    # Placeholder "music type" for the ASSR experiment, which has no music dimension.
    ASSR = "ASSR"


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


class ExclusionCategories(Enum):
    """
    All reasons for exclusion of the experiment series from the data.
    """

    BAD_MUSIC = "bad_music"  # Wrong TAG channel signal
    BAD_POWER_SPECTRUM = (
        "bad_power_spectrum"  # Abnormal power spectrum (bad data quality).
    )
    MISSING_TRIALS = (
        "missing_trials"  # Some of the trials are missing for the participant
    )
    ARTIFACTS = "artifacts"  # Too many artifacts in the data (also after preprocessing)
    WRONG_CONDITION = (
        "wrong_condition"  # Recorded under the wrong measurement condition.
    )
    ORPHAN_BGIN = "orphan_bgin"  # `bgin` annotation label without corresponding stimulus label (fam+ in ASSR).


# Exclusion categories applied when *fitting* an alignment, per experiment.
#
# Deliberately minimal, and not the same thing as the exclusions an analysis applies.
# A recording dropped here can never be selected later, because it will not have been
# aligned; and because the alignment trims to group minima, one bad recording degrades
# the result for everyone. Only categories that would corrupt the fit belong here:
# BAD_MUSIC (a wrong TAG channel wrecks the cross-correlation) and WRONG_CONDITION
# (the recording is not what its metadata claims).
#
# Both the alignment scripts and
# :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.load_pre_alignment_data` read this,
# so the stored crops and the wavelet cache are guaranteed to describe the same splice.
ALIGNMENT_EXCLUSIONS: dict[ExperimentNames, tuple[ExclusionCategories, ...]] = {
    ExperimentNames.PSILO_MUSIC: (ExclusionCategories.BAD_MUSIC,),
    ExperimentNames.ASSR: (ExclusionCategories.WRONG_CONDITION,),
}


class FrequencyBandNames(Enum):
    """
    Standard EEG frequency band names used for band-specific analyses.
    """

    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


class AnalysisVariants(Enum):
    """
    All analysis keywords accepted by the unified analysis entrypoint.
    """

    ISC = "isc"
    MEAN_VARIANCE = "mean_variance"
    WAVELET_POWER = "wavelet_power"
    WAVELET_PHASE = "wavelet_phase"


class SpectrumTypeVariants(Enum):
    """
    Canonical subdirectory names splitting analysis outputs (plots, result
    CSVs) and wavelet caches into full-spectrum vs per-frequency-band layouts.

    Each value is used verbatim as a subdirectory name when resolving output
    paths, e.g. ``<save_dir>/<SpectrumTypeVariants.BROADBAND.value>/...``.
    """

    BROADBAND = "broadband"  # Full-spectrum (non-band-split) outputs and caches.
    BANDS = "bands"  # Per-frequency-band outputs (each band in its own subdir).


class IvaVariants(Enum):
    """
    All IVA decomposition variants, values are the canonical subdirectory names
    shared by the plot layout and by the stored component products
    (:mod:`src.io.iva_store`).

    The value names *which axis the decomposition treats as the mixing
    (independent) dimension* and how the conditions are pooled, because that pair
    of choices decides the shape of every product a variant can offer:

    * :attr:`CHANNEL` — mixing = channels, samples = time x frequency. Each
      component is a shared spectro-temporal source ``(F, T)`` plus a per-recording
      channel topography ``(C,)``.
    * :attr:`FREQUENCY_CHANNEL` — mixing = channel x frequency, samples = time. Each
      component is a timecourse ``(T,)`` plus a spectro-spatial pattern ``(F, C)``;
      there is no per-component time-frequency map.
    * :attr:`TIME` — mixing = time, samples = channel x frequency. The transposed
      companion of :attr:`FREQUENCY_CHANNEL`: a temporal pattern ``(T,)`` plus a
      spectro-spatial score map ``(F, C)``.
    * :attr:`CHANNEL_JOINED` — :attr:`CHANNEL` run on the subject-axis join
      (:attr:`ConditionVariants.JOINED`): every recording is one dataset, so a
      participant occupies one row per condition.
    * :attr:`CHANNEL_JOINED_TRACKS` — :attr:`CHANNEL` run on the time-axis join
      (:attr:`ConditionVariants.JOINED_TRACKS`): one dataset per participant, whose
      time axis carries both condition tracks end to end, so the row is shared by
      the conditions and the split is a slice of the time axis.
    """

    CHANNEL = "iva_channel"
    FREQUENCY_CHANNEL = "iva_frequency_channel"
    TIME = "iva_time"
    CHANNEL_JOINED = "iva_channel_joined"
    CHANNEL_JOINED_TRACKS = "iva_channel_joined_tracks"


class IvaComponentArrays(Enum):
    """
    Canonical names of the per-component arrays kept in the IVA results store.

    Every array is indexed ``(recording, component, ...)`` — the leading two axes
    are the same for all of them, so per-recording bookkeeping (the participant and
    condition of each row) applies unchanged to any of them. Which arrays a run
    writes depends on its :class:`IvaVariants`; a reader must therefore ask for a
    name rather than assume it is present.

    * :attr:`TF_MAP` — ``(S, K, F, T)`` per-recording time-frequency source map.
    * :attr:`CHANNEL_PATTERN` — ``(S, K, C)`` forward (mixing) channel topography.
    * :attr:`TIMECOURSE` — ``(S, K, T)`` temporal profile of the component.
    * :attr:`SPECTRAL_PROFILE` — ``(S, K, F)`` spectral profile of the component.
    * :attr:`FREQUENCY_CHANNEL_PATTERN` — ``(S, K, F, C)`` spectro-spatial pattern.
    """

    TF_MAP = "tf_map"
    CHANNEL_PATTERN = "channel_pattern"
    TIMECOURSE = "timecourse"
    SPECTRAL_PROFILE = "spectral_profile"
    FREQUENCY_CHANNEL_PATTERN = "frequency_channel_pattern"


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
    RAW_CROPPED = "cropped"  # Raw dataseries after cropping to the common time window across all participants (after time alignment). The alignment is fitted once over every recording of both conditions, so all conditions share one time base and any participant subset can be selected afterwards without re-aligning.
    CONCATENATED = "concatenated"  # Concatenated data across all participants (after stacking into one array).


# All data variants that are `mne.io.Raw` types.
RAW_DATA_VARIANTS = [
    PreprocessedDataVariants.RAW_BEFORE_ICA,
    PreprocessedDataVariants.RAW_AFTER_ICA,
    PreprocessedDataVariants.RAW_EXCLUDED_IC,
    PreprocessedDataVariants.RAW_CROPPED,
]

# Intermediate data variants that should be stored in the interim directory
# (not the final processed directory).
INTERIM_DATA_VARIANTS = [
    PreprocessedDataVariants.RAW_BEFORE_ICA,
    PreprocessedDataVariants.ICA_COMPONENTS,
    PreprocessedDataVariants.IC_PROBABILITIES,
]
