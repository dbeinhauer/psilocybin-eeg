"""
Tests for src/definitions/fields.py — Enum definitions and constants.
"""

import pytest

from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
    ExcludedICsMetadata,
    EEGConditions,
    ConditionVariants,
    MusicTypeVariants,
    ChannelTypes,
    ICLabelComponentsClasses,
    ExclusionCategories,
    AnalysisVariants,
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
    INTERIM_DATA_VARIANTS,
)


class TestExperimentNames:
    def test_psilo_music_value(self):
        assert ExperimentNames.PSILO_MUSIC.value == "psilo_music"

    def test_assr_value(self):
        assert ExperimentNames.ASSR.value == "assr"


class TestCoordinateSystems:
    def test_hydrogel_257(self):
        assert CoordinateSystems.HYDROGEL_257.value == "GSN-HydroCel-257"

    def test_hydrogel_257_no_fiducials(self):
        assert (
            CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value
            == "GSN-HydroCel-257_no-fiducials"
        )


class TestSingleDataMetadata:
    def test_all_fields_present(self):
        expected = {
            "participant_id",
            "eeg_condition_id",
            "condition",
            "music_type",
            "filename",
            "explanation",
        }
        values = {m.value for m in SingleDataMetadata}
        assert values == expected


class TestEEGConditions:
    def test_condition_a(self):
        assert EEGConditions.CONDITION_A.value == "A"

    def test_condition_b(self):
        assert EEGConditions.CONDITION_B.value == "B"

    def test_only_two_conditions(self):
        assert len(EEGConditions) == 2


class TestConditionVariants:
    def test_placebo(self):
        assert ConditionVariants.PLACEBO.value == "Placebo"

    def test_psilocybin(self):
        assert ConditionVariants.PSILOCYBIN.value == "Psilocybin"


class TestMusicTypeVariants:
    def test_classical(self):
        assert MusicTypeVariants.CLASSICAL.value == "CLASSIC"

    def test_psytrance(self):
        assert MusicTypeVariants.PSYTRANCE.value == "PSYTRANCE"

    def test_assr(self):
        assert MusicTypeVariants.ASSR.value == "ASSR"


class TestChannelTypes:
    def test_eeg(self):
        assert ChannelTypes.EEG.value == "eeg"

    def test_tag(self):
        assert ChannelTypes.TAG.value == "tag"

    def test_ecg(self):
        assert ChannelTypes.ECG.value == "ecg"


class TestICLabelComponentsClasses:
    def test_all_classes_present(self):
        expected = {"brain", "muscle", "eog", "ecg", "line_noise", "ch_noise", "other"}
        values = {c.value for c in ICLabelComponentsClasses}
        assert values == expected


class TestExclusionCategories:
    def test_all_categories(self):
        expected = {
            "bad_music",
            "bad_power_spectrum",
            "missing_trials",
            "artifacts",
            "wrong_condition",
        }
        values = {c.value for c in ExclusionCategories}
        assert values == expected


class TestPreprocessedDataVariants:
    def test_raw_before_ica(self):
        assert PreprocessedDataVariants.RAW_BEFORE_ICA.value == "before_ica"

    def test_raw_after_ica(self):
        assert PreprocessedDataVariants.RAW_AFTER_ICA.value == "after_ica"

    def test_ica_components(self):
        assert PreprocessedDataVariants.ICA_COMPONENTS.value == "ica_components"

    def test_ic_probabilities(self):
        assert PreprocessedDataVariants.IC_PROBABILITIES.value == "ic_probabilities"

    def test_raw_cropped(self):
        assert PreprocessedDataVariants.RAW_CROPPED.value == "cropped"

    def test_concatenated(self):
        assert PreprocessedDataVariants.CONCATENATED.value == "concatenated"


class TestAnalysisVariants:
    def test_contains_expected_analysis_keywords(self):
        values = {v.value for v in AnalysisVariants}
        assert values == {"isc", "mean_variance", "wavelet_power", "wavelet_phase"}


class TestRawDataVariants:
    def test_contains_expected_members(self):
        assert PreprocessedDataVariants.RAW_BEFORE_ICA in RAW_DATA_VARIANTS
        assert PreprocessedDataVariants.RAW_AFTER_ICA in RAW_DATA_VARIANTS
        assert PreprocessedDataVariants.RAW_EXCLUDED_IC in RAW_DATA_VARIANTS
        assert PreprocessedDataVariants.RAW_CROPPED in RAW_DATA_VARIANTS

    def test_does_not_contain_non_raw(self):
        assert PreprocessedDataVariants.ICA_COMPONENTS not in RAW_DATA_VARIANTS
        assert PreprocessedDataVariants.IC_PROBABILITIES not in RAW_DATA_VARIANTS
        assert PreprocessedDataVariants.CONCATENATED not in RAW_DATA_VARIANTS


class TestInterimDataVariants:
    def test_contains_expected_members(self):
        assert PreprocessedDataVariants.RAW_BEFORE_ICA in INTERIM_DATA_VARIANTS
        assert PreprocessedDataVariants.ICA_COMPONENTS in INTERIM_DATA_VARIANTS
        assert PreprocessedDataVariants.IC_PROBABILITIES in INTERIM_DATA_VARIANTS

    def test_does_not_contain_final_outputs(self):
        assert PreprocessedDataVariants.RAW_AFTER_ICA not in INTERIM_DATA_VARIANTS
        assert PreprocessedDataVariants.RAW_CROPPED not in INTERIM_DATA_VARIANTS
        assert PreprocessedDataVariants.CONCATENATED not in INTERIM_DATA_VARIANTS
