"""
Tests for src/filtering/dataset_filter.py — DataFrame filtering by metadata.
"""

import pytest
import pandas as pd

from src.definitions.fields import (
    SingleDataMetadata,
    ConditionVariants,
    MusicTypeVariants,
    ExclusionCategories,
)
from src.filtering.dataset_filter import DatasetFilter


@pytest.fixture
def dataset_metadata():
    """Sample dataset metadata DataFrame using enum keys."""
    return pd.DataFrame(
        {
            SingleDataMetadata.PARTICIPANT_ID: [
                "001",
                "001",
                "002",
                "002",
                "003",
                "003",
            ],
            SingleDataMetadata.CONDITION: [
                ConditionVariants.PLACEBO,
                ConditionVariants.PLACEBO,
                ConditionVariants.PSILOCYBIN,
                ConditionVariants.PSILOCYBIN,
                ConditionVariants.PLACEBO,
                ConditionVariants.PLACEBO,
            ],
            SingleDataMetadata.MUSIC_TYPE: [
                MusicTypeVariants.CLASSICAL,
                MusicTypeVariants.PSYTRANCE,
                MusicTypeVariants.CLASSICAL,
                MusicTypeVariants.PSYTRANCE,
                MusicTypeVariants.CLASSICAL,
                MusicTypeVariants.PSYTRANCE,
            ],
            SingleDataMetadata.FILENAME: [
                "f1.edf",
                "f2.edf",
                "f3.edf",
                "f4.edf",
                "f5.edf",
                "f6.edf",
            ],
        }
    )


@pytest.fixture
def excluded_metadata():
    """Sample excluded participants metadata using string keys."""
    return pd.DataFrame(
        {
            SingleDataMetadata.PARTICIPANT_ID.value: ["003", "003"],
            SingleDataMetadata.CONDITION.value: ["Placebo", "Placebo"],
            SingleDataMetadata.MUSIC_TYPE.value: ["CLASSIC", "PSYTRANCE"],
            SingleDataMetadata.EXCLUSION_EXPLANATION.value: [
                "bad_music",
                "bad_music",
            ],
        }
    )


class TestFilterByMusicType:
    def test_filter_classical_only(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_music_type(
            dataset_metadata, [MusicTypeVariants.CLASSICAL]
        )
        assert len(result) == 3

    def test_filter_psytrance_only(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_music_type(
            dataset_metadata, [MusicTypeVariants.PSYTRANCE]
        )
        assert len(result) == 3

    def test_filter_both_types(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_music_type(
            dataset_metadata,
            [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE],
        )
        assert len(result) == 6

    def test_empty_filter_returns_empty(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_music_type(dataset_metadata, [])
        assert len(result) == 0


class TestFilterByCondition:
    def test_filter_placebo(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_condition(
            dataset_metadata, [ConditionVariants.PLACEBO]
        )
        assert len(result) == 4

    def test_filter_psilocybin(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_condition(
            dataset_metadata, [ConditionVariants.PSILOCYBIN]
        )
        assert len(result) == 2

    def test_filter_both_conditions(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_condition(
            dataset_metadata,
            [ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN],
        )
        assert len(result) == 6


class TestFilterByParticipantIds:
    def test_filter_single_participant(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_participant_ids(
            dataset_metadata, ["001"]
        )
        assert len(result) == 2

    def test_filter_multiple_participants(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_participant_ids(
            dataset_metadata, ["001", "002"]
        )
        assert len(result) == 4

    def test_filter_nonexistent_participant(self, dataset_metadata):
        result = DatasetFilter.filter_dataset_by_participant_ids(
            dataset_metadata, ["999"]
        )
        assert len(result) == 0


class TestFilterByExclusionCategories:
    def test_filter_bad_music(self, excluded_metadata):
        result = DatasetFilter.filter_dataset_by_exclusion_categories(
            excluded_metadata, [ExclusionCategories.BAD_MUSIC]
        )
        assert len(result) == 2

    def test_filter_missing_category(self, excluded_metadata):
        result = DatasetFilter.filter_dataset_by_exclusion_categories(
            excluded_metadata, [ExclusionCategories.BAD_POWER_SPECTRUM]
        )
        assert len(result) == 0


class TestGetUniqueValues:
    def test_get_unique_participant_ids(self, dataset_metadata):
        result = DatasetFilter.get_unique_values(
            dataset_metadata, SingleDataMetadata.PARTICIPANT_ID
        )
        assert set(result) == {"001", "002", "003"}

    def test_get_unique_conditions(self, dataset_metadata):
        result = DatasetFilter.get_unique_values(
            dataset_metadata, SingleDataMetadata.CONDITION
        )
        assert set(result) == {ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN}
