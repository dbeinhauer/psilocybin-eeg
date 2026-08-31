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


class TestFilterByAllCategoriesExclusionWildcards:
    """Blank condition / music type in exclusion metadata acts as a wildcard."""

    def test_blank_music_type_excludes_all_music_types(self, dataset_metadata):
        # Participant 001 excluded with no music type -> dropped from every music
        # type of the selected (Placebo) condition.
        excluded = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID.value: ["001"],
                SingleDataMetadata.CONDITION.value: ["Placebo"],
                SingleDataMetadata.MUSIC_TYPE.value: [""],
                SingleDataMetadata.EXCLUSION_EXPLANATION.value: ["artifacts"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            dataset_metadata,
            excluded,
            [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE],
            [ConditionVariants.PLACEBO],
            [ExclusionCategories.ARTIFACTS],
        )
        assert "001" not in result[SingleDataMetadata.PARTICIPANT_ID].tolist()
        assert "003" in result[SingleDataMetadata.PARTICIPANT_ID].tolist()

    def test_blank_condition_excludes_all_conditions(self, dataset_metadata):
        # Participant 002 excluded with no condition -> dropped regardless of the
        # requested condition.
        excluded = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID.value: ["002"],
                SingleDataMetadata.CONDITION.value: [""],
                SingleDataMetadata.MUSIC_TYPE.value: [""],
                SingleDataMetadata.EXCLUSION_EXPLANATION.value: ["missing_trials"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            dataset_metadata,
            excluded,
            [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE],
            [ConditionVariants.PSILOCYBIN],
            [ExclusionCategories.MISSING_TRIALS],
        )
        assert "002" not in result[SingleDataMetadata.PARTICIPANT_ID].tolist()

    def test_exclusion_respects_specified_condition(self, dataset_metadata):
        # Exclusion specified for Psilocybin only must not drop the Placebo rows.
        excluded = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID.value: ["001"],
                SingleDataMetadata.CONDITION.value: ["Psilocybin"],
                SingleDataMetadata.MUSIC_TYPE.value: [""],
                SingleDataMetadata.EXCLUSION_EXPLANATION.value: ["artifacts"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            dataset_metadata,
            excluded,
            [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE],
            [ConditionVariants.PLACEBO],
            [ExclusionCategories.ARTIFACTS],
        )
        assert "001" in result[SingleDataMetadata.PARTICIPANT_ID].tolist()


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


@pytest.fixture
def paired_metadata():
    """Metadata for a crossover design, deliberately not in condition order.

    Participants 001 and 002 have both conditions; 003 has only Placebo, so it is not
    a complete pair and must be dropped from a JOINED selection.
    """
    return pd.DataFrame(
        {
            SingleDataMetadata.PARTICIPANT_ID: [
                "002",
                "001",
                "003",
                "001",
                "002",
            ],
            SingleDataMetadata.CONDITION: [
                ConditionVariants.PSILOCYBIN,
                ConditionVariants.PLACEBO,
                ConditionVariants.PLACEBO,
                ConditionVariants.PSILOCYBIN,
                ConditionVariants.PLACEBO,
            ],
            SingleDataMetadata.MUSIC_TYPE: [MusicTypeVariants.CLASSICAL] * 5,
            SingleDataMetadata.FILENAME: [
                "p002_psilo.edf",
                "p001_placebo.edf",
                "p003_placebo.edf",
                "p001_psilo.edf",
                "p002_placebo.edf",
            ],
        }
    )


@pytest.fixture
def no_exclusions():
    """An empty excluded-participants table with the expected string columns."""
    return pd.DataFrame(
        {
            SingleDataMetadata.PARTICIPANT_ID.value: [],
            SingleDataMetadata.CONDITION.value: [],
            SingleDataMetadata.MUSIC_TYPE.value: [],
            SingleDataMetadata.EXCLUSION_EXPLANATION.value: [],
        }
    )


class TestExpandConditions:
    def test_real_conditions_pass_through(self):
        expanded, is_joined = DatasetFilter.expand_conditions(
            [ConditionVariants.PLACEBO]
        )
        assert expanded == [ConditionVariants.PLACEBO]
        assert is_joined is False

    def test_joined_expands_to_both(self):
        expanded, is_joined = DatasetFilter.expand_conditions(
            [ConditionVariants.JOINED]
        )
        assert expanded == [ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN]
        assert is_joined is True

    def test_joined_alongside_real_condition_deduplicates(self):
        expanded, _ = DatasetFilter.expand_conditions(
            [ConditionVariants.PSILOCYBIN, ConditionVariants.JOINED]
        )
        assert expanded == [ConditionVariants.PSILOCYBIN, ConditionVariants.PLACEBO]


class TestJoinedCondition:
    def test_joined_keeps_only_complete_pairs(self, paired_metadata, no_exclusions):
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        # 003 has no Psilocybin recording, so the pair is incomplete.
        assert result[SingleDataMetadata.PARTICIPANT_ID].tolist() == [
            "001",
            "002",
            "001",
            "002",
        ]

    def test_joined_orders_placebo_block_then_psilocybin_block(
        self, paired_metadata, no_exclusions
    ):
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        conditions = result[SingleDataMetadata.CONDITION].tolist()
        assert conditions == [
            ConditionVariants.PLACEBO,
            ConditionVariants.PLACEBO,
            ConditionVariants.PSILOCYBIN,
            ConditionVariants.PSILOCYBIN,
        ]

    def test_subject_k_pairs_with_k_plus_n_pairs(self, paired_metadata, no_exclusions):
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        participants = result[SingleDataMetadata.PARTICIPANT_ID].tolist()
        n_pairs = len(participants) // 2
        assert participants[:n_pairs] == participants[n_pairs:]

    def test_joined_never_yields_the_virtual_condition(
        self, paired_metadata, no_exclusions
    ):
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        assert ConditionVariants.JOINED not in result[SingleDataMetadata.CONDITION]

    def test_exclusion_in_one_condition_drops_the_whole_pair(self, paired_metadata):
        """A participant excluded under one condition cannot form a pair."""
        excluded = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID.value: ["001"],
                SingleDataMetadata.CONDITION.value: ["Placebo"],
                SingleDataMetadata.MUSIC_TYPE.value: ["CLASSIC"],
                SingleDataMetadata.EXCLUSION_EXPLANATION.value: ["artifacts"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            excluded,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [ExclusionCategories.ARTIFACTS],
        )
        assert result[SingleDataMetadata.PARTICIPANT_ID].tolist() == ["002", "002"]

    def test_single_condition_selection_is_unaffected(
        self, paired_metadata, no_exclusions
    ):
        """The pairing restriction must not leak into normal single-condition runs."""
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.PLACEBO],
            [],
        )
        assert sorted(result[SingleDataMetadata.PARTICIPANT_ID].tolist()) == [
            "001",
            "002",
            "003",
        ]

    def test_condition_column_as_strings_after_csv_roundtrip(self, no_exclusions):
        """A sidecar round-trip turns enum values into plain strings."""
        metadata = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID: ["001", "001", "002"],
                SingleDataMetadata.CONDITION: ["Placebo", "Psilocybin", "Placebo"],
                SingleDataMetadata.MUSIC_TYPE: ["CLASSIC"] * 3,
                SingleDataMetadata.FILENAME: ["a.edf", "b.edf", "c.edf"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        assert result[SingleDataMetadata.PARTICIPANT_ID].tolist() == ["001", "001"]
        assert result[SingleDataMetadata.CONDITION].tolist() == [
            "Placebo",
            "Psilocybin",
        ]


class TestJoinedTracksMatchesJoined:
    """Both virtual conditions must select exactly the same cohort."""

    def test_same_rows_as_joined(self, paired_metadata, no_exclusions):
        joined = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED],
            [],
        )
        tracks = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            no_exclusions,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED_TRACKS],
            [],
        )
        pd.testing.assert_frame_equal(joined, tracks)

    def test_exclusions_apply_the_same_way(self, paired_metadata):
        excluded = pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID.value: ["001"],
                SingleDataMetadata.CONDITION.value: ["Psilocybin"],
                SingleDataMetadata.MUSIC_TYPE.value: ["CLASSIC"],
                SingleDataMetadata.EXCLUSION_EXPLANATION.value: ["artifacts"],
            }
        )
        result = DatasetFilter.filter_dataset_by_all_categories(
            paired_metadata,
            excluded,
            [MusicTypeVariants.CLASSICAL],
            [ConditionVariants.JOINED_TRACKS],
            [ExclusionCategories.ARTIFACTS],
        )
        assert result[SingleDataMetadata.PARTICIPANT_ID].tolist() == ["002", "002"]
