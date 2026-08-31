"""
Tests for src/analysis/condition_tracks.py — participant-matched two-condition
datasets, joined along time (``JOINED_TRACKS``) or along the subject axis (``JOINED``).
"""

import dataclasses

import numpy as np
import pytest

from src.analysis.condition_tracks import (
    ZSCORE_MODES,
    PairedConditionTracks,
    PooledConditionSubjects,
    concatenate_condition_tracks,
    pool_condition_subjects,
)
from src.analysis.data_representations import AnalysisData, DataRepresentation
from src.analysis.wavelet_ica import zscore_by_time
from src.definitions.fields import ConditionVariants

PLACEBO = ConditionVariants.PLACEBO
PSILOCYBIN = ConditionVariants.PSILOCYBIN


def _track(n_subjects, n_times, *, label, seed=0, scale=1.0, offset=0.0):
    """A 4-D (S, C, F, T) wavelet-power-shaped AnalysisData."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_subjects, 3, 4, n_times)) * scale + offset
    return AnalysisData(
        data=data,
        sfreq=250.0,
        representation=DataRepresentation.WAVELET_POWER,
        label=label,
        feature_names=["C1", "C2", "C3"],
    )


@pytest.fixture
def tracks():
    """Placebo has participants 001/002/003, Psilocybin has 002/003/004."""
    return {
        PLACEBO: _track(3, 20, label="Placebo_ASSR", seed=1),
        PSILOCYBIN: _track(3, 30, label="Psilocybin_ASSR", seed=2),
    }


@pytest.fixture
def participants():
    return {
        PLACEBO: ["001", "002", "003"],
        PSILOCYBIN: ["002", "003", "004"],
    }


class TestConcatenateConditionTracks:
    def test_keeps_only_participants_present_in_both(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.participants == ("002", "003")
        assert paired.n_pairs == 2

    def test_concatenates_along_the_time_axis(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.data.data.shape == (2, 3, 4, 50)
        assert paired.segment_lengths == (20, 30)
        assert paired.total_length == 50

    def test_conditions_need_not_share_a_time_base(self, tracks, participants):
        """The whole point: 20 and 30 samples concatenate without any trimming."""
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.segment_lengths[0] != paired.segment_lengths[1]

    def test_subject_rows_are_matched_by_participant_not_position(self, participants):
        # Give each subject a constant, participant-identifying value so a mis-matched
        # row is detectable in the output.
        placebo = _track(3, 8, label="Placebo_ASSR")
        psilocybin = _track(3, 8, label="Psilocybin_ASSR")
        placebo.data[:] = np.array([1.0, 2.0, 3.0])[:, None, None, None]
        psilocybin.data[:] = np.array([2.0, 3.0, 4.0])[:, None, None, None]

        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="none",
        )
        # Matched participants are 002 and 003: Placebo rows 1,2 and Psilocybin rows 0,1.
        placebo_track = paired.condition_track(paired.data.data, PLACEBO)
        psilocybin_track = paired.condition_track(paired.data.data, PSILOCYBIN)
        assert placebo_track[:, 0, 0, 0].tolist() == [2.0, 3.0]
        assert psilocybin_track[:, 0, 0, 0].tolist() == [2.0, 3.0]

    def test_segment_order_follows_the_conditions_argument(self, tracks, participants):
        paired = concatenate_condition_tracks(
            tracks, participants, conditions=[PSILOCYBIN, PLACEBO]
        )
        assert paired.conditions == (PSILOCYBIN, PLACEBO)
        assert paired.segment_lengths == (30, 20)

    def test_default_label_cannot_collide_with_crop_based_joined(
        self, tracks, participants
    ):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.data.label == "Placebo_ASSR+Psilocybin_ASSR"

    def test_metadata_records_the_pairing(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        metadata = paired.data.metadata
        assert metadata["paired_participants"] == ["002", "003"]
        assert metadata["paired_segment_lengths"] == [20, 30]
        assert metadata["paired_zscore_mode"] == "per_condition"

    def test_inputs_are_not_mutated(self, tracks, participants):
        before = {c: ad.data.copy() for c, ad in tracks.items()}
        concatenate_condition_tracks(tracks, participants)
        for condition, original in before.items():
            np.testing.assert_array_equal(tracks[condition].data, original)


class TestUnequalCohortSizes:
    """The two conditions arrive with every recording they have, not the matched set.

    On the real ASSR set that is 15 Placebo recordings against 16 Psilocybin ones, and
    reconciling them is exactly what the participant matching does. A shape check that
    included the subject axis would reject every cohort in which the conditions are not
    already the same size — i.e. the normal case — so these tests pin that it does not.
    """

    PLACEBO_ONLY = ["031", "036", "024"]
    PSILOCYBIN_ONLY = ["033", "018", "027", "040"]
    MATCHED = ["019", "023", "026", "029"]

    @pytest.fixture
    def participants(self):
        return {
            PLACEBO: self.MATCHED + self.PLACEBO_ONLY,
            PSILOCYBIN: self.PSILOCYBIN_ONLY + self.MATCHED,
        }

    @pytest.fixture
    def tracks(self, participants):
        return {
            PLACEBO: _track(
                len(participants[PLACEBO]), 20, label="Placebo_ASSR", seed=1
            ),
            PSILOCYBIN: _track(
                len(participants[PSILOCYBIN]), 30, label="Psilocybin_ASSR", seed=2
            ),
        }

    def test_conditions_of_different_sizes_concatenate(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert tracks[PLACEBO].data.shape[0] != tracks[PSILOCYBIN].data.shape[0]
        assert paired.n_pairs == len(self.MATCHED)
        assert paired.data.data.shape[0] == len(self.MATCHED)

    def test_only_participants_present_in_both_survive(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.participants == tuple(sorted(self.MATCHED))
        for unmatched in self.PLACEBO_ONLY + self.PSILOCYBIN_ONLY:
            assert unmatched not in paired.participants

    def test_rows_are_the_same_participant_in_both_segments(self, participants):
        """The point of matching: row k must be one person, not two."""
        placebo = _track(len(participants[PLACEBO]), 8, label="Placebo_ASSR")
        psilocybin = _track(len(participants[PSILOCYBIN]), 8, label="Psilocybin_ASSR")
        # Stamp each row with a value identifying its participant.
        for index, label in enumerate(participants[PLACEBO]):
            placebo.data[index] = float(label)
        for index, label in enumerate(participants[PSILOCYBIN]):
            psilocybin.data[index] = float(label)

        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="none",
        )
        for k, participant in enumerate(paired.participants):
            for condition in (PLACEBO, PSILOCYBIN):
                track = paired.condition_track(paired.data.data, condition)
                assert track[k].min() == track[k].max() == float(participant)

    def test_the_subject_axis_is_still_checked_against_its_labels(
        self, tracks, participants
    ):
        """Excluding it from the cross-condition check must not drop the check."""
        participants[PSILOCYBIN] = participants[PSILOCYBIN][:-1]
        with pytest.raises(ValueError, match="participant label"):
            concatenate_condition_tracks(tracks, participants)

    def test_a_real_feature_mismatch_is_still_rejected(self, tracks, participants):
        tracks[PSILOCYBIN] = dataclasses.replace(
            tracks[PSILOCYBIN], data=tracks[PSILOCYBIN].data[:, :2]
        )
        with pytest.raises(ValueError, match="feature dimensions"):
            concatenate_condition_tracks(tracks, participants)

    def test_the_subject_axis_join_agrees(self, tracks, participants):
        """Both joins must accept the same unequal cohort and keep the same pairs."""
        equal_length = {
            PLACEBO: tracks[PLACEBO],
            PSILOCYBIN: dataclasses.replace(
                tracks[PSILOCYBIN], data=tracks[PSILOCYBIN].data[..., :20]
            ),
        }
        pooled = pool_condition_subjects(equal_length, participants)
        paired = concatenate_condition_tracks(tracks, participants)
        assert sorted(set(pooled.participants)) == sorted(paired.participants)
        assert pooled.n_pairs == paired.n_pairs


class TestZscoreModes:
    def test_joint_preserves_the_between_condition_offset(self, participants):
        """A power difference between conditions must survive joint z-scoring."""
        placebo = _track(3, 200, label="Placebo_ASSR", seed=3, offset=0.0)
        psilocybin = _track(3, 200, label="Psilocybin_ASSR", seed=4, offset=10.0)
        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="joint",
        )
        placebo_mean = paired.condition_track(paired.data.data, PLACEBO).mean()
        psilocybin_mean = paired.condition_track(paired.data.data, PSILOCYBIN).mean()
        assert psilocybin_mean - placebo_mean > 1.0

    def test_per_condition_removes_the_between_condition_offset(self, participants):
        """Documented trade-off: separate z-scoring erases the amplitude contrast."""
        placebo = _track(3, 200, label="Placebo_ASSR", seed=3, offset=0.0)
        psilocybin = _track(3, 200, label="Psilocybin_ASSR", seed=4, offset=10.0)
        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="per_condition",
        )
        for condition in (PLACEBO, PSILOCYBIN):
            track = paired.condition_track(paired.data.data, condition)
            np.testing.assert_allclose(track.mean(axis=-1), 0.0, atol=1e-10)
            np.testing.assert_allclose(track.std(axis=-1), 1.0, atol=1e-10)

    def test_none_leaves_values_untouched(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants, zscore_mode="none")
        np.testing.assert_array_equal(
            paired.condition_track(paired.data.data, PLACEBO),
            tracks[PLACEBO].data[[1, 2]],
        )

    def test_default_is_per_condition(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.zscore_mode == "per_condition"
        for condition in paired.conditions:
            track = paired.condition_track(paired.data.data, condition)
            np.testing.assert_allclose(track.mean(axis=-1), 0.0, atol=1e-10)
            np.testing.assert_allclose(track.std(axis=-1), 1.0, atol=1e-10)

    def test_unknown_mode_raises(self, tracks, participants):
        with pytest.raises(ValueError, match="Unknown zscore_mode"):
            concatenate_condition_tracks(tracks, participants, zscore_mode="standard")

    def test_every_documented_mode_is_accepted(self, tracks, participants):
        for mode in ZSCORE_MODES:
            paired = concatenate_condition_tracks(
                tracks, participants, zscore_mode=mode
            )
            assert paired.zscore_mode == mode


class TestReverseAccess:
    def test_segments_tile_the_time_axis_without_gaps(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.boundaries == (0, 20, 50)
        assert paired.segment(PLACEBO) == slice(0, 20)
        assert paired.segment(PSILOCYBIN) == slice(20, 50)

    def test_condition_track_works_on_iva_shaped_sources(self, tracks, participants):
        """The reverse split must not care about the leading axes."""
        paired = concatenate_condition_tracks(tracks, participants)
        sources = np.zeros((paired.n_pairs, 5, 4, paired.total_length))
        assert paired.condition_track(sources, PSILOCYBIN).shape == (2, 5, 4, 30)

    def test_condition_track_works_on_a_bare_tf_map(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        tf_map = np.zeros((4, paired.total_length))
        assert paired.condition_track(tf_map, PLACEBO).shape == (4, 20)

    def test_condition_track_rejects_a_foreign_time_axis(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        with pytest.raises(ValueError, match="does not come from this concatenation"):
            paired.condition_track(np.zeros((2, 3, 4, 999)), PLACEBO)

    def test_condition_track_round_trips_the_concatenation(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants, zscore_mode="none")
        rebuilt = np.concatenate(
            [
                paired.condition_track(paired.data.data, condition)
                for condition in paired.conditions
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(rebuilt, paired.data.data)

    def test_unknown_condition_raises(self, tracks, participants):
        paired = concatenate_condition_tracks(
            tracks, participants, conditions=[PLACEBO]
        )
        with pytest.raises(KeyError, match="not part of this concatenation"):
            paired.segment(PSILOCYBIN)


class TestStimulusOnsets:
    def test_local_onsets_index_into_the_split_track(self, tracks, participants):
        paired = concatenate_condition_tracks(
            tracks,
            participants,
            onsets={PLACEBO: np.array([0, 5, 10]), PSILOCYBIN: np.array([1, 6])},
        )
        np.testing.assert_array_equal(
            paired.condition_onsets(PSILOCYBIN), np.array([1, 6])
        )

    def test_absolute_onsets_are_shifted_by_the_segment_start(
        self, tracks, participants
    ):
        paired = concatenate_condition_tracks(
            tracks,
            participants,
            onsets={PLACEBO: np.array([0, 5]), PSILOCYBIN: np.array([1, 6])},
        )
        np.testing.assert_array_equal(
            paired.condition_onsets(PLACEBO, absolute=True), np.array([0, 5])
        )
        np.testing.assert_array_equal(
            paired.condition_onsets(PSILOCYBIN, absolute=True), np.array([21, 26])
        )

    def test_absolute_onsets_stay_inside_their_segment(self, tracks, participants):
        paired = concatenate_condition_tracks(
            tracks,
            participants,
            onsets={PLACEBO: np.array([19]), PSILOCYBIN: np.array([29])},
        )
        for condition in paired.conditions:
            segment = paired.segment(condition)
            onsets = paired.condition_onsets(condition, absolute=True)
            assert onsets.min() >= segment.start
            assert onsets.max() < segment.stop

    def test_missing_onsets_stay_none(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert paired.condition_onsets(PLACEBO) is None


class TestValidation:
    def test_mismatched_channel_count_raises(self, participants):
        placebo = _track(3, 10, label="Placebo_ASSR")
        psilocybin = AnalysisData(
            data=np.zeros((3, 9, 4, 10)),
            sfreq=250.0,
            representation=DataRepresentation.WAVELET_POWER,
            label="Psilocybin_ASSR",
        )
        with pytest.raises(ValueError, match="feature dimensions"):
            concatenate_condition_tracks(
                {PLACEBO: placebo, PSILOCYBIN: psilocybin}, participants
            )

    def test_mismatched_sfreq_raises(self, tracks, participants):
        tracks[PSILOCYBIN].sfreq = 500.0
        with pytest.raises(ValueError, match="sampled at"):
            concatenate_condition_tracks(tracks, participants)

    def test_participant_count_must_match_subject_count(self, tracks, participants):
        participants[PLACEBO] = ["001", "002"]
        with pytest.raises(ValueError, match="3 subject.* but 2 participant"):
            concatenate_condition_tracks(tracks, participants)

    def test_duplicate_participant_raises(self, tracks, participants):
        participants[PLACEBO] = ["002", "002", "003"]
        with pytest.raises(ValueError, match="more than once"):
            concatenate_condition_tracks(tracks, participants)

    def test_no_common_participant_raises(self, tracks, participants):
        participants[PSILOCYBIN] = ["007", "008", "009"]
        with pytest.raises(ValueError, match="No participant is present"):
            concatenate_condition_tracks(tracks, participants)

    def test_missing_track_raises(self, tracks, participants):
        del tracks[PSILOCYBIN]
        with pytest.raises(ValueError, match="No track supplied"):
            concatenate_condition_tracks(tracks, participants)

    def test_missing_participant_labels_raise(self, tracks, participants):
        del participants[PSILOCYBIN]
        with pytest.raises(ValueError, match="No participant labels"):
            concatenate_condition_tracks(tracks, participants)


class TestDataclassContract:
    def test_result_is_immutable(self, tracks, participants):
        paired = concatenate_condition_tracks(tracks, participants)
        assert isinstance(paired, PairedConditionTracks)
        with pytest.raises(Exception):
            paired.participants = ("999",)


class TestZscoreIdempotence:
    """The property that lets JOINED_TRACKS reuse every existing consumer.

    The cached JoinedTracks wavelets are stored already per-condition z-scored. Every
    downstream workflow then calls ``zscore_by_time`` as its standard first step. That
    must be an exact identity, otherwise the joint re-normalisation would undo the
    per-condition standardisation the cache was built to provide.
    """

    def test_joint_zscore_of_per_condition_output_is_an_identity(self, participants):
        # Deliberately unequal lengths and wildly different scales/offsets.
        placebo = _track(3, 137, label="Placebo_ASSR", seed=11, scale=3.0, offset=12.0)
        psilocybin = _track(
            3, 211, label="Psilocybin_ASSR", seed=12, scale=0.2, offset=-4.0
        )
        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="per_condition",
        )
        np.testing.assert_allclose(
            zscore_by_time(paired.data.data), paired.data.data, atol=1e-9
        )

    def test_pooled_series_is_already_standardised(self, participants):
        placebo = _track(3, 137, label="Placebo_ASSR", seed=13, scale=5.0, offset=1.0)
        psilocybin = _track(
            3, 211, label="Psilocybin_ASSR", seed=14, scale=0.1, offset=99.0
        )
        paired = concatenate_condition_tracks(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin},
            participants,
            zscore_mode="per_condition",
        )
        pooled = paired.data.data
        np.testing.assert_allclose(pooled.mean(axis=-1), 0.0, atol=1e-10)
        np.testing.assert_allclose(pooled.std(axis=-1), 1.0, atol=1e-10)


class TestFromAnalyzer:
    """Reconstructing the split from a loaded analyser / its on-disk sidecar."""

    class _FakeAnalyzer:
        def __init__(self, boundaries, frame, onsets=None, tracks=True):
            self.concatenates_condition_tracks = tracks
            self.segment_boundaries = boundaries
            self.filtered_df = frame
            self.stimulus_onsets = onsets

        def to_analysis_data(self, label=None):
            total = int(self.segment_boundaries[-1])
            return AnalysisData(
                data=np.zeros((2, 3, total)),
                sfreq=250.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="JoinedTracks_ASSR",
            )

    @staticmethod
    def _frame():
        import pandas as pd

        from src.definitions.fields import SingleDataMetadata

        return pd.DataFrame(
            {
                SingleDataMetadata.PARTICIPANT_ID: ["019", "024", "019", "024"],
                SingleDataMetadata.CONDITION: [
                    PLACEBO,
                    PLACEBO,
                    PSILOCYBIN,
                    PSILOCYBIN,
                ],
                # A participant's two rows share one subject index.
                SingleDataMetadata.CONCATENATED_PERSON_INDEX: [0, 1, 0, 1],
            }
        )

    def test_rebuilds_segments_from_stored_boundaries(self):
        analyzer = self._FakeAnalyzer(np.array([0, 20, 50]), self._frame())
        paired = PairedConditionTracks.from_analyzer(analyzer)
        assert paired.segment_lengths == (20, 30)
        assert paired.segment(PSILOCYBIN) == slice(20, 50)
        assert paired.participants == ("019", "024")

    def test_split_works_on_an_iva_shaped_array(self):
        analyzer = self._FakeAnalyzer(np.array([0, 20, 50]), self._frame())
        paired = PairedConditionTracks.from_analyzer(analyzer)
        sources = np.zeros((2, 4, 5, 50))
        assert paired.condition_track(sources, PLACEBO).shape == (2, 4, 5, 20)

    def test_first_segment_onsets_are_carried_through(self):
        analyzer = self._FakeAnalyzer(
            np.array([0, 20, 50]), self._frame(), onsets=np.array([1, 9])
        )
        paired = PairedConditionTracks.from_analyzer(analyzer)
        np.testing.assert_array_equal(paired.condition_onsets(PLACEBO), [1, 9])

    def test_rejects_a_non_track_analyzer(self):
        analyzer = self._FakeAnalyzer(None, self._frame(), tracks=False)
        with pytest.raises(ValueError, match="not constructed with"):
            PairedConditionTracks.from_analyzer(analyzer)

    def test_rejects_missing_boundaries(self):
        analyzer = self._FakeAnalyzer(None, self._frame())
        with pytest.raises(ValueError, match="no segment boundaries"):
            PairedConditionTracks.from_analyzer(analyzer)

    def test_rejects_boundaries_that_do_not_match_the_conditions(self):
        analyzer = self._FakeAnalyzer(np.array([0, 20, 35, 50]), self._frame())
        with pytest.raises(ValueError, match="describe 3 segment"):
            PairedConditionTracks.from_analyzer(analyzer)


# ──────────────────────────────────────────────────────────────────────
# Subject-axis pooling (ConditionVariants.JOINED)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def shared_base_tracks():
    """Both conditions on one time base, as subject-axis pooling requires."""
    return {
        PLACEBO: _track(3, 20, label="Placebo_ASSR", seed=1),
        PSILOCYBIN: _track(3, 20, label="Psilocybin_ASSR", seed=2),
    }


class TestPoolConditionSubjects:
    def test_keeps_only_participants_present_in_both(
        self, shared_base_tracks, participants
    ):
        pooled = pool_condition_subjects(shared_base_tracks, participants)
        assert pooled.n_pairs == 2
        assert pooled.participants == ("002", "003", "002", "003")

    def test_stacks_along_the_subject_axis(self, shared_base_tracks, participants):
        pooled = pool_condition_subjects(shared_base_tracks, participants)
        assert pooled.data.data.shape == (4, 3, 4, 20)
        assert pooled.n_subjects == 4

    def test_one_block_per_condition_in_the_given_order(
        self, shared_base_tracks, participants
    ):
        pooled = pool_condition_subjects(
            shared_base_tracks, participants, conditions=[PSILOCYBIN, PLACEBO]
        )
        assert pooled.conditions == (PSILOCYBIN, PLACEBO)
        assert pooled.subject_conditions == (
            PSILOCYBIN,
            PSILOCYBIN,
            PLACEBO,
            PLACEBO,
        )

    def test_subject_rows_are_matched_by_participant_not_position(self, participants):
        placebo = _track(3, 8, label="Placebo_ASSR")
        psilocybin = _track(3, 8, label="Psilocybin_ASSR")
        placebo.data[:] = np.array([1.0, 2.0, 3.0])[:, None, None, None]
        psilocybin.data[:] = np.array([2.0, 3.0, 4.0])[:, None, None, None]

        pooled = pool_condition_subjects(
            {PLACEBO: placebo, PSILOCYBIN: psilocybin}, participants
        )
        # Matched participants are 002 and 003: Placebo rows 1,2 and Psilocybin rows 0,1.
        placebo_block = pooled.condition_subjects(pooled.data.data, PLACEBO)
        psilocybin_block = pooled.condition_subjects(pooled.data.data, PSILOCYBIN)
        assert placebo_block[:, 0, 0, 0].tolist() == [2.0, 3.0]
        assert psilocybin_block[:, 0, 0, 0].tolist() == [2.0, 3.0]

    def test_values_are_not_standardised(self, shared_base_tracks, participants):
        """Pooling is verbatim; z-scoring is the caller's downstream step."""
        pooled = pool_condition_subjects(shared_base_tracks, participants)
        np.testing.assert_array_equal(
            pooled.condition_subjects(pooled.data.data, PLACEBO),
            shared_base_tracks[PLACEBO].data[[1, 2]],
        )

    def test_default_label_is_the_canonical_joined_name(
        self, shared_base_tracks, participants
    ):
        pooled = pool_condition_subjects(shared_base_tracks, participants)
        assert pooled.data.label == "Joined_ASSR"

    def test_label_falls_back_when_the_suffixes_disagree(self, participants):
        tracks = {
            PLACEBO: _track(3, 20, label="Placebo_CLASSIC", seed=1),
            PSILOCYBIN: _track(3, 20, label="Psilocybin_PSYTRANCE", seed=2),
        }
        pooled = pool_condition_subjects(tracks, participants)
        assert pooled.data.label == "Placebo_CLASSIC+Psilocybin_PSYTRANCE"

    def test_metadata_records_the_pooling(self, shared_base_tracks, participants):
        metadata = pool_condition_subjects(
            shared_base_tracks, participants
        ).data.metadata
        assert metadata["pooled_conditions"] == ["Placebo", "Psilocybin"]
        assert metadata["pooled_participants"] == ["002", "003"]
        assert metadata["pooled_subject_conditions"] == [
            "Placebo",
            "Placebo",
            "Psilocybin",
            "Psilocybin",
        ]

    def test_inputs_are_not_mutated(self, shared_base_tracks, participants):
        before = {c: ad.data.copy() for c, ad in shared_base_tracks.items()}
        pool_condition_subjects(shared_base_tracks, participants)
        for condition, original in before.items():
            np.testing.assert_array_equal(shared_base_tracks[condition].data, original)

    def test_differing_time_bases_are_rejected(self, tracks, participants):
        """The time-axis fixture (20 vs 30 samples) cannot be stacked on subjects."""
        with pytest.raises(ValueError, match="non-subject dimensions"):
            pool_condition_subjects(tracks, participants)

    def test_differing_sampling_rates_are_rejected(
        self, shared_base_tracks, participants
    ):
        shared_base_tracks[PSILOCYBIN] = dataclasses.replace(
            shared_base_tracks[PSILOCYBIN], sfreq=500.0
        )
        with pytest.raises(ValueError, match="sampled at"):
            pool_condition_subjects(shared_base_tracks, participants)

    def test_participant_count_must_match_the_subject_axis(
        self, shared_base_tracks, participants
    ):
        participants[PLACEBO] = ["001", "002"]
        with pytest.raises(ValueError, match="participant label"):
            pool_condition_subjects(shared_base_tracks, participants)

    def test_no_common_participant_raises(self, shared_base_tracks, participants):
        participants[PSILOCYBIN] = ["004", "005", "006"]
        with pytest.raises(ValueError, match="No participant is present"):
            pool_condition_subjects(shared_base_tracks, participants)

    def test_missing_track_raises(self, shared_base_tracks, participants):
        del shared_base_tracks[PSILOCYBIN]
        with pytest.raises(ValueError, match="No track supplied"):
            pool_condition_subjects(shared_base_tracks, participants)


class TestPooledReverseAccess:
    @pytest.fixture
    def pooled(self, shared_base_tracks, participants):
        return pool_condition_subjects(
            shared_base_tracks,
            participants,
            onsets={PLACEBO: np.array([2, 9]), PSILOCYBIN: None},
        )

    def test_condition_mask_selects_one_block(self, pooled):
        assert pooled.condition_mask(PLACEBO).tolist() == [True, True, False, False]

    def test_condition_subjects_works_on_iva_shaped_sources(self, pooled):
        """The reverse split must not care about the trailing axes."""
        sources = np.zeros((pooled.n_subjects, 5, 4, 20))
        assert pooled.condition_subjects(sources, PSILOCYBIN).shape == (2, 5, 4, 20)

    def test_condition_subjects_works_on_per_subject_topographies(self, pooled):
        patterns = np.zeros((pooled.n_subjects, 5, 3))
        assert pooled.condition_subjects(patterns, PLACEBO).shape == (2, 5, 3)

    def test_condition_subjects_rejects_a_foreign_subject_axis(self, pooled):
        with pytest.raises(ValueError, match="does not come from this pooling"):
            pooled.condition_subjects(np.zeros((7, 3, 4, 20)), PLACEBO)

    def test_condition_subjects_returns_participant_matched_rows(self, pooled):
        """Row k of either block must be the same participant."""
        placebo_rows = [
            p
            for p, c in zip(pooled.participants, pooled.subject_conditions)
            if c is PLACEBO
        ]
        psilocybin_rows = [
            p
            for p, c in zip(pooled.participants, pooled.subject_conditions)
            if c is PSILOCYBIN
        ]
        assert placebo_rows == psilocybin_rows

    def test_unknown_condition_raises(self, pooled):
        with pytest.raises(KeyError, match="not part of this pooling"):
            pooled.condition_mask(ConditionVariants.JOINED)

    def test_partner_index_pairs_the_two_blocks(self, pooled):
        assert pooled.partner_index.tolist() == [2, 3, 0, 1]

    def test_subject_labels_are_unique(self, pooled):
        assert pooled.subject_labels == (
            "002 Placebo",
            "003 Placebo",
            "002 Psilocybin",
            "003 Psilocybin",
        )
        assert len(set(pooled.subject_labels)) == pooled.n_subjects

    def test_onsets_need_no_shifting(self, pooled):
        np.testing.assert_array_equal(pooled.condition_onsets(PLACEBO), [2, 9])
        assert pooled.condition_onsets(PSILOCYBIN) is None


class TestSelectParticipants:
    @pytest.fixture
    def pooled(self, shared_base_tracks, participants):
        return pool_condition_subjects(shared_base_tracks, participants)

    def test_keeps_the_participant_in_every_condition(self, pooled):
        subset = pooled.select_participants(["003"])
        assert subset.participants == ("003", "003")
        assert subset.subject_conditions == (PLACEBO, PSILOCYBIN)
        assert subset.n_pairs == 1
        assert subset.data.data.shape == (2, 3, 4, 20)

    def test_stays_a_pooled_dataset(self, pooled):
        subset = pooled.select_participants(["003"])
        assert isinstance(subset, PooledConditionSubjects)
        assert subset.conditions == pooled.conditions
        assert subset.partner_index.tolist() == [1, 0]

    def test_rows_keep_their_pooled_order(self, pooled):
        subset = pooled.select_participants(["003", "002"])
        assert subset.participants == pooled.participants

    def test_data_rows_follow_the_selection(self, pooled):
        subset = pooled.select_participants(["003"])
        np.testing.assert_array_equal(
            subset.condition_subjects(subset.data.data, PLACEBO)[0],
            pooled.condition_subjects(pooled.data.data, PLACEBO)[1],
        )

    def test_metadata_follows_the_selection(self, pooled):
        subset = pooled.select_participants(["003"])
        assert subset.data.metadata["pooled_participants"] == ["003"]
        assert subset.data.metadata["pooled_subject_conditions"] == [
            "Placebo",
            "Psilocybin",
        ]

    def test_unknown_participant_raises(self, pooled):
        with pytest.raises(ValueError, match="not in this pooled cohort"):
            pooled.select_participants(["999"])

    def test_empty_selection_raises(self, pooled):
        with pytest.raises(ValueError, match="No participants selected"):
            pooled.select_participants([])
