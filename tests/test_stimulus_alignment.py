"""
Tests for annotation-based stimulus alignment (src.preprocessing.stimulus_alignment).
"""

import numpy as np
import pytest

from src.definitions.constants import AssrEpoch
from src.definitions.fields import ExperimentNames
from src.preprocessing.stimulus_alignment import (
    DEFAULT_STIMULUS_LABEL,
    EXPERIMENT_STIMULUS_MARKERS,
    StimulusAligner,
    StimulusMarker,
    get_stimulus_onset_samples,
    coarse_crop_to_stimulus_span,
    align_raws,
    apply_keep_segments_to_array,
    resolve_per_recording_offsets,
    resolve_stimulus_marker,
)

SFREQ = 1000.0  # 1 ms per sample -> sample index == milliseconds


class TestStimulusAlignerExample:
    """The 3-subject example from the feature spec (single interval)."""

    @pytest.fixture
    def aligner(self):
        # One interval per subject, onsets at 0 and {750, 1000, 700} ms.
        # No edge windows so the interval logic is isolated.
        onsets = [
            np.array([0, 750]),  # subject 1: 750 ms interval
            np.array([0, 1000]),  # subject 2: 1000 ms interval
            np.array([0, 700]),  # subject 3: 700 ms interval (shortest)
        ]
        lengths = [750, 1000, 700]
        return StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.1,
            pre_window_sec=0.0,
            post_window_sec=0.0,
        )

    def test_target_interval_is_group_minimum(self, aligner):
        assert aligner.interval_targets.tolist() == [700]

    def test_aligned_onsets_shared(self, aligner):
        assert aligner.aligned_onset_samples.tolist() == [0, 700]
        assert aligner.total_length == 700

    def test_segments_subject1_remove_600_650(self, aligner):
        # Keep [0,600] and [650,750]; i.e. remove [600,650].
        assert aligner.keep_segments[0] == [(0, 600), (650, 750)]

    def test_segments_subject2_remove_600_900(self, aligner):
        # Keep [0,600] and [900,1000]; i.e. remove [600,900].
        assert aligner.keep_segments[1] == [(0, 600), (900, 1000)]

    def test_segments_subject3_untouched(self, aligner):
        # Shortest subject: nothing removed, single contiguous segment.
        assert aligner.keep_segments[2] == [(0, 700)]

    def test_all_recordings_same_length(self, aligner):
        for segments in aligner.keep_segments:
            assert sum(end - start for start, end in segments) == aligner.total_length


class TestStimulusAlignerEdges:
    """Edge-window equalization before the first and after the last onset."""

    def test_edges_kept_and_equalized(self):
        # Onsets at 500 and 1500 ms; recordings extend 400 ms past last onset.
        onsets = [np.array([500, 1500]), np.array([500, 1700])]
        lengths = [1900, 2100]
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.1,
            pre_window_sec=0.1,  # 100 ms before first onset
            post_window_sec=0.1,  # 100 ms after last onset
        )
        # pre_target = min(100, 500) = 100; post_target = min(100, [400,400]) = 100.
        assert aligner.pre_target == 100
        assert aligner.post_target == 100
        # First onset sits 100 ms into the spliced output.
        assert aligner.aligned_onset_samples[0] == 100
        # total = pre + target_interval(min(1000,1200)=1000) + post
        assert aligner.total_length == 100 + 1000 + 100
        for segments in aligner.keep_segments:
            assert sum(e - s for s, e in segments) == aligner.total_length

    def test_uncapped_edges_use_group_minimum(self):
        # Default (None) edge windows keep the per-subject shortest lead-in/lead-out.
        onsets = [np.array([300, 1300]), np.array([800, 1800])]
        lengths = [1300 + 200, 1800 + 500]  # lead-outs: 200 and 500
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.1,
            pre_window_sec=None,
            post_window_sec=None,
        )
        # pre_target = min(lead-in) = min(300, 800) = 300 (uncapped).
        assert aligner.pre_target == 300
        # post_target = min(lead-out) = min(200, 500) = 200 (uncapped).
        assert aligner.post_target == 200
        for segments in aligner.keep_segments:
            assert sum(e - s for s, e in segments) == aligner.total_length

    def test_pre_window_clamped_to_available(self):
        # Subject with only 30 ms before first onset clamps the shared pre window.
        onsets = [np.array([30, 1030]), np.array([500, 1500])]
        lengths = [1030, 1500]
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.1,
            pre_window_sec=0.1,
            post_window_sec=0.0,
        )
        assert aligner.pre_target == 30


class TestStimulusAlignerCountMismatch:
    """Differing stimulus counts -> common count, trailing extras dropped."""

    def test_common_count_used(self):
        onsets = [
            np.array([0, 700, 1400, 2100]),  # 4 onsets
            np.array([0, 700, 1400]),  # 3 onsets (limits common count)
        ]
        lengths = [2100, 1400]
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.05,
            pre_window_sec=0.0,
            post_window_sec=0.0,
        )
        assert aligner.common_count == 3
        assert aligner.original_counts == [4, 3]
        assert len(aligner.interval_targets) == 2

    def test_post_window_capped_before_surplus_onset(self):
        # Subject 1 has a 4th (surplus) onset 300 ms after the last common onset,
        # but 1000 ms of recording past it. The post-window must stop just before
        # that surplus onset (300 samples), not extend to the recording end.
        onsets = [
            np.array([0, 700, 1400, 1700]),  # surplus onset at 1700 (300 after 1400)
            np.array([0, 700, 1400]),  # limits common count to 3
        ]
        lengths = [2700, 2400]  # plenty of room past the last common onset
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.05,
            pre_window_sec=None,
            post_window_sec=None,
        )
        # subj1 capped at 1700-1400 = 300; subj2 lead-out = 2400-1400 = 1000.
        assert aligner.post_target == 300
        # The kept span for subj1 ends exactly at the surplus onset (excluded).
        assert aligner.keep_segments[0][-1][1] == 1700
        for segments in aligner.keep_segments:
            assert sum(e - s for s, e in segments) == aligner.total_length

    def test_post_window_uncapped_without_surplus(self):
        # Same shape but no surplus onset -> post-window uses the group minimum
        # lead-out, unaffected by the cap.
        onsets = [
            np.array([0, 700, 1400]),
            np.array([0, 700, 1400]),
        ]
        lengths = [1400 + 300, 1400 + 1000]
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.05,
            pre_window_sec=None,
            post_window_sec=None,
        )
        assert aligner.post_target == 300  # min lead-out, not a surplus cap


class TestStimulusAlignerValidation:
    """Construction-time validation."""

    def test_too_few_common_onsets_raises(self):
        with pytest.raises(ValueError):
            StimulusAligner([np.array([100])], [1000], SFREQ)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            StimulusAligner([np.array([0, 700])], [700, 700], SFREQ)


def _make_raw_with_onsets(onset_ms, n_ms, sfreq=SFREQ):
    """Build a tiny annotated Raw: a ramp signal + fam+ onsets at given ms."""
    import mne

    info = mne.create_info(["ch0", "ch1"], sfreq, "eeg")
    # Distinct per-sample values so splicing can be verified by content.
    data = np.tile(np.arange(n_ms, dtype=float), (2, 1))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[ms / sfreq for ms in onset_ms],
            duration=[0.0] * len(onset_ms),
            description=["fam+"] * len(onset_ms),
        )
    )
    return raw


class TestMarkerOnsetOffset:
    """The marker→onset offset: applied once, everywhere onsets are read."""

    def test_default_returns_raw_marker_positions(self):
        raw = _make_raw_with_onsets([1000, 3000], 4000)
        assert get_stimulus_onset_samples(raw, "fam+").tolist() == [1000, 3000]

    def test_negative_offset_moves_onsets_earlier(self):
        # -0.4 s at 1 kHz = 400 samples earlier, applied to every onset.
        raw = _make_raw_with_onsets([1000, 3000], 4000)
        onsets = get_stimulus_onset_samples(raw, "fam+", onset_offset_s=-0.4)
        assert onsets.tolist() == [600, 2600]

    def test_positive_offset_moves_onsets_later(self):
        raw = _make_raw_with_onsets([1000, 3000], 4000)
        onsets = get_stimulus_onset_samples(raw, "fam+", onset_offset_s=0.25)
        assert onsets.tolist() == [1250, 3250]

    def test_offset_survives_a_cropped_recording(self):
        # first_samp > 0 (as after the coarse crop): the offset must compose with
        # the first_samp correction, not replace or double it.
        raw = _make_raw_with_onsets([1000, 3000], 4000)
        cropped = raw.copy().crop(tmin=0.5)
        assert cropped.first_samp == 500
        assert get_stimulus_onset_samples(cropped, "fam+").tolist() == [500, 2500]
        assert get_stimulus_onset_samples(
            cropped, "fam+", onset_offset_s=-0.4
        ).tolist() == [100, 2100]

    def test_offset_before_recording_start_raises(self):
        # Dropping the onset instead would silently misalign the group.
        raw = _make_raw_with_onsets([100, 3000], 4000)
        with pytest.raises(ValueError, match="outside the recording"):
            get_stimulus_onset_samples(raw, "fam+", onset_offset_s=-0.4)

    def test_offset_past_recording_end_raises(self):
        raw = _make_raw_with_onsets([1000, 3900], 4000)
        with pytest.raises(ValueError, match="outside the recording"):
            get_stimulus_onset_samples(raw, "fam+", onset_offset_s=0.4)

    def test_assr_marker_is_registered_with_the_fallback_offset(self):
        marker = EXPERIMENT_STIMULUS_MARKERS[ExperimentNames.ASSR]
        assert marker.label == DEFAULT_STIMULUS_LABEL
        assert marker.onset_offset_s == AssrEpoch.MARKER_ONSET_OFFSET_S
        # The marker lags the stimulus, and by less than the train length —
        # otherwise the epoch geometry in AssrEpoch cannot describe it.
        assert -AssrEpoch.STIMULUS_DURATION_S < marker.onset_offset_s < 0.0

    def test_coarse_crop_keeps_min_keep_before_the_true_onset(self):
        # With the offset applied the crop must protect the *stimulus*, not the
        # marker: min_keep is measured from the true onset (marker - 400 ms).
        raw = _make_raw_with_onsets([2000, 5000], 8000)
        cropped = coarse_crop_to_stimulus_span(
            raw, "fam+", trim_sec=10.0, min_keep_sec=0.5, onset_offset_s=-0.4
        )
        # True onsets 1600/4600. start = min(10000, 1600-500) = 1100.
        onsets = get_stimulus_onset_samples(cropped, "fam+", onset_offset_s=-0.4)
        assert onsets[0] == 500
        assert cropped.first_samp == 1100

    def test_align_raws_plans_around_true_onsets(self):
        # Same recordings aligned with and without the offset: the spliced output
        # must be shifted by exactly the offset, with identical geometry.
        raws = [_make_raw_with_onsets([1000, 2000], 3000) for _ in range(2)]
        _, plain = align_raws(raws, "fam+", keep_tail_sec=0.1)
        _, shifted = align_raws(raws, "fam+", keep_tail_sec=0.1, onset_offset_s=-0.4)
        assert plain.total_length == shifted.total_length
        assert plain.interval_targets.tolist() == shifted.interval_targets.tolist()
        # pre_target shrinks by the 400-sample offset, so onsets land 400 earlier.
        assert plain.pre_target - shifted.pre_target == 400


class TestPerRecordingOffsets:
    """
    The offset is per recording, not per experiment.

    Two recordings whose markers are late by different amounts must still end up
    aligned on the *stimulus*. A shared constant would align them on their markers
    instead, leaving each mis-timed by its own error — the bug this replaces.
    """

    def test_resolved_assr_marker_carries_the_shipped_calibration(self):
        marker = resolve_stimulus_marker(ExperimentNames.ASSR)
        assert marker.label == DEFAULT_STIMULUS_LABEL
        assert marker.onset_offset_s == AssrEpoch.MARKER_ONSET_OFFSET_S
        if not marker.per_recording_offsets:
            pytest.skip("No ASSR calibration checked in.")
        # Resolution must actually differ per recording — a table that collapsed
        # to one value would silently be the constant-offset bug again.
        assert len(set(marker.per_recording_offsets.values())) > 1

    def test_experiment_without_stimuli_resolves_to_nothing(self):
        assert resolve_stimulus_marker(ExperimentNames.PSILO_MUSIC) is None

    def test_marker_falls_back_when_a_recording_is_not_calibrated(self):
        marker = StimulusMarker("fam+", -0.4, {"rec_a": -0.42})
        assert marker.onset_offset_for("rec_a") == -0.42
        assert marker.onset_offset_for("unknown") == -0.4

    def test_marker_lookup_ignores_the_file_extension(self):
        marker = StimulusMarker("fam+", -0.4, {"rec_a": -0.42})
        assert marker.onset_offset_for("rec_a.edf") == -0.42

    def test_marker_without_a_recording_uses_the_fallback(self):
        marker = StimulusMarker("fam+", -0.4, {"rec_a": -0.42})
        assert marker.onset_offset_for(None) == -0.4

    def test_offsets_for_preserves_group_order(self):
        marker = StimulusMarker("fam+", -0.4, {"rec_a": -0.42, "rec_b": -0.38})
        assert marker.offsets_for(["rec_b", "rec_a", "rec_c"]) == [-0.38, -0.42, -0.4]

    def test_scalar_offset_is_broadcast_to_every_recording(self):
        assert resolve_per_recording_offsets(-0.4, 3) == [-0.4, -0.4, -0.4]

    def test_sequence_offsets_are_passed_through(self):
        assert resolve_per_recording_offsets([-0.4, -0.42], 2) == [-0.4, -0.42]

    def test_mismatched_offset_count_is_rejected(self):
        # Recycling or truncating would misalign the group by hundreds of ms
        # without any visible failure, so it must be an error.
        with pytest.raises(ValueError, match="must correspond one-to-one"):
            resolve_per_recording_offsets([-0.4, -0.42], 3)

    def test_differing_offsets_align_the_group_on_the_stimulus(self):
        # Two recordings with identical marker positions but markers that are late
        # by 400 ms and 300 ms. Their true onsets are therefore 100 ms apart, and
        # the aligner must absorb that difference: the spliced outputs must have
        # the same length and the same aligned onset positions.
        raws = [_make_raw_with_onsets([1000, 2000], 3000) for _ in range(2)]
        aligned, aligner = align_raws(
            raws, "fam+", keep_tail_sec=0.1, onset_offset_s=[-0.4, -0.3]
        )
        assert len({raw.n_times for raw in aligned}) == 1
        # The shortest available lead-in wins: recording 0's onset sits at
        # 1000-400=600, recording 1's at 1000-300=700, so pre_target is 600.
        assert aligner.pre_target == 600
        assert aligner.aligned_onset_samples[0] == 600

    def test_each_recording_keeps_its_own_marker_to_onset_distance(self):
        # After the splice the *onsets* coincide, so the annotations must NOT:
        # each marker still sits its own offset after the onset it marks.
        raws = [_make_raw_with_onsets([1000, 2000], 3000) for _ in range(2)]
        aligned, aligner = align_raws(
            raws, "fam+", keep_tail_sec=0.1, onset_offset_s=[-0.4, -0.3]
        )
        onset = int(aligner.aligned_onset_samples[0])
        assert int(get_stimulus_onset_samples(aligned[0], "fam+")[0]) == onset + 400
        assert int(get_stimulus_onset_samples(aligned[1], "fam+")[0]) == onset + 300


class TestCoarseCrop:
    """Stimulus-aware coarse crop (single continuous crop, no splicing)."""

    def test_short_lead_keeps_min_only(self):
        # Lead-in/out (1000 ms) shorter than trim (10 s) -> keep only min_keep each end.
        raw = _make_raw_with_onsets([1000, 3000], 4000)
        cropped = coarse_crop_to_stimulus_span(
            raw, "fam+", trim_sec=10.0, min_keep_sec=0.5
        )
        # start = min(10000, 1000-500) = 500; post lead = 4000-1-3000 = 999 ->
        # removed_end = min(10000, 999-500=499) = 499; end = 3999-499 = 3500.
        # kept [500, 3500] inclusive = 3001 samples.
        assert cropped.n_times == 3001
        onsets = get_stimulus_onset_samples(cropped, "fam+")
        assert onsets.tolist() == [500, 2500]

    def test_long_lead_trims_exactly_trim_sec(self):
        # Lead-in and lead-out both exceed trim+min_keep -> remove exactly trim_sec.
        raw = _make_raw_with_onsets([15000, 17000], 32000)
        cropped = coarse_crop_to_stimulus_span(
            raw, "fam+", trim_sec=10.0, min_keep_sec=0.5
        )
        # start = min(10000, 15000-500) = 10000; first onset -> 5000.
        # post lead = 32000-1-17000 = 14999 -> removed_end = min(10000, 14499)=10000;
        # end = 31999-10000 = 21999. kept [10000, 21999] = 12000 samples.
        assert cropped.n_times == 12000
        onsets = get_stimulus_onset_samples(cropped, "fam+")
        assert onsets.tolist() == [5000, 7000]

    def test_clipped_at_recording_bounds(self):
        # Onsets within min_keep of the recording bounds -> nothing trimmable.
        raw = _make_raw_with_onsets([200, 1200], 1500)
        cropped = coarse_crop_to_stimulus_span(
            raw, "fam+", trim_sec=10.0, min_keep_sec=0.5
        )
        # start = min(10000, max(0,200-500)) = 0; post lead = 1499-1200 = 299 ->
        # removed_end = min(10000, max(0,299-500)) = 0; end = 1499. kept all 1500.
        assert cropped.n_times == 1500
        assert get_stimulus_onset_samples(cropped, "fam+").tolist() == [200, 1200]

    def test_no_annotations_returns_copy(self):
        raw = _make_raw_with_onsets([], 1000)
        cropped = coarse_crop_to_stimulus_span(raw, "fam+", trim_sec=10.0)
        assert cropped.n_times == raw.n_times


class TestApplyKeepSegmentsToArray:
    """apply_keep_segments_to_array — trim numpy arrays the same way Raw objects are trimmed."""

    def test_single_segment_2d(self):
        data = np.arange(20).reshape(2, 10)
        result = apply_keep_segments_to_array(data, [(2, 7)])
        np.testing.assert_array_equal(result, data[:, 2:7])

    def test_two_segments_concatenated(self):
        data = np.arange(30).reshape(3, 10)
        result = apply_keep_segments_to_array(data, [(0, 3), (7, 10)])
        expected = np.concatenate([data[:, 0:3], data[:, 7:10]], axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_3d_array_time_on_last_axis(self):
        # (n_channels, n_freqs, n_times)
        data = np.random.default_rng(0).standard_normal((4, 5, 20))
        result = apply_keep_segments_to_array(data, [(2, 8), (12, 18)])
        expected = np.concatenate([data[..., 2:8], data[..., 12:18]], axis=-1)
        np.testing.assert_array_equal(result, expected)

    def test_output_length_matches_aligner_total_length(self):
        # Use the 3-subject fixture: aligner.total_length == 700.
        onsets = [np.array([0, 750]), np.array([0, 1000]), np.array([0, 700])]
        lengths = [750, 1000, 700]
        aligner = StimulusAligner(
            onsets,
            lengths,
            SFREQ,
            keep_tail_sec=0.1,
            pre_window_sec=0.0,
            post_window_sec=0.0,
        )
        for i, (n_times, segments) in enumerate(zip(lengths, aligner.keep_segments)):
            data = np.zeros((3, n_times))
            result = apply_keep_segments_to_array(data, segments)
            assert result.shape == (3, aligner.total_length), (
                f"Subject {i}: expected ({3}, {aligner.total_length}), "
                f"got {result.shape}"
            )

    def test_clamps_segments_to_array_bounds(self):
        data = np.ones((2, 10))
        # Segment extends past array end; should be clamped silently.
        result = apply_keep_segments_to_array(data, [(0, 15)])
        np.testing.assert_array_equal(result, data[:, 0:10])

    def test_no_valid_segments_raises(self):
        data = np.ones((2, 10))
        with pytest.raises(ValueError):
            apply_keep_segments_to_array(data, [(15, 20)])  # entirely out of range

    def test_matches_raw_splice_content(self):
        # Verify that trimming a numpy array gives the same sample values as
        # the MNE Raw splice produced by apply_keep_segments.
        raws = [
            _make_raw_with_onsets([0, 750], 750),
            _make_raw_with_onsets([0, 1000], 1000),
        ]
        aligned, aligner = align_raws(
            raws, "fam+", keep_tail_sec=0.1, pre_window_sec=0.0, post_window_sec=0.0
        )
        for raw, aligned_raw, segments in zip(raws, aligned, aligner.keep_segments):
            arr = raw.get_data()
            trimmed = apply_keep_segments_to_array(arr, segments)
            np.testing.assert_array_equal(trimmed, aligned_raw.get_data())


class TestApplyToRaw:
    """End-to-end: extract onsets, plan, splice MNE Raws."""

    def test_onset_extraction(self):
        raw = _make_raw_with_onsets([100, 800, 1500], 2000)
        samples = get_stimulus_onset_samples(raw, "fam+")
        assert samples.tolist() == [100, 800, 1500]

    def test_aligned_raws_equal_length_and_locked_onsets(self):
        raws = [
            _make_raw_with_onsets([0, 750], 750),
            _make_raw_with_onsets([0, 1000], 1000),
            _make_raw_with_onsets([0, 700], 700),
        ]
        aligned, aligner = align_raws(
            raws, "fam+", keep_tail_sec=0.1, pre_window_sec=0.0, post_window_sec=0.0
        )
        # All aligned recordings have identical length.
        lengths = {a.n_times for a in aligned}
        assert lengths == {aligner.total_length} == {700}
        # The second fam+ lands at the same sample in every aligned recording.
        second_onsets = {int(get_stimulus_onset_samples(a, "fam+")[1]) for a in aligned}
        assert second_onsets == {aligner.aligned_onset_samples[1]} == {700}
