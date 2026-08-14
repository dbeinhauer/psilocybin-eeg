"""
Tests for the per-recording stimulus-marker calibration
(src.preprocessing.marker_shift).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import mne

from src.preprocessing.marker_shift import (
    MARKER_SHIFT_COLUMNS,
    MarkerShift,
    anchor_shift,
    annotation_times,
    event_file_path,
    evt_clock_anchor,
    evt_marker_times,
    load_marker_shift_table,
    marker_shifts_to_frame,
    measure_marker_shift,
    read_evt_file,
    write_marker_shift_table,
)

SFREQ = 1000.0
HEADER_START = "2018-01-02T10:00:00"


def _write_evt(
    path,
    marker_times_s,
    label="fam+",
    anchor_time_s=1.0,
    anchor_clock="2018-01-02T10:00:00.400",
    duplicate=False,
):
    """Write a minimal `.evt` export: an anchor row plus the marker rows."""
    rows = [(int(round(anchor_time_s * 1e6)), "41", anchor_clock, "clock")]
    for time_s in marker_times_s:
        tmu = int(round(time_s * 1e6))
        rows.append((tmu, "2", "0", label))
        if duplicate:
            # The same event written once per hardware trigger line.
            rows.append((tmu, "1", "601", label))
    frame = pd.DataFrame(rows, columns=["Tmu", "Code", "TriNo", "Comnt"])
    frame["Ver-C"] = ""
    frame.to_csv(path, sep="\t", index=False)
    return path


def _make_raw(annotation_times_s, label="fam+", meas_date=HEADER_START):
    """Build a tiny annotated Raw whose header start is a whole second."""
    info = mne.create_info(["ch0"], SFREQ, "eeg")
    raw = mne.io.RawArray(np.zeros((1, 20_000)), info, verbose="ERROR")
    raw.set_meas_date(pd.Timestamp(meas_date, tz="UTC").to_pydatetime())
    raw.set_annotations(
        mne.Annotations(
            onset=list(annotation_times_s),
            duration=[0.0] * len(annotation_times_s),
            description=[label] * len(annotation_times_s),
            orig_time=raw.info["meas_date"],
        )
    )
    return raw


class TestEvtReading:
    """Parsing the recording-native `.evt` export."""

    def test_times_are_converted_from_microseconds(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0])
        events = read_evt_file(path)
        assert events["time_s"].tolist() == pytest.approx([1.0, 2.5, 4.0])

    def test_marker_times_are_sorted_and_deduplicated(self, tmp_path):
        # An export listing each event twice must still yield one time per event:
        # counting the repeats would break the pairing against the annotations.
        path = _write_evt(tmp_path / "rec.evt", [4.0, 2.5], duplicate=True)
        times = evt_marker_times(read_evt_file(path), "fam+")
        assert times.tolist() == pytest.approx([2.5, 4.0])

    def test_marker_lookup_is_case_insensitive(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5], label="FAM+")
        assert len(evt_marker_times(read_evt_file(path), "fam+")) == 1

    def test_unknown_label_yields_no_times(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5])
        assert evt_marker_times(read_evt_file(path), "bgin").size == 0

    def test_missing_time_column_is_rejected(self, tmp_path):
        path = tmp_path / "bad.evt"
        pd.DataFrame({"Nope": ["1"]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="not a recognised"):
            read_evt_file(path)

    def test_clock_anchor_is_the_iso_stamped_row(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5], anchor_time_s=1.5)
        time_s, clock = evt_clock_anchor(read_evt_file(path))
        assert time_s == pytest.approx(1.5)
        assert clock == pd.Timestamp("2018-01-02T10:00:00.400")

    def test_export_without_an_anchor_row_is_rejected(self, tmp_path):
        path = tmp_path / "rec.evt"
        pd.DataFrame(
            {"Tmu": ["2500000"], "Code": ["2"], "TriNo": ["0"], "Comnt": ["fam+"]}
        ).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="no wall-clock anchor"):
            evt_clock_anchor(read_evt_file(path))

    def test_event_file_path_uses_the_recording_stem(self, tmp_path):
        assert event_file_path(tmp_path, "REC_01.edf") == tmp_path / "REC_01.evt"
        assert event_file_path(tmp_path, "REC_01") == tmp_path / "REC_01.evt"


class TestAnchorEstimator:
    """The origin-offset estimate, from the wall-clock anchor and the header."""

    def test_offset_is_the_gap_between_the_two_origins(self, tmp_path):
        # Anchor: `.evt` time 1.0 s is absolute 10:00:00.400, so the `.evt` clock
        # starts at 09:59:59.400 — 0.6 s BEFORE a 10:00:00 header start.
        path = _write_evt(tmp_path / "rec.evt", [2.5], anchor_time_s=1.0)
        raw = _make_raw([2.5])
        assert anchor_shift(raw, read_evt_file(path)) == pytest.approx(-0.6)

    def test_offset_is_positive_when_the_evt_origin_is_later(self, tmp_path):
        # `.evt` time 0.0 is absolute 10:00:00.400 -> its origin is 400 ms after
        # the header start, which is exactly how late the annotations are read.
        path = _write_evt(tmp_path / "rec.evt", [2.5], anchor_time_s=0.0)
        raw = _make_raw([2.5])
        assert anchor_shift(raw, read_evt_file(path)) == pytest.approx(0.4)

    def test_recording_without_a_measurement_date_is_rejected(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5])
        raw = _make_raw([2.5])
        raw.set_meas_date(None)
        with pytest.raises(ValueError, match="no measurement date"):
            anchor_shift(raw, read_evt_file(path))


class TestAnnotationTimes:
    """Reading annotations in the file's own frame."""

    def test_times_are_returned_sorted(self):
        raw = _make_raw([4.0, 2.5])
        assert annotation_times(raw, "fam+").tolist() == pytest.approx([2.5, 4.0])

    def test_other_labels_are_ignored(self):
        raw = _make_raw([2.5], label="bgin")
        assert annotation_times(raw, "fam+").size == 0


class TestMeasureMarkerShift:
    """Choosing, checking and reporting the per-recording shift."""

    def test_marker_estimate_is_preferred_and_measures_the_lateness(self, tmp_path):
        # `.evt` events at 2.5/4.0; annotations 400 ms later.
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9, 4.4])
        shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "markers"
        assert shift.shift_s == pytest.approx(0.4)
        assert shift.n_markers == 2
        assert shift.jitter_s == pytest.approx(0.0, abs=1e-9)

    def test_onset_offset_is_the_negated_shift(self, tmp_path):
        # The sign convention lives in exactly one place; this pins it.
        path = _write_evt(tmp_path / "rec.evt", [2.5], anchor_time_s=0.0)
        raw = _make_raw([2.9])
        shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.onset_offset_s == pytest.approx(-shift.shift_s)
        assert shift.onset_offset_s == pytest.approx(-0.4)

    def test_filename_is_reduced_to_its_stem(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5], anchor_time_s=0.0)
        raw = _make_raw([2.9])
        shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.filename == "rec"

    def test_duplicate_export_rows_do_not_break_the_pairing(self, tmp_path):
        path = _write_evt(
            tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0, duplicate=True
        )
        raw = _make_raw([2.9, 4.4])
        shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "markers"
        assert shift.shift_s == pytest.approx(0.4)

    def test_count_mismatch_falls_back_to_the_anchor(self, tmp_path, caplog):
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9])  # one annotation, two `.evt` events
        with caplog.at_level(logging.WARNING):
            shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "anchor"
        assert shift.shift_s == pytest.approx(0.4)
        assert shift.n_markers == 0
        assert "cannot be paired" in caplog.text

    def test_non_constant_discrepancy_falls_back_to_the_anchor(self, tmp_path, caplog):
        # A shift that is not one constant cannot be corrected by one number, so
        # the marker estimate must be rejected rather than averaged.
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9, 4.5])  # 400 ms then 500 ms late
        with caplog.at_level(logging.WARNING):
            shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "anchor"
        assert "not constant across markers" in caplog.text

    def test_jitter_within_tolerance_is_accepted(self, tmp_path):
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9, 4.4001])  # 0.1 ms apart: grid quantisation
        shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "markers"

    def test_estimator_disagreement_is_reported_but_markers_win(self, tmp_path, caplog):
        # Anchor says 0.4 s late; the markers say 0.9 s. The two share no inputs,
        # so a gap means one source is wrong and must not pass silently.
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([3.4, 4.9])
        with caplog.at_level(logging.WARNING):
            shift = measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert shift.method == "markers"
        assert shift.shift_s == pytest.approx(0.9)
        assert shift.anchor_shift_s == pytest.approx(0.4)
        assert "disagree" in caplog.text

    def test_agreeing_estimators_log_nothing(self, tmp_path, caplog):
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9, 4.4])
        with caplog.at_level(logging.WARNING):
            measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")
        assert caplog.text == ""

    def test_unmeasurable_recording_is_rejected(self, tmp_path):
        # No pairable markers and no usable anchor -> no shift may be invented.
        path = _write_evt(tmp_path / "rec.evt", [2.5, 4.0], anchor_time_s=0.0)
        raw = _make_raw([2.9])
        raw.set_meas_date(None)
        with pytest.raises(ValueError, match="neither the marker pairing"):
            measure_marker_shift(raw, read_evt_file(path), "rec.edf", "fam+")


def _shift(filename, shift_s=0.42):
    """A measured shift, for the table round-trip tests."""
    return MarkerShift(
        filename=filename,
        shift_s=shift_s,
        n_markers=150,
        jitter_s=1e-6,
        anchor_shift_s=shift_s,
        method="markers",
    )


class TestShiftTable:
    """Persisting and re-loading the calibration."""

    def test_frame_has_the_declared_layout_sorted_by_filename(self):
        frame = marker_shifts_to_frame([_shift("b_rec"), _shift("a_rec")])
        assert list(frame.columns) == MARKER_SHIFT_COLUMNS
        assert frame["filename"].tolist() == ["a_rec", "b_rec"]

    def test_frame_stores_the_offset_not_the_shift(self):
        frame = marker_shifts_to_frame([_shift("rec", shift_s=0.42)])
        assert frame["onset_offset_s"].iloc[0] == pytest.approx(-0.42)
        assert frame["marker_shift_s"].iloc[0] == pytest.approx(0.42)

    def test_participant_columns_are_filled_when_supplied(self):
        frame = marker_shifts_to_frame(
            [_shift("rec")], participants={"rec": "018"}, eeg_sessions={"rec": "A"}
        )
        assert frame["participant"].iloc[0] == "018"
        assert frame["eeg"].iloc[0] == "A"

    def test_unknown_participants_become_empty(self):
        frame = marker_shifts_to_frame([_shift("rec")], participants={"other": "018"})
        assert frame["participant"].iloc[0] == ""

    def test_round_trip_preserves_the_offsets(self, tmp_path):
        path = tmp_path / "assr_time_shift.csv"
        write_marker_shift_table(
            path, marker_shifts_to_frame([_shift("rec_a", 0.42), _shift("rec_b", 0.38)])
        )
        assert load_marker_shift_table(path) == pytest.approx(
            {"rec_a": -0.42, "rec_b": -0.38}
        )

    def test_write_creates_missing_directories(self, tmp_path):
        path = tmp_path / "nested" / "assr_time_shift.csv"
        write_marker_shift_table(path, marker_shifts_to_frame([_shift("rec")]))
        assert path.exists()

    def test_write_rejects_an_incomplete_table(self, tmp_path):
        frame = marker_shifts_to_frame([_shift("rec")]).drop(columns=["method"])
        with pytest.raises(ValueError, match="missing columns"):
            write_marker_shift_table(tmp_path / "out.csv", frame)

    def test_missing_file_yields_an_empty_mapping(self, tmp_path):
        # An uncalibrated experiment must fall back, not crash.
        assert load_marker_shift_table(tmp_path / "absent.csv") == {}

    def test_load_accepts_filenames_with_an_extension(self, tmp_path):
        path = tmp_path / "table.csv"
        pd.DataFrame({"filename": ["rec.edf"], "onset_offset_s": [-0.42]}).to_csv(
            path, sep=";", index=False
        )
        assert load_marker_shift_table(path) == {"rec": -0.42}

    def test_load_rejects_a_table_without_the_required_columns(self, tmp_path):
        path = tmp_path / "table.csv"
        pd.DataFrame({"filename": ["rec"], "shift": [0.42]}).to_csv(
            path, sep=";", index=False
        )
        with pytest.raises(ValueError, match="missing required column"):
            load_marker_shift_table(path)

    def test_load_rejects_conflicting_duplicate_rows(self, tmp_path):
        # Silently picking one of two offsets would mis-time the recording by
        # whatever the two disagree on.
        path = tmp_path / "table.csv"
        pd.DataFrame(
            {"filename": ["rec", "rec"], "onset_offset_s": [-0.42, -0.38]}
        ).to_csv(path, sep=";", index=False)
        with pytest.raises(ValueError, match="conflicting offsets"):
            load_marker_shift_table(path)

    def test_load_tolerates_an_exact_duplicate_row(self, tmp_path):
        path = tmp_path / "table.csv"
        pd.DataFrame(
            {"filename": ["rec", "rec"], "onset_offset_s": [-0.42, -0.42]}
        ).to_csv(path, sep=";", index=False)
        assert load_marker_shift_table(path) == {"rec": -0.42}


class TestShippedAssrCalibration:
    """The calibration checked into config/participant_mappings."""

    @pytest.fixture
    def offsets(self):
        from src.definitions.constants import ProjectPaths
        from src.definitions.fields import ExperimentNames

        path = ProjectPaths.get_marker_shift_mapping_path(ExperimentNames.ASSR)
        if not path.exists():
            pytest.skip(f"No ASSR calibration at {path}.")
        return load_marker_shift_table(path)

    def test_every_recording_is_calibrated(self, offsets):
        assert len(offsets) == 38

    def test_offsets_are_lags_within_the_measured_range(self, offsets):
        # All negative (the markers lag the sound) and inside the range the
        # diagnosis notebook measured. A value outside it means the calibration
        # was regenerated from different data than the analyses assume.
        values = np.array(list(offsets.values()))
        assert np.all(values < 0.0)
        assert values.min() == pytest.approx(-0.455, abs=1e-6)
        assert values.max() == pytest.approx(-0.372, abs=1e-6)

    def test_the_fallback_constant_is_the_median_of_the_calibration(self, offsets):
        from src.definitions.constants import AssrEpoch

        median = float(np.median(list(offsets.values())))
        assert AssrEpoch.MARKER_ONSET_OFFSET_S == pytest.approx(median, abs=5e-4)
