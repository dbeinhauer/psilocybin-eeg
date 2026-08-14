"""
Per-recording calibration of the stimulus-marker timing error.

Stimulus annotations read from a raw recording can be systematically late with
respect to the acoustic event they mark. In the ASSR dataset the cause is a
file-format limitation: the EDF header stores the recording start only to the
nearest **whole second**, while the recording-native ``.evt`` export uses its own
origin a few hundred milliseconds later. Reading the EDF faithfully therefore
returns every annotation late by exactly the sub-second remainder the header could
not carry — a *different* constant in every recording (measured range 372-455 ms;
see ``notebooks/00-preprocessing/assr_annotation_discrepancy.ipynb``).

The remainder is recoverable, so this module measures it once per recording and
persists it as a mapping that the preprocessing pipeline consumes:

    ``config/participant_mappings/<experiment>_time_shift.csv``

Two independent estimators are available and they must agree:

* **markers** — the median of (annotation time − ``.evt`` event time) over every
  paired stimulus marker. The direct measurement of the quantity being corrected,
  and the default whenever the two sources list the same number of events.
* **anchor** — the distance between the two files' time origins, obtained from the
  ``.evt`` wall-clock anchor row and the recording header. Needs a single row
  rather than a paired marker sequence, so it still works when the export has
  duplicate or missing marker rows; used as the fallback and as a cross-check.

Sign convention, fixed here and nowhere else: :attr:`MarkerShift.shift_s` is how
much **later** the annotation is than the true event, so the correction added to a
marker time is :attr:`MarkerShift.onset_offset_s` ``= -shift_s``. That is the
column the pipeline reads, and it plugs directly into the ``onset_offset_s``
parameter of :func:`~src.preprocessing.stimulus_alignment.get_stimulus_onset_samples`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import mne

_logger = logging.getLogger(__name__)

# Separator of the mapping CSVs in ``config/participant_mappings``.
MAPPING_CSV_SEPARATOR = ";"

# ── `.evt` export layout ─────────────────────────────────────────────────────
EVT_FILE_SUFFIX = ".evt"
EVT_TIME_COLUMN = "Tmu"  # event time, microseconds, in the `.evt` time base
EVT_LABEL_COLUMN = "Comnt"  # event label ("fam+", "bgin", ...)
EVT_TRIGGER_COLUMN = "TriNo"  # numeric trigger code, or an ISO stamp on anchor rows

# ── Agreement tolerances ─────────────────────────────────────────────────────
# One tick of the EDF annotation grid (1 ms). Within a recording the discrepancy
# is a single constant, so anything above this is a genuine per-event timing
# problem and not the quantisation of the two time grids.
DEFAULT_JITTER_TOLERANCE_S = 1e-3
# The two estimators measure the same quantity by different routes; a gap wider
# than one annotation tick means one of the two sources is not what it claims.
DEFAULT_AGREEMENT_TOLERANCE_S = 1e-3

# Columns of the persisted mapping. ``onset_offset_s`` is the operational value —
# everything else is provenance, recorded so a stored mapping can be audited
# without re-reading the recordings.
MARKER_SHIFT_COLUMNS = [
    "filename",  # recording file stem; the lookup key
    "participant",  # participant id, for readability
    "eeg",  # EEG session id, for readability
    "onset_offset_s",  # AUTHORITATIVE: seconds added to a marker to get the onset
    "marker_shift_s",  # measured lateness of the annotations (= -onset_offset_s)
    "n_markers",  # markers the measurement is based on
    "jitter_s",  # within-recording peak-to-peak spread of the discrepancy
    "method",  # which estimator produced the value
]


@dataclass(frozen=True)
class MarkerShift:
    """
    The stimulus-marker timing error measured for one recording.

    :param filename: Recording file stem the measurement belongs to.
    :param shift_s: How much later the annotations are than the true events, in
        seconds. Positive means the annotation lags the event.
    :param onset_offset_s: The correction to add to a marker time, ``-shift_s``.
    :param n_markers: Number of paired markers behind a ``"markers"`` measurement
        (0 for ``"anchor"``).
    :param jitter_s: Peak-to-peak spread of the per-marker discrepancy. Near zero
        confirms the error really is one constant; ``nan`` when not measurable.
    :param anchor_shift_s: The independent origin-offset estimate, kept even when
        it was not the one used, so the two routes can be compared afterwards.
    :param method: ``"markers"`` or ``"anchor"``.
    """

    filename: str
    shift_s: float
    n_markers: int
    jitter_s: float
    anchor_shift_s: float
    method: str

    @property
    def onset_offset_s(self) -> float:
        """
        Seconds added to a marker time to obtain the true event onset.

        :return: ``-shift_s`` — the annotations are late, so the correction is
            negative.
        """
        return -self.shift_s


def event_file_path(events_dir: Path, filename: str) -> Path:
    """
    Locate the ``.evt`` export belonging to a recording.

    :param events_dir: Directory holding the ``.evt`` exports.
    :param filename: Raw data filename, with or without extension.
    :return: Path to the matching ``.evt`` file (not checked for existence).
    """
    return events_dir / (Path(filename).stem + EVT_FILE_SUFFIX)


def read_evt_file(path: Path) -> pd.DataFrame:
    """
    Read a recording-native ``.evt`` event export.

    :param path: Path to the ``.evt`` file (tab-separated, one header row).
    :return: DataFrame with the original columns stripped of padding, plus
        ``time_s`` — the event time in seconds in the ``.evt`` time base.
    :raises ValueError: If the expected time column is missing.
    """
    events = pd.read_csv(path, sep="\t", header=0, dtype=str)
    events.columns = [column.strip() for column in events.columns]
    if EVT_TIME_COLUMN not in events.columns:
        raise ValueError(
            f"{path} has no '{EVT_TIME_COLUMN}' column (found {list(events.columns)}); "
            "it is not a recognised `.evt` export."
        )
    for column in events.columns:
        events[column] = events[column].astype(str).str.strip()
    events["time_s"] = events[EVT_TIME_COLUMN].astype(np.int64) / 1e6
    return events


def evt_marker_times(events: pd.DataFrame, label: str) -> np.ndarray:
    """
    Sorted, de-duplicated times of one marker label in the ``.evt`` time base.

    De-duplication is not cosmetic: some exports list a single event once per
    hardware trigger line, at an identical microsecond. Counting those twice would
    break the index-wise pairing against the recording's annotations.

    :param events: DataFrame from :func:`read_evt_file`.
    :param label: Marker label (case-insensitive), e.g. ``"fam+"``.
    :return: Sorted array of unique event times in seconds.
    """
    if EVT_LABEL_COLUMN not in events.columns:
        return np.array([], dtype=float)
    mask = events[EVT_LABEL_COLUMN].str.lower() == label.strip().lower()
    return np.unique(events.loc[mask, "time_s"].to_numpy())


def evt_clock_anchor(events: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    """
    The ``.evt`` wall-clock anchor: one event time tied to absolute time.

    Anchor rows are those whose trigger field holds an ISO timestamp instead of a
    numeric code. They are what makes the ``.evt`` time base comparable to the
    recording header's start time.

    :param events: DataFrame from :func:`read_evt_file`.
    :return: ``(time_s, wall_clock)`` of the first anchor row.
    :raises ValueError: If the export contains no anchor row.
    """
    if EVT_TRIGGER_COLUMN not in events.columns:
        raise ValueError(
            f"`.evt` export has no '{EVT_TRIGGER_COLUMN}' column; cannot locate a "
            "wall-clock anchor row."
        )
    stamps = pd.to_datetime(
        events[EVT_TRIGGER_COLUMN], errors="coerce", format="ISO8601"
    )
    anchors = np.flatnonzero(stamps.notna().to_numpy())
    if not len(anchors):
        raise ValueError(
            "`.evt` export contains no wall-clock anchor row (no ISO timestamp in "
            f"'{EVT_TRIGGER_COLUMN}'); the recording's time origin cannot be "
            "reconstructed from it."
        )
    row = int(anchors[0])
    return float(events["time_s"].iloc[row]), stamps.iloc[row]


def annotation_times(raw: mne.io.Raw, label: str) -> np.ndarray:
    """
    Sorted annotation times of one label, exactly as they are read from the file.

    Returned in the file's own frame — seconds from ``raw.annotations.orig_time``,
    i.e. from the recording start written in the header. No offset and no
    ``first_time`` bookkeeping is applied: this is the raw quantity whose error is
    being measured.

    :param raw: Recording whose annotations are read.
    :param label: Annotation description (case-insensitive).
    :return: Sorted array of annotation onsets in seconds.
    """
    target = label.strip().lower()
    return np.sort(
        np.array(
            [
                onset
                for onset, description in zip(
                    raw.annotations.onset, raw.annotations.description
                )
                if description.strip().lower() == target
            ],
            dtype=float,
        )
    )


def anchor_shift(raw: mne.io.Raw, events: pd.DataFrame) -> float:
    """
    Offset between the two files' time origins, from the ``.evt`` wall-clock anchor.

    The anchor row gives one ``.evt`` time and its absolute wall clock; subtracting
    the former from the latter yields the absolute start of the ``.evt`` clock. Its
    distance from the recording's header start time is how much later the ``.evt``
    origin sits, which is exactly how much later the recording's annotations are
    read.

    :param raw: Recording whose header start time is used.
    :param events: DataFrame from :func:`read_evt_file`.
    :return: Shift in seconds (positive = annotations are late).
    :raises ValueError: If the recording has no measurement date, or the export has
        no anchor row.
    """
    if raw.info["meas_date"] is None:
        raise ValueError(
            "Recording has no measurement date; the anchor estimator needs the "
            "header start time."
        )
    anchor_time_s, wall_clock = evt_clock_anchor(events)
    header_start = pd.Timestamp(raw.info["meas_date"].replace(tzinfo=None))
    evt_origin = wall_clock - pd.Timedelta(seconds=anchor_time_s)
    return float((evt_origin - header_start).total_seconds())


def measure_marker_shift(
    raw: mne.io.Raw,
    events: pd.DataFrame,
    filename: str,
    stimulus_label: str,
    jitter_tolerance_s: float = DEFAULT_JITTER_TOLERANCE_S,
    agreement_tolerance_s: float = DEFAULT_AGREEMENT_TOLERANCE_S,
    logger=None,
) -> MarkerShift:
    """
    Measure how late one recording's stimulus annotations are.

    Prefers the direct marker-pairing estimate and falls back to the wall-clock
    anchor when the two sources disagree on the number of markers. Whichever is
    used, the other is computed too and a disagreement is logged — the two share no
    inputs, so their agreement is the only available check that either is right.

    :param raw: Recording whose annotations are measured (header and annotations
        only; the signal is never touched, so ``preload=False`` is fine).
    :param events: DataFrame from :func:`read_evt_file` for the same recording.
    :param filename: Recording filename; its stem becomes the mapping key.
    :param stimulus_label: Annotation description marking a stimulus.
    :param jitter_tolerance_s: Peak-to-peak spread above which the discrepancy is
        not a single constant and the marker estimate is rejected.
    :param agreement_tolerance_s: Gap above which the two estimators are reported
        as disagreeing.
    :return: The measured :class:`MarkerShift`.
    :raises ValueError: If neither estimator can be evaluated.
    """
    log = logger or _logger
    stem = Path(filename).stem

    evt_times = evt_marker_times(events, stimulus_label)
    raw_times = annotation_times(raw, stimulus_label)

    # The anchor estimate is always computed: it is either the answer or the check.
    try:
        anchor_estimate = anchor_shift(raw, events)
    except ValueError as error:
        anchor_estimate = float("nan")
        log.warning(f"{stem}: anchor estimate unavailable ({error}).")

    marker_estimate = float("nan")
    jitter = float("nan")
    if len(evt_times) and len(evt_times) == len(raw_times):
        differences = raw_times - evt_times
        marker_estimate = float(np.median(differences))
        jitter = float(np.ptp(differences))
        if jitter > jitter_tolerance_s:
            log.warning(
                f"{stem}: the discrepancy is not constant across markers "
                f"(peak-to-peak {jitter * 1e3:.3f} ms > "
                f"{jitter_tolerance_s * 1e3:.3f} ms). A single per-recording shift "
                "cannot describe it; falling back to the anchor estimate."
            )
            marker_estimate = float("nan")
    elif len(evt_times) != len(raw_times):
        log.warning(
            f"{stem}: '{stimulus_label}' count differs between the `.evt` export "
            f"({len(evt_times)}) and the recording ({len(raw_times)}); markers "
            "cannot be paired, using the anchor estimate."
        )

    if not np.isnan(marker_estimate) and not np.isnan(anchor_estimate):
        gap = abs(marker_estimate - anchor_estimate)
        if gap > agreement_tolerance_s:
            log.warning(
                f"{stem}: the two estimators disagree by {gap * 1e3:.3f} ms "
                f"(markers {marker_estimate:+.6f} s, anchor {anchor_estimate:+.6f} s). "
                "Using the marker estimate, but the sources should be re-checked."
            )

    if not np.isnan(marker_estimate):
        shift, method, n_markers = marker_estimate, "markers", len(evt_times)
    elif not np.isnan(anchor_estimate):
        shift, method, n_markers = anchor_estimate, "anchor", 0
    else:
        raise ValueError(
            f"{stem}: neither the marker pairing nor the wall-clock anchor could be "
            "evaluated; no shift can be measured for this recording."
        )

    return MarkerShift(
        filename=stem,
        shift_s=shift,
        n_markers=int(n_markers),
        jitter_s=jitter,
        anchor_shift_s=anchor_estimate,
        method=method,
    )


def marker_shifts_to_frame(
    shifts: list[MarkerShift],
    participants: dict[str, str] | None = None,
    eeg_sessions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Lay measured shifts out as the persisted mapping table.

    :param shifts: Measured shifts, one per recording.
    :param participants: Optional ``stem -> participant id`` for the readability
        column; missing entries become empty.
    :param eeg_sessions: Optional ``stem -> EEG session id``, likewise.
    :return: DataFrame with :data:`MARKER_SHIFT_COLUMNS`, sorted by filename.
    """
    participants = participants or {}
    eeg_sessions = eeg_sessions or {}
    frame = pd.DataFrame(
        [
            {
                "filename": shift.filename,
                "participant": participants.get(shift.filename, ""),
                "eeg": eeg_sessions.get(shift.filename, ""),
                "onset_offset_s": shift.onset_offset_s,
                "marker_shift_s": shift.shift_s,
                "n_markers": shift.n_markers,
                "jitter_s": shift.jitter_s,
                "method": shift.method,
            }
            for shift in shifts
        ],
        columns=MARKER_SHIFT_COLUMNS,
    )
    return frame.sort_values("filename").reset_index(drop=True)


def write_marker_shift_table(path: Path, frame: pd.DataFrame) -> None:
    """
    Persist the shift mapping.

    :param path: Destination CSV path; parent directories are created.
    :param frame: Table from :func:`marker_shifts_to_frame`.
    :raises ValueError: If required columns are missing.
    """
    missing = set(MARKER_SHIFT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Shift table is missing columns: {sorted(missing)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=MAPPING_CSV_SEPARATOR, index=False)


def load_marker_shift_table(path: Path) -> dict[str, float]:
    """
    Load the per-recording marker→onset offsets.

    Reads only ``onset_offset_s``: the other columns are provenance for humans
    auditing the file, and deriving the operational value from them at load time
    would be a second place for the sign convention to go wrong.

    :param path: Path to the mapping CSV. A missing file yields an empty mapping,
        so an experiment without a calibration simply falls back to its registered
        constant.
    :return: Mapping from recording file stem to the offset in seconds.
    :raises ValueError: If the file exists but lacks the required columns, or lists
        a recording twice with conflicting offsets.
    """
    if not path.exists():
        _logger.info(
            f"No marker-shift mapping at {path}; falling back to the experiment's "
            "registered constant offset."
        )
        return {}

    table = pd.read_csv(path, sep=MAPPING_CSV_SEPARATOR)
    required = {"filename", "onset_offset_s"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}; expected the "
            f"layout written by write_marker_shift_table ({MARKER_SHIFT_COLUMNS})."
        )

    table["filename"] = table["filename"].astype(str).str.strip()
    table["filename"] = table["filename"].map(lambda name: Path(name).stem)
    duplicated = table.groupby("filename")["onset_offset_s"].nunique()
    conflicting = duplicated[duplicated > 1]
    if len(conflicting):
        raise ValueError(
            f"{path} lists conflicting offsets for {list(conflicting.index)}; each "
            "recording must appear at most once."
        )
    return {
        str(row.filename): float(row.onset_offset_s)
        for row in table.drop_duplicates("filename").itertuples()
    }
