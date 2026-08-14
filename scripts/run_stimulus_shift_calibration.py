"""Calibrate the per-recording stimulus-marker timing error of an experiment.

Stimulus annotations read from the raw recordings can be systematically late with
respect to the events they mark. In the ASSR dataset the cause is a file-format
limitation — the EDF header stores the recording start only to the nearest whole
second, while the recording-native ``.evt`` export uses its own origin a few
hundred milliseconds later — so every annotation comes back late by the sub-second
remainder the header could not carry. That remainder is a *different* constant in
every recording (measured range 372-455 ms), which is why one global offset cannot
remove it; see ``notebooks/00-preprocessing/assr_annotation_discrepancy.ipynb``.

This script measures the error once per recording and writes the mapping the
preprocessing pipeline reads::

    config/participant_mappings/<experiment>_time_shift.csv

Only headers and annotations are read — the EEG signal is never loaded, so a whole
experiment calibrates in seconds.

Two independent estimators are computed for every recording (see
:mod:`src.preprocessing.marker_shift`): the median discrepancy over paired stimulus
markers, and the offset between the two files' time origins taken from the ``.evt``
wall-clock anchor. They share no inputs, so their agreement is the check that
either is right; disagreements and non-constant discrepancies are reported and, by
default, abort the write rather than silently persisting a bad calibration.

Rerun this whenever the ``.evt`` exports or the raw recordings change. Its output
re-times every onset-locked analysis, so the stimulus-aligned products on disk
(``RAW_CROPPED``, the concatenated array with its ``.stimulus_onsets.npy``, and the
wavelet cache) must be regenerated afterwards, in that order.

Examples::

    # Calibrate the ASSR experiment and write the mapping.
    python scripts/run_stimulus_shift_calibration.py --experiment assr

    # Inspect what would be written without touching the file.
    python scripts/run_stimulus_shift_calibration.py --experiment assr --dry_run

    # Persist anyway when a recording fails a consistency check.
    python scripts/run_stimulus_shift_calibration.py --experiment assr --force
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import ExperimentNames, SingleDataMetadata  # noqa: E402
from src.preprocessing.marker_shift import (  # noqa: E402
    DEFAULT_AGREEMENT_TOLERANCE_S,
    DEFAULT_JITTER_TOLERANCE_S,
    MarkerShift,
    event_file_path,
    marker_shifts_to_frame,
    measure_marker_shift,
    read_evt_file,
    write_marker_shift_table,
)
from src.preprocessing.pipeline import DatasetHandler  # noqa: E402
from src.preprocessing.stimulus_alignment import (  # noqa: E402
    EXPERIMENT_STIMULUS_MARKERS,
)
from src.definitions.fields import CoordinateSystems  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=ExperimentNames.ASSR.value,
        choices=[experiment.value for experiment in ExperimentNames],
        help="Experiment to calibrate (default: %(default)s).",
    )
    parser.add_argument(
        "--stimulus_label",
        type=str,
        default=None,
        help=(
            "Annotation label to calibrate against. Defaults to the label "
            "registered for the experiment."
        ),
    )
    parser.add_argument(
        "--events_dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the `.evt` exports "
            f"(default: {ProjectPaths.EVENTS_DIR})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination CSV. Defaults to the experiment's registered mapping path "
            "under config/participant_mappings/."
        ),
    )
    parser.add_argument(
        "--jitter_tolerance_s",
        type=float,
        default=DEFAULT_JITTER_TOLERANCE_S,
        help=(
            "Peak-to-peak spread of the per-marker discrepancy above which a "
            "recording is not describable by one shift (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--agreement_tolerance_s",
        type=float,
        default=DEFAULT_AGREEMENT_TOLERANCE_S,
        help=(
            "Gap above which the two estimators are treated as disagreeing "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Measure and report, but do not write the mapping file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Write the mapping even when some recordings failed a consistency "
            "check or could not be measured."
        ),
    )
    return parser.parse_args()


def measure_experiment(
    dataset_handler: DatasetHandler,
    events_dir: Path,
    stimulus_label: str,
    jitter_tolerance_s: float,
    agreement_tolerance_s: float,
) -> tuple[list[MarkerShift], list[str], list[str]]:
    """
    Measure the marker shift of every recording of an experiment.

    :param dataset_handler: Handler providing the dataset metadata and raw paths.
    :param events_dir: Directory holding the ``.evt`` exports.
    :param stimulus_label: Annotation label to calibrate against.
    :param jitter_tolerance_s: See :func:`measure_marker_shift`.
    :param agreement_tolerance_s: See :func:`measure_marker_shift`.
    :return: Tuple of (measured shifts, recordings skipped for lack of an export,
        recordings that could not be measured).
    """
    shifts: list[MarkerShift] = []
    without_export: list[str] = []
    failed: list[str] = []

    for _, row in dataset_handler.dataset_metadata.iterrows():
        filename = row[SingleDataMetadata.FILENAME]
        evt_path = event_file_path(events_dir, filename)
        if not evt_path.exists():
            without_export.append(Path(filename).stem)
            continue
        try:
            raw = dataset_handler.load_data_file(filename, preload=False)
            shifts.append(
                measure_marker_shift(
                    raw,
                    read_evt_file(evt_path),
                    filename,
                    stimulus_label,
                    jitter_tolerance_s=jitter_tolerance_s,
                    agreement_tolerance_s=agreement_tolerance_s,
                    logger=logger,
                )
            )
        except (ValueError, OSError) as error:
            logger.error(f"{Path(filename).stem}: {error}")
            failed.append(Path(filename).stem)

    return shifts, without_export, failed


def report(shifts: list[MarkerShift], agreement_tolerance_s: float) -> list[str]:
    """
    Summarise the calibration and collect the recordings that failed a check.

    :param shifts: Measured shifts.
    :param agreement_tolerance_s: Gap above which the estimators disagree.
    :return: Stems of recordings whose two estimators disagree.
    """
    values = np.array([shift.shift_s for shift in shifts], dtype=float)
    logger.info(
        f"Measured {len(shifts)} recording(s): shift "
        f"min {values.min():.6f} s, median {np.median(values):.6f} s, "
        f"max {values.max():.6f} s (spread {np.ptp(values) * 1e3:.1f} ms)"
    )

    by_method = pd.Series([shift.method for shift in shifts]).value_counts()
    logger.info(f"Estimator used: {by_method.to_dict()}")

    jitters = np.array([shift.jitter_s for shift in shifts], dtype=float)
    if np.isfinite(jitters).any():
        logger.info(
            f"Worst within-recording jitter: "
            f"{np.nanmax(jitters) * 1e6:.1f} µs "
            "(a single shift describes each recording)"
        )

    disagreeing = [
        shift.filename
        for shift in shifts
        if np.isfinite(shift.anchor_shift_s)
        and abs(shift.shift_s - shift.anchor_shift_s) > agreement_tolerance_s
    ]
    if disagreeing:
        logger.warning(
            f"{len(disagreeing)} recording(s) where the two estimators disagree: "
            f"{disagreeing}"
        )
    else:
        logger.info("Both estimators agree on every recording.")
    return disagreeing


def main() -> int:
    """
    Entry point: measure every recording and persist the mapping.

    :return: Process exit code (0 on success, 1 when checks failed without
        ``--force``).
    """
    args = parse_args()
    setup_logging()
    mne.set_log_level("ERROR")

    experiment_name = ExperimentNames(args.experiment)
    marker = EXPERIMENT_STIMULUS_MARKERS.get(experiment_name)
    stimulus_label = args.stimulus_label or (marker.label if marker else None)
    if stimulus_label is None:
        logger.error(
            f"Experiment '{experiment_name.value}' has no registered stimulus "
            "label; pass --stimulus_label explicitly."
        )
        return 1

    events_dir = args.events_dir or ProjectPaths.EVENTS_DIR
    output_path = args.output or ProjectPaths.get_marker_shift_mapping_path(
        experiment_name
    )

    logger.info(
        f"Calibrating '{stimulus_label}' markers of {experiment_name.value} "
        f"against {events_dir}"
    )

    dataset_handler = DatasetHandler(
        experiment_name, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )
    shifts, without_export, failed = measure_experiment(
        dataset_handler,
        events_dir,
        stimulus_label,
        args.jitter_tolerance_s,
        args.agreement_tolerance_s,
    )

    if without_export:
        logger.warning(
            f"{len(without_export)} recording(s) have no `.evt` export and will "
            f"fall back to the registered constant offset: {without_export}"
        )
    if not shifts:
        logger.error("No recording could be measured; nothing to write.")
        return 1

    disagreeing = report(shifts, args.agreement_tolerance_s)

    metadata = dataset_handler.dataset_metadata
    stems = metadata[SingleDataMetadata.FILENAME].map(lambda name: Path(name).stem)
    participants = dict(zip(stems, metadata[SingleDataMetadata.PARTICIPANT_ID]))
    eeg_sessions = dict(zip(stems, metadata[SingleDataMetadata.EEG_CONDITION_ID]))
    frame = marker_shifts_to_frame(
        shifts,
        participants={key: str(value) for key, value in participants.items()},
        eeg_sessions={
            key: getattr(value, "value", str(value))
            for key, value in eeg_sessions.items()
        },
    )

    problems = failed or disagreeing
    if problems and not args.force:
        logger.error(
            "Consistency checks failed; refusing to write a calibration that would "
            "silently mis-time these recordings. Investigate, or re-run with "
            "--force to persist anyway."
        )
        return 1

    if args.dry_run:
        logger.info(f"--dry_run: would write {len(frame)} row(s) to {output_path}")
        print(frame.to_string(index=False))
        return 0

    write_marker_shift_table(output_path, frame)
    logger.info(f"Wrote {len(frame)} row(s) to {output_path}")
    logger.warning(
        "The stimulus-aligned products on disk are now stale. Regenerate "
        "RAW_CROPPED, the concatenated array, and the wavelet cache, in that "
        "order, before re-running any onset-locked analysis."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
