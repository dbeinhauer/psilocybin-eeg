"""
Export every ASSR recording, stimulus-aligned, as its own FIF file.

Standalone, outside the analysis workflow (the products feed another project): no
participant is excluded and no metadata is consulted. Every ``.fif`` of the input
directory forms ONE alignment group, so stimulus ``k`` lands at the same sample in
every output file and each trial has the same length everywhere. Each inter-stimulus
interval is cut to the group minimum by removing its middle, keeping
``--keep_tail_sec`` of continuous data before every onset (see
:mod:`src.preprocessing.stimulus_alignment`). Surplus trailing stimuli are dropped.

The true onsets (marker plus the per-recording calibrated offset) are written as
``stimulus_onset`` annotations, and once more, shared by every file, in
``stimulus_onsets.csv`` beside them (written last, so its presence marks a complete
export). The original ``fam+`` markers are kept, still at the uncorrected marker
time. Every cut carries MNE's ``BAD boundary`` / ``EDGE boundary``
annotations, so an ``mne.Epochs`` reaching further back than ``--keep_tail_sec``
drops the epoch unless ``reject_by_annotation=False``.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import ExperimentNames, PreprocessedDataVariants  # noqa: E402
from src.preprocessing.stimulus_alignment import (  # noqa: E402
    StimulusAligner,
    apply_keep_segments,
    get_stimulus_onset_samples,
    resolve_stimulus_marker,
)

_logger = logging.getLogger(__name__)

STIMULUS_ONSET_DESCRIPTION = "stimulus_onset"
ONSETS_CSV_NAME = "stimulus_onsets.csv"
_ASSR_DIR = ProjectPaths.PROCESSED_DATA_DIR / ExperimentNames.ASSR.value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align every ASSR recording on its stimulus onsets (no exclusions) and "
            "store each as a FIF file on the shared time base."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=_ASSR_DIR / PreprocessedDataVariants.RAW_AFTER_ICA.value,
        help="Directory of the recordings to align (every *.fif in it).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_ASSR_DIR / "aligned_fif",
        help="Where the aligned FIF files are written.",
    )
    parser.add_argument(
        "--keep_tail_sec",
        type=float,
        default=AssrEpoch.PRE_ONSET_S,
        help="Continuous data kept immediately before every onset.",
    )
    parser.add_argument(
        "--pre_window_sec",
        type=float,
        default=None,
        help="Cap on the data kept before the first onset (default: group minimum).",
    )
    parser.add_argument(
        "--post_window_sec",
        type=float,
        default=None,
        help="Cap on the data kept after the last onset (default: group minimum).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into an output directory that already holds FIF files.",
    )
    return parser


def _add_onset_annotations(raw: mne.io.BaseRaw, onset_samples: np.ndarray) -> None:
    sfreq = raw.info["sfreq"]
    orig_time = raw.annotations.orig_time
    # With an orig_time, annotation onsets count from it, not from the first sample.
    shift = raw.first_time if orig_time is not None else 0.0
    onsets = mne.Annotations(
        onset=shift + onset_samples / sfreq,
        duration=np.zeros(len(onset_samples)),
        description=[STIMULUS_ONSET_DESCRIPTION] * len(onset_samples),
        orig_time=orig_time,
    )
    raw.set_annotations(raw.annotations + onsets)


def _save_atomic(raw: mne.io.BaseRaw, path: Path) -> None:
    partial = path.with_name(f"{path.stem}.partial.fif")
    raw.save(partial, overwrite=True, verbose="error")
    os.replace(partial, path)


def _save_onsets_csv(onset_samples: np.ndarray, sfreq: float, path: Path) -> None:
    """One row per trial; the onsets are identical in every exported file."""
    partial = path.with_name(f"{path.stem}.partial.csv")
    pd.DataFrame(
        {
            "trial": np.arange(len(onset_samples)),
            "onset_sample": onset_samples,
            "onset_s": onset_samples / sfreq,
        }
    ).to_csv(partial, index=False)
    os.replace(partial, path)


def run(args: argparse.Namespace) -> None:
    """Plan the alignment over every recording, then cut and store them one by one."""
    paths = sorted(args.input_dir.glob("*.fif"))
    if not paths:
        raise FileNotFoundError(f"No .fif recordings in {args.input_dir}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("*.fif")) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already holds FIF files; pass --overwrite to replace "
            "them (a partial set from another run would not share this time base)."
        )

    marker = resolve_stimulus_marker(ExperimentNames.ASSR)
    if marker is None:
        raise RuntimeError("No stimulus marker is registered for the ASSR experiment.")
    uncalibrated = [p.name for p in paths if p.stem not in marker.per_recording_offsets]
    if uncalibrated:
        _logger.warning(
            f"No calibrated marker offset for {uncalibrated}; using the fallback "
            f"{marker.onset_offset_s:+.4f} s."
        )

    # Headers only: the plan needs onsets and lengths, not samples.
    raws = [mne.io.read_raw_fif(p, preload=False, verbose="error") for p in paths]
    sfreqs = {raw.info["sfreq"] for raw in raws}
    if len(sfreqs) != 1:
        raise ValueError(f"Recordings differ in sampling frequency: {sorted(sfreqs)}.")
    sfreq = sfreqs.pop()

    aligner = StimulusAligner(
        [
            get_stimulus_onset_samples(
                raw, marker.label, marker.onset_offset_for(p.name)
            )
            for raw, p in zip(raws, paths)
        ],
        [raw.n_times for raw in raws],
        sfreq,
        keep_tail_sec=args.keep_tail_sec,
        pre_window_sec=args.pre_window_sec,
        post_window_sec=args.post_window_sec,
    )
    onsets = aligner.aligned_onset_samples
    _logger.info(
        f"{len(paths)} recording(s), onset counts {aligner.original_counts} -> "
        f"{aligner.common_count} common; {aligner.total_length} samples "
        f"({aligner.total_length / sfreq:.2f} s) each, first onset at "
        f"{onsets[0] / sfreq:.3f} s, trial lengths "
        f"{aligner.interval_targets.min() / sfreq:.3f}-"
        f"{aligner.interval_targets.max() / sfreq:.3f} s."
    )

    for raw, path, segments in zip(raws, paths, aligner.keep_segments):
        # Not preloaded: each segment is a cheap view, and only the kept samples are
        # read from disk when the result is saved.
        aligned = apply_keep_segments(raw, segments)
        if aligned.n_times != aligner.total_length:
            raise RuntimeError(
                f"{path.name}: aligned to {aligned.n_times} samples, expected "
                f"{aligner.total_length}."
            )
        _add_onset_annotations(aligned, onsets)
        stored = get_stimulus_onset_samples(aligned, STIMULUS_ONSET_DESCRIPTION)
        if not np.array_equal(stored, onsets):
            raise RuntimeError(f"{path.name}: onset annotations do not round-trip.")

        _save_atomic(aligned, args.output_dir / path.name)
        _logger.info(f"Saved {args.output_dir / path.name}.")
        del aligned
        raw.close()

    _save_onsets_csv(onsets, sfreq, args.output_dir / ONSETS_CSV_NAME)
    _logger.info(f"Saved {args.output_dir / ONSETS_CSV_NAME}.")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(_build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
