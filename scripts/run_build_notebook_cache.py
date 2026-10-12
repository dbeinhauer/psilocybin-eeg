"""
Build notebook-sized strips of the wavelet store.

The per-subject store under ``data/processed/<experiment>/wavelets/`` is memory-mapped,
so a notebook can already read a slice of it cheaply. A strip is for when the full store
is not there — a laptop, a copy of the project without the data volume: a small slice of
the same data, in the **same on-disk layout**, under ``notebooks/cache/``, so every
notebook reads it with the ordinary reader by pointing ``--wavelet_data_dir`` there.

Subsetting, in order of how much it saves:

* **frequencies** — a narrow window around the frequency of interest; a 40 +/- 5 Hz strip
  of a 1-50 Hz grid is ~4.5x smaller
* **time** — a few seconds instead of the whole recording; 12 s of 188 s is ~15x
* **subjects** — the first N participants (by id) present in *every* condition, so the
  strip stays paired and the analyses that need pairing still run

Channels are **never** subset: a spatial filter is indexed by the stored channel axis, and
dropping columns from it produces a different and meaningless operator.

Example::

    python scripts/run_build_notebook_cache.py --experiment assr \
        --n_subjects 6 --n_times 3000 --center_freq 40 --halfwidth 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ExperimentNames,
    MusicTypeVariants,
)
from src.io.wavelet_store import (  # noqa: E402
    WaveletSet,
    find_wavelet_set,
    iter_wavelet_subjects,
    wavelet_label_dir,
    write_wavelet_subject,
)

_logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    parser.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--n_subjects", type=int, default=6)
    parser.add_argument("--n_times", type=int, default=3000)
    parser.add_argument("--center_freq", type=float, default=40.0)
    parser.add_argument(
        "--halfwidth",
        type=float,
        default=5.0,
        help="Frequency half-width kept around --center_freq; 0 keeps the whole grid.",
    )
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ProjectPaths.NOTEBOOKS_DIR / "cache",
        help="Root the strips are written under; pass this as --wavelet_data_dir.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser


def paired_participants(sets: list[WaveletSet], n_subjects: int) -> list[str]:
    """The first *n_subjects* participants (by id) stored under every condition.

    Keeping the same people in every strip is what lets a paired analysis run on it.

    :param n_subjects: How many to keep; ``0`` or fewer keeps all.
    :raises ValueError: If the conditions share no participant.
    """
    shared = sorted(
        set(sets[0].participants).intersection(*(s.participants for s in sets))
    )
    if not shared:
        raise ValueError("No participant is stored under every condition.")
    return shared if n_subjects <= 0 else shared[:n_subjects]


def frequency_window(freqs: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    """Bins within *halfwidth* of *center*; the nearest bin if none; all if 0."""
    if halfwidth <= 0:
        return np.arange(freqs.size)
    indices = np.flatnonzero(np.abs(freqs - center) <= halfwidth)
    return (
        indices if indices.size else np.array([int(np.argmin(np.abs(freqs - center)))])
    )


def run(args: argparse.Namespace) -> None:
    """Build a strip for every requested condition."""
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    conditions = args.conditions or [c.value for c in REAL_CONDITIONS]
    labels = [f"{condition}_{music_type.value}" for condition in conditions]

    sets = [
        find_wavelet_set(wavelet_label_dir(args.source_dir, experiment, label))
        for label in labels
    ]
    keep = paired_participants(sets, args.n_subjects)
    _logger.info(f"keeping {len(keep)} paired participant(s): {keep}")

    for label, wavelet_set in zip(labels, sets):
        freq_indices = frequency_window(
            wavelet_set.freqs, args.center_freq, args.halfwidth
        )
        n_times = min(args.n_times, wavelet_set.open(keep[0]).shape[-1])
        out_dir = wavelet_label_dir(args.out_dir, experiment, label)
        for participant, strip in iter_wavelet_subjects(
            wavelet_set, keep, freq_indices=freq_indices, n_times=n_times
        ):
            write_wavelet_subject(
                out_dir,
                participant,
                strip,
                label=wavelet_set.label,
                representation=wavelet_set.representation,
                freqs=wavelet_set.freqs[freq_indices],
                n_cycles=wavelet_set.n_cycles[freq_indices],
                sfreq=wavelet_set.sfreq,
                channel_names=wavelet_set.channel_names,
            )
            _logger.info(f"  {label} {participant}: {strip.shape}")

    _logger.info(f"done; point --wavelet_data_dir at {args.out_dir}")


def main() -> None:
    """Parse arguments and run."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
