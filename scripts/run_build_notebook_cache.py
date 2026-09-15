"""
Build notebook-sized strips of the wavelet caches.

The source-of-truth caches under ``data/processed/<experiment>/wavelets/`` are ~50 GB per
condition, so a notebook that reads them costs as much as the job it is meant to rehearse.
This writes a small strip of the same data, in the **same on-disk layout**, under
``notebooks/cache/`` — so every notebook reads it with the ordinary
:func:`~src.io.loading.wavelet_cache_path` by pointing ``--wavelet_data_dir`` there, and
no analysis code needs a notebook-specific path.

Subsetting, in order of how much it saves:

* **frequencies** — a narrow window around the frequency of interest; a 40 +/- 5 Hz strip
  of a 1-50 Hz grid is ~4.5x smaller
* **time** — a few seconds instead of the whole recording; 12 s of 188 s is ~15x
* **subjects** — the first N participants present in *every* condition, so the strip stays
  paired and the analyses that need pairing still run

Channels are **never** subset: a spatial filter is indexed by the stored channel axis, and
dropping columns from it produces a different and meaningless operator.

Building a strip costs one full pass over the source cache (~15 min per condition), which
is the point — it is paid once, and every notebook run afterwards is cheap.

Example::

    python scripts/run_build_notebook_cache.py --experiment assr \
        --n_subjects 6 --n_times 3000 --center_freq 40 --halfwidth 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ExperimentNames,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.io.loading import (  # noqa: E402
    WAVELET_FEATURE_SEPARATOR,
    cache_participant_labels,
    concatenated_dir,
    read_wavelet_cache_header,
    stream_wavelet_cache_subjects,
    wavelet_cache_path,
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


def paired_participants(
    concat_dir: Path, labels: list[str], n_subjects: int
) -> list[str]:
    """The first *n_subjects* participants present in every condition, in a stable order.

    Keeping the same people in every strip is what lets a paired analysis run on it.

    :param concat_dir: Concatenation directory holding the metadata sidecars.
    :param labels: One dataset label per condition.
    :param n_subjects: How many to keep; ``0`` or fewer keeps all.
    :return: Participant ids common to every condition.
    :raises ValueError: If the conditions share no participant.
    """
    per_condition = [cache_participant_labels(concat_dir, label) for label in labels]
    shared = set(per_condition[0]).intersection(*per_condition[1:])
    if not shared:
        raise ValueError(f"No participant is present in all of {labels}.")
    ordered = [p for p in per_condition[0] if p in shared]
    return ordered if n_subjects <= 0 else ordered[:n_subjects]


def strip_one(
    source: Path,
    out_path: Path,
    keep: list[str],
    cache_labels: list[str],
    freq_indices: np.ndarray,
    n_times: int | None,
) -> None:
    """Write one strip, streaming the source a subject at a time.

    Only the kept subjects are retained, so peak memory is one subject's block plus the
    output array. The result is written to a sibling temp file and renamed, because a
    truncated write on a full volume would leave an unreadable cache behind.

    :param source: Full wavelet cache to read.
    :param out_path: Where the strip goes.
    :param keep: Participant ids to retain, in output order.
    :param cache_labels: Participant id per subject index of *source*.
    :param freq_indices: Frequency bins to retain.
    :param n_times: Time samples to retain, or ``None`` for all.
    """
    header = read_wavelet_cache_header(source)
    wanted = {name: row for row, name in enumerate(keep)}
    keep_times = header.n_times if n_times is None else min(n_times, header.n_times)
    n_channels = len(header.channel_names)
    data = np.empty(
        (len(keep), n_channels * freq_indices.size, keep_times), dtype=np.float32
    )
    seen = np.zeros(len(keep), dtype=bool)

    _logger.info(
        f"{source.name}: {source.stat().st_size / 1e9:.1f} GB -> "
        f"{len(keep)} subject(s) x {n_channels} channel(s) x {freq_indices.size} bin(s) "
        f"x {keep_times} sample(s)"
    )
    for subject, block in stream_wavelet_cache_subjects(
        source, freq_indices=freq_indices, n_times=keep_times, header=header
    ):
        name = cache_labels[subject] if subject < len(cache_labels) else None
        if name is None or name not in wanted:
            continue
        row = wanted[name]
        data[row] = block.reshape(n_channels * freq_indices.size, keep_times)
        seen[row] = True
        _logger.info(f"  {name} ({int(seen.sum())}/{len(keep)})")
        if seen.all():
            # The stream is subject-major, so once the last wanted one is in hand there
            # is nothing further to decompress.
            break

    if not seen.all():
        missing = [p for p, ok in zip(keep, seen) if not ok]
        raise ValueError(f"{source.name} held no block for {missing}.")

    freqs = header.freqs[freq_indices]
    feature_names = [
        f"{channel}{WAVELET_FEATURE_SEPARATOR}{freq:.3f}"
        for channel in header.channel_names
        for freq in freqs
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    try:
        # Pass the open handle, not the name: given a name, np.savez appends its own
        # ".npz" and the rename below would miss the file it actually wrote.
        with os.fdopen(handle, "wb") as stream:
            np.savez(
                stream,
                data=data,
                freqs=freqs,
                sfreq=header.sfreq,
                label=header.label,
                has_feature_names=True,
                feature_names=np.array(feature_names),
                # The strip keeps only some subjects, so its order no longer matches the
                # concatenation metadata. Without these labels a reader would name the
                # rows from that sidecar and silently mislabel every one.
                participants=np.array(keep),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _logger.info(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")


def run(args: argparse.Namespace) -> None:
    """Build a strip for every requested condition."""
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    conditions = args.conditions or [c.value for c in REAL_CONDITIONS]
    labels = [f"{condition}_{music_type.value}" for condition in conditions]

    concat_dir = concatenated_dir(experiment, args.source_dir)
    keep = paired_participants(concat_dir, labels, args.n_subjects)
    _logger.info(f"keeping {len(keep)} paired participant(s): {keep}")

    out_root = (
        Path(args.out_dir)
        / experiment.value
        / "wavelets"
        / SpectrumTypeVariants.BROADBAND.value
    )
    for label in labels:
        source = wavelet_cache_path(args.source_dir, experiment, label)
        header = read_wavelet_cache_header(source)
        if args.halfwidth > 0:
            freq_indices = np.flatnonzero(
                np.abs(header.freqs - args.center_freq) <= args.halfwidth
            )
            if freq_indices.size == 0:
                freq_indices = np.array(
                    [int(np.argmin(np.abs(header.freqs - args.center_freq)))]
                )
        else:
            freq_indices = np.arange(header.freqs.size)

        freqs = header.freqs[freq_indices]
        signature = f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{freqs.size}"
        out_path = out_root / f"{label}__wavelet_power__{signature}__freqdim1.npz"
        strip_one(
            source,
            out_path,
            keep,
            cache_participant_labels(concat_dir, label),
            freq_indices,
            args.n_times,
        )

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
