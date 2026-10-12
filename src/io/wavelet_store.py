"""
The on-disk wavelet store: one float32 array per participant.

::

    data/processed/<experiment>/wavelets/<Condition>_<MusicType>/
        wavelet_<repr>__<f0>_<f1>_<n>.json                axes every subject shares
        <participant>__wavelet_<repr>__<f0>_<f1>_<n>.npy  (channels, freqs, times)

Per subject and uncompressed, because the old single compressed file per label cost
more than it saved: DEFLATE shrank float wavelet power by ~5 %, yet a compressed member
cannot be read from the middle, so every read decompressed whole subjects in order. An
uncompressed ``.npy`` is memory-mapped, so a reader takes any participant, channel
range, frequency bin or time span directly, and participants can be read in parallel.
float32 halves the store; wavelet power needs nowhere near float64's precision.

The participant id is part of the filename, so a row's identity never depends on an
ordering sidecar, and a reader asks for participants by name in whatever order it needs.
All participants of one label share a time base (stage-00 alignment), so their arrays
have equal length; :func:`load_wavelet_subjects` checks it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.definitions.constants import ProjectPaths
from src.definitions.fields import ExperimentNames

_logger = logging.getLogger(__name__)

WAVELETS_DIR_NAME = "wavelets"
STORE_DTYPE = np.float32
REPRESENTATIONS = ("power", "phase")


def grid_token(freqs: np.ndarray) -> str:
    """``"<f0>_<f1>_<n>"`` naming a frequency grid, e.g. ``"1.000_50.000_50"``."""
    freqs = np.asarray(freqs, dtype=float)
    return f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{freqs.size}"


def safe_label(label: str) -> str:
    """A dataset label made safe as a directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
    return cleaned or f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"


def wavelet_root(root: Path | None, experiment: ExperimentNames) -> Path:
    """``<root>/<experiment>/wavelets``; ``None`` uses ``data/processed``."""
    base = ProjectPaths.PROCESSED_DATA_DIR if root is None else Path(root)
    return base / experiment.value / WAVELETS_DIR_NAME


def wavelet_label_dir(
    root: Path | None, experiment: ExperimentNames, label: str
) -> Path:
    """Directory of one ``<Condition>_<MusicType>`` label's per-subject files."""
    return wavelet_root(root, experiment) / safe_label(label)


def _set_token(representation: str, freqs: np.ndarray) -> str:
    return f"wavelet_{representation}__{grid_token(freqs)}"


def _metadata_path(directory: Path, representation: str, freqs: np.ndarray) -> Path:
    return Path(directory) / f"{_set_token(representation, freqs)}.json"


@dataclass(frozen=True)
class WaveletSet:
    """One label's wavelets at one representation and frequency grid.

    Holds the axes shared by every subject; the subject arrays are opened on demand.
    """

    directory: Path
    label: str
    representation: str
    freqs: np.ndarray
    n_cycles: np.ndarray
    sfreq: float
    channel_names: list[str]

    @property
    def token(self) -> str:
        return _set_token(self.representation, self.freqs)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def n_freqs(self) -> int:
        return int(self.freqs.size)

    def subject_path(self, participant: str) -> Path:
        return self.directory / f"{participant}__{self.token}.npy"

    @property
    def participants(self) -> list[str]:
        """Every participant with a file in this set, sorted by id."""
        suffix = f"__{self.token}.npy"
        return sorted(
            path.name.removesuffix(suffix) for path in self.directory.glob(f"*{suffix}")
        )

    def missing(self, participants: Sequence[str]) -> list[str]:
        return [p for p in participants if not self.subject_path(p).exists()]

    def open(self, participant: str) -> np.ndarray:
        """One subject's ``(channels, freqs, times)`` array, memory-mapped read-only.

        :raises FileNotFoundError: If the participant has no file in this set.
        :raises ValueError: If the array's channel or frequency axis disagrees with
            the set's metadata.
        """
        path = self.subject_path(participant)
        if not path.exists():
            raise FileNotFoundError(
                f"No {self.representation} wavelets for participant {participant} in "
                f"{self.directory} (grid {grid_token(self.freqs)})."
            )
        array = np.load(path, mmap_mode="r")
        if array.ndim != 3 or array.shape[:2] != (self.n_channels, self.n_freqs):
            raise ValueError(
                f"{path.name} is {array.shape}; the set declares "
                f"({self.n_channels} channels, {self.n_freqs} freqs, times)."
            )
        return array


def find_wavelet_set(
    directory: Path,
    representation: str = "power",
    freqs: np.ndarray | None = None,
) -> WaveletSet:
    """Open the wavelet set of one label directory.

    :param directory: The label directory (:func:`wavelet_label_dir`).
    :param representation: ``"power"`` or ``"phase"``.
    :param freqs: The frequency grid wanted. ``None`` takes the only grid present, or
        the last in name order when several are (logged).
    :raises FileNotFoundError: If no matching set exists.
    """
    directory = Path(directory)
    if freqs is not None:
        candidates = [_metadata_path(directory, representation, freqs)]
        candidates = [path for path in candidates if path.exists()]
    else:
        candidates = sorted(directory.glob(f"wavelet_{representation}__*.json"))
    if not candidates:
        grid = "" if freqs is None else f" on grid {grid_token(freqs)}"
        raise FileNotFoundError(
            f"No {representation} wavelets{grid} in {directory}. Build them with "
            "jobs/metacentrum/03-wavelet-analysis/store_wavelets.pbs (music) or "
            "store_assr_wavelets.pbs (ASSR)."
        )
    if len(candidates) > 1:
        _logger.info(
            f"{directory.name}: {len(candidates)} grids stored, using "
            f"{candidates[-1].stem}; pass freqs to choose."
        )
    with open(candidates[-1]) as handle:
        meta = json.load(handle)
    return WaveletSet(
        directory=directory,
        label=meta["label"],
        representation=meta["representation"],
        freqs=np.asarray(meta["freqs"], dtype=float),
        n_cycles=np.asarray(meta["n_cycles"], dtype=float),
        sfreq=float(meta["sfreq"]),
        channel_names=list(meta["channel_names"]),
    )


def _replace_atomically(path: Path, write) -> None:
    """Write through a sibling temp file and rename it over *path*.

    A failed write (the volume runs near quota) then never leaves a truncated file
    under the real name for a later run to trust.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_wavelet_subject(
    directory: Path,
    participant: str,
    data: np.ndarray,
    *,
    label: str,
    representation: str,
    freqs: np.ndarray,
    n_cycles: np.ndarray | float,
    sfreq: float,
    channel_names: Sequence[str],
) -> Path:
    """Store one subject's ``(channels, freqs, times)`` wavelets as float32.

    The set's metadata is written with the first subject and checked against every
    later one, so one set never mixes channel layouts or sampling rates.

    :raises ValueError: If *data* does not match the axes, or the set already holds
        subjects with different axes.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}.")
    freqs = np.asarray(freqs, dtype=float)
    expected = (len(channel_names), freqs.size)
    if data.ndim != 3 or data.shape[:2] != expected:
        raise ValueError(
            f"Subject {participant}: wavelets are {data.shape}; expected "
            f"({expected[0]} channels, {expected[1]} freqs, times)."
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": label,
        "representation": representation,
        "freqs": freqs.tolist(),
        "n_cycles": np.broadcast_to(np.asarray(n_cycles, float), freqs.shape).tolist(),
        "sfreq": float(sfreq),
        "channel_names": list(channel_names),
    }
    meta_path = _metadata_path(directory, representation, freqs)
    if meta_path.exists():
        with open(meta_path) as handle:
            stored = json.load(handle)
        for key in ("channel_names", "sfreq", "n_cycles"):
            if not np.array_equal(np.asarray(stored[key]), np.asarray(meta[key])):
                raise ValueError(
                    f"{meta_path.name}: {key} differs from the subjects already "
                    f"stored; write this set to a fresh directory instead."
                )
    else:
        _replace_atomically(
            meta_path, lambda handle: handle.write(json.dumps(meta, indent=1).encode())
        )

    path = directory / f"{participant}__{_set_token(representation, freqs)}.npy"
    array = np.ascontiguousarray(data, dtype=STORE_DTYPE)
    _replace_atomically(path, lambda handle: np.save(handle, array))
    return path


def _selection(
    wavelet_set: WaveletSet,
    participants: Sequence[str],
    n_channels: int | None,
    n_times: int | None,
) -> tuple[int, int]:
    """Resolve the channel and time extent shared by *participants*.

    :raises FileNotFoundError: If any participant has no file.
    :raises ValueError: If the subjects' time axes differ or *n_times* exceeds them.
    """
    missing = wavelet_set.missing(participants)
    if missing:
        raise FileNotFoundError(
            f"{wavelet_set.directory}: no {wavelet_set.token} file for participant(s) "
            f"{missing}."
        )
    lengths = {p: wavelet_set.open(p).shape[-1] for p in participants}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"{wavelet_set.directory.name}: subjects have different time axes "
            f"{lengths}; one label must share one aligned time base."
        )
    available = next(iter(lengths.values()))
    if n_times is not None and n_times > available:
        raise ValueError(
            f"n_times={n_times} exceeds the stored {available}-sample time axis."
        )
    keep_channels = wavelet_set.n_channels if n_channels is None else n_channels
    return min(keep_channels, wavelet_set.n_channels), (
        available if n_times is None else n_times
    )


def iter_wavelet_subjects(
    wavelet_set: WaveletSet,
    participants: Sequence[str],
    *,
    freq_indices: Sequence[int] | None = None,
    n_channels: int | None = None,
    n_times: int | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(participant, (channels, freqs, times) float32)`` one subject at a time.

    For a whole-cohort pass that reduces each subject before the next is read, so
    peak memory is one subject's selection.
    """
    keep_channels, keep_times = _selection(
        wavelet_set, participants, n_channels, n_times
    )
    freq_sel = slice(None) if freq_indices is None else np.asarray(freq_indices, int)
    for participant in participants:
        array = wavelet_set.open(participant)
        yield (
            participant,
            np.array(array[:keep_channels, freq_sel, :keep_times], dtype=STORE_DTYPE),
        )


def load_wavelet_subjects(
    wavelet_set: WaveletSet,
    participants: Sequence[str],
    *,
    freq_indices: Sequence[int] | None = None,
    n_channels: int | None = None,
    n_times: int | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Stack *participants* into ``(subjects, channels, freqs, times)`` float32.

    Only the requested extent is read from each memory-mapped file: the first
    *n_channels* channels, the bins in *freq_indices* and the first *n_times*
    samples. Subjects are read by *n_jobs* threads (``-1`` = one per CPU), which
    overlap the file reads.

    :raises FileNotFoundError: If a participant has no file.
    :raises ValueError: If the subjects' time axes differ or *n_times* exceeds them.
    """
    keep_channels, keep_times = _selection(
        wavelet_set, participants, n_channels, n_times
    )
    freq_sel = slice(None) if freq_indices is None else np.asarray(freq_indices, int)
    n_keep_freqs = (
        wavelet_set.n_freqs if freq_indices is None else len(np.asarray(freq_indices))
    )
    out = np.empty(
        (len(participants), keep_channels, n_keep_freqs, keep_times), STORE_DTYPE
    )

    def _read(row: int) -> None:
        array = wavelet_set.open(participants[row])
        out[row] = array[:keep_channels, freq_sel, :keep_times]

    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    if workers > 1 and len(participants) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_read, range(len(participants))))
    else:
        for row in range(len(participants)):
            _read(row)
    return out
