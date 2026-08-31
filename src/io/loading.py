"""
This module provides functions for loading raw and processed EEG data files,
plus the checked-in electrode-name lists that select channels for an analysis.
"""

import logging
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import mne
import pandas as pd

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    CoordinateSystems,
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
    INTERIM_DATA_VARIANTS,
)

_logger = logging.getLogger(__name__)


def _resolve_data_dir(
    processed_data_dir: Path,
    interim_data_dir: Path | None,
    data_type: PreprocessedDataVariants,
) -> Path:
    """
    Return the correct base directory for a given data type.

    Interim variants (before_ica, ica_components, ic_probabilities) go to
    ``interim_data_dir``; all other variants go to ``processed_data_dir``.

    :param processed_data_dir: Root directory for final processed data.
    :param interim_data_dir: Root directory for intermediate products.
        If ``None``, falls back to ``processed_data_dir`` for backwards compatibility.
    :param data_type: The preprocessing data variant.
    :return: The resolved base directory.
    """
    if interim_data_dir is not None and data_type in INTERIM_DATA_VARIANTS:
        return interim_data_dir
    return processed_data_dir


def get_preprocessing_results_path(
    processed_data_dir: Path,
    filename: str,
    data_type: PreprocessedDataVariants,
    interim_data_dir: Path | None = None,
) -> Path:
    """
    Gets the path to a specified processing results.

    :param processed_data_dir: Root directory for final processed data.
    :param filename: Name of the experiment file.
    :param data_type: Type of the processed data.
    :param interim_data_dir: Root directory for intermediate products.
        If ``None``, falls back to ``processed_data_dir``.
    :return: Returns path to the specified processing results.
    """
    suffix = ""
    if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
        suffix = ".fif"
    elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
        suffix = ".npy"

    base_dir = _resolve_data_dir(processed_data_dir, interim_data_dir, data_type)
    return base_dir / data_type.value / (filename + suffix)


def load_data_file(
    raw_data_dir: Path,
    processed_data_dir: Path,
    data_filename: str,
    is_processed: bool = False,
    processed_data_type: PreprocessedDataVariants = PreprocessedDataVariants.RAW_AFTER_ICA,
    preload=True,
    interim_data_dir: Path | None = None,
) -> mne.io.Raw | np.ndarray:
    """
    Loads one EEG sequence (one data example).

    :param raw_data_dir: Directory containing raw data files.
    :param processed_data_dir: Directory containing final processed data files.
    :param data_filename: Name of the file containing the wanted data.
    :param is_processed: Flag whether the data to load is already processed or not
    (from where we want to load the data).
    :param processed_data_type: Type of the processed file to load
    :param preload: Whether to preload data into memory.
    :param interim_data_dir: Directory for intermediate products. If ``None``,
        falls back to ``processed_data_dir``.
    :return: Returns loaded data in the Raw data type.
    """
    data_path = raw_data_dir / data_filename
    if is_processed:
        # Load processed data file
        data_path = get_preprocessing_results_path(
            processed_data_dir,
            data_filename.split(".")[0],
            data_type=processed_data_type,
            interim_data_dir=interim_data_dir,
        )
        if processed_data_type == PreprocessedDataVariants.IC_PROBABILITIES:
            # Load IC Probabilities
            return np.load(data_path)
        elif processed_data_type == PreprocessedDataVariants.ICA_COMPONENTS:
            # Load ICA components
            return mne.preprocessing.read_ica(
                data_path,
            )
        else:
            # We need this else for Raw dataseries are in '.fif' format.
            return mne.io.read_raw_fif(
                data_path,
                preload=preload,
            )

    # Load unprocessed raw data are in '.edf' format.
    return mne.io.read_raw_edf(
        data_path,
        preload=preload,
    )


def load_electrode_name_list(electrode_list_path: Path) -> list[str]:
    """
    Load a checked-in electrode-name list CSV.

    Shared reader for every one-column ``electrode_name`` list under ``config/``:
    the exclude-lists in ``config/excluded_electrodes/`` and the ASSR include-lists
    in ``config/assr_electrodes/``. Keeping one reader is what keeps the two
    formats identical.

    Names are stripped of surrounding whitespace, because the checked-in
    exclude-lists store them padded (``"E67 "``).

    :param electrode_list_path: Path to the CSV.
    :return: Electrode names, in file order.
    :raises FileNotFoundError: If the CSV does not exist.
    """
    if not electrode_list_path.exists():
        # Name what *is* there: not every coordinate system has every list, so a
        # missing file is usually the wrong CoordinateSystem rather than a typo.
        if electrode_list_path.parent.exists():
            available = sorted(p.name for p in electrode_list_path.parent.glob("*.csv"))
            hint = f"Available lists in {electrode_list_path.parent}: {available}"
        else:
            hint = f"Directory {electrode_list_path.parent} does not exist."
        raise FileNotFoundError(
            f"Electrode list not found: {electrode_list_path}. {hint}"
        )
    return pd.read_csv(electrode_list_path)["electrode_name"].str.strip().tolist()


def load_assr_electrodes(
    coordinate_system: CoordinateSystems = (
        CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    ),
) -> list[str]:
    """
    Load the electrodes to KEEP for the ASSR baseline check.

    The standard fronto-central selection the 40 Hz steady-state response is read
    from. An include-list, unlike the exclude-lists used during preprocessing.

    :param coordinate_system: Coordinate system whose list to load. Only the
        systems an ASSR check has been defined for have one.
    :return: Electrode names, in file order.
    :raises FileNotFoundError: If no list exists for *coordinate_system*.
    """
    return load_electrode_name_list(
        ProjectPaths.get_assr_electrodes_file_path(coordinate_system)
    )


def assr_electrode_mask(
    channel_names: list[str],
    coordinate_system: CoordinateSystems = (
        CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    ),
    *,
    strict: bool = True,
) -> np.ndarray:
    """
    Boolean mask selecting the ASSR-relevant electrodes out of *channel_names*.

    The mask is aligned to *channel_names*, so it indexes a data array's channel
    axis directly::

        mask = assr_electrode_mask(channel_names)
        assr_data = data[mask]                  # (n_assr_channels, ...)
        assr_names = [n for n, keep in zip(channel_names, mask) if keep]

    *strict* guards the case that actually happens: preprocessing drops the
    boundary electrodes, so a recording carries fewer channels than the montage
    and a listed electrode can be absent. Silently returning a smaller selection
    would quietly change what the baseline check averages over, so by default that
    raises rather than under-selecting. Pass ``strict=False`` to accept the
    intersection and log which electrodes were missing.

    :param channel_names: Channel names of the data to be filtered, in data order.
    :param coordinate_system: Coordinate system whose ASSR list to use.
    :param strict: Raise if a listed electrode is absent from *channel_names*.
    :return: Boolean array of ``len(channel_names)``, true for ASSR electrodes.
    :raises FileNotFoundError: If no ASSR list exists for *coordinate_system*.
    :raises ValueError: If *strict* and any listed electrode is missing, or if
        *channel_names* contains duplicates.
    """
    duplicates = sorted(
        {name for name in channel_names if channel_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            f"channel_names contains duplicate entries {duplicates}; the mask "
            f"would be ambiguous."
        )

    assr_electrodes = load_assr_electrodes(coordinate_system)
    available = set(channel_names)
    missing = [name for name in assr_electrodes if name not in available]
    if missing:
        message = (
            f"{len(missing)}/{len(assr_electrodes)} ASSR electrode(s) are absent "
            f"from the data: {missing}. Preprocessing drops the boundary "
            f"electrodes, so a recording can legitimately be missing some — but "
            f"the baseline check would then average over a smaller selection than "
            f"the list defines."
        )
        if strict:
            raise ValueError(f"{message} Pass strict=False to accept the rest.")
        _logger.warning(message)

    keep = set(assr_electrodes)
    return np.array([name in keep for name in channel_names], dtype=bool)


# ---------------------------------------------------------------------------
# Streaming reads of the source-of-truth wavelet cache
# ---------------------------------------------------------------------------

#: Name of the tensor entry inside a wavelet-cache ``.npz``.
WAVELET_CACHE_ARRAY = "data"

#: Separator between channel name and frequency in a wavelet cache's
#: ``feature_names`` (``"E1@40.0Hz"``). The flattened feature axis is
#: **channel-major** — every channel's whole frequency block precedes the next
#: channel's — which is what makes the channel-block streaming read below possible.
WAVELET_FEATURE_SEPARATOR = "@"


@dataclass(frozen=True)
class WaveletCacheHeader:
    """What a wavelet cache holds, read without decompressing its tensor.

    The source-of-truth caches under ``data/processed/<experiment>/wavelets/`` are
    tens of gigabytes of DEFLATE-compressed ``float64``, so ``np.load(...)["data"]``
    is not an option — it materialises the whole tensor, and being compressed the
    entry cannot be memory-mapped either. Everything needed to decide *what to read*
    lives in the small entries beside it, and this reads only those.

    :param path: The cache file.
    :param shape: On-disk tensor shape, ``(subjects, channels * freqs, times)`` — the
        frequency axis is folded into the feature axis, which is what the ``freqdim1``
        filename token records.
    :param dtype: On-disk dtype.
    :param channel_names: Channel names in data order, recovered from the flattened
        ``feature_names``.
    :param freqs: Morlet frequency grid in Hz.
    :param sfreq: Sampling frequency in Hz.
    :param label: Dataset label the cache was written under.
    """

    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    channel_names: list[str]
    freqs: np.ndarray
    sfreq: float
    label: str

    @property
    def n_subjects(self) -> int:
        """Length of the subject axis."""
        return int(self.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels on the folded feature axis."""
        return len(self.channel_names)

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins on the folded feature axis."""
        return int(self.freqs.size)

    @property
    def n_times(self) -> int:
        """Length of the time axis."""
        return int(self.shape[-1])


def read_wavelet_cache_header(path: Path) -> WaveletCacheHeader:
    """Describe a wavelet cache without decompressing its tensor.

    :param path: Path to a ``__wavelet_<repr>__<f0>_<f1>_<n>__freqdim1.npz`` cache.
    :return: The cache's axes and bookkeeping.
    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If the file has no ``data`` entry, the tensor is not 3-D, it
        carries no feature names, or the folded feature axis does not factor into the
        channel and frequency counts the small entries declare.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No wavelet cache at {path}.")

    with zipfile.ZipFile(path) as archive:
        names = {info.filename for info in archive.infolist()}
        if f"{WAVELET_CACHE_ARRAY}.npy" not in names:
            raise ValueError(
                f"{path.name} has no '{WAVELET_CACHE_ARRAY}' entry; it is not a "
                f"wavelet cache (holds {sorted(n.removesuffix('.npy') for n in names)})."
            )
        with archive.open(f"{WAVELET_CACHE_ARRAY}.npy") as handle:
            shape, _fortran, dtype = _read_npy_header(handle)

    # The remaining entries are kilobytes, so reading them normally is free.
    with np.load(path) as cached:
        freqs = np.asarray(cached["freqs"], dtype=float)
        sfreq = float(cached["sfreq"])
        label = str(cached["label"])
        if not cached["has_feature_names"].item():
            raise ValueError(
                f"{path.name} carries no feature names, so its channel axis cannot be "
                "named — and a spatial filter cannot be aligned to it."
            )
        feature_names = [str(name) for name in cached["feature_names"]]

    if len(shape) != 3:
        raise ValueError(
            f"{path.name} holds a {len(shape)}-D tensor {shape}; a wavelet cache is "
            "(subjects, channels * freqs, times)."
        )

    n_freqs = int(freqs.size)
    channel_names = [
        name.split(WAVELET_FEATURE_SEPARATOR, 1)[0] for name in feature_names[::n_freqs]
    ]
    if len(channel_names) * n_freqs != shape[1]:
        raise ValueError(
            f"{path.name}: the feature axis is {shape[1]} long but "
            f"{len(channel_names)} channel(s) x {n_freqs} frequency bin(s) is "
            f"{len(channel_names) * n_freqs}."
        )

    return WaveletCacheHeader(
        path=path,
        shape=tuple(int(n) for n in shape),
        dtype=np.dtype(dtype),
        channel_names=channel_names,
        freqs=freqs,
        sfreq=sfreq,
        label=label,
    )


def _read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    """``(shape, fortran_order, dtype)`` of a ``.npy`` stream, left positioned at data.

    :param handle: Binary stream positioned at the start of a ``.npy`` payload.
    :return: The header triple ``numpy`` itself parses.
    :raises ValueError: If the ``.npy`` format version is not supported.
    """
    version = np.lib.format.read_magic(handle)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ValueError(f"Unsupported .npy format version {version}.")
    return readers[version](handle)


def _read_exact(handle: BinaryIO, n_bytes: int) -> bytes:
    """Read exactly *n_bytes*, looping until the stream delivers them.

    A ``ZipExtFile`` over a DEFLATE member returns whatever the decompressor happens
    to have ready, so one ``read(n)`` is not guaranteed to fill the request.

    :param handle: Stream to read from.
    :param n_bytes: Number of bytes wanted.
    :return: Exactly *n_bytes* bytes.
    :raises EOFError: If the stream ends first.
    """
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining > 0:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError(
                f"Wavelet cache stream ended {remaining} byte(s) short of the "
                f"{n_bytes} requested; the file is truncated."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def stream_wavelet_cache_subjects(
    path: Path,
    *,
    freq_indices: Sequence[int] | None = None,
    n_times: int | None = None,
    header: WaveletCacheHeader | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield one subject at a time from a wavelet cache, keeping only some frequencies.

    The source caches are ~50 GB of DEFLATE-compressed ``float64`` — too large to
    load, and not memory-mappable because they are compressed. But the tensor is
    stored ``(subject, channel * frequency, time)`` in C order, so it decompresses in
    **subject-major, then channel-major** order: a single sequential pass can hand
    back one subject's channels while holding only one channel's frequency block in
    memory.

    Restricting *freq_indices* is what makes the pass worth doing for a narrow-band
    analysis. The discarded frequencies are still decompressed — there is no seeking
    inside a DEFLATE stream — but they are never accumulated, so each yielded array is
    a factor ``n_freqs / len(freq_indices)`` smaller than the tensor's own subject
    block. Because a spatial filter contracts only the channel axis, slicing frequency
    here is exactly equivalent to slicing it after the projection, and far cheaper.

    :param path: Path to the wavelet cache.
    :param freq_indices: Frequency-bin indices to keep, in the order wanted. ``None``
        keeps every bin.
    :param n_times: Keep only the first *n_times* samples. ``None`` keeps all.
    :param header: A previously read header, to avoid re-reading it.
    :yield: ``(subject_index, array)`` with *array* shaped
        ``(channels, len(freq_indices), n_times)`` in the cache's own dtype.
    :raises IndexError: If a frequency index is out of range.
    :raises ValueError: If *n_times* exceeds the cache's time axis.
    """
    resolved = header if header is not None else read_wavelet_cache_header(path)

    selection = (
        np.arange(resolved.n_freqs)
        if freq_indices is None
        else np.asarray(freq_indices, dtype=int)
    )
    if selection.size and (selection.min() < 0 or selection.max() >= resolved.n_freqs):
        raise IndexError(
            f"Frequency index out of range: the cache has {resolved.n_freqs} bin(s) "
            f"({resolved.freqs[0]:.1f}-{resolved.freqs[-1]:.1f} Hz)."
        )
    keep_times = resolved.n_times if n_times is None else int(n_times)
    if keep_times > resolved.n_times:
        raise ValueError(
            f"n_times={keep_times} exceeds the cache's {resolved.n_times}-sample "
            "time axis."
        )

    itemsize = resolved.dtype.itemsize
    block_bytes = resolved.n_freqs * resolved.n_times * itemsize

    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{WAVELET_CACHE_ARRAY}.npy") as handle:
            _read_npy_header(handle)  # advances past the header to the payload
            for subject in range(resolved.n_subjects):
                out = np.empty(
                    (resolved.n_channels, selection.size, keep_times),
                    dtype=resolved.dtype,
                )
                for channel in range(resolved.n_channels):
                    block = np.frombuffer(
                        _read_exact(handle, block_bytes), dtype=resolved.dtype
                    ).reshape(resolved.n_freqs, resolved.n_times)
                    out[channel] = block[selection, :keep_times]
                yield subject, out
