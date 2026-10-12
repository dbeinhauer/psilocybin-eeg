"""
This module provides functions for loading raw and processed EEG data files,
plus the checked-in electrode-name lists that select channels for an analysis.
Wavelet transforms are read through :mod:`src.io.wavelet_store`.
"""

import logging
from pathlib import Path

import numpy as np
import mne
import pandas as pd

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    CoordinateSystems,
    ExperimentNames,
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
# Concatenation sidecars
# ---------------------------------------------------------------------------


def concatenated_dir(experiment: ExperimentNames, root: Path | None = None) -> Path:
    """Directory holding one experiment's concatenated arrays and their sidecars.

    :param experiment: Which experiment.
    :param root: Processed-data root; ``None`` uses the project's.
    :return: The concatenation directory.
    """
    base = ProjectPaths.PROCESSED_DATA_DIR if root is None else Path(root)
    return base / experiment.value / PreprocessedDataVariants.CONCATENATED.value
