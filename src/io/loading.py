"""
This module provides functions for loading raw and processed EEG data files.
"""

import logging
from pathlib import Path

import numpy as np
import mne

from src.definitions.fields import (
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
)

_logger = logging.getLogger(__name__)


def get_preprocessing_results_path(
    processed_data_dir: Path, filename: str, data_type: PreprocessedDataVariants
) -> Path:
    """
    Gets the path to a specified processing results.

    :param processed_data_dir: Root directory for processed data.
    :param filename: Name of the experiment file.
    :param data_type: Type of the processed data.
    :return: Returns path to the specified processing results.
    """
    suffix = ""
    if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
        suffix = ".fif"
    elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
        suffix = ".npy"

    return processed_data_dir / data_type.value / (filename + suffix)


def load_data_file(
    raw_data_dir: Path,
    processed_data_dir: Path,
    data_filename: str,
    is_processed: bool = False,
    processed_data_type: PreprocessedDataVariants = PreprocessedDataVariants.RAW_AFTER_ICA,
    preload=True,
) -> mne.io.Raw | np.ndarray:
    """
    Loads one EEG sequence (one data example).

    :param raw_data_dir: Directory containing raw data files.
    :param processed_data_dir: Directory containing processed data files.
    :param data_filename: Name of the file containing the wanted data.
    :param is_processed: Flag whether the data to load is already processed or not
    (from where we want to load the data).
    :param processed_data_type: Type of the processed file to load
    :param preload: Whether to preload data into memory.
    :return: Returns loaded data in the Raw data type.
    """
    data_path = raw_data_dir / data_filename
    if is_processed:
        # Load processed data file
        data_path = get_preprocessing_results_path(
            processed_data_dir,
            data_filename.split(".")[0],
            data_type=processed_data_type,
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
