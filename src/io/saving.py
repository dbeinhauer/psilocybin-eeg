"""
This module provides functions for saving processed EEG data outputs.
"""

import logging
from pathlib import Path

import numpy as np
import mne

from src.definitions.fields import (
    PreprocessedDataVariants,
    RAW_DATA_VARIANTS,
)
from src.io.loading import get_preprocessing_results_path

_logger = logging.getLogger(__name__)


def save_data_file(
    data: mne.io.Raw | mne.preprocessing.ICA | np.ndarray,
    processed_data_dir: Path,
    filename: str,
    data_type: PreprocessedDataVariants,
    logger=None,
):
    """
    Stores the selected data after preprocessing.

    :param data: Data to be stored.
    :param processed_data_dir: Root directory for processed data.
    :param filename: Name of the file where the data should be stored (the path is computed
    based on the data type and default parameters.)
    :param data_type: Type of the data to be processed.
    :param logger: Optional logger instance. Falls back to module-level logger.
    """
    log = logger or _logger

    log.info(
        f"Saving the '{data_type.value}' data into the file {filename}."
    )
    # Path to results file.
    data_path = get_preprocessing_results_path(processed_data_dir, filename, data_type)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
        data.save(data_path, overwrite=True)
    elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
        np.save(data_path, data)
    else:
        log.warning(
            f"Wrong datatype: '{data_type.value}' to store. Skipping!"
        )
    log.info("Data saved successfully!")
