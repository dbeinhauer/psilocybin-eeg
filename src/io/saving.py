"""
This module provides functions for saving processed EEG data outputs.
"""

import logging
import os
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
    interim_data_dir: Path | None = None,
):
    """
    Stores the selected data after preprocessing.

    :param data: Data to be stored.
    :param processed_data_dir: Root directory for final processed data.
    :param filename: Name of the file where the data should be stored (the path is computed
    based on the data type and default parameters.)
    :param data_type: Type of the data to be processed.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :param interim_data_dir: Root directory for intermediate products. If ``None``,
        falls back to ``processed_data_dir``.
    """
    log = logger or _logger

    log.info(f"Saving the '{data_type.value}' data into the file {filename}.")
    # Path to results file.
    data_path = get_preprocessing_results_path(
        processed_data_dir, filename, data_type, interim_data_dir=interim_data_dir
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_type in RAW_DATA_VARIANTS + [PreprocessedDataVariants.ICA_COMPONENTS]:
        data.save(data_path, overwrite=True)
    elif data_type == PreprocessedDataVariants.IC_PROBABILITIES:
        np.save(data_path, data)
    else:
        log.warning(f"Wrong datatype: '{data_type.value}' to store. Skipping!")
    log.info("Data saved successfully!")


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """``np.savez`` to a temporary name, then renamed into place, so a full disk never
    leaves a truncated file."""
    partial = path.with_name(f"{path.stem}.partial.npz")
    np.savez(partial, **arrays)
    os.replace(partial, path)


def save_snr_values(
    path: Path,
    values: dict[tuple[str, str, str], dict[str, np.ndarray]],
    participants: list[str],
    sources: list[str],
) -> None:
    """Write the per-participant values an ASSR read-out's tests are run on.

    Keys ``value__<variant>__<selection>__<window>__<condition>`` hold ``(P, S)``;
    ``participants`` and ``sources`` are the axes. Read back with
    :func:`src.io.loading.load_snr_values`.

    :param values: ``(variant, selection, window)`` → condition → ``(P, S)``, each
        participant's median over trials of each source's window mean.
    """
    arrays = {"participants": np.asarray(participants), "sources": np.asarray(sources)}
    for (variant, selection, window), by_condition in values.items():
        for condition, value in by_condition.items():
            arrays[f"value__{variant}__{selection}__{window}__{condition}"] = value
    save_npz_atomic(path, **arrays)


def save_snr_courses(
    path: Path,
    courses: dict[tuple[str, str], dict[str, np.ndarray]],
    participants: list[str],
    sources: list[str],
    times: np.ndarray,
) -> None:
    """Write the per-participant trial courses behind an ASSR SNR grid's course figure.

    Keys ``course__<variant>__<selection>__<condition>`` hold ``(P, S, W)``;
    ``participants``, ``sources`` and ``times`` are the axes. Read back with
    :func:`src.io.loading.load_snr_courses`.

    :param courses: ``(variant, selection)`` → condition → ``(P, S, W)``, each
        participant's median over trials of each source's baseline-referenced course.
    """
    arrays = {
        "participants": np.asarray(participants),
        "sources": np.asarray(sources),
        "times": np.asarray(times),
    }
    for (variant, selection), by_condition in courses.items():
        for condition, course in by_condition.items():
            arrays[f"course__{variant}__{selection}__{condition}"] = course.astype(
                np.float32
            )
    save_npz_atomic(path, **arrays)
