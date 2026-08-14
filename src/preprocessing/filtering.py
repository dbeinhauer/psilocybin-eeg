"""
This module contains functions for EEG signal filtering, including
notch/FIR filtering, RANSAC bad channel detection, and bad epoch handling.
"""

import logging

import numpy as np
import mne
from autoreject import Ransac, AutoReject

_logger = logging.getLogger(__name__)


def apply_filters_to_data(
    data: mne.io.Raw,
    notch_frequencies: tuple[int, int, int] = (50, 250, 50),
    filter_boundaries: tuple[float, float] = (1.0, 100.0),
    logger=None,
) -> mne.io.Raw:
    """
    Apply notch filter to filter the line noise.

    :param data: Data to be filtered.
    :param notch_frequencies: Frequencies to filter by notch filter.
    :param filter_boundaries: Boundaries to aply FIR filter on.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns filtered data.
    """
    log = logger or _logger

    log.info("Applying notch filter and FIR filter to remove line noise.")
    freqs = np.arange(*notch_frequencies)
    data = data.notch_filter(freqs=freqs)  # Filter 50 Hz line noise and harmonics
    return data.filter(
        l_freq=filter_boundaries[0], h_freq=filter_boundaries[1], method="fir"
    )


def apply_ransac_filter(
    data: mne.io.Raw,
    epoch_duration: float = 2.0,
    ransac_epochs: int = 100,
    n_jobs: int = 1,
    logger=None,
):
    """
    Applies Ransac bad channel detection and marks bad
    channels in the data itself.

    :param data: Data to be processes..
    :param epoch_duration: Duration of the time step (for discretization).
    :param ransac_epochs: Number of epochs in the Ransac processing.
    :param n_jobs: Number of parallel jobs for Ransac. Default is 1 (no parallelism).
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns data labeled as good/bad channels based on the Ransac.
    """
    log = logger or _logger

    log.info("Applying Ransac algorithm to detect bad channels.")
    epochs = mne.make_fixed_length_epochs(data, duration=epoch_duration, preload=True)

    # Automatic detection of bad channels with RANSAC -> interpolation later
    ransac = Ransac(
        n_resample=ransac_epochs,
        min_channels=0.5,
        min_corr=0.7,
        unbroken_time=0.2,
        random_state=97,
        n_jobs=n_jobs,
    )
    ransac.fit(epochs)

    # ransac.bad_chs_ is a list of channel names it considers bad
    data.info["bads"] = list(set(data.info["bads"]).union(ransac.bad_chs_))

    return data


def remove_bad_epoch_annotations(data: mne.io.Raw) -> mne.io.Raw:
    """
    Removes bad epoch annotations from the data.

    :param data: Data to be processed.
    :return: _description_
    """
    old_annotations = data.annotations
    keep_annotations = [
        i for i, desc in enumerate(old_annotations.description) if desc != "BAD_epoch"
    ]

    new_annotations = mne.Annotations(
        onset=old_annotations.onset[keep_annotations],
        duration=old_annotations.duration[keep_annotations],
        description=old_annotations.description[keep_annotations],
        orig_time=old_annotations.orig_time,
    )
    return data.set_annotations(new_annotations)


def detect_bad_epochs(
    data: mne.io.Raw, epoch_len: float = 2.0, n_jobs: int = 1, logger=None
):
    """
    Runs Autoreject bad epochs detection and annotates putatively bad epochs in the raw data.

    :param data: Data to be analyzed.
    :param epoch_len: Length of the epochs.
    :param n_jobs: Number of parallel jobs for AutoReject. Default is 1 (no parallelism).
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns annotated data with bad epochs.
    """
    log = logger or _logger

    log.info("Running AutoReject to detect bad epochs.")
    epochs = mne.make_fixed_length_epochs(data, duration=epoch_len, preload=True)

    # Bad epochs detection
    ar = AutoReject(n_jobs=n_jobs, random_state=42, verbose=False)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    bad_epoch_indices = np.where(reject_log.bad_epochs)[0]
    log.info(f"Found {len(bad_epoch_indices)} bad epochs out of {len(epochs)}")

    # This marks bad segments WITHOUT removing them. The new annotations must share
    # the same time origin as any existing annotations (e.g. stimulus markers on the
    # ASSR data, which carry the recording's meas_date); otherwise MNE refuses to
    # concatenate them.
    bad_annotations = mne.Annotations(
        onset=[epochs.events[i, 0] / data.info["sfreq"] for i in bad_epoch_indices],
        duration=[epoch_len] * len(bad_epoch_indices),  # duration of each epoch
        description=["BAD_epoch"] * len(bad_epoch_indices),
        orig_time=data.annotations.orig_time,
    )

    # Add annotations to raw data (preserving any pre-existing annotations).
    return data.set_annotations(data.annotations + bad_annotations)


def interpolate_bad_channels(data: mne.io.Raw, logger=None) -> mne.io.Raw:
    """
    Interpolates the channels that are marked as bad using average reference.

    :param data: Data to be interpolated (the bad channels needs to be alredy labeled).
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Returns copy of the original data with interpolated bad channels.
    """
    log = logger or _logger

    log.info("Interpolating bad channels using average reference.")
    interpolated_data = data.interpolate_bads(reset_bads=True)
    return interpolated_data.set_eeg_reference("average", ch_type="eeg")
