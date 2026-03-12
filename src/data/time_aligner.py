"""
This module implements time alignment of EEG recordings across participants.

It uses the TAG (music/stimulus) channel to cross-correlate signals and determine
temporal offsets between recordings, enabling consistent time-locked analysis across
all participants in the experiment.

Key classes:
    - TAGObject: Container for TAG signal data and metadata.
    - TimeAligner: Performs cross-correlation-based alignment of TAG signals
      and crops raw EEG data to the common overlapping time window.
"""

import numpy as np
from scipy.signal import correlate, correlation_lags

# from src.data.dataset_handler import DatasetHandler
from src.utils.logging_config import LoggerMixin


class TAGObject:
    """
    Object to summarize all necessary info about the tag signal.
    """

    def __init__(self, filename: str, tag_signal):
        """
        :param filename: Original filename of the recording this TAG signal belongs to.
        :param tag_signal: The extracted TAG (stimulus/music) channel signal array.
        """
        self.filename = filename
        self.tag_signal = tag_signal


class TimeAligner(LoggerMixin):
    """
    Class to perform the alignment of the time signals across
    all individuals based on the TAG signal.
    """

    def __init__(self, all_tags: list[TAGObject], sfreq: float):
        """
        Initialize the TimeAligner and compute alignment parameters.

        Cross-correlates all TAG signal pairs, identifies the best reference signal,
        computes per-signal shifts, and determines the common overlapping time window.

        :param all_tags: List of TAGObject instances containing TAG signals to align.
        :param sfreq: Sampling frequency of the TAG signals in Hz.
        """
        # All TAG signals to be aligned, along with their filenames for reference and sampling frequency.
        self.all_tags = all_tags
        self.sfreq = sfreq

        # Correlation matrix and lag matrix (for highest correlation) between all pairs of the TAG signals.
        self.corr_matrix, self.lag_matrix = (
            self._compute_all_signal_cross_correlations()
        )

        # Reference signal index (the one with the highest average correlation to all others),
        # and the shifts needed to align all signals to this reference.
        self.reference_idx, self.shifts = self._find_reference_signal()

        # Common overlapping region across all signals after applying the shifts to the reference signal.
        self.start, self.end = self._find_latest_start_and_earliest_end_of_signals()

    def _compute_all_signal_cross_correlations(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the cross-correlation between all pairs of TAG signals and determine the optimal lag for alignment.
        :return: Correlation matrix and lag matrix for all pairs of TAG signals
        """
        n = len(self.all_tags)
        signals = [tag.tag_signal for tag in self.all_tags]

        corr_matrix = np.zeros((n, n))
        lag_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = signals[i], signals[j]
                # Normalize before correlating for fair comparison
                s1_norm = (s1 - s1.mean()) / (s1.std() + 1e-10)
                s2_norm = (s2 - s2.mean()) / (s2.std() + 1e-10)

                corr = correlate(s1_norm, s2_norm, mode="full")
                lags = correlation_lags(len(s1_norm), len(s2_norm), mode="full")

                corr_normalized = corr / max(len(s1_norm), len(s2_norm))

                best_lag_idx = np.argmax(corr_normalized)
                best_corr = corr_normalized[best_lag_idx]
                best_lag = lags[best_lag_idx]

                corr_matrix[i, j] = best_corr
                corr_matrix[j, i] = best_corr
                lag_matrix[i, j] = best_lag  # shift to apply to s2 to align with s1
                lag_matrix[j, i] = -best_lag  # opposite shift for reverse direction

        return corr_matrix, lag_matrix

    def _find_reference_signal(self) -> tuple[int, np.ndarray]:
        """
        Find the index of the reference signal to which all
        others will be aligned (the one with the highest average correlation to all others).
        :return: Index of the reference signal and shifts for alignment.
        """
        avg_correlation = self.corr_matrix.mean(axis=1)
        reference_idx = np.argmax(avg_correlation)
        shifts = self.lag_matrix[reference_idx]

        self.logger.info(
            f"Best reference: index {reference_idx} ({self.all_tags[reference_idx].filename})"
        )
        self.logger.info(f"Average correlations: {np.round(avg_correlation, 4)}")
        return int(reference_idx), shifts

    def _find_latest_start_and_earliest_end_of_signals(
        self,
    ) -> tuple[int, int]:
        """
        Find the latest start and earliest end of all signals after applying the shifts
        to reference signal. This determines the common overlapping region across all signals.

        :return: Tuple of (latest start index, earliest end index) in the reference signal's time frame.
        """
        start = max(
            self.shifts
        )  # latest start across all signals (just the highest shift).

        # Earliest end across signals is determined by the lowest index of the ends after shifts are applied.
        signals = [tag.tag_signal for tag in self.all_tags]
        ends = [start + len(sig) - shift for sig, shift in zip(signals, self.shifts)]
        end = min(ends)

        return start, end

    def get_crop_indices_for_signal(self, shift: int) -> tuple[int, int]:
        """
        Calculates start and end index for the signal cropping based on
        the shift to reference.

        :param shift: Shift of the signal relative to reference (in samples).
        :return: Tuple of (crop_start, crop_end) indices in the reference signal's time frame (in samples).
        """
        crop_start = self.start - shift
        crop_end = crop_start + (self.end - self.start)
        return crop_start, crop_end

    def crop_to_overlap(
        self,
    ) -> list[np.ndarray]:
        """
        Crop all signals to their common overlapping region given the shifts.
        `shifts[i]` = how many samples signal i is shifted relative to the reference.

        :return: List of cropped signals.
        """
        signals = [tag.tag_signal for tag in self.all_tags]
        expected_length = self.end - self.start

        cropped = []
        for sig, shift in zip(signals, self.shifts):
            crop_start, crop_end = self.get_crop_indices_for_signal(shift)
            cropped.append(sig[crop_start:crop_end])

        assert all(len(sig) == expected_length for sig in cropped), (
            f"All cropped signals should have the same length ({expected_length}), got signals with lengths {[len(sig) for sig in cropped]}."
        )
        return cropped
