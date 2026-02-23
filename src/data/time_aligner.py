import gc
import sys
from typing import Callable

sys.path.append("/home/david/source/psilocybin-eeg")

from pathlib import Path
import pandas as pd
import numpy as np
import mne
from mne_icalabel.iclabel import iclabel_label_components
from scipy.signal import correlate, correlation_lags
import seaborn as sns
import matplotlib.pyplot as plt

# from src.data.dataset_handler import DatasetHandler
from src.data.dataset_preprocessing import DatasetPreprocessor
from src.data.dataset_filtering import DatasetFilter
from src.definitions.fields import (
    SingleDataMetadata,
    ChannelTypes,
    ExperimentNames,
    CoordinateSystems,
    PreprocessedDataVariants,
    ICLabelComponentsClasses,
    ExcludedICsMetadata,
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
)
from src.utils.logging_config import LoggerMixin
from src.definitions.constants import ProjectPaths


class TAGObject:
    """
    Object to summarize all necessary info about the tag signal.
    """

    def __init__(self, filename: str, tag_signal):
        self.filename = filename
        self.tag_signal = tag_signal


class TimeAligner(LoggerMixin):
    """
    Class to perform the alignment of the time signals across
    all individuals based on the TAG signal.
    """

    def __init__(self, all_tags: list[TAGObject], sfreq: float):
        self.all_tags = all_tags
        self.sfreq = sfreq
        self.corr_matrix, self.lag_matrix = (
            self._compute_all_signal_cross_correlations()
        )
        self.reference_idx, self.shifts = self._find_reference_signal()

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

    def crop_to_overlap(self) -> list[np.ndarray]:
        """
        Crop all signals to their common overlapping region given the shifts.
        `shifts[i]` = how many samples signal i is shifted relative to the reference.

        :return: List of cropped signals aligned to the reference.
        """
        # The overlap starts at the latest start and ends at the earliest end
        start = max(self.shifts)  # latest start across all signals
        signals = [tag.tag_signal for tag in self.all_tags]
        ends = [start + len(sig) - shift for sig, shift in zip(signals, self.shifts)]
        end = min(ends)  # earliest end across all signals

        cropped = []
        for sig, shift in zip(signals, self.shifts):
            crop_start = start - shift
            crop_end = crop_start + (end - start)
            cropped.append(sig[crop_start:crop_end])

        assert all(
            len(sig) == len(cropped[0]) for sig in cropped
        ), "All cropped signals should have the same length"
        return cropped
