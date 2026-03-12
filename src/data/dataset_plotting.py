"""
This source serves to plot the dataset in various ways, such as topomaps,
power spectra, and correlation heatmaps for aligned signals.
"""

from pathlib import Path
from typing import Literal

import mne
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import numpy as np
from mne.viz import plot_topomap
import seaborn as sns

from src.definitions.constants import ProjectPaths
from src.definitions.fields import PreprocessedDataVariants, RAW_DATA_VARIANTS
from src.utils.logging_config import LoggerMixin


class DatasetPlotter(LoggerMixin):
    """
    Utility class for plotting EEG data including topomaps, power spectra,
    signal overlaps, and cross-correlation heatmaps for alignment analysis.
    """

    @staticmethod
    def get_plot_path(
        filename: str, subdir_name: str, variant_name: str, custom_full_path: str
    ) -> Path:
        """
        Based on the provided parameters create path where to store the plot.

        Path to a plot will be created based on the following pattern:
            `ProjectPaths.PLOTS_PATH / subdir_name / variant_name / filename`

        :param filename: Name of the plot file.
        :param subdir_name: Name of the plot type subdirectory.
        :param variant_name: Name of the data variant type.
        :param custom_full_path: Full path to a plot (in case we want custom one).
        :return: Returns path to a plot.
        """
        if custom_full_path:
            # If full path provided -> we just take it and
            return Path(custom_full_path)

        return ProjectPaths.PLOTS_PATH / subdir_name / variant_name / filename

    def plot_topomap_combined(data: mne.io.Raw, title: str = ""):
        """
        Plots topomap for all bands separately and combined.
        """
        bands = {
            "Delta": (1, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta": (12, 30),
            "Gamma": (30, 45),
        }

        # Create figure with 6 subplots (5 bands + 1 combined)
        fig, axes = plt.subplots(1, len(bands) + 1, figsize=(18, 3))

        # Store RMS power for combined plot
        all_rms_powers = []

        # Plot individual bands
        for idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            # Filter data to this band
            data_band = data.copy().filter(
                l_freq=fmin, h_freq=fmax, picks="eeg", verbose=False
            )
            # Calculate RMS power per channel
            band_data = data_band.get_data(picks="eeg")
            rms_power = np.sqrt(np.mean(band_data**2, axis=1))
            all_rms_powers.append(rms_power)

            # Plot topomap
            im, _ = plot_topomap(
                rms_power,
                data.info,
                axes=axes[idx],
                show=False,
                cmap="RdBu_r",
                contours=6,
            )
            axes[idx].set_title(f"{band_name}\n({fmin}-{fmax} Hz)", fontsize=11)
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Plot combined (all bands)
        data_broadband = data.copy().filter(
            l_freq=1, h_freq=45, picks="eeg", verbose=False
        )
        broadband_data = data_broadband.get_data(picks="eeg")
        combined_rms = np.sqrt(np.mean(broadband_data**2, axis=1))

        im, _ = plot_topomap(
            combined_rms,
            data.info,
            axes=axes[-1],
            show=False,
            cmap="RdBu_r",
            contours=6,
        )
        axes[-1].set_title("ALL BANDS\n(1-45 Hz)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_raw_dataseries(
        data: mne.io.Raw,
        save_fig: str = "",
        is_excluded: bool = False,
        excluded_ic_id: int = -1,
        plot_variant: Literal["power_spectrum", "topomap"] = "power_spectrum",
        variant_name: PreprocessedDataVariants = PreprocessedDataVariants.RAW_AFTER_ICA,
        custom_full_path: str = "",
        title="",
        fmax: int = 125,
    ):
        """
        Plot Raw dataseries object using MNE plotting functionalities from `compute_psd` base.

        :param data: Raw data to be plotted.
        :param save_fig: Whether save figure or not, if "" just show it and do not save.
        :param plot_variant: Which plot we want to create. Either "power_spectrum", or "topomap".
        :param is_excluded: Flag whether we are plotting excluded ICs.
        :param excluded_ic_id: ID of the excluded IC to plot (if we plot it).
        :param variant_name: Name of the data variant (plot will be stored in appropriate subdirectory).
        :param custom_full_path: In case one wants to store the plot in custom path.
        :param title: Title of the plot.
        :param fmax: Maximal frequency to include in plot, defaults to 125
        """
        show = save_fig == "" and custom_full_path == ""
        fig = None

        if plot_variant == "power_spectrum":
            # Power spectrum plotting.after_ica
            spectrum = data.compute_psd(fmax=fmax)
            fig = spectrum.plot(
                average=True, picks="data", exclude="bads", amplitude=False, show=show
            )
        elif plot_variant == "topomap":
            # Topomap plotting.
            if is_excluded:
                fig = (
                    data.plot_components(excluded_ic_id),
                )  # data.exclude[excluded_ic_id])
                if isinstance(fig, tuple):
                    fig = fig[0]
                fig.suptitle(title)
            else:
                fig = DatasetPlotter.plot_topomap_combined(data, title=title)
        else:
            return

        if not show:
            # Save the plot.
            plot_path = DatasetPlotter.get_plot_path(
                save_fig, plot_variant, variant_name.value, custom_full_path
            )
            # If the path does not exist. Create the parents.
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)

    @staticmethod
    def compute_tag_signal_correlation_matrix(signals: list[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise correlations between aligned signals and return the correlation matrix.
        :param signals: List of aligned TAG signals to compare.
        :return: Correlation matrix for all pairs of signals.
        """
        n = len(signals)

        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    # Simple Pearson correlation — signals are already aligned and same length
                    corr = np.corrcoef(signals[i], signals[j])[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        return corr_matrix

    @staticmethod
    def print_correlation_statistics(signals: list[np.ndarray]):
        """
        Print statistics about the correlation matrix,
        such as mean, median, and distribution of correlations.
        :param signals: List of aligned TAG signals to compute correlation statistics for.
        """
        corr_matrix = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        n = corr_matrix.shape[0]
        upper = corr_matrix[
            np.triu_indices(n, k=1)
        ]  # Get upper triangle without diagonal

        print(f"Mean pairwise correlation: {upper.mean():.4f}")
        print(f"Median pairwise correlation: {np.median(upper):.4f}")
        print(f"Min pairwise correlation: {upper.min():.4f}")
        print(f"Max pairwise correlation: {upper.max():.4f}")

    @staticmethod
    def plot_alignment_correlation_heatmap(
        signals: list[np.ndarray], save_fig: str = ""
    ):
        """
        Compute pairwise correlations between aligned signals and plot as heatmap.
        :param signals: List of aligned TAG signals to compare.
        :param save_fig: Whether save figure or not (to provided_path), if "" just show it and do not save.
        """

        corr_matrix = DatasetPlotter.compute_tag_signal_correlation_matrix(signals)
        n = corr_matrix.shape[0]
        upper = corr_matrix[
            np.triu_indices(n, k=1)
        ]  # Get upper triangle without diagonal

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Full correlation matrix
        sns.heatmap(
            corr_matrix,
            ax=axes[0],
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            center=0,
            cmap="RdBu_r",
        )
        axes[0].set_title("Pairwise Correlation Matrix")

        # Distribution of pairwise correlations
        axes[1].hist(upper, bins=20, edgecolor="black")
        axes[1].axvline(
            upper.mean(), color="red", linestyle="--", label=f"Mean: {upper.mean():.3f}"
        )
        axes[1].axvline(
            np.median(upper),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(upper):.3f}",
        )
        axes[1].set_xlabel("Pearson Correlation")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Pairwise Correlations")
        axes[1].legend()

        plt.tight_layout()
        if save_fig:
            fig.savefig(save_fig)
        else:
            plt.show()

    @staticmethod
    def plot_signal_overlap(
        signals: list[np.ndarray],
        sfreq: float,
        t_start: float = 0,
        time_duration: float = 10,
        save_fig: str = "",
    ):
        """
        Plot all aligned signals overlaid on the same axis.
        :param signals: List of aligned TAG signals to plot.
        :param sfreq: Sampling frequency of the signals (to convert time to samples).
        :param t_start: Start time in seconds for the plot.
        :param time_duration: Duration in seconds to plot from the start time.
        :param save_fig: Whether save figure or not (to provided_path), if "" just show it and do not save.
        """

        labels = [f"Signal {i}" for i in range(len(signals))]

        time = np.arange(signals[0].shape[0]) / sfreq

        t_end = t_start + time_duration  # seconds

        sample_start = int(t_start * sfreq)
        sample_end = int(t_end * sfreq)

        time = np.arange(sample_start, sample_end) / sfreq

        fig, ax = plt.subplots(figsize=(14, 4))

        for sig, label in zip(signals, labels):
            ax.plot(
                time,
                sig[sample_start:sample_end],
                alpha=0.6,
                linewidth=0.8,
                label=label,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Aligned Signal Overlap")

        plt.tight_layout()
        if save_fig:
            fig.savefig(save_fig)
        else:
            plt.show()

    @staticmethod
    def plot_crosscorr_vs_shift(
        s1,
        s2,
        sfreq,
        max_lag_sec=5.0,
        label1="Signal 1",
        label2="Signal 2",
        save_fig: str = "",
    ):
        """
        Plot normalized cross-correlation as a function of lag.
        :param s1: First signal (e.g. reference).
        :param s2: Second signal to compare.
        :param sfreq: Sampling frequency of the signals (to convert lag to seconds).
        :param max_lag_sec: Maximum lag in seconds to display on the plot.
        :param label1: Label for the first signal (for legend).
        :param label2: Label for the second signal (for legend).
        :param save_fig: Whether save figure or not (to provided_path), if "" just show it and do not save.
        """
        s1_norm = (s1 - s1.mean()) / s1.std()
        s2_norm = (s2 - s2.mean()) / s2.std()

        corr = correlate(s1_norm, s2_norm, mode="full")
        lags = correlation_lags(len(s1_norm), len(s2_norm), mode="full")

        # Overlap at each lag for normalization
        overlap = np.array(
            [
                min(i + 1, len(s1), len(s2), len(s1) + len(s2) - 1 - i)
                for i in range(len(corr))
            ]
        )
        corr_normalized = corr / overlap

        # Restrict to ±max_lag_sec
        max_lag_samples = int(max_lag_sec * sfreq)
        valid = np.abs(lags) <= max_lag_samples
        lags_sec = lags[valid] / sfreq
        corr_valid = corr_normalized[valid]

        best_idx = np.argmax(corr_valid)
        best_lag = lags_sec[best_idx]
        best_corr = corr_valid[best_idx]

        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(lags_sec, corr_valid, linewidth=0.9, color="steelblue")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Zero lag")
        ax.axvline(
            best_lag,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Best lag: {best_lag:.3f} s (corr={best_corr:.4f})",
        )

        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized Correlation")
        ax.set_title(f"Cross-Correlation: {label1} vs {label2}")
        ax.legend()

        plt.tight_layout()
        if save_fig:
            fig.savefig(save_fig)
        else:
            plt.show()
