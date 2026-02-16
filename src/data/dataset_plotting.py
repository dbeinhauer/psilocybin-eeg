from pathlib import Path
from typing import Literal

import mne

import matplotlib.pyplot as plt
import numpy as np
from mne.viz import plot_topomap

from src.definitions.constants import ProjectPaths
from src.definitions.fields import PreprocessedDataVariants, RAW_DATA_VARIANTS
from src.utils.logging_config import LoggerMixin


class DatasetPlotter(LoggerMixin):

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
