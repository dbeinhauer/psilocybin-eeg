from pathlib import Path

import mne

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

    @staticmethod
    def plot_raw_dataseries(
        data: mne.io.Raw,
        save_fig: str = "",
        plot_variant: str = "power_spectrum",
        variant_name: PreprocessedDataVariants = PreprocessedDataVariants.RAW_BEFORE_ICA,
        custom_full_path: str = "",
        fmax: int = 125,
    ):
        """
        Plot Raw dataseries object using MNE plotting functionalities from `compute_psd` base.

        :param data: Raw data to be plotted.
        :param save_fig: Whether save figure or not, if "" just show it and do not save.
        :param plot_variant: Which plot we want to create. Either "power_spectrum", or "topomap".
        :param variant_name: Name of the data variant (plot will be stored in appropriate subdirectory).
        :param custom_full_path: In case one wants to store the plot in custom path.
        :param fmax: Maximal frequency to include in plot, defaults to 125
        """
        show = save_fig == "" and custom_full_path == ""
        spectrum = data.compute_psd(fmax=fmax)
        fig = None

        if plot_variant == "power_spectrum":
            # Power spectrum plotting.
            fig = spectrum.plot(
                average=True, picks="data", exclude="bads", amplitude=False, show=show
            )
        elif plot_variant == "topomap":
            # Topomap plotting.
            fig = spectrum.plot_topomap(normalize=False, show=show)
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
