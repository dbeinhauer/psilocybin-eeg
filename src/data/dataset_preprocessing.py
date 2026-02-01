"""
This module contains functionalities for preprocessing EEG datasets.
"""

import re
from pathlib import Path

import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from autoreject import Ransac


from src.definitions.fields import ChannelTypes, ICLabelComponentsClasses
from src.definitions.mappings import (
    RAW_CHANNEL_NAMES,
    MONTAGE_CHANNEL_NAMES,
    CHANNEL_TYPES_TO_MNE_TYPES_MAPPING,
)
from src.utils.logging_config import LoggerMixin


class DatasetPreprocessor(LoggerMixin):

    def __init__(self, coordinates_file_path: Path):
        self.montage = DatasetPreprocessor._load_coordinates_file(coordinates_file_path)

    @staticmethod
    def _load_coordinates_file(coordinates_file_path: Path) -> mne.channels.DigMontage:
        """
        Loads coordinates (montage) from the provided SFP file.

        :param coordinates_file_path: Path to the coordinates file.
        :return: Returns the montage object loaded from the provided file.
        """
        return mne.channels.read_custom_montage(coordinates_file_path)

    def _rename_channels(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Rename raw data channels to match the montage file naming conventions.

        :param data: MNE-Python Raw object containing the EEG data.
        :return: MNE-Python Raw object with renamed channels to match the montage naming conventions.
        """

        self.logger.info("Renaming channels to match montage file...")
        channel_name_mapping = {}
        for ch_name in data.ch_names:
            # Handle "EEG VREF" -> "Cz"
            if ch_name == RAW_CHANNEL_NAMES[ChannelTypes.EEG_REF]:
                channel_name_mapping[ch_name] = MONTAGE_CHANNEL_NAMES[
                    ChannelTypes.EEG_REF
                ]
                # Handle "EEG {num}" -> "E{num}"
            elif ch_name.startswith(RAW_CHANNEL_NAMES[ChannelTypes.EEG]):
                # Extract numbers using regex (handles multiple spaces, etc.)
                match = re.search(
                    rf"{RAW_CHANNEL_NAMES[ChannelTypes.EEG]}\s+(\d+)", ch_name
                )
                if match:
                    num = match.group(1)
                    channel_name_mapping[ch_name] = (
                        f"{MONTAGE_CHANNEL_NAMES[ChannelTypes.EEG]}{num}"
                    )

        # Rename channels to match montage:
        data.rename_channels(channel_name_mapping)
        self.logger.info("Channels renamed successfully.")
        return data

    def _set_channel_types(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Set channel types in the raw data according to predefined mappings.

        Note: This function assumes that the channel names have already been renamed to match the
        montage naming conventions e.g. by running function `_rename_channels`.

        :param data: MNE-Python Raw object containing the EEG data.
        :return: MNE-Python Raw object with updated channel types.
        """
        self.logger.info("Setting channel types...")
        channel_types_mapping = {
            ch: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.EEG]
            for ch in data.ch_names
            if ch in MONTAGE_CHANNEL_NAMES.values()
        }  # All channels in montage file are EEG channels (either normal or reference).
        channel_types_mapping.update(
            {
                RAW_CHANNEL_NAMES[ChannelTypes.ECG]: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[
                    ChannelTypes.ECG
                ],
                RAW_CHANNEL_NAMES[ChannelTypes.TAG]: CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[
                    ChannelTypes.TAG
                ],
            }
        )
        data.set_channel_types(channel_types_mapping)
        self.logger.info("Channel types set successfully.")

        return data

    def _add_coordinates_montage(
        self, data: mne.io.Raw, montage: mne.channels.DigMontage
    ) -> mne.io.Raw:
        """
        Add channel coordinates montage to the raw data.

        :param raw_data: MNE-Python Raw object containing the EEG data.
        :param montage: MNE-Python DigMontage object containing the channel coordinates.
        :return: MNE-Python Raw object with added montage.
        """
        self.logger.info("Adding channel coordinates montage...")
        data = self._set_channel_types(self._rename_channels(data)).set_montage(
            montage, on_missing="warn"
        )
        self.logger.info("Montage added successfully.")

        return data

    def _apply_filters_to_data(
        self,
        data: mne.io.Raw,
        notch_frequencies: tuple[int, int, int] = (50, 250, 50),
        filter_boundaries: tuple[float, float] = (1.0, 100.0),
    ) -> mne.io.Raw:
        """
        Apply notch filter to filter the line noise.

        :param data: Data to be filtered.
        :param notch_frequencies: Frequencies to filter by notch filter.
        :param filter_boundaries: Boundaries to aply FIR filter on.
        :return: Returns filtered data.
        """
        freqs = np.arange(*notch_frequencies)
        data = data.notch_filter(freqs=freqs)  # Filter 50 Hz line noise and harmonics
        return data.filter(
            l_freq=filter_boundaries[0], h_freq=filter_boundaries[1], method="fir"
        )

    def _apply_ransac_filter(
        self,
        data: mne.io.Raw,
        epoch_duration: float = 2.0,
        ransac_epochs: int = 100,
    ):
        """
        Applies Ransac bad channel detection and marks bad
        channels in the data itself.

        :param data: Data to be processes..
        :param epoch_duration: Duration of the time step (for discretization).
        :param ransac_epochs: Number of epochs in the Ransac processing.
        :return: Returns data labeled as good/bad channels based on the Ransac.
        """
        epochs = mne.make_fixed_length_epochs(
            data, duration=epoch_duration, preload=True
        )

        # Automatic detection of bad channels with RANSAC -> interpolation later
        ransac = Ransac(
            n_resample=ransac_epochs,
            min_channels=0.5,
            min_corr=0.7,
            unbroken_time=0.2,
            random_state=97,
        )
        ransac.fit(epochs)

        # ransac.bad_chs_ is a list of channel names it considers bad
        data.info["bads"] = list(set(data.info["bads"]).union(ransac.bad_chs_))

        return data

    def _interpolate_bad_channels(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Interpolates the channels that are marked as bad using average.

        :param data: Data to be interpolated (the bad channels needs to be alredy labeled).
        :return: Returns copy of the original data with interpolated bad channels.
        """
        data = data.copy().interpolate_bads(reset_bads=True)
        return data.set_eeg_reference("average", ch_type="eeg")

    @staticmethod
    def _get_ic_labeling_selected_probabilities(
        component_probabilities,
    ) -> dict[ICLabelComponentsClasses, np.ndarray]:
        """
        Select proper probability values for selected components from the
        ICLabel tool.

        Note: In our current implementation we select only brain, muscle, eye and heart.

        :param component_probabilities: Probabilites of all ICLabel Components.
        :return: Returns dictionary of key ICLabel component and its probabalities in np.ndarray form.
        """
        classes_order = [
            ICLabelComponentsClasses.BRAIN.value,
            ICLabelComponentsClasses.MUSCLE.value,
            ICLabelComponentsClasses.EYE.value,
            ICLabelComponentsClasses.HEART.value,
            ICLabelComponentsClasses.LINE.value,
            ICLabelComponentsClasses.CHANNEL.value,
            ICLabelComponentsClasses.OTHER.value,
        ]

        selected_classes = [
            ICLabelComponentsClasses.EYE,
            ICLabelComponentsClasses.MUSCLE,
            ICLabelComponentsClasses.HEART,
            ICLabelComponentsClasses.BRAIN,
        ]

        selected_idxs = {
            component: classes_order.index(component.value)
            for component in selected_classes
        }
        return {
            component: component_probabilities[:, selected_idxs[component]]
            for component in selected_idxs
        }

    def _mark_ic_for_exclusion(
        self,
        component_probabilities,
        ica: ICA,
        component_thresholds: dict[ICLabelComponentsClasses, float] = {
            ICLabelComponentsClasses.EYE: 0.40,
            ICLabelComponentsClasses.MUSCLE: 0.60,
            ICLabelComponentsClasses.HEART: 0.40,
            ICLabelComponentsClasses.BRAIN: 0.30,
        },
    ):
        self.logger.info("Getting probabilites of selected IC components.")
        selected_component_probabilites = (
            DatasetPreprocessor._get_ic_labeling_selected_probabilities(
                component_probabilities
            )
        )
        # --- Auto-exclusion rule (tune thresholds to taste) ---
        # Conservative defaults: remove clear artifacts, keep 'brain' and usually keep 'other'
        # ---- Tuning thresholds ----
        exclude = []

        self.logger.info(
            "Starting exclusion of the components passing selected threshold."
        )
        for i in range(len(component_probabilities)):
            # Exclude strong eye components
            if (
                selected_component_probabilites[ICLabelComponentsClasses.EYE][i]
                >= component_thresholds[ICLabelComponentsClasses.EYE]
                and selected_component_probabilites[ICLabelComponentsClasses.BRAIN][i]
                < component_thresholds[ICLabelComponentsClasses.BRAIN]
            ):
                exclude.append(i)

            # Exclude strong muscle components
            if (
                selected_component_probabilites[ICLabelComponentsClasses.MUSCLE][i]
                >= component_thresholds[ICLabelComponentsClasses.MUSCLE]
                and selected_component_probabilites[ICLabelComponentsClasses.BRAIN][i]
                < component_thresholds[ICLabelComponentsClasses.BRAIN]
            ):
                exclude.append(i)

            # Exclude heart artifacts if present
            if (
                selected_component_probabilites[ICLabelComponentsClasses.HEART][i]
                >= component_thresholds[ICLabelComponentsClasses.HEART]
                and selected_component_probabilites[ICLabelComponentsClasses.BRAIN][i]
                < component_thresholds[ICLabelComponentsClasses.BRAIN]
            ):
                exclude.append(i)

        ica.exclude = sorted(set(exclude))

        self.logger.info("All problematic components excluded.")
        self.logger.info(f"ICs to exclude: {ica.exclude}")

        return ica

    def _apply_ica_component_filtering(
        self, filtered_data: mne.io.Raw, interpolated_data: mne.io.Raw
    ) -> mne.io.Raw:

        self.logger.info("Starting ICA decomposition.")
        # Run ICA on the interpolated data.
        ica = ICA(
            n_components=0.99,  # or an int
            method="infomax",  # extended infomax recommended :contentReference[oaicite:4]{index=4}
            fit_params=dict(extended=True),
            random_state=97,
            max_iter="auto",
        )
        self.logger.info("Start ICA fitting.")
        ica.fit(interpolated_data, reject_by_annotation=True)
        self.logger.info("ICA fitting finished.")

        # IC labeling (select each component probabilities).
        self.logger.info("Start computing ICLabel components.")
        component_probabilities = iclabel_label_components(interpolated_data, ica)
        self.logger.info("ICLabel component probabilites found.")
        # Based on the IC labeling select components for exclusion
        ica = self._mark_ic_for_exclusion(
            component_probabilities,
            ica,
        )

        # Apply ICA to the ORIGINAL raw (often you fit on filtered, apply to unfiltered)
        # TODO: Not sure if use raw_data or raw_interp
        return ica.apply(filtered_data.copy())

    def preprocess_one_raw_data(self, data: mne.io.Raw) -> mne.io.Raw:
        # Apply all the filters and label the bad channels.
        data = self._apply_ransac_filter(
            self._apply_filters_to_data(
                self._add_coordinates_montage(data, self.montage)
            )
        )
        # Interpolate the bad channels.
        interpolated_data = self._interpolate_bad_channels(data)
