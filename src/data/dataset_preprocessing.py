"""
This module contains functionalities for preprocessing EEG datasets.
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from autoreject import Ransac, AutoReject


from src.definitions.fields import ChannelTypes, ICLabelComponentsClasses
from src.definitions.mappings import (
    RAW_CHANNEL_NAMES,
    MONTAGE_CHANNEL_NAMES,
    CHANNEL_TYPES_TO_MNE_TYPES_MAPPING,
)
from src.utils.logging_config import LoggerMixin


class DatasetPreprocessor(LoggerMixin):

    # Order of the classes in ICLabel tool.
    # Based on https://mne.tools/mne-icalabel/dev/generated/api/mne_icalabel.iclabel.iclabel_label_components.html#mne_icalabel.iclabel.iclabel_label_components
    ic_label_classes_order = [
        ICLabelComponentsClasses.BRAIN.value,
        ICLabelComponentsClasses.MUSCLE.value,
        ICLabelComponentsClasses.EYE.value,
        ICLabelComponentsClasses.HEART.value,
        ICLabelComponentsClasses.LINE.value,
        ICLabelComponentsClasses.CHANNEL.value,
        ICLabelComponentsClasses.OTHER.value,
    ]

    def __init__(self, coordinates_file_path: Path, excluded_coordinates_path: Path):
        self.montage = DatasetPreprocessor._load_coordinates_file(coordinates_file_path)
        # List of all electrodes that we want to exclude.
        self.electrodes_to_exclude: list[str] = (
            pd.read_csv(excluded_coordinates_path)["electrode_name"]
            .str.strip()
            .tolist()
        )

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
        Add channel coordinates montage to the raw data and rename channel based on the template.

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

    def _exclude_selected_channels(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Exclude the selected channels from the raw data.

        We want to exclude the selected electrodes with the problematic placement
        (the electrodes are typically located at the cheeks or other problematic
        positions in the head.)

        :param data: Data from which we want to exclude the electrodes.
        :return: Returns provided data without the selected electrodes.
        """
        self.logger.info("Dropping the selected channels (from problematic positions).")
        return data.drop_channels(self.electrodes_to_exclude)

    def _crop_start_and_end_of_dataseries(
        self, data: mne.io.Raw, start_offset: float = 10.0, end_offset: float = 10.0
    ) -> mne.io.Raw:
        """
        Crops start and end time blocks from the data (they are always noisy).

        :param data: Data to be cropped.
        :param start_offset: Start offset in seconds.
        :param end_offset: End offset in seconds.
        :return: Returns cropped raw data series.
        """
        self.logger.info(
            f"Cropping the first {start_offset} and last {end_offset} seconds from the dataseries."
        )
        start_time = data.times[0]
        end_time = data.times[-1]

        return data.crop(tmin=start_time + start_offset, tmax=end_time - end_offset)

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
        self.logger.info("Applying notch filter and FIR filter to remove line noise.")
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

        self.logger.info("Applying Ransac algorithm to detect bad channels.")
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
            n_jobs=-1,
        )
        ransac.fit(epochs)

        # ransac.bad_chs_ is a list of channel names it considers bad
        data.info["bads"] = list(set(data.info["bads"]).union(ransac.bad_chs_))

        return data

    def remove_bad_epoch_annotations(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Removes bad epoch annotations from the data.

        :param data: Data to be processed.
        :return: _description_
        """
        old_annotations = data.annotations
        keep_annotations = [
            i
            for i, desc in enumerate(old_annotations.description)
            if desc != "BAD_epoch"
        ]

        new_annotations = mne.Annotations(
            onset=old_annotations.onset[keep_annotations],
            duration=old_annotations.duration[keep_annotations],
            description=old_annotations.description[keep_annotations],
        )
        return data.set_annotations(new_annotations)

    def _detect_bad_epochs(self, data: mne.io.Raw, epoch_len: float = 2.0):
        """
        Runs Autoreject bad epochs detection and annotates putatively bad epochs in the raw data.

        :param data: Data to be analyzed.
        :param epoch_len: Length of the epochs.
        :return: Returns annotated data with bad epochs.
        """
        epochs = mne.make_fixed_length_epochs(data, duration=epoch_len, preload=True)

        # Bad epochs detection
        ar = AutoReject(n_jobs=-1, random_state=42, verbose=True)
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)

        bad_epoch_indices = np.where(reject_log.bad_epochs)[0]
        print(f"Found {len(bad_epoch_indices)} bad epochs out of {len(epochs)}")

        # This marks bad segments WITHOUT removing them
        bad_annotations = mne.Annotations(
            onset=[epochs.events[i, 0] / data.info["sfreq"] for i in bad_epoch_indices],
            duration=[epoch_len] * len(bad_epoch_indices),  # duration of each epoch
            description=["BAD_epoch"] * len(bad_epoch_indices),
        )

        # Add annotations to raw data
        return data.set_annotations(data.annotations + bad_annotations)

    def _data_preparation(self, data: mne.io.Raw) -> tuple[mne.io.Raw, mne.io.Raw]:
        """
        Do first preparation of the raw data.

        Namely it adds the electrode coordinates, excludes the electrodes
        that has problematic placement, crops first and last 10 seconds of
        the data series (typically noisy).

        :param data: Data to be processed.
        :return: Returns tuple of prepared data splitted on EEG and rest of channels.
        """
        prepared_data = self._crop_start_and_end_of_dataseries(
            self._exclude_selected_channels(
                self._add_coordinates_montage(data, self.montage)
            )
        )

        eeg_data = prepared_data.copy().pick_types(eeg=True)
        aux_data = prepared_data.copy().pick_types(eeg=False, ecg=True, stim=True)

        return eeg_data, aux_data

    def _filter_data(self, eeg_data: mne.io.Raw) -> mne.io.Raw:
        """
        Applies notch and FIR filters to remove the line noise, and
        applies Ransac algorithm to detect badchannels.

        :param eeg_data: EEG data to do the preprocessing on.
        :return: Returns filtered EEG data.
        """
        return self._apply_ransac_filter(self._apply_filters_to_data(eeg_data))

    def _interpolate_bad_channels(self, data: mne.io.Raw) -> mne.io.Raw:
        """
        Interpolates the channels that are marked as bad using average reference.

        :param data: Data to be interpolated (the bad channels needs to be alredy labeled).
        :return: Returns copy of the original data with interpolated bad channels.
        """
        self.logger.info("Interpolating bad channels using average reference.")
        interpolated_data = data.interpolate_bads(reset_bads=True)
        return interpolated_data.set_eeg_reference("average", ch_type="eeg")

    @staticmethod
    def get_ic_labeling_probabilities(
        component_probabilities,
    ) -> dict[ICLabelComponentsClasses, np.ndarray]:
        """
        Links each IC label to its probability.

        :param component_probabilities: Probabilities of all ICLabel Components.
        :return: Returns dictionary of key ICLabel component and its probabilities in np.ndarray form.
        """

        selected_idxs = {
            component: DatasetPreprocessor.ic_label_classes_order.index(component.value)
            for component in ICLabelComponentsClasses
        }
        return {
            component: component_probabilities[:, selected_idxs[component]]
            for component in selected_idxs
        }

    @staticmethod
    def _check_ic_component_probability(
        all_probabilities: dict[ICLabelComponentsClasses, np.ndarray],
        ic_idx: int,
        tested_component: ICLabelComponentsClasses,
        tested_threshold: float = 0.6,
        brain_threshold: float = 0.3,
    ) -> bool:
        return (
            all_probabilities[tested_component][ic_idx] >= tested_threshold
            and all_probabilities[ICLabelComponentsClasses.BRAIN][ic_idx]
            < brain_threshold
        )

    def _mark_ic_for_exclusion(
        self,
        component_probabilities,
        ica: ICA,
        brain_threshold=0.3,
        component_thresholds: dict[ICLabelComponentsClasses, float] = {
            ICLabelComponentsClasses.EYE: 0.40,
            ICLabelComponentsClasses.MUSCLE: 0.60,
            ICLabelComponentsClasses.HEART: 0.40,
            ICLabelComponentsClasses.CHANNEL: 0.5,
        },
    ):
        """
        Based on the provided predicted probabilities of the ICA classes, mark
        putative artifact components for exclusion.

        We want to exclude muscle, eye and heartbeat .

        :param component_probabilities: Probability of each ICLabel class for each IC.
        :param ica: ICA decomposition results.
        :param component_thresholds: Dictionary of applied threshold for selected subset of ICLabel
        classes that we want to use in our IC exclusion. NOTE: The threshold for 'brain` signal
        defines the lowest value it is considered to be partially brain signal (will be kept). For
        other classes if the thresholds is surpasses -> it belongs to this class.
        :return: Returns ICA decomposition with ICs marked for exclusion (if classified as artifact components).
        """
        self.logger.info("Getting probabilites of IC components.")
        labeled_probabilities = DatasetPreprocessor.get_ic_labeling_probabilities(
            component_probabilities
        )
        # --- Auto-exclusion rule (tune thresholds to taste) ---
        # Conservative defaults: remove clear artifacts, keep 'brain' and usually keep 'other'
        # ---- Tuning thresholds ----
        exclude = []

        self.logger.info(
            "Starting exclusion of the components passing selected threshold."
        )

        for ic_idx in range(len(component_probabilities)):
            for component, threshold in component_thresholds.items():
                if DatasetPreprocessor._check_ic_component_probability(
                    labeled_probabilities,
                    ic_idx,
                    component,
                    tested_threshold=threshold,
                    brain_threshold=brain_threshold,
                ):
                    exclude.append(ic_idx)

        ica.exclude = sorted(set(exclude))

        self.logger.info("All problematic components excluded.")
        self.logger.info(f"ICs to exclude: {ica.exclude}")

        return ica

    def initial_preprocessing_and_bad_channel_interpolation(
        self, data: mne.io.Raw
    ) -> tuple[mne.io.Raw, mne.io.Raw]:
        """
        Run initial preprocessing steps (electrode naming,
        time trimming, electrodes exclusion, line noise filtering, bad
        channel interpolation).

        :param data: Raw EEG data series from one measurement.
        :return: Tuple of preprocessed data with interpolated bad channels and rest of channels (other than EEG).
        """
        self.logger.info(
            "Starting initial data preprocessing and interpolation of the bad channels."
        )
        # Prepare the data for preprocessing.
        eeg_data, aux_data = self._data_preparation(data)
        # Filter, interpolate the bad channels by using average reference and annotate bad epochs.
        eeg_data = self._detect_bad_epochs(
            self._interpolate_bad_channels(self._filter_data(eeg_data))
        )
        return eeg_data, aux_data

    def apply_ica_component_filtering(
        self,
        interpolated_data: mne.io.Raw,
    ) -> tuple[mne.io.Raw, ICA, np.ndarray]:
        """
        Runs ICA and then executes ICLabel tool to predict probabilities
        of the several artifacts in each component, excludes the artifact
        components and returns the filtered signal.

        :param interpolated_data: Already preprocessed and interpolated data without artifacts.
        :return: Returns tuple of processed data series by applying ICLabeling, ICAs with
        marked artifact ICs and array of probabilities of each IC artifact class.
        """

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

        # Apply ICA to interpolated data (however, mne might be able to
        # work with the Raw data before interpolation).
        return ica.apply(interpolated_data), ica, component_probabilities
