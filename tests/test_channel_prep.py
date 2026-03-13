"""
Tests for src/preprocessing/channel_prep.py — Channel renaming, type setting, and exclusion.
"""

import pytest
import numpy as np
import mne

from src.preprocessing.channel_prep import (
    rename_channels,
    set_channel_types,
    exclude_selected_channels,
    crop_start_and_end_of_dataseries,
)


@pytest.fixture
def raw_with_eeg_channels():
    """Create a minimal Raw object with channel names matching raw EDF format."""
    sfreq = 250.0
    n_samples = 25000  # 100 seconds to accommodate default 10+10 s crop
    # Simulate raw EDF channel names: "EEG  1", "EEG  2", "EEG VREF", "ECG", "TAG"
    ch_names = ["EEG  1", "EEG  2", "EEG  3", "EEG VREF", "ECG", "TAG"]
    n_channels = len(ch_names)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_channels, n_samples)) * 1e-6  # microvolts

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")
    raw = mne.io.RawArray(data, info)
    return raw


class TestRenameChannels:
    def test_eeg_channels_renamed(self, raw_with_eeg_channels):
        raw = rename_channels(raw_with_eeg_channels)
        assert "E1" in raw.ch_names
        assert "E2" in raw.ch_names
        assert "E3" in raw.ch_names
        assert "Cz" in raw.ch_names

    def test_non_eeg_channels_unchanged(self, raw_with_eeg_channels):
        raw = rename_channels(raw_with_eeg_channels)
        assert "ECG" in raw.ch_names
        assert "TAG" in raw.ch_names

    def test_original_names_gone(self, raw_with_eeg_channels):
        raw = rename_channels(raw_with_eeg_channels)
        assert "EEG  1" not in raw.ch_names
        assert "EEG VREF" not in raw.ch_names


class TestSetChannelTypes:
    def test_channel_types_set(self, raw_with_eeg_channels):
        raw = rename_channels(raw_with_eeg_channels)
        raw = set_channel_types(raw)
        # ECG channel should be 'ecg'
        idx = raw.ch_names.index("ECG")
        assert raw.get_channel_types()[idx] == "ecg"
        # TAG should be 'stim'
        idx = raw.ch_names.index("TAG")
        assert raw.get_channel_types()[idx] == "stim"


class TestExcludeSelectedChannels:
    def test_channels_dropped(self, raw_with_eeg_channels):
        raw = rename_channels(raw_with_eeg_channels)
        original_n = len(raw.ch_names)
        raw = exclude_selected_channels(raw, ["E1"])
        assert "E1" not in raw.ch_names
        assert len(raw.ch_names) == original_n - 1


class TestCropStartAndEnd:
    def test_crop_shortens_data(self, raw_with_eeg_channels):
        raw = raw_with_eeg_channels
        original_duration = raw.times[-1] - raw.times[0]
        cropped = crop_start_and_end_of_dataseries(
            raw, start_offset=1.0, end_offset=1.0
        )
        cropped_duration = cropped.times[-1] - cropped.times[0]
        assert cropped_duration < original_duration
        assert cropped_duration == pytest.approx(original_duration - 2.0, abs=0.01)

    def test_crop_default_offsets(self, raw_with_eeg_channels):
        raw = raw_with_eeg_channels
        original_duration = raw.times[-1] - raw.times[0]
        cropped = crop_start_and_end_of_dataseries(raw)
        # Default offsets are 10s each
        expected_duration = original_duration - 20.0
        assert cropped.times[-1] - cropped.times[0] == pytest.approx(
            expected_duration, abs=0.01
        )
