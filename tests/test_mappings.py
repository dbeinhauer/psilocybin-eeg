"""
Tests for src/definitions/mappings.py — Channel name and type mappings.
"""

import pytest

from src.definitions.fields import ChannelTypes
from src.definitions.mappings import (
    RAW_CHANNEL_NAMES,
    MONTAGE_CHANNEL_NAMES,
    CHANNEL_TYPES_TO_MNE_TYPES_MAPPING,
)


class TestRawChannelNames:
    def test_eeg_prefix(self):
        assert RAW_CHANNEL_NAMES[ChannelTypes.EEG] == "EEG"

    def test_eeg_ref(self):
        assert RAW_CHANNEL_NAMES[ChannelTypes.EEG_REF] == "EEG VREF"

    def test_ecg(self):
        assert RAW_CHANNEL_NAMES[ChannelTypes.ECG] == "ECG"

    def test_tag(self):
        assert RAW_CHANNEL_NAMES[ChannelTypes.TAG] == "TAG"


class TestMontageChannelNames:
    def test_eeg_prefix(self):
        assert MONTAGE_CHANNEL_NAMES[ChannelTypes.EEG] == "E"

    def test_eeg_ref(self):
        assert MONTAGE_CHANNEL_NAMES[ChannelTypes.EEG_REF] == "Cz"


class TestChannelTypesToMneTypesMapping:
    def test_eeg_maps_to_eeg(self):
        assert CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.EEG] == "eeg"

    def test_eeg_ref_maps_to_eeg(self):
        assert CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.EEG_REF] == "eeg"

    def test_ecg_maps_to_ecg(self):
        assert CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.ECG] == "ecg"

    def test_tag_maps_to_stim(self):
        assert CHANNEL_TYPES_TO_MNE_TYPES_MAPPING[ChannelTypes.TAG] == "stim"
