"""
Tests for src/preprocessing/filtering.py — Signal filtering functions.

Tests the functions that do not require actual RANSAC or AutoReject,
which need real EEG-like data.
"""

import pytest
import numpy as np
import mne

from src.preprocessing.filtering import (
    remove_bad_epoch_annotations,
    interpolate_bad_channels,
)


@pytest.fixture
def raw_with_annotations():
    """Create a Raw object with annotations including BAD_epoch."""
    sfreq = 250.0
    n_samples = 10000
    rng = np.random.default_rng(42)
    data = rng.normal(size=(3, n_samples)) * 1e-6
    info = mne.create_info(ch_names=["E1", "E2", "E3"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Add annotations
    annotations = mne.Annotations(
        onset=[0.0, 2.0, 4.0, 6.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        description=["BAD_epoch", "stimulus", "BAD_epoch", "response"],
    )
    raw.set_annotations(annotations)
    return raw


class TestRemoveBadEpochAnnotations:
    def test_bad_epoch_annotations_removed(self, raw_with_annotations):
        result = remove_bad_epoch_annotations(raw_with_annotations)
        descriptions = list(result.annotations.description)
        assert "BAD_epoch" not in descriptions

    def test_other_annotations_kept(self, raw_with_annotations):
        result = remove_bad_epoch_annotations(raw_with_annotations)
        descriptions = list(result.annotations.description)
        assert "stimulus" in descriptions
        assert "response" in descriptions

    def test_annotation_count(self, raw_with_annotations):
        result = remove_bad_epoch_annotations(raw_with_annotations)
        # 2 out of 4 annotations are BAD_epoch
        assert len(result.annotations) == 2


class TestInterpolateBadChannels:
    def test_bad_channels_interpolated(self):
        sfreq = 250.0
        n_samples = 1000
        rng = np.random.default_rng(42)
        n_channels = 10
        ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
        data = rng.normal(size=(n_channels, n_samples)) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Create a minimal montage so interpolation works
        montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
        # Rename channels to match the montage
        mapping = {f"E{i}": f"E{i}" for i in range(1, n_channels + 1)}
        raw.set_montage(montage, on_missing="ignore")

        # Mark channel as bad
        raw.info["bads"] = ["E1"]

        result = interpolate_bad_channels(raw)
        assert len(result.info["bads"]) == 0  # bads reset after interpolation
