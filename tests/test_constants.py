"""
Tests for src/definitions/constants.py — ProjectPaths and AssrEpoch.
"""

import numpy as np
import pytest
from pathlib import Path

from src.definitions.constants import AssrEpoch, ProjectPaths
from src.definitions.fields import (
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
)

SFREQ = 250.0
ASSR_MIN_GAP = 313  # shortest observed inter-onset gap in the ASSR dataset


class TestProjectPaths:
    """Test that all path constants are correctly defined."""

    def test_project_root_is_directory(self):
        assert ProjectPaths.PROJECT_ROOT.is_dir()

    def test_config_dir(self):
        assert ProjectPaths.CONFIG_DIR == ProjectPaths.PROJECT_ROOT / "config"

    def test_data_dir(self):
        assert ProjectPaths.DATA_DIR == ProjectPaths.PROJECT_ROOT / "data"

    def test_coordinates_dir(self):
        assert ProjectPaths.COORDINATES_DIR == ProjectPaths.CONFIG_DIR / "coordinates"

    def test_raw_data_dir(self):
        assert ProjectPaths.RAW_DATA_DIR == ProjectPaths.DATA_DIR / "raw"

    def test_interim_data_dir(self):
        assert ProjectPaths.INTERIM_DATA_DIR == ProjectPaths.DATA_DIR / "interim"

    def test_processed_data_dir(self):
        assert ProjectPaths.PROCESSED_DATA_DIR == ProjectPaths.DATA_DIR / "processed"

    def test_participant_mapping_dir(self):
        assert (
            ProjectPaths.PARTICIPANT_MAPPING_DIR
            == ProjectPaths.CONFIG_DIR / "participant_mappings"
        )

    def test_excluded_electrodes_dir(self):
        assert (
            ProjectPaths.EXCLUDED_ELECTRODES_DIR
            == ProjectPaths.CONFIG_DIR / "excluded_electrodes"
        )


class TestGetExperimentDataDir:
    """Test get_experiment_data_dir method."""

    def test_processed_dir(self):
        data_dir, mapping_path = ProjectPaths.get_experiment_data_dir(
            ExperimentNames.PSILO_MUSIC, is_processed=True
        )
        assert data_dir == ProjectPaths.PROCESSED_DATA_DIR / "psilo_music"
        assert mapping_path == (
            ProjectPaths.PARTICIPANT_MAPPING_DIR / "psilo_music.csv"
        )

    def test_raw_dir(self):
        data_dir, mapping_path = ProjectPaths.get_experiment_data_dir(
            ExperimentNames.PSILO_MUSIC, is_processed=False
        )
        assert data_dir == ProjectPaths.RAW_DATA_DIR / "psilo_music"
        assert mapping_path == (
            ProjectPaths.PARTICIPANT_MAPPING_DIR / "psilo_music.csv"
        )


class TestGetExperimentInterimDir:
    """Test get_experiment_interim_dir method."""

    def test_interim_dir(self):
        result = ProjectPaths.get_experiment_interim_dir(ExperimentNames.PSILO_MUSIC)
        assert result == ProjectPaths.INTERIM_DATA_DIR / "psilo_music"


class TestGetIvaResultsDir:
    """Test get_iva_results_dir method."""

    def test_store_root_without_a_condition(self):
        result = ProjectPaths.get_iva_results_dir(ExperimentNames.PSILO_MUSIC)
        assert result == ProjectPaths.PROCESSED_DATA_DIR / "psilo_music" / "iva_results"

    def test_the_condition_is_the_subdirectory(self):
        result = ProjectPaths.get_iva_results_dir(
            ExperimentNames.ASSR, ConditionVariants.JOINED_TRACKS
        )
        assert result == (
            ProjectPaths.PROCESSED_DATA_DIR / "assr" / "iva_results" / "JoinedTracks"
        )

    def test_a_custom_root_redirects_the_whole_store(self, tmp_path):
        result = ProjectPaths.get_iva_results_dir(
            ExperimentNames.ASSR,
            ConditionVariants.PLACEBO,
            processed_data_dir=tmp_path,
        )
        assert result == tmp_path / "assr" / "iva_results" / "Placebo"

    def test_nothing_is_created_by_asking_for_the_path(self, tmp_path):
        path = ProjectPaths.get_iva_results_dir(
            ExperimentNames.ASSR,
            ConditionVariants.PLACEBO,
            processed_data_dir=tmp_path,
        )
        assert not path.exists()


class TestGetCoordinatesFilePath:
    """Test get_coordinates_file_path method."""

    def test_hydrogel_257(self):
        coord_path, excl_path = ProjectPaths.get_coordinates_file_path(
            CoordinateSystems.HYDROGEL_257
        )
        assert coord_path == (ProjectPaths.COORDINATES_DIR / "GSN-HydroCel-257.sfp")
        assert excl_path == (
            ProjectPaths.EXCLUDED_ELECTRODES_DIR / "GSN-HydroCel-257.csv"
        )

    def test_hydrogel_257_no_fiducials(self):
        coord_path, excl_path = ProjectPaths.get_coordinates_file_path(
            CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
        )
        assert coord_path == (
            ProjectPaths.COORDINATES_DIR / "GSN-HydroCel-257_no-fiducials.sfp"
        )
        assert excl_path == (
            ProjectPaths.EXCLUDED_ELECTRODES_DIR / "GSN-HydroCel-257_no-fiducials.csv"
        )


def assr_epoch_times(sfreq=SFREQ, min_gap=ASSR_MIN_GAP):
    """Epoch time axis the ASSR analyses build from ``AssrEpoch``."""
    pre = AssrEpoch.pre_onset_samples(sfreq)
    post = AssrEpoch.post_onset_samples(sfreq, min_gap=min_gap)
    return np.arange(-pre, post) / sfreq


class TestAssrEpochTiming:
    """Paradigm timing values and their sample conversions."""

    def test_paradigm_timing_values(self):
        assert AssrEpoch.PRE_ONSET_S == 0.1
        assert AssrEpoch.STIMULUS_DURATION_S == 0.5
        assert AssrEpoch.POST_STIMULUS_S == 0.5
        # The fallback offset for recordings without a per-recording calibration:
        # the median of the 38 measured ASSR shifts.
        assert AssrEpoch.MARKER_ONSET_OFFSET_S == -0.4245

    def test_marker_offset_is_a_lag_shorter_than_the_train(self):
        # The `fam+` marker lags the acoustic onset, landing inside the train.
        # A value outside this range would mean the epoch windows above no longer
        # describe the paradigm and the offset was mis-signed or mis-scaled.
        assert -AssrEpoch.STIMULUS_DURATION_S < AssrEpoch.MARKER_ONSET_OFFSET_S < 0.0

    def test_post_onset_span_is_stimulus_plus_post_stimulus(self):
        assert AssrEpoch.POST_ONSET_S == pytest.approx(
            AssrEpoch.STIMULUS_DURATION_S + AssrEpoch.POST_STIMULUS_S
        )

    def test_sample_counts_at_the_project_sampling_rate(self):
        assert AssrEpoch.pre_onset_samples(SFREQ) == 25
        assert AssrEpoch.post_onset_samples(SFREQ) == 250

    @pytest.mark.parametrize("sfreq", [100.0, 250.0, 500.0, 1000.0])
    def test_sample_counts_scale_with_sampling_rate(self, sfreq):
        assert AssrEpoch.pre_onset_samples(sfreq) == round(
            AssrEpoch.PRE_ONSET_S * sfreq
        )
        assert AssrEpoch.post_onset_samples(sfreq) == round(
            AssrEpoch.POST_ONSET_S * sfreq
        )


class TestAssrEpochCapping:
    """The shortest inter-onset gap caps the window so epochs never overlap."""

    def test_full_epoch_fits_inside_the_assr_inter_onset_gap(self):
        pre = AssrEpoch.pre_onset_samples(SFREQ)
        post = AssrEpoch.post_onset_samples(SFREQ, min_gap=ASSR_MIN_GAP)
        assert pre + post <= ASSR_MIN_GAP

    def test_min_gap_caps_the_window(self):
        assert AssrEpoch.post_onset_samples(SFREQ, min_gap=100) == 100

    def test_min_gap_wider_than_paradigm_does_not_extend_the_window(self):
        assert AssrEpoch.post_onset_samples(
            SFREQ, min_gap=10_000
        ) == AssrEpoch.post_onset_samples(SFREQ)

    def test_post_onset_is_at_least_one_sample(self):
        assert AssrEpoch.post_onset_samples(SFREQ, min_gap=1) == 1

    @pytest.mark.parametrize("bad_gap", [0, -5])
    def test_non_positive_min_gap_is_rejected(self, bad_gap):
        with pytest.raises(ValueError):
            AssrEpoch.post_onset_samples(SFREQ, min_gap=bad_gap)


class TestAssrEpochStimulusMask:
    """The mask isolating the driven interval from the silent remainder."""

    def test_selects_only_the_driven_interval(self):
        times = assr_epoch_times()
        mask = AssrEpoch.stimulus_mask(times)
        assert times[mask].min() >= 0.0
        assert times[mask].max() < AssrEpoch.STIMULUS_DURATION_S
        assert mask.sum() == round(AssrEpoch.STIMULUS_DURATION_S * SFREQ)

    def test_excludes_baseline_and_post_stimulus(self):
        times = assr_epoch_times()
        mask = AssrEpoch.stimulus_mask(times)
        assert not mask[times < 0.0].any()
        assert not mask[times >= AssrEpoch.STIMULUS_DURATION_S].any()

    def test_is_half_the_post_onset_window(self):
        """Why the mask exists: the post-onset window is half silence."""
        times = assr_epoch_times()
        mask = AssrEpoch.stimulus_mask(times)
        assert mask.sum() == pytest.approx((times >= 0.0).sum() / 2, rel=0.02)

    def test_shape_and_dtype(self):
        times = assr_epoch_times()
        mask = AssrEpoch.stimulus_mask(times)
        assert mask.dtype == bool
        assert mask.shape == times.shape

    def test_does_not_mutate_input(self):
        times = assr_epoch_times()
        before = times.copy()
        AssrEpoch.stimulus_mask(times)
        np.testing.assert_array_equal(times, before)
