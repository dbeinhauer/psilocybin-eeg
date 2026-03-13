"""
Tests for src/definitions/constants.py — ProjectPaths.
"""

import pytest
from pathlib import Path

from src.definitions.constants import ProjectPaths
from src.definitions.fields import ExperimentNames, CoordinateSystems


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
