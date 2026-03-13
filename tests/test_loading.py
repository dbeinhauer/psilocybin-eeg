"""
Tests for src/io/loading.py — Data loading and path resolution.
"""

import pytest
from pathlib import Path

from src.definitions.fields import PreprocessedDataVariants, INTERIM_DATA_VARIANTS
from src.io.loading import _resolve_data_dir, get_preprocessing_results_path


class TestResolveDataDir:
    """Test _resolve_data_dir helper for interim vs processed routing."""

    def setup_method(self):
        self.processed = Path("/data/processed/exp")
        self.interim = Path("/data/interim/exp")

    def test_interim_variant_goes_to_interim(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.RAW_BEFORE_ICA
        )
        assert result == self.interim

    def test_ica_components_goes_to_interim(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.ICA_COMPONENTS
        )
        assert result == self.interim

    def test_ic_probabilities_goes_to_interim(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.IC_PROBABILITIES
        )
        assert result == self.interim

    def test_after_ica_goes_to_processed(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.RAW_AFTER_ICA
        )
        assert result == self.processed

    def test_cropped_goes_to_processed(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.RAW_CROPPED
        )
        assert result == self.processed

    def test_concatenated_goes_to_processed(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.CONCATENATED
        )
        assert result == self.processed

    def test_none_interim_falls_back_to_processed(self):
        """When interim_data_dir is None, all variants go to processed."""
        for variant in INTERIM_DATA_VARIANTS:
            result = _resolve_data_dir(self.processed, None, variant)
            assert result == self.processed

    def test_excluded_ic_goes_to_processed(self):
        result = _resolve_data_dir(
            self.processed, self.interim, PreprocessedDataVariants.RAW_EXCLUDED_IC
        )
        assert result == self.processed


class TestGetPreprocessingResultsPath:
    """Test get_preprocessing_results_path path construction."""

    def setup_method(self):
        self.processed = Path("/data/processed/exp")
        self.interim = Path("/data/interim/exp")

    def test_raw_before_ica_fif_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.RAW_BEFORE_ICA,
            interim_data_dir=self.interim,
        )
        assert path == self.interim / "before_ica" / "PSI001.fif"

    def test_raw_after_ica_fif_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.RAW_AFTER_ICA,
        )
        assert path == self.processed / "after_ica" / "PSI001.fif"

    def test_ica_components_fif_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.ICA_COMPONENTS,
            interim_data_dir=self.interim,
        )
        assert path == self.interim / "ica_components" / "PSI001.fif"

    def test_ic_probabilities_npy_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.IC_PROBABILITIES,
            interim_data_dir=self.interim,
        )
        assert path == self.interim / "ic_probabilities" / "PSI001.npy"

    def test_cropped_fif_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.RAW_CROPPED,
        )
        assert path == self.processed / "cropped" / "PSI001.fif"

    def test_concatenated_no_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.CONCATENATED,
        )
        assert path == self.processed / "concatenated" / "PSI001"

    def test_excluded_ic_fif_suffix(self):
        path = get_preprocessing_results_path(
            self.processed,
            "PSI001",
            PreprocessedDataVariants.RAW_EXCLUDED_IC,
        )
        assert path == self.processed / "raw_excluded_ic" / "PSI001.fif"
