"""
Tests for src/analysis/data_representations.py — AnalysisData container & adapters.
"""

import pytest
import numpy as np

from src.analysis.data_representations import (
    AnalysisData,
    DataRepresentation,
    from_array,
    to_analytic_amplitude,
    to_wavelet_phase,
    to_wavelet_power,
)

class TestDataRepresentation:
    """Test DataRepresentation enum."""

    def test_time_domain(self):
        assert DataRepresentation.TIME_DOMAIN.value == "time_domain"

    def test_ica_activations(self):
        assert DataRepresentation.ICA_ACTIVATIONS.value == "ica_activations"

    def test_wavelet_power(self):
        assert DataRepresentation.WAVELET_POWER.value == "wavelet_power"

    def test_wavelet_phase(self):
        assert DataRepresentation.WAVELET_PHASE.value == "wavelet_phase"

    def test_mean_response(self):
        assert DataRepresentation.MEAN_RESPONSE.value == "mean_response"


class TestAnalysisData:
    """Test AnalysisData dataclass."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        return AnalysisData(
            data=rng.normal(size=(5, 3, 100)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="Test data",
            feature_names=["ch1", "ch2", "ch3"],
        )

    def test_shape_properties(self, sample_data):
        assert sample_data.n_items == 5
        assert sample_data.n_features == 3
        assert sample_data.n_samples == 100

    def test_feature_axis_label(self, sample_data):
        assert sample_data.feature_axis_label == "Channel"

    def test_item_axis_label(self, sample_data):
        assert sample_data.item_axis_label == "Subject"

    def test_value_label(self, sample_data):
        assert sample_data.value_label == "Amplitude (µV)"

    def test_normalize_returns_new_object(self, sample_data):
        normalized = sample_data.normalize()
        assert normalized is not sample_data
        assert normalized.data.shape == sample_data.data.shape
        # Z-scored data should have mean ≈ 0, std ≈ 1 along axis=2
        np.testing.assert_allclose(normalized.data.mean(axis=2), 0.0, atol=1e-10)

    def test_normalize_inplace(self, sample_data):
        original_shape = sample_data.data.shape
        sample_data.normalize_inplace()
        assert sample_data.data.shape == original_shape
        np.testing.assert_allclose(sample_data.data.mean(axis=2), 0.0, atol=1e-10)

    def test_copy_is_independent(self, sample_data):
        copied = sample_data.copy()
        copied.data[0, 0, 0] = 999.0
        assert sample_data.data[0, 0, 0] != 999.0

    def test_repr_string(self, sample_data):
        r = repr(sample_data)
        assert "Test data" in r
        assert "time_domain" in r
        assert "5 items" in r

    def test_ica_activations_labels(self):
        data = AnalysisData(
            data=np.zeros((4, 2, 50)),
            sfreq=250.0,
            representation=DataRepresentation.ICA_ACTIVATIONS,
            label="ICA",
        )
        assert data.feature_axis_label == "IC component"
        assert data.item_axis_label == "Subject"

    def test_mean_response_labels(self):
        data = AnalysisData(
            data=np.zeros((2, 3, 50)),
            sfreq=250.0,
            representation=DataRepresentation.MEAN_RESPONSE,
            label="Mean",
        )
        assert data.item_axis_label == "Group"


class TestFromArray:
    """Test from_array factory function."""

    def test_valid_3d_array(self):
        arr = np.zeros((3, 4, 100))
        result = from_array(arr, sfreq=250.0, label="test")
        assert isinstance(result, AnalysisData)
        assert result.n_items == 3
        assert result.sfreq == 250.0

    def test_invalid_2d_array_raises(self):
        arr = np.zeros((3, 100))
        with pytest.raises(ValueError, match="Expected 3D"):
            from_array(arr, sfreq=250.0)

    def test_default_representation(self):
        arr = np.zeros((2, 2, 50))
        result = from_array(arr, sfreq=100.0)
        assert result.representation == DataRepresentation.TIME_DOMAIN


class TestToAnalyticAmplitude:
    """Test analytic (Hilbert) amplitude adapter."""

    def test_output_representation(self):
        rng = np.random.default_rng(42)
        ad = AnalysisData(
            data=rng.normal(size=(3, 2, 100)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="raw",
        )
        result = to_analytic_amplitude(ad)
        assert result.representation == DataRepresentation.WAVELET_AMPLITUDE

    def test_amplitude_is_non_negative(self):
        rng = np.random.default_rng(43)
        ad = AnalysisData(
            data=rng.normal(size=(2, 2, 200)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="raw",
        )
        result = to_analytic_amplitude(ad)
        assert np.all(result.data >= 0)

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(44)
        ad = AnalysisData(
            data=rng.normal(size=(3, 4, 150)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="raw",
        )
        result = to_analytic_amplitude(ad)
        assert result.data.shape == ad.data.shape


class TestToWaveletPower:
    """Test Morlet-wavelet power adapter."""

    @pytest.fixture
    def sample_ad(self):
        rng = np.random.default_rng(60)
        return AnalysisData(
            data=rng.normal(size=(3, 4, 500)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="raw",
        )

    @pytest.fixture
    def freqs(self):
        return np.linspace(4.0, 30.0, 5)

    def test_output_representation(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs)
        assert result.representation == DataRepresentation.WAVELET_POWER

    def test_output_shape_preserved(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs)
        assert result.data.shape == sample_ad.data.shape

    def test_power_is_non_negative(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs)
        assert np.all(result.data >= 0)

    def test_label_contains_freq_range(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs)
        assert "wavelet power" in result.label

    def test_metadata_contains_freqs(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs)
        assert "freqs" in result.metadata
        np.testing.assert_array_equal(result.metadata["freqs"], freqs)

    def test_keep_frequency_dim_preserves_frequency_axis(self, sample_ad, freqs):
        result = to_wavelet_power(sample_ad, freqs, keep_frequency_dim=True)
        assert result.data.shape == (
            sample_ad.n_items,
            sample_ad.n_features * len(freqs),
            sample_ad.n_samples,
        )
        assert result.metadata["keep_frequency_dim"]


class TestToWaveletPhase:
    """Test Morlet-wavelet phase adapter."""

    @pytest.fixture
    def sample_ad(self):
        rng = np.random.default_rng(70)
        return AnalysisData(
            data=rng.normal(size=(3, 4, 500)),
            sfreq=250.0,
            representation=DataRepresentation.TIME_DOMAIN,
            label="raw",
        )

    @pytest.fixture
    def freqs(self):
        return np.linspace(4.0, 30.0, 5)

    def test_output_representation(self, sample_ad, freqs):
        result = to_wavelet_phase(sample_ad, freqs)
        assert result.representation == DataRepresentation.WAVELET_PHASE

    def test_output_shape_preserved(self, sample_ad, freqs):
        result = to_wavelet_phase(sample_ad, freqs)
        assert result.data.shape == sample_ad.data.shape

    def test_phase_range(self, sample_ad, freqs):
        result = to_wavelet_phase(sample_ad, freqs)
        assert np.all(result.data >= -np.pi)
        assert np.all(result.data <= np.pi)

    def test_label_contains_freq_range(self, sample_ad, freqs):
        result = to_wavelet_phase(sample_ad, freqs)
        assert "wavelet phase" in result.label

    def test_keep_frequency_dim_preserves_frequency_axis(self, sample_ad, freqs):
        result = to_wavelet_phase(sample_ad, freqs, keep_frequency_dim=True)
        assert result.data.shape == (
            sample_ad.n_items,
            sample_ad.n_features * len(freqs),
            sample_ad.n_samples,
        )
        assert result.metadata["keep_frequency_dim"]
