"""
Tests for src/preprocessing/time_alignment.py — TAGObject and TimeAligner.
"""

import pytest
import numpy as np

from src.preprocessing.time_alignment import TAGObject, TimeAligner


class TestTAGObject:
    """Test TAGObject container."""

    def test_attributes(self):
        signal = np.array([1.0, 2.0, 3.0])
        tag = TAGObject("file1.edf", signal)
        assert tag.filename == "file1.edf"
        np.testing.assert_array_equal(tag.tag_signal, signal)


class TestTimeAligner:
    """Test TimeAligner cross-correlation alignment."""

    @pytest.fixture
    def aligned_tags(self):
        """Create perfectly aligned identical signals."""
        rng = np.random.default_rng(42)
        base_signal = rng.normal(size=500)
        tags = [TAGObject(f"file{i}.edf", base_signal.copy()) for i in range(3)]
        return tags

    @pytest.fixture
    def shifted_tags(self):
        """Create signals with known shifts from the same underlying source.

        Uses a pulse-like signal so cross-correlation detects exact shifts.
        """
        rng = np.random.default_rng(42)
        # Create a base signal with a distinctive spike pattern
        base = np.zeros(3000)
        # Add a unique spike sequence
        base[500] = 10.0
        base[700] = -8.0
        base[1000] = 5.0
        base[1500] = -12.0
        # Add some smooth variation
        t = np.arange(3000)
        base += np.sin(t / 50.0) + rng.normal(0, 0.01, 3000)

        # Signals taken from different offsets, all 1000 samples
        tags = [
            TAGObject("file0.edf", base[100:1100].copy()),
            TAGObject("file1.edf", base[110:1110].copy()),
            TAGObject("file2.edf", base[120:1120].copy()),
        ]
        return tags

    def test_identical_signals_zero_shifts(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        # With identical signals, all shifts should be 0
        np.testing.assert_array_equal(aligner.shifts, [0, 0, 0])

    def test_correlation_matrix_shape(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        assert aligner.corr_matrix.shape == (3, 3)
        assert aligner.lag_matrix.shape == (3, 3)

    def test_correlation_matrix_symmetric(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        np.testing.assert_allclose(
            aligner.corr_matrix, aligner.corr_matrix.T, atol=1e-10
        )

    def test_crop_to_overlap_equal_lengths(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        cropped = aligner.crop_to_overlap()
        lengths = [len(s) for s in cropped]
        assert len(set(lengths)) == 1  # all same length

    def test_shifted_signals_detected(self, shifted_tags):
        """Shifted signals should produce correct shift detection."""
        aligner = TimeAligner(shifted_tags, sfreq=250.0)
        # Shifts should be detected: signal 0 is -10, signal 1 is 0, signal 2 is +10
        # (relative to the reference, which is signal 1)
        assert aligner.reference_idx == 1
        np.testing.assert_array_equal(aligner.shifts, [-10, 0, 10])

    def test_get_crop_indices_for_signal(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        crop_start, crop_end = aligner.get_crop_indices_for_signal(shift=0)
        assert crop_start >= 0
        assert crop_end > crop_start

    def test_start_before_end(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        assert aligner.start < aligner.end

    def test_reference_idx_valid(self, aligned_tags):
        aligner = TimeAligner(aligned_tags, sfreq=250.0)
        assert 0 <= aligner.reference_idx < len(aligned_tags)
