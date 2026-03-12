"""
Tests for src/preprocessing/ica.py — ICA utility functions (not the full ICA pipeline).
"""

import pytest
import numpy as np

from src.definitions.fields import ICLabelComponentsClasses
from src.preprocessing.ica import (
    IC_LABEL_CLASSES_ORDER,
    get_ic_labeling_probabilities,
    check_ic_component_probability,
    mark_ic_for_exclusion,
)


class TestICLabelClassesOrder:
    """Test IC_LABEL_CLASSES_ORDER constant."""

    def test_length(self):
        assert len(IC_LABEL_CLASSES_ORDER) == 7

    def test_brain_is_first(self):
        assert IC_LABEL_CLASSES_ORDER[0] == ICLabelComponentsClasses.BRAIN.value

    def test_all_classes_present(self):
        all_values = {c.value for c in ICLabelComponentsClasses}
        assert set(IC_LABEL_CLASSES_ORDER) == all_values


class TestGetICLabelingProbabilities:
    """Test get_ic_labeling_probabilities function."""

    def test_output_keys(self):
        # 5 ICs, 7 classes
        probs = np.random.default_rng(42).random((5, 7))
        result = get_ic_labeling_probabilities(probs)
        expected_keys = set(ICLabelComponentsClasses)
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self):
        n_ics = 10
        probs = np.random.default_rng(42).random((n_ics, 7))
        result = get_ic_labeling_probabilities(probs)
        for component, values in result.items():
            assert len(values) == n_ics

    def test_correct_indexing(self):
        """Each probability column should map to the right class."""
        # Create a matrix where each IC has a unique pattern
        probs = np.eye(7)  # 7 ICs, each one is "100%" one class
        result = get_ic_labeling_probabilities(probs)
        # The brain column is at index 0, so IC 0 should have brain=1.0
        assert result[ICLabelComponentsClasses.BRAIN][0] == 1.0
        # Muscle is at index 1
        assert result[ICLabelComponentsClasses.MUSCLE][1] == 1.0


class TestCheckICComponentProbability:
    """Test check_ic_component_probability threshold logic."""

    def setup_method(self):
        # 3 ICs: IC0 is brain, IC1 is eye artifact, IC2 is mixed
        self.probs = {
            ICLabelComponentsClasses.BRAIN: np.array([0.8, 0.1, 0.4]),
            ICLabelComponentsClasses.EYE: np.array([0.05, 0.7, 0.3]),
            ICLabelComponentsClasses.MUSCLE: np.array([0.05, 0.05, 0.1]),
            ICLabelComponentsClasses.HEART: np.array([0.05, 0.05, 0.1]),
            ICLabelComponentsClasses.LINE: np.array([0.02, 0.05, 0.05]),
            ICLabelComponentsClasses.CHANNEL: np.array([0.02, 0.03, 0.03]),
            ICLabelComponentsClasses.OTHER: np.array([0.01, 0.02, 0.02]),
        }

    def test_brain_component_not_excluded(self):
        """IC0 has high brain probability — should NOT be flagged."""
        result = check_ic_component_probability(
            self.probs, ic_idx=0, tested_component=ICLabelComponentsClasses.EYE
        )
        assert not result

    def test_eye_artifact_excluded(self):
        """IC1 has high eye, low brain — should be flagged."""
        result = check_ic_component_probability(
            self.probs,
            ic_idx=1,
            tested_component=ICLabelComponentsClasses.EYE,
            tested_threshold=0.6,
            brain_threshold=0.3,
        )
        assert result

    def test_mixed_component_not_excluded_when_above_brain_threshold(self):
        """IC2 has brain=0.4 which is >= brain_threshold=0.3 → keep."""
        result = check_ic_component_probability(
            self.probs,
            ic_idx=2,
            tested_component=ICLabelComponentsClasses.EYE,
            tested_threshold=0.2,
            brain_threshold=0.3,
        )
        assert not result

    def test_below_threshold_not_excluded(self):
        """Component below artifact threshold should not be excluded."""
        result = check_ic_component_probability(
            self.probs,
            ic_idx=1,
            tested_component=ICLabelComponentsClasses.MUSCLE,
            tested_threshold=0.6,
        )
        assert not result


class TestMarkICForExclusion:
    """Test mark_ic_for_exclusion function with a mock ICA."""

    def test_marks_artifact_components(self):
        """Test that clear artifact ICs are marked for exclusion."""
        from unittest.mock import MagicMock

        # 4 ICs with probabilities: brain, eye artifact, muscle artifact, other
        probs = np.array(
            [
                [0.8, 0.05, 0.05, 0.02, 0.02, 0.02, 0.04],  # IC0: brain
                [0.1, 0.05, 0.7, 0.05, 0.05, 0.03, 0.02],   # IC1: eye
                [0.1, 0.7, 0.05, 0.05, 0.05, 0.03, 0.02],   # IC2: muscle
                [0.4, 0.05, 0.2, 0.05, 0.05, 0.03, 0.22],   # IC3: mixed/brain
            ]
        )
        ica = MagicMock()
        ica.exclude = []

        result = mark_ic_for_exclusion(probs, ica)
        # IC0 is brain (keep), IC3 is mixed with brain=0.4 (keep)
        # IC1 has eye=0.7 with brain=0.1 (exclude), IC2 has muscle=0.7 with brain=0.1 (exclude)
        assert 0 not in result.exclude
        assert 1 in result.exclude
        assert 2 in result.exclude
        assert 3 not in result.exclude
