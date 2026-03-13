"""
Tests for src/io/parsing.py — check_enum_value_in_variants utility function.
"""

import pytest

from src.definitions.fields import (
    ConditionVariants,
    MusicTypeVariants,
    EEGConditions,
    ICLabelComponentsClasses,
)
from src.io.parsing import check_enum_value_in_variants


class TestCheckEnumValueInVariants:
    """Test the check_enum_value_in_variants helper."""

    def test_valid_condition_variant(self):
        result = check_enum_value_in_variants(ConditionVariants, "Placebo")
        assert result == ConditionVariants.PLACEBO

    def test_valid_music_type(self):
        result = check_enum_value_in_variants(MusicTypeVariants, "CLASSIC")
        assert result == MusicTypeVariants.CLASSICAL

    def test_invalid_value_returns_none(self):
        result = check_enum_value_in_variants(ConditionVariants, "invalid")
        assert result is None

    def test_eeg_conditions(self):
        result = check_enum_value_in_variants(EEGConditions, "A")
        assert result == EEGConditions.CONDITION_A

    def test_case_sensitive(self):
        result = check_enum_value_in_variants(EEGConditions, "a")
        assert result is None

    def test_iclabel_classes(self):
        result = check_enum_value_in_variants(ICLabelComponentsClasses, "brain")
        assert result == ICLabelComponentsClasses.BRAIN
