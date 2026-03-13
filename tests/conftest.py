"""
Shared test fixtures for the psilocybin-eeg test suite.
"""

import pytest
import tempfile
from pathlib import Path

import pandas as pd


@pytest.fixture
def sample_participant_map(tmp_path):
    """Create a sample participant mapping CSV for testing."""
    data = {
        "participant": [
            "PSI000",
            "PSI000",
            "PSI018",
            "PSI018",
            "PSI019",
            "PSI019",
            "PSI999",
            "PSI999",
        ],
        "eeg": [
            "EEGA",
            "EEGB",
            "EEGA",
            "EEGB",
            "EEGA",
            "EEGB",
            "EEGA",
            "EEGB",
        ],
        "condition": [
            "Placebo",
            "Psilocybin",
            "Placebo",
            "Psilocybin",
            "Placebo",
            "Psilocybin",
            "Placebo",
            "Psilocybin",
        ],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_participant_map.csv"
    df.to_csv(csv_path, sep=";", index=False)
    return csv_path
