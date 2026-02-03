"""
Script to run all dataset preprocessing pipeline. Typically used on the
computational cluster Umbriel.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_handler import DatasetHandler
from src.definitions.fields import ExperimentNames, CoordinateSystems

if __name__ == "__main__":
    data_handler = DatasetHandler(
        ExperimentNames.PSILO_MUSIC, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )
    data_handler.preprocess_all_dataset(save_processing_info=True)
