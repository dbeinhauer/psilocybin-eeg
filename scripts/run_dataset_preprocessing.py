"""
Script to run all dataset preprocessing pipeline. Typically used on the
computational cluster Umbriel.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_handler import DatasetHandler
from src.definitions.fields import ExperimentNames, CoordinateSystems

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Preprocessing.")
    # Paths and directories:
    parser.set_defaults(only_excluded_ic=False)
    parser.add_argument(
        "--only_excluded_ic",
        type=bool,
        help="Whether we just want to generated dataseries from excluded ICs.",
    )

    args = parser.parse_args()

    data_handler = DatasetHandler(
        ExperimentNames.PSILO_MUSIC, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )

    if not args.only_excluded_ic:
        # We want to preprocess raw dataset with all steps.
        data_handler.preprocess_all_dataset(save_processing_info=True)

    # Exclusion of the IC components and creation of their timeseries.
    data_handler.generate_all_excluded_ic_timeseries()
