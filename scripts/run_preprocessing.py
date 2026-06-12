"""
Script to run all dataset preprocessing pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DatasetHandler
from src.definitions.fields import ExperimentNames, CoordinateSystems

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Preprocessing.")
    # Paths and directories:
    parser.add_argument(
        "--experiment",
        type=str,
        default=ExperimentNames.PSILO_MUSIC.value,
        # default=ExperimentNames.ASSR.value,
        choices=[experiment.value for experiment in ExperimentNames],
        help="Which experiment dataset to preprocess.",
    )
    parser.set_defaults(raw_processing=False)
    parser.add_argument(
        "--raw_processing",
        action="store_true",
        help="Whether we want to process the raw unfiltered data.",
    )
    parser.set_defaults(process_excluded_ic=False)
    parser.add_argument(
        "--process_excluded_ic",
        action="store_true",
        help="Whether we want to process metadata from excluded ICs.",
    )
    parser.set_defaults(plot_results=False)
    parser.add_argument(
        "--plot_results",
        action="store_true",
        help="Flag whether we want to plot the results.",
    )

    args = parser.parse_args()

    data_handler = DatasetHandler(
        ExperimentNames(args.experiment),
        CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
    )

    if args.raw_processing:
        # We want to preprocess raw dataset with all steps.
        data_handler.preprocess_all_dataset(save_processing_info=True)

    if args.process_excluded_ic:
        # Exclusion of the IC components and creation of their timeseries.
        data_handler.extract_all_excluded_ic_metadata()

    if args.plot_results:
        data_handler.plot_all_one_variant(plot_variant="power_spectrum")
        data_handler.plot_all_one_variant(plot_variant="topomap")
