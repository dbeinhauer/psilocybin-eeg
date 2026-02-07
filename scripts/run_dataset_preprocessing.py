"""
Script to run all dataset preprocessing pipeline.
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
    parser.set_defaults(raw_processing=False)
    parser.add_argument(
        "--raw_processing",
        type=bool,
        help="Whether we want to process the raw unfiltered data.",
    )
    parser.set_defaults(process_excluded_ic=False)
    parser.add_argument(
        "--process_excluded_ic",
        type=bool,
        help="Whether we want to generated dataseries from excluded ICs.",
    )
    parser.set_defaults(plot_results=False)
    parser.add_argument(
        "--plot_results",
        type=bool,
        help="Flag whether we want to plot the results.",
    )

    args = parser.parse_args()

    data_handler = DatasetHandler(
        ExperimentNames.PSILO_MUSIC, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )

    if args.raw_processing:
        # We want to preprocess raw dataset with all steps.
        data_handler.preprocess_all_dataset(save_processing_info=True)

    if args.process_excluded_ic:
        # Exclusion of the IC components and creation of their timeseries.
        data_handler.generate_all_excluded_ic_timeseries()

    if args.plot_results:
        data_handler.plot_all_one_variant(plot_variant="power_spectrum")
        data_handler.plot_all_one_variant(plot_variant="topomap")
