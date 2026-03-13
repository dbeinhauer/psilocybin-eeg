"""
This script serves as a standalone utility to align time signals across participants based on the TAG signal.
It stored the cropped results of the alignment for future analysis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DatasetHandler
from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    PreprocessedDataVariants,
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align time signals across participants based on the TAG signal."
    )
    # Paths and directories:
    parser.set_defaults()
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[cond.value for cond in ConditionVariants],
        help=f"The condition to process, should be one of {[cond.value for cond in ConditionVariants]}.",
    )
    parser.add_argument(
        "--music_type",
        type=str,
        default=MusicTypeVariants.CLASSICAL.value,
        choices=[mt.value for mt in MusicTypeVariants],
        help=f"The music type to process, should be one of {[mt.value for mt in MusicTypeVariants]}.",
    )

    args = parser.parse_args()
    dataset_handler = DatasetHandler(
        ExperimentNames.PSILO_MUSIC, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )
    selected_conditions = [ConditionVariants(args.condition)]
    selected_music_types = [MusicTypeVariants(args.music_type)]
    exclusion_categories = [ExclusionCategories.BAD_MUSIC]

    aligned_signals, time_aligner = dataset_handler.align_time_series(
        selected_music_types[0],
        selected_conditions[0],
        exclusion_categories,
        plot_alignment_results=False,
        data_type_to_load=PreprocessedDataVariants.RAW_AFTER_ICA,
    )
    dataset_handler.crop_all_raw_to_alignment(time_aligner)
