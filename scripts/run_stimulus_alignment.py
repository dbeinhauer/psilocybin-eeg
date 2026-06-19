"""
This script serves as a standalone utility to align stimulus onsets across
participants based on the stimulus-onset annotations (e.g. ``fam+`` in the ASSR
experiment). It stores the cropped (spliced) results of the alignment for future
analysis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DatasetHandler
from src.preprocessing.stimulus_alignment import DEFAULT_STIMULUS_LABEL
from src.definitions.fields import (
    ExperimentNames,
    CoordinateSystems,
    PreprocessedDataVariants,
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    SingleDataMetadata,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Align stimulus onsets across participants based on stimulus-onset "
            "annotations and store the cropped (spliced) recordings."
        )
    )
    parser.set_defaults()
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[cond.value for cond in ConditionVariants],
        help=f"The condition to process, should be one of {[cond.value for cond in ConditionVariants]}.",
    )
    parser.add_argument(
        "--stimulus_label",
        type=str,
        default=DEFAULT_STIMULUS_LABEL,
        help=f"Annotation description marking a stimulus onset (default '{DEFAULT_STIMULUS_LABEL}').",
    )

    args = parser.parse_args()
    dataset_handler = DatasetHandler(
        ExperimentNames.ASSR, CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
    )
    selected_condition = ConditionVariants(args.condition)
    # ASSR has a single (placeholder) music type and no music dimension.
    selected_music_type = MusicTypeVariants.ASSR
    exclusion_categories = [ExclusionCategories.WRONG_CONDITION, ExclusionCategories.ARTIFACTS]

    filtered_df, aligned, aligner = dataset_handler.align_stimuli_by_annotations(
        selected_music_type,
        selected_condition,
        exclusion_categories,
        stimulus_label=args.stimulus_label,
        data_type_to_load=PreprocessedDataVariants.RAW_AFTER_ICA,
    )

    # Persist each spliced recording to the cropped stage. ``aligned`` is returned
    # in the same row order as ``filtered_df``.
    for (_, row), cropped in zip(filtered_df.iterrows(), aligned):
        filename = row[SingleDataMetadata.FILENAME]
        assert cropped.n_times == aligner.total_length, (
            f"Aligned signal of file {filename} has length {cropped.n_times}, "
            f"expected {aligner.total_length}!"
        )
        dataset_handler.save_data_file(
            cropped, filename.split(".")[0], PreprocessedDataVariants.RAW_CROPPED
        )
