"""
This script serves as a standalone utility to align stimulus onsets across
participants based on the stimulus-onset annotations (e.g. ``fam+`` in the ASSR
experiment). It stores the cropped (spliced) results of the alignment for future
analysis.

The alignment is fitted **once**, over every recording of both conditions, so all
conditions come out with the same length and carry exactly the same stimuli. Condition
selection and any custom participant subsetting then happen at analysis time as pure
filtering, and never change the time axis — so results stay comparable across analyses
and no wavelet ever needs recomputing for a different selection.

Because ``StimulusAligner`` trims every inter-stimulus interval to the group minimum,
widening the group costs a little data: over all 38 ASSR recordings the common stimulus
count is 148 (vs 148 Placebo-only and 149 Psilocybin-only). Keeping every recording
available for later selection is worth that.
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
    ExclusionCategories,
    SingleDataMetadata,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Align stimulus onsets across participants based on stimulus-onset "
            "annotations and store the cropped (spliced) recordings. Runs over both "
            "conditions at once so they share a time base."
        )
    )
    parser.set_defaults()
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
    # ASSR has a single (placeholder) music type and no music dimension.
    selected_music_type = MusicTypeVariants.ASSR
    # Deliberately minimal: a recording dropped here can never be selected later,
    # because it will not have been aligned. Every ASSR recording carries 148-150
    # stimulus onsets, so none of them degrades the group minimum enough to be worth
    # excluding — only genuinely mis-recorded conditions are dropped.
    exclusion_categories = [ExclusionCategories.WRONG_CONDITION]

    filtered_df, aligned, aligner = dataset_handler.align_stimuli_by_annotations(
        selected_music_type,
        exclusion_categories,
        stimulus_label=args.stimulus_label,
        data_type_to_load=PreprocessedDataVariants.RAW_AFTER_ICA,
    )

    # ``aligned`` is returned in the same row order as ``filtered_df``.
    for (_, row), cropped in zip(filtered_df.iterrows(), aligned):
        filename = row[SingleDataMetadata.FILENAME]
        assert cropped.n_times == aligner.total_length, (
            f"Aligned signal of file {filename} has length {cropped.n_times}, "
            f"expected {aligner.total_length}!"
        )
        dataset_handler.save_data_file(
            cropped, filename.split(".")[0], PreprocessedDataVariants.RAW_CROPPED
        )
