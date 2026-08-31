"""
This script serves as a standalone utility to align time signals across participants
based on the TAG signal. It stores the cropped results of the alignment for future
analysis.

The alignment is fitted **once**, over every recording of both conditions, so Placebo
and Psilocybin come out on one time base. Condition selection and any custom
participant subsetting then happen at analysis time as pure filtering, and never change
the time axis — so results stay comparable across analyses and no wavelet ever needs
recomputing for a different selection.

Only :attr:`~src.definitions.fields.ExclusionCategories.BAD_MUSIC` is excluded here: a
wrong TAG channel would corrupt the cross-correlation for the whole group. Everything
else is kept, because a recording dropped at this stage can never be selected later.
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
    ExclusionCategories,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Align time signals across participants based on the TAG signal. Runs over "
            "both conditions at once so they share a time base."
        )
    )
    parser.set_defaults()
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
    selected_music_type = MusicTypeVariants(args.music_type)
    # A wrong TAG channel would corrupt the cross-correlation for every recording in
    # the group, so it has to go. Nothing else is excluded — anything dropped here
    # becomes permanently unselectable, since it will not have been aligned.
    exclusion_categories = [ExclusionCategories.BAD_MUSIC]

    aligned_signals, time_aligner = dataset_handler.align_time_series(
        selected_music_type,
        exclusion_categories,
        plot_alignment_results=False,
        data_type_to_load=PreprocessedDataVariants.RAW_AFTER_ICA,
    )
    dataset_handler.crop_all_raw_to_alignment(time_aligner)
