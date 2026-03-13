"""
Script to run ISC and group-level analysis on preprocessed EEG data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISC and group-level analysis.")
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

    # Import here to defer heavy dependency loading

    print(
        f"Analysis entry point for condition={args.condition}, music_type={args.music_type}"
    )
    print("Instantiate EEGSummarizedAnalyzer and run desired analysis methods.")
