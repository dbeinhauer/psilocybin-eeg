"""
This module contains functions for filtering dataset metadata based on music types and conditions.
"""

from typing import List
import pandas as pd

from src.definitions.fields import (
    SingleDataMetadata,
    ConditionVariants,
    MusicTypeVariants,
    ExclusionCategories,
)


class DatasetFilter:
    """
    Class to filter the data for selection from metadata dataframe.
    """

    @staticmethod
    def filter_dataset_by_all_categories(
        dataset_metadata: pd.DataFrame,
        excluded_participants_metadata: pd.DataFrame,
        music_types: List[MusicTypeVariants],
        conditions: List[ConditionVariants],
        exclusion_categories: List[ExclusionCategories],
    ) -> pd.DataFrame:
        """
        Filter dataset metadata by all categories (music type, condition, and exclusion categories).

        :param dataset_metadata: The dataset metadata DataFrame to filter
        :param excluded_participants_metadata: The excluded participants metadata DataFrame to filter
        :param music_types: List of music types to include.
        :param conditions: List of conditions to include.
        :param exclusion_categories: List of exclusion categories to include.
        :return: Filtered DataFrame containing only rows that match the specified music types, conditions, and exclusion categories.
        """
        # Prepare excluded participants list based on the selected condition and music type and exclusion categories.
        excluded_participants = DatasetFilter.filter_dataset_by_exclusion_categories(
            DatasetFilter.filter_dataset_by_music_type(
                DatasetFilter.filter_dataset_by_condition(
                    excluded_participants_metadata,
                    conditions,
                ),
                music_types,
            ),
            exclusion_categories,
        )

        included_participants_ids = [
            p
            for p in dataset_metadata[SingleDataMetadata.PARTICIPANT_ID].tolist()
            if p
            not in excluded_participants[
                SingleDataMetadata.PARTICIPANT_ID.value
            ].tolist()
        ]

        return DatasetFilter.filter_dataset_by_participant_ids(
            DatasetFilter.filter_dataset_by_music_type(
                DatasetFilter.filter_dataset_by_condition(dataset_metadata, conditions),
                music_types,
            ),
            included_participants_ids,
        )

    @staticmethod
    def filter_dataset_by_music_type(
        dataset_metadata: pd.DataFrame, selected_music_types: List[MusicTypeVariants]
    ) -> pd.DataFrame:
        """
        Filter dataset metadata by selected music types.

        :param dataset_metadata: The dataset metadata DataFrame to filter
        :param selected_music_types: List of music types to include.
        :return: Filtered DataFrame containing only rows with the specified music types
        """
        # music_type_values = [mt for mt in selected_music_types]
        music_type_values = {
            mt if isinstance(mt, str) else mt.value for mt in selected_music_types
        } | {mt for mt in selected_music_types}
        key = (
            SingleDataMetadata.MUSIC_TYPE
            if SingleDataMetadata.MUSIC_TYPE in dataset_metadata
            else SingleDataMetadata.MUSIC_TYPE.value
        )
        return dataset_metadata[dataset_metadata[key].isin(music_type_values)]

    @staticmethod
    def filter_dataset_by_condition(
        dataset_metadata: pd.DataFrame, selected_conditions: List[ConditionVariants]
    ) -> pd.DataFrame:
        """
        Filter dataset metadata by selected conditions.

        :param dataset_metadata: The dataset metadata DataFrame to filter
        :param selected_conditions: List of conditions to include (e.g., [ConditionVariants.PLACEBO])
        :return: Filtered DataFrame containing only rows with the specified conditions
        """
        # condition_values = [c for c in selected_conditions]
        condition_values = {
            mt if isinstance(mt, str) else mt.value for mt in selected_conditions
        } | {mt for mt in selected_conditions}
        key = (
            SingleDataMetadata.CONDITION
            if SingleDataMetadata.CONDITION in dataset_metadata
            else SingleDataMetadata.CONDITION.value
        )
        return dataset_metadata[dataset_metadata[key].isin(condition_values)]

    @staticmethod
    def get_unique_values(
        dataset_metadata: pd.DataFrame, column: SingleDataMetadata
    ) -> List[str]:
        """
        Get unique values for a specific column in the dataset metadata.

        :param dataset_metadata: The dataset metadata DataFrame
        :param column: The column to get unique values for
        :return: List of unique values in the specified column
        """
        return dataset_metadata[column].unique().tolist()

    @staticmethod
    def filter_dataset_by_participant_ids(
        dataset_metadata: pd.DataFrame, participant_ids: List[str]
    ) -> pd.DataFrame:
        """
        Filter dataset metadata by specific participant IDs.

        :param dataset_metadata: The dataset metadata DataFrame to filter
        :param participant_ids: List of participant IDs to include
        :return: Filtered DataFrame containing only rows for the specified participants
        """
        key = (
            SingleDataMetadata.PARTICIPANT_ID
            if SingleDataMetadata.PARTICIPANT_ID in dataset_metadata
            else SingleDataMetadata.PARTICIPANT_ID.value
        )
        return dataset_metadata[dataset_metadata[key].isin(participant_ids)]

    @staticmethod
    def filter_dataset_by_exclusion_categories(
        excluded_participants_metadata: pd.DataFrame,
        exclusion_categories: List[ExclusionCategories],
    ) -> pd.DataFrame:
        """
        Filter excluded participants metadata by specific exclusion categories.

        :param excluded_participants_metadata: The excluded participants metadata DataFrame to filter
        :param exclusion_categories: List of exclusion categories to include.
        :return: Filtered DataFrame containing only rows for the specified exclusion categories.
        """
        categories = [category.value for category in exclusion_categories]
        return excluded_participants_metadata[
            excluded_participants_metadata[
                SingleDataMetadata.EXCLUSION_EXPLANATION.value
            ].isin(categories)
        ]
