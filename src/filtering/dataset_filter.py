"""
This module contains functions for filtering dataset metadata based on music types and conditions.
"""

from typing import List
import pandas as pd

from src.definitions.fields import (
    JOINED_CONDITIONS,
    REAL_CONDITIONS,
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
    def expand_conditions(
        conditions: List[ConditionVariants],
    ) -> tuple[List[ConditionVariants], bool]:
        """
        Resolve the virtual conditions to the real conditions they stand for.

        :attr:`~src.definitions.fields.ConditionVariants.JOINED` and
        :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` are selectors,
        not values any recording carries, so every filter that touches the metadata
        must expand them first. They expand identically — the difference between them
        is only which axis the conditions are pooled along downstream, not which
        recordings are selected. Expansion is idempotent and order-preserving, and
        de-duplicates if a caller passes a virtual condition alongside a real one.

        :param conditions: Requested conditions, possibly including a virtual one.
        :return: Tuple ``(expanded, is_joined)`` where *expanded* contains only real
            conditions and *is_joined* records whether a virtual condition was
            requested (and therefore whether the complete-pair restriction applies).
        """
        is_joined = any(condition in JOINED_CONDITIONS for condition in conditions)
        if not is_joined:
            return list(conditions), False

        expanded: List[ConditionVariants] = []
        for condition in conditions:
            for resolved in (
                REAL_CONDITIONS if condition in JOINED_CONDITIONS else [condition]
            ):
                if resolved not in expanded:
                    expanded.append(resolved)
        return expanded, True

    @staticmethod
    def restrict_to_complete_pairs(
        dataset_metadata: pd.DataFrame,
        conditions: List[ConditionVariants],
    ) -> pd.DataFrame:
        """
        Keep only participants contributing a recording to *every* given condition,
        ordered by condition block and then by participant.

        This is what makes :attr:`~src.definitions.fields.ConditionVariants.JOINED` a
        balanced within-subject design. Exclusions are already applied by the caller
        and — because
        :meth:`filter_dataset_by_all_categories` drops by participant ID rather than by
        (participant, condition) — a participant excluded under *either* condition is
        gone from both by that point. What this adds is the case of a participant who
        was never excluded but simply has no recording for one condition.

        The row order defines the subject axis of the concatenated array (see
        :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.load_and_prepare_data`,
        which writes it into
        :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX`).
        Rows are emitted as one contiguous block per condition, in the order given by
        *conditions*, with participants in the same order within each block. Subject
        ``k`` and subject ``k + n_pairs`` are therefore the same participant under the
        two conditions.

        :param dataset_metadata: Already condition- and exclusion-filtered metadata.
        :param conditions: The real conditions that must all be present per participant.
        :return: Metadata restricted to complete participants, in block order.
        """
        if dataset_metadata.empty:
            return dataset_metadata

        participant_key = DatasetFilter._resolve_key(
            dataset_metadata, SingleDataMetadata.PARTICIPANT_ID
        )
        condition_key = DatasetFilter._resolve_key(
            dataset_metadata, SingleDataMetadata.CONDITION
        )

        condition_values = [c if isinstance(c, str) else c.value for c in conditions]
        present = dataset_metadata.groupby(participant_key)[condition_key].apply(
            lambda column: {
                value if isinstance(value, str) else value.value for value in column
            }
        )
        complete = sorted(
            participant
            for participant, values in present.items()
            if all(value in values for value in condition_values)
        )

        in_cohort = dataset_metadata[participant_key].isin(complete)
        blocks = [
            # Match on both the enum member and its value: the column holds enums
            # in a freshly-parsed frame and plain strings after a CSV round-trip.
            dataset_metadata[
                in_cohort
                & dataset_metadata[condition_key].isin({condition, condition_value})
            ].sort_values(participant_key, kind="stable")
            for condition, condition_value in zip(conditions, condition_values)
        ]
        return pd.concat(blocks) if blocks else dataset_metadata.iloc[:0]

    @staticmethod
    def _resolve_key(dataset_metadata: pd.DataFrame, field: SingleDataMetadata):
        """
        Return the column key *field* is stored under, tolerating a CSV round-trip.

        Sidecar metadata written to CSV comes back with plain-string column names, so
        every lookup has to accept either the enum member or its value.

        :param dataset_metadata: The DataFrame to inspect.
        :param field: The metadata field to look up.
        :return: Either *field* itself or ``field.value``, whichever indexes the frame.
        """
        return field if field in dataset_metadata else field.value

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
        :param conditions: List of conditions to include. May contain
            :attr:`~src.definitions.fields.ConditionVariants.JOINED`, which selects both
            real conditions and additionally restricts the result to participants
            present in both (see :meth:`restrict_to_complete_pairs`).
        :param exclusion_categories: List of exclusion categories to include.
        :return: Filtered DataFrame containing only rows that match the specified music types, conditions, and exclusion categories.
        """
        # JOINED is a selector over the real conditions, so resolve it before anything
        # touches the metadata. Note that requesting both conditions also widens the
        # exclusion step below: it drops by participant ID, so a participant excluded
        # under either condition is dropped from both — which is exactly what a paired
        # design needs.
        conditions, is_joined = DatasetFilter.expand_conditions(conditions)

        # Prepare excluded participants list based on the selected condition and music type and exclusion categories.
        # A blank (empty/NaN) condition or music type in the excluded-participants
        # metadata is a wildcard: the exclusion applies to every variant of that
        # field (e.g. an exclusion with no music type excludes the participant from
        # all music types). This wildcard handling is specific to exclusions, so the
        # generic condition/music-type filters are not used here.
        excluded_participants = DatasetFilter._filter_excluded_by_condition_and_music(
            DatasetFilter.filter_dataset_by_exclusion_categories(
                excluded_participants_metadata,
                exclusion_categories,
            ),
            conditions,
            music_types,
        )

        included_participants_ids = [
            p
            for p in dataset_metadata[SingleDataMetadata.PARTICIPANT_ID].tolist()
            if p
            not in excluded_participants[
                SingleDataMetadata.PARTICIPANT_ID.value
            ].tolist()
        ]

        filtered = DatasetFilter.filter_dataset_by_participant_ids(
            DatasetFilter.filter_dataset_by_music_type(
                DatasetFilter.filter_dataset_by_condition(dataset_metadata, conditions),
                music_types,
            ),
            included_participants_ids,
        )

        if is_joined:
            filtered = DatasetFilter.restrict_to_complete_pairs(filtered, conditions)
        return filtered

    @staticmethod
    def _filter_excluded_by_condition_and_music(
        excluded_participants_metadata: pd.DataFrame,
        conditions: List[ConditionVariants],
        music_types: List[MusicTypeVariants],
    ) -> pd.DataFrame:
        """
        Restrict excluded-participants metadata to the requested conditions and music
        types, treating a blank (empty/NaN) field as a wildcard.

        Unlike the generic condition / music-type filters, a row whose condition (or
        music type) is unspecified matches *every* requested condition (or music
        type). This expresses exclusions that apply to a participant across all
        variants of a field — e.g. an exclusion with no music type drops the
        participant from every music type of the selected condition(s).

        :param excluded_participants_metadata: Excluded-participants metadata to filter.
        :param conditions: Requested conditions.
        :param music_types: Requested music types.
        :return: Filtered excluded-participants metadata.
        """

        def matches(column_key, selected) -> pd.Series:
            key = (
                column_key
                if column_key in excluded_participants_metadata
                else column_key.value
            )
            column = excluded_participants_metadata[key]
            blank = column.isna() | (column.astype(str).str.strip() == "")
            allowed = {s if isinstance(s, str) else s.value for s in selected} | set(
                selected
            )
            return blank | column.isin(allowed)

        return excluded_participants_metadata[
            matches(SingleDataMetadata.CONDITION, conditions)
            & matches(SingleDataMetadata.MUSIC_TYPE, music_types)
        ]

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

        Note this applies the condition selection only. When
        :attr:`~src.definitions.fields.ConditionVariants.JOINED` is requested it expands
        to both real conditions, but the complete-pair restriction that makes a joined
        dataset balanced is *not* applied here — it must run after exclusions, so it
        lives in :meth:`filter_dataset_by_all_categories`.

        :param dataset_metadata: The dataset metadata DataFrame to filter
        :param selected_conditions: List of conditions to include (e.g., [ConditionVariants.PLACEBO])
        :return: Filtered DataFrame containing only rows with the specified conditions
        """
        selected_conditions, _ = DatasetFilter.expand_conditions(selected_conditions)
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
