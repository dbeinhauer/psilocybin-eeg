"""
Tests for src/analysis/summary.py -- summarized analyzer loading behavior.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import mne

from src.analysis.summary import EEGSummarizedAnalyzer
from src.definitions.fields import (
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    MusicTypeVariants,
    PreprocessedDataVariants,
    SingleDataMetadata,
)
from src.preprocessing.stimulus_alignment import (
    ResampledPlan,
    StimulusAligner,
    get_stimulus_onset_samples,
    rescale_onsets,
)


class TestLoadAndPrepareData:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_adds_data_axis0_index_mapping_to_filtered_df(
        self, mock_dataset_handler_cls, mock_filter
    ):
        filtered_df = pd.DataFrame(
            {
                SingleDataMetadata.FILENAME: ["first_raw.fif", "second_raw.fif"],
            },
            index=[10, 20],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()

        mock_raw = MagicMock()
        mock_raw.pick.return_value = mock_raw
        mock_raw.resample.return_value = mock_raw
        mock_raw.info = {"sfreq": 200.0, "ch_names": ["Cz", "Pz"]}
        mock_raw.get_data.side_effect = [
            np.zeros((2, 3)),
            np.ones((2, 3)),
        ]
        mock_dataset_handler.load_data_file.return_value = mock_raw
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        data, _ = analyzer.load_and_prepare_data(resample_freq=200.0, n_jobs=1)

        assert data.shape == (2, 2, 3)
        np.testing.assert_array_equal(data[0], np.zeros((2, 3)))
        np.testing.assert_array_equal(data[1], np.ones((2, 3)))
        assert (
            SingleDataMetadata.CONCATENATED_PERSON_INDEX in analyzer.filtered_df.columns
        )
        assert analyzer.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0, 1]
        assert analyzer.filtered_df.index.tolist() == [10, 20]


class TestSaveLoadDataWithMetadata:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_metadata_persists_through_save_load_cycle(
        self, mock_dataset_handler_cls, mock_filter, tmp_path
    ):
        filtered_df = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["first_raw.fif", "second_raw.fif"]},
            index=[10, 20],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        analyzer.data = np.arange(12).reshape(2, 2, 3)
        analyzer.filtered_df = filtered_df.copy()
        analyzer.filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = [0, 1]
        save_path = tmp_path / "concatenated.npy"
        metadata_path = tmp_path / "concatenated.metadata.csv"
        analyzer.save_data(save_path=save_path, metadata_path=metadata_path)

        restored = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )
        restored.load_data(load_path=save_path, metadata_path=metadata_path)

        np.testing.assert_array_equal(restored.data, analyzer.data)
        assert (
            SingleDataMetadata.CONCATENATED_PERSON_INDEX in restored.filtered_df.columns
        )
        assert restored.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0, 1]
        assert restored.filtered_df[SingleDataMetadata.FILENAME].tolist() == [
            "first_raw.fif",
            "second_raw.fif",
        ]
        assert restored.filtered_df.index.tolist() == [10, 20]

    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_default_metadata_path_round_trips_without_explicit_path(
        self, mock_dataset_handler_cls, mock_filter, tmp_path
    ):
        """save_data writes ``<stem>.metadata.csv``; load_data must find it.

        Regression: load_data derived ``<stem>.csv``, so the sidecar was never
        picked up and ``filtered_df`` silently kept the freshly-filtered rows
        from ``__init__`` — which carry no CONCATENATED_PERSON_INDEX, breaking
        every subject-index -> participant lookup.
        """
        filtered_df = pd.DataFrame(
            {
                SingleDataMetadata.FILENAME: ["first_raw.fif", "second_raw.fif"],
                SingleDataMetadata.PARTICIPANT_ID: ["031", "019"],
            },
            index=[10, 20],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        def _make_analyzer():
            return EEGSummarizedAnalyzer(
                experiment_name=ExperimentNames.PSILO_MUSIC,
                coordinate_system=CoordinateSystems.HYDROGEL_257,
                music_types=[MusicTypeVariants.CLASSICAL],
                conditions=[ConditionVariants.PLACEBO],
                exclusion_categories=[],
            )

        analyzer = _make_analyzer()
        analyzer.data = np.arange(12).reshape(2, 2, 3)
        analyzer.filtered_df = filtered_df.copy()
        analyzer.filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = [0, 1]
        save_path = tmp_path / "Placebo_CLASSIC.npy"
        analyzer.save_data(save_path=save_path)

        assert (tmp_path / "Placebo_CLASSIC.metadata.csv").exists()

        restored = _make_analyzer()
        # No metadata_path: the sidecar must be found from the data path alone.
        restored.load_data(load_path=save_path)

        assert (
            SingleDataMetadata.CONCATENATED_PERSON_INDEX in restored.filtered_df.columns
        )
        assert restored.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0, 1]

    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_legacy_bare_csv_sidecar_is_still_loaded(
        self, mock_dataset_handler_cls, mock_filter, tmp_path
    ):
        """Sidecars written as ``<stem>.csv`` before the suffixes were aligned."""
        mock_filter.return_value = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["first_raw.fif"]}, index=[10]
        )
        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        save_path = tmp_path / "Placebo_CLASSIC.npy"
        np.save(save_path, np.arange(6).reshape(1, 2, 3))
        pd.DataFrame(
            {
                "SingleDataMetadata.FILENAME": ["first_raw.fif"],
                "SingleDataMetadata.CONCATENATED_PERSON_INDEX": [0],
            },
            index=[10],
        ).to_csv(tmp_path / "Placebo_CLASSIC.csv", index=True)

        restored = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )
        restored.load_data(load_path=save_path)

        assert restored.filtered_df[
            SingleDataMetadata.CONCATENATED_PERSON_INDEX
        ].tolist() == [0]


def _make_mock_raw(n_channels: int, n_times: int, sfreq: float, onset_sec: list[float]):
    """Return a mock MNE Raw with fam+ annotations at the given onset seconds."""
    raw = MagicMock()
    raw.info = {"sfreq": sfreq, "ch_names": [f"ch{i}" for i in range(n_channels)]}
    raw.n_times = n_times
    raw.pick.return_value = raw
    raw.resample.return_value = raw
    raw.get_data.return_value = np.zeros((n_channels, n_times))
    annotations = mne.Annotations(
        onset=onset_sec,
        duration=[0.0] * len(onset_sec),
        description=["fam+"] * len(onset_sec),
    )
    raw.annotations = annotations
    raw.first_time = 0.0
    raw.time_as_index = lambda t, use_rounding=False: np.round(
        np.asarray(t) * sfreq
    ).astype(int)
    return raw


class TestLoadPreAlignmentData:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_returns_list_plan_and_info(self, mock_dataset_handler_cls, mock_filter):
        sfreq = 250.0
        filtered_df = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["subj1.fif", "subj2.fif"]},
            index=[0, 1],
        )
        mock_filter.return_value = filtered_df

        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        # Two subjects: 3 s and 4 s recordings, two fam+ onsets each, further apart
        # than the ASSR epoch, which a shorter gap would refuse.
        raw1 = _make_mock_raw(2, int(3 * sfreq), sfreq, [0.5, 1.8])
        raw2 = _make_mock_raw(2, int(4 * sfreq), sfreq, [0.5, 2.0])
        mock_dataset_handler.load_data_file.side_effect = [raw1, raw2]

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.ASSR,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[MusicTypeVariants.ASSR],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        arrays, segments, plan, info = analyzer.load_pre_alignment_data(
            resample_freq=sfreq, n_jobs=1
        )

        # One array per subject.
        assert len(arrays) == 2
        assert arrays[0].shape == (2, int(3 * sfreq))
        assert arrays[1].shape == (2, int(4 * sfreq))

        # The plan has one keep_segments list per subject.
        assert isinstance(plan, ResampledPlan)
        assert len(plan.keep_segments) == 2

        # All subjects' segments sum to the same total_length.
        for segs in plan.keep_segments:
            assert sum(e - s for s, e in segs) == plan.total_length

        # RAW_AFTER_ICA was loaded, not RAW_CROPPED.
        for call in mock_dataset_handler.load_data_file.call_args_list:
            assert (
                call.kwargs.get("processed_data_type")
                == PreprocessedDataVariants.RAW_AFTER_ICA
            )

    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_plan_is_fitted_natively_and_rescaled(
        self, mock_dataset_handler_cls, mock_filter
    ):
        # Recorded at 1000 Hz, stored at 250 Hz: the plan's onsets are the native
        # plan's onsets rescaled, as RAW_CROPPED's are, not a second fit at 250 Hz.
        native, target = 1000.0, 250.0
        filenames = ["subj1.fif", "subj2.fif"]
        mock_filter.return_value = pd.DataFrame(
            {SingleDataMetadata.FILENAME: filenames}, index=[0, 1]
        )
        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        raws = [
            _make_mock_raw(2, 9000, native, [1.0, 2.3014, 3.5523, 4.8531]),
            _make_mock_raw(2, 9500, native, [1.4, 2.6987, 3.9991, 5.2546]),
        ]
        for raw in raws:
            resampled = _make_mock_raw(2, raw.n_times // 4, target, [])
            raw.resample.return_value = resampled
        mock_dataset_handler.load_data_file.side_effect = raws

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.ASSR,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[MusicTypeVariants.ASSR],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )
        arrays, segments, plan, _info = analyzer.load_pre_alignment_data(
            resample_freq=target, n_jobs=1
        )

        marker = analyzer._stimulus_marker
        native_aligner = StimulusAligner(
            [
                get_stimulus_onset_samples(r, marker.label, marker.onset_offset_for(f))
                for r, f in zip(raws, filenames)
            ],
            [r.n_times for r in raws],
            native,
            keep_tail_sec=0.1,
        )
        np.testing.assert_array_equal(
            plan.aligned_onset_samples,
            rescale_onsets(native_aligner.aligned_onset_samples, native, target),
        )
        assert [a.shape[-1] for a in arrays] == [2250, 2375]
        assert all(sum(e - s for s, e in seg) == plan.total_length for seg in segments)

    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_no_stimulus_label_raises(self, mock_dataset_handler_cls, mock_filter):
        mock_filter.return_value = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["subj1.fif"]}, index=[0]
        )
        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler

        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,  # no stimulus label
            coordinate_system=CoordinateSystems.HYDROGEL_257,
            music_types=[MusicTypeVariants.CLASSICAL],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )

        with pytest.raises(ValueError, match="no registered stimulus label"):
            analyzer.load_pre_alignment_data()


class TestExtractStimulusOnsets:
    @patch("src.analysis.summary.DatasetFilter.filter_dataset_by_all_categories")
    @patch("src.analysis.summary.DatasetHandler")
    def test_onsets_are_read_natively_and_rescaled(
        self, mock_dataset_handler_cls, mock_filter
    ):
        mock_filter.return_value = pd.DataFrame(
            {SingleDataMetadata.FILENAME: ["subj1.fif"]}, index=[0]
        )
        mock_dataset_handler = MagicMock()
        mock_dataset_handler.dataset_metadata = pd.DataFrame()
        mock_dataset_handler.excluded_participants_metadata = pd.DataFrame()
        mock_dataset_handler_cls.return_value = mock_dataset_handler
        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.ASSR,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[MusicTypeVariants.ASSR],
            conditions=[ConditionVariants.PLACEBO],
            exclusion_categories=[],
        )
        raw = _make_mock_raw(2, 6000, 1000.0, [1.0, 2.3014, 3.5523])
        marker = analyzer._stimulus_marker

        onsets = analyzer._extract_stimulus_onsets(raw, "subj1.fif", 250.0)

        native = get_stimulus_onset_samples(
            raw, marker.label, marker.onset_offset_for("subj1.fif")
        )
        np.testing.assert_array_equal(onsets, rescale_onsets(native, 1000.0, 250.0))
