"""
Tests for scripts/analysis_common.py — argument parsing and workflow helpers.
"""

import argparse
from unittest.mock import patch
import runpy

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (
    _wavelet_transform,
    add_common_arguments,
    participant_label,
    participant_labels,
    resolve_wavelet_dir,
    run_wavelet_workflow,
)
from src.analysis.data_representations import AnalysisData, DataRepresentation
from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    AnalysisVariants,
    ExperimentNames,
    SingleDataMetadata,
)
from src.definitions.frequency import (
    WAVELET_FREQ_MAX,
    WAVELET_FREQ_MIN,
    WAVELET_N_FREQS,
)


class TestAddCommonArguments:
    """Test that add_common_arguments registers all expected CLI parameters."""

    @pytest.fixture
    def parser(self):
        p = argparse.ArgumentParser()
        add_common_arguments(p)
        return p

    # ── --analysis ────────────────────────────────────────────────

    def test_analysis_defaults_to_wavelet_power(self, parser):
        args = parser.parse_args([])
        assert set(args.analysis) == {
            AnalysisVariants.WAVELET_POWER.value,
        }

    def test_analysis_accepts_wavelet_power(self, parser):
        args = parser.parse_args(["--analysis", AnalysisVariants.WAVELET_POWER.value])
        assert AnalysisVariants.WAVELET_POWER.value in args.analysis

    def test_analysis_accepts_wavelet_phase(self, parser):
        args = parser.parse_args(["--analysis", AnalysisVariants.WAVELET_PHASE.value])
        assert AnalysisVariants.WAVELET_PHASE.value in args.analysis

    def test_analysis_accepts_all_wavelet_choices(self, parser):
        args = parser.parse_args(
            [
                "--analysis",
                AnalysisVariants.WAVELET_POWER.value,
                AnalysisVariants.WAVELET_PHASE.value,
            ]
        )
        assert set(args.analysis) == {
            AnalysisVariants.WAVELET_POWER.value,
            AnalysisVariants.WAVELET_PHASE.value,
        }

    def test_analysis_rejects_isc(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--analysis", AnalysisVariants.ISC.value])

    def test_analysis_rejects_mean_variance(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--analysis", AnalysisVariants.MEAN_VARIANCE.value])

    def test_analysis_rejects_invalid_choice(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--analysis", "invalid_choice"])

    # ── --n_jobs ──────────────────────────────────────────────────

    def test_n_jobs_default_is_one(self, parser):
        args = parser.parse_args([])
        assert args.n_jobs == 1

    def test_n_jobs_accepts_minus_one(self, parser):
        args = parser.parse_args(["--n_jobs", "-1"])
        assert args.n_jobs == -1

    def test_n_jobs_accepts_positive_int(self, parser):
        args = parser.parse_args(["--n_jobs", "4"])
        assert args.n_jobs == 4

    # ── wavelet frequency parameters ─────────────────────────────

    def test_wavelet_freq_min_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_freq_min == pytest.approx(WAVELET_FREQ_MIN)

    def test_wavelet_freq_max_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_freq_max == pytest.approx(WAVELET_FREQ_MAX)

    def test_wavelet_n_freqs_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_n_freqs == WAVELET_N_FREQS

    def test_wavelet_freq_min_custom(self, parser):
        args = parser.parse_args(["--wavelet_freq_min", "4.0"])
        assert args.wavelet_freq_min == pytest.approx(4.0)

    def test_wavelet_freq_max_custom(self, parser):
        args = parser.parse_args(["--wavelet_freq_max", "100.0"])
        assert args.wavelet_freq_max == pytest.approx(100.0)

    def test_wavelet_n_freqs_custom(self, parser):
        args = parser.parse_args(["--wavelet_n_freqs", "50"])
        assert args.wavelet_n_freqs == 50

    def test_wavelet_bands_default(self, parser):
        args = parser.parse_args([])
        assert args.wavelet_bands is None

    def test_wavelet_bands_custom_subset(self, parser):
        args = parser.parse_args(["--wavelet_bands", "delta", "alpha"])
        assert args.wavelet_bands == ["delta", "alpha"]

    def test_skip_wavelet_broadband_default(self, parser):
        args = parser.parse_args([])
        assert args.skip_wavelet_broadband is False

    def test_skip_wavelet_broadband_flag(self, parser):
        args = parser.parse_args(["--skip_wavelet_broadband"])
        assert args.skip_wavelet_broadband is True

    def test_wavelet_data_dir_default(self, parser):
        # The flag itself defaults to None; the experiment-aware path is built
        # by resolve_wavelet_dir (see TestResolveWaveletDir).
        args = parser.parse_args([])
        assert args.wavelet_data_dir is None

    def test_wavelet_data_dir_custom(self, parser):
        args = parser.parse_args(["--wavelet_data_dir", "/tmp/data"])
        assert args.wavelet_data_dir == "/tmp/data"

    def test_reuse_wavelets_flag(self, parser):
        args = parser.parse_args(["--reuse_wavelets"])
        assert args.reuse_wavelets is True

    def test_backward_compatible_wavelet_cache_aliases(self, parser):
        args = parser.parse_args(
            ["--wavelet_cache_dir", "/tmp/wavelets", "--reuse_wavelet_cache"]
        )
        assert args.wavelet_data_dir == "/tmp/wavelets"
        assert args.reuse_wavelets is True

    def test_wavelet_keep_frequency_dim_flag(self, parser):
        args = parser.parse_args(["--wavelet_keep_frequency_dim"])
        assert args.wavelet_keep_frequency_dim is True

    def test_wavelet_reshape_frequency_dim_flag(self, parser):
        args = parser.parse_args(["--wavelet_reshape_frequency_dim"])
        assert args.wavelet_reshape_frequency_dim is True

    def test_wavelet_freqs_linspace_construction(self, parser):
        """Check that wavelet_freq_{min,max,n_freqs} produce a valid linspace."""
        args = parser.parse_args(
            [
                "--wavelet_freq_min",
                "1.0",
                "--wavelet_freq_max",
                "40.0",
                "--wavelet_n_freqs",
                "20",
            ]
        )
        freqs = np.linspace(
            args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
        )
        assert freqs.shape == (20,)
        assert freqs[0] == pytest.approx(1.0)
        assert freqs[-1] == pytest.approx(40.0)

    # ── existing parameters still present ────────────────────────

    def test_window_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.window_sec == pytest.approx(5.0)

    def test_step_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.step_sec == pytest.approx(2.5)

    def test_isc_threshold_default(self, parser):
        args = parser.parse_args([])
        assert args.isc_threshold == pytest.approx(0.035)

    # ── new multi-scale ISC parameters ───────────────────────────

    def test_window_fine_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.window_fine_sec == pytest.approx(1.0)

    def test_window_fine_sec_custom(self, parser):
        args = parser.parse_args(["--window_fine_sec", "2.0"])
        assert args.window_fine_sec == pytest.approx(2.0)

    def test_window_large_sec_default(self, parser):
        args = parser.parse_args([])
        assert args.window_large_sec == pytest.approx(15.0)

    def test_window_large_sec_custom(self, parser):
        args = parser.parse_args(["--window_large_sec", "30.0"])
        assert args.window_large_sec == pytest.approx(30.0)

    def test_n_ch_subsample_default(self, parser):
        args = parser.parse_args([])
        assert args.n_ch_subsample == 64

    def test_n_ch_subsample_custom(self, parser):
        args = parser.parse_args(["--n_ch_subsample", "128"])
        assert args.n_ch_subsample == 128

    def test_n_ch_subsample_zero_disables_subsampling(self, parser):
        args = parser.parse_args(["--n_ch_subsample", "0"])
        assert args.n_ch_subsample == 0


class TestResolveWaveletDir:
    """Test that resolve_wavelet_dir always scopes the cache by experiment."""

    def test_default_base_is_processed_data_dir(self):
        for experiment in ExperimentNames:
            assert resolve_wavelet_dir(None, experiment) == (
                ProjectPaths.PROCESSED_DATA_DIR / experiment.value / "wavelets"
            )

    def test_custom_base_gets_experiment_suffix(self):
        result = resolve_wavelet_dir("/tmp/data", ExperimentNames.ASSR)
        assert result == Path("/tmp/data") / ExperimentNames.ASSR.value / "wavelets"

    def test_experiment_segment_drives_path_not_the_base(self):
        # Same base, different experiments → different, non-overlapping dirs.
        psilo = resolve_wavelet_dir("/tmp/data", ExperimentNames.PSILO_MUSIC)
        assr = resolve_wavelet_dir("/tmp/data", ExperimentNames.ASSR)
        assert psilo != assr
        assert ExperimentNames.ASSR.value not in psilo.parts
        assert ExperimentNames.PSILO_MUSIC.value not in assr.parts

    def test_rejects_experiment_specific_base(self):
        # A stale '.../psilo_music/wavelets' base must be rejected rather than
        # silently nesting or mis-routing the cache across experiments.
        with pytest.raises(ValueError, match="experiment-specific"):
            resolve_wavelet_dir(
                "/mnt/data/processed/psilo_music/wavelets",
                ExperimentNames.ASSR,
            )

    def test_rejects_any_experiment_name_segment(self):
        for experiment in ExperimentNames:
            with pytest.raises(ValueError):
                resolve_wavelet_dir(
                    f"/data/{experiment.value}", ExperimentNames.PSILO_MUSIC
                )


class TestRunWaveletWorkflowValidation:
    """Test input validation in run_wavelet_workflow."""

    @pytest.fixture
    def sample_datasets(self, tmp_path):
        rng = np.random.default_rng(0)
        return {
            "TEST": AnalysisData(
                data=rng.normal(size=(3, 4, 500)),
                sfreq=250.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="TEST",
            )
        }

    def test_invalid_representation_raises(self, sample_datasets, tmp_path):
        freqs = np.linspace(4.0, 30.0, 5)
        with pytest.raises(
            ValueError, match="representation must be 'power' or 'phase'"
        ):
            run_wavelet_workflow(
                sample_datasets,
                analyzers={},
                representation="invalid",
                freqs=freqs,
                save_dir=tmp_path / "out",
            )


class TestWaveletStorageBehavior:
    @pytest.fixture
    def sample_dataset(self):
        rng = np.random.default_rng(42)
        return {
            "TEST": AnalysisData(
                data=rng.normal(size=(2, 3, 120)),
                sfreq=120.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="TEST",
                feature_names=["Fz", "Cz", "Pz"],
            )
        }

    def test_wavelet_store_always_saves_frequency_dim(self, tmp_path, sample_dataset):
        freqs = np.linspace(4.0, 8.0, 3)
        _wavelet_transform(
            sample_dataset,
            freqs,
            representation="power",
            keep_frequency_dim=False,
            wavelet_dir=tmp_path / "wavelets",
            reuse_wavelets=False,
        )

        saved = list((tmp_path / "wavelets").glob("*.npz"))
        assert len(saved) == 1
        loaded = np.load(saved[0])
        assert bool(loaded["keep_frequency_dim"]) is True
        assert loaded["data"].shape == (2, 3 * len(freqs), 120)
        assert "_freqdim1.npz" in saved[0].name


class TestWaveletLoadShapeBehavior:
    @pytest.fixture
    def sample_dataset(self):
        rng = np.random.default_rng(7)
        return {
            "TEST": AnalysisData(
                data=rng.normal(size=(2, 3, 120)),
                sfreq=120.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="TEST",
                feature_names=["Fz", "Cz", "Pz"],
            )
        }

    def test_wavelet_reuse_can_reshape_to_4d(self, tmp_path, sample_dataset):
        freqs = np.linspace(4.0, 8.0, 3)
        wavelet_dir = tmp_path / "wavelets"

        _wavelet_transform(
            sample_dataset,
            freqs,
            representation="power",
            keep_frequency_dim=False,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=False,
        )

        loaded = _wavelet_transform(
            sample_dataset,
            freqs,
            representation="power",
            keep_frequency_dim=True,
            reshape_frequency_dim=True,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=True,
        )["TEST"]

        assert loaded.data.shape == (2, 3, len(freqs), 120)
        assert loaded.metadata["keep_frequency_dim"] is True

    def test_wavelet_reshape_requires_keep_frequency_dim(self, sample_dataset):
        freqs = np.linspace(4.0, 8.0, 3)
        with pytest.raises(
            ValueError,
            match="reshape_frequency_dim=True requires keep_frequency_dim=True",
        ):
            _wavelet_transform(
                sample_dataset,
                freqs,
                representation="power",
                keep_frequency_dim=False,
                reshape_frequency_dim=True,
                wavelet_dir=None,
                reuse_wavelets=False,
            )


class TestRunAnalysisWaveletReshapeArgPropagation:
    def test_run_analysis_passes_reshape_frequency_dim(self):
        with (
            patch(
                "scripts.analysis_common.load_analyzers",
                return_value={"TEST": object()},
            ),
            patch(
                "scripts.analysis_common.analyzers_to_datasets",
                return_value={"TEST": object()},
            ),
            patch(
                "scripts.analysis_common.run_wavelet_workflow"
            ) as run_wavelet_workflow,
            patch.object(
                sys,
                "argv",
                [
                    "scripts/run_analysis.py",
                    "--analysis",
                    "wavelet_power",
                    "--wavelet_keep_frequency_dim",
                    "--wavelet_reshape_frequency_dim",
                ],
            ),
        ):
            runpy.run_path(
                str(Path(__file__).parent.parent / "scripts" / "run_analysis.py"),
                run_name="__main__",
            )

        assert run_wavelet_workflow.call_count == 1
        call_kwargs = run_wavelet_workflow.call_args.kwargs
        assert call_kwargs["keep_frequency_dim"] is True
        assert call_kwargs["reshape_frequency_dim"] is True


class TestRunAnalysisArgValidation:
    """Test CLI-level validation in run_analysis.py."""

    def test_reshape_without_keep_raises(self):
        """--wavelet_reshape_frequency_dim without --wavelet_keep_frequency_dim should fail."""
        with (
            patch(
                "scripts.analysis_common.load_analyzers",
                return_value={"TEST": object()},
            ),
            patch(
                "scripts.analysis_common.analyzers_to_datasets",
                return_value={"TEST": object()},
            ),
            patch.object(
                sys,
                "argv",
                [
                    "scripts/run_analysis.py",
                    "--analysis",
                    "wavelet_power",
                    "--wavelet_reshape_frequency_dim",
                    # intentionally omitting --wavelet_keep_frequency_dim
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            runpy.run_path(
                str(Path(__file__).parent.parent / "scripts" / "run_analysis.py"),
                run_name="__main__",
            )
        assert exc_info.value.code != 0


class TestRunWaveletWorkflowReshapeRaisesInWorkflow:
    """reshape_frequency_dim=True must raise early in run_wavelet_workflow."""

    @pytest.fixture
    def sample_datasets(self, tmp_path):
        rng = np.random.default_rng(0)
        return {
            "TEST": AnalysisData(
                data=rng.normal(size=(3, 4, 500)),
                sfreq=250.0,
                representation=DataRepresentation.TIME_DOMAIN,
                label="TEST",
            )
        }

    def test_reshape_frequency_dim_raises_in_workflow(self, sample_datasets, tmp_path):
        freqs = np.linspace(4.0, 30.0, 5)
        with pytest.raises(
            ValueError, match="reshape_frequency_dim=True is not supported"
        ):
            run_wavelet_workflow(
                sample_datasets,
                analyzers={},
                representation="power",
                freqs=freqs,
                save_dir=tmp_path / "out",
                reshape_frequency_dim=True,
                keep_frequency_dim=True,
            )


class TestParticipantLabels:
    """Subject-index -> 3-digit participant mapping for per-subject plots."""

    @staticmethod
    def _metadata(participant_ids, person_indices=None):
        df = pd.DataFrame({SingleDataMetadata.PARTICIPANT_ID: participant_ids})
        if person_indices is not None:
            df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = person_indices
        return df

    @pytest.mark.parametrize(
        "participant_id,expected",
        [
            ("031", "031"),
            (31, "031"),  # sidecar round-trip drops the zero padding
            ("19", "019"),
            ("PSI019", "019"),  # a PSI prefix on the input is stripped
            ("PSI019_EEGA_ASSR.edf", "019"),
        ],
    )
    def test_participant_label_formats(self, participant_id, expected):
        assert participant_label(participant_id) == expected

    def test_uses_concatenated_person_index_not_row_order(self):
        # Rows deliberately out of index order: the mapping must follow
        # CONCATENATED_PERSON_INDEX, not the DataFrame's row order.
        df = self._metadata(["031", "019", "024"], person_indices=[2, 0, 1])
        assert participant_labels(df, 3) == ["019", "024", "031"]

    def test_subset_returns_only_requested_subjects(self):
        df = self._metadata(["031", "019", "024"], person_indices=[0, 1, 2])
        assert participant_labels(df, 2) == ["031", "019"]

    def test_falls_back_to_row_order_without_person_index(self):
        df = self._metadata(["031", "019"])
        assert participant_labels(df, 2) == ["031", "019"]

    def test_raises_when_metadata_missing(self):
        with pytest.raises(ValueError, match="No participant metadata"):
            participant_labels(None, 2)

    def test_raises_when_participant_id_column_missing(self):
        df = pd.DataFrame({SingleDataMetadata.CONCATENATED_PERSON_INDEX: [0, 1]})
        with pytest.raises(ValueError, match="no PARTICIPANT_ID column"):
            participant_labels(df, 2)

    def test_raises_when_metadata_covers_fewer_subjects_than_data(self):
        df = self._metadata(["031"], person_indices=[0])
        with pytest.raises(ValueError, match="covers 1 recording"):
            participant_labels(df, 2)

    def test_raises_when_person_index_does_not_cover_every_subject(self):
        df = self._metadata(["031", "019"], person_indices=[0, 5])
        with pytest.raises(ValueError, match="missing subject index"):
            participant_labels(df, 2)
