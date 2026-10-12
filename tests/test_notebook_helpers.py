"""
Tests for scripts/notebook_helpers.py — how wavelet datasets are computed or loaded.

``compute_wavelet_datasets`` decides one thing on its own: whether a stimulus
experiment needs the pre-alignment transform first. That decision is pinned with the
expensive steps stubbed; the read itself is exercised against a real per-subject store.
"""

import numpy as np
import pytest

import scripts.notebook_helpers as nh
from src.analysis.data_representations import AnalysisData, DataRepresentation, subset
from src.definitions.fields import ExperimentNames

FREQS = np.linspace(4.0, 8.0, 3)
LABEL = "Placebo_CLASSIC"


def _time_domain(n_subjects=3, n_channels=4, n_times=120) -> AnalysisData:
    rng = np.random.default_rng(0)
    return AnalysisData(
        data=rng.normal(size=(n_subjects, n_channels, n_times)),
        sfreq=120.0,
        representation=DataRepresentation.TIME_DOMAIN,
        label=LABEL,
        feature_names=[f"E{i + 1}" for i in range(n_channels)],
        metadata={"participants": [f"9{i:02d}" for i in range(n_subjects)]},
    )


class _FakeAnalyzer:
    concatenates_condition_tracks = False


@pytest.fixture
def calls(monkeypatch):
    """Stub both expensive steps, recording which ran and with what reuse flag."""
    record: list[tuple[str, object]] = []
    monkeypatch.setattr(
        nh,
        "precompute_pre_alignment_wavelet_cache",
        lambda *args, **kwargs: record.append(("precompute", None)),
    )
    monkeypatch.setattr(
        nh,
        "wavelet_transform",
        lambda datasets, *args, **kwargs: record.append(
            ("transform", kwargs["reuse_wavelets"])
        )
        or {},
    )
    return record


def _compute(tmp_path, experiment, reuse, dataset=None):
    dataset = dataset or _time_domain()
    return nh.compute_wavelet_datasets(
        {LABEL: dataset},
        {LABEL: _FakeAnalyzer()},
        FREQS,
        "power",
        wavelet_dir=tmp_path,
        reuse_wavelets=reuse,
        experiment_name=experiment,
    )


class TestPreAlignmentDecision:
    def test_stimulus_experiment_precomputes_then_reads(self, tmp_path, calls):
        _compute(tmp_path, ExperimentNames.ASSR, reuse=False)
        assert calls == [("precompute", None), ("transform", True)]

    def test_reuse_skips_the_precompute(self, tmp_path, calls):
        _compute(tmp_path, ExperimentNames.ASSR, reuse=True)
        assert calls == [("transform", True)]

    def test_music_never_precomputes(self, tmp_path, calls):
        _compute(tmp_path, ExperimentNames.PSILO_MUSIC, reuse=False)
        assert calls == [("transform", False)]


class TestStoreRoundTrip:
    def test_reads_back_at_the_requested_extent(self, tmp_path):
        full = _compute(tmp_path, ExperimentNames.PSILO_MUSIC, reuse=False)[LABEL]
        assert full.data.shape == (3, 4, len(FREQS), 120)

        part = subset(_time_domain(), n_items=2, n_features=3, n_samples=50)
        loaded = _compute(tmp_path, ExperimentNames.PSILO_MUSIC, True, part)[LABEL]
        np.testing.assert_allclose(loaded.data, full.data[:2, :3, :, :50])
        assert loaded.feature_names == ["E1", "E2", "E3"]
