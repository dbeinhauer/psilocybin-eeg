"""
Tests for scripts/notebook_helpers.py — the notebook-level wavelet subset cache.

The subset cache exists to stop the multi-GB source-of-truth cache from being read
more than once per extent, so the tests assert on *whether the expensive read
happened*, not only on the values that came back. ``wavelet_transform`` is stubbed to
count its calls and stand in for that read.
"""

import dataclasses

import numpy as np
import pytest

import scripts.notebook_helpers as nh
from src.analysis.data_representations import AnalysisData, DataRepresentation
from src.definitions.fields import ExperimentNames

FULL_CHANNELS, N_FREQS, FULL_TIMES = 8, 5, 60
FREQS = np.linspace(1.0, 50.0, N_FREQS)
LABEL = "Placebo_ASSR"


def _time_domain(n_subjects=3, n_channels=FULL_CHANNELS, n_times=FULL_TIMES):
    """The (possibly subset) time-domain dataset that defines the requested extent."""
    return AnalysisData(
        data=np.zeros((n_subjects, n_channels, n_times)),
        sfreq=250.0,
        representation=DataRepresentation.TIME_DOMAIN,
        label=LABEL,
        feature_names=[f"E{i + 1}" for i in range(n_channels)],
    )


class _FakeAnalyzer:
    """Just enough analyser for compute_wavelet_datasets' track-path check."""

    concatenates_condition_tracks = False


@pytest.fixture
def source_reads(monkeypatch):
    """Stub the source-of-truth read, recording every label it is asked for."""
    reads: list[str] = []

    def fake_wavelet_transform(datasets, freqs, representation, **kwargs):
        out = {}
        for label, ad in datasets.items():
            reads.append(label)
            n_subjects = ad.data.shape[0]
            # A value that identifies (subject, channel), so a mis-sliced or
            # mis-keyed cache is detectable rather than merely differently shaped.
            full = np.zeros((n_subjects, FULL_CHANNELS, len(freqs), FULL_TIMES))
            for s in range(n_subjects):
                full[s] = s * 1000 + np.arange(FULL_CHANNELS)[:, None, None]
            out[label] = AnalysisData(
                data=full,
                sfreq=250.0,
                representation=DataRepresentation.WAVELET_POWER,
                label=label,
                feature_names=[f"E{i + 1}" for i in range(FULL_CHANNELS)],
                metadata={
                    "freqs": freqs,
                    "n_cycles": freqs / 2.0,
                    "keep_frequency_dim": True,
                },
            )
        return out

    monkeypatch.setattr(nh, "wavelet_transform", fake_wavelet_transform)
    return reads


def _compute(cache_dir, reference_ad, *, reuse_subset_cache=True):
    return nh.compute_wavelet_datasets(
        {LABEL: reference_ad},
        {LABEL: _FakeAnalyzer()},
        FREQS,
        "power",
        wavelet_dir=cache_dir / "unused-source-dir",
        reuse_wavelets=True,
        experiment_name=ExperimentNames.ASSR,
        subset_cache_dir=cache_dir,
        reuse_subset_cache=reuse_subset_cache,
    )


class TestResolveNotebookWaveletCacheDir:
    def test_points_at_the_stage_03_notebook(self):
        resolved = nh.resolve_notebook_wavelet_cache_dir(ExperimentNames.ASSR)
        assert resolved.parts[-3:] == (
            "03-wavelet-analysis",
            "wavelet_cache",
            "assr",
        )

    def test_is_per_experiment(self):
        assr = nh.resolve_notebook_wavelet_cache_dir(ExperimentNames.ASSR)
        music = nh.resolve_notebook_wavelet_cache_dir(ExperimentNames.PSILO_MUSIC)
        assert assr != music
        assert assr.parent == music.parent

    def test_is_not_the_source_of_truth_cache(self):
        """The two caches must never resolve to the same directory."""
        from scripts.analysis_common import resolve_wavelet_dir

        assert nh.resolve_notebook_wavelet_cache_dir(
            ExperimentNames.ASSR
        ) != resolve_wavelet_dir(None, ExperimentNames.ASSR)


class TestWaveletSubsetCache:
    def test_first_run_reads_the_source_and_writes_the_cache(
        self, tmp_path, source_reads
    ):
        _compute(tmp_path, _time_domain(n_channels=4, n_times=20))
        assert source_reads == [LABEL]
        assert len(list(tmp_path.glob("*.npz"))) == 1

    def test_second_run_does_not_read_the_source(self, tmp_path, source_reads):
        """The whole point: a hit must not open the multi-GB cache."""
        reference = _time_domain(n_channels=4, n_times=20)
        first = _compute(tmp_path, reference)
        source_reads.clear()
        second = _compute(tmp_path, reference)
        assert source_reads == []
        np.testing.assert_array_equal(second[LABEL].data, first[LABEL].data)

    def test_cached_values_survive_the_round_trip(self, tmp_path, source_reads):
        reference = _time_domain(n_channels=4, n_times=20)
        _compute(tmp_path, reference)
        source_reads.clear()
        cached = _compute(tmp_path, reference)[LABEL]
        assert cached.data.shape == (3, 4, N_FREQS, 20)
        for subject in range(3):
            np.testing.assert_allclose(
                cached.data[subject, :, 0, 0], subject * 1000 + np.arange(4)
            )

    def test_cached_metadata_is_restored(self, tmp_path, source_reads):
        reference = _time_domain(n_channels=4, n_times=20)
        _compute(tmp_path, reference)
        source_reads.clear()
        cached = _compute(tmp_path, reference)[LABEL]
        assert cached.label == LABEL
        assert cached.sfreq == 250.0
        assert cached.representation is DataRepresentation.WAVELET_POWER
        assert cached.feature_names == ["E1", "E2", "E3", "E4"]
        np.testing.assert_allclose(cached.metadata["freqs"], FREQS)
        np.testing.assert_allclose(cached.metadata["n_cycles"], FREQS / 2.0)
        assert cached.metadata["loaded_from_subset_cache"].endswith(".npz")

    def test_a_different_extent_is_a_different_cache(self, tmp_path, source_reads):
        _compute(tmp_path, _time_domain(n_channels=4, n_times=20))
        source_reads.clear()
        _compute(tmp_path, _time_domain(n_channels=4, n_times=40))
        assert source_reads == [LABEL], "a different extent reused the wrong cache"
        assert len(list(tmp_path.glob("*.npz"))) == 2

    def test_a_different_channel_count_is_a_different_cache(
        self, tmp_path, source_reads
    ):
        _compute(tmp_path, _time_domain(n_channels=4, n_times=20))
        source_reads.clear()
        result = _compute(tmp_path, _time_domain(n_channels=6, n_times=20))
        assert source_reads == [LABEL]
        assert result[LABEL].data.shape[1] == 6

    def test_reuse_subset_cache_false_recomputes_and_overwrites(
        self, tmp_path, source_reads
    ):
        reference = _time_domain(n_channels=4, n_times=20)
        _compute(tmp_path, reference)
        source_reads.clear()
        _compute(tmp_path, reference, reuse_subset_cache=False)
        assert source_reads == [LABEL]
        assert len(list(tmp_path.glob("*.npz"))) == 1, (
            "the stale cache was not replaced"
        )

    def test_no_cache_dir_writes_nothing(self, tmp_path, source_reads):
        """The default must leave existing call-sites completely unchanged."""
        nh.compute_wavelet_datasets(
            {LABEL: _time_domain(n_channels=4, n_times=20)},
            {LABEL: _FakeAnalyzer()},
            FREQS,
            "power",
            wavelet_dir=tmp_path / "source",
            reuse_wavelets=True,
            experiment_name=ExperimentNames.ASSR,
        )
        assert list(tmp_path.glob("*.npz")) == []

    def test_no_temporary_files_are_left_behind(self, tmp_path, source_reads):
        _compute(tmp_path, _time_domain(n_channels=4, n_times=20))
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_the_cache_directory_is_created_on_demand(self, tmp_path, source_reads):
        nested = tmp_path / "wavelet_cache" / "assr" / "broadband"
        _compute(nested, _time_domain(n_channels=4, n_times=20))
        assert len(list(nested.glob("*.npz"))) == 1

    def test_partial_hits_only_recompute_the_misses(self, tmp_path, source_reads):
        """Two datasets, one already cached: only the uncached one is read."""
        other = "Psilocybin_ASSR"
        reference = _time_domain(n_channels=4, n_times=20)
        _compute(tmp_path, reference)  # caches Placebo_ASSR only
        source_reads.clear()

        result = nh.compute_wavelet_datasets(
            {LABEL: reference, other: dataclasses.replace(reference, label=other)},
            {LABEL: _FakeAnalyzer(), other: _FakeAnalyzer()},
            FREQS,
            "power",
            wavelet_dir=tmp_path / "unused-source-dir",
            reuse_wavelets=True,
            experiment_name=ExperimentNames.ASSR,
            subset_cache_dir=tmp_path,
        )
        assert source_reads == [other]
        assert list(result) == [LABEL, other], "caller key order was not preserved"

    def test_a_truncated_cache_is_not_silently_reused(self, tmp_path, source_reads):
        """An atomic write is what guarantees this; assert the guarantee holds."""
        reference = _time_domain(n_channels=4, n_times=20)
        _compute(tmp_path, reference)
        cache_file = next(tmp_path.glob("*.npz"))
        cache_file.write_bytes(cache_file.read_bytes()[:1024])
        with pytest.raises(Exception):
            _compute(tmp_path, reference)
