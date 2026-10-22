"""Tests for src/analysis/assr_variants.py — reading a store on the cache's channels."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.assr_variants import (
    project_condition,
    store_channel_index,
    zscore_time,
)


def _cache(names):
    return SimpleNamespace(channel_names=names, directory=Path("cache"))


def test_a_store_on_the_cache_s_own_channels_needs_no_index():
    assert store_channel_index(_cache(["E1", "E2"]), ["E1", "E2"]) is None


def test_a_subset_store_reads_its_channels_in_its_own_order():
    index = store_channel_index(_cache(["E1", "E2", "E3", "E4"]), ["E4", "E2"])
    np.testing.assert_array_equal(index, [3, 1])


def test_a_store_channel_missing_from_the_cache_is_refused():
    with pytest.raises(ValueError, match="E9"):
        store_channel_index(_cache(["E1", "E2"]), ["E1", "E9"])


# ---------------------------------------------------------------------------
# project_condition: the complement filter and the full-frequency projections
# ---------------------------------------------------------------------------

N_CHANNELS, N_FREQS, N_TIMES, N_COMPONENTS = 6, 8, 40, 2
CHANNELS = [f"E{i + 1}" for i in range(N_CHANNELS)]
PARTICIPANTS = ["001", "002"]
GRID_BINS = np.array([3, 4])  # the 40 Hz selections
ROI = np.array([True, True, False, False, True, False])


@pytest.fixture
def cache(monkeypatch):
    rng = np.random.default_rng(0)
    arrays = {
        p: rng.random((N_CHANNELS, N_FREQS, N_TIMES)).astype(np.float32)
        for p in PARTICIPANTS
    }

    def iterate(wavelet_set, participants, *, freq_indices, n_times):
        for p in participants:
            yield p, arrays[p][:, freq_indices, :n_times]

    monkeypatch.setattr("src.analysis.assr_variants.iter_wavelet_subjects", iterate)
    wavelet_set = SimpleNamespace(
        channel_names=CHANNELS,
        directory=Path("cache"),
        n_freqs=N_FREQS,
        open=lambda p: arrays[p],
    )
    return wavelet_set, arrays


@pytest.fixture
def filters():
    return np.random.default_rng(1).standard_normal((N_COMPONENTS, N_CHANNELS))


def _project(wavelet_set, filters, **kwargs):
    binary = ROI / ROI.sum()
    return project_condition(
        wavelet_set,
        "Placebo",
        PARTICIPANTS,
        lambda participant, condition: 0,
        binary,
        binary[None, :],
        GRID_BINS,
        CHANNELS,
        None,
        np.flatnonzero(ROI),
        (filters * ROI)[None],
        **kwargs,
    )


def test_asking_for_tf_bins_leaves_the_selection_rows_unchanged(cache, filters):
    wavelet_set, _ = cache
    plain = _project(wavelet_set, filters)
    wide = _project(wavelet_set, filters, tf_freq_indices=np.arange(N_FREQS))
    for name in ("reference_raw", "pca_z", "roi_raw", "masked_z"):
        np.testing.assert_allclose(
            getattr(wide, name), getattr(plain, name), rtol=1e-5, err_msg=name
        )
    assert plain.masked_tf_z is None and plain.outside_tf_z is None


def test_the_two_masked_filters_add_up_to_the_full_one(cache, filters):
    wavelet_set, arrays = cache
    projected = _project(
        wavelet_set,
        filters,
        outside_filters=(filters * ~ROI)[None],
        tf_freq_indices=np.arange(N_FREQS),
    )
    for row, p in enumerate(PARTICIPANTS):
        full = np.tensordot(filters, zscore_time(arrays[p]), axes=([1], [0]))
        np.testing.assert_allclose(
            projected.masked_tf_z[row] + projected.outside_tf_z[row],
            full,
            rtol=1e-4,
            atol=1e-4,
        )
        # The full-frequency rows hold the selection rows at those bins.
        np.testing.assert_allclose(
            projected.outside_tf_z[row][:, GRID_BINS],
            projected.outside_z[row],
            rtol=1e-5,
        )
