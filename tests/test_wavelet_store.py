"""
Tests for src/io/wavelet_store.py — the per-subject wavelet store.

Pinned: a subject round-trips as float32 with its axes named, a read returns exactly the
requested participants and extent in the requested order, and the store refuses what
would silently mix data — another channel layout in one set, subjects of different
length, a time extent it does not hold.
"""

import numpy as np
import pytest

from src.definitions.fields import ExperimentNames
from src.io.wavelet_store import (
    find_wavelet_set,
    iter_wavelet_subjects,
    load_wavelet_subjects,
    wavelet_label_dir,
    write_wavelet_subject,
)

FREQS = np.linspace(1.0, 5.0, 5)
CHANNELS = ["E1", "E2", "E3"]


def _write(directory, participant, data, channels=CHANNELS, freqs=FREQS):
    return write_wavelet_subject(
        directory,
        participant,
        data,
        label="Placebo_ASSR",
        representation="power",
        freqs=freqs,
        n_cycles=freqs / 2,
        sfreq=250.0,
        channel_names=channels,
    )


def _subject(value: float, n_times: int = 40) -> np.ndarray:
    """A subject whose every entry encodes (value, channel, freq, time)."""
    c, f, t = np.meshgrid(np.arange(3), np.arange(5), np.arange(n_times), indexing="ij")
    return value * 1e4 + c * 1e3 + f * 1e2 + t


@pytest.fixture
def store(tmp_path):
    directory = wavelet_label_dir(tmp_path, ExperimentNames.ASSR, "Placebo_ASSR")
    for value, participant in enumerate(["903", "901", "902"]):
        _write(directory, participant, _subject(value))
    return find_wavelet_set(directory)


def test_layout_and_round_trip(tmp_path, store):
    assert store.directory == tmp_path / "assr" / "wavelets" / "Placebo_ASSR"
    assert store.participants == ["901", "902", "903"]
    assert store.channel_names == CHANNELS and store.sfreq == 250.0
    np.testing.assert_allclose(store.freqs, FREQS)
    array = store.open("903")
    assert array.dtype == np.float32
    np.testing.assert_allclose(array, _subject(0))
    assert not [p for p in store.directory.iterdir() if p.name.startswith(".")]


def test_load_returns_the_requested_extent_in_the_requested_order(store):
    data = load_wavelet_subjects(
        store, ["902", "903"], n_channels=2, freq_indices=[4, 1], n_times=10
    )
    assert data.shape == (2, 2, 2, 10)
    np.testing.assert_allclose(data[0], _subject(2)[:2][:, [4, 1], :10])
    np.testing.assert_allclose(data[1], _subject(0)[:2][:, [4, 1], :10])
    parallel = load_wavelet_subjects(
        store, ["902", "903"], n_channels=2, freq_indices=[4, 1], n_times=10, n_jobs=2
    )
    np.testing.assert_array_equal(parallel, data)


def test_iteration_yields_one_subject_at_a_time(store):
    names, arrays = zip(*iter_wavelet_subjects(store, ["901", "902"], n_times=5))
    assert names == ("901", "902")
    np.testing.assert_allclose(arrays[0], _subject(1)[:, :, :5])


def test_refuses_what_would_mix_data(tmp_path, store):
    with pytest.raises(FileNotFoundError, match="999"):
        load_wavelet_subjects(store, ["901", "999"])
    with pytest.raises(ValueError, match="exceeds"):
        load_wavelet_subjects(store, ["901"], n_times=41)
    with pytest.raises(ValueError, match="channel_names differs"):
        _write(store.directory, "904", _subject(4), channels=["E1", "E2", "X"])

    _write(store.directory, "905", _subject(5, n_times=30))
    with pytest.raises(ValueError, match="different time axes"):
        load_wavelet_subjects(store, ["901", "905"])


def test_grids_are_separate_sets(store):
    other = np.linspace(1.0, 9.0, 5)
    _write(store.directory, "901", _subject(9), freqs=other)
    assert find_wavelet_set(store.directory, freqs=FREQS).participants == [
        "901",
        "902",
        "903",
    ]
    assert find_wavelet_set(store.directory, freqs=other).participants == ["901"]
    with pytest.raises(FileNotFoundError):
        find_wavelet_set(store.directory, "phase")
