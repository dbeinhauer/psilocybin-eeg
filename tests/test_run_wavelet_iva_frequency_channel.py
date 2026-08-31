"""
Tests for scripts/run_wavelet_iva_frequency_channel.py — the frequency-channel IVA CLI.

Scoped to the component store: figure content and the plot layout are not asserted
here. What is pinned is that this variant stores what it actually recovers — a
timecourse and a ``(frequencies, channels)`` pattern, and **no** time-frequency map,
which only the channel-as-mixing variant produces — and that it refuses to store
anything it could not label by participant.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.run_wavelet_iva_frequency_channel import _run_iva  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import list_iva_results, load_iva_components  # noqa: E402

SFREQ = 100.0
N_SUBJECTS = 3
N_CHANNELS = 6
N_FREQS = 4
N_TIMES = 200
N_PCA = 3
VARIANT = IvaVariants.FREQUENCY_CHANNEL


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def info():
    mne.set_log_level("ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:N_CHANNELS], sfreq=SFREQ, ch_types="eeg")
    info.set_montage(montage)
    return info


@pytest.fixture
def data_4d():
    return np.random.default_rng(0).standard_normal(
        (N_SUBJECTS, N_CHANNELS, N_FREQS, N_TIMES)
    )


def _run(info, data_4d, tmp_path, **kwargs):
    """Drive the pipeline on tiny synthetic data, overriding any argument."""
    params = {
        "n_pca": N_PCA,
        "n_top": 2,
        "n_bottom": 1,
        "random_state": 42,
        "iva_opt_approach": "newton",
        "iva_max_iter": 16,
        "iva_w_diff_stop": 1e-6,
        "save_dir": tmp_path / "plots",
        "band": None,
        "experiment_name": ExperimentNames.ASSR,
        "condition": ConditionVariants.PLACEBO,
        "music_type": MusicTypeVariants.ASSR,
        "subject_ids": [f"PSI{s:03d}" for s in range(N_SUBJECTS)],
    }
    params.update(kwargs)
    _run_iva(
        data_4d,
        SFREQ,
        np.linspace(10.0, 60.0, N_FREQS),
        info,
        label="Placebo_ASSR",
        **params,
    )


def _load(store, n_pca=N_PCA, band=None):
    return load_iva_components(
        experiment=ExperimentNames.ASSR,
        condition=ConditionVariants.PLACEBO,
        variant=VARIANT,
        music_type=MusicTypeVariants.ASSR,
        band=band,
        n_pca=n_pca,
        processed_data_dir=store,
    )


class TestComponentStore:
    def test_nothing_is_written_unless_asked(self, info, data_4d, tmp_path):
        store = tmp_path / "processed"
        _run(info, data_4d, tmp_path)
        assert list_iva_results(ExperimentNames.ASSR, processed_data_dir=store) == []

    def test_stores_the_products_this_variant_actually_has(
        self, info, data_4d, tmp_path
    ):
        store = tmp_path / "processed"
        _run(info, data_4d, tmp_path, store_root=store)
        loaded = _load(store)

        assert loaded.variant is VARIANT
        assert set(loaded.arrays) == {"frequency_channel_pattern", "timecourse"}
        assert loaded.array(IvaComponentArrays.TIMECOURSE).shape == (
            N_SUBJECTS,
            N_PCA,
            N_TIMES,
        )
        assert loaded.array(IvaComponentArrays.FREQUENCY_CHANNEL_PATTERN).shape == (
            N_SUBJECTS,
            N_PCA,
            N_FREQS,
            N_CHANNELS,
        )
        # Frequency is on the mixing axis here, so there is no TF map to store.
        assert not loaded.has(IvaComponentArrays.TF_MAP)
        with pytest.raises(KeyError, match="has no 'tf_map'"):
            loaded.tf_maps

    def test_the_participant_mapping_travels_with_the_arrays(
        self, info, data_4d, tmp_path
    ):
        store = tmp_path / "processed"
        _run(info, data_4d, tmp_path, store_root=store)
        loaded = _load(store)
        assert loaded.participants == tuple(f"PSI{s:03d}" for s in range(N_SUBJECTS))
        assert loaded.subject_conditions == ("Placebo",) * N_SUBJECTS
        assert loaded.row("PSI002") == 2
        assert list(loaded.channel_names) == list(info["ch_names"])
        assert loaded.extras["rank_score"].shape == (N_PCA,)

    def test_storing_without_participant_labels_is_refused(
        self, info, data_4d, tmp_path
    ):
        with pytest.raises(ValueError, match="without participant labels"):
            _run(
                info,
                data_4d,
                tmp_path,
                subject_ids=None,
                store_root=tmp_path / "processed",
            )

    def test_storing_without_a_run_descriptor_is_refused(self, info, data_4d, tmp_path):
        with pytest.raises(ValueError, match="condition"):
            _run(
                info,
                data_4d,
                tmp_path,
                condition=None,
                store_root=tmp_path / "processed",
            )
