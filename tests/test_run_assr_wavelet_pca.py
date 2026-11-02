"""
Tests for scripts/run_assr_wavelet_pca.py — the trial-normalised TF maps of the
streaming pass.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import zscore

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_assr_wavelet_pca import stream_wavelet_pca  # noqa: E402
from src.analysis.assr_trials import baseline_tf_map  # noqa: E402

N_CHANNELS, N_FREQS, N_TIMES = 6, 4, 600
PRE, POST = 10, 60
ONSETS = np.arange(40, 520, 80)


@pytest.fixture
def store():
    rng = np.random.default_rng(0)
    power = {
        p: rng.gamma(2.0, 1.0, (N_CHANNELS, N_FREQS, N_TIMES)) for p in ("019", "023")
    }
    return SimpleNamespace(
        n_channels=N_CHANNELS,
        n_freqs=N_FREQS,
        open=lambda participant: power[participant],
    ), power


def _stream(wavelet_set, **kwargs):
    epoch_times = np.arange(-PRE, POST)
    return stream_wavelet_pca(
        wavelet_set,
        ["019", "023"],
        ONSETS,
        PRE,
        POST,
        2,
        epoch_times >= 0,
        epoch_times < 0,
        **kwargs,
    )


def test_maps_are_the_projected_zscored_power_trial_normalised(store):
    wavelet_set, power = store
    _maps, loadings, _ev, _n, reference, tf_baseline = _stream(
        wavelet_set, trial_normalised=True
    )
    base_mask = np.arange(-PRE, POST) < 0
    for subj, participant in enumerate(["019", "023"]):
        z = zscore(power[participant], axis=2)
        sources = np.tensordot(loadings[:, subj], z, axes=(1, 0))  # (K, F, T)
        np.testing.assert_allclose(
            tf_baseline[:, subj], baseline_tf_map(sources, ONSETS, PRE, POST, base_mask)
        )
        np.testing.assert_allclose(
            reference.chan_mean_tf_baseline[subj],
            baseline_tf_map(z.mean(axis=0), ONSETS, PRE, POST, base_mask),
        )


def test_maps_are_only_built_when_asked_for(store):
    wavelet_set, _power = store
    *_rest, reference, tf_baseline = _stream(wavelet_set)
    assert tf_baseline is None
    assert reference.chan_mean_tf_baseline is None


def test_a_flipped_component_flips_its_map():
    # The polarity step applies the loading's sign to the trial-normalised map.
    sources = np.random.default_rng(1).standard_normal((N_FREQS, N_TIMES))
    base_mask = np.arange(-PRE, POST) < 0
    np.testing.assert_allclose(
        baseline_tf_map(-sources, ONSETS, PRE, POST, base_mask),
        -baseline_tf_map(sources, ONSETS, PRE, POST, base_mask),
    )
