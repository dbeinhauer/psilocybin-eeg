"""
Tests for scripts/run_corrca_raw_weighted.py — raw voltages weighted by the stage-06b
CorrCA topographies, read as their PC1 and as their mean over all electrodes.

What is pinned: the weights (absolute value, zero off the fit); the order z-score ->
weight -> readout -> onset normalisation of the read-out signal (the mean curve equals
that pipeline written out by hand, the uniform PC1 direction is stage 00's, both
curves are zero before onset and blind to the weights' overall scale); and that the
PC1 sign alignment undoes a flipped recording.
"""

import numpy as np
import pytest

from scripts.run_assr_raw_pca import channel_pca_components
from scripts.run_corrca_raw_weighted import (
    abs_pattern_weights,
    align_recordings,
    binary_roi_weights,
    mean_onset_curve,
    pc1_onset_curve,
    zscored_trials,
)
from src.analysis.assr_variants import zscore_time

READ = ["E1", "E2", "E3", "E4"]
PRE, POST = 25, 75
ONSETS = np.arange(100, 2900, 150)
MASK = np.arange(PRE + POST) < PRE


@pytest.fixture
def recording():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3000)) * np.array([[3.0], [0.1], [10.0], [1]])
    # A shared evoked response with opposite signs on two electrodes (a dipole).
    for o in ONSETS:
        data[0, o + 10 : o + 30] += 6.0
        data[2, o + 10 : o + 30] -= 20.0
    return data


@pytest.fixture
def trials(recording):
    return zscored_trials(recording, ONSETS, PRE, POST)


def test_weights_are_abs_and_zero_off_the_fit():
    patterns = np.array([[2.0, -2.0], [0.0, -1.0]])
    weights = abs_pattern_weights(patterns, ["E3", "E1"], READ)
    np.testing.assert_allclose(weights, [[2, 0, 2, 0], [1, 0, 0, 0]])


def test_zero_pattern_is_refused():
    with pytest.raises(ValueError):
        abs_pattern_weights(np.zeros((1, 2)), ["E1", "E2"], READ)


def test_binary_roi_weights_mark_the_assr_electrodes():
    from src.io.loading import load_assr_electrodes

    channels = [*load_assr_electrodes()[:2], "E900", "E901"]
    np.testing.assert_allclose(binary_roi_weights(channels), [1.0, 1.0, 0.0, 0.0])


def test_mean_curve_is_the_pipeline_written_out(recording, trials):
    w = np.array([1.0, 0.5, 2.0, 0.0])
    signal = (w[:, None] * zscore_time(recording)).mean(axis=0)
    cut = np.stack([signal[o - PRE : o + POST] for o in ONSETS])
    base = cut[:, :PRE]
    sd = np.sqrt((base.std(axis=1, ddof=1) ** 2).mean())
    expected = ((cut - base.mean(axis=1, keepdims=True)) / sd).mean(axis=0)
    np.testing.assert_allclose(mean_onset_curve(trials, w, MASK), expected)


def test_uniform_pc1_direction_is_stage_00s(trials):
    _, loading, ratio = pc1_onset_curve(trials, np.ones(4), MASK)
    _, ref_loading, ref_ratio = channel_pca_components(trials.mean(axis=1), 1)
    np.testing.assert_allclose(np.abs(loading), np.abs(ref_loading[0]))
    assert ratio == pytest.approx(ref_ratio[0])


def test_curves_are_onset_normalised_and_blind_to_the_weights_scale(trials):
    weights = np.array([1.0, 0.5, 2.0, 0.0])
    pc1, loading, _ = pc1_onset_curve(trials, weights, MASK)
    mean = mean_onset_curve(trials, weights, MASK)
    for curve in (pc1, mean):
        np.testing.assert_allclose(curve[MASK].mean(), 0.0, atol=1e-12)
    # PC1 follows the dipole: its response stands out of the pre-onset noise.
    assert np.abs(pc1[PRE + 10 : PRE + 30]).mean() > 3 * pc1[MASK].std()
    assert loading[3] == pytest.approx(0.0, abs=1e-12)
    scaled_pc1, _, _ = pc1_onset_curve(trials, 7.0 * weights, MASK)
    np.testing.assert_allclose(np.abs(scaled_pc1), np.abs(pc1))
    np.testing.assert_allclose(mean_onset_curve(trials, 7.0 * weights, MASK), mean)


def test_alignment_unflips_a_sign_flipped_recording():
    rng = np.random.default_rng(2)
    loading = np.array([1.0, 0.5, -0.2, 0.8])
    loadings = loading + 0.05 * rng.standard_normal((3, 4))
    scores = rng.standard_normal((3, 10))
    loadings[1], scores[1] = -loadings[1], -scores[1]
    s, ld, consistency, _ = align_recordings(scores, loadings, np.ones(4), READ)
    assert consistency.n_agreeing == 3
    assert np.all(ld @ loading > 0)
    # Each recording's score is flipped with its loading.
    np.testing.assert_allclose(s * ld[:, :1], scores * loadings[:, :1])
