"""
Tests for src/analysis/iva_quality.py — channel-IVA decomposition-quality scoring.

The compute half is pure, so everything here is exercised on small synthetic
arrays with a known answer.
"""

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning

from src.analysis.iva_quality import (
    ASSR_FREQ,
    RESP_DURATION_S,
    IvaQualityResult,
    boxcar,
    boxcar_correlation,
    compute_iva_quality,
    epoch_average,
    onset_average,
    onset_window,
    quality_score,
    reference_topomap,
    response_duration_samples,
    sign_agreement,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

SFREQ = 100.0
N_SUBJECTS = 4
N_CHANNELS = 6
N_FREQS = 5
N_TIMES = 600


@pytest.fixture
def onsets() -> np.ndarray:
    """Evenly spaced onsets that all fit inside ``N_TIMES``."""
    return np.arange(20, N_TIMES - 130, 60)


@pytest.fixture
def ch_names() -> list[str]:
    return [f"CH{i}" for i in range(N_CHANNELS)]


#: Topography planted in both the raw evoked response and component 0, so the
#: reference and that component agree on the x-axis.
_RESPONSE_TOPO = np.linspace(1.0, -1.0, N_CHANNELS)


def _quality_inputs(onsets: np.ndarray, seed: int = 0):
    """``(iva_components, iva_sources, freqs, raw_voltage)`` with a planted response.

    Component 0 is the "good" one: its channel pattern matches the evoked
    topography *and* its source responds for 50 samples after every onset, so it
    should score near ``(1, 1)``. The other two are noise.
    """
    rng = np.random.default_rng(seed)
    components = rng.standard_normal((N_SUBJECTS, 3, N_CHANNELS))
    components[:, 0, :] = _RESPONSE_TOPO + 0.01 * rng.standard_normal(
        (N_SUBJECTS, N_CHANNELS)
    )
    sources = rng.standard_normal((N_SUBJECTS, 3, N_FREQS, N_TIMES)) * 0.1
    # Component 0 responds for 50 samples after every onset, in every subject.
    for o in onsets:
        sources[:, 0, :, o : o + 50] += 3.0
    raw = rng.standard_normal((N_SUBJECTS, N_CHANNELS, N_TIMES))
    for o in onsets:
        raw[:, :, o : o + 50] += _RESPONSE_TOPO[None, :, None]
    freqs = np.linspace(1.0, 50.0, N_FREQS)
    return components, sources, freqs, raw


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------


class TestEpochAverage:
    def test_averages_constant_windows(self) -> None:
        arr = np.zeros((2, 100))
        arr[:, 10:20] = 5.0
        arr[:, 40:50] = 5.0
        avg, n_used = epoch_average(arr, np.array([10, 40]), 0, 10)
        assert n_used == 2
        np.testing.assert_allclose(avg, 5.0)

    def test_preserves_leading_axes(self) -> None:
        arr = np.arange(3 * 4 * 100, dtype=float).reshape(3, 4, 100)
        avg, _ = epoch_average(arr, np.array([20, 50]), 5, 5)
        assert avg.shape == (3, 4, 10)

    def test_drops_out_of_range_windows(self) -> None:
        arr = np.ones((1, 50))
        # Only the middle onset has a full window on both sides.
        _avg, n_used = epoch_average(arr, np.array([1, 25, 49]), 5, 5)
        assert n_used == 1

    def test_raises_when_nothing_fits(self) -> None:
        with pytest.raises(ValueError, match="No onset window fits"):
            epoch_average(np.ones((1, 10)), np.array([5]), 20, 20)


class TestOnsetWindow:
    def test_post_capped_by_shortest_gap(self) -> None:
        # Gaps of 30 samples are shorter than EPOCH_POST_S * sfreq (= 100).
        pre, post = onset_window(np.arange(0, 300, 30), N_TIMES, SFREQ)
        assert post == 30
        assert pre == 10  # EPOCH_PRE_S = 0.1 s at 100 Hz

    def test_uncapped_when_onsets_far_apart(self) -> None:
        _pre, post = onset_window(np.array([0, 500]), 1000, SFREQ)
        assert post == 100  # EPOCH_POST_S = 1.0 s at 100 Hz

    def test_ignores_onsets_past_the_time_axis(self) -> None:
        # The 5-sample gap lies beyond n_times and must not shrink the window.
        _pre, post = onset_window(np.array([10, 900, 905]), 500, SFREQ)
        assert post == 100


class TestOnsetAverage:
    def test_recovers_planted_response(self) -> None:
        x = np.zeros(300)
        onsets = np.array([50, 150, 250])
        for o in onsets:
            x[o : o + 10] = 2.0
        avg = onset_average(x, onsets, 5, 20, len(x))
        assert avg.shape == (25,)
        np.testing.assert_allclose(avg[:5], 0.0)
        np.testing.assert_allclose(avg[5:15], 2.0)

    def test_raises_when_nothing_fits(self) -> None:
        with pytest.raises(ValueError, match="No onset window fits"):
            onset_average(np.zeros(10), np.array([5]), 20, 20, 10)


# ---------------------------------------------------------------------------
# The boxcar reference
# ---------------------------------------------------------------------------


class TestBoxcar:
    def test_shape_and_placement(self) -> None:
        b = boxcar(pre=10, win=50, d_samples=20)
        assert b.shape == (50,)
        np.testing.assert_array_equal(b[:10], 0.0)
        np.testing.assert_array_equal(b[10:30], 1.0)
        np.testing.assert_array_equal(b[30:], 0.0)

    def test_response_duration_clamped_to_post(self) -> None:
        # RESP_DURATION_S * sfreq = 50 samples, but post is only 20.
        assert response_duration_samples(20, SFREQ) == 20
        assert response_duration_samples(500, SFREQ) == int(RESP_DURATION_S * SFREQ)
        assert response_duration_samples(0, SFREQ) == 1  # never below 1


class TestBoxcarCorrelation:
    def test_perfect_match_scores_one(self) -> None:
        pre, win, d = 10, 50, 20
        box = boxcar(pre, win, d)
        onset_avgs = np.tile(box, (3, 2, 1))  # (S=3, K=2, W)
        corr = boxcar_correlation(onset_avgs, pre, win, d)
        assert corr.shape == (3, 2)
        np.testing.assert_allclose(corr, 1.0, atol=1e-10)

    def test_inverted_response_scores_minus_one(self) -> None:
        pre, win, d = 10, 50, 20
        onset_avgs = np.tile(-boxcar(pre, win, d), (2, 1, 1))
        np.testing.assert_allclose(
            boxcar_correlation(onset_avgs, pre, win, d), -1.0, atol=1e-10
        )

    def test_flat_response_scores_zero_not_nan(self) -> None:
        # A constant series has zero variance -> undefined r, mapped to 0.
        onset_avgs = np.ones((2, 1, 50))
        with pytest.warns(ConstantInputWarning):
            corr = boxcar_correlation(onset_avgs, 10, 50, 20)
        assert not np.isnan(corr).any()
        np.testing.assert_allclose(corr, 0.0)


# ---------------------------------------------------------------------------
# Reference topography
# ---------------------------------------------------------------------------


class TestReferenceTopomap:
    def test_shape_matches_iva_subset(self, onsets, ch_names) -> None:
        _c, _s, _f, raw = _quality_inputs(onsets)
        subset = ch_names[:4]
        ref, _anchor = reference_topomap(raw, onsets, ch_names, subset, SFREQ)
        assert ref.shape == (4,)

    def test_selects_channels_by_name_not_position(self, onsets, ch_names) -> None:
        # Reversing the requested order must reverse the returned values.
        _c, _s, _f, raw = _quality_inputs(onsets)
        forward, _ = reference_topomap(raw, onsets, ch_names, ch_names, SFREQ)
        reverse, _ = reference_topomap(raw, onsets, ch_names, ch_names[::-1], SFREQ)
        np.testing.assert_allclose(reverse, forward[::-1])

    def test_anchor_falls_back_when_channel_absent(self, onsets, ch_names) -> None:
        _c, _s, _f, raw = _quality_inputs(onsets)
        _ref, anchor = reference_topomap(
            raw, onsets, ch_names, ch_names, SFREQ, polarity_channel="Cz"
        )
        assert anchor in ch_names  # "Cz" is absent -> largest-magnitude channel

    def test_anchor_used_when_present(self, onsets) -> None:
        names = ["CH0", "Cz", "CH2"]
        raw = np.random.default_rng(1).standard_normal((N_SUBJECTS, 3, N_TIMES))
        _ref, anchor = reference_topomap(raw, onsets, names, names, SFREQ)
        assert anchor == "Cz"

    def test_rejects_channel_count_mismatch(self, onsets, ch_names) -> None:
        _c, _s, _f, raw = _quality_inputs(onsets)
        with pytest.raises(ValueError, match="raw voltage channels"):
            reference_topomap(raw, onsets, ch_names[:-1], ch_names[:-1], SFREQ)

    def test_rejects_unknown_iva_channel(self, onsets, ch_names) -> None:
        _c, _s, _f, raw = _quality_inputs(onsets)
        with pytest.raises(ValueError, match="missing from the reference"):
            reference_topomap(raw, onsets, ch_names, ["NOPE"], SFREQ)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestSignAgreement:
    def test_unanimous_is_one(self) -> None:
        np.testing.assert_allclose(sign_agreement(np.ones((5, 2))), 1.0)

    def test_even_split_is_half(self) -> None:
        corr = np.array([[1.0], [1.0], [-1.0], [-1.0]])
        # Mean is 0 -> sign(0) falls back to +1, so the two positives agree.
        np.testing.assert_allclose(sign_agreement(corr), 0.5)

    def test_minority_flip_reported(self) -> None:
        corr = np.array([[1.0], [1.0], [1.0], [-1.0]])
        np.testing.assert_allclose(sign_agreement(corr), 0.75)


class TestQualityScore:
    def test_ideal_corner_scores_one(self) -> None:
        ones = np.ones((3, 2))
        assert quality_score(ones, ones, 0) == pytest.approx(1.0)

    def test_averages_both_axes(self) -> None:
        topo = np.array([[1.0], [1.0]])
        time = np.array([[0.0], [0.0]])
        assert quality_score(topo, time, 0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end compute
# ---------------------------------------------------------------------------


class TestComputeIvaQuality:
    def test_shapes_and_types(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        assert isinstance(res, IvaQualityResult)
        n_comp = comps.shape[1]
        win = res.n_epoch_pre + res.n_epoch_post
        assert res.ref_topo.shape == (N_CHANNELS,)
        assert res.patterns.shape == comps.shape
        assert res.topo_corr.shape == (N_SUBJECTS, n_comp)
        assert res.time_corr_pca.shape == (N_SUBJECTS, n_comp)
        assert res.onset_avg_pca.shape == (N_SUBJECTS, n_comp, win)
        assert res.epoch_times.shape == (win,)
        assert res.n_subjects == N_SUBJECTS
        assert res.n_components == n_comp

    def test_planted_component_wins_both_axes(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        # Component 0 carries both the matching topography and the onset-locked
        # response, so it must win on x and y and land near the (1, 1) corner.
        mean_topo = res.topo_corr.mean(axis=0)
        mean_time = res.time_corr_pca.mean(axis=0)
        assert int(np.argmax(mean_topo)) == 0
        assert int(np.argmax(mean_time)) == 0
        assert mean_topo[0] > 0.9
        assert mean_time[0] > 0.9
        assert quality_score(res.topo_corr, res.time_corr_pca, 0) > 0.9

    def test_noise_components_score_below_the_planted_one(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        scores = [
            quality_score(res.topo_corr, res.time_corr_pca, k)
            for k in range(comps.shape[1])
        ]
        assert scores[0] == max(scores)
        assert all(s < scores[0] - 0.3 for s in scores[1:])

    def test_epoch_times_zero_at_onset(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        assert res.epoch_times[res.n_epoch_pre] == pytest.approx(0.0)

    def test_orientation_makes_mean_topo_corr_non_negative(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        assert (res.topo_corr.mean(axis=0) >= 0).all()
        assert set(np.unique(res.sign_per_comp)) <= {-1.0, 1.0}

    def test_patterns_oriented_consistently_with_scores(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        # The stored patterns are the inputs times the same per-component sign,
        # then rescaled to unit L2 norm per (subject, component) so participants
        # stay comparable under the shared colour limit of the topomap figures.
        oriented = comps * res.sign_per_comp[np.newaxis, :, np.newaxis]
        np.testing.assert_allclose(
            res.patterns,
            oriented / np.linalg.norm(oriented, axis=2)[:, :, np.newaxis],
        )
        np.testing.assert_allclose(np.linalg.norm(res.patterns, axis=2), 1.0)
        # ...so recomputing the correlation from them reproduces topo_corr.
        for s in range(N_SUBJECTS):
            for k in range(comps.shape[1]):
                r = np.corrcoef(res.patterns[s, k], res.ref_topo)[0, 1]
                assert r == pytest.approx(res.topo_corr[s, k], abs=1e-8)

    def test_40hz_variant_flagged_off_outside_range(self, onsets, ch_names) -> None:
        comps, sources, _freqs, raw = _quality_inputs(onsets)
        low = np.linspace(1.0, 10.0, N_FREQS)  # ASSR_FREQ is outside
        res = compute_iva_quality(
            comps, sources, low, raw, onsets, SFREQ, ch_names, ch_names
        )
        assert res.have_40hz is False
        np.testing.assert_allclose(res.time_corr_40, 0.0)
        assert [tag for *_rest, tag in res.variants()] == ["pca"]

    def test_40hz_variant_present_in_range(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        assert freqs.min() <= ASSR_FREQ <= freqs.max()
        res = compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        assert res.have_40hz is True
        assert [tag for *_rest, tag in res.variants()] == ["pca", "40hz"]

    def test_input_not_mutated(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        c0, s0, r0 = comps.copy(), sources.copy(), raw.copy()
        compute_iva_quality(
            comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names
        )
        np.testing.assert_array_equal(comps, c0)
        np.testing.assert_array_equal(sources, s0)
        np.testing.assert_array_equal(raw, r0)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda c, s, f: (c[0], s, f), "iva_components must be"),
            (lambda c, s, f: (c, s[0], f), "iva_sources must be"),
            (lambda c, s, f: (c, s[:, :-1], f), "do not match"),
            (lambda c, s, f: (c, s, f[:-1]), "frequency bins"),
        ],
    )
    def test_rejects_inconsistent_shapes(self, onsets, ch_names, mutate, match) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        c, s, f = mutate(comps, sources, freqs)
        with pytest.raises(ValueError, match=match):
            compute_iva_quality(c, s, f, raw, onsets, SFREQ, ch_names, ch_names)

    def test_rejects_channel_name_count_mismatch(self, onsets, ch_names) -> None:
        comps, sources, freqs, raw = _quality_inputs(onsets)
        with pytest.raises(ValueError, match="channel names were given"):
            compute_iva_quality(
                comps, sources, freqs, raw, onsets, SFREQ, ch_names, ch_names[:-1]
            )
