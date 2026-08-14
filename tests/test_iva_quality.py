"""
Tests for src/analysis/iva_quality.py — channel-IVA decomposition-quality scoring.

The compute half is pure, so everything here is exercised on small synthetic
arrays with a known answer.
"""

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning, pearsonr

from src.analysis.iva_quality import (
    ASSR_FREQ,
    RESP_DURATION_S,
    SUBJECT_SIGN_ANCHOR,
    TF_ANCHOR_HALFWIDTH_HZ,
    TF_BANDS,
    IvaQualityResult,
    anchor_signs_reference,
    anchor_signs_tf,
    boxcar,
    boxcar_correlation,
    compute_iva_quality,
    epoch_average,
    equalize_subject_influence,
    frequency_band_mask,
    full_tf_group_mean,
    onset_average,
    onset_window,
    quality_score,
    response_duration_samples,
    sign_agreement,
    subject_scales,
    tf_map_correlation,
    wavelet_reference,
)
from src.analysis.wavelet_ica import zscore_by_time

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


#: Topography planted in both the wavelet response and component 0, so the
#: reference and that component agree on the x-axis.
_RESPONSE_TOPO = np.linspace(1.0, -1.0, N_CHANNELS)


def _wavelet_inputs(onsets: np.ndarray, seed: int = 1) -> np.ndarray:
    """``(S, C, F, T)`` time-z-scored wavelet power carrying the same response.

    The onset-locked burst is scaled by :data:`_RESPONSE_TOPO` across channels, so
    the channel-PCA of the trial average recovers that topography as PC1 and a
    map peaking over the driven window as PC1's score.
    """
    rng = np.random.default_rng(seed)
    wav = rng.standard_normal((N_SUBJECTS, N_CHANNELS, N_FREQS, N_TIMES)) * 0.1
    for o in onsets:
        wav[:, :, :, o : o + 50] += _RESPONSE_TOPO[None, :, None, None]
    return zscore_by_time(wav)


def _quality_inputs(onsets: np.ndarray, seed: int = 0):
    """``(components, sources, freqs, wavelet)`` with a planted response.

    Component 0 is the "good" one: its channel pattern matches the topography the
    wavelet reference recovers *and* its source responds for 50 samples after
    every onset, so it should score near ``(1, 1)`` on every axis. The other two
    are noise.
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
    freqs = np.linspace(1.0, 50.0, N_FREQS)
    return components, sources, freqs, _wavelet_inputs(onsets)


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
# Wavelet reference
# ---------------------------------------------------------------------------

# Epoch used directly by the wavelet-reference tests (onset_window's values at
# SFREQ with the fixture's 60-sample gaps).
_PRE, _POST = 10, 60


class TestWaveletReference:
    def test_shapes(self, onsets, ch_names) -> None:
        wav = _wavelet_inputs(onsets)
        ref_topo, ref_tf, _anchor, _cons = wavelet_reference(
            wav, onsets, _PRE, _POST, ch_names
        )
        assert ref_topo.shape == (N_CHANNELS,)
        assert ref_tf.shape == (N_FREQS, _PRE + _POST)

    def test_recovers_the_planted_topography(self, onsets, ch_names) -> None:
        wav = _wavelet_inputs(onsets)
        ref_topo, _ref_tf, _anchor, cons = wavelet_reference(
            wav, onsets, _PRE, _POST, ch_names
        )
        # PC1 must be the planted spatial pattern, up to a global sign. Not an
        # exact match: the per-channel z-scoring divides each channel by its own
        # SD, which is exactly what compresses the planted amplitude ratios.
        r = np.corrcoef(ref_topo, _RESPONSE_TOPO)[0, 1]
        assert abs(r) > 0.9
        # ...and every subject must agree with the group once aligned.
        assert cons.n_agreeing == cons.n_subjects
        assert cons.median_pairwise_r > 0.9

    def test_tf_map_peaks_inside_the_driven_window(self, onsets, ch_names) -> None:
        wav = _wavelet_inputs(onsets)
        ref_topo, ref_tf, _anchor, _cons = wavelet_reference(
            wav, onsets, _PRE, _POST, ch_names
        )
        # Score and loading share one sign, so read the map in the loading's
        # orientation before asking where it is large.
        oriented = ref_tf * np.sign(np.corrcoef(ref_topo, _RESPONSE_TOPO)[0, 1])
        driven = oriented[:, _PRE : _PRE + 50].mean()
        baseline = oriented[:, :_PRE].mean()
        assert driven > baseline

    def test_anchor_used_when_present(self, onsets) -> None:
        names = ["CH0", "Cz", "CH2"]
        wav = zscore_by_time(
            np.random.default_rng(3).standard_normal((N_SUBJECTS, 3, N_FREQS, N_TIMES))
        )
        _t, _m, anchor, _c = wavelet_reference(wav, onsets, _PRE, _POST, names)
        assert anchor == "Cz"

    def test_input_not_mutated(self, onsets, ch_names) -> None:
        wav = _wavelet_inputs(onsets)
        before = wav.copy()
        wavelet_reference(wav, onsets, _PRE, _POST, ch_names)
        np.testing.assert_array_equal(wav, before)

    def test_rejects_wrong_ndim(self, onsets, ch_names) -> None:
        with pytest.raises(ValueError, match=r"must be \(S, C, F, T\)"):
            wavelet_reference(
                np.zeros((N_SUBJECTS, N_CHANNELS, N_TIMES)),
                onsets,
                _PRE,
                _POST,
                ch_names,
            )

    def test_rejects_channel_name_count_mismatch(self, onsets, ch_names) -> None:
        wav = _wavelet_inputs(onsets)
        with pytest.raises(ValueError, match="channel names were given"):
            wavelet_reference(wav, onsets, _PRE, _POST, ch_names[:-1])


class TestTfMapCorrelation:
    def test_perfect_match_scores_one(self) -> None:
        ref = np.random.default_rng(0).standard_normal((4, 20))
        onset_tf = np.tile(ref, (3, 2, 1, 1))  # (S=3, K=2, F, W)
        corr = tf_map_correlation(onset_tf, ref)
        assert corr.shape == (3, 2)
        np.testing.assert_allclose(corr, 1.0, atol=1e-10)

    def test_inverted_map_scores_minus_one(self) -> None:
        ref = np.random.default_rng(1).standard_normal((4, 20))
        onset_tf = np.tile(-ref, (2, 1, 1, 1))
        np.testing.assert_allclose(tf_map_correlation(onset_tf, ref), -1.0, atol=1e-10)

    def test_uses_the_whole_map_not_just_time(self) -> None:
        # Same time profile, wrong frequency row: the flattened correlation must
        # not reward it the way a frequency-collapsed score would.
        ref = np.zeros((4, 20))
        ref[0, 5:10] = 1.0
        wrong = np.zeros((1, 1, 4, 20))
        wrong[0, 0, 3, 5:10] = 1.0
        right = np.zeros((1, 1, 4, 20))
        right[0, 0, 0, 5:10] = 1.0
        assert tf_map_correlation(right, ref)[0, 0] > 0.99
        assert tf_map_correlation(wrong, ref)[0, 0] < 0.1

    def test_flat_map_scores_zero_not_nan(self) -> None:
        ref = np.random.default_rng(2).standard_normal((4, 20))
        with pytest.warns(ConstantInputWarning):
            corr = tf_map_correlation(np.ones((2, 1, 4, 20)), ref)
        assert not np.isnan(corr).any()
        np.testing.assert_allclose(corr, 0.0)

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match=r"must be \(S, K, F, W\)"):
            tf_map_correlation(np.zeros((2, 4, 20)), np.zeros((4, 20)))

    def test_rejects_map_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="do not match the reference"):
            tf_map_correlation(np.zeros((2, 1, 4, 20)), np.zeros((4, 19)))


class TestFrequencyBandMask:
    def test_selects_the_bins_inside_the_band(self) -> None:
        freqs = np.array([1.0, 5.0, 12.0, 35.0, 45.0, 60.0])
        np.testing.assert_array_equal(
            frequency_band_mask(freqs, 30.0, 50.0),
            [False, False, False, True, True, False],
        )

    def test_edges_are_inclusive(self) -> None:
        freqs = np.array([1.0, 10.0, 11.0])
        np.testing.assert_array_equal(
            frequency_band_mask(freqs, 1.0, 10.0), [True, True, False]
        )

    def test_band_outside_the_range_selects_nothing(self) -> None:
        # A legitimate answer, not an error: the caller skips that band.
        assert not frequency_band_mask(np.linspace(8.0, 13.0, 5), 30.0, 50.0).any()

    def test_rejects_inverted_band(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            frequency_band_mask(np.linspace(1.0, 50.0, 5), 50.0, 30.0)


class TestBandLimitedTfMapCorrelation:
    """``freq_mask`` scores one frequency band instead of the whole map."""

    @staticmethod
    def _split_band_maps():
        """Reference and one component agreeing at low freqs, opposed at high ones."""
        freqs = np.array([2.0, 5.0, 35.0, 45.0])
        ref = np.zeros((4, 20))
        ref[:2, 5:10] = 1.0  # low band: a burst
        ref[2:, 5:10] = 1.0  # high band: the same burst
        comp = np.zeros((1, 1, 4, 20))
        comp[0, 0, :2, 5:10] = 1.0  # low band: matches
        comp[0, 0, 2:, 5:10] = -1.0  # high band: inverted
        return freqs, ref, comp

    def test_band_score_ignores_the_other_band(self) -> None:
        freqs, ref, comp = self._split_band_maps()
        low = tf_map_correlation(comp, ref, frequency_band_mask(freqs, 1.0, 10.0))
        high = tf_map_correlation(comp, ref, frequency_band_mask(freqs, 30.0, 50.0))
        assert low[0, 0] > 0.99
        assert high[0, 0] < -0.99

    def test_whole_map_score_sits_between_the_bands(self) -> None:
        # The point of the band variants: a component matching in one band only is
        # diluted by the rest of the spectrum when the whole map is scored.
        freqs, ref, comp = self._split_band_maps()
        whole = tf_map_correlation(comp, ref)[0, 0]
        low = tf_map_correlation(comp, ref, frequency_band_mask(freqs, 1.0, 10.0))[0, 0]
        assert whole == pytest.approx(0.0, abs=1e-10)
        assert low > whole

    def test_all_true_mask_matches_the_whole_map(self) -> None:
        ref = np.random.default_rng(3).standard_normal((4, 20))
        onset_tf = np.random.default_rng(4).standard_normal((2, 3, 4, 20))
        np.testing.assert_allclose(
            tf_map_correlation(onset_tf, ref, np.ones(4, dtype=bool)),
            tf_map_correlation(onset_tf, ref),
        )

    def test_rejects_mask_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="does not match the"):
            tf_map_correlation(
                np.zeros((1, 1, 4, 20)), np.zeros((4, 20)), np.ones(3, dtype=bool)
            )

    def test_rejects_empty_mask(self) -> None:
        with pytest.raises(ValueError, match="selects no frequency bin"):
            tf_map_correlation(
                np.zeros((1, 1, 4, 20)), np.zeros((4, 20)), np.zeros(4, dtype=bool)
            )


# ---------------------------------------------------------------------------
# Per-participant sign anchoring
# ---------------------------------------------------------------------------


class TestAnchorSignsReference:
    """The one per-pair flip, from the topography against the reference."""

    def test_flips_participants_that_anti_correlate_with_the_reference(self) -> None:
        ref = np.linspace(1.0, -1.0, N_CHANNELS)
        patterns = np.stack([[ref, -ref], [-ref, -ref], [ref, ref]])  # (3, 2, C)
        signs = anchor_signs_reference(patterns, ref)
        np.testing.assert_array_equal(signs, [[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
        # Applying them makes every pair agree with the reference.
        aligned = patterns * signs[:, :, np.newaxis]
        for s in range(aligned.shape[0]):
            for k in range(aligned.shape[1]):
                assert np.corrcoef(aligned[s, k], ref)[0, 1] > 0

    def test_ignores_per_pair_scaling(self) -> None:
        # A correlation's sign survives any positive rescaling, so unit-norming
        # the patterns beforehand cannot change the answer.
        ref = np.linspace(1.0, -1.0, N_CHANNELS)
        patterns = np.stack([[ref], [-ref]])  # (2, 1, C)
        scaled = patterns * np.array([[10.0], [0.01]])[:, :, np.newaxis]
        np.testing.assert_array_equal(
            anchor_signs_reference(scaled, ref), anchor_signs_reference(patterns, ref)
        )

    def test_follows_the_reference_rather_than_a_fixed_rule(self) -> None:
        # Inverting the reference inverts every sign: the participants agree with
        # the reference, whatever polarity it happens to carry.
        ref = np.linspace(1.0, -1.0, N_CHANNELS)
        patterns = np.stack([[ref], [-ref]])
        np.testing.assert_array_equal(
            anchor_signs_reference(patterns, -ref),
            -anchor_signs_reference(patterns, ref),
        )

    def test_undefined_correlation_keeps_the_pattern_as_is(self) -> None:
        # A constant pattern has no correlation with anything; +1 leaves it alone.
        with pytest.warns(ConstantInputWarning):
            signs = anchor_signs_reference(
                np.zeros((2, 1, N_CHANNELS)), np.linspace(1.0, -1.0, N_CHANNELS)
            )
        np.testing.assert_array_equal(signs, np.ones((2, 1)))

    def test_rejects_shape_mismatches(self) -> None:
        with pytest.raises(ValueError, match=r"must be \(S, K, C\)"):
            anchor_signs_reference(np.zeros((2, 3)), np.zeros(N_CHANNELS))
        with pytest.raises(ValueError, match="ref_topo shape"):
            anchor_signs_reference(
                np.zeros((2, 1, N_CHANNELS)), np.zeros(N_CHANNELS - 1)
            )


class TestAnchorSignsTf:
    """Per-participant flips from the driven response in the stimulus window."""

    @staticmethod
    def _maps(freqs: np.ndarray, times: np.ndarray) -> np.ndarray:
        """``(2, 1, F, W)``: a 40 Hz burst in the window, inverted for subject 1."""
        maps = np.zeros((2, 1, len(freqs), len(times)))
        driven = int(np.argmin(np.abs(freqs - ASSR_FREQ)))
        window = (times >= 0.0) & (times <= RESP_DURATION_S)
        maps[0, 0, driven, window] = 1.0
        maps[1, 0, driven, window] = -1.0
        return maps

    def test_flips_the_inverted_participant(self) -> None:
        freqs = np.linspace(1.0, 60.0, 12)
        times = np.linspace(-0.1, 1.0, 60)
        signs, have = anchor_signs_tf(self._maps(freqs, times), freqs, times)
        assert have is True
        np.testing.assert_array_equal(signs, [[1.0], [-1.0]])

    def test_ignores_activity_outside_the_stimulus_window(self) -> None:
        # A post-stimulus 40 Hz burst of the opposite sign must not decide the flip.
        freqs = np.linspace(1.0, 60.0, 12)
        times = np.linspace(-0.1, 1.0, 60)
        maps = self._maps(freqs, times)
        driven = int(np.argmin(np.abs(freqs - ASSR_FREQ)))
        maps[:, :, driven, times > RESP_DURATION_S] = -5.0
        signs, _have = anchor_signs_tf(maps, freqs, times)
        np.testing.assert_array_equal(signs, [[1.0], [-1.0]])

    def test_ignores_activity_outside_the_driven_band(self) -> None:
        freqs = np.linspace(1.0, 60.0, 12)
        times = np.linspace(-0.1, 1.0, 60)
        maps = self._maps(freqs, times)
        maps[:, :, 0, :] = -5.0  # a strong 1 Hz component of the opposite sign
        signs, _have = anchor_signs_tf(maps, freqs, times)
        np.testing.assert_array_equal(signs, [[1.0], [-1.0]])

    def test_reports_no_anchor_when_the_band_is_out_of_range(self) -> None:
        alpha = np.linspace(8.0, 13.0, 5)
        times = np.linspace(-0.1, 1.0, 60)
        signs, have = anchor_signs_tf(
            np.ones((2, 1, len(alpha), len(times))), alpha, times
        )
        assert have is False
        np.testing.assert_array_equal(signs, np.ones((2, 1)))

    def test_rejects_axis_mismatches(self) -> None:
        freqs, times = np.linspace(1.0, 60.0, 4), np.linspace(-0.1, 0.5, 10)
        with pytest.raises(ValueError, match=r"must be \(S, K, F, W\)"):
            anchor_signs_tf(np.zeros((2, 4, 10)), freqs, times)
        with pytest.raises(ValueError, match="frequency bins"):
            anchor_signs_tf(np.zeros((2, 1, 3, 10)), freqs, times)
        with pytest.raises(ValueError, match="epoch times"):
            anchor_signs_tf(np.zeros((2, 1, 4, 9)), freqs, times)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestSubjectScales:
    def test_one_scale_per_subject(self) -> None:
        arr = np.ones((N_SUBJECTS, 3, 4))
        assert subject_scales(arr).shape == (N_SUBJECTS,)

    def test_tracks_per_subject_amplitude(self) -> None:
        arr = np.ones((3, 2, 8))
        arr[1] *= 5.0
        arr[2] *= 0.2
        np.testing.assert_allclose(subject_scales(arr), [1.0, 5.0, 0.2])

    def test_measured_over_selected_components_only(self) -> None:
        arr = np.ones((2, 2, 4))
        arr[:, 1] *= 100.0  # a component no figure would show
        np.testing.assert_allclose(subject_scales(arr, [0]), [1.0, 1.0])

    def test_all_zero_subject_gets_unit_scale(self) -> None:
        arr = np.ones((2, 2, 4))
        arr[0] = 0.0
        np.testing.assert_allclose(subject_scales(arr), [1.0, 1.0])

    def test_robust_to_a_single_hot_bin(self) -> None:
        # The 99th percentile, not the maximum: one extreme cell must not set the
        # scale, or a single artefactual bin would silence a whole participant.
        arr = np.ones((1, 1, 500))
        arr[0, 0, 0] = 1000.0
        assert subject_scales(arr)[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "arr, comp_indices, match",
        [
            (np.ones((3, 4)), None, "at least 3-D"),
            (np.ones((0, 2, 4)), None, "at least one subject"),
            (np.ones((2, 2, 4)), [], "at least one component"),
            (np.ones((2, 2, 4)), [2], "out of range"),
            (np.ones((2, 2, 4)), [-1], "out of range"),
        ],
    )
    def test_rejects_bad_inputs(self, arr, comp_indices, match) -> None:
        with pytest.raises(ValueError, match=match):
            subject_scales(arr, comp_indices)


class TestEqualizeSubjectInfluence:
    def test_every_subject_ends_on_the_same_range(self) -> None:
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((N_SUBJECTS, 3, 20))
        arr *= np.array([1.0, 10.0, 0.1, 4.0])[:, None, None]
        out = equalize_subject_influence(arr)
        np.testing.assert_allclose(subject_scales(out), 1.0)

    def test_preserves_relative_component_strength_within_a_subject(self) -> None:
        # The scale is per subject, not per (subject, component): a component
        # with no response must stay weak, or every component would look equal.
        arr = np.ones((2, 2, 6))
        arr[:, 1] *= 0.01
        out = equalize_subject_influence(arr)
        ratio = np.abs(out[:, 1]).max() / np.abs(out[:, 0]).max()
        assert ratio == pytest.approx(0.01)

    def test_equalizes_the_group_mean_contributions(self) -> None:
        # The point of the rescaling: without it the loud subject *is* the mean.
        arr = np.zeros((2, 1, 4))
        arr[0, 0] = [1.0, -1.0, 1.0, -1.0]
        arr[1, 0] = [-100.0, 100.0, -100.0, 100.0]
        assert np.abs(arr.mean(axis=0)).max() > 40.0  # unweighted: subject 1 wins
        out = equalize_subject_influence(arr)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)

    def test_scale_invariant_correlations_unaffected(self) -> None:
        rng = np.random.default_rng(4)
        arr = rng.standard_normal((3, 2, 30))
        out = equalize_subject_influence(arr)
        for s in range(3):
            for k in range(2):
                assert pearsonr(arr[s, k], out[s, k])[0] == pytest.approx(1.0)

    def test_does_not_mutate_input(self) -> None:
        arr = np.arange(24, dtype=float).reshape(2, 3, 4)
        before = arr.copy()
        equalize_subject_influence(arr)
        np.testing.assert_array_equal(arr, before)

    def test_accepts_four_dimensional_tf_maps(self) -> None:
        rng = np.random.default_rng(5)
        arr = rng.standard_normal((3, 2, 4, 10))
        out = equalize_subject_influence(arr, [0, 1])
        assert out.shape == arr.shape
        np.testing.assert_allclose(subject_scales(out, [0, 1]), 1.0)


class TestFullTfGroupMean:
    def test_reduces_the_subject_axis(self) -> None:
        rng = np.random.default_rng(6)
        sources = rng.standard_normal((3, 2, 4, 20))
        out = full_tf_group_mean(sources, np.ones((3, 2)))
        assert out.shape == (2, 4, 20)

    def test_applies_the_per_pair_signs(self) -> None:
        # Two subjects carrying the same map at opposite polarity: the raw mean
        # cancels, the signed one does not. This is why the QC figure exists.
        sources = np.zeros((2, 1, 2, 10))
        sources[0, 0] = 1.0
        sources[1, 0] = -1.0
        np.testing.assert_allclose(sources.mean(axis=0), 0.0)
        out = full_tf_group_mean(sources, np.array([[1.0], [-1.0]]))
        np.testing.assert_allclose(out, 1.0)

    def test_weighs_every_subject_equally(self) -> None:
        # A subject 100x louder must not stand in for the group, exactly as in
        # equalize_subject_influence — the scale here is just measured streaming.
        sources = np.zeros((2, 1, 2, 10))
        sources[0, 0] = 1.0
        sources[1, 0] = -100.0
        out = full_tf_group_mean(sources, np.ones((2, 1)))
        np.testing.assert_allclose(out, 0.0, atol=1e-9)

    def test_matches_the_equalized_mean(self) -> None:
        rng = np.random.default_rng(7)
        sources = (
            rng.standard_normal((3, 2, 4, 20))
            * np.array([1.0, 8.0, 0.3])[:, None, None, None]
        )
        signs = np.array([[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(
            full_tf_group_mean(sources, signs),
            equalize_subject_influence(sources * signs[:, :, None, None]).mean(axis=0),
        )

    def test_all_zero_subject_gets_unit_scale(self) -> None:
        sources = np.zeros((2, 1, 2, 10))
        sources[1, 0] = 1.0
        out = full_tf_group_mean(sources, np.ones((2, 1)))
        np.testing.assert_allclose(out, 0.5)

    def test_does_not_mutate_input(self) -> None:
        sources = np.arange(2 * 1 * 2 * 5, dtype=float).reshape(2, 1, 2, 5)
        before = sources.copy()
        full_tf_group_mean(sources, np.array([[1.0], [-1.0]]))
        np.testing.assert_array_equal(sources, before)

    @pytest.mark.parametrize(
        "sources, signs, match",
        [
            (np.ones((2, 1, 4)), np.ones((2, 1)), r"must be \(S, K, F, T\)"),
            (np.ones((0, 1, 2, 4)), np.ones((0, 1)), "at least one subject"),
            (np.ones((2, 1, 2, 4)), np.ones((2, 2)), "signs shape"),
            (np.ones((2, 1, 2, 4)), np.ones(2), "signs shape"),
        ],
    )
    def test_rejects_bad_inputs(self, sources, signs, match) -> None:
        with pytest.raises(ValueError, match=match):
            full_tf_group_mean(sources, signs)


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
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
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
        # The reference pair and the TF axis.
        assert res.ref_topo.shape == (N_CHANNELS,)
        assert res.ref_tf.shape == (N_FREQS, win)
        assert res.tf_corr.shape == (N_SUBJECTS, n_comp)
        assert res.onset_tf.shape == (N_SUBJECTS, n_comp, N_FREQS, win)
        assert res.freqs.shape == (N_FREQS,)
        assert res.wavelet_consistency.n_subjects == N_SUBJECTS

    def test_planted_component_wins_both_axes_of_the_reference_pair(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        mean_topo = res.topo_corr.mean(axis=0)
        mean_tf = res.tf_corr.mean(axis=0)
        assert int(np.argmax(mean_topo)) == 0
        assert int(np.argmax(mean_tf)) == 0
        assert mean_topo[0] > 0.8
        assert mean_tf[0] > 0.8

    def test_reference_is_the_wavelet_pair(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        # ref_topo / ref_tf are what wavelet_reference returns for this tensor and
        # epoch — there is one reference, built from the IVA's own input — up to
        # the one global sign the driven-response anchoring may flip (below).
        ref_topo, ref_tf, anchor, _cons = wavelet_reference(
            wav, onsets, res.n_epoch_pre, res.n_epoch_post, ch_names
        )
        global_sign = np.sign(np.dot(res.ref_topo, ref_topo))
        np.testing.assert_allclose(res.ref_topo, global_sign * ref_topo)
        np.testing.assert_allclose(res.ref_tf, global_sign * ref_tf)
        assert res.anchor_channel == anchor

    def test_reference_polarity_is_anchored_on_the_driven_response(
        self, onsets, ch_names
    ) -> None:
        # The pair's PCA sign is arbitrary but the boxcar y-axis is not, so the
        # reference is flipped until its own driven response reads positive.
        # Without this a component following the stimulus could score -1 on the
        # boxcar purely because of the reference's polarity.
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        signs, have_anchor = anchor_signs_tf(
            res.ref_tf[np.newaxis, np.newaxis],
            res.freqs,
            res.epoch_times,
            resp_duration_s=res.resp_duration_s,
        )
        assert have_anchor is True
        assert signs[0, 0] == 1.0
        # ...and the planted component then reads positive on the boxcar too.
        assert res.time_corr_pca.mean(axis=0)[0] > 0.9

    def test_tf_corr_recomputable_from_stored_maps(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        # onset_tf and tf_corr carry the same component sign, so recomputing the
        # correlation from the stored maps must reproduce the reported score.
        np.testing.assert_allclose(
            tf_map_correlation(res.onset_tf, res.ref_tf), res.tf_corr, atol=1e-10
        )

    def test_planted_component_wins_both_axes(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
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
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        scores = [
            quality_score(res.topo_corr, res.time_corr_pca, k)
            for k in range(comps.shape[1])
        ]
        assert scores[0] == max(scores)
        assert all(s < scores[0] - 0.3 for s in scores[1:])

    def test_epoch_times_zero_at_onset(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        assert res.epoch_times[res.n_epoch_pre] == pytest.approx(0.0)

    def test_orientation_makes_mean_topo_corr_non_negative(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        assert (res.topo_corr.mean(axis=0) >= 0).all()
        assert set(np.unique(res.sign_per_comp)) <= {-1.0, 1.0}

    def test_patterns_oriented_consistently_with_scores(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
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

    def test_per_participant_signs_are_not_applied_to_the_stored_arrays(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        n_comp = comps.shape[1]
        assert res.sign_per_subject.shape == (N_SUBJECTS, n_comp)
        assert res.sign_tf_per_subject.shape == (N_SUBJECTS, n_comp)
        assert set(np.unique(res.sign_per_subject)) <= {-1.0, 1.0}
        assert set(np.unique(res.sign_tf_per_subject)) <= {-1.0, 1.0}
        # The scores keep the one-flip-per-component convention; only the views
        # apply the per-participant flip.
        patterns_view, topo_view = res.topomap_view()
        np.testing.assert_allclose(
            patterns_view, res.patterns * res.sign_per_subject[:, :, np.newaxis]
        )
        np.testing.assert_allclose(topo_view, res.topo_corr * res.sign_per_subject)
        assert 0.0 <= res.anchor_agreement() <= 1.0

    def test_both_views_apply_the_same_per_pair_sign(self, onsets, ch_names) -> None:
        # The point of the change: a pair's topography and its TF map are one
        # decomposition, so they are never flipped apart.
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        signs = res.sign_per_subject
        _patterns_view, topo_view = res.topomap_view()
        tf_view, tf_corr_view, bands_view = res.tf_view()
        np.testing.assert_allclose(
            tf_view, res.onset_tf * signs[:, :, np.newaxis, np.newaxis]
        )
        np.testing.assert_allclose(tf_corr_view, res.tf_corr * signs)
        for tag, corr in bands_view.items():
            np.testing.assert_allclose(corr, res.tf_corr_bands[tag] * signs)
        # ...and that shared sign is the topography's, so the annotated r of the
        # topomap panels is non-negative throughout.
        assert (topo_view >= 0).all()

    def test_sign_is_the_topomap_correlation_sign(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        # Exactly the rule the module documents: correlate the pair's topography
        # with the reference, flip when negative.
        expected = np.sign(res.topo_corr)
        expected[expected == 0] = 1.0
        np.testing.assert_array_equal(res.sign_per_subject, expected)
        np.testing.assert_array_equal(
            res.sign_per_subject, anchor_signs_reference(res.patterns, res.ref_topo)
        )
        assert res.subject_anchor_name == SUBJECT_SIGN_ANCHOR

    def test_full_tf_mean_carries_the_shared_signs(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        n_comp = comps.shape[1]
        assert res.full_tf_mean.shape == (n_comp, N_FREQS, N_TIMES)
        assert res.full_times.shape == (N_TIMES,)
        np.testing.assert_allclose(res.full_times, np.arange(N_TIMES) / SFREQ)
        # It is the group mean of the raw sources under the total per-pair flip:
        # the component orientation times the per-participant sign.
        np.testing.assert_allclose(
            res.full_tf_mean,
            full_tf_group_mean(
                sources,
                res.sign_per_subject * res.sign_per_comp[np.newaxis, :],
            ),
        )

    def test_full_tf_mean_survives_pairs_that_arrive_inverted(
        self, onsets, ch_names
    ) -> None:
        # The QC map's reason to exist: under IVA's arbitrary per-pair polarity
        # the plain group mean cancels, and the aligned one does not.
        comps, sources, freqs, wav = _quality_inputs(onsets)
        flips = np.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        res = compute_iva_quality(
            comps * flips[:, :, np.newaxis],
            sources * flips[:, :, np.newaxis, np.newaxis],
            freqs,
            onsets,
            SFREQ,
            ch_names,
            wav,
        )
        raw_mean = (sources * flips[:, :, np.newaxis, np.newaxis]).mean(axis=0)
        assert np.abs(res.full_tf_mean[0]).max() > np.abs(raw_mean[0]).max()

    def test_topomap_view_aligns_participants_that_arrive_inverted(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        # Invert half the (subject, component) pairs, which is exactly the freedom
        # IVA leaves: flipping a pattern together with its source changes nothing
        # about the data it explains.
        flips = np.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        res = compute_iva_quality(
            comps * flips[:, :, np.newaxis],
            sources * flips[:, :, np.newaxis, np.newaxis],
            freqs,
            onsets,
            SFREQ,
            ch_names,
            wav,
        )
        patterns_view, topo_view = res.topomap_view()
        # The planted component's group mean survives the alignment and cancels
        # without it — the whole point of anchoring the sign per participant.
        assert (
            np.abs(patterns_view[:, 0].mean(axis=0)).max()
            > np.abs(res.patterns[:, 0].mean(axis=0)).max()
        )
        # Every participant now agrees with the reference, so the r annotated
        # beside a panel is non-negative and describes the map drawn there.
        assert (topo_view >= 0).all()
        for s in range(N_SUBJECTS):
            for k in range(comps.shape[1]):
                r = np.corrcoef(patterns_view[s, k], res.ref_topo)[0, 1]
                assert r == pytest.approx(topo_view[s, k], abs=1e-8)

    def test_tf_view_uncancels_the_group_mean_of_inverted_pairs(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        flips = np.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        res = compute_iva_quality(
            comps * flips[:, :, np.newaxis],
            sources * flips[:, :, np.newaxis, np.newaxis],
            freqs,
            onsets,
            SFREQ,
            ch_names,
            wav,
        )
        tf_view, _tf_corr_view, _bands = res.tf_view()
        # The planted component's group-mean map no longer cancels, even though
        # the sign came from the topography rather than from the map itself.
        assert (
            np.abs(tf_view[:, 0].mean(axis=0)).max()
            > np.abs(res.onset_tf[:, 0].mean(axis=0)).max()
        )

    def test_anchor_agreement_reports_the_driven_response_polarity(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        assert res.tf_anchor_name != "reference TF-map correlation"
        # The diagnostic is what anchor_signs_tf says about the stored maps, and
        # anchor_agreement is how often it matches the sign actually applied.
        expected, _have = anchor_signs_tf(
            res.onset_tf,
            res.freqs,
            res.epoch_times,
            resp_duration_s=res.resp_duration_s,
        )
        np.testing.assert_array_equal(res.sign_tf_per_subject, expected)
        assert res.anchor_agreement() == pytest.approx(
            float(np.mean(res.sign_per_subject == expected))
        )
        # The planted component matches the reference *and* responds positively,
        # so its pattern and its source agree for every participant.
        assert (res.sign_per_subject[:, 0] == res.sign_tf_per_subject[:, 0]).all()
        # And under the shared sign its driven response does read positive.
        tf_view, _corr, _bands = res.tf_view()
        band = frequency_band_mask(
            res.freqs,
            ASSR_FREQ - TF_ANCHOR_HALFWIDTH_HZ,
            ASSR_FREQ + TF_ANCHOR_HALFWIDTH_HZ,
        )
        window = (res.epoch_times >= 0.0) & (res.epoch_times <= res.resp_duration_s)
        driven = tf_view[:, 0][:, band][:, :, window].mean(axis=(1, 2))
        assert (driven > 0).all()

    def test_band_tf_variants_computed_and_recomputable(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        # The synthetic frequency axis spans 1-50 Hz, so both bands are available.
        assert set(res.tf_corr_bands) == {tag for *_rest, tag in TF_BANDS}
        for _name, fmin, fmax, tag in TF_BANDS:
            assert res.tf_corr_bands[tag].shape == (N_SUBJECTS, comps.shape[1])
            # Same sign convention as the maps, so the stored score is exactly the
            # correlation of the stored maps over that band.
            np.testing.assert_allclose(
                tf_map_correlation(
                    res.onset_tf, res.ref_tf, frequency_band_mask(res.freqs, fmin, fmax)
                ),
                res.tf_corr_bands[tag],
                atol=1e-10,
            )

    def test_wavelet_variants_lists_the_whole_map_first(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        variants = res.wavelet_variants()
        assert [tag for *_rest, tag in variants] == [
            "wavelet_tf",
            *[tag for *_rest, tag in TF_BANDS],
        ]
        # The whole-map entry is the unnamed one; the bands carry their names.
        assert variants[0][0] is None
        assert [name for name, *_rest in variants[1:]] == [
            name for name, *_rest in TF_BANDS
        ]
        np.testing.assert_array_equal(variants[0][1], res.tf_corr)

    def test_aligned_wavelet_variants_carry_the_tf_view_sign(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        _tf_view, tf_corr_view, bands_view = res.tf_view()
        aligned = res.wavelet_variants(aligned=True)
        # Same entries in the same order as the unaligned call, each measured on
        # the maps tf_view draws rather than on the stored orientation.
        assert [tag for *_rest, tag in aligned] == [
            tag for *_rest, tag in res.wavelet_variants()
        ]
        np.testing.assert_allclose(aligned[0][1], tf_corr_view)
        for _band_name, corr, tag in aligned[1:]:
            np.testing.assert_allclose(corr, bands_view[tag])

    def test_aligned_variants_flip_a_time_course_with_its_source(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        signs = res.sign_per_subject
        for (name, corr, avg, tag), (a_name, a_corr, a_avg, a_tag) in zip(
            res.variants(), res.variants(aligned=True)
        ):
            assert (a_name, a_tag) == (name, tag)
            np.testing.assert_allclose(a_avg, avg * signs[:, :, np.newaxis])
            np.testing.assert_allclose(a_corr, corr * signs)
            # The aligned score is the boxcar correlation of the aligned time
            # course, not just its sign flipped by hand: pattern, source and score
            # all describe one flipped pair.
            win = res.n_epoch_pre + res.n_epoch_post
            np.testing.assert_allclose(
                a_corr,
                boxcar_correlation(
                    a_avg,
                    res.n_epoch_pre,
                    win,
                    response_duration_samples(res.n_epoch_post, SFREQ),
                ),
                atol=1e-10,
            )

    def test_aligned_scores_uncancel_the_component_score(
        self, onsets, ch_names
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        # The same half-inverted pairs the view tests use: IVA is free to hand
        # them over like this, and a score averaged over mixed polarities cancels.
        flips = np.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
        res = compute_iva_quality(
            comps * flips[:, :, np.newaxis],
            sources * flips[:, :, np.newaxis, np.newaxis],
            freqs,
            onsets,
            SFREQ,
            ch_names,
            wav,
        )
        _patterns, topo_view = res.topomap_view()
        raw_tf = res.wavelet_variants()[0][1]
        aligned_tf = res.wavelet_variants(aligned=True)[0][1]
        # The planted component scores near the ideal corner once every pair is
        # scored under the sign its maps are drawn with, and well below that
        # without — the scatter equivalent of a group mean cancelling.
        assert quality_score(topo_view, aligned_tf, 0) > quality_score(
            res.topo_corr, raw_tf, 0
        )
        assert (topo_view >= 0).all()

    def test_band_tf_variant_skipped_outside_the_frequency_range(
        self, onsets, ch_names
    ) -> None:
        comps, sources, _freqs, wav = _quality_inputs(onsets)
        # 13-30 Hz reaches neither the 1-10 Hz nor the 30-50 Hz band.
        beta = np.linspace(13.0, 29.0, N_FREQS)
        res = compute_iva_quality(comps, sources, beta, onsets, SFREQ, ch_names, wav)
        assert res.tf_corr_bands == {}
        assert [tag for *_rest, tag in res.wavelet_variants()] == ["wavelet_tf"]

    def test_40hz_variant_flagged_off_outside_range(self, onsets, ch_names) -> None:
        comps, sources, _freqs, wav = _quality_inputs(onsets)
        low = np.linspace(1.0, 10.0, N_FREQS)  # ASSR_FREQ is outside
        res = compute_iva_quality(comps, sources, low, onsets, SFREQ, ch_names, wav)
        assert res.have_40hz is False
        np.testing.assert_allclose(res.time_corr_40, 0.0)
        assert [tag for *_rest, tag in res.variants()] == ["pca"]

    def test_40hz_variant_present_in_range(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        assert freqs.min() <= ASSR_FREQ <= freqs.max()
        res = compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        assert res.have_40hz is True
        assert [tag for *_rest, tag in res.variants()] == ["pca", "40hz"]

    def test_input_not_mutated(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        c0, s0, w0 = comps.copy(), sources.copy(), wav.copy()
        compute_iva_quality(comps, sources, freqs, onsets, SFREQ, ch_names, wav)
        np.testing.assert_array_equal(comps, c0)
        np.testing.assert_array_equal(sources, s0)
        np.testing.assert_array_equal(wav, w0)

    @pytest.mark.parametrize(
        "wav_slice, match",
        [
            (lambda w: w[:, :, :, 0], r"must be \(S, C, F, T\)"),
            (lambda w: w[:-1], "does not match the IVA extent"),
            (lambda w: w[:, :-1], "does not match the IVA extent"),
            (lambda w: w[:, :, :-1], "does not match the IVA extent"),
            (lambda w: w[:, :, :, :-1], "does not match the IVA extent"),
        ],
    )
    def test_rejects_wavelet_extent_mismatch(
        self, onsets, ch_names, wav_slice, match
    ) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        with pytest.raises(ValueError, match=match):
            compute_iva_quality(
                comps, sources, freqs, onsets, SFREQ, ch_names, wav_slice(wav)
            )

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
        comps, sources, freqs, wav = _quality_inputs(onsets)
        c, s, f = mutate(comps, sources, freqs)
        with pytest.raises(ValueError, match=match):
            compute_iva_quality(c, s, f, onsets, SFREQ, ch_names, wav)

    def test_rejects_channel_name_count_mismatch(self, onsets, ch_names) -> None:
        comps, sources, freqs, wav = _quality_inputs(onsets)
        with pytest.raises(ValueError, match="channel names were given"):
            compute_iva_quality(
                comps, sources, freqs, onsets, SFREQ, ch_names[:-1], wav
            )
