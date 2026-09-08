"""Tests for :mod:`src.analysis.assr_trials`."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import assr_trials as at
from src.definitions.constants import AssrEpoch


# ---------------------------------------------------------------------------
# Labels and frequency selection
# ---------------------------------------------------------------------------


class TestSourceLabels:
    def test_components_then_binary_filter(self):
        assert at.source_labels(3) == ["IC 1", "IC 2", "IC 3", at.BINARY_FILTER_LABEL]

    def test_binary_filter_is_last(self):
        labels = at.source_labels(5)
        assert labels[-1] == at.BINARY_FILTER_LABEL
        assert at.BINARY_FILTER_LABEL not in labels[:-1]

    def test_no_components_still_has_the_reference(self):
        assert at.source_labels(0) == [at.BINARY_FILTER_LABEL]


class TestFrequencySelection:
    @pytest.fixture
    def freqs(self):
        return np.arange(1.0, 51.0)

    def test_zero_halfwidth_takes_the_single_nearest_bin(self, freqs):
        assert at.frequency_selection(freqs, 40.0, 0.0).tolist() == [39]

    def test_nearest_bin_when_centre_is_off_grid(self, freqs):
        assert at.frequency_selection(freqs, 40.4, 0.0).tolist() == [39]
        assert at.frequency_selection(freqs, 40.6, 0.0).tolist() == [40]

    def test_halfwidth_takes_the_closed_interval(self, freqs):
        selected = at.frequency_selection(freqs, 40.0, 5.0)
        assert freqs[selected].tolist() == list(np.arange(35.0, 46.0))

    def test_never_returns_nothing(self):
        # An interval falling between two bins would otherwise give a NaN track.
        coarse = np.array([10.0, 50.0])
        assert at.frequency_selection(coarse, 40.0, 1.0).size == 1

    def test_rejects_negative_halfwidth(self, freqs):
        with pytest.raises(ValueError, match="halfwidth"):
            at.frequency_selection(freqs, 40.0, -1.0)

    def test_rejects_empty_grid(self):
        with pytest.raises(ValueError, match="empty"):
            at.frequency_selection(np.array([]), 40.0)


class TestSelectionLabel:
    def test_single_bin(self):
        assert at.selection_label(40.0, 0.0) == "40hz"

    def test_band(self):
        assert at.selection_label(40.0, 5.0) == "35-45hz"


# ---------------------------------------------------------------------------
# Spatial filters
# ---------------------------------------------------------------------------


class TestRecoverSpatialFilters:
    @pytest.fixture
    def filters(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=(4, 3, 12))  # participants, components, channels

    def test_round_trips_through_the_pseudo_inverse(self, filters):
        patterns = np.stack([np.linalg.pinv(f).T for f in filters])
        recovered = at.recover_spatial_filters(patterns)
        assert np.allclose(recovered, filters)

    def test_sign_flips_propagate(self, filters):
        # A per-participant sign flip is a diagonal +-1 scaling on the component
        # index, so it must survive the inversion unchanged.
        signs = np.array([1.0, -1.0, 1.0])[None, :, None]
        patterns = np.stack([np.linalg.pinv(f).T for f in filters * signs])
        assert np.allclose(at.recover_spatial_filters(patterns), filters * signs)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="participants, components, channels"):
            at.recover_spatial_filters(np.zeros((3, 4)))

    def test_rejects_rank_deficient_patterns(self):
        # Two identical component rows cannot be inverted back to a filter.
        patterns = np.zeros((1, 2, 5))
        patterns[0, 0] = patterns[0, 1] = np.arange(5.0)
        with pytest.raises(ValueError, match="rank-deficient"):
            at.recover_spatial_filters(patterns)


class TestBinaryFilterWeights:
    def test_normalised_weights_average(self):
        mask = np.array([True, False, True, True])
        weights = at.binary_filter_weights(mask)
        assert weights.sum() == pytest.approx(1.0)
        assert weights.tolist() == pytest.approx([1 / 3, 0.0, 1 / 3, 1 / 3])

    def test_unnormalised_weights_sum(self):
        mask = np.array([True, False, True])
        assert at.binary_filter_weights(mask, normalize=False).tolist() == [
            1.0,
            0.0,
            1.0,
        ]

    def test_rejects_empty_selection(self):
        with pytest.raises(ValueError, match="no channel"):
            at.binary_filter_weights(np.zeros(5, dtype=bool))


class TestStackFilters:
    def test_binary_row_is_appended_last(self):
        learned = np.zeros((3, 2, 6))
        binary = np.arange(6.0)
        stacked = at.stack_filters(learned, binary)
        assert stacked.shape == (3, 3, 6)
        assert np.allclose(stacked[:, -1], binary)
        assert np.allclose(stacked[:, :-1], learned)

    def test_binary_row_is_identical_for_every_participant(self):
        stacked = at.stack_filters(np.zeros((4, 1, 3)), np.array([1.0, 2.0, 3.0]))
        assert np.allclose(stacked[:, -1] - stacked[0, -1], 0.0)

    def test_rejects_channel_mismatch(self):
        with pytest.raises(ValueError, match="channel axis"):
            at.stack_filters(np.zeros((2, 2, 6)), np.ones(5))


class TestProjectChannels:
    def test_contracts_only_the_channel_axis(self):
        filters = np.eye(3)
        track = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        assert np.allclose(at.project_channels(filters, track), track)

    def test_frequency_slicing_commutes_with_projection(self):
        # The reason the streaming reader may slice frequency before projecting.
        rng = np.random.default_rng(1)
        filters, track = rng.normal(size=(2, 6)), rng.normal(size=(6, 8, 10))
        after = at.project_channels(filters, track)[:, 3, :]
        before = at.project_channels(filters, track[:, 3, :])
        assert np.allclose(after, before)

    def test_rejects_channel_mismatch(self):
        with pytest.raises(ValueError, match="channel"):
            at.project_channels(np.zeros((2, 6)), np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Polarity anchoring
# ---------------------------------------------------------------------------


class TestPolarityFlip:
    @pytest.fixture
    def mask(self):
        keep = np.zeros(10, dtype=bool)
        keep[:4] = True
        return keep

    def test_flips_participants_loading_negatively(self, mask):
        patterns = np.zeros((3, 1, 10))
        patterns[0, 0, :4] = 1.0  # positive on the mask
        patterns[1, 0, :4] = -1.0  # negative on the mask
        patterns[2, 0, :4] = 2.0
        flip, _strength = at.polarity_flip(patterns, mask)
        assert flip[:, 0].tolist() == [1.0, -1.0, 1.0]

    def test_flip_is_plus_or_minus_one(self, mask):
        rng = np.random.default_rng(2)
        flip, _ = at.polarity_flip(rng.normal(size=(6, 3, 10)), mask)
        assert set(np.unique(flip)).issubset({-1.0, 1.0})

    def test_rejects_mask_mismatch(self):
        with pytest.raises(ValueError, match="electrode_mask"):
            at.polarity_flip(np.zeros((2, 1, 10)), np.ones(5, dtype=bool))

    def test_orients_every_participant_against_the_rest_of_the_head(self, mask):
        rng = np.random.default_rng(4)
        patterns = rng.normal(size=(8, 2, 10))
        flip, _ = at.polarity_flip(patterns, mask)
        oriented = patterns * flip[:, :, None]
        inside = oriented[:, :, mask].mean(axis=2)
        outside = oriented[:, :, ~mask].mean(axis=2)
        assert (inside - outside >= 0).all()

    def test_a_global_offset_cannot_flip_it(self, mask):
        # The pattern is MORE negative on the mask than off it, but a large positive
        # offset makes its mask MEAN positive. An anchor reading only the mask mean
        # would be fooled into leaving it; the correlation compares inside against
        # outside, so it flips correctly.
        patterns = np.zeros((1, 1, 10))
        patterns[0, 0, :4] = -1.0
        patterns += 5.0
        assert patterns[0, 0, mask].mean() > 0  # the mean-based anchor's blind spot
        flip, _ = at.polarity_flip(patterns, mask)
        assert flip[0, 0] == -1.0

    def test_strength_is_the_absolute_correlation(self, mask):
        patterns = np.zeros((1, 1, 10))
        patterns[0, 0, :4] = 1.0
        _flip, strength = at.polarity_flip(patterns, mask)
        indicator = mask.astype(float)
        expected = abs(np.corrcoef(patterns[0, 0], indicator)[0, 1])
        assert strength[0, 0] == pytest.approx(expected)
        assert 0.0 <= strength[0, 0] <= 1.0

    def test_sign_is_independent_per_component(self, mask):
        # Two components of ONE participant, loading oppositely: the flip must not be
        # shared between them.
        patterns = np.zeros((1, 2, 10))
        patterns[0, 0, :4] = 1.0
        patterns[0, 1, :4] = -1.0
        flip, _ = at.polarity_flip(patterns, mask)
        assert flip[0].tolist() == [1.0, -1.0]

    def test_flat_pattern_gets_zero_strength_and_no_flip(self, mask):
        flip, strength = at.polarity_flip(np.full((1, 1, 10), 3.0), mask)
        assert flip[0, 0] == 1.0
        assert strength[0, 0] == 0.0

    def test_rejects_degenerate_masks(self):
        for bad in (np.zeros(10, dtype=bool), np.ones(10, dtype=bool)):
            with pytest.raises(ValueError, match="some but not all"):
                at.polarity_flip(np.zeros((2, 1, 10)), bad)


class TestRoiChannelwiseSnr:
    @pytest.fixture
    def baseline_mask(self):
        m = np.zeros(40, dtype=bool)
        m[:20] = True
        return m

    def _synthetic(self, gains, rng, n_trials=24, n_samples=40, response=3.0):
        """One participant, channels differing only in GAIN, identical SNR."""
        n_ch = len(gains)
        noise = rng.normal(size=(1, n_ch, n_trials, n_samples))
        signal = np.zeros(n_samples)
        signal[20:] = response
        data = (noise + signal) * np.asarray(gains)[None, :, None, None]
        return data + 100.0 * np.asarray(gains)[None, :, None, None]

    def test_output_is_in_units_of_its_own_baseline_sd(self, baseline_mask):
        rng = np.random.default_rng(0)
        data = self._synthetic([1.0] * 8, rng)
        snr, _ = at.roi_channelwise_snr(data, baseline_mask)
        # Step 3 is exactly what makes this true; without it the baseline SD would be
        # ~1/sqrt(n_channels), not 1.
        assert snr[..., baseline_mask].std(ddof=1) == pytest.approx(1.0, rel=0.15)

    def test_gain_differences_do_not_reweight_the_roi(self, baseline_mask):
        # Same SNR in every channel, gains spanning 30x. An equal-weight ROI must give
        # the same answer as it would with equal gains.
        equal = at.roi_channelwise_snr(
            self._synthetic([1.0] * 8, np.random.default_rng(1)), baseline_mask
        )[0]
        spread = at.roi_channelwise_snr(
            self._synthetic(
                [0.1, 0.3, 1, 3, 0.2, 2, 0.5, 3.0], np.random.default_rng(1)
            ),
            baseline_mask,
        )[0]
        assert np.allclose(equal, spread, atol=1e-9)

    def test_a_dead_channel_dilutes_rather_than_dominates(self, baseline_mask):
        rng = np.random.default_rng(2)
        data = self._synthetic([1.0] * 8, rng)
        with_dead = data.copy()
        with_dead[0, 0] = rng.normal(size=with_dead[0, 0].shape) + 50.0  # no response
        good = at.roi_channelwise_snr(data, baseline_mask)[0]
        mixed = at.roi_channelwise_snr(with_dead, baseline_mask)[0]
        drive = slice(20, None)
        assert 0 < mixed[..., drive].mean() < good[..., drive].mean()

    def test_diagnostics_report_the_effective_channel_count(self, baseline_mask):
        rng = np.random.default_rng(3)
        data = self._synthetic([1.0] * 9, rng)
        _snr, diag = at.roi_channelwise_snr(data, baseline_mask)
        # Independent channels: the ROI mean's baseline SD ~ 1/sqrt(9).
        assert diag["roi_baseline_sd"] == pytest.approx(1 / 3, rel=0.3)
        assert diag["effective_channels"] == pytest.approx(9, rel=0.6)
        assert diag["max_abs_z1"] > 0

    def test_a_quiet_baseline_trial_does_not_create_a_post_stimulus_plateau(self):
        """The defect the pooled divisor exists to prevent.

        Two channels get a baseline whose loudness varies trial to trial, so some trials
        draw a very quiet one, plus a drift that outlasts the stimulus. A per-trial
        divisor would blow that drift up on exactly those trials and — because the ROI
        is a mean over channels — drag the whole course to a plateau that never returns
        to baseline. Pooling the divisor over trials removes the lottery, so the course
        has to come back down.
        """
        rng = np.random.default_rng(11)
        mask = np.zeros(60, dtype=bool)
        mask[:20] = True
        data = rng.normal(size=(1, 8, 40, 60))
        data[..., 20:40] += 3.0  # the stimulus response, in every channel
        for channel in (0, 1):
            loudness = rng.lognormal(0.0, 1.5, size=40)
            data[0, channel, :, :20] *= loudness[:, None]
            data[0, channel, :, 40:] += 1.0  # drift outlasting the stimulus

        snr, diagnostics = at.roi_channelwise_snr(data, mask)
        after = snr[..., 40:].mean()
        peak = snr[..., 20:40].mean()
        assert after / peak < 0.05  # returns to baseline rather than plateauing
        # The runaway a per-trial divisor produces is orders of magnitude larger.
        assert diagnostics["max_abs_z1"] < 50

    def test_scaling_is_one_constant_not_per_participant(self, baseline_mask):
        """A participant with noisier electrodes must not be rescaled to match.

        Per-participant scaling is what would break a paired contrast: the divisor is
        estimated per (participant, condition), so it rescales one participant's two
        conditions differently for reasons unrelated to the response.
        """
        rng = np.random.default_rng(20)
        data = self._synthetic([1.0] * 6, rng, n_trials=30)
        loud = data.copy()
        loud[0, :, :, :] *= 1.0
        # Make ONE participant's whole recording noisier by adding a second "subject".
        pair = np.concatenate([data, data * 3.0], axis=0)  # (2, C, N, W)
        snr, _ = at.roi_channelwise_snr(pair, baseline_mask)
        drive = slice(20, None)
        a = snr[0][..., drive].mean()
        b = snr[1][..., drive].mean()
        # Both are pure gain changes, which step 1 already removes, so they must agree —
        # and neither is renormalised to the other by a per-participant divisor.
        assert a == pytest.approx(b, rel=1e-6)

    def test_a_shared_scale_can_be_passed_in(self, baseline_mask):
        rng = np.random.default_rng(21)
        data = self._synthetic([1.0] * 6, rng)
        auto, diag = at.roi_channelwise_snr(data, baseline_mask)
        forced, forced_diag = at.roi_channelwise_snr(
            data, baseline_mask, scale=2 * diag["roi_baseline_sd"]
        )
        assert forced_diag["roi_baseline_sd"] == pytest.approx(
            2 * diag["roi_baseline_sd"]
        )
        assert np.allclose(forced, auto / 2)

    def test_reports_the_spread_it_declined_to_apply(self, baseline_mask):
        rng = np.random.default_rng(22)
        pair = np.concatenate(
            [
                self._synthetic([1.0] * 6, rng),
                self._synthetic([1.0] * 6, rng, n_trials=24) * 3.0,
            ],
            axis=0,
        )
        _snr, diag = at.roi_channelwise_snr(pair, baseline_mask)
        assert diag["participant_sd_spread"] >= 1.0

    def test_rejects_a_non_positive_scale(self, baseline_mask):
        data = np.random.default_rng(23).normal(size=(1, 2, 4, 40))
        with pytest.raises(ValueError, match="positive"):
            at.roi_channelwise_snr(data, baseline_mask, scale=0.0)

    def test_rejects_wrong_rank(self, baseline_mask):
        with pytest.raises(ValueError, match="participants, channels"):
            at.roi_channelwise_snr(np.zeros((2, 3, 40)), baseline_mask)

    def test_rejects_short_baseline(self):
        short = np.zeros(40, dtype=bool)
        short[0] = True
        with pytest.raises(ValueError, match="at least 2"):
            at.roi_channelwise_snr(np.zeros((1, 2, 3, 40)), short)


class TestPolarityWeakCount:
    def test_counts_pairs_below_the_floor(self):
        strength = np.array([[0.05, 0.5], [0.2, 0.01]])
        assert at.polarity_weak_count(strength) == (2, 4)

    def test_floor_is_configurable(self):
        strength = np.array([[0.05, 0.5]])
        assert at.polarity_weak_count(strength, floor=0.6) == (2, 2)


class TestPolarityIsDetermined:
    def test_strong_projection_is_determined(self):
        assert at.polarity_is_determined(np.full((10, 2), 0.9)).tolist() == [True, True]

    def test_weak_projection_is_not(self):
        assert at.polarity_is_determined(np.full((10, 1), 0.05)).tolist() == [False]

    def test_disagreement_across_participants_is_not_the_criterion(self):
        # Participants splitting on sign is exactly what the flip reconciles; it must
        # not by itself mark the anchor as undetermined.
        strength = np.full((12, 1), 0.9)
        assert at.polarity_is_determined(strength).tolist() == [True]


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------


class TestEpochGeometry:
    def test_window_comes_from_the_paradigm(self):
        onsets = np.array([500, 900, 1300])
        _inside, pre, post = at.epoch_geometry(onsets, 2000, 250.0)
        assert pre == AssrEpoch.pre_onset_samples(250.0)
        assert post == AssrEpoch.post_onset_samples(250.0, min_gap=400)

    def test_short_gap_caps_the_post_window(self):
        onsets = np.array([100, 150, 200])
        _inside, _pre, post = at.epoch_geometry(onsets, 1000, 250.0)
        assert post == 50

    def test_drops_onsets_outside_the_axis(self):
        inside, _pre, _post = at.epoch_geometry(np.array([10, 500, 5000]), 1000, 250.0)
        assert inside.tolist() == [10, 500]

    def test_raises_when_no_onset_is_inside(self):
        with pytest.raises(ValueError, match="No stimulus onset"):
            at.epoch_geometry(np.array([5000]), 100, 250.0)


class TestCommonEpochWindow:
    def test_takes_the_shorter_post(self):
        geometries = {
            "a": (np.array([1]), 25, 250),
            "b": (np.array([1]), 25, 200),
        }
        assert at.common_epoch_window(geometries) == (25, 200)

    def test_rejects_disagreeing_pre(self):
        geometries = {
            "a": (np.array([1]), 25, 250),
            "b": (np.array([1]), 30, 250),
        }
        with pytest.raises(ValueError, match="pre-onset baseline"):
            at.common_epoch_window(geometries)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="No condition"):
            at.common_epoch_window({})


class TestCutTrials:
    def test_trial_axis_sits_before_time(self):
        array = np.arange(2 * 3 * 1000, dtype=float).reshape(2, 3, 1000)
        trials, kept = at.cut_trials(array, np.array([100, 400, 700]), 25, 250)
        assert trials.shape == (2, 3, 3, 275)
        assert kept.tolist() == [100, 400, 700]

    def test_windows_hold_the_right_samples(self):
        array = np.arange(1000, dtype=float)[None, :]
        trials, _kept = at.cut_trials(array, np.array([100]), 10, 20)
        assert trials[0, 0].tolist() == list(range(90, 120))

    def test_drops_overhanging_windows_and_reports_what_survived(self):
        array = np.zeros((1, 300))
        # 5 underflows the pre window, 295 overflows the post one.
        trials, kept = at.cut_trials(array, np.array([5, 150, 295]), 25, 50)
        assert kept.tolist() == [150]
        assert trials.shape == (1, 1, 75)

    def test_raises_when_nothing_fits(self):
        with pytest.raises(ValueError, match="fits inside"):
            at.cut_trials(np.zeros((1, 30)), np.array([15]), 25, 250)


class TestEpochTimeBase:
    def test_zero_sits_at_the_onset(self):
        times = at.epoch_time_base(25, 250, 250.0)
        assert times.size == 275
        assert times[25] == pytest.approx(0.0)
        assert times[0] == pytest.approx(-0.1)

    def test_agrees_with_the_stimulus_mask(self):
        times = at.epoch_time_base(25, 250, 250.0)
        mask = AssrEpoch.stimulus_mask(times)
        assert mask.sum() == int(AssrEpoch.STIMULUS_DURATION_S * 250.0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestBaselineNormalise:
    @pytest.fixture
    def baseline_mask(self):
        mask = np.zeros(100, dtype=bool)
        mask[:25] = True
        return mask

    def test_z_is_defined_for_every_trial(self, baseline_mask):
        rng = np.random.default_rng(4)
        trials = rng.normal(size=(3, 2, 5, 100))  # signed, like an IVA source
        z, _rel, _positive = at.baseline_normalise(trials, baseline_mask)
        assert np.isfinite(z).all()

    def test_z_puts_every_trial_baseline_at_zero_mean(self, baseline_mask):
        rng = np.random.default_rng(5)
        trials = rng.normal(3.0, 2.0, size=(2, 1, 4, 100))
        z, _rel, _positive = at.baseline_normalise(trials, baseline_mask)
        # The MEAN is removed per trial, so this holds trial by trial.
        assert np.allclose(z[..., baseline_mask].mean(axis=-1), 0.0, atol=1e-12)

    def test_the_pooled_sd_is_unit_across_trials_not_within_one(self, baseline_mask):
        """What pooling the divisor guarantees, and what it deliberately gives up.

        Dividing by a per-trial SD would force EVERY trial's baseline to unit SD, which
        is what lets one quiet baseline inflate that trial's whole epoch. Pooling makes
        the unit hold across the trial ensemble instead, so a trial that happened to be
        quiet stays quiet rather than being amplified to match its neighbours.
        """
        rng = np.random.default_rng(5)
        trials = rng.normal(3.0, 2.0, size=(2, 1, 8, 100))
        trials[0, 0, 0, :25] *= 0.05  # one trial with a freakishly quiet baseline
        z, _rel, _positive = at.baseline_normalise(trials, baseline_mask)

        per_trial = z[..., baseline_mask].std(axis=-1, ddof=1)
        assert np.allclose(np.sqrt((per_trial**2).mean(axis=-1)), 1.0)  # pooled == 1
        assert per_trial[0, 0, 0] < 0.2  # the quiet trial was NOT rescaled up to 1

    def test_rel_is_nan_where_the_baseline_is_not_positive(self, baseline_mask):
        trials = np.ones((1, 1, 2, 100))
        trials[0, 0, 1, :25] = -1.0  # negative baseline on the second trial
        _z, rel, positive = at.baseline_normalise(trials, baseline_mask)
        assert positive.tolist() == [[[True, False]]]
        assert np.isfinite(rel[0, 0, 0]).all()
        assert np.isnan(rel[0, 0, 1]).all()

    def test_rel_is_the_relative_change_where_valid(self, baseline_mask):
        trials = np.ones((1, 1, 1, 100))
        trials[..., 25:] = 1.5
        _z, rel, _positive = at.baseline_normalise(trials, baseline_mask)
        assert rel[0, 0, 0, 50] == pytest.approx(0.5)

    def test_is_invariant_to_a_per_trial_scale(self, baseline_mask):
        # The reason it removes participant scale: z is scale-free.
        rng = np.random.default_rng(6)
        trials = rng.normal(size=(2, 1, 3, 100))
        scaled = trials * np.array([1.0, 1e6])[:, None, None, None]
        z_plain, _rel, _pos = at.baseline_normalise(trials, baseline_mask)
        z_scaled, _rel2, _pos2 = at.baseline_normalise(scaled, baseline_mask)
        assert np.allclose(z_plain, z_scaled)

    def test_rejects_a_baseline_too_short_for_a_standard_deviation(self):
        mask = np.zeros(10, dtype=bool)
        mask[0] = True
        with pytest.raises(ValueError, match="at least 2"):
            at.baseline_normalise(np.zeros((1, 1, 1, 10)), mask)


# ---------------------------------------------------------------------------
# Tests over participants
# ---------------------------------------------------------------------------


class TestPairedTest:
    def test_reports_the_median_and_the_sign_count(self):
        result = at.paired_test(np.array([1.0, 2.0, 3.0, -1.0]))
        assert result["median"] == pytest.approx(1.5)
        assert result["same sign"] == "3/4"

    def test_all_one_way_hits_the_floor(self):
        result = at.paired_test(np.ones(12))
        assert result["p"] == pytest.approx(at.p_floor(12))

    def test_one_sided_halves_the_floor(self):
        result = at.paired_test(np.ones(12), "greater")
        assert result["p"] == pytest.approx(at.p_floor(12, one_sided=True))

    def test_is_calibrated_under_the_null(self):
        rng = np.random.default_rng(7)
        hits = sum(at.paired_test(rng.normal(size=12))["p"] <= 0.05 for _ in range(400))
        assert hits / 400 <= 0.10  # nominal 0.05, generous for 400 draws

    def test_absolute_differences_would_not_be_calibrated(self):
        # Guards the reasoning in the docstring: |d| >= 0 by construction, so a test
        # of it fires on pure noise. This is why the module never offers one.
        rng = np.random.default_rng(8)
        hits = sum(
            at.paired_test(np.abs(rng.normal(size=12)))["p"] <= 0.05 for _ in range(50)
        )
        assert hits == 50

    def test_ignores_non_finite_values(self):
        result = at.paired_test(np.array([1.0, np.nan, 2.0, 3.0]))
        assert result["same sign"] == "3/3"

    def test_rejects_too_few_participants(self):
        with pytest.raises(ValueError, match="at least 2"):
            at.paired_test(np.array([1.0]))


class TestPFloor:
    def test_two_sided(self):
        assert at.p_floor(12) == pytest.approx(2 / 4096)

    def test_one_sided_is_half(self):
        assert at.p_floor(12, one_sided=True) == pytest.approx(1 / 4096)


class TestContrasts:
    @pytest.fixture
    def values(self):
        return {
            "Placebo": np.array([[3.0, 1.0], [5.0, 2.0]]),
            "Psilocybin": np.array([[1.0, 1.0], [2.0, 3.0]]),
        }

    def test_condition_contrast_subtracts_in_order(self, values):
        got = at.condition_contrast(values, ["Placebo", "Psilocybin"], 0)
        assert got.tolist() == [2.0, 3.0]

    def test_discrimination_gain_is_the_interaction(self, values):
        got = at.discrimination_gain(values, ["Placebo", "Psilocybin"], 0, 1)
        # (3-1) - (1-1) = 2 ; (5-2) - (2-3) = 4
        assert got.tolist() == [2.0, 4.0]

    def test_interaction_equals_the_difference_of_within_condition_gaps(self, values):
        conditions = ["Placebo", "Psilocybin"]
        interaction = at.discrimination_gain(values, conditions, 0, 1)
        within = (values["Placebo"][:, 0] - values["Placebo"][:, 1]) - (
            values["Psilocybin"][:, 0] - values["Psilocybin"][:, 1]
        )
        assert np.allclose(interaction, within)

    def test_reference_against_itself_is_zero(self, values):
        got = at.discrimination_gain(values, ["Placebo", "Psilocybin"], 1, 1)
        assert np.allclose(got, 0.0)


# ---------------------------------------------------------------------------
# The stored product
# ---------------------------------------------------------------------------


@pytest.fixture
def trial_set():
    rng = np.random.default_rng(9)
    conditions = ["Placebo", "Psilocybin"]
    times = at.epoch_time_base(25, 50, 250.0)
    # Deliberately different trial counts, which is why the file keys per condition.
    counts = {"Placebo": 4, "Psilocybin": 6}
    trials, z, rel, positive, onsets = {}, {}, {}, {}, {}
    for condition in conditions:
        raw = rng.normal(size=(3, 2, counts[condition], times.size))
        trials[condition] = raw
        z[condition], rel[condition], positive[condition] = at.baseline_normalise(
            raw, times < 0.0
        )
        onsets[condition] = np.arange(counts[condition]) * 300 + 100
    return at.AssrTrialSet(
        trials=trials,
        trials_z=z,
        trials_rel=rel,
        baseline_positive=positive,
        onsets=onsets,
        participants=("019", "023", "026"),
        labels=("IC 1", at.BINARY_FILTER_LABEL),
        times=times,
        sfreq=250.0,
        freqs=np.array([40.0]),
        selection="40hz",
        binary_channels=("E4", "E5"),
        metadata={"units": "wavelet power"},
    )


class TestAssrTrialSet:
    def test_axes(self, trial_set):
        assert trial_set.conditions == ("Placebo", "Psilocybin")
        assert trial_set.n_participants == 3
        assert trial_set.n_sources == 2

    def test_source_index(self, trial_set):
        assert trial_set.source_index("IC 1") == 0
        assert trial_set.reference_index == 1

    def test_unknown_source_raises(self, trial_set):
        with pytest.raises(KeyError, match="No source labelled"):
            trial_set.source_index("IC 9")

    def test_masks_partition_the_epoch(self, trial_set):
        assert not (trial_set.baseline_mask() & trial_set.stimulus_mask()).any()
        assert trial_set.baseline_mask().sum() == 25

    def test_participant_frame(self, trial_set):
        frame = trial_set.participant_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 6  # 2 conditions x 3 participants
        placebo = frame[frame["condition"] == "Placebo"]
        assert placebo["n_trials"].unique().tolist() == [4]


class TestSaveLoadRoundTrip:
    def test_arrays_survive(self, trial_set, tmp_path):
        path = at.save_assr_trials(tmp_path / "trials.npz", trial_set)
        loaded = at.load_assr_trials(path)
        for condition in trial_set.conditions:
            assert np.allclose(loaded.trials[condition], trial_set.trials[condition])
            assert np.allclose(
                loaded.trials_z[condition], trial_set.trials_z[condition]
            )
            assert np.allclose(
                loaded.trials_rel[condition],
                trial_set.trials_rel[condition],
                equal_nan=True,
            )
            assert loaded.onsets[condition].tolist() == (
                trial_set.onsets[condition].tolist()
            )

    def test_bookkeeping_survives(self, trial_set, tmp_path):
        loaded = at.load_assr_trials(
            at.save_assr_trials(tmp_path / "trials.npz", trial_set)
        )
        assert loaded.participants == trial_set.participants
        assert loaded.labels == trial_set.labels
        assert loaded.selection == "40hz"
        assert loaded.binary_channels == ("E4", "E5")
        assert loaded.metadata["units"] == "wavelet power"
        assert np.allclose(loaded.times, trial_set.times)

    def test_differing_trial_counts_survive(self, trial_set, tmp_path):
        loaded = at.load_assr_trials(
            at.save_assr_trials(tmp_path / "trials.npz", trial_set)
        )
        assert loaded.trials["Placebo"].shape[2] == 4
        assert loaded.trials["Psilocybin"].shape[2] == 6

    def test_creates_parent_directories(self, trial_set, tmp_path):
        path = at.save_assr_trials(tmp_path / "deep" / "er" / "trials.npz", trial_set)
        assert path.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No trial file"):
            at.load_assr_trials(tmp_path / "absent.npz")


class TestTrialsFilename:
    def test_carries_every_setting_that_decides_the_content(self):
        name = at.trials_filename("iva_channel_joined_tracks", "ASSR", "40hz", 5)
        assert name == "assr_trials__iva_channel_joined_tracks__ASSR__40hz__pca5.npz"

    def test_selections_do_not_collide(self):
        first = at.trials_filename("v", "ASSR", "40hz", 5)
        second = at.trials_filename("v", "ASSR", "35-45hz", 5)
        assert first != second


class TestResolveConditions:
    def test_accepts_enums_and_strings(self):
        from src.definitions.fields import ConditionVariants

        got = at.resolve_conditions([ConditionVariants.PLACEBO, "Psilocybin"])
        assert got == ["Placebo", "Psilocybin"]


class TestReferenceSnrTests:
    """The third paired family: is a learned source below the fixed reference?"""

    def _value(self):
        # Three participants; IC 1 sits above the reference, IC 2 below it.
        return np.array(
            [
                [2.0, 0.2, 1.0],
                [2.5, 0.1, 1.2],
                [2.2, 0.3, 1.1],
            ]
        )

    def test_one_record_per_non_reference_source(self):
        labels = ["IC 1", "IC 2", at.BINARY_FILTER_LABEL]
        records = at.reference_snr_tests(
            self._value(), labels, condition="Placebo", alternative="less"
        )
        assert [row["source"] for row in records] == ["IC 1", "IC 2"]
        assert {row["condition"] for row in records} == {"Placebo"}

    def test_the_below_reference_count_reads_the_right_direction(self):
        labels = ["IC 1", "IC 2", at.BINARY_FILTER_LABEL]
        records = at.reference_snr_tests(self._value(), labels, condition="Placebo")
        by_source = {row["source"]: row for row in records}
        # IC 1 is above the reference for all three, IC 2 below for all three.
        assert by_source["IC 1"]["IC<ref"] == "0/3"
        assert by_source["IC 2"]["IC<ref"] == "3/3"
        assert by_source["IC 2"]["median"] < 0 < by_source["IC 1"]["median"]

    def test_the_reference_column_can_be_named_explicitly(self):
        labels = [at.BINARY_FILTER_LABEL, "IC 1", "IC 2"]
        value = self._value()[:, [2, 0, 1]]
        records = at.reference_snr_tests(
            value, labels, condition="Placebo", reference_index=0
        )
        assert [row["source"] for row in records] == ["IC 1", "IC 2"]

    def test_a_mismatched_value_array_is_refused(self):
        with pytest.raises(ValueError, match="to match labels"):
            at.reference_snr_tests(
                np.zeros((3, 2)), ["a", "b", "c"], condition="Placebo"
            )
