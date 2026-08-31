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

    def test_applying_the_flip_orients_every_participant(self, mask):
        rng = np.random.default_rng(3)
        patterns = rng.normal(size=(8, 2, 10))
        flip, _ = at.polarity_flip(patterns, mask)
        oriented = (patterns * flip[:, :, None])[:, :, mask].mean(axis=2)
        assert (oriented >= 0).all()

    def test_strength_is_relative_to_the_whole_pattern(self, mask):
        patterns = np.zeros((1, 1, 10))
        patterns[0, 0, :4] = 1.0  # only the mask channels carry weight
        _flip, strength = at.polarity_flip(patterns, mask)
        # mean over mask = 1.0; mean |.| over all = 0.4
        assert strength[0, 0] == pytest.approx(2.5)

    def test_rejects_mask_mismatch(self):
        with pytest.raises(ValueError, match="electrode_mask"):
            at.polarity_flip(np.zeros((2, 1, 10)), np.ones(5, dtype=bool))


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

    def test_z_puts_the_baseline_at_zero_mean_unit_sd(self, baseline_mask):
        rng = np.random.default_rng(5)
        trials = rng.normal(3.0, 2.0, size=(2, 1, 4, 100))
        z, _rel, _positive = at.baseline_normalise(trials, baseline_mask)
        assert np.allclose(z[..., baseline_mask].mean(axis=-1), 0.0, atol=1e-12)
        assert np.allclose(z[..., baseline_mask].std(axis=-1, ddof=1), 1.0)

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
