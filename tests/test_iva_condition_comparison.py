"""
Tests for src/analysis/iva_condition_comparison.py — TF-PC1 sign alignment and
per-condition aggregation for a subject-axis-pooled IVA run.

The alignment's job is to make per-recording maps agree in polarity so they can be
averaged and contrasted, so most tests assert on that outcome — recordings agreeing
with their group mean, the mean not cancelling, signs recovered up to one global sign
per component — rather than on particular numbers.
"""

import numpy as np
import pytest

from src.analysis.iva_condition_comparison import (
    TfPc1Polarity,
    align_tf_pc1_signs,
    apply_component_signs,
    condition_component_means,
    condition_difference,
    stack_conditions_on_subject_axis,
)

N_SUBJECTS, N_COMPONENTS, N_FREQS, N_TIMES, N_CHANNELS = 6, 3, 5, 8, 4


@pytest.fixture
def shared():
    """A strong shared ``(K, F, T)`` map per component."""
    return (
        np.random.default_rng(0).standard_normal((N_COMPONENTS, N_FREQS, N_TIMES)) * 5
    )


@pytest.fixture
def sources(shared):
    """Aligned per-recording maps: the shared map plus a little noise."""
    noise = np.random.default_rng(1).standard_normal(
        (N_SUBJECTS, N_COMPONENTS, N_FREQS, N_TIMES)
    )
    return shared[None] + noise * 0.3


@pytest.fixture
def true_flips():
    """Arbitrary per-``(recording, component)`` signs, as IVA would leave them."""
    return np.random.default_rng(2).choice([-1.0, 1.0], size=(N_SUBJECTS, N_COMPONENTS))


@pytest.fixture
def scrambled(sources, true_flips):
    return sources * true_flips[:, :, None, None]


@pytest.fixture
def patterns(true_flips):
    """Channel patterns carrying the same per-recording signs as the maps."""
    base = np.random.default_rng(3).standard_normal(
        (N_SUBJECTS, N_COMPONENTS, N_CHANNELS)
    )
    return base * true_flips[:, :, None]


@pytest.fixture
def conditions():
    return ["Placebo", "Psilocybin"]


@pytest.fixture
def subject_conditions():
    return ["Placebo"] * 3 + ["Psilocybin"] * 3


class TestAlignTfPc1Signs:
    def test_returns_the_documented_shapes(self, scrambled):
        polarity = align_tf_pc1_signs(scrambled)
        assert isinstance(polarity, TfPc1Polarity)
        assert polarity.signs.shape == (N_SUBJECTS, N_COMPONENTS)
        assert polarity.loadings.shape == (N_SUBJECTS, N_COMPONENTS)
        assert polarity.pc1.shape == (N_COMPONENTS, N_FREQS, N_TIMES)
        assert polarity.explained_variance_ratio.shape == (N_COMPONENTS,)

    def test_signs_are_plus_or_minus_one(self, scrambled):
        assert set(np.unique(align_tf_pc1_signs(scrambled).signs)) <= {-1.0, 1.0}

    def test_every_recording_agrees_with_its_component_mean(self, scrambled):
        """The outcome the alignment exists for."""
        polarity = align_tf_pc1_signs(scrambled)
        aligned = apply_component_signs(scrambled, polarity.signs)
        for k in range(N_COMPONENTS):
            mean = aligned[:, k].mean(axis=0).ravel()
            for s in range(N_SUBJECTS):
                assert np.corrcoef(aligned[s, k].ravel(), mean)[0, 1] > 0.5

    def test_the_aligned_mean_recovers_the_shared_map(self, scrambled, shared):
        polarity = align_tf_pc1_signs(scrambled)
        aligned = apply_component_signs(scrambled, polarity.signs)
        for k in range(N_COMPONENTS):
            mean = aligned[:, k].mean(axis=0)
            assert abs(np.corrcoef(mean.ravel(), shared[k].ravel())[0, 1]) > 0.99
            # Not cancelled: the mean keeps the shared map's amplitude.
            assert np.abs(mean).mean() > 0.9 * np.abs(shared[k]).mean()

    def test_aligning_never_weakens_the_mean(self, scrambled):
        polarity = align_tf_pc1_signs(scrambled)
        aligned = apply_component_signs(scrambled, polarity.signs)
        for k in range(N_COMPONENTS):
            assert (
                np.abs(aligned[:, k].mean(axis=0)).mean()
                >= np.abs(scrambled[:, k].mean(axis=0)).mean() - 1e-12
            )

    def test_is_idempotent(self, scrambled):
        polarity = align_tf_pc1_signs(scrambled)
        aligned = apply_component_signs(scrambled, polarity.signs)
        assert align_tf_pc1_signs(aligned).n_flipped == 0

    def test_recovers_the_true_signs_up_to_one_global_sign_per_component(
        self, scrambled, true_flips
    ):
        """The group's overall orientation is free; the *relative* signs are not."""
        polarity = align_tf_pc1_signs(scrambled)
        ratio = polarity.signs * true_flips
        for k in range(N_COMPONENTS):
            assert len(set(ratio[:, k])) == 1

    def test_pc1_is_anchored_strongest_bin_positive(self, scrambled):
        pc1 = align_tf_pc1_signs(scrambled).pc1
        for k in range(N_COMPONENTS):
            flat = pc1[k].ravel()
            assert flat[int(np.argmax(np.abs(flat)))] > 0

    def test_is_independent_of_recording_order(self, scrambled):
        """A reproducible anchor must not depend on how recordings happen to arrive."""
        polarity = align_tf_pc1_signs(scrambled)
        perm = np.random.default_rng(4).permutation(N_SUBJECTS)
        permuted = align_tf_pc1_signs(scrambled[perm])
        np.testing.assert_allclose(permuted.pc1, polarity.pc1, atol=1e-10)
        np.testing.assert_allclose(permuted.signs, polarity.signs[perm])

    def test_loadings_are_the_pre_flip_projections(self, scrambled):
        polarity = align_tf_pc1_signs(scrambled)
        # signs invert exactly the negative loadings, nothing else.
        expected = np.where(polarity.loadings < 0, -1.0, 1.0)
        np.testing.assert_array_equal(polarity.signs, expected)

    def test_explained_variance_is_high_for_a_dominant_shared_map(self, scrambled):
        evr = align_tf_pc1_signs(scrambled).explained_variance_ratio
        assert (evr > 0.9).all(), evr

    def test_explained_variance_is_low_for_unstructured_maps(self):
        """The diagnostic must say so when there is no shared map to align to."""
        noise = np.random.default_rng(5).standard_normal(
            (N_SUBJECTS, 1, N_FREQS, N_TIMES)
        )
        assert align_tf_pc1_signs(noise).explained_variance_ratio[0] < 0.6

    def test_flip_counters(self, scrambled):
        polarity = align_tf_pc1_signs(scrambled)
        assert polarity.n_flipped == int((polarity.signs < 0).sum())
        np.testing.assert_array_equal(
            polarity.flipped_per_component(), (polarity.signs < 0).sum(axis=0)
        )
        assert polarity.flipped_per_component().shape == (N_COMPONENTS,)

    def test_an_all_zero_component_is_left_alone(self):
        polarity = align_tf_pc1_signs(np.zeros((3, 2, 2, 2)))
        assert polarity.n_flipped == 0
        assert polarity.explained_variance_ratio.tolist() == [0.0, 0.0]

    def test_inputs_are_not_mutated(self, scrambled):
        before = scrambled.copy()
        align_tf_pc1_signs(scrambled)
        np.testing.assert_array_equal(scrambled, before)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match=r"must be \(S, K, F, T\)"):
            align_tf_pc1_signs(np.zeros((3, 2, 4)))

    def test_empty_axis_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            align_tf_pc1_signs(np.zeros((3, 0, 4, 5)))


class TestApplyComponentSigns:
    def test_orients_tf_maps(self, scrambled):
        signs = align_tf_pc1_signs(scrambled).signs
        out = apply_component_signs(scrambled, signs)
        np.testing.assert_allclose(out, scrambled * signs[:, :, None, None])

    def test_orients_channel_patterns_with_the_same_signs(self, scrambled, patterns):
        """Map and topography must never disagree about which way is up."""
        signs = align_tf_pc1_signs(scrambled).signs
        out = apply_component_signs(patterns, signs)
        np.testing.assert_allclose(out, patterns * signs[:, :, None])

    def test_orients_marginals(self, scrambled):
        signs = align_tf_pc1_signs(scrambled).signs
        marginal = scrambled.mean(axis=2)  # (S, K, T)
        np.testing.assert_allclose(
            apply_component_signs(marginal, signs), marginal * signs[:, :, None]
        )

    def test_orients_a_bare_two_axis_array(self, scrambled):
        signs = align_tf_pc1_signs(scrambled).signs
        flat = scrambled[..., 0, 0]
        np.testing.assert_allclose(apply_component_signs(flat, signs), flat * signs)

    def test_inputs_are_not_mutated(self, patterns, scrambled):
        before = patterns.copy()
        apply_component_signs(patterns, align_tf_pc1_signs(scrambled).signs)
        np.testing.assert_array_equal(patterns, before)

    def test_applying_twice_is_the_identity(self, patterns, scrambled):
        signs = align_tf_pc1_signs(scrambled).signs
        once = apply_component_signs(patterns, signs)
        np.testing.assert_allclose(apply_component_signs(once, signs), patterns)

    def test_mismatched_leading_axes_raise(self, patterns):
        with pytest.raises(ValueError, match="leading axes"):
            apply_component_signs(patterns, np.ones((N_SUBJECTS, N_COMPONENTS + 1)))

    def test_non_2d_signs_raise(self, patterns):
        with pytest.raises(ValueError, match=r"signs must be \(S, K\)"):
            apply_component_signs(patterns, np.ones(N_SUBJECTS))

    def test_too_few_axes_raise(self):
        with pytest.raises(ValueError, match=r"values must be \(S, K, \.\.\.\)"):
            apply_component_signs(np.zeros(3), np.ones((3, 1)))


class TestConditionComponentMeans:
    def test_averages_within_each_condition(
        self, sources, subject_conditions, conditions
    ):
        means = condition_component_means(sources, subject_conditions, conditions)
        np.testing.assert_allclose(means["Placebo"], sources[:3].mean(axis=0))
        np.testing.assert_allclose(means["Psilocybin"], sources[3:].mean(axis=0))

    def test_drops_the_recording_axis(self, sources, subject_conditions, conditions):
        means = condition_component_means(sources, subject_conditions, conditions)
        assert means["Placebo"].shape == (N_COMPONENTS, N_FREQS, N_TIMES)

    def test_key_order_follows_the_conditions_argument(
        self, sources, subject_conditions
    ):
        means = condition_component_means(
            sources, subject_conditions, ["Psilocybin", "Placebo"]
        )
        assert list(means) == ["Psilocybin", "Placebo"]

    def test_none_uses_first_appearance_order(self, sources, subject_conditions):
        means = condition_component_means(sources, subject_conditions)
        assert list(means) == ["Placebo", "Psilocybin"]

    def test_works_on_patterns_too(self, patterns, subject_conditions, conditions):
        means = condition_component_means(patterns, subject_conditions, conditions)
        assert means["Placebo"].shape == (N_COMPONENTS, N_CHANNELS)

    def test_mismatched_bookkeeping_raises(self, sources, conditions):
        with pytest.raises(ValueError, match="subject_conditions has"):
            condition_component_means(sources, ["Placebo"] * 2, conditions)

    def test_absent_condition_raises(self, sources, subject_conditions):
        with pytest.raises(ValueError, match="has no recordings"):
            condition_component_means(sources, subject_conditions, ["Joined"])


class TestConditionDifference:
    def test_is_second_minus_first(self, sources, subject_conditions, conditions):
        means = condition_component_means(sources, subject_conditions, conditions)
        np.testing.assert_allclose(
            condition_difference(means, conditions),
            means["Psilocybin"] - means["Placebo"],
        )

    def test_order_flips_the_sign(self, sources, subject_conditions, conditions):
        means = condition_component_means(sources, subject_conditions, conditions)
        np.testing.assert_allclose(
            condition_difference(means, conditions),
            -condition_difference(means, list(reversed(conditions))),
        )

    def test_needs_exactly_two_conditions(
        self, sources, subject_conditions, conditions
    ):
        means = condition_component_means(sources, subject_conditions, conditions)
        with pytest.raises(ValueError, match="exactly two conditions"):
            condition_difference(means, conditions[:1])

    def test_missing_mean_raises(self, sources, subject_conditions, conditions):
        means = condition_component_means(sources, subject_conditions, conditions)
        with pytest.raises(ValueError, match="No mean for condition"):
            condition_difference(means, ["Placebo", "Joined"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="disagree in shape"):
            condition_difference(
                {"a": np.zeros((2, 3)), "b": np.zeros((2, 4))}, ["a", "b"]
            )


class TestStackConditionsOnSubjectAxis:
    """The bridge from a time-axis join's split back to the subject-axis layout."""

    PARTICIPANTS = ["003", "007", "011"]

    @pytest.fixture
    def per_condition(self):
        rng = np.random.default_rng(0)
        return {
            "Placebo": rng.standard_normal((3, 2, 4, 20)),
            "Psilocybin": rng.standard_normal((3, 2, 4, 25)),
        }

    def test_stacks_one_block_per_condition(self, per_condition):
        stacked, participants, conditions = stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        assert stacked.shape == (6, 2, 4, 20)
        assert participants == self.PARTICIPANTS * 2
        assert conditions == ["Placebo"] * 3 + ["Psilocybin"] * 3

    def test_trims_to_the_shorter_condition(self, per_condition):
        """The tracks keep their own time bases, so stacking needs a common axis."""
        stacked, _p, _c = stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        assert stacked.shape[-1] == 20
        np.testing.assert_allclose(stacked[3:], per_condition["Psilocybin"][..., :20])

    def test_rows_keep_their_per_condition_order(self, per_condition):
        stacked, _p, _c = stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        np.testing.assert_allclose(stacked[:3], per_condition["Placebo"][..., :20])

    def test_block_order_follows_the_conditions_argument(self, per_condition):
        _s, _p, conditions = stack_conditions_on_subject_axis(
            per_condition, ["Psilocybin", "Placebo"], self.PARTICIPANTS
        )
        assert conditions[0] == "Psilocybin"

    def test_the_result_is_participant_paired(self, per_condition):
        """Row k and row k + P must be the same participant."""
        _s, participants, conditions = stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        n = len(self.PARTICIPANTS)
        assert participants[:n] == participants[n:]
        assert len(set(conditions[:n])) == 1

    def test_works_on_a_marginal_with_no_frequency_axis(self):
        per_condition = {
            "Placebo": np.zeros((3, 2, 20)),
            "Psilocybin": np.zeros((3, 2, 20)),
        }
        stacked, _p, _c = stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        assert stacked.shape == (6, 2, 20)

    def test_a_single_condition_is_a_passthrough(self, per_condition):
        stacked, participants, conditions = stack_conditions_on_subject_axis(
            per_condition, ["Placebo"], self.PARTICIPANTS
        )
        assert stacked.shape == per_condition["Placebo"].shape
        assert participants == self.PARTICIPANTS
        assert conditions == ["Placebo"] * 3

    def test_inputs_are_not_mutated(self, per_condition):
        before = {k: v.copy() for k, v in per_condition.items()}
        stack_conditions_on_subject_axis(
            per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
        )
        for key, original in before.items():
            np.testing.assert_array_equal(per_condition[key], original)

    def test_missing_condition_raises(self, per_condition):
        with pytest.raises(ValueError, match="No array supplied"):
            stack_conditions_on_subject_axis(
                {"Placebo": per_condition["Placebo"]},
                ["Placebo", "Psilocybin"],
                self.PARTICIPANTS,
            )

    def test_disagreeing_non_time_axes_raise(self, per_condition):
        per_condition["Psilocybin"] = per_condition["Psilocybin"][:, :1]
        with pytest.raises(ValueError, match="non-time axes"):
            stack_conditions_on_subject_axis(
                per_condition, ["Placebo", "Psilocybin"], self.PARTICIPANTS
            )

    def test_participant_count_must_match_the_rows(self, per_condition):
        with pytest.raises(ValueError, match="participant label"):
            stack_conditions_on_subject_axis(
                per_condition, ["Placebo", "Psilocybin"], ["003"]
            )

    def test_no_conditions_raises(self, per_condition):
        with pytest.raises(ValueError, match="At least one condition"):
            stack_conditions_on_subject_axis(per_condition, [], self.PARTICIPANTS)
