"""
Tests for src/analysis/pca_polarity.py — cross-participant PC1 sign alignment.

Everything here is pure array maths, so the cases are small synthetic loadings
whose correct orientation is known by construction.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.analysis.pca_polarity import (
    REFERENCE_CHANNEL,
    SignConsistency,
    align_pc1_signs,
    apply_pc1_signs,
    topography_consistency,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def shared_mode(n_subj=12, n_ch=24, noise=0.05, seed=0):
    """One shared topography per subject, with random per-subject sign flips."""
    rng = np.random.default_rng(seed)
    pattern = rng.normal(size=n_ch)
    truth = rng.choice([-1.0, 1.0], size=n_subj)
    loadings = pattern[None, :] * truth[:, None] + noise * rng.normal(
        size=(n_subj, n_ch)
    )
    return loadings, truth, pattern


# ---------------------------------------------------------------------------
# align_pc1_signs — the core job: make subjects agree with each other
# ---------------------------------------------------------------------------


def test_recovers_the_flips_that_were_applied():
    loadings, truth, _ = shared_mode()
    signs, _ = align_pc1_signs(loadings)
    aligned = apply_pc1_signs(loadings, signs)
    # Up to one arbitrary global orientation, the recovered signs match the truth.
    assert np.all(signs == truth) or np.all(signs == -truth)
    # Every subject now correlates positively with the group mean.
    group = aligned.mean(axis=0)
    assert np.all([np.corrcoef(row, group)[0, 1] > 0 for row in aligned])


def test_alignment_beats_leaving_signs_unresolved():
    """The point of aligning: the group mean keeps its amplitude instead of cancelling."""
    rng = np.random.default_rng(17)
    pattern = rng.normal(size=24)
    # Exactly balanced flips, so the unresolved mean cancels to ~noise.
    truth = np.array([1.0] * 8 + [-1.0] * 8)
    loadings = pattern[None, :] * truth[:, None] + 0.05 * rng.normal(size=(16, 24))
    signs, _ = align_pc1_signs(loadings)
    aligned_amp = np.linalg.norm(apply_pc1_signs(loadings, signs).mean(axis=0))
    unresolved_amp = np.linalg.norm(loadings.mean(axis=0))
    assert aligned_amp > 10.0 * unresolved_amp


def test_anti_correlated_subject_is_flipped_not_kept():
    """A subject that is simply inverted must end up agreeing with the rest."""
    rng = np.random.default_rng(1)
    pattern = rng.normal(size=20)
    loadings = np.vstack([pattern] * 5 + [-pattern])
    signs, _ = align_pc1_signs(loadings)
    aligned = apply_pc1_signs(loadings, signs)
    assert signs[5] != signs[0]
    np.testing.assert_allclose(aligned[5], aligned[0])


def test_signs_do_not_depend_on_subject_order():
    loadings, _, _ = shared_mode(n_subj=10, seed=2)
    names = [f"E{i}" for i in range(loadings.shape[1])]
    order = np.random.default_rng(3).permutation(10)
    a, _ = align_pc1_signs(loadings, channel_names=names)
    b, _ = align_pc1_signs(loadings[order], channel_names=names)
    np.testing.assert_array_equal(a[order], b)


def test_reference_channel_anchors_the_overall_orientation():
    """The anchored channel reads positive in the aligned group mean."""
    loadings, _, _ = shared_mode(n_subj=8, n_ch=6, seed=4)
    names = ["E1", "E2", REFERENCE_CHANNEL, "E4", "E5", "E6"]
    signs, anchor = align_pc1_signs(loadings, channel_names=names)
    assert anchor == REFERENCE_CHANNEL
    group = apply_pc1_signs(loadings, signs).mean(axis=0)
    assert group[names.index(REFERENCE_CHANNEL)] > 0


def test_anchor_is_flip_invariant():
    """Inverting the whole input must not change the anchored output."""
    loadings, _, _ = shared_mode(n_subj=8, n_ch=6, seed=5)
    names = ["E1", "E2", REFERENCE_CHANNEL, "E4", "E5", "E6"]
    s1, _ = align_pc1_signs(loadings, channel_names=names)
    s2, _ = align_pc1_signs(-loadings, channel_names=names)
    np.testing.assert_allclose(
        apply_pc1_signs(loadings, s1), apply_pc1_signs(-loadings, s2)
    )


def test_missing_reference_falls_back_to_strongest_channel():
    loadings, _, _ = shared_mode(n_subj=6, n_ch=8, seed=6)
    names = [f"E{i}" for i in range(8)]
    signs, anchor = align_pc1_signs(loadings, channel_names=names)
    assert anchor in names
    group = apply_pc1_signs(loadings, signs).mean(axis=0)
    assert group[names.index(anchor)] > 0


def test_without_channel_names_no_anchor_is_reported():
    loadings, _, _ = shared_mode(n_subj=5, seed=7)
    signs, anchor = align_pc1_signs(loadings)
    assert anchor is None
    assert set(np.unique(signs)).issubset({-1.0, 1.0})


def test_dipolar_loadings_are_still_aligned():
    """The case a per-participant sum rule gets wrong: near-zero-sum topographies."""
    rng = np.random.default_rng(8)
    half = rng.normal(size=10)
    pattern = np.concatenate([half, -half])  # sums to ~0
    truth = rng.choice([-1.0, 1.0], size=14)
    loadings = pattern[None, :] * truth[:, None] + 0.02 * rng.normal(size=(14, 20))
    assert abs(loadings.sum(axis=1)).max() < 0.5 * np.abs(loadings).sum(axis=1).min()
    signs, _ = align_pc1_signs(loadings)
    assert np.all(signs == truth) or np.all(signs == -truth)


def test_signs_are_exactly_plus_or_minus_one():
    loadings, _, _ = shared_mode(n_subj=30, seed=9)
    signs, _ = align_pc1_signs(loadings)
    assert set(np.unique(signs)).issubset({-1.0, 1.0})


def test_single_subject_is_handled():
    signs, _ = align_pc1_signs(np.array([[1.0, -2.0, 3.0]]))
    assert signs.shape == (1,)


def test_input_is_not_mutated():
    loadings, _, _ = shared_mode(n_subj=4, seed=10)
    before = loadings.copy()
    signs, _ = align_pc1_signs(loadings)
    apply_pc1_signs(loadings, signs)
    topography_consistency(loadings)
    np.testing.assert_array_equal(loadings, before)


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((0, 4)), np.zeros((2, 0))])
def test_align_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        align_pc1_signs(bad)


def test_align_rejects_channel_name_length_mismatch():
    with pytest.raises(ValueError):
        align_pc1_signs(np.zeros((3, 4)), channel_names=["a", "b"])


# ---------------------------------------------------------------------------
# apply_pc1_signs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trailing", [(5,), (4, 7), (3, 4, 5)])
def test_broadcasts_over_any_trailing_shape(trailing):
    """Same call orients loadings (S, C), time courses (S, T) and TF maps (S, F, T)."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=(3, *trailing))
    out = apply_pc1_signs(values, np.array([1.0, -1.0, 1.0]))
    assert out.shape == values.shape
    np.testing.assert_allclose(out[0], values[0])
    np.testing.assert_allclose(out[1], -values[1])


def test_loading_and_score_stay_consistent():
    """Flipping both with the same signs preserves loading-score products."""
    rng = np.random.default_rng(12)
    loadings, scores = rng.normal(size=(6, 12)), rng.normal(size=(6, 40))
    signs, _ = align_pc1_signs(loadings)
    before = loadings.sum(axis=1) * scores.sum(axis=1)
    after = apply_pc1_signs(loadings, signs).sum(axis=1) * apply_pc1_signs(
        scores, signs
    ).sum(axis=1)
    np.testing.assert_allclose(before, after)


def test_apply_rejects_subject_axis_mismatch():
    with pytest.raises(ValueError):
        apply_pc1_signs(np.zeros((3, 4)), np.array([1.0, -1.0]))


def test_apply_rejects_non_1d_signs():
    with pytest.raises(ValueError):
        apply_pc1_signs(np.zeros((2, 4)), np.ones((2, 1)))


# ---------------------------------------------------------------------------
# topography_consistency
# ---------------------------------------------------------------------------


def test_all_subjects_agree_after_alignment():
    loadings, _, _ = shared_mode(n_subj=12, seed=13)
    signs, _ = align_pc1_signs(loadings)
    result = topography_consistency(apply_pc1_signs(loadings, signs))
    assert isinstance(result, SignConsistency)
    assert result.n_agreeing == result.n_subjects == 12
    assert result.median_pairwise_r > 0.8
    assert result.min_subject_r > 0.0


def test_unaligned_loadings_are_reported_as_disagreeing():
    loadings, _, _ = shared_mode(n_subj=12, seed=14)
    result = topography_consistency(loadings)  # deliberately NOT aligned
    assert result.n_agreeing < result.n_subjects
    assert result.min_subject_r < 0.0


def test_genuine_outlier_survives_alignment_and_is_flagged():
    """An unrelated topography is not a sign problem; it must show up as one short."""
    rng = np.random.default_rng(15)
    pattern = rng.normal(size=30)
    loadings = np.vstack([pattern + 0.01 * rng.normal(size=30) for _ in range(9)])
    orthogonal = rng.normal(size=30)
    orthogonal -= orthogonal @ pattern / (pattern @ pattern) * pattern
    loadings = np.vstack([loadings, orthogonal])
    signs, _ = align_pc1_signs(loadings)
    result = topography_consistency(apply_pc1_signs(loadings, signs))
    assert result.min_subject_r < result.median_pairwise_r


def test_constant_loading_counts_as_non_agreeing_without_raising():
    loadings = np.vstack([np.array([1.0, 2.0, 3.0]), np.zeros(3)])
    result = topography_consistency(loadings)
    assert result.n_agreeing == 1
    assert result.n_subjects == 2


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((3, 1))])
def test_consistency_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        topography_consistency(bad)


# ---------------------------------------------------------------------------
# Integration with an actual PCA fit
# ---------------------------------------------------------------------------


def test_aligns_real_pca_output():
    """Subjects sharing one spatial mode end with one polarity after alignment."""
    rng = np.random.default_rng(16)
    n_ch, n_time = 20, 300
    topo = rng.normal(size=n_ch)
    loadings, scores = [], []
    for _ in range(10):
        source = np.sin(2 * np.pi * 8 * np.arange(n_time) / 250.0)
        data = np.outer(topo, source) + 0.05 * rng.normal(size=(n_ch, n_time))
        pca = PCA(n_components=1)
        scores.append(pca.fit_transform(data.T)[:, 0])
        loadings.append(pca.components_[0])
    loadings, scores = np.array(loadings), np.array(scores)
    # PCA happened to agree on a sign here, so flip half the subjects to create
    # the ambiguity that real data has.
    scrambled = np.array([1.0, -1.0] * 5)
    loadings = apply_pc1_signs(loadings, scrambled)
    scores = apply_pc1_signs(scores, scrambled)

    signs, _ = align_pc1_signs(loadings)
    result = topography_consistency(apply_pc1_signs(loadings, signs))
    assert result.n_agreeing == 10
    assert result.median_pairwise_r > 0.9
    # Aligned scores average coherently instead of cancelling.
    aligned_scores = apply_pc1_signs(scores, signs)
    assert (
        np.abs(aligned_scores.mean(axis=0)).max()
        > 5.0 * np.abs(scores.mean(axis=0)).max()
    )
