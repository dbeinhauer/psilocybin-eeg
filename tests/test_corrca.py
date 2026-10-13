"""
Tests for src/analysis/corrca.py — Correlated Component Analysis across subjects.

The data are synthetic: one source shared by every subject, projected through a
common topography, buried in subject-specific noise. CorrCA's job is to find that
source, so most tests assert on recovering it rather than on particular numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.corrca import (
    anchor_component_signs,
    component_isc,
    condition_isc,
    corrca_covariances,
    decompose_channel_corrca,
    fit_corrca,
    fit_corrca_from_covariances,
    forward_patterns,
    isc_from_covariances,
    leave_one_out_isc,
    paired_condition_test,
    projected_isc,
)

N_SUBJECTS, N_CHANNELS, N_SAMPLES = 8, 12, 4000


@pytest.fixture
def shared_source():
    return np.random.default_rng(0).standard_normal(N_SAMPLES)


@pytest.fixture
def topography():
    return np.random.default_rng(1).standard_normal(N_CHANNELS)


@pytest.fixture
def data(shared_source, topography):
    """``(S, C, N)``: the shared source on one topography plus correlated noise."""
    rng = np.random.default_rng(2)
    mixing = rng.standard_normal((N_SUBJECTS, N_CHANNELS, N_CHANNELS))
    noise = np.einsum(
        "scd,sdn->scn",
        mixing,
        rng.standard_normal((N_SUBJECTS, N_CHANNELS, N_SAMPLES)),
    )
    return 5.0 * topography[None, :, None] * shared_source[None, None, :] + noise


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_recovers_shared_source(data, shared_source):
    result = fit_corrca(data)
    projected = result.transform(data)  # (S, K, N)
    for s in range(N_SUBJECTS):
        assert abs(_corr(projected[s, 0], shared_source)) > 0.9


def test_recovers_topography(data, topography):
    result = fit_corrca(data)
    assert abs(_corr(result.patterns[0], topography)) > 0.95


def test_isc_sorted_and_separates_signal(data):
    result = fit_corrca(data)
    assert np.all(np.diff(result.isc) <= 1e-12)
    assert result.isc[0] > 0.5
    assert result.isc[1] < 0.1


def test_noise_only_has_low_isc():
    noise = np.random.default_rng(3).standard_normal(
        (N_SUBJECTS, N_CHANNELS, N_SAMPLES)
    )
    assert fit_corrca(noise).isc[0] < 0.05


def test_filters_whiten_within_covariance(data):
    cov = corrca_covariances(data)
    result = fit_corrca_from_covariances(cov)
    gram = result.filters.T @ cov.within @ result.filters
    np.testing.assert_allclose(gram, np.eye(result.n_components), atol=1e-8)


def test_isc_from_covariances_reproduces_fit(data):
    cov = corrca_covariances(data)
    result = fit_corrca_from_covariances(cov)
    np.testing.assert_allclose(
        isc_from_covariances(cov, result.filters), result.isc, atol=1e-10
    )


def test_isc_is_mean_pairwise_correlation_for_equal_variance(data):
    """With per-subject unit variance the ISC equals the mean pairwise Pearson r."""
    result = fit_corrca(data)
    y = result.transform(data)[:, 0]
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    r = np.corrcoef(y)
    mean_r = r[~np.eye(N_SUBJECTS, dtype=bool)].mean()
    np.testing.assert_allclose(component_isc(y[:, None, :], np.ones((1, 1))), mean_r)


def test_sign_convention(data):
    result = fit_corrca(data)
    peaks = result.patterns[
        np.arange(result.n_components), np.abs(result.patterns).argmax(axis=1)
    ]
    assert np.all(peaks > 0)


def test_sign_shared_across_subjects(data, shared_source):
    """One filter for all subjects: every subject's projection has the same sign."""
    projected = fit_corrca(data).transform(data)[:, 0]
    signs = np.sign([_corr(p, shared_source) for p in projected])
    assert len(set(signs)) == 1


def test_rank_deficient_input_keeps_numerical_rank(data):
    referenced = data - data.mean(axis=1, keepdims=True)  # average reference
    result = fit_corrca(referenced)
    assert result.n_retained == N_CHANNELS - 1
    assert np.all(np.isfinite(result.isc))


def test_shrinkage_makes_full_rank(data):
    referenced = data - data.mean(axis=1, keepdims=True)
    assert fit_corrca(referenced, gamma=0.1).n_retained == N_CHANNELS


def test_n_pca_sets_component_count(data):
    assert fit_corrca(data, n_pca=4).n_components == 4


def test_held_out_isc_lower_than_training(data):
    train, test = data[:5], data[5:]
    result = fit_corrca(train)
    held_out = component_isc(test, result.filters)
    assert held_out[0] > 0.5
    assert held_out[0] <= result.isc[0] + 0.05


def test_forward_patterns_shape(data):
    result = fit_corrca(data, n_pca=3)
    cov = corrca_covariances(data)
    assert forward_patterns(cov.subject[0], result.filters).shape == (3, N_CHANNELS)


def test_input_not_mutated(data):
    before = data.copy()
    fit_corrca(data)
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize(
    "kwargs, message",
    [({"gamma": 1.5}, "gamma"), ({"n_pca": 0}, "n_pca"), ({"n_pca": 99}, "n_pca")],
)
def test_invalid_parameters(data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_corrca(data, **kwargs)


def test_requires_3d_and_two_subjects(data):
    with pytest.raises(ValueError, match="3-D"):
        corrca_covariances(data[0])
    with pytest.raises(ValueError, match="two subjects"):
        corrca_covariances(data[:1])


def test_decompose_channel_corrca_shapes(shared_source, topography):
    rng = np.random.default_rng(4)
    n_freqs, n_times = 5, 400
    envelope = np.abs(shared_source[: n_freqs * n_times]).reshape(n_freqs, n_times)
    data_4d = topography[None, :, None, None] * envelope[
        None, None
    ] + rng.standard_normal((N_SUBJECTS, N_CHANNELS, n_freqs, n_times))
    out = decompose_channel_corrca(data_4d, label="test", n_pca=4)
    assert out.sources.shape == (N_SUBJECTS, 4, n_freqs, n_times)
    assert out.patterns.shape == (N_SUBJECTS, 4, N_CHANNELS)
    for s in range(N_SUBJECTS):
        assert abs(_corr(out.patterns[s, 0], topography)) > 0.8


# ---------------------------------------------------------------------------
# Projected-component ISC and the condition read-out
# ---------------------------------------------------------------------------


def test_projected_isc_matches_component_isc(data):
    result = fit_corrca(data, n_pca=4)
    projected = result.transform(data)
    np.testing.assert_allclose(
        projected_isc(projected), component_isc(data, result.filters), atol=1e-10
    )


def test_projected_isc_flattens_trailing_axes(data):
    projected = fit_corrca(data, n_pca=2).transform(data)
    as_tf = projected.reshape(N_SUBJECTS, 2, 40, N_SAMPLES // 40)
    np.testing.assert_allclose(projected_isc(as_tf), projected_isc(projected))


def test_leave_one_out_isc_matches_hand_computed(data):
    y = fit_corrca(data, n_pca=2).transform(data)
    loo = leave_one_out_isc(y)
    for s in range(N_SUBJECTS):
        others = np.delete(y, s, axis=0).mean(axis=0)
        for k in range(2):
            np.testing.assert_allclose(
                loo[s, k], np.corrcoef(y[s, k], others[k])[0, 1], atol=1e-10
            )


def test_projected_isc_needs_two_subjects(data):
    with pytest.raises(ValueError, match="two subjects"):
        projected_isc(data[:1])


@pytest.fixture
def two_conditions(shared_source):
    """Sources of one component in two conditions: synchronised, then much less."""
    rng = np.random.default_rng(5)
    participants = [f"PSI{p:03d}" for p in range(10)]
    strong = shared_source[None, None] + 0.5 * rng.standard_normal((10, 1, N_SAMPLES))
    weak = shared_source[None, None] + 3.0 * rng.standard_normal((10, 1, N_SAMPLES))
    return {"Placebo": strong, "Psilocybin": weak}, participants


def test_condition_isc_frames(two_conditions):
    sources, participants = two_conditions
    group, per_participant = condition_isc(sources, {c: participants for c in sources})
    assert list(group.columns) == ["component", "condition", "isc", "n_subjects"]
    assert set(per_participant.columns) == {
        "component",
        "condition",
        "participant",
        "loo_isc",
    }
    assert len(per_participant) == 2 * len(participants)
    isc = group.set_index("condition")["isc"]
    assert isc["Placebo"] > isc["Psilocybin"]


def test_condition_isc_rejects_mismatched_labels(two_conditions):
    sources, participants = two_conditions
    with pytest.raises(ValueError, match="participant label"):
        condition_isc(sources, {c: participants[:3] for c in sources})


def test_paired_test_detects_a_synchrony_drop(two_conditions):
    sources, participants = two_conditions
    _group, per_participant = condition_isc(sources, {c: participants for c in sources})
    tests = paired_condition_test(per_participant, ["Placebo", "Psilocybin"])
    row = tests.iloc[0]
    assert row["n_pairs"] == len(participants)
    assert row["median_difference"] < 0
    assert row["p_value"] < 0.01


def test_paired_test_pairs_by_label_not_position():
    """Rows in a different order per condition must still pair correctly."""
    per_participant = pd.DataFrame(
        {
            "component": [1] * 6,
            "condition": ["A"] * 3 + ["B"] * 3,
            "participant": ["p1", "p2", "p3", "p3", "p1", "p2"],
            "loo_isc": [0.1, 0.2, 0.3, 0.6, 0.4, 0.5],
        }
    )
    row = paired_condition_test(per_participant, ["A", "B"]).iloc[0]
    assert row["mean_difference"] == pytest.approx(0.3)


def test_paired_test_needs_two_conditions():
    with pytest.raises(ValueError, match="two conditions"):
        paired_condition_test(pd.DataFrame(), ["A"])


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


@pytest.fixture
def decomposition(shared_source, topography):
    rng = np.random.default_rng(6)
    n_freqs, n_times = 4, 500
    envelope = np.abs(shared_source[: n_freqs * n_times]).reshape(n_freqs, n_times)
    data_4d = 5.0 * topography[None, :, None, None] * envelope[
        None, None
    ] + rng.standard_normal((N_SUBJECTS, N_CHANNELS, n_freqs, n_times))
    return decompose_channel_corrca(data_4d, label="test", n_pca=3)


def test_oriented_flips_every_quantity_of_a_component(decomposition):
    signs = np.array([-1.0, 1.0, -1.0])
    flipped = decomposition.oriented(signs)
    np.testing.assert_allclose(
        flipped.model.filters, decomposition.model.filters * signs
    )
    np.testing.assert_allclose(
        flipped.model.patterns, decomposition.model.patterns * signs[:, None]
    )
    np.testing.assert_allclose(
        flipped.sources, decomposition.sources * signs[None, :, None, None]
    )
    np.testing.assert_allclose(
        flipped.patterns, decomposition.patterns * signs[None, :, None]
    )
    np.testing.assert_allclose(flipped.model.isc, decomposition.model.isc)


def test_oriented_rejects_non_unit_signs(decomposition):
    with pytest.raises(ValueError, match="signs"):
        decomposition.oriented(np.array([1.0, 0.5, 1.0]))


def test_anchor_makes_the_shared_pattern_positive_over_the_mask(decomposition):
    mask = np.zeros(N_CHANNELS, dtype=bool)
    # Anchor on the channels where the first component's pattern is most negative,
    # so the anchor has to flip it.
    mask[np.argsort(decomposition.model.patterns[0])[:3]] = True
    oriented, flip, strength = anchor_component_signs(decomposition, mask)
    assert flip.shape == strength.shape == (3,)
    assert flip[0] == -1.0
    inside = oriented.model.patterns[:, mask].mean(axis=1)
    outside = oriented.model.patterns[:, ~mask].mean(axis=1)
    assert np.all(inside >= outside)
