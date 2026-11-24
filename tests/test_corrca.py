"""
Tests for src/analysis/corrca.py — Correlated Component Analysis across subjects.

The data are synthetic: one source shared by every subject, projected through a
common topography, buried in subject-specific noise. CorrCA's job is to find that
source, so most tests assert on recovering it rather than on particular numbers.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.analysis.corrca as corrca_module
from src.analysis.corrca import (
    INFLUENCE_Z_THRESHOLD,
    CorrcaDecomposition,
    anchor_component_signs,
    annotate_influence,
    channel_corrca_influence,
    component_isc,
    condition_isc,
    corrca_covariances,
    decompose_channel_corrca,
    fit_corrca,
    fit_corrca_from_covariances,
    forward_patterns,
    isc_from_covariances,
    jackknife_statistics,
    leave_group_out_influence,
    leave_subsets_out,
    low_topography_flags,
    map_influence,
    leave_one_out_isc,
    matched_component_correlation,
    paired_condition_test,
    participant_groups,
    projected_isc,
    robust_influence_z,
    subset_refit_filters,
)
from src.analysis.wavelet_ica import zscore_by_time

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


def test_decompose_fit_rows_fits_on_them_and_reads_every_subject(
    shared_source, topography
):
    rng = np.random.default_rng(5)
    n_freqs, n_times = 4, 300
    envelope = np.abs(shared_source[: n_freqs * n_times]).reshape(n_freqs, n_times)
    data_4d = topography[None, :, None, None] * envelope[
        None, None
    ] + rng.standard_normal((N_SUBJECTS, N_CHANNELS, n_freqs, n_times))
    fit_rows = [0, 1, 2, 3]
    out = decompose_channel_corrca(data_4d, label="test", n_pca=3, fit_rows=fit_rows)
    alone = decompose_channel_corrca(data_4d[fit_rows], label="test", n_pca=3)
    np.testing.assert_allclose(out.model.filters, alone.model.filters)
    np.testing.assert_allclose(out.model.isc, alone.model.isc)
    np.testing.assert_allclose(out.sources[fit_rows], alone.sources)
    np.testing.assert_allclose(out.patterns[fit_rows], alone.patterns)
    # A subject outside the fit is read through the same filters, its pattern
    # through its own covariance.
    x = zscore_by_time(data_4d).reshape(N_SUBJECTS, N_CHANNELS, -1)
    held_out = out.model.transform(x[4:])
    np.testing.assert_allclose(out.sources[4:].reshape(held_out.shape), held_out)
    for s in range(4, N_SUBJECTS):
        xs = x[s] - x[s].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(
            out.patterns[s], forward_patterns(xs @ xs.T, out.model.filters)
        )


def test_covariance_rows_match_the_subset_and_are_validated(data):
    sub = corrca_covariances(data, rows=[5, 1, 2])
    direct = corrca_covariances(data[[5, 1, 2]])
    np.testing.assert_allclose(sub.subject, direct.subject)
    np.testing.assert_allclose(sub.total, direct.total)
    with pytest.raises(ValueError, match="distinct"):
        corrca_covariances(data, rows=[1, 1])
    with pytest.raises(ValueError, match="distinct"):
        corrca_covariances(data, rows=[0, N_SUBJECTS])
    with pytest.raises(ValueError, match="two subjects"):
        corrca_covariances(data, rows=[0])


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
    assert np.all(oriented.model.patterns[:, mask].mean(axis=1) >= 0.0)


def test_anchor_reads_the_mask_mean_not_the_contrast(decomposition):
    # A pattern negative everywhere but less so over the mask: the correlation anchor
    # would keep it, the mask-mean anchor must flip it.
    patterns = np.full_like(decomposition.model.patterns, -2.0)
    mask = np.zeros(N_CHANNELS, dtype=bool)
    mask[:3] = True
    patterns[:, mask] = -1.0
    shifted = CorrcaDecomposition(
        model=replace(decomposition.model, patterns=patterns),
        sources=decomposition.sources,
        patterns=decomposition.patterns,
    )
    _oriented, flip, strength = anchor_component_signs(shifted, mask)
    assert np.all(flip == -1.0)
    assert np.all(strength > 0.0)


def test_matched_component_correlation_pairs_by_index_and_keeps_sign():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 20))
    b = np.stack([2.0 * a[0] + 5.0, -a[1], a[0]])
    r = matched_component_correlation(a, b)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-1.0)
    # Component 3 is matched to component 3, not to the copy of component 1.
    assert abs(r[2]) < 1.0
    with pytest.raises(ValueError):
        matched_component_correlation(a, b[:, :10])


# ---------------------------------------------------------------------------
# Leave-one-participant-out influence
# ---------------------------------------------------------------------------

PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def test_participant_groups_keeps_first_appearance_order():
    assert participant_groups(["b", "a", "b", "c", "a"]) == ((0, 2), (1, 4), (3,))


def test_jackknife_drop_matches_covariances_of_the_reduced_data(data):
    # Shuffled, uneven chunks: the statistics are sums over samples.
    edges = [0, 700, 2500, N_SAMPLES]
    chunks = [data[..., a:b] for a, b in zip(edges[:-1], edges[1:])][::-1]
    stats = jackknife_statistics(chunks, PAIRS)
    full = corrca_covariances(data)
    np.testing.assert_allclose(stats.covariances.subject, full.subject, atol=1e-6)
    np.testing.assert_allclose(stats.covariances.total, full.total, atol=1e-6)
    for g, rows in enumerate(PAIRS):
        expected = corrca_covariances(np.delete(data, rows, axis=0))
        got = stats.without(g)
        np.testing.assert_allclose(got.subject, expected.subject, atol=1e-6)
        np.testing.assert_allclose(got.total, expected.total, rtol=1e-9, atol=1e-6)


def test_insample_isc_is_leave_one_out_isc_for_single_rows(data):
    model = fit_corrca(data, n_pca=4)
    stats = jackknife_statistics([data], [(s,) for s in range(N_SUBJECTS)])
    influence = leave_group_out_influence(stats, model, n_pca=4, gamma=0.0)
    expected = leave_one_out_isc(model.transform(data))
    got = influence.pivot(index="row", columns="component", values="insample_isc")
    np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-8)


def test_heldout_isc_uses_the_refit_filter(data):
    model = fit_corrca(data, n_pca=4)
    stats = jackknife_statistics([data], PAIRS)
    influence = leave_group_out_influence(stats, model, n_pca=4, gamma=0.0)
    rows = PAIRS[0]
    refit = fit_corrca(np.delete(data, rows, axis=0), n_pca=4)
    y = np.einsum("d,sdn->sn", refit.filters[:, 0], data)
    y = y - y.mean(axis=1, keepdims=True)
    reference = np.delete(y, rows, axis=0).sum(axis=0)
    first = influence[(influence["row"] == 0) & (influence["component"] == 1)]
    assert first["heldout_isc"].iloc[0] == pytest.approx(_corr(y[0], reference))
    full = model.transform(data)[:, 0]
    own = influence[(influence["row"] == 1) & (influence["component"] == 1)]
    assert own["within_participant_isc"].iloc[0] == pytest.approx(
        _corr(full[1], full[0])
    )
    # The shared source survives dropping any pair: same topography, same ISC.
    top = influence[influence["component"] == 1]
    assert (top["pattern_r"] > 0.95).all()
    assert top["influence"].abs().max() < 0.05


def test_influence_flags_a_participant_driving_a_component(data, topography):
    """Participant 0's two recordings share a strong private source: a component
    only that participant carries, which the pooled between-recording term rewards."""
    rng = np.random.default_rng(7)
    private_topo = rng.standard_normal(N_CHANNELS)
    private = 6.0 * private_topo[:, None] * rng.standard_normal(N_SAMPLES)[None]
    driven = data.copy()
    driven[list(PAIRS[0])] += private
    model = fit_corrca(driven, n_pca=6)
    k = int(np.argmax([abs(_corr(p, private_topo)) for p in model.patterns]))

    stats = jackknife_statistics([driven], PAIRS)
    influence = leave_group_out_influence(stats, model, n_pca=6, gamma=0.0)
    influence["influence_z"] = robust_influence_z(influence)
    comp = influence[influence["component"] == k + 1]
    driver = comp[comp["group"] == 0]
    others = comp[comp["group"] != 0]
    assert driver["influence"].iloc[0] > others["influence"].max() + 0.1
    assert driver["influence_z"].iloc[0] > INFLUENCE_Z_THRESHOLD
    assert driver["pattern_r"].iloc[0] < others["pattern_r"].min()
    # Synchronised with its own other recording, not with anybody else.
    assert (driver["within_participant_isc"] > 0.8).all()
    assert (driver[["insample_isc", "heldout_isc"]].abs() < 0.2).all().all()


def test_channel_influence_chunking_and_labels(monkeypatch, shared_source, topography):
    rng = np.random.default_rng(4)
    n_freqs, n_times = 4, 300
    envelope = np.abs(shared_source[: n_freqs * n_times]).reshape(n_freqs, n_times)
    data_4d = topography[None, :, None, None] * envelope[
        None, None
    ] + rng.standard_normal((N_SUBJECTS, N_CHANNELS, n_freqs, n_times))
    model = decompose_channel_corrca(data_4d, label="test", n_pca=3).model
    participants = [f"P{s // 2}" for s in range(N_SUBJECTS)]
    conditions = ["Placebo", "Psilocybin"] * (N_SUBJECTS // 2)
    whole = channel_corrca_influence(
        data_4d, model, participants, conditions, n_pca=3, gamma=0.0
    )
    # Force one frequency row and a few dozen samples per chunk.
    monkeypatch.setattr(corrca_module, "_JACKKNIFE_CHUNK_VALUES", 50 * N_SUBJECTS)
    chunked = channel_corrca_influence(
        data_4d, model, participants, conditions, n_pca=3, gamma=0.0
    )
    numeric = ["isc_full", "isc_without", "pattern_r", "insample_isc", "heldout_isc"]
    np.testing.assert_allclose(chunked[numeric], whole[numeric], atol=1e-8)
    np.testing.assert_allclose(
        whole.drop_duplicates("component")["isc_full"], model.isc, atol=1e-8
    )
    assert list(whole["participant"].unique()) == ["P0", "P1", "P2", "P3"]
    assert set(whole["condition"]) == {"Placebo", "Psilocybin"}
    assert whole["outlier"].dtype == bool
    # The shared component: losses near zero, so no z, however large, flags anyone.
    assert not whole.loc[whole["component"] == 1, "outlier"].any()


def test_jackknife_needs_a_partition_into_three_groups(data):
    with pytest.raises(ValueError, match="partition"):
        jackknife_statistics([data], [(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="at least two groups"):
        jackknife_statistics([data], [(0, 1, 2, 3), (4, 5, 6, 7)])


# ---------------------------------------------------------------------------
# Leave-subset-out refits
# ---------------------------------------------------------------------------


def test_dropping_several_groups_matches_the_reduced_data(data):
    # An offset per row and channel, so the centring of the cross terms is exercised.
    shifted = data + np.random.default_rng(5).standard_normal(data.shape[:2])[..., None]
    edges = [0, 900, 3100, N_SAMPLES]
    chunks = [shifted[..., a:b] for a, b in zip(edges[:-1], edges[1:])][::-1]
    stats = jackknife_statistics(chunks, PAIRS, group_cross=True)
    for dropped in [(0, 2), (1, 3), (3,)]:
        rows = [r for g in dropped for r in PAIRS[g]]
        expected = corrca_covariances(np.delete(shifted, rows, axis=0))
        got = stats.without(dropped)
        np.testing.assert_allclose(got.subject, expected.subject, atol=1e-6)
        np.testing.assert_allclose(got.total, expected.total, rtol=1e-9, atol=1e-5)


def test_dropping_several_groups_needs_the_cross_terms(data):
    stats = jackknife_statistics([data], PAIRS)
    with pytest.raises(ValueError, match="group_cross"):
        stats.without((0, 1))
    with pytest.raises(ValueError, match="fewer than two"):
        jackknife_statistics([data], PAIRS, group_cross=True).without((0, 1, 2))


def test_leave_subsets_out_enumerates_each_size():
    subsets = leave_subsets_out(6, [3, 1])
    assert len(subsets) == 6 + 20
    assert subsets[0] == (0,) and subsets[6] == (0, 1, 2)
    with pytest.raises(ValueError, match="at least two"):
        leave_subsets_out(4, [3])


def test_subset_refits_are_the_direct_refits_signed_to_the_full_fit(data):
    model = fit_corrca(data, n_pca=4)
    model = model.oriented(-np.ones(model.n_components))  # any anchored sign
    stats = jackknife_statistics([data], PAIRS, group_cross=True)
    subsets = leave_subsets_out(len(PAIRS), [1, 2])
    refits = subset_refit_filters(stats, model, subsets, n_pca=4, gamma=0.0)
    assert refits.filters.shape == (len(subsets), N_CHANNELS, model.n_components)
    for i, subset in enumerate(subsets):
        rows = [r for g in subset for r in PAIRS[g]]
        direct = fit_corrca(np.delete(data, rows, axis=0), n_pca=4)
        # The shared component: the refit's leading filter, up to its sign.
        got = refits.filters[i][:, 0]
        r = _corr(got, direct.filters[:, 0])
        assert abs(r) > 0.999
        # ... signed so its pattern agrees with the full fit's.
        pattern = forward_patterns(corrca_covariances(data).within, got[:, None])[0]
        assert _corr(pattern, model.patterns[0]) > 0.9
    assert np.nanmin(refits.pattern_r[:, 0]) > 0.9
    assert np.nanmin(refits.filter_r[:, 0]) > 0.5


# ---------------------------------------------------------------------------
# Influence on maps (the onset-locked reading)
# ---------------------------------------------------------------------------


def _recording_maps(seed: int, n_participants: int = 5):
    """``(2P, K=2, 6, 30)`` maps: component 1 shared by everyone, 2 pure noise."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal((6, 30))
    maps = rng.standard_normal((2 * n_participants, 2, 6, 30))
    maps[:, 0] += 3.0 * shared
    participants = [f"P{i}" for i in range(n_participants) for _ in range(2)]
    conditions = ["Placebo", "Psilocybin"] * n_participants
    return maps, participants, conditions


def test_map_influence_with_unchanged_refits_reads_in_sample_everywhere():
    maps, participants, conditions = _recording_maps(0)
    table = map_influence(maps, dict.fromkeys(set(participants), maps), participants)
    np.testing.assert_allclose(table["insample_isc"], table["heldout_isc"])
    # The group ISC without a participant is the remaining maps' own.
    without = table[(table["group"] == 0) & (table["component"] == 1)]
    assert without["isc_without"].iloc[0] == pytest.approx(projected_isc(maps[2:])[0])
    # In-sample: the recording against the sum of every OTHER participant's maps.
    row, k = 0, 0
    others = maps[2:, k].reshape(8, -1)
    expected = _corr(maps[row, k].ravel(), others.sum(axis=0))
    got = table[(table["row"] == row) & (table["component"] == k + 1)]
    assert got["insample_isc"].iloc[0] == pytest.approx(expected)
    # Own pair: against the same participant's other recording.
    assert got["within_participant_isc"].iloc[0] == pytest.approx(
        _corr(maps[0, 0].ravel(), maps[1, 0].ravel())
    )
    assert table.loc[table["component"] == 1, "isc_full"].iloc[0] > 0.5


def test_map_influence_flags_a_participant_the_component_needs():
    maps, participants, conditions = _recording_maps(1)
    refits = {p: maps for p in set(participants)}
    # Without P2, the refit no longer finds the shared component: its maps are noise.
    lost = maps.copy()
    lost[:, 0] = np.random.default_rng(9).standard_normal(lost[:, 0].shape)
    refits["P2"] = lost
    table = annotate_influence(
        map_influence(maps, refits, participants), participants, conditions
    )
    first = table[table["component"] == 1].drop_duplicates("participant")
    top = first.loc[first["influence"].idxmax()]
    assert top.participant == "P2" and top.outlier
    assert not first.loc[first["participant"] != "P2", "outlier"].any()


def test_map_influence_needs_every_participant_s_refit():
    maps, participants, _ = _recording_maps(2)
    with pytest.raises(ValueError, match="No refit maps"):
        map_influence(maps, {"P0": maps}, participants)


def test_low_topography_flags_only_a_real_drop():
    rng = np.random.default_rng(3)
    r = 0.99 + 0.002 * rng.standard_normal(30)
    r[4] = 0.70  # a refit whose topography moved
    r[9] = 0.975  # a robust outlier, but a negligible drop
    flags = low_topography_flags(r)
    assert flags[4] and not flags[9]
    assert flags.sum() == 1
    assert not low_topography_flags(np.array([np.nan, 0.5, 0.9])).any()
