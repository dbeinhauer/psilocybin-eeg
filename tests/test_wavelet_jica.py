"""
Tests for src/analysis/wavelet_jica.py — the joint-ICA core.

What is pinned here is the bookkeeping and the conventions, not the numerics of
FastICA: the join's block layout, that the standardisation makes every stacked column
unit-variance (which is what makes the stacking fair), that both axes fold back
exactly after the fit, that the sign/order convention leaves the back-projection
untouched, and that the two joins differ in exactly one way — whether a participant's
two conditions share a channel block.

The module deliberately stops at the decomposition: everything downstream lives in
:mod:`src.analysis.assr_trials` and is tested there. One test here checks the bridge
between them, because a shape mismatch at that seam would be silent.
"""

import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.linalg
from sklearn.decomposition import FastICA

from src.analysis import assr_trials as at
from src.analysis.wavelet_jica import (
    IDENTITY_TOLERANCE,
    SIGNAL_VARIANTS,
    VARIANT_RECIPE,
    VARIANT_REFERENCE_FROM,
    VARIANT_UNITS,
    assemble_join,
    bootstrap_median_ci,
    fit_joint_ica,
    orient_components,
    participant_spread,
)
from src.definitions.fields import ConditionVariants

P, C, F, T = 4, 6, 5, 80
CONDITIONS = ["Placebo", "Psilocybin"]
PARTICIPANTS = [f"PSI{i + 1:03d}" for i in range(P)]
BOTH_JOINS = [ConditionVariants.JOINED, ConditionVariants.JOINED_TRACKS]


def _raw(seed: int = 0, n_times: int = T) -> dict[str, np.ndarray]:
    """Positive, wavelet-power-like tensors with a planted shared source."""
    rng = np.random.default_rng(seed)
    shared = np.sin(np.linspace(0, 40, F * n_times)).reshape(F, n_times)
    out = {}
    for gain, condition in zip((1.0, 0.5), CONDITIONS):
        block = rng.gamma(2.0, 0.5, size=(P, C, F, n_times))
        block += gain * shared[np.newaxis, np.newaxis]
        out[condition] = block
    return out


class TestAssembleJoin:
    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_block_axis_and_sample_axis_match_the_join(self, join):
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        expected_blocks = 2 * P if join is ConditionVariants.JOINED else P
        expected_times = T if join is ConditionVariants.JOINED else 2 * T
        assert layout.n_blocks == expected_blocks
        assert layout.n_times == expected_times
        assert layout.matrix.shape == (F * expected_times, expected_blocks * C)

    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_every_stacked_column_has_unit_variance(self, join):
        """What makes the stacking fair: no block can dominate the whitening by gain."""
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        assert np.allclose(layout.matrix.std(axis=0), 1.0, atol=1e-6)

    def test_joined_gives_every_recording_its_own_block(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        first = layout.block_of(PARTICIPANTS[0], CONDITIONS[0])
        second = layout.block_of(PARTICIPANTS[0], CONDITIONS[1])
        assert first != second
        # Condition-major order, matching how the standardised blocks were stacked.
        assert layout.blocks[first] == (PARTICIPANTS[0], CONDITIONS[0])
        assert layout.block_rows(CONDITIONS[0]) == list(range(P))
        assert layout.block_rows(CONDITIONS[1]) == list(range(P, 2 * P))

    def test_joined_tracks_shares_one_block_between_the_conditions(self):
        """The defining property: the same operator reads both conditions."""
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        for participant in PARTICIPANTS:
            assert layout.block_of(participant, CONDITIONS[0]) == layout.block_of(
                participant, CONDITIONS[1]
            )
        assert layout.block_rows(CONDITIONS[0]) == layout.block_rows(CONDITIONS[1])

    def test_inputs_are_not_mutated(self):
        raw = _raw()
        before = {name: array.copy() for name, array in raw.items()}
        assemble_join(raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED)
        for name, array in raw.items():
            np.testing.assert_array_equal(array, before[name])

    def test_time_join_accepts_conditions_of_different_length(self):
        """A time join never needs the two tracks to share a time base."""
        raw = _raw()
        raw[CONDITIONS[1]] = raw[CONDITIONS[1]][..., : T - 7]
        layout = assemble_join(
            raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        assert layout.n_times == T + (T - 7)

    def test_unknown_join_is_refused(self):
        with pytest.raises(ValueError, match="JOINED or JOINED_TRACKS"):
            assemble_join(_raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.PLACEBO)

    def test_missing_condition_is_refused(self):
        raw = _raw()
        del raw[CONDITIONS[1]]
        with pytest.raises(ValueError, match="No array supplied"):
            assemble_join(raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED)

    def test_row_count_must_match_the_participant_labels(self):
        with pytest.raises(ValueError, match="participant label"):
            assemble_join(
                _raw(), PARTICIPANTS[:-1], CONDITIONS, ConditionVariants.JOINED
            )

    def test_disagreeing_channel_axes_are_refused(self):
        raw = _raw()
        raw[CONDITIONS[1]] = raw[CONDITIONS[1]][:, :-1]
        with pytest.raises(ValueError, match="non-time axes"):
            assemble_join(raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED)

    def test_block_of_refuses_an_unknown_pair(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        with pytest.raises(KeyError, match="No channel block"):
            layout.block_of("PSI999", CONDITIONS[0])


class TestFitJointIca:
    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_every_axis_folds_back_to_its_own_shape(self, join):
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        assert result.tf_maps.shape == (3, F, layout.n_times)
        assert result.filters.shape == (3, layout.n_blocks, C)
        assert result.patterns.shape == (layout.n_blocks, 3, C)
        assert result.block_ic_energy.shape == (layout.n_blocks, 3)
        assert result.n_components == 3

    def test_filter_and_pattern_are_a_pseudo_inverse_pair(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        assert result.identity_error() < IDENTITY_TOLERANCE

    def test_retained_variance_is_a_fraction(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        assert 0.0 < result.retained <= 1.0

    def test_block_energy_sums_to_ic_variance(self):
        """The per-block split is exact because the feature axis partitions by block."""
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        np.testing.assert_allclose(
            result.block_ic_energy.sum(axis=0), result.ic_variance, rtol=1e-9
        )

    def test_loading_is_proportional_to_the_squared_block_norm(self):
        """The claim the loading figures rest on: energy share == ||pattern block||^2."""
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        ratio = result.block_ic_energy / (result.patterns**2).sum(axis=2)
        np.testing.assert_allclose(ratio / ratio[0:1, :], 1.0, rtol=1e-9)

    def test_the_reduction_never_hands_the_data_matrix_to_lapack(self):
        """Regression: the full-extent run died inside scipy's SVD.

        ``FastICA``'s own whitening calls ``scipy.linalg.svd`` on the transposed input,
        and LAPACK indexes with 32-bit integers, so a matrix with more than
        ``2**31 - 1`` elements is refused outright — a hard ceiling no node size lifts.
        At the full ASSR extent (4,698,000 x 2,340) the input is 5x over it. The fit
        therefore reduces through the ``(features, features)`` covariance instead, and
        must never pass anything data-sized to a LAPACK driver.

        Checked by shape rather than by running a 44 GB fit: the arrays FastICA sees
        must be bounded by the FEATURE count, not the sample count.
        """
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        seen = []
        real_svd = scipy.linalg.svd

        def spy(a, *args, **kwargs):
            seen.append(np.shape(a))
            return real_svd(a, *args, **kwargs)

        with mock.patch.object(scipy.linalg, "svd", spy):
            fit_joint_ica(layout, n_ica=3, max_iter=200)
        n_samples, n_features = layout.matrix.shape
        n_ica = 3
        assert n_samples > n_features, "the fixture must be sample-heavy to be a test"
        assert seen, "the spy saw no SVD at all, so it is not testing anything"
        # A LAPACK input may scale with the samples ONLY through the K-dimensional
        # scores (FastICA still whitens those, which is a (K, n_samples) SVD and is
        # tiny). What it must never be is the data matrix itself.
        budget = max(n_features**2, n_ica * n_samples)
        for shape in seen:
            elements = int(np.prod(shape))
            assert elements <= budget, (
                f"a LAPACK SVD saw a {shape} array ({elements:,} elements), above the "
                f"{budget:,} that the covariance and the scores account for. Anything "
                f"scaling as samples x features ({n_samples * n_features:,} here) "
                "overflows int32 at the full extent."
            )

    def test_the_retained_subspace_matches_sklearns_own_whitening(self):
        """The reduction must be the same decomposition, not merely a cheaper one.

        ``u`` and ``d`` of ``svd(X.T)`` are exactly the eigenvectors and
        root-eigenvalues of ``X.T X``, so the covariance route spans the identical
        subspace. Compared through the orthogonal projector onto the unmixing's row
        space, which is invariant to the sign and permutation freedom ICA leaves.
        """
        layout = assemble_join(
            _raw(seed=5), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        n_ica = 3
        new = fit_joint_ica(layout, n_ica=n_ica, max_iter=2000, tol=1e-9)

        reference = FastICA(
            n_components=n_ica,
            whiten="unit-variance",
            random_state=42,
            max_iter=2000,
            tol=1e-9,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference.fit(layout.matrix)

        def projector(unmixing):
            return np.linalg.pinv(unmixing) @ unmixing

        np.testing.assert_allclose(
            projector(new.filters.reshape(n_ica, -1)),
            projector(np.asarray(reference.components_, dtype=float)),
            atol=1e-10,
        )

    def test_n_ica_outside_the_feature_count_is_refused(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        with pytest.raises(ValueError, match="n_ica must be between"):
            fit_joint_ica(layout, n_ica=0)
        with pytest.raises(ValueError, match="n_ica must be between"):
            fit_joint_ica(layout, n_ica=layout.n_blocks * C + 1)

    def test_convergence_is_reported_not_hidden(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        # One iteration cannot converge, and the flag has to say so rather than the
        # warning being printed once and lost.
        result = fit_joint_ica(layout, n_ica=3, max_iter=1)
        assert result.converged is False


class TestOrientComponents:
    def _fitted(self, join=ConditionVariants.JOINED):
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        return layout, fit_joint_ica(layout, n_ica=3, max_iter=200)

    def test_largest_tf_excursion_is_positive(self):
        _layout, result = self._fitted()
        flat = result.tf_maps.reshape(result.n_components, -1)
        peak = np.take_along_axis(
            flat, np.abs(flat).argmax(axis=1)[:, np.newaxis], axis=1
        )
        assert (peak[:, 0] > 0).all()

    def test_components_are_ordered_by_size(self):
        _layout, result = self._fitted()
        assert np.all(np.diff(result.ic_variance) <= 0)

    def test_reorienting_leaves_the_back_projection_untouched(self):
        """Flipping and reordering rearrange bookkeeping, not the decomposition."""
        _layout, result = self._fitted()

        def back_projection(res):
            k = res.n_components
            return res.tf_maps.reshape(k, -1).T @ res.patterns.transpose(
                1, 0, 2
            ).reshape(k, -1)

        # Applying the convention a second time must be a no-op, and must not change
        # what the components add up to.
        again = orient_components(result)
        np.testing.assert_allclose(
            back_projection(again), back_projection(result), atol=1e-9
        )
        assert (again.signs > 0).all()


class TestRecordingOperators:
    def test_joined_tracks_hands_the_same_operator_to_both_conditions(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        np.testing.assert_array_equal(
            result.recording_filters(layout, CONDITIONS[0]),
            result.recording_filters(layout, CONDITIONS[1]),
        )
        np.testing.assert_array_equal(
            result.recording_patterns(layout, CONDITIONS[0]),
            result.recording_patterns(layout, CONDITIONS[1]),
        )

    def test_joined_gives_each_condition_its_own_operator(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        assert not np.array_equal(
            result.recording_filters(layout, CONDITIONS[0]),
            result.recording_filters(layout, CONDITIONS[1]),
        )

    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_operators_have_the_shape_assr_trials_expects(self, join):
        """The bridge to the shared trial machinery; a mismatch here would be silent."""
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        filters = result.recording_filters(layout, CONDITIONS[0])
        assert filters.shape == (P, 3, C)
        # The shape recover_spatial_filters produces for a stored IVA run, so
        # stack_filters and project_channels take it unchanged.
        mask = np.array([True] * 3 + [False] * (C - 3))
        stacked = at.stack_filters(filters, at.binary_filter_weights(mask))
        assert stacked.shape == (P, 4, C)
        assert at.source_labels(3)[-1] == at.BINARY_FILTER_LABEL

    def test_a_mismatched_layout_is_refused(self):
        layout = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED
        )
        other = assemble_join(
            _raw(), PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        with pytest.raises(ValueError, match="do not belong together"):
            result.recording_filters(other, CONDITIONS[0])


class TestSharedUtilities:
    def test_bootstrap_interval_brackets_the_median(self):
        rng = np.random.default_rng(0)
        d = rng.normal(1.0, 0.2, size=20)
        low, high = bootstrap_median_ci(d, n_bootstrap=500, seed=1)
        assert low < np.median(d) < high

    def test_bootstrap_drops_non_finite_and_degrades_gracefully(self):
        assert np.isnan(bootstrap_median_ci(np.array([np.nan, 1.0]))).all()

    def test_sem_band_is_symmetric_about_the_mean(self):
        values = np.arange(12.0).reshape(3, 4)
        centre, low, high = participant_spread(values, "sem")
        np.testing.assert_allclose(centre, values.mean(axis=0))
        np.testing.assert_allclose(high - centre, centre - low)

    def test_iqr_band_brackets_the_median(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((9, 5))
        centre, low, high = participant_spread(values, "iqr")
        assert (low <= centre).all() and (centre <= high).all()

    def test_unknown_spread_mode_is_refused(self):
        with pytest.raises(ValueError, match="'sem' or 'iqr'"):
            participant_spread(np.zeros((2, 3)), "stderr")

    def test_every_signal_variant_has_units_and_a_recipe(self):
        assert set(SIGNAL_VARIANTS) <= set(VARIANT_UNITS)
        assert set(SIGNAL_VARIANTS) == set(VARIANT_RECIPE)

    def test_the_recipes_fill_the_reachable_cells_of_the_2x2(self):
        """Raw-without-a-baseline is the unreachable cell: it has no usable units."""
        cells = {(r["zscore"], r["baseline"]) for r in VARIANT_RECIPE.values()}
        assert cells == {(False, True), (True, False), (True, True)}

    def test_zscored_is_the_exact_component_and_prestim_is_not(self):
        """The z-scored route applies the filter to the signal the fit actually saw."""
        assert VARIANT_RECIPE["zscored"]["zscore"] is True
        assert VARIANT_RECIPE["zscored_prestim"]["zscore"] is True
        assert VARIANT_RECIPE["prestim"]["zscore"] is False

    def test_only_the_borrowing_variant_takes_a_reference_from_elsewhere(self):
        assert VARIANT_REFERENCE_FROM == {"zscored_prestim": "prestim"}
        for variant, source in VARIANT_REFERENCE_FROM.items():
            assert variant in SIGNAL_VARIANTS
            assert source in SIGNAL_VARIANTS
