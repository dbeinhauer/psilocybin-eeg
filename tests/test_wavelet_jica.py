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

import dataclasses
import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.linalg
from sklearn.decomposition import FastICA

from src.analysis import assr_trials as at
from src.analysis.wavelet_jica import (
    DEFAULT_SIGNAL_VARIANTS,
    IDENTITY_TOLERANCE,
    SIGNAL_VARIANTS,
    VARIANT_RECIPE,
    VARIANT_REFERENCE_FROM,
    VARIANT_UNITS,
    assemble_join,
    bootstrap_median_ci,
    cohort_mean_filter,
    cohort_mean_pattern,
    component_polarity,
    fit_joint_ica,
    orient_components,
    participant_spread,
    subspace_mask_rows,
    subspace_reference_rows,
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


class TestSubspaceReference:
    """The ``ASSR-mask (PCA)`` rows: the electrode mask read through the reduction.

    The point of the row is that a component is judged against a reference living in
    the same subspace it does, so what has to hold is that the operator really is the
    reduction's projector — not merely something mask-shaped.
    """

    MASK = np.array([True, True, False, False, True, False])  # 3 of C = 6 channels

    def _fitted(self, join=ConditionVariants.JOINED, n_ica=3):
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        return layout, fit_joint_ica(layout, n_ica=n_ica, max_iter=200)

    def _weights(self):
        return at.binary_filter_weights(self.MASK)

    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_the_rows_are_the_mask_carried_through_the_projector(self, join):
        """``m_b P^T P``, which the factorisation computes without forming ``P^T P``."""
        layout, result = self._fitted(join)
        weights = self._weights()
        rows = subspace_mask_rows(result, weights)
        assert rows.shape == (layout.n_blocks, layout.n_blocks * layout.n_channels)

        k = result.n_components
        mixing = result.patterns.transpose(1, 0, 2).reshape(k, -1).T
        unmixing = result.filters.reshape(k, -1)
        projector = mixing @ unmixing
        embedded = np.zeros((layout.n_blocks, layout.n_blocks * layout.n_channels))
        for block in range(layout.n_blocks):
            start = block * layout.n_channels
            embedded[block, start : start + layout.n_channels] = weights
        np.testing.assert_allclose(rows, embedded @ projector, atol=1e-8)

    def test_a_projector_that_keeps_everything_returns_the_plain_mask_average(self):
        """The property that says it IS a projector, not merely a mask-shaped operator.

        With a reduction that discards nothing, ``P^T P`` is the identity, so the row
        has to collapse to the electrode average on that block's own channels — the
        binary reference itself. Anything else means the operator is not the one the
        fit applied.
        """
        raw = _raw()
        layout = assemble_join(
            raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        _layout, result = self._fitted(ConditionVariants.JOINED_TRACKS)
        n_features = layout.n_blocks * layout.n_channels
        # A full-rank orthonormal reduction, as filters/patterns rather than a refit:
        # unmixing = P, mixing = P^T, so mixing @ unmixing is exactly the identity.
        basis, _r = scipy.linalg.qr(
            np.random.default_rng(0).normal(size=(n_features, n_features))
        )
        loadings = basis.T  # orthonormal ROWS
        full_rank = dataclasses.replace(
            result,
            filters=loadings.reshape(n_features, layout.n_blocks, layout.n_channels),
            patterns=loadings.reshape(
                n_features, layout.n_blocks, layout.n_channels
            ).transpose(1, 0, 2),
            # n_components is read off the maps, so they have to grow with the rest.
            tf_maps=np.zeros((n_features, F, layout.n_times)),
        )
        weights = self._weights()
        rows = subspace_mask_rows(full_rank, weights)
        reference = subspace_reference_rows(
            raw, layout, rows, bins=[1, 2], zscore=False
        )
        for condition in CONDITIONS:
            band = raw[condition][:, :, [1, 2], :].mean(axis=2)  # (P, C, T)
            np.testing.assert_allclose(
                reference[condition],
                np.einsum("c,pct->pt", weights, band),
                atol=1e-8,
            )

    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_the_zscored_row_reads_the_signal_the_fit_was_handed(self, join):
        """Per ``(block, channel, frequency)`` over that condition's own time axis."""
        raw = _raw()
        layout = assemble_join(raw, PARTICIPANTS, CONDITIONS, join)
        result = fit_joint_ica(layout, n_ica=3, max_iter=200)
        rows = subspace_mask_rows(result, self._weights())
        bins = [1, 2]
        reference = subspace_reference_rows(raw, layout, rows, bins, zscore=True)

        for condition in CONDITIONS:
            stacked = []
            for participant, block_condition in layout.blocks:
                source = raw[condition if block_condition is None else block_condition]
                band = source[PARTICIPANTS.index(participant)][:, bins, :]
                mean = band.mean(axis=-1, keepdims=True)
                deviation = band.std(axis=-1, keepdims=True)
                stacked.append(((band - mean) / deviation).mean(axis=1))
            projected = rows @ np.concatenate(stacked, axis=0)
            expected = projected[
                [layout.block_of(p, condition) for p in layout.participants]
            ]
            np.testing.assert_allclose(reference[condition], expected, atol=1e-8)

    def test_each_join_reads_the_recordings_its_own_way(self):
        """A recording join gives each condition its own block; a time join does not."""
        raw = _raw()
        weights = self._weights()
        joined = assemble_join(raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED)
        result = fit_joint_ica(joined, n_ica=3, max_iter=200)
        rows = subspace_mask_rows(result, weights)
        reference = subspace_reference_rows(raw, joined, rows, [1], zscore=False)
        # Different blocks read the same stacked samples, so the two conditions differ
        # by the OPERATOR as well as by the data.
        assert not np.allclose(reference[CONDITIONS[0]], reference[CONDITIONS[1]])

        tracks = assemble_join(
            raw, PARTICIPANTS, CONDITIONS, ConditionVariants.JOINED_TRACKS
        )
        track_result = fit_joint_ica(tracks, n_ica=3, max_iter=200)
        track_rows = subspace_mask_rows(track_result, weights)
        track_reference = subspace_reference_rows(
            raw, tracks, track_rows, [1], zscore=False
        )
        # One block per participant: the same operator, applied to each track.
        assert track_reference[CONDITIONS[0]].shape == (P, T)
        assert not np.allclose(
            track_reference[CONDITIONS[0]], track_reference[CONDITIONS[1]]
        )

    def test_a_row_that_does_not_match_the_layout_is_refused(self):
        layout, result = self._fitted()
        rows = subspace_mask_rows(result, self._weights())
        with pytest.raises(ValueError, match="to match the layout"):
            subspace_reference_rows(_raw(), layout, rows[:-1], [1], zscore=False)
        with pytest.raises(ValueError, match="to match the channel axis"):
            subspace_mask_rows(result, np.ones(C + 1))


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
        assert set(DEFAULT_SIGNAL_VARIANTS) <= set(SIGNAL_VARIANTS)

    def test_the_default_grid_covers_every_filter_by_signal_cell_that_exists(self):
        """Three spatial filters x (raw | z-scored), minus the two cells that do not.

        A restricted filter's weights presume the scaling they were fitted on, so the
        masked pair reads the z-scored signal only — applying them to raw power would
        weight each electrode by its own power level on top, which is the artefact the
        equal-weight reference row exists to remove.
        """
        cells = {
            (VARIANT_RECIPE[v]["filter"], VARIANT_RECIPE[v]["zscore"])
            for v in DEFAULT_SIGNAL_VARIANTS
        }
        assert cells == {
            ("own", False),
            ("own", True),
            ("mean", False),
            ("mean", True),
            ("masked", True),
            ("mean_masked", True),
        }

    def test_every_default_cell_is_per_trial_baselined(self):
        """That is what puts all four in the reference's units, so they compare."""
        assert all(VARIANT_RECIPE[v]["baseline"] for v in DEFAULT_SIGNAL_VARIANTS)
        # And the one variant that is not baselined is the one kept out of the default.
        assert not VARIANT_RECIPE["zscored"]["baseline"]
        assert "zscored" not in DEFAULT_SIGNAL_VARIANTS

    def test_zscored_is_the_exact_component_and_prestim_is_not(self):
        """The z-scored route applies the filter to the signal the fit actually saw."""
        assert VARIANT_RECIPE["zscored"]["zscore"] is True
        assert VARIANT_RECIPE["zscored_prestim"]["zscore"] is True
        assert VARIANT_RECIPE["prestim"]["zscore"] is False

    def test_every_baselined_variant_borrows_prestim_s_reference(self):
        """The invariant the grid rests on: only the component moves between cells.

        For a raw-signal variant the borrow is a no-op — the fixed rows do not depend
        on the learned filter, so ``mean_prestim`` would recompute the identical
        numbers — and it is declared anyway so the rule is stated once instead of
        inferred per variant. ``zscored`` is excluded because it has no per-trial
        baseline to put the borrowed rows in.
        """
        assert VARIANT_REFERENCE_FROM == {
            "zscored_prestim": "prestim",
            "mean_prestim": "prestim",
            "mean_zscored_prestim": "prestim",
            "masked_prestim": "prestim",
            "mean_masked_prestim": "prestim",
        }
        assert set(VARIANT_REFERENCE_FROM) == {
            v
            for v in SIGNAL_VARIANTS
            if VARIANT_RECIPE[v]["baseline"] and v != "prestim"
        }
        for variant, source in VARIANT_REFERENCE_FROM.items():
            assert variant in SIGNAL_VARIANTS
            assert source in SIGNAL_VARIANTS
            assert VARIANT_RECIPE[variant]["baseline"]
            assert not VARIANT_RECIPE[source]["zscore"]


class TestCohortFilter:
    """The operator the ``mean_*`` variants use, and the sign it anchors."""

    @staticmethod
    def _patterns(seed: int = 0, n_blocks: int = 5, n_components: int = 3):
        return np.random.default_rng(seed).normal(size=(n_blocks, n_components, C))

    def test_it_is_the_unit_normalised_block_mean(self):
        patterns = self._patterns()
        scaled = patterns / np.linalg.norm(patterns, axis=2, keepdims=True)
        cohort = cohort_mean_pattern(patterns)
        np.testing.assert_allclose(cohort.pattern, scaled.mean(axis=0))
        assert cohort.normalized and not cohort.aligned

    def test_normalising_stops_one_loud_block_from_setting_the_topography(self):
        """jICA's loadings are unconstrained, which is why this is not optional."""
        patterns = self._patterns()
        shouting = patterns.copy()
        shouting[0] *= 500.0
        np.testing.assert_allclose(
            cohort_mean_pattern(shouting).pattern,
            cohort_mean_pattern(patterns).pattern,
        )
        # Without it, that one block dominates.
        assert not np.allclose(
            cohort_mean_pattern(shouting, normalize=False).pattern,
            cohort_mean_pattern(patterns, normalize=False).pattern,
        )

    def test_an_all_zero_block_contributes_nothing_rather_than_dividing_by_zero(self):
        patterns = self._patterns()
        patterns[2] = 0.0
        cohort = cohort_mean_pattern(patterns)
        assert np.isfinite(cohort.pattern).all()
        np.testing.assert_allclose(cohort.cosine[2], 0.0)

    def test_the_cosine_reports_a_cancelling_mean(self):
        """Half the blocks inverted is a cohort with no consensus, and it must show."""
        patterns = self._patterns()
        patterns[len(patterns) // 2 :] = -patterns[: len(patterns) - len(patterns) // 2]
        cohort = cohort_mean_pattern(patterns)
        assert (cohort.disagreeing_blocks() > 0).any()

    def test_signs_are_left_alone_unless_an_anchor_is_given(self):
        """A jICA block's sign is a result, so aligning first would invent a consensus."""
        patterns = self._patterns()
        mask = np.zeros(C, dtype=bool)
        mask[: C // 2] = True
        plain = cohort_mean_pattern(patterns)
        aligned = cohort_mean_pattern(patterns, align_to=mask)
        assert not plain.aligned and aligned.aligned
        assert not np.allclose(plain.pattern, aligned.pattern)

    def test_the_filter_inverts_the_pattern_it_came_from(self):
        cohort = cohort_mean_pattern(self._patterns())
        backward = cohort_mean_filter(cohort.pattern)
        assert backward.shape == cohort.pattern.shape
        np.testing.assert_allclose(
            backward @ cohort.pattern.T, np.eye(cohort.n_components), atol=1e-9
        )

    def test_a_rank_deficient_cohort_model_is_refused(self):
        """Two components whose cohort topographies collapse cannot be separated."""
        cohort = cohort_mean_pattern(self._patterns())
        collapsed = cohort.pattern.copy()
        collapsed[1] = collapsed[0]
        with pytest.raises(ValueError, match="rank-deficient"):
            cohort_mean_filter(collapsed)

    def test_the_sign_is_one_per_component_and_matches_polarity_flip(self):
        """Same criterion and floor as stage 06, applied at jICA's granularity."""
        cohort = cohort_mean_pattern(self._patterns())
        mask = np.zeros(C, dtype=bool)
        mask[: C // 2] = True
        flip, strength = component_polarity(cohort.pattern, mask)
        assert flip.shape == (cohort.n_components,)
        assert set(np.unique(flip)) <= {-1.0, 1.0}
        reference_flip, reference_strength = at.polarity_flip(
            cohort.pattern[np.newaxis], mask
        )
        np.testing.assert_allclose(flip, reference_flip[0])
        np.testing.assert_allclose(strength, reference_strength[0])

    @pytest.mark.parametrize("join", BOTH_JOINS)
    def test_the_cohort_operator_is_the_same_for_both_conditions(self, join):
        """Pooling the blocks is what keeps it symmetric in the conditions.

        A condition-specific filter would break the exchangeability the paired contrast
        rests on — a difference could then come from the operator having changed.
        """
        layout = assemble_join(_raw(), PARTICIPANTS, CONDITIONS, join)
        result = fit_joint_ica(layout, n_ica=2, max_iter=200)
        cohort = cohort_mean_pattern(result.patterns)
        # Built from every block at once, so there is nothing per condition to differ.
        assert cohort.cosine.shape == (layout.n_blocks, result.n_components)
        mask = np.zeros(layout.n_channels, dtype=bool)
        mask[: layout.n_channels // 2] = True
        flip, _strength = component_polarity(cohort.pattern, mask)
        assert flip.shape == (result.n_components,)
