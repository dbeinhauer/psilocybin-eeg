"""
Tests for scripts/run_corrca_snr_grid.py — the stage-06b SNR grid.

The trial cutting, normalisation and tests are stage 06's (covered there). What is
CorrCA's own is where the spatial operators come from, so that is what is pinned: the
filter is the stored shared one, and the "(PCA)" reference reads through the subspace
those filters span.
"""

import numpy as np
import pytest

from scripts.run_corrca_snr_grid import _build_arg_parser, corrca_operators

N_CHANNELS, N_COMPONENTS = 12, 4


@pytest.fixture
def shared_filters():
    return np.random.default_rng(0).standard_normal((N_CHANNELS, N_COMPONENTS))


@pytest.fixture
def mask():
    mask = np.zeros(N_CHANNELS, dtype=bool)
    mask[:4] = True
    return mask


@pytest.fixture
def ops(shared_filters, mask):
    binary = mask / mask.sum()
    return corrca_operators(shared_filters, binary, mask)


def test_filters_are_the_stored_ones_transposed(ops, shared_filters):
    np.testing.assert_array_equal(ops["filters"], shared_filters.T)


def test_projector_is_orthogonal_onto_the_filter_span(ops, shared_filters):
    p = ops["projector"]
    np.testing.assert_allclose(p @ p, p, atol=1e-10)
    np.testing.assert_allclose(p, p.T, atol=1e-10)
    np.testing.assert_allclose(p @ shared_filters, shared_filters, atol=1e-10)
    assert np.linalg.matrix_rank(p) == N_COMPONENTS


def test_pca_row_is_the_mask_through_that_subspace(ops, mask):
    np.testing.assert_allclose(ops["pca_row"], ops["projector"] @ (mask / mask.sum()))


def test_masked_filters_read_only_the_anchor(ops, mask):
    assert np.all(ops["masked"][:, ~mask] == 0)
    np.testing.assert_array_equal(ops["masked"][:, mask], ops["filters"][:, mask])
    assert np.all((ops["masked_share"] > 0) & (ops["masked_share"] < 1))


def test_cli_defaults_and_variants():
    args = _build_arg_parser().parse_args([])
    assert args.variant == "corrca_channel_joined_separate"
    assert args.signal_variants == [
        "zscored",
        "prestim",
        "stored_prestim",
        "masked_prestim",
    ]
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--variant", "iva_channel_joined_separate"])
