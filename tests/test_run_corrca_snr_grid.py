"""
Tests for scripts/run_corrca_snr_grid.py — the stage-06b SNR grid.

The trial cutting, normalisation and tests are stage 06's (covered there). What is
CorrCA's own is where the spatial operators come from, so that is what is pinned: the
filter is the stored shared one, and the "(PCA)" reference reads through the subspace
those filters span.
"""

import numpy as np
import pytest

from scripts.run_assr_snr_grid import _accumulate_tests
from scripts.run_corrca_snr_grid import (
    _build_arg_parser,
    baseline_label,
    baseline_window,
    corrca_operators,
    run,
)
from src.analysis import assr_trials as at

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
    assert args.variant == "corrca_channel_joined_tracks"
    # Only the stored sources of the full store, unless asked for more.
    assert args.signal_variants == ["full"]
    assert args.masked_tf_maps is False
    assert (args.log_power, args.roi_fit, args.non_roi_fit) == (False, False, False)
    # The ROI store is read after its join's full one, not named on its own.
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
            ["--variant", "corrca_channel_joined_separate_roi"]
        )
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--signal_variants", "stored_prestim"])
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--variant", "iva_channel_joined_separate"])


def test_the_complement_reads_everything_but_the_anchor(ops, mask):
    assert np.all(ops["outside"][:, mask] == 0)
    np.testing.assert_allclose(ops["masked"] + ops["outside"], ops["filters"])
    # Disjoint supports, so the two L2 shares are the legs of one right triangle.
    np.testing.assert_allclose(ops["masked_share"] ** 2 + ops["outside_share"] ** 2, 1)


def test_an_roi_store_has_no_complement(shared_filters):
    everything = np.ones(N_CHANNELS, dtype=bool)
    ops = corrca_operators(shared_filters, everything / N_CHANNELS, everything)
    assert not ops["outside"].any()
    assert not ops["outside_share"].any()


def test_cli_reads_the_log_power_stores_with_their_run():
    # The log stores are read with the raw run, after the full store; not on their own.
    assert _build_arg_parser().parse_args([]).log_power is False
    assert _build_arg_parser().parse_args(["--log_power"]).log_power
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
            ["--variant", "corrca_channel_joined_tracks_log"]
        )


def test_cli_reads_the_individual_condition_stores():
    args = _build_arg_parser().parse_args(
        ["--variant", "corrca_channel_individual_conditions"]
    )
    assert args.variant == "corrca_channel_individual_conditions"


def test_the_band_swap_reads_only_a_broadband_store():
    # Its filters come from the band twin of a broadband store; a band store has none.
    args = _build_arg_parser().parse_args(
        ["--swap_band", "1-30hz", "--band", "30-50hz"]
    )
    with pytest.raises(ValueError, match="--band unset"):
        run(args)


def test_another_baseline_is_parsed_and_named_apart():
    parser = _build_arg_parser()
    assert parser.parse_args([]).baseline is None
    args = parser.parse_args(["--baseline=-0.3:-0.05"])
    assert args.baseline == (-0.3, -0.05)
    # The default writes the usual names; another baseline gets its own token.
    assert baseline_label(None) == ""
    assert baseline_label(args.baseline) == "baseline_-300_-50ms"
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline=-0.1:0.2"])  # runs past the onset


def test_the_baseline_window_selects_only_its_samples():
    times = np.arange(-75, 225) / 250.0  # -0.3 s .. 0.896 s at 250 Hz
    np.testing.assert_array_equal(baseline_window(times, None), times < 0)
    mask = baseline_window(times, (-0.3, -0.05))
    assert times[mask][0] == pytest.approx(-0.3)
    assert times[mask][-1] == pytest.approx(-0.052)
    assert mask.sum() == 63
    # An edge on a sample keeps it; an epoch starting just after -0.3 s still fits.
    assert baseline_window(times, (-0.3, -0.048))[times == -0.048].all()
    late_start = np.arange(-38, 115) / 128.0  # first sample -0.297 s
    assert baseline_window(late_start, (-0.3, -0.05))[0]
    with pytest.raises(ValueError, match="before the epoch"):
        baseline_window(times, (-0.5, -0.05))
    with pytest.raises(ValueError, match="at least 2"):
        baseline_window(times, (-0.001, 0.0))


def test_a_single_condition_is_tested_only_for_snr():
    n_participants, n_components = 6, 2
    comp_labels = [f"CC {k + 1}" for k in range(n_components)]
    labels = comp_labels + list(at.MASK_LABELS)
    value = {
        "Placebo": np.random.default_rng(1).standard_normal(
            (n_participants, len(labels))
        )
    }
    rows = []
    _accumulate_tests(
        rows,
        {"variant": "full"},
        value,
        ["Placebo"],
        labels,
        comp_labels,
        "greater",
        "less",
    )
    assert {r["family"] for r in rows} == {"snr"}
    assert len(rows) == len(at.MASK_LABELS) * n_components
