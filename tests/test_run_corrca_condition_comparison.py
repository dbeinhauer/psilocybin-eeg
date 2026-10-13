"""
Tests for scripts/run_corrca_condition_comparison.py — the joined-condition CorrCA CLI.

Figure content is not asserted. What is pinned is the output layout under its own
stage, the per-condition synchrony tables, and the stored bookkeeping that differs
from IVA: one sign per component. The loader stubs are the stage-06 IVA tests' own.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.run_corrca_condition_comparison import (  # noqa: E402
    _ANALYSIS_DIR,
    _STAGE_DIR,
    main,
)
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import load_iva_components  # noqa: E402
import test_run_iva_condition_comparison as iva_tests  # noqa: E402

# The stage-06 loader stubs, re-exposed as fixtures of this module.
N_FREQS, N_TIMES = iva_tests.N_FREQS, iva_tests.N_TIMES
info = iva_tests.info
stub_loaders = iva_tests.stub_loaders

N_PCA = 3


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _argv(save_dir: Path, *extra: str) -> list[str]:
    return [
        "--experiment",
        "assr",
        "--n_pca",
        str(N_PCA),
        "--n_channels",
        "6",
        "--n_times",
        str(N_TIMES),
        "--n_pairs",
        "3",
        "--reuse_wavelets",
        "--wavelet_freq_min",
        "1",
        "--wavelet_freq_max",
        "50",
        "--wavelet_n_freqs",
        str(N_FREQS),
        "--save_dir",
        str(save_dir),
        *extra,
    ]


def _out_dir(save_dir: Path) -> Path:
    return (
        save_dir
        / _STAGE_DIR
        / "JoinedSeparate_ASSR"
        / "broadband"
        / _ANALYSIS_DIR
        / f"pca_{N_PCA}"
    )


def test_lives_in_its_own_stage():
    assert _STAGE_DIR == "06b-corrca-condition-comparison"
    assert _ANALYSIS_DIR == IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE.value


def test_writes_the_condition_grids_and_the_synchrony_report(tmp_path, stub_loaders):
    main(_argv(tmp_path))
    out = _out_dir(tmp_path)
    for name in (
        "condition_mean_topomaps.png",
        "condition_mean_tf_maps.png",
        "condition_mean_tf_maps_onset.png",
        "corrca_shared_topomaps.png",
        "corrca_condition_isc.png",
    ):
        assert (out / name).exists(), name
    figures = sorted(p.name for p in (out / "participants").glob("*.png"))
    assert len(figures) == 3 * N_PCA
    assert all("_cc0" in name for name in figures)


def test_condition_isc_tables(tmp_path, stub_loaders):
    main(_argv(tmp_path, "--skip_onset_average"))
    out = _out_dir(tmp_path)
    group = pd.read_csv(out / "corrca_condition_isc.csv")
    per_participant = pd.read_csv(out / "corrca_participant_isc.csv")
    tests = pd.read_csv(out / "corrca_condition_isc_test.csv")
    assert set(group["condition"]) == {"Placebo", "Psilocybin"}
    assert sorted(group["component"].unique()) == list(range(1, N_PCA + 1))
    # 3 matched participants × 2 conditions × N_PCA components.
    assert len(per_participant) == 3 * 2 * N_PCA
    assert list(tests["n_pairs"]) == [3] * N_PCA


def test_components_are_one_based(tmp_path, stub_loaders):
    main(_argv(tmp_path, "--components", "2", "--skip_onset_average"))
    names = sorted(p.name for p in (_out_dir(tmp_path) / "participants").glob("*.png"))
    assert names == [
        "condition_tf_cc02_participants_JoinedSeparate_ASSR.png",
        "condition_topomap_cc02_participants_JoinedSeparate_ASSR.png",
    ]


def test_fewer_than_two_conditions_raises(tmp_path, stub_loaders):
    with pytest.raises(ValueError, match="at least two distinct"):
        main(_argv(tmp_path, "--conditions", "Placebo"))


def test_store_keeps_rows_and_corrca_extras(tmp_path, stub_loaders):
    store = tmp_path / "store"
    main(
        _argv(
            tmp_path / "plots",
            "--skip_onset_average",
            "--store_components",
            "--store_dir",
            str(store),
        )
    )
    loaded = load_iva_components(
        experiment=ExperimentNames.ASSR,
        condition=ConditionVariants.JOINED_SEPARATE,
        variant=IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE,
        music_type=MusicTypeVariants.ASSR,
        n_pca=N_PCA,
        processed_data_dir=store,
    )
    assert loaded.n_subjects == 6  # 3 participants × 2 conditions
    assert loaded.extras["corrca_filters"].shape == (6, N_PCA)
    assert loaded.extras["corrca_shared_patterns"].shape == (N_PCA, 6)
    assert loaded.extras["corrca_condition_isc"].shape == (2, N_PCA)
