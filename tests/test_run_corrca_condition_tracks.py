"""
Tests for scripts/run_corrca_condition_tracks.py — the time-concatenated CorrCA CLI.

Figure content is not asserted. What is pinned is the output layout under its own
stage, the per-condition synchrony tables over the matched participants, and the
unsplit store. The loader stubs (real ASSR cohort sizes) are the stage-06 IVA tests'.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.run_corrca_condition_tracks import (  # noqa: E402
    _ANALYSIS_DIR,
    _STAGE_DIR,
    _build_arg_parser,
    main,
)
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import load_iva_components  # noqa: E402
import test_run_iva_condition_tracks as iva_tests  # noqa: E402

# The stage-06 loader stubs, re-exposed as fixtures of this module.
N_FREQS, N_TIMES = iva_tests.N_FREQS, iva_tests.N_TIMES
info = iva_tests.info
stub_loaders = iva_tests.stub_loaders

N_PCA = 3
N_PAIRS = 4


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
        str(N_PAIRS),
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
        / "JoinedTracks_ASSR"
        / "broadband"
        / _ANALYSIS_DIR
        / f"pca_{N_PCA}"
    )


def test_lives_in_its_own_stage():
    assert _STAGE_DIR == "06b-corrca-condition-comparison"
    assert _ANALYSIS_DIR == IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS.value


def test_zscore_defaults_to_per_condition():
    assert _build_arg_parser().parse_args([]).zscore_mode == "per_condition"


def test_writes_the_grids_and_the_synchrony_report(tmp_path, stub_loaders):
    main(_argv(tmp_path))
    out = _out_dir(tmp_path)
    for name in (
        "shared_mean_topomaps.png",
        "condition_mean_tf_maps.png",
        "condition_mean_tf_maps_onset.png",
        "corrca_shared_topomaps.png",
        "corrca_condition_isc.png",
    ):
        assert (out / name).exists(), name


def test_condition_isc_pairs_every_matched_participant(tmp_path, stub_loaders):
    main(_argv(tmp_path, "--skip_onset_average"))
    out = _out_dir(tmp_path)
    per_participant = pd.read_csv(out / "corrca_participant_isc.csv")
    tests = pd.read_csv(out / "corrca_condition_isc_test.csv")
    assert set(per_participant["condition"]) == {"Placebo", "Psilocybin"}
    assert per_participant["participant"].nunique() == N_PAIRS
    assert list(tests["n_pairs"]) == [N_PAIRS] * N_PCA


def test_store_is_unsplit_with_segments(tmp_path, stub_loaders):
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
        condition=ConditionVariants.JOINED_TRACKS,
        variant=IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS,
        music_type=MusicTypeVariants.ASSR,
        n_pca=N_PCA,
        processed_data_dir=store,
    )
    assert loaded.n_subjects == N_PAIRS
    assert loaded.segment_conditions == ("Placebo", "Psilocybin")
    placebo = loaded.condition_track("tf_map", ConditionVariants.PLACEBO)
    assert placebo.shape[-1] == loaded.segment_lengths[0]
    assert loaded.extras["corrca_condition_isc"].shape == (2, N_PCA)
    assert loaded.extras["zscore_mode"].item() == "per_condition"
