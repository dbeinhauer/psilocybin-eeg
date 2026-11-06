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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.analysis_common import CORRCA_STAGE_DIR as _STAGE_DIR  # noqa: E402
from scripts.run_corrca_condition_tracks import (  # noqa: E402
    _ANALYSIS_DIR,
    _build_arg_parser,
    main,
)
from scripts.run_corrca_individual_conditions import (  # noqa: E402
    main as main_individual,
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


def test_skip_stored_reads_the_store_back_and_still_draws(tmp_path, stub_loaders):
    store = tmp_path / "store"
    # The stub montage has no ASSR electrode, so there is no '_roi' / '_non_roi' fit.
    stored = [
        "--skip_onset_average",
        "--store_components",
        "--store_dir",
        str(store),
    ]
    main(_argv(tmp_path / "first", *stored))
    (path,) = (store / "assr" / "iva_results" / "JoinedTracks").glob(
        "corrca_channel_joined_tracks__*.npz"
    )
    written = path.stat().st_mtime_ns
    first = pd.read_csv(_out_dir(tmp_path / "first") / "corrca_condition_isc.csv")

    main(_argv(tmp_path / "second", *stored, "--skip_stored"))
    out = _out_dir(tmp_path / "second")
    for name in (
        "shared_mean_topomaps.png",
        "condition_mean_tf_maps.png",
        "corrca_condition_isc.png",
    ):
        assert (out / name).exists(), name
    # Read back, not refit: the store is untouched and the synchrony is the fit's.
    assert path.stat().st_mtime_ns == written
    second = pd.read_csv(out / "corrca_condition_isc.csv")
    pd.testing.assert_frame_equal(first, second, rtol=1e-5)


def test_skip_stored_refuses_a_store_of_other_data(tmp_path, stub_loaders):
    stored = ["--store_components", "--store_dir", str(tmp_path / "store")]
    main(_argv(tmp_path / "first", "--skip_onset_average", *stored))
    trimmed = _argv(tmp_path / "second", "--skip_onset_average", *stored)
    trimmed[trimmed.index("--n_pairs") + 1] = str(N_PAIRS - 1)
    with pytest.raises(ValueError, match="not fitted on the loaded data"):
        main([*trimmed, "--skip_stored"])


def test_individual_conditions_fit_each_condition_alone(tmp_path, stub_loaders):
    store = tmp_path / "store"
    main_individual(
        _argv(
            tmp_path / "plots",
            "--skip_onset_average",
            "--store_components",
            "--store_dir",
            str(store),
        )
    )
    variant = IvaVariants.CORRCA_CHANNEL_INDIVIDUAL_CONDITIONS
    for condition in (ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN):
        out = (
            tmp_path
            / "plots"
            / _STAGE_DIR
            / "IndividualConditions_ASSR"
            / "broadband"
            / variant.value
            / condition.value
            / f"pca_{N_PCA}"
        )
        assert (out / "condition_mean_topomaps.png").exists()
        # One condition: its synchrony, but no paired test.
        per_participant = pd.read_csv(out / "corrca_participant_isc.csv")
        assert set(per_participant["condition"]) == {condition.value}
        assert not (out / "corrca_condition_isc_test.csv").exists()

        loaded = load_iva_components(
            experiment=ExperimentNames.ASSR,
            condition=condition,
            variant=variant,
            music_type=MusicTypeVariants.ASSR,
            n_pca=N_PCA,
            processed_data_dir=store,
        )
        assert loaded.n_subjects == N_PAIRS
        assert loaded.segment_conditions == (condition.value,)
        assert loaded.extras["corrca_condition_isc"].shape == (1, N_PCA)


def test_every_run_also_fits_on_log_power(tmp_path, stub_loaders):
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
    loaded = {
        variant: load_iva_components(
            experiment=ExperimentNames.ASSR,
            condition=ConditionVariants.JOINED_TRACKS,
            variant=variant,
            music_type=MusicTypeVariants.ASSR,
            n_pca=N_PCA,
            processed_data_dir=store,
        )
        for variant in (
            IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS,
            IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS_LOG,
        )
    }
    raw, log = loaded.values()
    assert log.n_subjects == raw.n_subjects == N_PAIRS
    # The log changes what is fitted, so the filters differ.
    assert not np.allclose(raw.extras["corrca_filters"], log.extras["corrca_filters"])
    out = _out_dir(tmp_path / "plots").parent.parent / (
        f"{IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS_LOG.value}/pca_{N_PCA}"
    )
    assert (out / "condition_mean_tf_maps.png").exists()


def test_skip_log_power_fits_raw_power_only(tmp_path, stub_loaders):
    store = tmp_path / "store"
    main(
        _argv(
            tmp_path / "plots",
            "--skip_onset_average",
            "--skip_log_power",
            "--store_components",
            "--store_dir",
            str(store),
        )
    )
    names = [
        p.name for p in (store / "assr" / "iva_results" / "JoinedTracks").iterdir()
    ]
    assert names and not any("_log" in name for name in names)
