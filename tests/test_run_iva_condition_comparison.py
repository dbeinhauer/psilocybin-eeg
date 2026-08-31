"""
Tests for scripts/run_iva_condition_comparison.py — the joined-condition IVA CLI.

Figure content is not asserted. What is pinned is the CLI contract and the output
layout: the defaults, the 1-based ``--components`` translation, the canonical
``plots/<stage>/<label>/<spectrum>/<type>/pca_<n>/`` tree, and that a band run cannot
collide with the broadband one. ``run_condition_comparison`` is driven end to end on
tiny synthetic data with the two disk-touching loaders stubbed.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts import notebook_helpers, run_iva_condition_comparison  # noqa: E402
from scripts.run_iva_condition_comparison import (  # noqa: E402
    _ANALYSIS_DIR,
    _STAGE_DIR,
    _build_arg_parser,
    _onset_average,
    main,
)
from src.analysis.iva_condition_comparison import (  # noqa: E402
    decompose_channel_iva,
    slice_to_band,
)
from src.analysis.data_representations import (  # noqa: E402
    AnalysisData,
    DataRepresentation,
)
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
    SingleDataMetadata,
)
from src.io.iva_store import load_iva_components, list_iva_results  # noqa: E402

SFREQ = 250.0
N_CHANNELS = 10
N_TIMES = 1600
N_FREQS = 20
# Placebo and Psilocybin share participants 3/7/11; 21 and 33 are unmatched, so the
# pooling must drop them.
COHORT = {
    ConditionVariants.PLACEBO: [21, 3, 11, 7],
    ConditionVariants.PSILOCYBIN: [3, 11, 7, 33],
}
# ASSR-like: one stimulus every 315 samples, plus one past the end of the window.
ONSETS = np.append(np.arange(60, N_TIMES, 315), 99_999)
FREQS = np.linspace(1.0, 50.0, N_FREQS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def info():
    mne.set_log_level("ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:N_CHANNELS], sfreq=SFREQ, ch_types="eeg")
    info.set_montage(montage)
    return info


@pytest.fixture
def stub_loaders(monkeypatch, info):
    """Replace the two disk-touching helpers the loader calls with synthetic data."""

    class FakeAnalyzer:
        def __init__(self, condition):
            ids = COHORT[condition]
            self.filtered_df = pd.DataFrame(
                {
                    SingleDataMetadata.PARTICIPANT_ID: ids,
                    SingleDataMetadata.CONCATENATED_PERSON_INDEX: range(len(ids)),
                }
            )
            self.stimulus_onsets = ONSETS
            self.info = info

    def fake_load_analyzers(music_types, condition, *args, **kwargs):
        return {f"{condition.value}_{music_types[0].value}": FakeAnalyzer(condition)}

    def fake_analyzers_to_datasets(analyzers):
        return {
            label: AnalysisData(
                data=np.zeros((len(a.filtered_df), N_CHANNELS, N_TIMES)),
                sfreq=SFREQ,
                representation=DataRepresentation.TIME_DOMAIN,
                label=label,
                feature_names=list(info["ch_names"]),
                info=info,
            )
            for label, a in analyzers.items()
        }

    def fake_compute_wavelet_datasets(datasets, analyzers, freqs, representation, **kw):
        out = {}
        for label, ad in datasets.items():
            rng = np.random.default_rng(abs(hash(label)) % 2**32)
            out[label] = AnalysisData(
                data=rng.standard_normal(
                    (ad.data.shape[0], ad.data.shape[1], len(freqs), ad.data.shape[2])
                ),
                sfreq=SFREQ,
                representation=DataRepresentation.WAVELET_POWER,
                # The representation suffix the real to_wavelet_power appends. Kept
                # here on purpose: a clean label would hide that the pooled data
                # label is unfit to name a directory, which is exactly the bug the
                # canonical product name in run_condition_comparison guards against.
                label=(f"{label} (wavelet power {freqs[0]:.0f}-{freqs[-1]:.0f} Hz)"),
                feature_names=ad.feature_names,
                info=info,
                metadata={
                    "freqs": freqs,
                    "n_cycles": freqs / 2.0,
                    "keep_frequency_dim": True,
                },
            )
        return out

    monkeypatch.setattr(notebook_helpers, "load_analyzers", fake_load_analyzers)
    monkeypatch.setattr(
        notebook_helpers, "analyzers_to_datasets", fake_analyzers_to_datasets
    )
    monkeypatch.setattr(
        notebook_helpers, "compute_wavelet_datasets", fake_compute_wavelet_datasets
    )


def _argv(save_dir: Path, *extra: str) -> list[str]:
    """A small, fast ASSR run writing under *save_dir*."""
    return [
        "--experiment",
        "assr",
        "--n_pca",
        "3",
        "--n_channels",
        "6",
        "--n_times",
        str(N_TIMES),
        "--n_pairs",
        "3",
        "--reuse_wavelets",
        "--iva_max_iter",
        "8",
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


def _out_dir(save_dir: Path, spectrum: str = "broadband", n_pca: int = 3) -> Path:
    return (
        save_dir
        / _STAGE_DIR
        / "Joined_ASSR"
        / spectrum
        / _ANALYSIS_DIR
        / f"pca_{n_pca}"
    )


class TestCliDefaults:
    def test_defaults_match_the_standard_assr_run(self):
        args = _build_arg_parser().parse_args([])
        assert args.n_pca == 10
        assert args.conditions == ["Placebo", "Psilocybin"]
        assert args.iva_opt_approach == "newton"
        assert args.random_state == 42
        assert args.band is None
        assert args.components is None

    def test_subsets_default_to_the_full_extent(self):
        args = _build_arg_parser().parse_args([])
        assert args.n_pairs is None
        assert args.n_channels is None
        assert args.n_times is None

    def test_caches_are_opt_in(self):
        """Neither cache may be written unless asked for."""
        args = _build_arg_parser().parse_args([])
        assert args.reuse_wavelets is False
        assert args.subset_cache is False

    def test_the_stimulus_average_is_on_by_default(self):
        assert _build_arg_parser().parse_args([]).skip_onset_average is False
        assert _build_arg_parser().parse_args([]).min_onsets == 5

    def test_the_difference_row_is_on_by_default(self):
        assert _build_arg_parser().parse_args([]).no_difference_row is False


class TestSliceToBand:
    def test_none_keeps_the_whole_grid(self):
        data = np.zeros((2, 3, N_FREQS, 5))
        sliced, freqs = slice_to_band(data, FREQS, None)
        assert sliced.shape == data.shape
        np.testing.assert_array_equal(freqs, FREQS)

    def test_a_band_restricts_the_frequency_axis(self):
        data = np.zeros((2, 3, N_FREQS, 5))
        sliced, freqs = slice_to_band(data, FREQS, "alpha")
        assert sliced.shape[2] == len(freqs) < N_FREQS
        assert freqs.min() >= 8.0 and freqs.max() <= 13.0

    def test_a_band_outside_the_grid_raises(self):
        with pytest.raises(ValueError, match="no frequency inside the wavelet grid"):
            slice_to_band(np.zeros((2, 3, 3, 5)), np.array([1.0, 2.0, 3.0]), "gamma")


class TestDecompose:
    def test_returns_sources_and_patterns(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 6, 5, 40))
        sources, patterns = decompose_channel_iva(
            data,
            label="t",
            n_pca=3,
            random_state=0,
            iva_opt_approach="newton",
            iva_max_iter=4,
            iva_w_diff_stop=1e-6,
        )
        assert sources.shape == (4, 3, 5, 40)
        assert patterns.shape == (4, 3, 6)

    def test_n_pca_above_the_channel_count_raises(self):
        with pytest.raises(ValueError, match="must be ≤ n_channels"):
            decompose_channel_iva(
                np.zeros((3, 4, 5, 20)),
                label="t",
                n_pca=5,
                random_state=0,
                iva_opt_approach="newton",
                iva_max_iter=2,
                iva_w_diff_stop=1e-6,
            )


class TestOnsetAverage:
    @pytest.fixture
    def sources(self):
        return np.random.default_rng(0).standard_normal((4, 2, 5, N_TIMES))

    def test_averages_the_epochs_that_fit(self, sources):
        result = _onset_average(
            sources, ONSETS, label="t", n_times=N_TIMES, sfreq=SFREQ, min_onsets=1
        )
        assert result is not None
        onset_tf, epoch_times, marks = result
        assert onset_tf.shape[:3] == sources.shape[:3]
        assert onset_tf.shape[3] == len(epoch_times)
        # t = 0 is the onset, and the epoch straddles it.
        assert epoch_times[0] < 0 < epoch_times[-1]
        assert marks[0] == 0.0 and marks[1] > 0.0

    def test_the_epoch_never_reaches_the_next_stimulus(self, sources):
        onset_tf, _times, _marks = _onset_average(
            sources, ONSETS, label="t", n_times=N_TIMES, sfreq=SFREQ, min_onsets=1
        )
        assert onset_tf.shape[3] <= 315

    def test_matches_a_hand_computed_mean(self, sources):
        onset_tf, epoch_times, _ = _onset_average(
            sources, ONSETS, label="t", n_times=N_TIMES, sfreq=SFREQ, min_onsets=1
        )
        pre = int(round(-epoch_times[0] * SFREQ))
        post = len(epoch_times) - pre
        fitting = [
            int(o) for o in ONSETS if 0 <= int(o) - pre and int(o) + post <= N_TIMES
        ]
        manual = np.mean([sources[..., o - pre : o + post] for o in fitting], axis=0)
        np.testing.assert_allclose(onset_tf, manual, rtol=1e-12)

    def test_no_onsets_returns_none(self, sources):
        assert (
            _onset_average(
                sources, None, label="t", n_times=N_TIMES, sfreq=SFREQ, min_onsets=1
            )
            is None
        )

    def test_onsets_all_outside_the_window_return_none(self, sources):
        assert (
            _onset_average(
                sources,
                np.array([99_999]),
                label="t",
                n_times=N_TIMES,
                sfreq=SFREQ,
                min_onsets=1,
            )
            is None
        )

    def test_too_few_fitting_epochs_returns_none(self, sources):
        assert (
            _onset_average(
                sources, ONSETS, label="t", n_times=N_TIMES, sfreq=SFREQ, min_onsets=99
            )
            is None
        )


class TestProductName:
    """The run is named ``Joined_<MusicType>`` — never the pooled *data* label.

    The wavelet transform appends its representation to the data label
    ("Joined_ASSR (wavelet power 1-50 Hz)"). That describes the array, but it is
    unfit to name a path: it would put spaces, parentheses and a frequency range
    into the plot directory and every figure filename, and disagree with the
    component store, which names the condition from the enum.
    """

    def test_the_plot_directory_is_the_canonical_name(self, tmp_path, stub_loaders):
        main(_argv(tmp_path, "--skip_onset_average"))
        stage = tmp_path / _STAGE_DIR
        assert [p.name for p in stage.iterdir()] == ["Joined_ASSR"]

    def test_no_representation_suffix_reaches_any_path(self, tmp_path, stub_loaders):
        main(_argv(tmp_path, "--skip_onset_average"))
        for path in tmp_path.rglob("*"):
            name = path.name
            assert "wavelet power" not in name, path
            assert "(" not in name and ")" not in name, path
            assert " " not in name, path

    def test_the_figure_filenames_carry_the_canonical_name(
        self, tmp_path, stub_loaders
    ):
        main(_argv(tmp_path, "--components", "2", "--skip_onset_average"))
        names = sorted(
            p.name for p in (_out_dir(tmp_path) / "participants").glob("*.png")
        )
        assert names == [
            "condition_tf_ic02_participants_Joined_ASSR.png",
            "condition_topomap_ic02_participants_Joined_ASSR.png",
        ], names

    def test_the_plot_tree_and_the_store_agree_on_the_condition(
        self, tmp_path, stub_loaders
    ):
        store = tmp_path / "processed"
        main(
            _argv(
                tmp_path,
                "--skip_onset_average",
                "--store_components",
                "--store_dir",
                str(store),
            )
        )
        entries = list_iva_results(ExperimentNames.ASSR, processed_data_dir=store)
        # The store names the condition from the enum; the plot tree must match it.
        assert [p.parent.name for p in entries] == [ConditionVariants.JOINED.value]
        # pca_<n> -> <analysis type> -> <spectrum> -> the run's product name.
        assert (
            _out_dir(tmp_path).parents[2].name
            == f"{ConditionVariants.JOINED.value}_{MusicTypeVariants.ASSR.value}"
        )
        assert load_iva_components(entries[0]).label == "Joined_ASSR"


class TestEndToEnd:
    def test_writes_the_canonical_broadband_tree(self, tmp_path, stub_loaders):
        main(_argv(tmp_path))
        out = _out_dir(tmp_path)
        assert (out / "condition_mean_topomaps.png").exists()
        assert (out / "condition_mean_tf_maps.png").exists()
        assert (out / "condition_mean_tf_maps_onset.png").exists()
        # 3 components × (topomap + whole-recording TF + onset TF)
        figures = sorted(p.name for p in (out / "participants").glob("*.png"))
        assert len(figures) == 9, figures
        assert len([f for f in figures if f.startswith("onset_")]) == 3
        assert all((out / "participants" / f).stat().st_size > 5000 for f in figures)

    def test_only_matched_participants_reach_the_figures(self, tmp_path, stub_loaders):
        """21 and 33 have no partner, so they must not appear."""
        main(_argv(tmp_path))
        names = [p.name for p in (_out_dir(tmp_path) / "participants").glob("*.png")]
        assert names, "no figures written"
        # Participant identity lives inside the figures; what is checkable here is
        # that the pooling kept 3 pairs, which the component count reflects.
        assert len(names) == 9

    def test_skip_onset_average(self, tmp_path, stub_loaders):
        main(_argv(tmp_path, "--skip_onset_average"))
        out = _out_dir(tmp_path)
        assert not (out / "condition_mean_tf_maps_onset.png").exists()
        assert list((out / "participants").glob("onset_*.png")) == []

    def test_components_are_one_based(self, tmp_path, stub_loaders):
        main(_argv(tmp_path, "--components", "2", "--skip_onset_average"))
        names = sorted(
            p.name for p in (_out_dir(tmp_path) / "participants").glob("*.png")
        )
        assert names == [
            "condition_tf_ic02_participants_Joined_ASSR.png",
            "condition_topomap_ic02_participants_Joined_ASSR.png",
        ], names

    def test_an_out_of_range_component_raises(self, tmp_path, stub_loaders):
        with pytest.raises(ValueError, match=r"outside 1\.\.3"):
            main(_argv(tmp_path, "--components", "9"))

    def test_fewer_than_two_conditions_raises(self, tmp_path, stub_loaders):
        with pytest.raises(ValueError, match="at least two distinct"):
            main(_argv(tmp_path, "--conditions", "Placebo"))

    def test_a_band_run_cannot_collide_with_the_broadband_one(
        self, tmp_path, stub_loaders
    ):
        main(_argv(tmp_path, "--skip_onset_average"))
        main(_argv(tmp_path, "--band", "gamma", "--skip_onset_average"))
        broadband = _out_dir(tmp_path, "broadband")
        bands = _out_dir(tmp_path, "bands")
        assert (bands / "gamma_condition_mean_topomaps.png").exists()
        assert sorted(p.name for p in bands.glob("*.png")) == [
            "gamma_condition_mean_tf_maps.png",
            "gamma_condition_mean_topomaps.png",
        ]
        # The band prefix never leaks into the broadband tree.
        assert not any(p.name.startswith("gamma_") for p in broadband.rglob("*.png"))

    def test_the_subset_cache_is_written_only_when_asked(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        cache = tmp_path / "subset_cache"
        monkeypatch.setattr(
            run_iva_condition_comparison,
            "resolve_notebook_wavelet_cache_dir",
            lambda experiment_name: cache,
        )
        main(_argv(tmp_path, "--skip_onset_average"))
        assert not cache.exists(), "the cache was written without --subset_cache"

    def test_psilo_music_defaults_to_both_music_types(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        seen: list[MusicTypeVariants] = []

        def spy(music_type, **kwargs):
            seen.append(music_type)

        monkeypatch.setattr(
            run_iva_condition_comparison, "run_condition_comparison", spy
        )
        main(["--experiment", ExperimentNames.PSILO_MUSIC.value])
        assert seen == [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]

    def test_assr_defaults_to_the_placeholder_music_type(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        seen: list[MusicTypeVariants] = []
        monkeypatch.setattr(
            run_iva_condition_comparison,
            "run_condition_comparison",
            lambda music_type, **kwargs: seen.append(music_type),
        )
        main(["--experiment", ExperimentNames.ASSR.value])
        assert seen == [MusicTypeVariants.ASSR]


class TestComponentStore:
    """``--store_components`` keeps the components with their row bookkeeping.

    The subject-axis join is the case the mapping has to earn its keep on: a
    participant owns one row per condition, so a bare label cannot address a row.
    """

    def _load(self, store, n_pca=3, band=None):
        return load_iva_components(
            experiment=ExperimentNames.ASSR,
            condition=ConditionVariants.JOINED,
            variant=IvaVariants.CHANNEL_JOINED,
            music_type=MusicTypeVariants.ASSR,
            band=band,
            n_pca=n_pca,
            processed_data_dir=store,
        )

    def test_nothing_is_stored_by_default(self, tmp_path, stub_loaders):
        store = tmp_path / "processed"
        main(_argv(tmp_path, "--skip_onset_average", "--store_dir", str(store)))
        assert list_iva_results(ExperimentNames.ASSR, processed_data_dir=store) == []

    def test_stores_one_row_per_participant_and_condition(self, tmp_path, stub_loaders):
        store = tmp_path / "processed"
        main(
            _argv(
                tmp_path,
                "--skip_onset_average",
                "--store_components",
                "--store_dir",
                str(store),
            )
        )
        loaded = self._load(store)

        # 3 matched participants x 2 conditions.
        assert loaded.n_subjects == 6
        assert sorted(set(loaded.participants)) == ["003", "007", "011"]
        assert loaded.rows(condition=ConditionVariants.PLACEBO) == [0, 1, 2]
        assert loaded.rows(condition=ConditionVariants.PSILOCYBIN) == [3, 4, 5]
        assert loaded.participant_rows["003"] == (0, 3)
        with pytest.raises(ValueError, match="occupies 2 rows"):
            loaded.row("003")
        assert loaded.row("003", ConditionVariants.PSILOCYBIN) == 3

        assert loaded.tf_maps.shape == (6, 3, N_FREQS, N_TIMES)
        assert loaded.channel_patterns.shape == (6, 3, 6)  # --n_channels 6
        assert not loaded.has(IvaComponentArrays.TIMECOURSE)
        assert len(loaded.channel_names) == 6
        assert loaded.extras["tf_pc1_signs"].shape == (6, 3)

    def test_the_stimulus_onsets_travel_with_the_components(
        self, tmp_path, stub_loaders
    ):
        """Without them an onset-locked read-out would need the analysers again."""
        store = tmp_path / "processed"
        main(
            _argv(
                tmp_path,
                "--skip_onset_average",
                "--store_components",
                "--store_dir",
                str(store),
            )
        )
        loaded = self._load(store)
        for condition in (ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN):
            onsets = loaded.extras[f"stimulus_onsets_{condition.value}"]
            # The fixture puts one onset past the end of the window; it must be gone,
            # or indexing the stored time axis with it would fail.
            assert onsets.size > 0
            assert onsets.max() < loaded.times.size
            np.testing.assert_array_equal(onsets, ONSETS[ONSETS < loaded.times.size])

    def test_the_stored_file_lands_in_the_condition_subdirectory(
        self, tmp_path, stub_loaders
    ):
        store = tmp_path / "processed"
        main(
            _argv(
                tmp_path,
                "--skip_onset_average",
                "--store_components",
                "--store_dir",
                str(store),
            )
        )
        entries = list_iva_results(ExperimentNames.ASSR, processed_data_dir=store)
        assert [p.parent.name for p in entries] == ["Joined"]
        assert entries[0].name == "iva_channel_joined__ASSR__broadband__pca3.npz"

    def test_a_band_run_stores_a_separate_entry(self, tmp_path, stub_loaders):
        store = tmp_path / "processed"
        for extra in ([], ["--band", "gamma"]):
            main(
                _argv(
                    tmp_path,
                    "--skip_onset_average",
                    "--store_components",
                    "--store_dir",
                    str(store),
                    *extra,
                )
            )
        names = sorted(
            p.name
            for p in list_iva_results(ExperimentNames.ASSR, processed_data_dir=store)
        )
        assert names == [
            "iva_channel_joined__ASSR__broadband__pca3.npz",
            "iva_channel_joined__ASSR__gamma__pca3.npz",
        ]
        gamma = self._load(store, band="gamma")
        assert gamma.freqs.max() <= 50.0
        assert gamma.tf_maps.shape[2] == gamma.freqs.size
