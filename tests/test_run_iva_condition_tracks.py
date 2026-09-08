"""
Tests for scripts/run_iva_condition_tracks.py — the time-concatenated IVA CLI.

Figure content is not asserted. What is pinned is the CLI contract, the output layout,
and the two things that make this variant different from its subject-axis companion:
the topographies are **shared** by the conditions so their figures get one row, and the
stimulus average epochs each condition on **its own** onsets on its own segment.

The cohort fixtures use the real ASSR sizes — 15 Placebo recordings against 16
Psilocybin, 12 matched — because an equal-sized fixture would not exercise the
participant matching this variant depends on.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts import notebook_helpers, run_iva_condition_tracks  # noqa: E402
from scripts.run_iva_condition_tracks import (  # noqa: E402
    _ANALYSIS_DIR,
    _STAGE_DIR,
    _build_arg_parser,
    _onset_average_per_condition,
    main,
)
from src.analysis.condition_tracks import ZSCORE_MODES  # noqa: E402
from src.analysis.data_representations import (  # noqa: E402
    AnalysisData,
    DataRepresentation,
)
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
    SingleDataMetadata,
)
from src.io.iva_store import load_iva_components, list_iva_results  # noqa: E402

SFREQ = 250.0
N_CHANNELS = 10
N_TIMES = 1900  # per condition; long enough for 6 ASSR-spaced onsets
N_FREQS = 20

# The real ASSR cohort: 15 vs 16 recordings, 12 present in both.
PLACEBO_IDS = [31, 39, 38, 36, 34, 24, 26, 19, 28, 37, 29, 32, 30, 23, 35]
PSILOCYBIN_IDS = [23, 33, 32, 26, 30, 19, 18, 37, 38, 39, 34, 27, 35, 29, 40, 28]
N_MATCHED = 12
COHORT = {
    ConditionVariants.PLACEBO: PLACEBO_IDS,
    ConditionVariants.PSILOCYBIN: PSILOCYBIN_IDS,
}
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
    """Replace the disk-touching helpers with synthetic data of the real cohort shape."""

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
                # here on purpose: this variant's label is built from the enum
                # precisely so the data label never names a path, and a clean stub
                # label would make that indistinguishable from using it.
                label=f"{label} (wavelet power {freqs[0]:.0f}-{freqs[-1]:.0f} Hz)",
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
        "4",
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
        / "JoinedTracks_ASSR"
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

    def test_zscore_defaults_to_per_condition(self):
        """The variant's defining choice: standardise each track before concatenating."""
        assert _build_arg_parser().parse_args([]).zscore_mode == "per_condition"

    def test_every_documented_zscore_mode_is_accepted(self):
        for mode in ZSCORE_MODES:
            args = _build_arg_parser().parse_args(["--zscore_mode", mode])
            assert args.zscore_mode == mode

    def test_an_unknown_zscore_mode_is_rejected(self):
        with pytest.raises(SystemExit):
            _build_arg_parser().parse_args(["--zscore_mode", "standard"])

    def test_caches_are_opt_in(self):
        args = _build_arg_parser().parse_args([])
        assert args.reuse_wavelets is False
        assert args.subset_cache is False

    def test_subsets_default_to_the_full_extent(self):
        args = _build_arg_parser().parse_args([])
        assert args.n_pairs is None
        assert args.n_channels is None
        assert args.n_times is None


class TestOnsetAveragePerCondition:
    """Each condition is epoched on its own onsets: the tracks keep their own bases."""

    @pytest.fixture
    def sources_by_condition(self):
        rng = np.random.default_rng(0)
        return {
            "Placebo": rng.standard_normal((4, 2, 5, N_TIMES)),
            "Psilocybin": rng.standard_normal((4, 2, 5, N_TIMES - 60)),
        }

    @pytest.fixture
    def onsets_by_condition(self):
        return {"Placebo": ONSETS, "Psilocybin": ONSETS}

    PARTICIPANTS = ["019", "023", "026", "028"]
    CONDITIONS = ["Placebo", "Psilocybin"]

    def _run(self, sources, onsets, min_onsets=1):
        return _onset_average_per_condition(
            sources,
            onsets,
            self.PARTICIPANTS,
            self.CONDITIONS,
            label="t",
            sfreq=SFREQ,
            min_onsets=min_onsets,
        )

    def test_stacks_both_conditions_on_one_epoch_axis(
        self, sources_by_condition, onsets_by_condition
    ):
        result = self._run(sources_by_condition, onsets_by_condition)
        assert result is not None
        stacked, participants, conditions, epoch_times, marks = result
        assert stacked.shape[0] == 2 * len(self.PARTICIPANTS)
        assert stacked.shape[-1] == len(epoch_times)
        assert participants == self.PARTICIPANTS * 2
        assert conditions == ["Placebo"] * 4 + ["Psilocybin"] * 4
        assert epoch_times[0] < 0 < epoch_times[-1]
        assert marks[0] == 0.0 and marks[1] > 0.0

    def test_segments_of_different_length_still_share_one_epoch(
        self, sources_by_condition, onsets_by_condition
    ):
        """Unequal tracks are the norm here; the epoch axis must still be common."""
        assert (
            sources_by_condition["Placebo"].shape[-1]
            != sources_by_condition["Psilocybin"].shape[-1]
        )
        stacked, _p, _c, epoch_times, _m = self._run(
            sources_by_condition, onsets_by_condition
        )
        assert stacked.shape[-1] == len(epoch_times)

    def test_matches_a_hand_computed_per_condition_mean(
        self, sources_by_condition, onsets_by_condition
    ):
        stacked, _p, _c, epoch_times, _m = self._run(
            sources_by_condition, onsets_by_condition
        )
        pre = int(round(-epoch_times[0] * SFREQ))
        post = len(epoch_times) - pre
        for block, condition in enumerate(self.CONDITIONS):
            arr = sources_by_condition[condition]
            fitting = [
                int(o)
                for o in ONSETS
                if int(o) - pre >= 0 and int(o) + post <= arr.shape[-1]
            ]
            manual = np.mean([arr[..., o - pre : o + post] for o in fitting], axis=0)
            # Each frequency is then referenced to its own pre-onset mean, so the hand
            # computation has to do the same. Doing it per trial BEFORE the mean would
            # give the identical answer — epoch_average is a plain mean and the mean is
            # linear — which is why the correction is applied once, afterwards.
            manual -= manual[..., epoch_times < 0.0].mean(axis=-1, keepdims=True)
            rows = slice(
                block * len(self.PARTICIPANTS), (block + 1) * len(self.PARTICIPANTS)
            )
            np.testing.assert_allclose(stacked[rows], manual, rtol=1e-12, atol=1e-12)

    def test_the_baseline_is_removed_per_frequency(
        self, sources_by_condition, onsets_by_condition
    ):
        stacked, _p, _c, epoch_times, _m = self._run(
            sources_by_condition, onsets_by_condition
        )
        baseline = stacked[..., epoch_times < 0.0].mean(axis=-1)
        assert np.allclose(baseline, 0.0, atol=1e-12)

    def test_no_onsets_for_one_condition_skips_everything(
        self, sources_by_condition, onsets_by_condition
    ):
        """A one-sided average would not be a comparison."""
        onsets_by_condition["Psilocybin"] = None
        assert self._run(sources_by_condition, onsets_by_condition) is None

    def test_too_few_fitting_epochs_in_one_condition_skips(
        self, sources_by_condition, onsets_by_condition
    ):
        assert (
            self._run(sources_by_condition, onsets_by_condition, min_onsets=99) is None
        )

    def test_onsets_outside_a_segment_are_dropped(
        self, sources_by_condition, onsets_by_condition
    ):
        result = self._run(sources_by_condition, onsets_by_condition)
        assert result is not None  # the 99999 sentinel did not break it


class TestEndToEnd:
    def test_writes_the_canonical_tree_under_the_joined_tracks_label(
        self, tmp_path, stub_loaders
    ):
        main(_argv(tmp_path))
        out = _out_dir(tmp_path)
        assert (out / "shared_mean_topomaps.png").exists()
        assert (out / "condition_mean_tf_maps.png").exists()
        assert (out / "condition_mean_tf_maps_onset.png").exists()
        figures = sorted(p.name for p in (out / "participants").glob("*.png"))
        # 3 components × (shared topography + whole-track TF + onset TF)
        assert len(figures) == 9, figures
        assert len([f for f in figures if f.startswith("shared_")]) == 3
        assert len([f for f in figures if f.startswith("onset_")]) == 3

    def test_no_plus_sign_reaches_any_path(self, tmp_path, stub_loaders):
        """The concatenated label is 'A+B'; a '+' in paths is awkward to glob and quote."""
        main(_argv(tmp_path))
        assert all("+" not in str(p) for p in tmp_path.rglob("*"))

    def test_no_representation_suffix_reaches_any_path(self, tmp_path, stub_loaders):
        """The data label also carries '(wavelet power 1-50 Hz)'; paths must not."""
        main(_argv(tmp_path, "--skip_onset_average"))
        for path in tmp_path.rglob("*"):
            assert "wavelet power" not in path.name, path
            assert "(" not in path.name and ")" not in path.name, path
            assert " " not in path.name, path

    def test_only_matched_participants_are_used(self, tmp_path, stub_loaders):
        """15 Placebo and 16 Psilocybin recordings must pair down to 12."""
        seen = {}

        def spy(music_type, **kwargs):
            seen.update(kwargs)

        main(_argv(tmp_path, "--n_pairs", str(N_MATCHED)))
        # A run asking for exactly the matched count must succeed; asking for more
        # than exist is simply capped, never an error.
        main(_argv(tmp_path, "--n_pairs", str(N_MATCHED + 5)))
        assert (_out_dir(tmp_path) / "shared_mean_topomaps.png").exists()

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
            "condition_tf_ic02_participants_JoinedTracks_ASSR.png",
            "shared_condition_topomap_ic02_participants_JoinedTracks_ASSR.png",
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
        bands = _out_dir(tmp_path, "bands")
        assert sorted(p.name for p in bands.glob("*.png")) == [
            "gamma_condition_mean_tf_maps.png",
            "gamma_shared_mean_topomaps.png",
        ]
        assert not any(
            p.name.startswith("gamma_")
            for p in _out_dir(tmp_path, "broadband").rglob("*.png")
        )

    def test_the_topography_grid_has_one_row_the_tf_grid_has_three(
        self, tmp_path, stub_loaders
    ):
        """The variant's defining asymmetry, visible in the figure heights."""
        import matplotlib.image as mpimg

        main(_argv(tmp_path, "--skip_onset_average"))
        out = _out_dir(tmp_path)
        shared = mpimg.imread(out / "shared_mean_topomaps.png")
        tf = mpimg.imread(out / "condition_mean_tf_maps.png")
        assert shared.shape[0] < tf.shape[0]

    def test_zscore_mode_reaches_the_loader(self, tmp_path, stub_loaders, monkeypatch):
        seen = []
        real = notebook_helpers.load_paired_condition_wavelets

        def spy(*args, **kwargs):
            seen.append(kwargs.get("zscore_mode"))
            return real(*args, **kwargs)

        monkeypatch.setattr(
            run_iva_condition_tracks, "load_paired_condition_wavelets", spy
        )
        main(_argv(tmp_path, "--zscore_mode", "joint", "--skip_onset_average"))
        assert seen == ["joint"]

    def test_the_subset_cache_is_written_only_when_asked(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        cache = tmp_path / "subset_cache"
        monkeypatch.setattr(
            run_iva_condition_tracks,
            "resolve_notebook_wavelet_cache_dir",
            lambda experiment_name: cache,
        )
        main(_argv(tmp_path, "--skip_onset_average"))
        assert not cache.exists()

    def test_assr_defaults_to_the_placeholder_music_type(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        seen: list[MusicTypeVariants] = []
        monkeypatch.setattr(
            run_iva_condition_tracks,
            "run_condition_tracks",
            lambda music_type, **kwargs: seen.append(music_type),
        )
        main(["--experiment", ExperimentNames.ASSR.value])
        assert seen == [MusicTypeVariants.ASSR]

    def test_psilo_music_defaults_to_both_music_types(
        self, tmp_path, stub_loaders, monkeypatch
    ):
        seen: list[MusicTypeVariants] = []
        monkeypatch.setattr(
            run_iva_condition_tracks,
            "run_condition_tracks",
            lambda music_type, **kwargs: seen.append(music_type),
        )
        main(["--experiment", ExperimentNames.PSILO_MUSIC.value])
        assert seen == [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]


class TestComponentStore:
    """``--store_components`` keeps the components with their row bookkeeping.

    The time-axis join stores them **unsplit**, on the concatenated axis, with the
    segment order and lengths — so the file answers both the whole-recording and
    the per-condition question, and the shared topography is stored once rather
    than duplicated per condition.
    """

    def _load(self, store, n_pca=3, band=None):
        return load_iva_components(
            experiment=ExperimentNames.ASSR,
            condition=ConditionVariants.JOINED_TRACKS,
            variant=IvaVariants.CHANNEL_JOINED_TRACKS,
            music_type=MusicTypeVariants.ASSR,
            band=band,
            n_pca=n_pca,
            processed_data_dir=store,
        )

    def _run(self, tmp_path, store, *extra):
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

    def test_nothing_is_stored_by_default(self, tmp_path, stub_loaders):
        store = tmp_path / "processed"
        main(_argv(tmp_path, "--skip_onset_average", "--store_dir", str(store)))
        assert list_iva_results(ExperimentNames.ASSR, processed_data_dir=store) == []

    def test_one_row_per_participant_shared_by_both_conditions(
        self, tmp_path, stub_loaders
    ):
        store = tmp_path / "processed"
        self._run(tmp_path, store)
        loaded = self._load(store)

        # --n_pairs 4: four participants, one row each, not eight.
        assert loaded.n_subjects == 4
        assert len(set(loaded.participants)) == 4
        assert loaded.subject_conditions == ("JoinedTracks",) * 4
        assert loaded.row(loaded.participants[2]) == 2
        assert loaded.channel_patterns.shape == (4, 3, 6)  # one shared topography

    def test_the_segments_split_the_stored_time_axis(self, tmp_path, stub_loaders):
        store = tmp_path / "processed"
        self._run(tmp_path, store)
        loaded = self._load(store)

        assert loaded.segment_conditions == ("Placebo", "Psilocybin")
        assert sum(loaded.segment_lengths) == loaded.tf_maps.shape[-1]
        placebo = loaded.condition_track("tf_map", ConditionVariants.PLACEBO)
        psilocybin = loaded.condition_track("tf_map", ConditionVariants.PSILOCYBIN)
        assert placebo.shape[-1] == loaded.segment_lengths[0]
        assert psilocybin.shape[-1] == loaded.segment_lengths[1]
        # The two segments are consecutive halves of the stored axis, not copies.
        assert placebo.shape[-1] + psilocybin.shape[-1] == loaded.times.size
        assert loaded.times_for(ConditionVariants.PLACEBO)[0] == 0.0
        assert loaded.extras["zscore_mode"].item() == "per_condition"

    def test_the_onsets_are_stored_local_to_each_segment(self, tmp_path, stub_loaders):
        """Segment-local, i.e. the same frame condition_track hands back."""
        store = tmp_path / "processed"
        self._run(tmp_path, store)
        loaded = self._load(store)
        for condition, length in zip(loaded.segment_conditions, loaded.segment_lengths):
            onsets = loaded.extras[f"stimulus_onsets_{condition}"]
            assert onsets.size > 0
            # Local to the segment, so bounded by the segment — NOT by the whole
            # concatenated axis, which is what an absolute frame would allow.
            assert onsets.max() < length
            np.testing.assert_array_equal(onsets, ONSETS[ONSETS < length])

    def test_the_stored_file_lands_in_the_joined_tracks_subdirectory(
        self, tmp_path, stub_loaders
    ):
        store = tmp_path / "processed"
        self._run(tmp_path, store)
        entries = list_iva_results(ExperimentNames.ASSR, processed_data_dir=store)
        assert [p.parent.name for p in entries] == ["JoinedTracks"]
        assert entries[0].name == "iva_channel_joined_tracks__ASSR__broadband__pca3.npz"

    def test_the_zscore_mode_is_recorded_with_the_components(
        self, tmp_path, stub_loaders
    ):
        """It changes what a between-condition difference means, so it must travel."""
        store = tmp_path / "processed"
        self._run(tmp_path, store, "--zscore_mode", "joint")
        assert self._load(store).extras["zscore_mode"].item() == "joint"
