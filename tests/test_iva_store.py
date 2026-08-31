"""
Tests for src/io/iva_store.py — the IVA component store.

What is pinned here is the contract a reader depends on months after the run: the
path layout (so a sweep cannot overwrite a sibling), the round trip of every
canonical array, and above all the **participant mapping** — that a row can be
resolved back to a participant and a condition, that an ambiguous label is
reported rather than silently resolved to its first match, and that a mislabelled
write (arrays disagreeing with the axes or with the participant list) is refused
at write time rather than becoming an unreadable file.
"""

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import (  # noqa: E402
    FORMAT_VERSION,
    ONSETS_EXTRA_PREFIX,
    iva_results_filename,
    iva_results_path,
    list_iva_results,
    load_iva_components,
    load_many_iva_components,
    save_iva_components,
)

SFREQ = 250.0
N_SUBJECTS = 4
N_PCA = 3
N_FREQS = 5
N_TIMES = 40
N_CHANNELS = 6

FREQS = np.linspace(1.0, 30.0, N_FREQS)
TIMES = np.arange(N_TIMES) / SFREQ
# Real GSN-HydroCel names (1-based), so topo_info finds every one in the montage
# instead of falling through to its on_missing="warn" path.
CHANNELS = [f"E{i + 1}" for i in range(N_CHANNELS)]


def _arrays(seed: int = 0) -> dict:
    """A channel-variant array set: a TF map plus a channel topography."""
    rng = np.random.default_rng(seed)
    return {
        IvaComponentArrays.TF_MAP: rng.standard_normal(
            (N_SUBJECTS, N_PCA, N_FREQS, N_TIMES)
        ),
        IvaComponentArrays.CHANNEL_PATTERN: rng.standard_normal(
            (N_SUBJECTS, N_PCA, N_CHANNELS)
        ),
    }


def _save(tmp_path, **overrides):
    """Write one store entry into *tmp_path*, overriding any argument."""
    params = {
        "experiment": ExperimentNames.ASSR,
        "condition": ConditionVariants.PLACEBO,
        "variant": IvaVariants.CHANNEL,
        "music_type": MusicTypeVariants.ASSR,
        "band": None,
        "n_pca": N_PCA,
        "sfreq": SFREQ,
        "participants": [f"{31 + s:03d}" for s in range(N_SUBJECTS)],
        "arrays": _arrays(),
        "freqs": FREQS,
        "times": TIMES,
        "channel_names": CHANNELS,
        "processed_data_dir": tmp_path,
    }
    params.update(overrides)
    return save_iva_components(**params)


class TestPaths:
    def test_the_condition_is_a_directory_and_the_rest_a_filename(self, tmp_path):
        path = iva_results_path(
            ExperimentNames.ASSR,
            ConditionVariants.PLACEBO,
            IvaVariants.CHANNEL,
            MusicTypeVariants.ASSR,
            None,
            10,
            processed_data_dir=tmp_path,
        )
        assert path.parent == (
            tmp_path / "assr" / ProjectPaths.IVA_RESULTS_DIR_NAME / "Placebo"
        )
        assert path.name == "iva_channel__ASSR__broadband__pca10.npz"

    def test_a_band_run_gets_its_own_name(self):
        assert (
            iva_results_filename(
                IvaVariants.CHANNEL, MusicTypeVariants.ASSR, "alpha", 10
            )
            == "iva_channel__ASSR__alpha__pca10.npz"
        )

    def test_a_pca_sweep_cannot_overwrite_itself(self, tmp_path):
        first = _save(tmp_path)
        second = _save(
            tmp_path,
            n_pca=2,
            arrays={
                IvaComponentArrays.TF_MAP: np.zeros((N_SUBJECTS, 2, N_FREQS, N_TIMES)),
            },
        )
        assert first != second
        assert first.exists() and second.exists()

    def test_a_rerun_replaces_its_own_entry(self, tmp_path):
        first = _save(tmp_path)
        second = _save(tmp_path, arrays=_arrays(seed=1))
        assert first == second
        assert (
            len(list_iva_results(ExperimentNames.ASSR, processed_data_dir=tmp_path))
            == 1
        )

    def test_default_paths_live_under_the_project_processed_data_dir(self):
        path = iva_results_path(
            ExperimentNames.PSILO_MUSIC,
            ConditionVariants.JOINED,
            IvaVariants.CHANNEL_JOINED,
            MusicTypeVariants.CLASSICAL,
            None,
            20,
        )
        assert path.is_relative_to(
            ProjectPaths.PROCESSED_DATA_DIR / ExperimentNames.PSILO_MUSIC.value
        )

    def test_listing_an_empty_store_is_not_an_error(self, tmp_path):
        assert list_iva_results(ExperimentNames.ASSR, processed_data_dir=tmp_path) == []

    def test_listing_spans_conditions_or_narrows_to_one(self, tmp_path):
        _save(tmp_path)
        _save(tmp_path, condition=ConditionVariants.PSILOCYBIN)
        assert (
            len(list_iva_results(ExperimentNames.ASSR, processed_data_dir=tmp_path))
            == 2
        )
        narrowed = list_iva_results(
            ExperimentNames.ASSR,
            ConditionVariants.PSILOCYBIN,
            processed_data_dir=tmp_path,
        )
        assert [p.parent.name for p in narrowed] == ["Psilocybin"]


class TestRoundTrip:
    def test_every_array_and_axis_comes_back(self, tmp_path):
        arrays = _arrays()
        path = _save(tmp_path, arrays=arrays)
        loaded = load_iva_components(path)

        assert loaded.variant is IvaVariants.CHANNEL
        assert loaded.experiment is ExperimentNames.ASSR
        assert loaded.condition is ConditionVariants.PLACEBO
        assert loaded.music_type is MusicTypeVariants.ASSR
        assert loaded.band is None
        assert (loaded.n_pca, loaded.n_components) == (N_PCA, N_PCA)
        assert loaded.sfreq == SFREQ
        assert loaded.n_subjects == N_SUBJECTS
        assert loaded.label == "Placebo_ASSR"
        assert loaded.path == path
        np.testing.assert_allclose(loaded.freqs, FREQS)
        np.testing.assert_allclose(loaded.times, TIMES)
        assert list(loaded.channel_names) == CHANNELS
        np.testing.assert_allclose(
            loaded.tf_maps, arrays[IvaComponentArrays.TF_MAP], atol=1e-6
        )
        np.testing.assert_allclose(
            loaded.channel_patterns,
            arrays[IvaComponentArrays.CHANNEL_PATTERN],
            atol=1e-6,
        )

    def test_a_descriptor_finds_the_same_file_as_the_path(self, tmp_path):
        path = _save(tmp_path)
        loaded = load_iva_components(
            experiment=ExperimentNames.ASSR,
            condition=ConditionVariants.PLACEBO,
            variant=IvaVariants.CHANNEL,
            music_type=MusicTypeVariants.ASSR,
            n_pca=N_PCA,
            processed_data_dir=tmp_path,
        )
        assert loaded.path == path

    def test_a_band_run_round_trips_its_band(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path, band="alpha"))
        assert loaded.band == "alpha"

    def test_float32_is_the_default_and_float64_is_available(self, tmp_path):
        assert load_iva_components(_save(tmp_path)).tf_maps.dtype == np.float32
        assert (
            load_iva_components(_save(tmp_path, dtype=np.float64)).tf_maps.dtype
            == np.float64
        )

    def test_extras_ride_along_untouched(self, tmp_path):
        path = _save(tmp_path, extras={"rank_score": np.arange(N_PCA, dtype=float)})
        loaded = load_iva_components(path)
        np.testing.assert_allclose(loaded.extras["rank_score"], np.arange(N_PCA))
        # An extra named like a coordinate axis cannot shadow it.
        loaded = load_iva_components(_save(tmp_path, extras={"freqs": np.zeros(2)}))
        np.testing.assert_allclose(loaded.freqs, FREQS)
        assert loaded.extras["freqs"].shape == (2,)

    def test_the_format_version_is_recorded(self, tmp_path):
        with np.load(_save(tmp_path)) as stored:
            assert int(stored["format_version"]) == FORMAT_VERSION

    def test_axes_are_optional(self, tmp_path):
        """A variant without a frequency axis stores none, and says so."""
        loaded = load_iva_components(
            _save(
                tmp_path,
                variant=IvaVariants.TIME,
                arrays={
                    IvaComponentArrays.TIMECOURSE: np.zeros(
                        (N_SUBJECTS, N_PCA, N_TIMES)
                    )
                },
                freqs=None,
                channel_names=None,
            )
        )
        assert loaded.freqs is None
        assert loaded.channel_names is None
        np.testing.assert_allclose(loaded.times, TIMES)

    def test_load_many_keeps_the_given_order(self, tmp_path):
        first = _save(tmp_path)
        second = _save(tmp_path, condition=ConditionVariants.PSILOCYBIN)
        loaded = load_many_iva_components([second, first])
        assert [r.condition for r in loaded] == [
            ConditionVariants.PSILOCYBIN,
            ConditionVariants.PLACEBO,
        ]


class TestParticipantMapping:
    def test_a_single_condition_run_maps_one_row_per_participant(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path))
        assert loaded.participants == ("031", "032", "033", "034")
        assert loaded.subject_conditions == ("Placebo",) * N_SUBJECTS
        assert loaded.row("033") == 2
        assert loaded.rows(condition=ConditionVariants.PLACEBO) == [0, 1, 2, 3]

    def test_a_subject_axis_join_keeps_both_rows_of_a_participant(self, tmp_path):
        loaded = load_iva_components(
            _save(
                tmp_path,
                condition=ConditionVariants.JOINED,
                variant=IvaVariants.CHANNEL_JOINED,
                participants=["031", "032", "031", "032"],
                subject_conditions=[
                    "Placebo",
                    "Placebo",
                    "Psilocybin",
                    "Psilocybin",
                ],
            )
        )
        assert loaded.participant_rows == {"031": (0, 2), "032": (1, 3)}
        assert loaded.row("031", ConditionVariants.PSILOCYBIN) == 2
        assert loaded.rows(condition="Psilocybin") == [2, 3]

    def test_an_ambiguous_label_is_reported_not_guessed(self, tmp_path):
        loaded = load_iva_components(
            _save(
                tmp_path,
                condition=ConditionVariants.JOINED,
                participants=["031", "032", "031", "032"],
                subject_conditions=["Placebo", "Placebo", "Psilocybin", "Psilocybin"],
            )
        )
        with pytest.raises(ValueError, match="occupies 2 rows"):
            loaded.row("031")

    def test_an_unknown_participant_raises_and_lists_what_there_is(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path))
        with pytest.raises(KeyError, match="999"):
            loaded.row("999")
        assert loaded.rows("999") == []

    def test_for_participant_slices_every_array_to_one_row(self, tmp_path):
        arrays = _arrays()
        loaded = load_iva_components(_save(tmp_path, arrays=arrays))
        one = loaded.for_participant("032")
        assert set(one) == {"tf_map", "channel_pattern"}
        np.testing.assert_allclose(
            one["tf_map"], arrays[IvaComponentArrays.TF_MAP][1], atol=1e-6
        )

    def test_the_frame_reports_the_row_bookkeeping(self, tmp_path):
        frame = load_iva_components(_save(tmp_path)).participant_frame()
        assert list(frame.columns) == ["subject_index", "participant", "condition"]
        assert frame["participant"].tolist() == ["031", "032", "033", "034"]

    def test_selecting_participants_keeps_the_stored_row_order(self, tmp_path):
        loaded = load_iva_components(
            _save(
                tmp_path,
                condition=ConditionVariants.JOINED,
                participants=["031", "032", "031", "032"],
                subject_conditions=["Placebo", "Placebo", "Psilocybin", "Psilocybin"],
            )
        )
        subset = loaded.select_participants(["032"])
        assert subset.participants == ("032", "032")
        assert subset.subject_conditions == ("Placebo", "Psilocybin")
        assert subset.tf_maps.shape[0] == 2
        np.testing.assert_allclose(subset.tf_maps[0], loaded.tf_maps[1], atol=1e-6)
        # The original is untouched.
        assert loaded.n_subjects == N_SUBJECTS

    def test_selecting_an_absent_participant_raises(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path))
        with pytest.raises(KeyError, match="999"):
            loaded.select_participants(["999"])


class TestRedrawingAndEpoching:
    """The accessors a downstream analysis of a stored run leans on."""

    def test_topo_info_rebuilds_the_stored_channel_axis(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path))
        info = loaded.topo_info()
        assert info["ch_names"] == CHANNELS
        assert info["sfreq"] == SFREQ
        assert loaded.n_channels == N_CHANNELS
        # A montage was applied, so the channels have positions to draw on.
        assert info.get_montage() is not None

    def test_topo_info_without_channel_names_is_refused(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path, channel_names=None))
        with pytest.raises(ValueError, match="stored no channel names"):
            loaded.topo_info()
        with pytest.raises(ValueError, match="stored no channel names"):
            loaded.n_channels

    def test_stimulus_onsets_round_trip_per_condition(self, tmp_path):
        onsets = np.array([5, 15, 25])
        loaded = load_iva_components(
            _save(
                tmp_path,
                extras={
                    f"{ONSETS_EXTRA_PREFIX}Placebo": onsets,
                    f"{ONSETS_EXTRA_PREFIX}Psilocybin": onsets + 1,
                },
            )
        )
        assert loaded.onset_conditions == ["Placebo", "Psilocybin"]
        np.testing.assert_array_equal(
            loaded.stimulus_onsets(ConditionVariants.PLACEBO), onsets
        )
        np.testing.assert_array_equal(loaded.stimulus_onsets("Psilocybin"), onsets + 1)

    def test_a_file_without_onsets_says_so_instead_of_failing(self, tmp_path):
        """Experiments without stimulus annotations, and files predating onsets."""
        loaded = load_iva_components(_save(tmp_path))
        assert loaded.onset_conditions == []
        assert loaded.stimulus_onsets(ConditionVariants.PLACEBO) is None


class TestMissingArrays:
    def test_asking_for_an_array_a_variant_never_made_says_what_it_has(self, tmp_path):
        loaded = load_iva_components(
            _save(
                tmp_path,
                variant=IvaVariants.FREQUENCY_CHANNEL,
                arrays={
                    IvaComponentArrays.TIMECOURSE: np.zeros(
                        (N_SUBJECTS, N_PCA, N_TIMES)
                    ),
                    IvaComponentArrays.FREQUENCY_CHANNEL_PATTERN: np.zeros(
                        (N_SUBJECTS, N_PCA, N_FREQS, N_CHANNELS)
                    ),
                },
            )
        )
        assert not loaded.has(IvaComponentArrays.TF_MAP)
        assert loaded.has(IvaComponentArrays.TIMECOURSE)
        with pytest.raises(KeyError, match="frequency_channel_pattern"):
            loaded.tf_maps


class TestTimeAxisJoin:
    def _tracks(self, tmp_path, lengths=(15, 25)):
        total = sum(lengths)
        return load_iva_components(
            _save(
                tmp_path,
                condition=ConditionVariants.JOINED_TRACKS,
                variant=IvaVariants.CHANNEL_JOINED_TRACKS,
                participants=["031", "032", "033", "034"],
                arrays={
                    IvaComponentArrays.TF_MAP: np.arange(
                        N_SUBJECTS * N_PCA * N_FREQS * total, dtype=float
                    ).reshape(N_SUBJECTS, N_PCA, N_FREQS, total),
                    IvaComponentArrays.CHANNEL_PATTERN: np.zeros(
                        (N_SUBJECTS, N_PCA, N_CHANNELS)
                    ),
                },
                times=np.arange(total) / SFREQ,
                segment_conditions=["Placebo", "Psilocybin"],
                segment_lengths=list(lengths),
            )
        )

    def test_the_row_is_shared_by_both_conditions(self, tmp_path):
        loaded = self._tracks(tmp_path)
        assert loaded.subject_conditions == ("JoinedTracks",) * N_SUBJECTS
        assert loaded.participant_rows["031"] == (0,)

    def test_a_condition_is_a_slice_of_the_time_axis(self, tmp_path):
        loaded = self._tracks(tmp_path, lengths=(15, 25))
        assert loaded.segment_boundaries == (0, 15, 40)
        assert loaded.segment_slice("Psilocybin") == slice(15, 40)
        placebo = loaded.condition_track(IvaComponentArrays.TF_MAP, "Placebo")
        psilocybin = loaded.condition_track(
            loaded.tf_maps, ConditionVariants.PSILOCYBIN
        )
        assert placebo.shape[-1] == 15
        assert psilocybin.shape[-1] == 25
        np.testing.assert_allclose(placebo, loaded.tf_maps[..., :15])
        assert loaded.times_for("Placebo").shape == (15,)
        assert loaded.times_for("Placebo")[0] == 0.0

    def test_an_unknown_segment_raises(self, tmp_path):
        loaded = self._tracks(tmp_path)
        with pytest.raises(KeyError, match="Joined"):
            loaded.segment_slice(ConditionVariants.JOINED)

    def test_a_single_condition_run_has_no_segments(self, tmp_path):
        loaded = load_iva_components(_save(tmp_path))
        assert loaded.segment_boundaries == ()
        with pytest.raises(ValueError, match="no time-axis segments"):
            loaded.segment_slice(ConditionVariants.PLACEBO)

    def test_an_array_off_the_concatenated_axis_is_refused(self, tmp_path):
        loaded = self._tracks(tmp_path)
        with pytest.raises(ValueError, match="not on the concatenated axis"):
            loaded.condition_track(np.zeros((N_SUBJECTS, N_PCA, 7)), "Placebo")


class TestWriteValidation:
    def test_no_participants_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="subject axis needs labels"):
            _save(tmp_path, participants=[])

    def test_no_arrays_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="No component arrays"):
            _save(tmp_path, arrays={})

    def test_an_unknown_array_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not a component array name"):
            _save(tmp_path, arrays={"topography": np.zeros((N_SUBJECTS, N_PCA, 3))})

    def test_a_row_count_disagreeing_with_the_labels_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="recording"):
            _save(
                tmp_path,
                arrays={
                    IvaComponentArrays.CHANNEL_PATTERN: np.zeros(
                        (N_SUBJECTS + 1, N_PCA, N_CHANNELS)
                    )
                },
            )

    def test_a_component_count_disagreeing_with_n_pca_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="component"):
            _save(
                tmp_path,
                arrays={
                    IvaComponentArrays.CHANNEL_PATTERN: np.zeros(
                        (N_SUBJECTS, N_PCA + 1, N_CHANNELS)
                    )
                },
            )

    def test_an_axis_disagreeing_with_its_coordinate_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="freq bin"):
            _save(
                tmp_path,
                arrays={
                    IvaComponentArrays.TF_MAP: np.zeros(
                        (N_SUBJECTS, N_PCA, N_FREQS + 2, N_TIMES)
                    )
                },
            )

    def test_a_wrongly_shaped_array_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="must be 3-D"):
            _save(
                tmp_path,
                arrays={
                    IvaComponentArrays.CHANNEL_PATTERN: np.zeros(
                        (N_SUBJECTS, N_PCA, N_CHANNELS, 2)
                    )
                },
            )

    def test_mismatched_subject_conditions_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="subject_conditions has 2 entries"):
            _save(tmp_path, subject_conditions=["Placebo", "Placebo"])

    def test_segments_must_come_as_a_pair(self, tmp_path):
        with pytest.raises(ValueError, match="must be given together"):
            _save(tmp_path, segment_conditions=["Placebo"])

    def test_segments_must_describe_the_time_axis(self, tmp_path):
        with pytest.raises(ValueError, match="do not describe it"):
            _save(
                tmp_path,
                segment_conditions=["Placebo", "Psilocybin"],
                segment_lengths=[3, 4],
            )


class TestReadValidation:
    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_iva_components(tmp_path / "absent.npz")

    def test_an_incomplete_descriptor_says_what_is_missing(self):
        with pytest.raises(ValueError, match="missing"):
            load_iva_components(experiment=ExperimentNames.ASSR)

    def test_a_foreign_npz_is_refused(self, tmp_path):
        alien = tmp_path / "alien.npz"
        np.savez_compressed(alien, data=np.zeros(3))
        with pytest.raises(ValueError, match="not a readable IVA component store"):
            load_iva_components(alien)

    def test_a_newer_format_version_is_refused(self, tmp_path):
        path = _save(tmp_path)
        with np.load(path) as stored:
            payload = {key: stored[key] for key in stored.files}
        payload["format_version"] = np.asarray(FORMAT_VERSION + 1)
        np.savez_compressed(path, **payload)
        with pytest.raises(ValueError, match="newer than the supported"):
            load_iva_components(path)
