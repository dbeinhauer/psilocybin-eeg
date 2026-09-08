"""
Tests for scripts/run_wavelet_jica.py — the joint-ICA CLI.

Figure content is not asserted. What is pinned is the CLI contract, the output layout,
and the two things that make the two joins different products rather than two runs of
the same thing: which side of the decomposition gets a **shared** row, and therefore
which figure names appear.

The fixtures use the real ASSR electrode names, because the read-out is built on the
checked-in fronto-central selection and a cohort of ``standard_1020`` names would make
the reference — and the guard that protects it — untestable.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts import notebook_helpers, run_wavelet_jica  # noqa: E402
from scripts.run_wavelet_jica import (  # noqa: E402
    _STAGE_DIR,
    _build_arg_parser,
    main,
)
from src.analysis.data_representations import (  # noqa: E402
    AnalysisData,
    DataRepresentation,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    JicaVariants,
    SingleDataMetadata,
)

SFREQ = 250.0
N_TIMES = 1900  # per condition; long enough for six ASSR-spaced onsets
N_FREQS = 8
N_ICA = 2

# The real ASSR cohort: 15 vs 16 recordings, 12 present in both — but trimmed with
# --n_pairs in the argv below so the fit stays fast.
PLACEBO_IDS = [31, 39, 38, 36, 34, 24, 26, 19, 28, 37, 29, 32, 30, 23, 35]
PSILOCYBIN_IDS = [23, 33, 32, 26, 30, 19, 18, 37, 38, 39, 34, 27, 35, 29, 40, 28]
COHORT = {
    ConditionVariants.PLACEBO: PLACEBO_IDS,
    ConditionVariants.PSILOCYBIN: PSILOCYBIN_IDS,
}
ONSETS = np.append(np.arange(60, N_TIMES, 315), 99_999)
FREQS = np.linspace(1.0, 50.0, N_FREQS)

#: The checked-in fronto-central selection the reference reads, plus filler, so
#: assr_electrode_mask(strict=True) finds every listed electrode.
ASSR_NAMES = (
    (
        (ProjectPaths.PROJECT_ROOT / "config/assr_electrodes")
        / "GSN-HydroCel-257_no-fiducials.csv"
    )
    .read_text()
    .split()[1:]
)
CHANNEL_NAMES = ASSR_NAMES + [f"E{300 + i}" for i in range(4)]
N_CHANNELS = len(CHANNEL_NAMES)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def info():
    """An ``Info`` on the ASSR electrode names, with positions so topomaps can draw."""
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((N_CHANNELS, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = {name: 0.09 * xyz for name, xyz in zip(CHANNEL_NAMES, directions)}
    info = mne.create_info(CHANNEL_NAMES, sfreq=SFREQ, ch_types="eeg")
    info.set_montage(
        mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head")
    )
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
                feature_names=list(CHANNEL_NAMES),
                info=info,
            )
            for label, a in analyzers.items()
        }

    def fake_compute_wavelet_datasets(datasets, analyzers, freqs, representation, **kw):
        out = {}
        for label, ad in datasets.items():
            rng = np.random.default_rng(abs(hash(label)) % 2**32)
            # Positive, wavelet-power-like, with a planted onset-locked 40 Hz response
            # that is stronger under Placebo — so the read-out has something to find.
            # Honour the trimmed shape: --n_channels / --n_times are applied to the
            # time-domain dataset, and the real transform inherits that extent.
            n_subjects, n_channels, n_times = ad.data.shape
            data = rng.gamma(
                2.0, 0.5, size=(n_subjects, n_channels, len(freqs), n_times)
            )
            drive = 0.9 if label.startswith(ConditionVariants.PLACEBO.value) else 0.4
            envelope = np.zeros(n_times)
            for onset in ONSETS[ONSETS < n_times]:
                envelope[onset : onset + int(0.5 * SFREQ)] = 1.0
            profile = np.exp(-0.5 * ((freqs - 40.0) / 3.0) ** 2)
            data += drive * profile[None, None, :, None] * envelope[None, None, None, :]
            out[label] = AnalysisData(
                data=data.astype(np.float32),
                sfreq=SFREQ,
                representation=DataRepresentation.WAVELET_POWER,
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


def _argv(save_dir: Path, results_dir: Path, *extra: str) -> list[str]:
    return [
        "--experiment",
        "assr",
        "--n_ica",
        str(N_ICA),
        "--n_pairs",
        "4",
        "--n_times",
        str(N_TIMES),
        "--reuse_wavelets",
        "--ica_max_iter",
        "60",
        "--n_bootstrap",
        "100",
        "--wavelet_freq_min",
        "1",
        "--wavelet_freq_max",
        "50",
        "--wavelet_n_freqs",
        str(N_FREQS),
        "--save_dir",
        str(save_dir),
        "--results_dir",
        str(results_dir),
        *extra,
    ]


def _product(join: str) -> str:
    return "JoinedTracks_ASSR" if join == "joined_tracks" else "Joined_ASSR"


def _decomposition_dir(save_dir: Path, join: str) -> Path:
    variant = (
        JicaVariants.CHANNEL_JOINED_TRACKS
        if join == "joined_tracks"
        else JicaVariants.CHANNEL_JOINED
    )
    return (
        save_dir
        / _STAGE_DIR
        / _product(join)
        / "broadband"
        / variant.value
        / f"ica_{N_ICA}"
    )


def _analysis_dir(save_dir: Path, join: str) -> Path:
    return (
        save_dir
        / _STAGE_DIR
        / _product(join)
        / "broadband"
        / JicaVariants.COMPONENT_ANALYSIS.value
        / f"ica_{N_ICA}"
    )


class TestCliDefaults:
    def test_defaults_match_the_standard_assr_run(self):
        args = _build_arg_parser().parse_args([])
        assert args.experiment == "assr"
        assert args.join == "joined_tracks"
        assert args.n_ica == 5
        assert args.conditions == ["Placebo", "Psilocybin"]
        assert args.random_state == 42

    def test_the_two_extents_that_break_the_analysis_default_to_full(self):
        """--n_channels and --n_pairs are not free knobs; see the module docstring."""
        args = _build_arg_parser().parse_args([])
        assert args.n_channels is None
        assert args.n_pairs is None

    def test_test_directions_encode_the_registered_priors(self):
        args = _build_arg_parser().parse_args([])
        # Psilocybin lowers the 40 Hz response -> Placebo - Psilocybin > 0.
        assert args.contrast_alternative == "greater"
        # A per-block filter is a contribution to a shared source, so it is expected
        # to recover less than a fixed selection aimed at the response.
        assert args.snr_alternative == "less"

    def test_all_three_signal_variants_run_by_default(self):
        assert _build_arg_parser().parse_args([]).signal_variants == [
            "prestim",
            "zscored",
            "zscored_prestim",
        ]

    def test_a_variant_that_borrows_its_reference_needs_the_source_variant(self):
        """Holding the reference fixed is what isolates how the component was derived."""
        with pytest.raises(ValueError, match="fixed-reference row"):
            main(["--signal_variants", "zscored_prestim", "--skip_analysis"])

    def test_a_single_condition_is_refused(self):
        with pytest.raises(ValueError, match="exactly two distinct conditions"):
            main(["--conditions", "Placebo", "Placebo", "--skip_analysis"])


@pytest.mark.parametrize("join", ["joined_tracks", "joined"])
class TestEndToEnd:
    def test_decomposition_and_readout_figures_are_written(
        self, join, stub_loaders, tmp_path
    ):
        save_dir, results_dir = tmp_path / "plots", tmp_path / "results"
        main(_argv(save_dir, results_dir, "--join", join))

        decomposition = _decomposition_dir(save_dir, join)
        assert (decomposition / "loading_bars_absolute.png").is_file()
        assert (decomposition / "loading_bars_within_component.png").is_file()

        analysis = _analysis_dir(save_dir, join)
        assert (analysis / "trial_course_by_source_40hz_prestim.png").is_file()
        assert (analysis / "trial_course_by_source_40hz_zscored.png").is_file()
        assert (analysis / "trial_course_by_source_40hz_zscored_prestim.png").is_file()
        assert (analysis / "pvalue_summary_40hz_prestim.png").is_file()
        assert (analysis / "snr_vs_reference_40hz.png").is_file()

    def test_the_shared_side_gets_the_shared_figure_name(
        self, join, stub_loaders, tmp_path
    ):
        """Which side is shared is exactly what the join decides."""
        save_dir = tmp_path / "plots"
        main(_argv(save_dir, tmp_path / "results", "--join", join))
        out = _decomposition_dir(save_dir, join)
        if join == "joined_tracks":
            # One block per participant: the topography is shared, the TF maps split.
            assert (out / "shared_mean_topomaps.png").is_file()
            assert (out / "condition_tf_maps.png").is_file()
            assert not (out / "condition_mean_topomaps.png").exists()
            assert not (out / "shared_tf_maps.png").exists()
        else:
            # One block per recording: the TF map is shared, the topographies split.
            assert (out / "condition_mean_topomaps.png").is_file()
            assert (out / "shared_tf_maps.png").is_file()
            assert not (out / "shared_mean_topomaps.png").exists()
            assert not (out / "condition_tf_maps.png").exists()

    def test_the_test_csv_carries_every_family_and_its_interval(
        self, join, stub_loaders, tmp_path
    ):
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        csv_path = results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv"
        assert csv_path.is_file()
        frame = pd.read_csv(csv_path)
        assert set(frame["family"]) == {"contrast", "vs_reference", "snr"}
        # Both --halfwidths are carried through, so a result can be checked against
        # the other frequency window without a second run.
        assert set(frame["selection"]) == {"40hz", "35-45hz"}
        assert set(frame["variant"]) == {"prestim", "zscored", "zscored_prestim"}
        for column in (
            "median",
            "effect",
            "same sign",
            "alt",
            "p",
            "ci_low",
            "ci_high",
        ):
            assert column in frame.columns
        assert ((frame["p"] > 0) & (frame["p"] <= 1)).all()
        assert (frame["n_participants"] == 4).all()

    def test_the_borrowed_reference_row_is_identical_and_the_components_are_not(
        self, join, stub_loaders, tmp_path
    ):
        """The invariant that makes 'zscored_prestim' interpretable.

        Its fixed-electrode row is 'prestim''s verbatim, so the two variants are judged
        against exactly the same numbers and any difference between them is the cost of
        'prestim' approximating the component rather than reproducing it. If the
        reference drifted too, the comparison would confound the two.
        """
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        frame = pd.read_csv(results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
        contrast = frame[
            (frame["family"] == "contrast") & (frame["selection"] == "40hz")
        ]
        by_variant = contrast.set_index(["variant", "source"])["median"]

        reference = "ASSR-mask"
        assert by_variant[("zscored_prestim", reference)] == pytest.approx(
            by_variant[("prestim", reference)]
        )
        # The learned rows are read a different way, so they must NOT match.
        assert by_variant[("zscored_prestim", "IC 1")] != pytest.approx(
            by_variant[("prestim", "IC 1")]
        )

    def test_the_reference_row_is_tested_alongside_the_components(
        self, join, stub_loaders, tmp_path
    ):
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        frame = pd.read_csv(results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
        contrast = frame[frame["family"] == "contrast"]
        # The fixed reference is contrasted like any other source; the interaction
        # family compares only the learned rows against it.
        assert "ASSR-mask" in set(contrast["source"])
        assert "ASSR-mask" not in set(
            frame[frame["family"] == "vs_reference"]["source"]
        )


class TestStageSelection:
    def test_skip_analysis_writes_no_readout_and_no_csv(self, stub_loaders, tmp_path):
        save_dir, results_dir = tmp_path / "plots", tmp_path / "results"
        main(_argv(save_dir, results_dir, "--skip_analysis"))
        assert (
            _decomposition_dir(save_dir, "joined_tracks") / "condition_tf_maps.png"
        ).is_file()
        assert not _analysis_dir(save_dir, "joined_tracks").exists()
        assert not results_dir.exists()

    def test_skip_decomposition_plots_still_runs_the_readout(
        self, stub_loaders, tmp_path
    ):
        save_dir, results_dir = tmp_path / "plots", tmp_path / "results"
        main(_argv(save_dir, results_dir, "--skip_decomposition_plots"))
        assert not _decomposition_dir(save_dir, "joined_tracks").exists()
        assert (
            _analysis_dir(save_dir, "joined_tracks")
            / "trial_course_by_source_40hz_prestim.png"
        ).is_file()

    def test_one_variant_writes_one_pair_of_readout_figures(
        self, stub_loaders, tmp_path
    ):
        save_dir = tmp_path / "plots"
        main(
            _argv(
                save_dir,
                tmp_path / "results",
                "--signal_variants",
                "prestim",
                "--skip_decomposition_plots",
            )
        )
        out = _analysis_dir(save_dir, "joined_tracks")
        assert (out / "trial_course_by_source_40hz_prestim.png").is_file()
        assert not (out / "trial_course_by_source_40hz_zscored.png").exists()

    def test_participant_grids_are_opt_in(self, stub_loaders, tmp_path):
        save_dir = tmp_path / "plots"
        main(_argv(save_dir, tmp_path / "results", "--skip_analysis"))
        assert not (
            _decomposition_dir(save_dir, "joined_tracks") / "participants"
        ).exists()
        main(
            _argv(
                save_dir,
                tmp_path / "results",
                "--skip_analysis",
                "--participant_grids",
            )
        )
        grids = _decomposition_dir(save_dir, "joined_tracks") / "participants"
        assert len(list(grids.glob("*.png"))) == N_ICA


class TestGuards:
    def test_a_channel_subset_that_loses_the_reference_is_refused(
        self, stub_loaders, tmp_path
    ):
        """The selection is spread over the montage, so a leading slice misses it."""
        with pytest.raises(ValueError, match="spread over the montage"):
            main(
                _argv(
                    tmp_path / "plots",
                    tmp_path / "results",
                    "--n_channels",
                    "4",
                    "--mask_not_strict",
                )
            )

    def test_a_component_outside_the_run_is_refused(self, stub_loaders, tmp_path):
        with pytest.raises(ValueError, match="--components names IC"):
            main(
                _argv(
                    tmp_path / "plots",
                    tmp_path / "results",
                    "--skip_analysis",
                    "--components",
                    str(N_ICA + 3),
                )
            )

    def test_an_untested_halfwidth_is_refused(self, stub_loaders, tmp_path):
        with pytest.raises(ValueError, match="--test_halfwidth"):
            main(
                _argv(
                    tmp_path / "plots",
                    tmp_path / "results",
                    "--skip_decomposition_plots",
                    "--test_halfwidth",
                    "3",
                )
            )

    def test_a_missing_reference_electrode_is_refused_by_default(
        self, monkeypatch, stub_loaders, tmp_path
    ):
        """strict=True is the default: under-selecting silently changes the reference."""
        short = CHANNEL_NAMES[:-6]
        monkeypatch.setattr(
            run_wavelet_jica,
            "_MIN_REFERENCE_ELECTRODES",
            1,  # so the other guard cannot fire first
        )
        original = run_wavelet_jica.assr_electrode_mask

        def fake_mask(channel_names, coordinate_system, *, strict=True):
            return original(short, coordinate_system, strict=strict)

        monkeypatch.setattr(run_wavelet_jica, "assr_electrode_mask", fake_mask)
        with pytest.raises(ValueError):
            main(
                _argv(
                    tmp_path / "plots",
                    tmp_path / "results",
                    "--skip_decomposition_plots",
                )
            )
