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
from src.analysis.assr_trials import (  # noqa: E402
    FULL_MASK_LABEL,
    MASK_LABELS,
    MASK_SLUGS,
)
from src.analysis.data_representations import (  # noqa: E402
    AnalysisData,
    DataRepresentation,
)
from src.analysis.wavelet_jica import VARIANT_RECIPE  # noqa: E402
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    JicaVariants,
    SingleDataMetadata,
    SpectrumTypeVariants,
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

    def test_the_filter_by_signal_grid_runs_by_default(self):
        """Six cells, every one per-trial baselined so all six share units.

        The IVA stage's grid cell for cell, which is what makes a row of either CSV
        readable against the other. ``zscored`` is deliberately absent: without a
        baseline it is not in the reference's units, which is the whole point of
        running a grid.
        """
        variants = _build_arg_parser().parse_args([]).signal_variants
        assert variants == [
            "prestim",
            "zscored_prestim",
            "mean_prestim",
            "mean_zscored_prestim",
            "masked_prestim",
            "mean_masked_prestim",
        ]
        assert "zscored" not in variants
        assert all(VARIANT_RECIPE[v]["baseline"] for v in variants)
        # Three spatial filters x (raw | z-scored), minus the two cells a restricted
        # filter does not have: the weights presume the scaling they were fitted on, so
        # the masked pair reads the z-scored signal only.
        assert sorted(VARIANT_RECIPE[v]["filter"] for v in variants) == [
            "masked",
            "mean",
            "mean",
            "mean_masked",
            "own",
            "own",
        ]
        assert sum(VARIANT_RECIPE[v]["zscore"] for v in variants) == 4

    def test_the_cohort_filter_is_not_pre_aligned_by_default(self):
        """A jICA block's sign is a result, so aligning before averaging invents one."""
        args = _build_arg_parser().parse_args([])
        assert args.mean_filter_align is False
        assert args.mean_filter_raw_patterns is False

    def test_a_variant_that_borrows_its_reference_needs_the_source_variant(self):
        """Holding the reference fixed is what isolates how the component was derived."""
        with pytest.raises(ValueError, match="fixed-reference row"):
            main(["--signal_variants", "zscored_prestim", "--skip_analysis"])

    def test_a_raw_signal_variant_may_still_run_on_its_own(
        self, stub_loaders, tmp_path
    ):
        """It borrows too, but the borrow is a no-op, so requiring `prestim` would be
        an obstacle rather than a safeguard: the fixed rows do not depend on the
        learned filter, so recomputing them gives the identical numbers.
        """
        results_dir = tmp_path / "results"
        main(
            _argv(
                tmp_path / "plots",
                results_dir,
                "--signal_variants",
                "mean_prestim",
            )
        )
        frame = pd.read_csv(
            results_dir / f"jica_tests__joined_tracks__ASSR__ica{N_ICA}.csv"
        )
        assert set(frame["variant"]) == {"mean_prestim"}

    def test_a_single_condition_is_refused(self):
        with pytest.raises(ValueError, match="exactly two distinct conditions"):
            main(["--conditions", "Placebo", "Placebo", "--skip_analysis"])


@pytest.mark.parametrize("join", ["joined_tracks", "joined"])
class TestWaveletCache:
    """The stage-03 caches are read, never silently recomputed."""

    def test_the_source_cache_path_carries_the_band_subdirectory(
        self, join, stub_loaders, monkeypatch, tmp_path
    ):
        """A path one level up is a cache miss, and a miss recomputes tens of GB.

        The loaders take the cache directory with the band subdirectory already on it,
        so resolving only ``<experiment>/wavelets`` matches nothing and sends the run
        through the Morlet transform again — on the spliced signal, which is not what
        the cache holds.
        """
        seen: list[Path] = []
        stubbed = notebook_helpers.compute_wavelet_datasets

        def spy(*args, **kwargs):
            seen.append(Path(kwargs["wavelet_dir"]))
            return stubbed(*args, **kwargs)

        monkeypatch.setattr(notebook_helpers, "compute_wavelet_datasets", spy)
        main(
            _argv(
                tmp_path / "plots",
                tmp_path / "results",
                "--join",
                join,
                "--skip_decomposition_plots",
                "--skip_analysis",
            )
        )

        assert seen, "the loaders were never asked for a wavelet dataset"
        for path in seen:
            assert path.parts[-3:] == (
                ExperimentNames.ASSR.value,
                "wavelets",
                SpectrumTypeVariants.BROADBAND.value,
            )


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
        # One course figure per cell of the (filter x signal) grid.
        for variant in (
            "prestim",
            "zscored_prestim",
            "mean_prestim",
            "mean_zscored_prestim",
            "masked_prestim",
            "mean_masked_prestim",
        ):
            assert (
                analysis / f"trial_course_by_source_40hz_{variant}.png"
            ).is_file(), variant
        # ...and one p-value / SNR figure per REFERENCE row, because "better than the
        # reference?" is a different question for each of the three.
        for slug in MASK_SLUGS.values():
            assert (analysis / f"pvalue_summary_40hz_prestim_{slug}.png").is_file(), (
                slug
            )
            assert (analysis / f"snr_vs_reference_40hz_{slug}.png").is_file(), slug
        # Off by default, so it must not have been drawn.
        assert not (analysis / "trial_course_by_source_40hz_zscored.png").exists()

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
        assert set(frame["variant"]) == {
            "prestim",
            "zscored_prestim",
            "mean_prestim",
            "mean_zscored_prestim",
            "masked_prestim",
            "mean_masked_prestim",
        }
        # Both reference-taking families name which of the three they were judged
        # against, so a row is never ambiguous about what it was compared to.
        for family in ("vs_reference", "snr"):
            assert set(frame[frame["family"] == family]["reference"]) == set(
                MASK_LABELS
            ), family
        assert frame[frame["family"] == "contrast"]["reference"].isna().all()
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

    def test_one_reference_row_serves_the_whole_grid(
        self, join, stub_loaders, tmp_path
    ):
        """The invariant that makes the grid readable.

        Every cell is judged against exactly the same fixed-electrode numbers, so a
        difference between two cells isolates the axis they differ on — the filter, or
        the signal — and never the reference. If the reference drifted too, the
        comparison would confound them.
        """
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        frame = pd.read_csv(results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
        contrast = frame[
            (frame["family"] == "contrast") & (frame["selection"] == "40hz")
        ]
        by_variant = contrast.set_index(["variant", "source"])["median"]

        for reference in MASK_LABELS:
            baseline = by_variant[("prestim", reference)]
            for variant in (
                "zscored_prestim",
                "mean_prestim",
                "mean_zscored_prestim",
                "masked_prestim",
                "mean_masked_prestim",
            ):
                assert by_variant[(variant, reference)] == pytest.approx(baseline), (
                    variant,
                    reference,
                )

    def test_each_axis_of_the_grid_actually_moves_the_components(
        self, join, stub_loaders, tmp_path
    ):
        """Otherwise a cell is a relabelling rather than a different read-out."""
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        frame = pd.read_csv(results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
        contrast = frame[
            (frame["family"] == "contrast") & (frame["selection"] == "40hz")
        ]
        by_variant = contrast.set_index(["variant", "source"])["median"]

        # Signal axis: raw power drops the per-channel 1/sd weighting the fit folded in.
        assert by_variant[("zscored_prestim", "IC 1")] != pytest.approx(
            by_variant[("prestim", "IC 1")]
        )
        # Filter axis: one cohort operator for everybody is not each block's own row.
        assert by_variant[("mean_prestim", "IC 1")] != pytest.approx(
            by_variant[("prestim", "IC 1")]
        )

    def test_the_reference_row_is_tested_alongside_the_components(
        self, join, stub_loaders, tmp_path
    ):
        results_dir = tmp_path / "results"
        main(_argv(tmp_path / "plots", results_dir, "--join", join))
        frame = pd.read_csv(results_dir / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
        contrast = frame[frame["family"] == "contrast"]
        # Every fixed reference is contrasted like any other source; the interaction
        # family compares only the learned rows against them — a reference judged
        # against another reference is not what either family asks.
        assert set(MASK_LABELS) <= set(contrast["source"])
        assert not set(MASK_LABELS) & set(
            frame[frame["family"] == "vs_reference"]["source"]
        )


@pytest.mark.parametrize("join", ["joined_tracks", "joined"])
class TestComponentPolarity:
    """jICA orients each component ONCE, never per recording.

    FastICA's objective is invariant under flipping an entire unmixing row and nothing
    else, so after ``orient_components`` a block that comes out negative is genuinely
    inverted relative to the rest — a result, not an ambiguity. IVA-G's is invariant
    under per-dataset flips, which is why the IVA stage must anchor per recording and
    this one must not.
    """

    @pytest.fixture
    def flip_spy(self, monkeypatch):
        """Record every flip vector the read-out applies to the reduced values."""
        applied = []
        original = run_wavelet_jica.component_polarity

        def spy(mean_pattern, electrode_mask):
            flip, strength = original(mean_pattern, electrode_mask)
            applied.append({"flip": flip, "strength": strength})
            return flip, strength

        monkeypatch.setattr(run_wavelet_jica, "component_polarity", spy)
        return applied

    def test_the_sign_is_resolved_once_for_the_whole_cohort(
        self, join, flip_spy, stub_loaders, tmp_path
    ):
        """One vector over components — not one per participant, not one per condition.

        A ``(participants, components)`` flip is the failure this test exists to catch.
        Under ``--join joined`` it would be resolved per (participant, condition), so a
        participant could flip under Placebo and not under Psilocybin — which turns the
        paired difference ``placebo - psilocybin`` into ``placebo + psilocybin``.
        """
        main(
            _argv(
                tmp_path / "plots",
                tmp_path / "results",
                "--join",
                join,
                "--skip_decomposition_plots",
            )
        )
        assert len(flip_spy) == 1, "the sign must be resolved once, not per condition"
        flip = flip_spy[0]["flip"]
        assert flip.shape == (N_ICA,)
        assert set(np.unique(flip)) <= {-1.0, 1.0}

    def test_the_anchor_negates_exactly_the_components_it_flips(
        self, join, flip_spy, stub_loaders, tmp_path
    ):
        """Turning it off must recover the fit's own signs, component by component.

        Asserted against the flip vector the run actually resolved rather than against
        an assumption about which way the fixture's components happen to point, so the
        test says what it means whether or not any component needs flipping.
        """
        results = tmp_path / "results"
        for name, extra in (("on", []), ("off", ["--no_polarity_anchor"])):
            main(
                _argv(
                    tmp_path / name,
                    results / name,
                    "--join",
                    join,
                    "--skip_decomposition_plots",
                    *extra,
                )
            )

        def _contrast(root):
            frame = pd.read_csv(root / f"jica_tests__{join}__ASSR__ica{N_ICA}.csv")
            rows = frame[
                (frame["family"] == "contrast") & (frame["selection"] == "40hz")
            ]
            return rows.set_index(["variant", "source"])["median"]

        on, off = _contrast(results / "on"), _contrast(results / "off")
        # Both runs resolve the same flip; only the second declines to apply it.
        flip = flip_spy[0]["flip"]
        for k in range(N_ICA):
            source = f"IC {k + 1}"
            assert on[("prestim", source)] == pytest.approx(
                flip[k] * off[("prestim", source)]
            ), source
        # The reference is never flipped under either setting: it is a non-negative
        # electrode average, so "higher = more power there" already holds for it.
        assert on[("prestim", FULL_MASK_LABEL)] == pytest.approx(
            off[("prestim", FULL_MASK_LABEL)]
        )


class TestReferenceRow:
    """How the fixed ASSR-electrode row every component is judged against is built.

    Not a detail of presentation: :func:`~src.analysis.assr_trials.reference_snr_tests`
    subtracts this row from every component row directly, so if it is not in the same
    units the whole comparison is off by a constant. These pin the two decisions that
    put it there — equal weight per electrode, and ONE scale shared by the conditions —
    which is the same construction the stage-06 ``run_assr_snr_grid.py`` uses.
    """

    @pytest.fixture
    def roi_spy(self, monkeypatch):
        """Record every ``roi_channelwise_snr`` call, delegating to the real one."""
        calls = []
        original = run_wavelet_jica.at.roi_channelwise_snr

        def spy(trials, baseline_mask, **kwargs):
            snr, diagnostics = original(trials, baseline_mask, **kwargs)
            calls.append({"trials": trials, "diagnostics": diagnostics})
            return snr, diagnostics

        monkeypatch.setattr(run_wavelet_jica.at, "roi_channelwise_snr", spy)
        return calls

    def test_the_roi_is_normalised_before_it_is_averaged(
        self, roi_spy, stub_loaders, tmp_path
    ):
        """Equal weight per electrode, which averaging raw power first destroys.

        The helper is handed the ROI electrodes UNCOMBINED — that is what lets it
        reference each one to its own pre-stimulus window before the ROI mean is taken.
        A ``(participants, trials, samples)`` argument would mean the mean had already
        happened, i.e. every electrode weighted by its own power level.
        """
        main(
            _argv(
                tmp_path / "plots",
                tmp_path / "results",
                "--skip_decomposition_plots",
            )
        )
        assert roi_spy, "the reference row was not built the equal-weight way"
        n_roi = len(ASSR_NAMES)
        for call in roi_spy:
            trials = call["trials"]
            assert trials.ndim == 4, "the ROI channels were averaged before normalising"
            assert trials.shape[0] == 4  # --n_pairs
            assert trials.shape[1] == n_roi

    def test_the_row_is_built_once_per_selection_and_condition(
        self, roi_spy, stub_loaders, tmp_path
    ):
        """Not once per variant: every baseline variant shares the one row.

        Rebuilding it per variant would be both wasteful and a chance for the reference
        to drift between variants that are supposed to be judged against the same
        numbers.
        """
        main(
            _argv(
                tmp_path / "plots",
                tmp_path / "results",
                "--skip_decomposition_plots",
            )
        )
        # 2 selections (--halfwidths) x 2 conditions, with three signal variants running.
        assert len(roi_spy) == 4

    def test_both_conditions_land_on_one_shared_scale(
        self, roi_spy, stub_loaders, tmp_path, caplog
    ):
        """A per-condition divisor would rescale one arm of the paired contrast.

        Each call derives its own ROI baseline SD from its own participants, and those
        differ between the conditions. Applying each to its own arm injects a
        per-condition rescaling straight into the difference the contrast tests, so the
        run picks one constant — the median — and rescales both onto it.
        """
        with caplog.at_level("INFO", logger=run_wavelet_jica._logger.name):
            main(
                _argv(
                    tmp_path / "plots",
                    tmp_path / "results",
                    "--skip_decomposition_plots",
                )
            )
        per_condition = [call["diagnostics"]["roi_baseline_sd"] for call in roi_spy]
        # Otherwise the test would pass even if each arm kept its own divisor.
        assert len(set(per_condition)) > 1

        shared = [
            line for line in caplog.text.splitlines() if "equal-weight ROI over" in line
        ]
        assert len(shared) == 2  # one per selection
        for line, (first, second) in zip(
            shared, [per_condition[:2], per_condition[2:]]
        ):
            expected = float(np.median([first, second]))
            assert f"shared baseline SD {expected:.3f}" in line
            assert "NOT applied" in line  # the per-participant spread is reported only


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
