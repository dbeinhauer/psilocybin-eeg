"""
Tests for scripts/run_wavelet_iva_channel.py — the channel-IVA CLI pipeline.

Figure content is not asserted; these tests pin the CLI defaults (the standard
ASSR run) and drive ``_run_iva`` end to end on tiny synthetic data, including the
``--n_bottom 0`` path — the default — where no bottom-ranked group exists and
every per-component figure has to lay itself out over the top group alone.
"""

import matplotlib

matplotlib.use("Agg")

import re  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts import run_wavelet_iva_channel  # noqa: E402
from scripts.run_wavelet_iva_channel import _build_arg_parser, _run_iva  # noqa: E402
from src.definitions.fields import ExperimentNames  # noqa: E402

SFREQ = 100.0
N_SUBJECTS = 3
N_CHANNELS = 6
N_FREQS = 4
N_TIMES = 300
N_PCA = 3


@pytest.fixture(autouse=True)
def _close_figures():
    """Never leak figures between tests."""
    yield
    plt.close("all")


@pytest.fixture
def info():
    """Minimal EEG ``Info`` with a real montage so topomaps can be drawn."""
    mne.set_log_level("ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:N_CHANNELS], sfreq=SFREQ, ch_types="eeg")
    info.set_montage(montage)
    return info


@pytest.fixture
def iva_inputs():
    """``(data_4d, freqs, onsets)`` for one synthetic dataset."""
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((N_SUBJECTS, N_CHANNELS, N_FREQS, N_TIMES)),
        np.linspace(10.0, 60.0, N_FREQS),
        np.arange(30, N_TIMES - 80, 100),
    )


def _run(info, iva_inputs, tmp_path, **kwargs):
    """Run the pipeline with small defaults, overriding any argument via ``kwargs``."""
    data_4d, freqs, onsets = iva_inputs
    params = {
        "n_pca": N_PCA,
        "n_top": N_PCA,
        "n_bottom": 0,
        "random_state": 42,
        "iva_opt_approach": "newton",
        "iva_max_iter": 32,
        "iva_w_diff_stop": 1e-6,
        "save_dir": tmp_path,
        "band": None,
        "quality": True,
        "stimulus_onsets": onsets,
        "subject_ids": [f"PSI{s:03d}" for s in range(N_SUBJECTS)],
    }
    params.update(kwargs)
    _run_iva(data_4d, SFREQ, freqs, info, label="Placebo_ASSR", **params)
    return tmp_path / "broadband" / "iva_channel" / f"pca_{params['n_pca']}"


class TestArgParser:
    def test_defaults_satisfy_the_pipeline_constraint(self) -> None:
        # n_top + n_bottom <= n_pca is validated inside _run_iva; the defaults
        # must not trip it, or a bare invocation cannot run at all.
        args = _build_arg_parser().parse_args([])
        assert args.n_top + args.n_bottom <= args.n_pca
        assert args.n_pca > 0

    def test_the_standard_assr_invocation_parses(self) -> None:
        # The setting run_channel_quality.pbs passes: 10 components, all of them
        # shown as "top", no bottom group.
        args = _build_arg_parser().parse_args(
            [
                "--experiment",
                ExperimentNames.ASSR.value,
                "--n_pca",
                "10",
                "--n_top",
                "10",
                "--n_bottom",
                "0",
                "--quality",
            ]
        )
        assert args.experiment == ExperimentNames.ASSR.value
        assert (args.n_pca, args.n_top, args.n_bottom) == (10, 10, 0)
        assert args.quality


class TestRunIvaWithoutBottomGroup:
    """``--n_bottom 0``: only the top group, and nothing may fall over."""

    def test_writes_top_figures_and_no_bottom_figures(
        self, info, iva_inputs, tmp_path
    ) -> None:
        out_dir = _run(info, iva_inputs, tmp_path)
        names = sorted(p.name for p in out_dir.iterdir())
        assert names, "no output written"
        assert any(n.endswith("_top.png") for n in names)
        assert not [n for n in names if n.endswith("_bottom.png")]

    def test_writes_every_grouped_figure_once(self, info, iva_inputs, tmp_path) -> None:
        # Every per-component grid figure is emitted for the top group; a missing
        # one means its layout silently skipped the group.
        out_dir = _run(info, iva_inputs, tmp_path)
        for stem in (
            "iva_isc_temporal",
            "iva_isc_spectral",
            "iva_topomap_mean_var",
            "iva_shared_time_frequency",
            "iva_mean_variance_over_time",
            "iva_mean_variance_over_freq",
            "iva_pairmap_time_subject",
            "iva_pairmap_subject_frequency",
            "iva_pairmap_subject_channel",
            "iva_subject_loadings",
        ):
            assert (out_dir / f"{stem}_Placebo_ASSR_top.png").exists(), stem

    def test_writes_the_ungrouped_summaries(self, info, iva_inputs, tmp_path) -> None:
        out_dir = _run(info, iva_inputs, tmp_path)
        assert (out_dir / "iva_component_ranking_Placebo_ASSR.png").exists()
        assert (out_dir / "pca_scree_Placebo_ASSR.png").exists()
        assert (out_dir / "iva_loo_isc_axis_summary_Placebo_ASSR.png").exists()

    def test_quality_analysis_runs(self, info, iva_inputs, tmp_path) -> None:
        out_dir = _run(info, iva_inputs, tmp_path)
        # One reference pair, and no raw-voltage reference figure.
        assert (out_dir / "quality_reference_wavelet_Placebo_ASSR.png").exists()
        assert not (out_dir / "quality_reference_topomap_Placebo_ASSR.png").exists()
        assert (out_dir / "quality_tf_maps_wavelet_tf_Placebo_ASSR.png").exists()
        # The whole-recording QC map, under the same per-participant signs.
        assert (out_dir / "quality_tf_maps_full_recording_Placebo_ASSR.png").exists()
        # The group-mean topography grid — one figure, one reference.
        assert (out_dir / "quality_topomaps_Placebo_ASSR.png").exists()
        # One participant figure per component, in both domains.
        for sub in ("quality_participant_topomaps", "quality_participant_tf_maps"):
            written = sorted((out_dir / f"{sub}_Placebo_ASSR").iterdir())
            assert len(written) == N_PCA
        scatters = sorted(
            p.name for p in (out_dir / "quality_scatters_Placebo_ASSR").iterdir()
        )
        # 5 y-axes against the one reference topography (boxcar-PCA, boxcar-40Hz,
        # and the TF score over the whole map + both bands), combined +
        # per-component each.
        assert len(scatters) == 10
        for tag in ("time_pca", "time_40hz", "wavelet_tf", "tf_1_10hz", "tf_30_50hz"):
            assert f"quality_scatter_{tag}_Placebo_ASSR.png" in scatters
            assert f"quality_scatter_{tag}_per_component_Placebo_ASSR.png" in scatters

    def test_leaves_no_open_figures(self, info, iva_inputs, tmp_path) -> None:
        _run(info, iva_inputs, tmp_path)
        assert plt.get_fignums() == []


class TestPanelOrdering:
    """``--quality`` lays the per-component panels out in IVA order, not by rank.

    The quality figures number their panels ``IC <k+1>`` in IVA component order,
    so the standard IVA grids must do the same or a component sits in a different
    cell in each set. The ranking still chooses *which* components are shown.
    """

    @pytest.fixture
    def captured_labels(self, monkeypatch) -> list[list[str]]:
        """Panel labels of every per-component ISC grid, in the order drawn."""
        seen: list[list[str]] = []

        def _record(corr_per_comp, *, labels, fig_title, save_path) -> None:
            seen.append(list(labels))

        monkeypatch.setattr(run_wavelet_iva_channel, "_plot_isc_grid", _record)
        return seen

    @staticmethod
    def _ic_numbers(labels: list[str]) -> list[int]:
        return [int(re.search(r"IC (\d+)", text).group(1)) for text in labels]

    def test_quality_keeps_iva_component_order(
        self, info, iva_inputs, tmp_path, captured_labels
    ) -> None:
        _run(info, iva_inputs, tmp_path, n_top=2, n_bottom=0)
        assert captured_labels, "no per-component grid was drawn"
        for labels in captured_labels:
            ics = self._ic_numbers(labels)
            assert ics == sorted(ics)
            # The rank is reported per panel instead of ordering the panels.
            assert all("rank" in text for text in labels)
            assert not any(text.startswith(("TOP", "BOT")) for text in labels)

    def test_without_quality_panels_follow_the_ranking(
        self, info, iva_inputs, tmp_path, captured_labels
    ) -> None:
        _run(info, iva_inputs, tmp_path, n_top=2, n_bottom=0, quality=False)
        for labels in captured_labels:
            assert labels[0].startswith("TOP 1 (IC")
            # Ranked order: the score never increases down the panels.
            scores = [float(re.search(r"r=([-+][\d.]+)", t).group(1)) for t in labels]
            assert scores == sorted(scores, reverse=True)

    def test_the_selected_components_are_the_same_either_way(
        self, info, iva_inputs, tmp_path, monkeypatch
    ) -> None:
        # Only the panel order changes: --quality must not change which
        # components the ranking selected.
        def _selection(**kwargs) -> set[int]:
            seen: list[list[str]] = []
            monkeypatch.setattr(
                run_wavelet_iva_channel,
                "_plot_isc_grid",
                lambda corr, *, labels, fig_title, save_path: seen.append(list(labels)),
            )
            _run(info, iva_inputs, tmp_path, n_top=2, n_bottom=0, **kwargs)
            return set(self._ic_numbers(seen[0]))

        assert _selection(quality=True) == _selection(quality=False)


class TestRunIvaWithBottomGroup:
    """A positive ``--n_bottom`` still produces the contrast figures."""

    def test_writes_both_groups(self, info, iva_inputs, tmp_path) -> None:
        out_dir = _run(
            info, iva_inputs, tmp_path, n_top=N_PCA - 1, n_bottom=1, quality=False
        )
        names = sorted(p.name for p in out_dir.iterdir())
        assert any(n.endswith("_top.png") for n in names)
        assert any(n.endswith("_bottom.png") for n in names)


class TestRunIvaValidation:
    def test_rejects_n_pca_above_channel_count(
        self, info, iva_inputs, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="must be ≤ n_channels"):
            _run(info, iva_inputs, tmp_path, n_pca=N_CHANNELS + 1, n_top=1)

    def test_rejects_more_shown_components_than_computed(
        self, info, iva_inputs, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="must be ≤ --n_pca"):
            _run(info, iva_inputs, tmp_path, n_top=N_PCA, n_bottom=1)
