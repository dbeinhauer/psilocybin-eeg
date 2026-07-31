"""
Tests for src/visualization/iva_quality_plots.py — channel-IVA quality figures.

Figure content is not asserted; these tests cover the layout maths, the files
written by ``plot_participant_topomaps``, and that every plotter runs and
returns a Figure.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.visualization import iva_quality_plots  # noqa: E402
from src.visualization.iva_quality_plots import (  # noqa: E402
    MAX_PANELS,
    _grid_shape,
    _participant_sort_key,
    axis_limit,
    component_colors,
    panel_indices,
    plot_onset_diagnostic,
    plot_participant_topomaps,
    plot_quality_scatter,
    plot_quality_scatter_per_component,
    plot_reference_topomap,
)

N_SUBJECTS = 3
N_COMPONENTS = 4
N_CHANNELS = 8
WIN = 40


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
    info = mne.create_info(montage.ch_names[:N_CHANNELS], sfreq=100.0, ch_types="eeg")
    info.set_montage(montage)
    return info


@pytest.fixture
def arrays():
    """``(patterns, ref_topo, topo_corr, time_corr, onset_avgs, epoch_times)``."""
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((N_SUBJECTS, N_COMPONENTS, N_CHANNELS)),
        rng.standard_normal(N_CHANNELS),
        rng.uniform(-1, 1, (N_SUBJECTS, N_COMPONENTS)),
        rng.uniform(-1, 1, (N_SUBJECTS, N_COMPONENTS)),
        rng.standard_normal((N_SUBJECTS, N_COMPONENTS, WIN)),
        np.linspace(-0.1, 0.3, WIN),
    )


@pytest.fixture
def subject_ids() -> list[str]:
    return [f"P{i:02d}" for i in range(N_SUBJECTS)]


# ---------------------------------------------------------------------------
# Layout / selection helpers
# ---------------------------------------------------------------------------


class TestGridShape:
    @pytest.mark.parametrize("n_panels", range(1, 40))
    def test_covers_every_panel(self, n_panels: int) -> None:
        nrows, ncols = _grid_shape(n_panels)
        assert nrows * ncols >= n_panels

    def test_never_strands_a_single_panel_on_its_own_row(self) -> None:
        # The regression: 7 panels used to lay out as 6 + 1.
        nrows, ncols = _grid_shape(7)
        assert (nrows, ncols) == (2, 4)

    def test_respects_max_cols(self) -> None:
        _nrows, ncols = _grid_shape(30, max_cols=6)
        assert ncols <= 6

    def test_single_panel(self) -> None:
        assert _grid_shape(1) == (1, 1)


class TestParticipantSortKey:
    def test_orders_psi_ids_numerically(self) -> None:
        ids = ["PSI010", "PSI2", "PSI1"]
        assert sorted(ids, key=_participant_sort_key) == ["PSI1", "PSI2", "PSI010"]

    def test_ids_without_a_number_sort_last(self) -> None:
        ids = ["group mean", "PSI010"]
        assert sorted(ids, key=_participant_sort_key) == ["PSI010", "group mean"]


class TestComponentColors:
    def test_one_color_per_component(self) -> None:
        colors = component_colors(list(range(5)))
        assert set(colors) == set(range(5))

    def test_distinct_for_small_sets(self) -> None:
        colors = component_colors(list(range(10)))
        assert len({tuple(c) for c in colors.values()}) == 10

    def test_handles_more_than_twenty(self) -> None:
        colors = component_colors(list(range(35)))
        assert len(colors) == 35


class TestAxisLimit:
    def test_pads_above_the_largest_magnitude(self) -> None:
        topo = np.array([[0.5, 0.1]])
        time = np.array([[0.2, 0.3]])
        assert axis_limit(topo, time, [0, 1]) > 0.5

    def test_capped_at_one(self) -> None:
        ones = np.ones((2, 2))
        assert axis_limit(ones, ones, [0, 1]) == pytest.approx(1.0)


class TestPanelIndices:
    def test_returns_all_when_under_the_cap(self) -> None:
        idx = list(range(5))
        assert panel_indices(np.zeros((2, 5)), np.zeros((2, 5)), idx) == idx

    def test_caps_and_keeps_iva_order(self) -> None:
        n = MAX_PANELS + 10
        topo = np.zeros((2, n))
        time = np.zeros((2, n))
        # Make the LAST components the best scoring ones.
        topo[:, -MAX_PANELS:] = 1.0
        got = panel_indices(topo, time, list(range(n)))
        assert len(got) == MAX_PANELS
        assert got == sorted(got)
        assert got == list(range(n - MAX_PANELS, n))


# ---------------------------------------------------------------------------
# Per-component participant-comparison figures
# ---------------------------------------------------------------------------


class TestPlotParticipantTopomaps:
    def test_one_flat_figure_per_component(self, info, arrays, subject_ids, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        comp_indices = [0, 2]
        written = plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            comp_indices,
            subject_ids,
            label="LBL",
            root_dir=tmp_path / "tree",
        )
        # One figure per component, written flat — no per-component directories.
        assert len(written) == len(comp_indices)
        assert [p.is_dir() for p in (tmp_path / "tree").iterdir()] == [False, False]
        # 1-based IVA numbering, so components 0 and 2 -> ic01 and ic03.
        assert sorted(p.name for p in written) == [
            "topomap_ic01_all_participants_LBL.png",
            "topomap_ic03_all_participants_LBL.png",
        ]

    def test_files_are_written_and_non_empty(self, info, arrays, subject_ids, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        written = plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [1],
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
        )
        assert len(written) == 1
        assert all(p.exists() and p.stat().st_size > 0 for p in written)
        assert written[0].name == "topomap_ic02_all_participants_LBL.png"

    def test_preserves_comp_indices_order(self, info, arrays, subject_ids, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        written = plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [3, 0, 1],
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
        )
        assert [p.name for p in written] == [
            "topomap_ic04_all_participants_LBL.png",
            "topomap_ic01_all_participants_LBL.png",
            "topomap_ic02_all_participants_LBL.png",
        ]

    def test_panels_are_ordered_by_participant_id(
        self, info, arrays, tmp_path, monkeypatch
    ):
        # Subjects arrive out of ID order; the panels must not follow suit.
        patterns, ref_topo, topo_corr, *_ = arrays
        shuffled = ["PSI010", "PSI002", "PSI001"][:N_SUBJECTS]
        titles: list[str] = []
        monkeypatch.setattr(
            iva_quality_plots,
            "_save_fig",
            lambda fig, path: titles.extend(ax.get_title() for ax in fig.axes),
        )
        plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [0],
            shuffled,
            label="LBL",
            root_dir=tmp_path,
        )
        participant_panels = [t.split()[0] for t in titles[:N_SUBJECTS]]
        assert participant_panels == sorted(shuffled, key=_participant_sort_key)

    def test_panel_labels_keep_each_subject_with_its_own_correlation(
        self, info, arrays, tmp_path, monkeypatch
    ):
        # Reordering the panels must reorder the r annotations with them.
        patterns, ref_topo, topo_corr, *_ = arrays
        shuffled = ["PSI010", "PSI002", "PSI001"][:N_SUBJECTS]
        titles: list[str] = []
        monkeypatch.setattr(
            iva_quality_plots,
            "_save_fig",
            lambda fig, path: titles.extend(ax.get_title() for ax in fig.axes),
        )
        plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [0],
            shuffled,
            label="LBL",
            root_dir=tmp_path,
        )
        expected = {sid: f"{topo_corr[s, 0]:+.2f}" for s, sid in enumerate(shuffled)}
        for title in titles[:N_SUBJECTS]:
            sid, r = title.split()[0], title.split()[1].strip("()")
            assert r == expected[sid]

    def test_prefix_applied(self, info, arrays, subject_ids, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        written = plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [0],
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
            prefix="alpha_",
        )
        assert all(p.name.startswith("alpha_") for p in written)

    def test_closes_all_figures(self, info, arrays, subject_ids, tmp_path):
        # A sweep over every component must not accumulate open figures.
        patterns, ref_topo, topo_corr, *_ = arrays
        plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            list(range(N_COMPONENTS)),
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
        )
        assert plt.get_fignums() == []

    def test_rejects_bad_pattern_ndim(self, info, arrays, subject_ids, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        with pytest.raises(ValueError, match="patterns must be"):
            plot_participant_topomaps(
                patterns[0],
                ref_topo,
                topo_corr,
                info,
                N_CHANNELS,
                [0],
                subject_ids,
                label="LBL",
                root_dir=tmp_path,
            )

    def test_rejects_subject_id_count_mismatch(self, info, arrays, tmp_path):
        patterns, ref_topo, topo_corr, *_ = arrays
        with pytest.raises(ValueError, match="subject_ids has"):
            plot_participant_topomaps(
                patterns,
                ref_topo,
                topo_corr,
                info,
                N_CHANNELS,
                [0],
                ["only-one"],
                label="LBL",
                root_dir=tmp_path,
            )

    def test_rejects_topo_corr_subject_mismatch(
        self, info, arrays, subject_ids, tmp_path
    ):
        patterns, ref_topo, topo_corr, *_ = arrays
        with pytest.raises(ValueError, match="topo_corr has"):
            plot_participant_topomaps(
                patterns,
                ref_topo,
                topo_corr[:-1],
                info,
                N_CHANNELS,
                [0],
                subject_ids,
                label="LBL",
                root_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# The remaining figures
# ---------------------------------------------------------------------------


class TestFigurePlotters:
    def test_reference_topomap(self, info, arrays, tmp_path) -> None:
        _p, ref_topo, *_ = arrays
        out = tmp_path / "ref.png"
        fig = plot_reference_topomap(
            ref_topo, info, N_CHANNELS, label="LBL", save_path=out
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_onset_diagnostic(self, arrays, tmp_path) -> None:
        _p, _r, topo_corr, time_corr, onset_avgs, epoch_times = arrays
        comp_indices = list(range(N_COMPONENTS))
        out = tmp_path / "onset.png"
        fig = plot_onset_diagnostic(
            onset_avgs,
            0.5,
            topo_corr,
            time_corr,
            epoch_times,
            comp_indices,
            component_colors(comp_indices),
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_quality_scatter(self, arrays, tmp_path) -> None:
        _p, _r, topo_corr, time_corr, *_ = arrays
        comp_indices = list(range(N_COMPONENTS))
        out = tmp_path / "scatter.png"
        fig = plot_quality_scatter(
            topo_corr,
            time_corr,
            comp_indices,
            component_colors(comp_indices),
            axis_limit(topo_corr, time_corr, comp_indices),
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_quality_scatter_per_component(self, arrays, subject_ids, tmp_path) -> None:
        _p, _r, topo_corr, time_corr, *_ = arrays
        comp_indices = list(range(N_COMPONENTS))
        out = tmp_path / "per_comp.png"
        fig = plot_quality_scatter_per_component(
            topo_corr,
            time_corr,
            comp_indices,
            component_colors(comp_indices),
            subject_ids,
            axis_limit(topo_corr, time_corr, comp_indices),
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_save_path_none_writes_nothing(self, arrays, tmp_path) -> None:
        _p, _r, topo_corr, time_corr, *_ = arrays
        comp_indices = list(range(N_COMPONENTS))
        plot_quality_scatter(
            topo_corr,
            time_corr,
            comp_indices,
            component_colors(comp_indices),
            1.0,
            variant_name="V",
            label="LBL",
        )
        assert list(tmp_path.iterdir()) == []

    def test_creates_missing_parent_directories(self, info, arrays, tmp_path) -> None:
        _p, ref_topo, *_ = arrays
        out = tmp_path / "a" / "b" / "ref.png"
        plot_reference_topomap(ref_topo, info, N_CHANNELS, label="LBL", save_path=out)
        assert out.exists()
