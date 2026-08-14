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

from src.analysis.iva_quality import equalize_subject_influence  # noqa: E402
from src.visualization import iva_quality_plots  # noqa: E402
from src.visualization.iva_quality_plots import (  # noqa: E402
    MAX_ONSET_MARKS,
    MAX_PANELS,
    _grid_shape,
    _participant_sort_key,
    axis_limit,
    component_colors,
    panel_indices,
    plot_full_tf_maps,
    plot_onset_diagnostic,
    plot_participant_tf_maps,
    plot_participant_topomaps,
    plot_quality_scatter,
    plot_quality_scatter_per_component,
    plot_tf_diagnostic,
    plot_topomap_diagnostic,
    plot_wavelet_reference,
)

N_SUBJECTS = 3
N_COMPONENTS = 4
N_CHANNELS = 8
N_FREQS = 5
WIN = 40
N_FULL_TIMES = 50  # whole-recording time axis, longer than the onset epoch


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
def tf_arrays():
    """``(onset_tf, ref_tf, freqs)`` for the wavelet-family figures."""
    rng = np.random.default_rng(1)
    return (
        rng.standard_normal((N_SUBJECTS, N_COMPONENTS, N_FREQS, WIN)),
        rng.standard_normal((N_FREQS, WIN)),
        np.linspace(2.0, 60.0, N_FREQS),
    )


@pytest.fixture
def full_tf():
    """``(full_tf_mean, freqs, times)`` for the whole-recording QC figure.

    Already a group mean, unlike ``tf_arrays``: the whole-recording sources are
    reduced over subjects before they reach the plotter.
    """
    rng = np.random.default_rng(2)
    return (
        rng.standard_normal((N_COMPONENTS, N_FREQS, N_FULL_TIMES)),
        np.linspace(2.0, 60.0, N_FREQS),
        np.arange(N_FULL_TIMES) / 100.0,
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

    def test_empty_comp_indices_writes_nothing(
        self, info, arrays, subject_ids, tmp_path
    ):
        # Mirrors plot_participant_tf_maps: no component selected is not an
        # error, and the per-participant scale has nothing to be measured from.
        patterns, ref_topo, topo_corr, *_ = arrays
        written = plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [],
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
        )
        assert written == []
        assert list(tmp_path.iterdir()) == []

    def _clims(self, info, patterns, arrays, subject_ids, tmp_path, monkeypatch):
        """Colour limits of every panel of the ``comp_indices=[0]`` figure."""
        _p, ref_topo, topo_corr, *_ = arrays
        figs: list[Figure] = []
        monkeypatch.setattr(
            iva_quality_plots, "_save_fig", lambda fig, path: figs.append(fig)
        )
        plot_participant_topomaps(
            patterns,
            ref_topo,
            topo_corr,
            info,
            N_CHANNELS,
            [0],
            subject_ids,
            label="LBL",
            root_dir=tmp_path,
        )
        return [ax.images[0].get_clim() for ax in figs[0].axes if ax.images]

    def test_a_loud_participant_cannot_take_over_the_figure(
        self, info, arrays, subject_ids, tmp_path, monkeypatch
    ):
        # Every participant is put on a common scale, so scaling one participant's
        # patterns up leaves the figure — panels, shared limit and group mean —
        # exactly as it was. Without the rescaling that participant would set the
        # shared limit and flatten everyone else.
        patterns = arrays[0]
        loud = patterns.copy()
        loud[1] *= 50.0
        base_clims = self._clims(
            info, patterns, arrays, subject_ids, tmp_path, monkeypatch
        )
        loud_clims = self._clims(info, loud, arrays, subject_ids, tmp_path, monkeypatch)
        np.testing.assert_allclose(np.array(loud_clims), np.array(base_clims))

    def test_group_mean_panel_is_drawn_on_its_own_scale(
        self, info, arrays, subject_ids, tmp_path, monkeypatch
    ):
        # The across-participant mean is weaker than the individual patterns, so
        # under their shared limit it reads flat; it gets its own, like the TF
        # counterpart's mean panel.
        patterns = arrays[0]
        clims = self._clims(info, patterns, arrays, subject_ids, tmp_path, monkeypatch)
        subject_clim, mean_clim = clims[0], clims[N_SUBJECTS]
        equalized = equalize_subject_influence(patterns, [0])
        expected = float(np.percentile(np.abs(equalized[:, 0].mean(axis=0)), 99))
        assert mean_clim == pytest.approx((-expected, expected))
        assert mean_clim[1] < subject_clim[1]

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


class TestPlotParticipantTfMaps:
    """The TF counterpart of ``plot_participant_topomaps``, same conventions."""

    def _call(self, arrays, tf_arrays, ids, tmp_path, **kw):
        """Call the plotter with the fixtures, overriding any argument via ``kw``."""
        *_rest, epoch_times = arrays
        onset_tf, ref_tf, freqs = tf_arrays
        return plot_participant_tf_maps(
            kw.pop("onset_tf", onset_tf),
            kw.pop("ref_tf", ref_tf),
            kw.pop("tf_corr", arrays[3]),
            freqs,
            epoch_times,
            kw.pop("comp_indices", list(range(N_COMPONENTS))),
            kw.pop("subject_ids", ids),
            resp_duration_s=0.5,
            assr_freq=40.0,
            label="LBL",
            root_dir=tmp_path,
            **kw,
        )

    def test_one_flat_figure_per_component(
        self, arrays, tf_arrays, subject_ids, tmp_path
    ) -> None:
        written = self._call(arrays, tf_arrays, subject_ids, tmp_path)
        assert len(written) == N_COMPONENTS
        assert [p.name for p in written] == [
            f"tf_ic{k + 1:02d}_all_participants_LBL.png" for k in range(N_COMPONENTS)
        ]
        # Flat directory, exactly one file per component, all non-empty.
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            p.name for p in written
        )
        assert all(p.exists() and p.stat().st_size > 0 for p in written)

    def test_preserves_comp_indices_order(
        self, arrays, tf_arrays, subject_ids, tmp_path
    ) -> None:
        written = self._call(
            arrays, tf_arrays, subject_ids, tmp_path, comp_indices=[2, 0]
        )
        assert [p.name for p in written] == [
            "tf_ic03_all_participants_LBL.png",
            "tf_ic01_all_participants_LBL.png",
        ]

    def test_prefix_applied(self, arrays, tf_arrays, subject_ids, tmp_path) -> None:
        written = self._call(
            arrays, tf_arrays, subject_ids, tmp_path, comp_indices=[0], prefix="alpha_"
        )
        assert written[0].name == "alpha_tf_ic01_all_participants_LBL.png"
        assert written[0].exists()

    def test_closes_all_figures(self, arrays, tf_arrays, subject_ids, tmp_path) -> None:
        self._call(arrays, tf_arrays, subject_ids, tmp_path)
        assert plt.get_fignums() == []

    def test_panels_are_ordered_by_participant_id(
        self, arrays, tf_arrays, tmp_path, monkeypatch
    ) -> None:
        # Deliberately unsorted IDs: the panels must come out numerically ordered,
        # matching plot_participant_topomaps so a participant sits in the same grid
        # position in both directories.
        shuffled = ["PSI010", "PSI002", "PSI001"][:N_SUBJECTS]
        titles: list[str] = []
        monkeypatch.setattr(
            iva_quality_plots,
            "_save_fig",
            lambda fig, path: titles.extend(ax.get_title() for ax in fig.axes),
        )
        self._call(arrays, tf_arrays, shuffled, tmp_path, comp_indices=[0])
        participant_panels = [t.split()[0] for t in titles[:N_SUBJECTS]]
        assert participant_panels == sorted(shuffled, key=_participant_sort_key)

    def test_panel_labels_keep_each_subject_with_its_own_correlation(
        self, arrays, tf_arrays, tmp_path, monkeypatch
    ) -> None:
        # Reordering the panels must carry each subject's r with it, or a map ends
        # up captioned with someone else's score.
        shuffled = ["PSI010", "PSI002", "PSI001"][:N_SUBJECTS]
        tf_corr = arrays[3]
        titles: list[str] = []
        monkeypatch.setattr(
            iva_quality_plots,
            "_save_fig",
            lambda fig, path: titles.extend(ax.get_title() for ax in fig.axes),
        )
        self._call(arrays, tf_arrays, shuffled, tmp_path, comp_indices=[0])
        expected = {shuffled[s]: f"{tf_corr[s, 0]:+.2f}" for s in range(N_SUBJECTS)}
        for title in titles[:N_SUBJECTS]:
            pid, r = title.split()[0], title.split()[1].strip("()")
            assert r == expected[pid]

    def test_group_mean_panel_is_drawn_on_its_own_scale(
        self, arrays, tf_arrays, subject_ids, tmp_path, monkeypatch
    ) -> None:
        # The across-subject mean is much weaker than the individual maps, so
        # under the participants' shared limit it reads flat — it gets the limit
        # of the mean itself (what plot_tf_diagnostic uses) instead. The mean is
        # of the equal-weighted maps, so the expectation equalises too.
        onset_tf, *_ = tf_arrays
        figs: list[Figure] = []
        monkeypatch.setattr(
            iva_quality_plots, "_save_fig", lambda fig, path: figs.append(fig)
        )
        self._call(arrays, tf_arrays, subject_ids, tmp_path, comp_indices=[0])
        mean_clim = figs[0].axes[N_SUBJECTS].images[0].get_clim()
        subject_clim = figs[0].axes[0].images[0].get_clim()
        equalized = equalize_subject_influence(onset_tf, [0])
        expected = float(np.abs(equalized[:, 0].mean(axis=0)).max())
        assert mean_clim == pytest.approx((-expected, expected))
        assert mean_clim[1] < subject_clim[1]

    def test_a_loud_participant_cannot_take_over_the_figure(
        self, arrays, tf_arrays, subject_ids, tmp_path, monkeypatch
    ) -> None:
        # As in the topomap counterpart: participants are put on a common scale,
        # so scaling one up leaves every panel limit — including the group mean's
        # — untouched instead of flattening everyone else.
        onset_tf, ref_tf, freqs = tf_arrays
        loud = onset_tf.copy()
        loud[1] *= 50.0

        def _clims(maps):
            figs: list[Figure] = []
            monkeypatch.setattr(
                iva_quality_plots, "_save_fig", lambda fig, path: figs.append(fig)
            )
            self._call(
                arrays,
                (maps, ref_tf, freqs),
                subject_ids,
                tmp_path,
                comp_indices=[0],
            )
            return np.array(
                [ax.images[0].get_clim() for ax in figs[0].axes if ax.images]
            )

        np.testing.assert_allclose(_clims(loud), _clims(onset_tf))

    def test_group_mean_scale_is_shared_across_components(
        self, arrays, tf_arrays, subject_ids, tmp_path, monkeypatch
    ) -> None:
        # One mean limit for the whole call, so a component's mean panel can be
        # read against another's rather than only against its own participants.
        figs: list[Figure] = []
        monkeypatch.setattr(
            iva_quality_plots, "_save_fig", lambda fig, path: figs.append(fig)
        )
        self._call(arrays, tf_arrays, subject_ids, tmp_path, comp_indices=[0, 1])
        clims = [fig.axes[N_SUBJECTS].images[0].get_clim() for fig in figs]
        assert clims[0] == pytest.approx(clims[1])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"onset_tf": np.zeros((N_SUBJECTS, N_COMPONENTS, WIN))}, "must be"),
            ({"subject_ids": ["only-one"]}, "subject_ids has"),
            ({"ref_tf": np.zeros((N_FREQS, WIN - 1))}, "do not match the reference"),
        ],
    )
    def test_rejects_bad_inputs(
        self, arrays, tf_arrays, subject_ids, tmp_path, kwargs, match
    ) -> None:
        with pytest.raises(ValueError, match=match):
            self._call(
                arrays, tf_arrays, subject_ids, tmp_path, comp_indices=[0], **kwargs
            )

    def test_rejects_tf_corr_subject_mismatch(
        self, arrays, tf_arrays, subject_ids, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="tf_corr has"):
            self._call(
                arrays,
                tf_arrays,
                subject_ids,
                tmp_path,
                comp_indices=[0],
                tf_corr=arrays[3][:-1],
            )


# ---------------------------------------------------------------------------
# The remaining figures
# ---------------------------------------------------------------------------


class TestFigurePlotters:
    def test_wavelet_reference(self, info, arrays, tf_arrays, tmp_path) -> None:
        _p, ref_topo, *_rest, epoch_times = arrays
        _onset_tf, ref_tf, freqs = tf_arrays
        out = tmp_path / "ref_wavelet.png"
        fig = plot_wavelet_reference(
            ref_topo,
            ref_tf,
            info,
            N_CHANNELS,
            freqs,
            epoch_times,
            label="LBL",
            resp_duration_s=0.5,
            assr_freq=40.0,
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_wavelet_reference_with_consistency(
        self, info, arrays, tf_arrays, tmp_path
    ) -> None:
        from src.analysis.pca_polarity import topography_consistency

        _p, ref_topo, *_rest, epoch_times = arrays
        _onset_tf, ref_tf, freqs = tf_arrays
        out = tmp_path / "ref_wavelet_cons.png"
        plot_wavelet_reference(
            ref_topo,
            ref_tf,
            info,
            N_CHANNELS,
            freqs,
            epoch_times,
            label="LBL",
            resp_duration_s=0.5,
            assr_freq=40.0,
            consistency=topography_consistency(
                np.random.default_rng(0).standard_normal((N_SUBJECTS, N_CHANNELS))
            ),
            save_path=out,
        )
        assert out.exists()

    def test_topomap_diagnostic(self, info, arrays, tmp_path) -> None:
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        out = tmp_path / "topomaps.png"
        fig = plot_topomap_diagnostic(
            patterns,
            ref_topo,
            topo_corr,
            time_corr,
            info,
            N_CHANNELS,
            list(range(N_COMPONENTS)),
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_topomap_diagnostic_subset_of_components(
        self, info, arrays, tmp_path
    ) -> None:
        # A partial component list must still leave room for the reference panel.
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        out = tmp_path / "topomaps_subset.png"
        plot_topomap_diagnostic(
            patterns,
            ref_topo,
            topo_corr,
            time_corr,
            info,
            N_CHANNELS,
            [0, 2],
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert out.exists()

    def test_topomap_diagnostic_weights_every_participant_equally(
        self, info, arrays
    ) -> None:
        # The group-mean patterns — and so the shared colour limit read off them —
        # must not change when one participant's patterns are scaled up.
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        loud = patterns.copy()
        loud[1] *= 50.0

        def _clims(maps):
            fig = plot_topomap_diagnostic(
                maps,
                ref_topo,
                topo_corr,
                time_corr,
                info,
                N_CHANNELS,
                list(range(N_COMPONENTS)),
                variant_name="V",
                label="LBL",
            )
            # One image per component panel; the reference panel has its own scale.
            return [
                ax.images[0].get_clim() for ax in fig.axes[:N_COMPONENTS] if ax.images
            ]

        np.testing.assert_allclose(_clims(loud), _clims(patterns))

    def test_topomap_diagnostic_reference_keeps_its_own_scale(
        self, info, arrays
    ) -> None:
        # The reference is a PC1 loading, not a pattern: it must not be drawn
        # under the components' shared limit.
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        fig = plot_topomap_diagnostic(
            patterns,
            ref_topo * 100.0,
            topo_corr,
            time_corr,
            info,
            N_CHANNELS,
            list(range(N_COMPONENTS)),
            variant_name="V",
            label="LBL",
        )
        panels = [ax.images[0].get_clim() for ax in fig.axes if ax.images]
        assert panels[N_COMPONENTS] != panels[0]

    def test_topomap_diagnostic_rejects_empty_component_list(
        self, info, arrays
    ) -> None:
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        with pytest.raises(ValueError, match="at least one component"):
            plot_topomap_diagnostic(
                patterns,
                ref_topo,
                topo_corr,
                time_corr,
                info,
                N_CHANNELS,
                [],
                variant_name="V",
                label="LBL",
            )

    def test_tf_diagnostic(self, arrays, tf_arrays, tmp_path) -> None:
        _p, _r, topo_corr, tf_corr, _onset_avgs, epoch_times = arrays
        onset_tf, ref_tf, freqs = tf_arrays
        comp_indices = list(range(N_COMPONENTS))
        out = tmp_path / "tf.png"
        fig = plot_tf_diagnostic(
            onset_tf,
            ref_tf,
            topo_corr,
            tf_corr,
            freqs,
            epoch_times,
            comp_indices,
            resp_duration_s=0.5,
            assr_freq=40.0,
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()

    def test_tf_diagnostic_subset_of_components(
        self, arrays, tf_arrays, tmp_path
    ) -> None:
        # A partial component list must still leave room for the reference panel.
        _p, _r, topo_corr, tf_corr, _onset_avgs, epoch_times = arrays
        onset_tf, ref_tf, freqs = tf_arrays
        out = tmp_path / "tf_subset.png"
        plot_tf_diagnostic(
            onset_tf,
            ref_tf,
            topo_corr,
            tf_corr,
            freqs,
            epoch_times,
            [0, 2],
            resp_duration_s=0.5,
            assr_freq=40.0,
            label="LBL",
            save_path=out,
        )
        assert out.exists()

    def test_tf_diagnostic_weights_every_participant_equally(
        self, arrays, tf_arrays
    ) -> None:
        # The group-mean maps — and so the shared colour limit read off them —
        # must not change when one participant's maps are scaled up.
        _p, _r, topo_corr, tf_corr, _onset_avgs, epoch_times = arrays
        onset_tf, ref_tf, freqs = tf_arrays
        loud = onset_tf.copy()
        loud[1] *= 50.0

        def _panel_data(maps):
            fig = plot_tf_diagnostic(
                maps,
                ref_tf,
                topo_corr,
                tf_corr,
                freqs,
                epoch_times,
                list(range(N_COMPONENTS)),
                resp_duration_s=0.5,
                assr_freq=40.0,
                label="LBL",
            )
            return np.array(
                [ax.images[0].get_array() for ax in fig.axes[:N_COMPONENTS]]
            )

        np.testing.assert_allclose(_panel_data(loud), _panel_data(onset_tf))

    def test_tf_diagnostic_rejects_empty_component_list(
        self, arrays, tf_arrays
    ) -> None:
        _p, _r, topo_corr, tf_corr, _onset_avgs, epoch_times = arrays
        onset_tf, ref_tf, freqs = tf_arrays
        with pytest.raises(ValueError, match="at least one component"):
            plot_tf_diagnostic(
                onset_tf,
                ref_tf,
                topo_corr,
                tf_corr,
                freqs,
                epoch_times,
                [],
                resp_duration_s=0.5,
                assr_freq=40.0,
                label="LBL",
            )

    def test_full_tf_maps(self, full_tf, tmp_path) -> None:
        full_tf_mean, freqs, times = full_tf
        out = tmp_path / "full_tf.png"
        fig = plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            list(range(N_COMPONENTS)),
            assr_freq=40.0,
            label="LBL",
            save_path=out,
        )
        assert isinstance(fig, Figure)
        assert out.exists()
        # One panel per component, all under one shared colour limit.
        panels = [ax.images[0] for ax in fig.axes if ax.images]
        assert len(panels) == N_COMPONENTS
        assert len({im.get_clim() for im in panels}) == 1

    def test_full_tf_maps_subset_of_components(self, full_tf, tmp_path) -> None:
        full_tf_mean, freqs, times = full_tf
        out = tmp_path / "full_tf_subset.png"
        plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            [0, 2],
            assr_freq=40.0,
            label="LBL",
            save_path=out,
        )
        assert out.exists()

    def test_full_tf_maps_marks_onsets_below_the_cap(self, full_tf) -> None:
        full_tf_mean, freqs, times = full_tf
        onsets = np.linspace(0.0, times[-1], 5)
        fig = plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            [0],
            assr_freq=40.0,
            label="LBL",
            onset_times=onsets,
        )
        # One line per onset plus the ASSR frequency marker.
        assert len(fig.axes[0].lines) == len(onsets) + 1
        assert "5 stimulus onsets marked" in fig._suptitle.get_text()

    def test_full_tf_maps_skips_onsets_above_the_cap(self, full_tf) -> None:
        # Hundreds of markers would cover the map instead of locating anything.
        full_tf_mean, freqs, times = full_tf
        onsets = np.linspace(0.0, times[-1], MAX_ONSET_MARKS + 1)
        fig = plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            [0],
            assr_freq=40.0,
            label="LBL",
            onset_times=onsets,
        )
        assert len(fig.axes[0].lines) == 1  # only the ASSR frequency marker
        assert "too many to mark" in fig._suptitle.get_text()

    def test_full_tf_maps_drops_onsets_past_the_recording(self, full_tf) -> None:
        # A cropped recording keeps its full onset list; a marker beyond the map
        # would only stretch the x-axis away from the data.
        full_tf_mean, freqs, times = full_tf
        onsets = np.array([0.0, times[-1] / 2, times[-1] + 5.0])
        fig = plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            [0],
            assr_freq=40.0,
            label="LBL",
            onset_times=onsets,
        )
        assert len(fig.axes[0].lines) == 3  # two onsets + the frequency marker
        assert "2 stimulus onsets marked" in fig._suptitle.get_text()
        assert fig.axes[0].get_xlim()[1] <= times[-1] + 1e-9

    def test_full_tf_maps_names_the_alignment(self, full_tf) -> None:
        full_tf_mean, freqs, times = full_tf
        fig = plot_full_tf_maps(
            full_tf_mean,
            freqs,
            times,
            [0],
            assr_freq=40.0,
            label="LBL",
            alignment_note="reference topomap correlation",
        )
        assert "reference topomap correlation" in fig._suptitle.get_text()

    @pytest.mark.parametrize(
        "maps, comp_indices, match",
        [
            (np.zeros((N_COMPONENTS, N_FREQS)), [0], r"must be \(K, F, T\)"),
            (np.zeros((N_COMPONENTS, N_FREQS + 1, 50)), [0], "frequency bins"),
            (np.zeros((N_COMPONENTS, N_FREQS, 49)), [0], "times were given"),
            (np.zeros((N_COMPONENTS, N_FREQS, 50)), [], "at least one component"),
        ],
    )
    def test_full_tf_maps_rejects_bad_inputs(
        self, full_tf, maps, comp_indices, match
    ) -> None:
        _m, freqs, times = full_tf
        with pytest.raises(ValueError, match=match):
            plot_full_tf_maps(
                maps,
                freqs,
                times,
                comp_indices,
                assr_freq=40.0,
                label="LBL",
            )

    def test_scatter_axis_labels_are_overridable(self, arrays, tmp_path) -> None:
        _p, _r, topo_corr, time_corr, *_ = arrays
        comp_indices = list(range(N_COMPONENTS))
        default = plot_quality_scatter(
            topo_corr,
            time_corr,
            comp_indices,
            component_colors(comp_indices),
            1.0,
            variant_name="V",
            label="LBL",
        )
        assert default.axes[0].get_xlabel() == iva_quality_plots.TOPO_AXIS_LABEL
        assert iva_quality_plots.TIME_AXIS_LABEL in default.axes[0].get_ylabel()
        custom = plot_quality_scatter(
            topo_corr,
            time_corr,
            comp_indices,
            component_colors(comp_indices),
            1.0,
            variant_name="V",
            label="LBL",
            xlabel=iva_quality_plots.TOPO_AXIS_LABEL,
            ylabel=iva_quality_plots.TF_AXIS_LABEL,
        )
        assert custom.axes[0].get_xlabel() == iva_quality_plots.TOPO_AXIS_LABEL
        assert custom.axes[0].get_ylabel() == iva_quality_plots.TF_AXIS_LABEL

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

    def test_onset_diagnostic_weights_every_participant_equally(self, arrays) -> None:
        # The plotted curve is the mean over participants; scaling one
        # participant's response up must not bend it towards that participant.
        _p, _r, topo_corr, time_corr, onset_avgs, epoch_times = arrays
        comp_indices = list(range(N_COMPONENTS))
        loud = onset_avgs.copy()
        loud[1] *= 50.0

        def _curves(avgs):
            fig = plot_onset_diagnostic(
                avgs,
                0.5,
                topo_corr,
                time_corr,
                epoch_times,
                comp_indices,
                component_colors(comp_indices),
                variant_name="V",
                label="LBL",
            )
            return np.array(
                [ax.lines[0].get_ydata() for ax in fig.axes[: len(comp_indices)]]
            )

        np.testing.assert_allclose(_curves(loud), _curves(onset_avgs))

    def test_onset_diagnostic_rejects_empty_component_list(self, arrays) -> None:
        _p, _r, topo_corr, time_corr, onset_avgs, epoch_times = arrays
        with pytest.raises(ValueError, match="at least one component"):
            plot_onset_diagnostic(
                onset_avgs,
                0.5,
                topo_corr,
                time_corr,
                epoch_times,
                [],
                {},
                variant_name="V",
                label="LBL",
            )

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
        patterns, ref_topo, topo_corr, time_corr, *_ = arrays
        out = tmp_path / "a" / "b" / "ref.png"
        plot_topomap_diagnostic(
            patterns,
            ref_topo,
            topo_corr,
            time_corr,
            info,
            N_CHANNELS,
            [0],
            variant_name="V",
            label="LBL",
            save_path=out,
        )
        assert out.exists()
