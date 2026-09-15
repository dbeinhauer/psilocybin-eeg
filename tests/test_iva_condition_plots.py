"""
Tests for src/visualization/iva_condition_plots.py — Placebo/Psilocybin comparison
figures for a subject-axis-pooled IVA run.

Figure *content* is not asserted; what is asserted is the layout contract the figures
are built on — participant columns ordered by ID and paired across the condition rows,
one row per condition plus the difference row, the files written, and the validation
that stops a mislabelled grid from being drawn.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.visualization.iva_condition_plots import (  # noqa: E402
    DIFFERENCE_ROW,
    MAX_TIME_MARKS,
    participant_grid,
    plot_condition_mean_tf_maps,
    plot_condition_mean_topomaps,
    plot_participant_condition_tf_maps,
    plot_participant_condition_topomaps,
)

N_PAIRS = 3
N_COMPONENTS = 2
N_CHANNELS = 8
N_FREQS = 5
N_TIMES = 20
CONDITIONS = ["Placebo", "Psilocybin"]
# Deliberately out of numeric order, so the column sorting has something to do.
PARTICIPANTS = ["021", "003", "011"]
LABEL = "Joined_ASSR"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def subject_participants():
    return PARTICIPANTS * len(CONDITIONS)


@pytest.fixture
def subject_conditions():
    return [c for c in CONDITIONS for _ in PARTICIPANTS]


@pytest.fixture
def info():
    mne.set_log_level("ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(montage.ch_names[:N_CHANNELS], sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    return info


@pytest.fixture
def patterns():
    return np.random.default_rng(0).standard_normal(
        (N_PAIRS * len(CONDITIONS), N_COMPONENTS, N_CHANNELS)
    )


@pytest.fixture
def sources():
    return np.random.default_rng(1).standard_normal(
        (N_PAIRS * len(CONDITIONS), N_COMPONENTS, N_FREQS, N_TIMES)
    )


@pytest.fixture
def freqs():
    return np.linspace(1.0, 50.0, N_FREQS)


@pytest.fixture
def times():
    return np.arange(N_TIMES) / 250.0


@pytest.fixture
def comp_indices():
    return list(range(N_COMPONENTS))


class TestParticipantGrid:
    def test_columns_are_ordered_by_participant_id(
        self, subject_participants, subject_conditions
    ):
        participants, _ = participant_grid(
            subject_participants, subject_conditions, CONDITIONS
        )
        assert participants == ["003", "011", "021"]

    def test_rows_are_participant_paired(
        self, subject_participants, subject_conditions
    ):
        """Column j must be the same person in every condition row."""
        participants, grid = participant_grid(
            subject_participants, subject_conditions, CONDITIONS
        )
        assert grid.shape == (len(CONDITIONS), len(participants))
        for j, participant in enumerate(participants):
            for i, condition in enumerate(CONDITIONS):
                assert subject_participants[grid[i, j]] == participant
                assert subject_conditions[grid[i, j]] == condition

    def test_row_order_follows_the_conditions_argument(
        self, subject_participants, subject_conditions
    ):
        _, grid = participant_grid(
            subject_participants, subject_conditions, list(reversed(CONDITIONS))
        )
        assert subject_conditions[grid[0, 0]] == "Psilocybin"

    def test_a_missing_recording_is_marked_not_dropped(self):
        """An unmatched participant keeps its column, so the pairing never shifts."""
        participants, grid = participant_grid(
            ["003", "011", "003"], ["Placebo", "Placebo", "Psilocybin"], CONDITIONS
        )
        assert participants == ["003", "011"]
        assert grid[1, participants.index("011")] == -1

    def test_ids_without_a_number_sort_last(self):
        participants, _ = participant_grid(
            ["zed", "003"], ["Placebo", "Placebo"], ["Placebo"]
        )
        assert participants == ["003", "zed"]

    def test_a_duplicate_recording_raises(self):
        with pytest.raises(ValueError, match="more than one"):
            participant_grid(["003", "003"], ["Placebo", "Placebo"], ["Placebo"])

    def test_conditions_outside_the_row_order_are_ignored(self):
        participants, grid = participant_grid(
            ["003", "003"], ["Placebo", "Psilocybin"], ["Placebo"]
        )
        assert grid.shape == (1, 1)
        assert participants == ["003"]


class TestConditionMeanFigures:
    def test_topomaps_lay_out_conditions_by_components(
        self, patterns, subject_participants, subject_conditions, info, comp_indices
    ):
        fig = plot_condition_mean_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            info,
            N_CHANNELS,
            comp_indices,
            label=LABEL,
        )
        assert isinstance(fig, Figure)
        # Two condition rows plus the difference row, one column per component.
        assert len(fig.axes) == (len(CONDITIONS) + 1) * len(comp_indices)

    def test_tf_maps_lay_out_conditions_by_components(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
        )
        assert len(fig.axes) == (len(CONDITIONS) + 1) * len(comp_indices)

    def test_difference_row_can_be_suppressed(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            show_difference=False,
        )
        assert len(fig.axes) == len(CONDITIONS) * len(comp_indices)

    def test_no_difference_row_for_a_single_condition(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        """A difference needs two conditions; one must not fabricate a row."""
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            ["Placebo"],
            freqs,
            times,
            comp_indices,
            label=LABEL,
        )
        assert len(fig.axes) == len(comp_indices)

    def test_saves_when_asked(
        self,
        tmp_path,
        patterns,
        subject_participants,
        subject_conditions,
        info,
        comp_indices,
    ):
        out = tmp_path / "nested" / "means.png"
        plot_condition_mean_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            info,
            N_CHANNELS,
            comp_indices,
            label=LABEL,
            save_path=out,
        )
        assert out.exists() and out.stat().st_size > 5000

    def test_empty_components_draw_nothing(
        self, patterns, subject_participants, subject_conditions, info
    ):
        assert (
            plot_condition_mean_topomaps(
                patterns,
                subject_participants,
                subject_conditions,
                CONDITIONS,
                info,
                N_CHANNELS,
                [],
                label=LABEL,
            )
            is None
        )

    def test_time_and_frequency_marks_are_accepted(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            time_marks=times[::5],
            freq_marks=[40.0],
        )
        assert isinstance(fig, Figure)

    def test_too_many_time_marks_are_dropped(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        """Above the cap the lines wash out the map they were meant to locate."""
        crowded = np.linspace(times[0], times[-1], MAX_TIME_MARKS + 1)
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            time_marks=crowded,
        )
        assert len(fig.axes[0].lines) == 0

    def test_epoch_marks_are_drawn_prominently(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        """Paradigm geometry gets solid black lines, not the faint onset styling."""
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            epoch_marks=[0.0, 0.05],
        )
        lines = fig.axes[0].lines
        assert len(lines) == 2
        assert [line.get_linestyle() for line in lines] == ["--", ":"]
        assert all(line.get_alpha() in (None, 1.0) for line in lines)

    def test_epoch_marks_are_not_capped(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        """Unlike time_marks, epoch geometry is never dropped for being crowded."""
        crowded = list(np.linspace(times[0], times[-1], MAX_TIME_MARKS + 1))
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            epoch_marks=crowded,
        )
        assert len(fig.axes[0].lines) == len(crowded)

    def test_epoch_and_time_marks_coexist(
        self,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        fig = plot_condition_mean_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            time_marks=times[::5],
            epoch_marks=[0.0],
            freq_marks=[40.0],
        )
        # 4 faint onsets + 1 epoch mark + 1 frequency line.
        assert len(fig.axes[0].lines) == len(times[::5]) + 2

    def test_mismatched_tf_axes_raise(
        self, sources, subject_participants, subject_conditions, times, comp_indices
    ):
        with pytest.raises(ValueError, match="do not match freqs/times"):
            plot_condition_mean_tf_maps(
                sources,
                subject_participants,
                subject_conditions,
                CONDITIONS,
                np.linspace(1.0, 50.0, N_FREQS + 1),
                times,
                comp_indices,
                label=LABEL,
            )

    def test_wrong_ndim_raises(
        self, sources, subject_participants, subject_conditions, info, comp_indices
    ):
        with pytest.raises(ValueError, match="must have 3 axes"):
            plot_condition_mean_topomaps(
                sources,
                subject_participants,
                subject_conditions,
                CONDITIONS,
                info,
                N_CHANNELS,
                comp_indices,
                label=LABEL,
            )

    def test_mismatched_bookkeeping_raises(
        self, patterns, subject_conditions, info, comp_indices
    ):
        with pytest.raises(ValueError, match="subject_participants has"):
            plot_condition_mean_topomaps(
                patterns,
                ["003"],
                subject_conditions,
                CONDITIONS,
                info,
                N_CHANNELS,
                comp_indices,
                label=LABEL,
            )

    def test_absent_condition_raises(
        self, patterns, subject_participants, subject_conditions, info, comp_indices
    ):
        with pytest.raises(ValueError, match="have no recordings"):
            plot_condition_mean_topomaps(
                patterns,
                subject_participants,
                subject_conditions,
                ["Placebo", "Joined"],
                info,
                N_CHANNELS,
                comp_indices,
                label=LABEL,
            )


class TestPerParticipantFigures:
    def test_one_topomap_figure_per_component(
        self,
        tmp_path,
        patterns,
        subject_participants,
        subject_conditions,
        info,
        comp_indices,
    ):
        paths = plot_participant_condition_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            info,
            N_CHANNELS,
            comp_indices,
            label=LABEL,
            root_dir=tmp_path,
        )
        assert len(paths) == len(comp_indices)
        assert all(p.exists() and p.stat().st_size > 5000 for p in paths)
        assert [p.name for p in paths] == [
            f"condition_topomap_ic{k + 1:02d}_participants_{LABEL}.png"
            for k in comp_indices
        ]

    def test_one_tf_figure_per_component(
        self,
        tmp_path,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        comp_indices,
    ):
        paths = plot_participant_condition_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            comp_indices,
            label=LABEL,
            root_dir=tmp_path,
        )
        assert [p.name for p in paths] == [
            f"condition_tf_ic{k + 1:02d}_participants_{LABEL}.png" for k in comp_indices
        ]

    def test_prefix_reaches_the_filename(
        self, tmp_path, sources, subject_participants, subject_conditions, freqs, times
    ):
        paths = plot_participant_condition_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            [0],
            label=LABEL,
            root_dir=tmp_path,
            prefix="alpha_",
        )
        assert paths[0].name.startswith("alpha_")

    def test_grid_is_conditions_by_participants_plus_the_mean_column(
        self,
        tmp_path,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        monkeypatch,
    ):
        """Row per condition, column per participant, one trailing mean column."""
        captured = []
        real = plt.subplots

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            captured.append(out[1].shape)
            return out

        monkeypatch.setattr(plt, "subplots", spy)
        plot_participant_condition_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            [0],
            label=LABEL,
            root_dir=tmp_path,
        )
        assert captured == [(len(CONDITIONS), N_PAIRS + 1)]

    def test_mean_column_can_be_suppressed(
        self,
        tmp_path,
        sources,
        subject_participants,
        subject_conditions,
        freqs,
        times,
        monkeypatch,
    ):
        captured = []
        real = plt.subplots

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            captured.append(out[1].shape)
            return out

        monkeypatch.setattr(plt, "subplots", spy)
        plot_participant_condition_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            [0],
            label=LABEL,
            root_dir=tmp_path,
            show_condition_mean=False,
        )
        assert captured == [(len(CONDITIONS), N_PAIRS)]

    def test_empty_components_write_nothing(
        self, tmp_path, patterns, subject_participants, subject_conditions, info
    ):
        assert (
            plot_participant_condition_topomaps(
                patterns,
                subject_participants,
                subject_conditions,
                CONDITIONS,
                info,
                N_CHANNELS,
                [],
                label=LABEL,
                root_dir=tmp_path,
            )
            == []
        )
        assert list(tmp_path.glob("*.png")) == []

    def test_an_unmatched_participant_still_gets_a_column(self, tmp_path, freqs, times):
        """A participant missing one condition must not shift the other columns."""
        sources = np.random.default_rng(2).standard_normal(
            (3, N_COMPONENTS, N_FREQS, N_TIMES)
        )
        paths = plot_participant_condition_tf_maps(
            sources,
            ["003", "011", "003"],
            ["Placebo", "Placebo", "Psilocybin"],
            CONDITIONS,
            freqs,
            times,
            [0],
            label=LABEL,
            root_dir=tmp_path,
        )
        assert len(paths) == 1 and paths[0].exists()

    def test_epoch_marks_reach_the_participant_panels(
        self, tmp_path, sources, subject_participants, subject_conditions, freqs, times
    ):
        """The onset-averaged view of Step 8 goes through this function."""
        paths = plot_participant_condition_tf_maps(
            sources,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            freqs,
            times,
            [0],
            label=LABEL,
            root_dir=tmp_path,
            prefix="onset_",
            epoch_marks=[0.0, 0.05],
            freq_marks=[40.0],
        )
        assert paths[0].name.startswith("onset_") and paths[0].exists()

    def test_the_onset_prefix_does_not_overwrite_the_whole_recording_figures(
        self, tmp_path, sources, subject_participants, subject_conditions, freqs, times
    ):
        common = dict(
            subject_participants=subject_participants,
            subject_conditions=subject_conditions,
            conditions=CONDITIONS,
            freqs=freqs,
            times=times,
            comp_indices=[0],
            label=LABEL,
            root_dir=tmp_path,
        )
        whole = plot_participant_condition_tf_maps(sources, **common)
        onset = plot_participant_condition_tf_maps(sources, prefix="onset_", **common)
        assert whole[0] != onset[0]
        assert len(list(tmp_path.glob("*.png"))) == 2

    def test_alignment_note_is_accepted(
        self, tmp_path, patterns, subject_participants, subject_conditions, info
    ):
        paths = plot_participant_condition_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            info,
            N_CHANNELS,
            [0],
            label=LABEL,
            root_dir=tmp_path,
            alignment_note="TF PC1",
        )
        assert paths[0].exists()


class TestSingleRowWording:
    """A time-axis join shares its topographies, so its grid has one row, not two.

    The suptitle must not then claim a scale is "shared by the condition rows" — that
    would describe a comparison the figure is deliberately not drawing.
    """

    def test_multi_row_grids_mention_the_shared_rows(
        self, patterns, subject_participants, subject_conditions, info, comp_indices
    ):
        fig = plot_condition_mean_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            CONDITIONS,
            info,
            N_CHANNELS,
            comp_indices,
            label=LABEL,
        )
        assert "shared by the condition rows" in fig._suptitle.get_text()

    def test_single_row_grids_do_not(self, patterns, info, comp_indices):
        shared = "Placebo + Psilocybin (shared)"
        # One row means one entry per participant: the time-axis variant estimates a
        # single mixing matrix per participant covering both conditions.
        fig = plot_condition_mean_topomaps(
            patterns[:N_PAIRS],
            PARTICIPANTS,
            [shared] * N_PAIRS,
            [shared],
            info,
            N_CHANNELS,
            comp_indices,
            label=LABEL,
        )
        title = fig._suptitle.get_text()
        assert "shared by the condition rows" not in title
        assert "own symmetric scale" in title

    def test_single_row_tf_grids_do_not(self, sources, freqs, times, comp_indices):
        shared = "both"
        fig = plot_condition_mean_tf_maps(
            sources[:N_PAIRS],
            PARTICIPANTS,
            [shared] * N_PAIRS,
            [shared],
            freqs,
            times,
            comp_indices,
            label=LABEL,
        )
        assert "shared by the condition rows" not in fig._suptitle.get_text()

    def test_single_row_participant_figures_name_the_row(
        self, tmp_path, patterns, info
    ):
        shared = "both (shared)"
        paths = plot_participant_condition_topomaps(
            patterns[:N_PAIRS],
            PARTICIPANTS,
            [shared] * N_PAIRS,
            [shared],
            info,
            N_CHANNELS,
            [0],
            label=LABEL,
            root_dir=tmp_path,
            prefix="shared_",
        )
        assert paths[0].name.startswith("shared_") and paths[0].exists()

    def test_a_participant_twice_in_one_row_is_rejected(
        self, patterns, subject_participants, info, comp_indices
    ):
        """The guard that would catch feeding a pooled 2P array to a one-row grid."""
        shared = "both"
        with pytest.raises(ValueError, match="more than one"):
            plot_condition_mean_topomaps(
                patterns,
                subject_participants,
                [shared] * len(subject_participants),
                [shared],
                info,
                N_CHANNELS,
                comp_indices,
                label=LABEL,
            )


def test_difference_row_name_is_not_a_condition():
    """The row key must not collide with a real condition label."""
    assert DIFFERENCE_ROW not in CONDITIONS
