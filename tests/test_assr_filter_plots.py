"""
Tests for src/visualization/assr_filter_plots.py — ASSR spatial-filter check figures.

Figure content is not asserted; these tests cover the panel layout, the guards, the
file written through ``save_path``, and that both plotters run and return a Figure.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.analysis import assr_trials as at  # noqa: E402
from src.visualization.assr_filter_plots import (  # noqa: E402
    plot_cohort_mean_topomaps,
    plot_masked_filter_topomaps,
)

N_COMPONENTS = 3
N_CHANNELS = 8
N_RECORDINGS = 6


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
def electrode_mask():
    keep = np.zeros(N_CHANNELS, dtype=bool)
    keep[:3] = True
    return keep


@pytest.fixture
def cohort(electrode_mask):
    rng = np.random.default_rng(7)
    patterns = rng.normal(size=(N_RECORDINGS, N_COMPONENTS, N_CHANNELS))
    return at.cohort_mean_pattern(patterns, electrode_mask)


class TestPlotCohortMeanTopomaps:
    def test_returns_a_figure(self, cohort, info, electrode_mask):
        fig = plot_cohort_mean_topomaps(
            cohort.pattern,
            info,
            electrode_mask,
            mask_corr=cohort.mask_corr,
            cosine_to_mean=cohort.cosine_to_mean,
            n_recordings=cohort.n_recordings,
        )
        assert isinstance(fig, Figure)

    def test_draws_one_panel_per_component_plus_the_mask(
        self, cohort, info, electrode_mask
    ):
        # The reference panel is the point of the figure as much as the components are.
        fig = plot_cohort_mean_topomaps(
            cohort.pattern,
            info,
            electrode_mask,
            mask_corr=cohort.mask_corr,
            cosine_to_mean=cohort.cosine_to_mean,
            n_recordings=cohort.n_recordings,
        )
        visible = [ax for ax in fig.axes if ax.axison]
        assert len(visible) == N_COMPONENTS + 1

    def test_comp_indices_selects_a_subset(self, cohort, info, electrode_mask):
        fig = plot_cohort_mean_topomaps(
            cohort.pattern,
            info,
            electrode_mask,
            mask_corr=cohort.mask_corr,
            cosine_to_mean=cohort.cosine_to_mean,
            n_recordings=cohort.n_recordings,
            comp_indices=[0],
        )
        assert len([ax for ax in fig.axes if ax.axison]) == 2

    def test_writes_the_save_path(self, cohort, info, electrode_mask, tmp_path):
        path = tmp_path / "nested" / "mean_filter_topomaps.png"
        plot_cohort_mean_topomaps(
            cohort.pattern,
            info,
            electrode_mask,
            mask_corr=cohort.mask_corr,
            cosine_to_mean=cohort.cosine_to_mean,
            n_recordings=cohort.n_recordings,
            save_path=path,
        )
        assert path.exists() and path.stat().st_size > 0

    def test_rejects_a_one_dimensional_pattern(self, cohort, info, electrode_mask):
        with pytest.raises(ValueError, match="components, channels"):
            plot_cohort_mean_topomaps(
                np.zeros(N_CHANNELS),
                info,
                electrode_mask,
                mask_corr=cohort.mask_corr,
                cosine_to_mean=cohort.cosine_to_mean,
                n_recordings=cohort.n_recordings,
            )

    def test_rejects_mask_mismatch(self, cohort, info):
        with pytest.raises(ValueError, match="electrode_mask"):
            plot_cohort_mean_topomaps(
                cohort.pattern,
                info,
                np.ones(N_CHANNELS + 2, dtype=bool),
                mask_corr=cohort.mask_corr,
                cosine_to_mean=cohort.cosine_to_mean,
                n_recordings=cohort.n_recordings,
            )


class TestPlotMaskedFilterTopomaps:
    @pytest.fixture
    def masked(self, cohort, electrode_mask):
        return at.restrict_filters_to_mask(cohort.spatial_filter, electrode_mask)

    @pytest.fixture
    def share(self, cohort, electrode_mask):
        return at.mask_weight_share(cohort.spatial_filter, electrode_mask)

    def test_returns_a_figure(self, masked, share, info):
        fig = plot_masked_filter_topomaps(
            masked,
            info,
            weight_share=share,
            n_mask_channels=3,
            variant="mean_masked_prestim",
        )
        assert isinstance(fig, Figure)

    def test_draws_one_panel_per_component_and_no_reference(self, masked, share, info):
        # No mask panel here: the restriction IS the mask, drawn into every panel.
        fig = plot_masked_filter_topomaps(
            masked,
            info,
            weight_share=share,
            n_mask_channels=3,
            variant="mean_masked_prestim",
        )
        assert len([ax for ax in fig.axes if ax.axison]) == N_COMPONENTS

    def test_writes_the_save_path(self, masked, share, info, tmp_path):
        path = tmp_path / "mean_masked_filter_topomaps.png"
        plot_masked_filter_topomaps(
            masked,
            info,
            weight_share=share,
            n_mask_channels=3,
            variant="mean_masked_prestim",
            save_path=path,
        )
        assert path.exists() and path.stat().st_size > 0

    def test_rejects_a_one_dimensional_filter(self, share, info):
        with pytest.raises(ValueError, match="components, channels"):
            plot_masked_filter_topomaps(
                np.zeros(N_CHANNELS),
                info,
                weight_share=share,
                n_mask_channels=3,
                variant="mean_masked_prestim",
            )
