"""
Full-recording ASSR SNR / condition-contrast grid for a stored **CorrCA**
decomposition (stage 06b).

The CorrCA companion of ``run_assr_snr_grid.py`` (stage 06): the same trials, the same
three ASSR references, the same three test families and the same figures, read off a
store written by ``run_corrca_condition_comparison.py``,
``run_corrca_condition_tracks.py`` or ``run_corrca_individual_conditions.py`` with
``--store_components``. What changes is where the spatial operators come from:

* **The filter is the fitted one.** CorrCA stores its shared filters
  (``corrca_filters``), so they are read back rather than recovered by inverting each
  recording's pattern, which for CorrCA would give a *different* operator: the stored
  per-recording pattern is the shared component seen through that recording's
  covariance, and its pseudo-inverse is not the filter the model applied.
* **One filter for every recording**, so there is no per-recording polarity flip: the
  one sign per component was anchored at fit time, from the shared pattern against the
  ASSR electrodes, and the stored sources carry it.
* **``ASSR-mask (PCA)`` reads through the CorrCA subspace.** The IVA grid projects the
  electrode average onto each recording's PCA subspace; here the analogue is the one
  ``--n_pca``-dimensional subspace the filters were sought in (the span of the filters,
  orthogonal projector ``Q Qᵀ``). Same label, same role: the like-for-like reference
  for a component that could only see that subspace.

Signal variants, stage 06's two and their complement:

  - ``full`` — the stored sources (the fit's own output on z-scored data), given a
    per-trial pre-stimulus baseline
  - ``roi_masked`` — the shared filter with every weight outside the ASSR electrodes
    zeroed, on the z-scored cache, given the same baseline
  - ``non_roi_masked`` — the same filter with the ASSR electrodes zeroed instead. The
    two masked filters add up to the full one, so together they split a component's
    response into what the ROI carries and what the rest of the head carries. An ROI
    store has no electrode outside the ROI, so it skips this variant.
  - ``swapped_filter`` — the filters of the store's twin on the other power scale
    (raw <-> log, same electrodes; ``CORRCA_SWAPPED_OF``) on THIS store's input, given
    the same baseline: how the other fit's topography reads what this one was handed.
    Component *k* is the twin's *k*, with the twin's own sign. The matched-component
    correlations of the two fits' patterns and filters are logged. Skipped, with a
    warning, when the twin store is missing.
  - ``band_swapped_filter_<band>`` — only with ``--swap_band <band>`` on a broadband
    store, and then the only reading: the filters of the store's band-restricted
    twin (same variant and electrodes, fitted on ``<band>``) on THIS store's broadband
    input, at the 40 Hz selections. For ``1-30hz``, whether a topography chosen below
    30 Hz carries the 40 Hz response. Its TF maps are drawn by
    ``run_corrca_band_swap.py``; here only the statistics, into separate result files
    (``corrca_band_swap_snr__*``), so the store's own grid is never overwritten.

**Another baseline.** Every trial is normalised by its pre-stimulus window: its own
mean subtracted, divided by the SD pooled over trials (``at.baseline_normalise``; the
ROI reference likewise). By default that window is the whole pre-onset span, ``t < 0``.
``--baseline=LO:HI`` (seconds, e.g. ``--baseline=-0.3:-0.05``) normalises by that window
instead. The trials are not stored, only their medians, so this re-reads the cache.
Everything it draws goes to ``baseline_<lo>_<hi>ms/`` inside the fit's directory, laid
out like the fit's directory itself:

* ``snr_grid/<selection>/<window>/`` — the grid figures, same names as the default's;
* the **stimulus-averaged TF maps** under that baseline, with the fit's plotting calls
  and file names: ``condition_mean_tf_maps_onset.png`` of the stored sources (every
  component) and ``participants/onset_*``. With ``--masked_tf_maps`` (the cache read at
  every store frequency, several times slower) also ``roi_masked_*``,
  ``non_roi_masked_*`` (every component), ``swapped_filter_*`` and, with
  ``--swap_band``, ``band_swapped_filter_<band>_*`` (the leading
  ``--swapped_components``). A band-swap run draws only its own reading; the stored
  sources' maps are the store's own run's.

Its results go to ``assr_snr_grid_baseline_<lo>_<hi>ms__*``
(``corrca_band_swap_snr_baseline_..__*`` with ``--swap_band``), so the default grid is
never overwritten; no ``tf_baseline`` file is written, the maps are drawn instead.

All carry the same reference rows (the masks on the raw cache, the full mask swapped
for the equal-weight ROI row), so only the learned rows move. Tests (participants as
the unit, exact Wilcoxon, uncorrected): ``contrast`` (Placebo − Psilocybin per source),
``discrimination`` (does a component separate the conditions better than a reference)
and ``snr`` (is a component's response weaker than a reference's, per condition).

**Every store of the run.** The full store first, then — in the order the fit scripts
made them — its log-power twin (``_log``), the fits on the ASSR electrodes alone
(``_roi``) and on every other electrode (``_non_roi``), and their log-power twins
(``_log_roi``, ``_log_non_roi``). A missing one is skipped with a warning;
``--skip_log_power`` / ``--skip_roi`` / ``--skip_non_roi`` skip them outright.

An ROI store's cache is read on its electrodes only, and ``roi_masked`` coincides with
``full`` by construction — every weight is already inside the ROI — which is a free
check that the store and the cache agree. A non-ROI store holds no ASSR electrode, so
its cache is read on the full montage with the filters zero on the ASSR electrodes:
only its ``full`` rows are tested, and only against ``ASSR-mask (full)`` — a
reference read through its subspace would be identically zero.

**Log-power stores.** A ``*_log*`` store was fitted on ``log10`` power, z-scored. Its
masked rows read the cache the same way (log, then z-score), and it gets a fourth
reference, ``ASSR-mask (log, PCA, z)``: the PCA mask on that same signal, the
like-for-like reference for what the decomposition was handed. The other three
references stay as for every store, so the two variants share them.

**Individual conditions.** ``--variant corrca_channel_individual_conditions`` reads
the per-condition stores, one pass per ``--conditions`` entry, each on its own
condition alone. Their filters differ between the conditions, so there is nothing to
contrast: only the ``snr`` family is tested and the p-value summary is not drawn.

**Masked TF maps.** With ``--masked_tf_maps`` the masked variants' full
trial-normalised TF maps are stored too, like the stored sources' ``tf_baseline``: the
cache is then read at every frequency of the store, not only at the 40 Hz selections,
which costs several times the read time.

**Figures.** The statistical ones, per grid cell. The TF maps and topographies are
the fit's, drawn in its own directory; the trial-normalised maps computed here are
stored, not drawn. The one exception is the ``swapped_filter`` reading, which no fit
can draw (its twin may still be fitting in parallel): with ``--masked_tf_maps`` its
TF maps are drawn here, last, into the fit's own directory beside the fit's
``condition_mean_tf_maps``, with the same plotting calls and stimulus average, for
the leading ``--swapped_components``. ``swapped_filter_topomap_correlation.csv``
goes beside them.

Outputs::

    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__values.npz
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__courses.npz
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__tf_baseline.npz
    results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__robustness.csv
        # + __robustness_summary.csv: the contrast with participants left out, on the
        # same filters and on the fit's leave-subset-out refits (<store>__subset_refits.npz)
    results/<experiment>/corrca_band_swap_snr__<variant>__<band>__pca<N>.csv
        # --swap_band <band>: + __courses.npz, __values.npz; no tf_baseline
        # tf_baseline__<Condition>; with --masked_tf_maps also
        # tf_baseline__<signal variant>__<Condition> for the masked variants
        # an individual condition's files carry it: <variant>__<Condition>__<spectrum>
    plots/06b-corrca-condition-comparison/<Join>_<Music>/<spectrum>/
        <variant>/pca_<N>/snr_grid/   # inside the fit's own directory; <variant>_roi/,
                                      # <variant>_non_roi/ for those stores;
                                      # IndividualConditions_<Music>/<spectrum>/
                                      # <variant>/<Condition>/pca_<N>/snr_grid/
            <selection>/<window>/
                pvalue_summary__<variant>.png
                snr_by_condition__<variant>.png
                trial_course_by_source__<variant>.png
                trial_course_by_participant__<variant>.png   # CC 1, CC 2, (PCA, z)
                robustness__full.png   # the contrast's dependence on the participants
            snr_by_condition__band_swapped_filter_<band>.png   # --swap_band, and its
                                      # pvalue_summary__ / trial_course_by_source__
        swapped_filter_condition_mean_tf_maps.png   # in the fit's directory
        swapped_filter_condition_mean_tf_maps_onset.png
        swapped_filter_topomap_correlation.csv
        participants/swapped_filter_*.png

Examples::

    python scripts/run_corrca_snr_grid.py --experiment assr \\
        --variant corrca_channel_joined_separate --n_pca 10 \\
        --halfwidths 0 5 --stimulus_intervals 0:0.5 0.2:0.5

    python scripts/run_corrca_snr_grid.py --variant corrca_channel_joined_tracks \\
        --n_pca 10 --n_times 3000 --skip_plots

    python scripts/run_corrca_snr_grid.py --variant corrca_channel_joined_separate \\
        --n_pca 5 --swap_band 1-30hz

    python scripts/run_corrca_snr_grid.py --variant corrca_channel_joined_separate \\
        --n_pca 5 --baseline=-0.3:-0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    CORRCA_ALIGNMENT_NOTE,
    CORRCA_ALIGNMENT_NOTE_NONE,
    CORRCA_INDIVIDUAL_VARIANTS,
    CORRCA_LOG_OF,
    CORRCA_LOG_VARIANTS,
    CORRCA_SWAPPED_OF,
    CORRCA_SWAPPED_READING,
    STABILITY_COMPONENTS,
    corrca_plots_dir,
    corrca_refit_path,
    corrca_run_plan,
    read_corrca_filters,
)

# The cache/onset conventions and the test bookkeeping are stage 06's, so both stages
# read the same trials and write the same CSV schema.
from scripts.run_assr_snr_grid import (  # noqa: E402
    _accumulate_tests,
    _cache_frequency_grid,
    _load_onsets,
    _stimulus_interval,
)

# The swapped filters' stimulus average is the fit's own.
from scripts.run_iva_condition_tracks import _onset_average_per_condition  # noqa: E402
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.analysis.corrca import matched_component_correlation  # noqa: E402
from src.analysis.assr_variants import (  # noqa: E402
    project_condition,
    replicate_snr_values,
)
from src.analysis.iva_condition_comparison import (  # noqa: E402
    stack_conditions_on_subject_axis,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.io.iva_store import iva_results_path, load_iva_components  # noqa: E402
from src.io.loading import (  # noqa: E402
    assr_electrode_mask,
    concatenated_dir as _concatenated_dir,
)
from src.io.saving import (  # noqa: E402
    save_npz_atomic,
    save_snr_courses,
    save_snr_values,
)
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.assr_snr_plots import (  # noqa: E402
    plot_cell,
    plot_contrast_robustness,
    plot_courses,
    plot_participant_courses,
    plot_snr,
)
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_tf_maps,
    plot_participant_condition_tf_maps,
)

_logger = logging.getLogger(__name__)

#: The grid's own subdirectory of the fit's plot directory, so its figures never sit
#: among the fit's.
_PLOTS_SUBDIR = "snr_grid"
#: The fit scripts' ``--min_onsets`` default and their 40 Hz frequency mark, so the
#: swapped_filter TF maps are drawn exactly as the fit's own.
_MIN_ONSETS = 5
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]
_MASK_LABELS = at.MASK_LABELS
_FULL_LABEL = at.FULL_MASK_LABEL
_SIGNAL_VARIANTS = ("full", "roi_masked", "non_roi_masked", CORRCA_SWAPPED_READING)
#: The variants projected here from the cache rather than read from the store: signal
#: variant -> the operator in the ops dict and the projection in ProjectedVariants at
#: the 40 Hz selections and at every store frequency.
_MASKED = {
    "roi_masked": ("masked", "masked_z", "masked_tf_z"),
    "non_roi_masked": ("outside", "outside_z", "outside_tf_z"),
    CORRCA_SWAPPED_READING: ("swapped", "swapped_z", "swapped_tf_z"),
}
_LOG_LABEL = at.LOG_PCA_Z_MASK_LABEL
#: Components drawn per participant beside the ROI reference.
_PARTICIPANT_COURSE_COMPONENTS = 2
#: Every reference row project_condition can give, in the order its blocks stack.
_REFERENCE_ORDER = (*_MASK_LABELS, _LOG_LABEL)
#: The band-swapped reading's label (``run_corrca_band_swap.py``'s); its signal
#: variant carries the band: ``band_swapped_filter_<band>``.
BAND_SWAPPED_READING = "band_swapped_filter"
_STORE_CONDITIONS = {
    IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE: ConditionVariants.JOINED_SEPARATE,
    IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS: ConditionVariants.JOINED_TRACKS,
    # Fitted on Placebo alone, but every recording of the join is stored.
    IvaVariants.CORRCA_CHANNEL_PLACEBO_FIT: ConditionVariants.JOINED_SEPARATE,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-recording ASSR SNR / contrast grid for a stored CorrCA run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stored = parser.add_argument_group("the stored decomposition to read")
    stored.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    stored.add_argument(
        "--variant",
        default=IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE.value,
        choices=[v.value for v in CORRCA_LOG_OF],
        help="Which stage-06b run to read: subject axis (separate), time axis, "
        "each condition alone, or the subject axis with the filters fitted on "
        "Placebo alone (placebo_fit). Its full store is read first, then its '_log', "
        "'_roi', '_non_roi', '_log_roi' and '_log_non_roi' ones — the fit order.",
    )
    stored.add_argument(
        "--skip_log_power",
        action="store_true",
        help="Do not read the three log-power ('*_log') stores.",
    )
    stored.add_argument(
        "--skip_roi",
        action="store_true",
        help="Do not read the '_roi' store.",
    )
    stored.add_argument(
        "--skip_non_roi",
        action="store_true",
        help="Do not read the '_non_roi' store.",
    )
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument("--band", default=None)
    stored.add_argument(
        "--swap_band",
        default=None,
        help="Read each broadband store through its twin fitted on this band (e.g. "
        "1-30hz or 30-50hz) instead: the band_swapped_filter_<band> rows only, into "
        "corrca_band_swap_snr__* result files. Needs --band unset.",
    )
    stored.add_argument("--n_pca", type=int, default=10)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument("--wavelet_data_dir", type=Path, default=None)
    data.add_argument(
        "--concat_dir",
        type=Path,
        default=None,
        help="Processed-data root holding the concatenated arrays' onset sidecars.",
    )
    data.add_argument(
        "--coordinate_system", default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value
    )
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Two conditions, contrast order; for the individual-condition stores, "
        "the conditions to read, each on its own. Default Placebo Psilocybin.",
    )
    data.add_argument(
        "--mask_sum",
        action="store_true",
        help="Sum the ASSR electrodes instead of averaging them.",
    )
    data.add_argument(
        "--lenient_mask",
        action="store_true",
        help="Accept the electrode intersection when some are missing.",
    )
    data.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Trim the time axis (a quick shape check; not for real runs).",
    )

    analysis = parser.add_argument_group("the extraction, the grid and the tests")
    analysis.add_argument("--center_freq", type=float, default=iva_quality.ASSR_FREQ)
    analysis.add_argument(
        "--halfwidths",
        type=float,
        nargs="+",
        default=[0.0, iva_quality.TF_ANCHOR_HALFWIDTH_HZ],
        help="One frequency selection per half-width in Hz (0 = single 40 Hz bin).",
    )
    analysis.add_argument(
        "--stimulus_intervals",
        type=_stimulus_interval,
        nargs="+",
        default=[
            (0.0, AssrEpoch.STIMULUS_DURATION_S),
            (0.2, AssrEpoch.STIMULUS_DURATION_S),
        ],
        help="One test window per 'lo:hi' seconds, e.g. 0:0.5 0.2:0.5.",
    )
    analysis.add_argument(
        "--baseline",
        type=_baseline_interval,
        default=None,
        help="Normalise every trial by this pre-stimulus window, 'lo:hi' seconds, "
        "instead of the whole pre-onset span; write as --baseline=-0.3:-0.05 (a "
        "leading minus would read as an option). Into snr_grid_baseline_<lo>_<hi>ms/ "
        "and assr_snr_grid_baseline_<lo>_<hi>ms__* files.",
    )
    analysis.add_argument(
        "--signal_variants",
        nargs="+",
        default=list(_SIGNAL_VARIANTS),
        choices=list(_SIGNAL_VARIANTS),
        help="Which readings of the learned rows to test side by side. "
        "'swapped_filter' reads the raw <-> log twin store's filters on this store's "
        "input.",
    )
    analysis.add_argument(
        "--masked_tf_maps",
        action="store_true",
        help="Also store and draw full trial-normalised TF maps of the masked "
        "variants. Reads the cache at every store frequency.",
    )
    analysis.add_argument(
        "--contrast_alternative",
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="Direction of Placebo - Psilocybin. 'greater' = psilocybin lowers it.",
    )
    analysis.add_argument(
        "--snr_alternative",
        default="less",
        choices=["less", "greater", "two-sided"],
        help="Direction of (component SNR - reference SNR) in the SNR family.",
    )
    analysis.add_argument("--alpha", type=float, default=0.05)
    analysis.add_argument(
        "--drop_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Robustness of the contrast: rerun it with every subset of this many "
        "participants left out, on the same filters. The refit replicates are the "
        "fit's own (--refit_subset_sizes of the fit scripts).",
    )
    analysis.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip the robustness of the contrast to the participants (sign test, "
        "leave-subset-out on fixed and on refitted filters, held-out values).",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--save_dir", type=Path, default=None)
    output.add_argument("--results_dir", type=Path, default=None)
    output.add_argument("--skip_plots", action="store_true")
    output.add_argument(
        "--swapped_components",
        type=int,
        default=2,
        help="Draw the swapped_filter TF maps of this many leading components, into "
        "the fit's own directory beside its condition_mean_tf_maps (needs "
        "--masked_tf_maps). 0 draws none; the SNR rows always cover every component.",
    )
    output.add_argument("--log_level", default="INFO")
    return parser


def _baseline_interval(text: str) -> tuple[float, float]:
    """Parse ``"lo:hi"`` seconds into a pre-stimulus window, ``lo < hi <= 0``."""
    lo, hi = _stimulus_interval(text)
    if hi > 0:
        raise argparse.ArgumentTypeError(
            f"the baseline must end at or before the onset (hi <= 0); got {text!r}."
        )
    return lo, hi


def baseline_label(interval: tuple[float, float] | None) -> str:
    """Directory / file token of a baseline window; ``""`` for the default one."""
    if interval is None:
        return ""
    lo, hi = (round(edge * 1000) for edge in interval)
    return f"baseline_{lo}_{hi}ms"


def baseline_window(
    times: np.ndarray, interval: tuple[float, float] | None
) -> np.ndarray:
    """Epoch samples of a baseline window; ``None`` is the whole pre-onset span.

    :raises ValueError: If the window starts before the epoch or holds < 2 samples.
    """
    if interval is None:
        return times < 0.0
    lo, hi = interval
    # The epoch's first sample is -round(0.3 * sfreq) / sfreq, which can sit just after
    # -0.3 s: a start within one sample of it is the epoch start. The edges themselves
    # get only float slack, so the window never grows by a sample.
    sample = float(np.min(np.diff(times))) if times.size > 1 else 0.0
    if lo < times[0] - sample:
        raise ValueError(
            f"baseline {lo:g}-{hi:g} s starts before the epoch ({times[0]:g} s)."
        )
    tol = 1e-6
    mask = (times >= lo - tol) & (times <= hi + tol) & (times < 0.0)
    if int(mask.sum()) < 2:
        raise ValueError(
            f"baseline {lo:g}-{hi:g} s holds {int(mask.sum())} epoch sample(s); "
            "at least 2 are needed."
        )
    return mask


def corrca_operators(
    shared_filters: np.ndarray, binary: np.ndarray, electrode_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """The spatial operators of a CorrCA store, in the shapes the stage-06 grid uses.

    :param shared_filters: ``(channels, components)`` stored ``corrca_filters``.
    :param binary: ``(channels,)`` electrode-average weights.
    :param electrode_mask: Boolean ``(channels,)`` anchor mask.
    :return: ``filters`` ``(K, C)``, ``projector`` ``(C, C)`` — the orthogonal projector
        onto the filters' span — ``pca_row`` ``(C,)`` (the mask through that subspace),
        ``masked`` ``(K, C)`` and ``masked_share`` ``(K,)``, and their complement
        ``outside`` / ``outside_share`` (``masked + outside == filters``; all zero when
        the mask covers every channel).
    """
    basis, _ = np.linalg.qr(np.asarray(shared_filters, dtype=float))
    projector = basis @ basis.T
    filters = np.asarray(shared_filters, dtype=float).T
    outside = filters * ~np.asarray(electrode_mask, dtype=bool)
    total = np.linalg.norm(filters, axis=-1)
    return {
        "filters": filters,
        "projector": projector,
        "pca_row": projector @ binary,
        "masked": at.restrict_filters_to_mask(filters, electrode_mask),
        "masked_share": at.mask_weight_share(filters, electrode_mask),
        "outside": outside,
        "outside_share": np.divide(
            np.linalg.norm(outside, axis=-1),
            total,
            out=np.zeros_like(total),
            where=total > 0,
        ),
    }


def _embed(
    filters: np.ndarray, store_channels: list[str], read_channels: list[str]
) -> np.ndarray:
    """``(K, store channels)`` filters on *read_channels*, zero on every other one."""
    position = {name: i for i, name in enumerate(read_channels)}
    embedded = np.zeros((filters.shape[0], len(read_channels)))
    embedded[:, [position[name] for name in store_channels]] = filters
    return embedded


def _swapped_filters(
    args: argparse.Namespace,
    variant: IvaVariants,
    store_condition: ConditionVariants,
    music_type: MusicTypeVariants,
    results,
    store_channels: list[str],
    *,
    swap_band: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """The filters of the store's twin on the other power scale, or ``None``.

    Read on this store's input (its log or not, its z-scoring), so its rows ask how
    the twin's topography reads what this decomposition was handed. Component *k* is
    the twin's component *k*, with the twin's own sign anchor. Runs after every fit
    of the job is stored, so the twin store is complete.

    :param swap_band: Take the twin fitted on this band instead: the same variant (same
        power scale and electrodes), band-restricted.

    :return: ``(filters, correlations)`` — the twin's ``(K, C)`` filters and one row per
        component with ``pattern_r`` (Pearson r over channels of the two shared forward
        patterns: the topomaps) and ``filter_r`` (of the two filters).
    """
    twin = variant if swap_band is not None else CORRCA_SWAPPED_OF[variant]
    reading = (
        CORRCA_SWAPPED_READING
        if swap_band is None
        else f"{BAND_SWAPPED_READING}_{swap_band}"
    )
    path = iva_results_path(
        ExperimentNames(args.experiment),
        store_condition,
        twin,
        music_type,
        args.band if swap_band is None else swap_band,
        args.n_pca,
        processed_data_dir=args.store_dir,
    )
    if not path.is_file():
        _logger.warning(f"no {twin.value} store at {path}: no {reading} rows.")
        return None
    channels, filters, patterns = read_corrca_filters(path)
    own_filters = np.asarray(results.extras["corrca_filters"], dtype=float).T
    if channels != store_channels or filters.shape != own_filters.shape:
        _logger.warning(
            f"{path.name} does not match this store's channels and component count: "
            f"no {reading} rows."
        )
        return None
    pattern_r = matched_component_correlation(
        np.asarray(results.extras["corrca_shared_patterns"], dtype=float), patterns
    )
    filter_r = matched_component_correlation(own_filters, filters)
    _logger.info(
        f"{reading} rows: {path.name}'s filters on this store's input; "
        "matched-component r "
        + "; ".join(
            f"{COMPONENT_PREFIX} {k + 1}: pattern {pattern_r[k]:+.3f}, "
            f"filter {filter_r[k]:+.3f}"
            for k in range(pattern_r.size)
        )
    )
    table = pd.DataFrame(
        {
            "component": np.arange(1, pattern_r.size + 1),
            "variant": variant.value,
            "twin_variant": twin.value,
            "pattern_r": pattern_r,
            "filter_r": filter_r,
        }
    )
    return filters, table


def run(args: argparse.Namespace) -> None:
    """Run the grid on every store of the run, in the order the fits were made.

    The full montage, its log-power twin, then the ROI and non-ROI fits and their
    log-power twins. Only the full store must exist. An individual-condition run has
    one set of stores per condition, each read alone.
    """
    variant = IvaVariants(args.variant)
    if args.swap_band is not None and args.band is not None:
        raise ValueError(
            "--swap_band reads a broadband store through its band twin's filters; "
            f"leave --band unset (got --band {args.band})."
        )
    plan = corrca_run_plan(
        variant,
        argparse.Namespace(
            skip_log_power=args.skip_log_power,
            skip_roi_fit=args.skip_roi,
            skip_non_roi_fit=args.skip_non_roi,
        ),
    )
    if variant in CORRCA_INDIVIDUAL_VARIANTS:
        stores = [
            (ConditionVariants(c), [c])
            for c in at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))
        ]
    else:
        stores = [(_STORE_CONDITIONS[variant], args.conditions)]
    for store_condition, conditions in stores:
        for fit, _channels, _log_power in plan:
            # A band-swap reading of a fit without a band twin is skipped like a
            # missing store, so it is optional for every fit.
            run_store(
                args,
                fit,
                store_condition,
                conditions,
                optional=fit != variant or args.swap_band is not None,
            )


def run_store(
    args: argparse.Namespace,
    variant: IvaVariants,
    store_condition: ConditionVariants,
    conditions: list[str] | None,
    *,
    optional: bool = False,
) -> None:
    """Run the full grid for one stored CorrCA decomposition.

    :param store_condition: The store's condition: the join, or the one real
        condition of an individual-condition fit.
    :param conditions: Conditions to read, in contrast order; ``None`` is Placebo
        then Psilocybin. A single condition tests only the ``snr`` family.
    :param optional: Skip, with a warning, when the store does not exist.
    """
    experiment = ExperimentNames(args.experiment)
    per_recording = store_condition == ConditionVariants.JOINED_SEPARATE
    music_type = MusicTypeVariants(args.music_type)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    conditions = at.resolve_conditions(conditions or list(REAL_CONDITIONS))

    # ---- 1. store and operators -------------------------------------------
    try:
        results = load_iva_components(
            experiment=experiment,
            condition=store_condition,
            variant=variant,
            music_type=music_type,
            band=args.band,
            n_pca=args.n_pca,
            processed_data_dir=args.store_dir,
        )
    except FileNotFoundError as exc:
        if not optional:
            raise
        _logger.warning(f"no {variant.value} store to read ({exc}); skipped.")
        return
    _logger.info(f"loaded {results.path}")
    if "corrca_filters" not in results.extras:
        raise ValueError(
            f"{results.path} holds no 'corrca_filters'; it was not written by a "
            "stage-06b CorrCA run with --store_components."
        )
    store_channels = list(results.channel_names)
    shared_filters = np.asarray(results.extras["corrca_filters"], dtype=float)
    # A fit cut to the non-ROI electrodes holds no ASSR electrode, so the cache is read
    # on the full montage instead — the references then read the same electrodes as
    # every other store's — and the stored filters are embedded with zeros on the ASSR
    # electrodes. z-scoring is per channel, so the embedded filters read exactly what
    # the fit saw. A reference read through the component subspace is identically zero
    # there, so only the full-montage reference is kept.
    no_roi = not assr_electrode_mask(
        store_channels, coordinate_system, strict=False
    ).any()
    read_channels = store_channels
    if no_roi:
        read_channels = list(
            find_wavelet_set(
                wavelet_label_dir(
                    args.wavelet_data_dir,
                    experiment,
                    f"{conditions[0]}_{music_type.value}",
                )
            ).channel_names
        )
        shared_filters = _embed(shared_filters.T, store_channels, read_channels).T
    electrode_mask = assr_electrode_mask(
        read_channels, coordinate_system, strict=not args.lenient_mask
    )
    binary = at.binary_filter_weights(electrode_mask, normalize=not args.mask_sum)
    ops = corrca_operators(shared_filters, binary, electrode_mask)
    n_components = ops["filters"].shape[0]
    n_rows = results.n_subjects
    n_channels = len(read_channels)
    comp_labels = [f"{COMPONENT_PREFIX} {k + 1}" for k in range(n_components)]
    # A log-power fit is also read against the PCA mask on log power, then z-scored —
    # what it was handed.
    log_power = variant in CORRCA_LOG_VARIANTS
    if no_roi:
        mask_labels = (_FULL_LABEL,)
    elif log_power:
        mask_labels = (*_MASK_LABELS, _LOG_LABEL)
    else:
        mask_labels = _MASK_LABELS
    # Rows of the (binary raw, PCA raw, PCA z[, PCA log z]) reference block kept.
    reference_rows = [_REFERENCE_ORDER.index(m) for m in mask_labels]
    labels = comp_labels + list(mask_labels)
    full_mask_column = labels.index(_FULL_LABEL)
    anchor = str(results.extras.get("polarity_anchor", np.asarray("none")))
    _logger.info(
        f"variant {variant.value}: one shared filter, {n_components} component(s), "
        f"{int(electrode_mask.sum())} ASSR electrode(s) read"
        + (" beside the store's non-ROI ones" if no_roi else "")
        + f"; sign anchor '{anchor}'"
    )
    signals = list(args.signal_variants)
    swapped_r: pd.DataFrame | None = None
    # Signal variant -> (operator, 40 Hz projection, TF projection), with the band-swap
    # reading on the swapped operator.
    masked_of = dict(_MASKED)
    band_signal = None
    if args.swap_band is not None:
        band_signal = f"{BAND_SWAPPED_READING}_{args.swap_band}"
        masked_of[band_signal] = _MASKED[CORRCA_SWAPPED_READING]
        swapped = _swapped_filters(
            args,
            variant,
            store_condition,
            music_type,
            results,
            store_channels,
            swap_band=args.swap_band,
        )
        if swapped is None:
            return
        swapped = swapped[0]
        if no_roi:
            swapped = _embed(swapped, store_channels, read_channels)
        ops["swapped"] = swapped
        signals = [band_signal]
    elif CORRCA_SWAPPED_READING in signals:
        swapped = _swapped_filters(
            args, variant, store_condition, music_type, results, store_channels
        )
        if swapped is None:
            signals.remove(CORRCA_SWAPPED_READING)
        else:
            swapped, swapped_r = swapped
            if no_roi:
                swapped = _embed(swapped, store_channels, read_channels)
            ops["swapped"] = swapped
    if no_roi:
        # Its filters have no ASSR weight: roi_masked is zero and non_roi_masked = full.
        _logger.info(
            "a non-ROI store: only the full (and swapped_filter) rows, against the "
            "full reference."
        )
        signals = [s for s in signals if s == "full" or masked_of[s][0] == "swapped"]
    else:
        for k in range(n_components):
            _logger.info(
                f"  {comp_labels[k]}: |w| kept by the ROI restriction "
                f"{ops['masked_share'][k]:.1%}"
            )
    if "non_roi_masked" in signals and electrode_mask.all():
        _logger.info(
            "every store channel is an ASSR electrode: no non_roi_masked rows."
        )
        signals.remove("non_roi_masked")
    masked_signals = [s for s in signals if s in masked_of]

    def resolve(participant: str, condition: str) -> int:
        if per_recording:
            return results.row(participant, condition)
        return results.row(participant)

    # ---- 2. onsets, caches, paired cohort ---------------------------------
    concat_dir = _concatenated_dir(experiment, args.concat_dir)
    onsets, wavelet_sets, cached = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        wavelet_sets[condition] = find_wavelet_set(
            wavelet_label_dir(args.wavelet_data_dir, experiment, label)
        )
        cached[condition] = set(wavelet_sets[condition].participants)

    # ---- 3. frequency selections, resolved on the store's AND the cache's grid
    # The 'full' rows read the stored sources on the store's own bins; on a band store
    # without the centre (40 Hz in a 1-30 Hz fit) the nearest bin is the band edge,
    # which would be reported as the centre.
    store_freqs = np.asarray(results.freqs, dtype=float)
    if not store_freqs.min() <= args.center_freq <= store_freqs.max():
        raise ValueError(
            f"--center_freq {args.center_freq:g} Hz lies outside the store's "
            f"{store_freqs.min():g}-{store_freqs.max():g} Hz: its 'full' rows would "
            "be read at the band edge. Read this band's filters at that frequency "
            "with scripts/run_corrca_band_swap.py instead."
        )
    cache_freqs = _cache_frequency_grid(wavelet_sets, conditions)
    store_selections, cache_selections = {}, {}
    for halfwidth in args.halfwidths:
        name = at.selection_label(args.center_freq, halfwidth)
        store_selections[name] = at.frequency_selection(
            results.freqs, args.center_freq, halfwidth
        )
        cache_selections[name] = at.frequency_selection(
            cache_freqs, args.center_freq, halfwidth
        )
    union = np.unique(np.concatenate(list(cache_selections.values())))
    union_position = {int(b): i for i, b in enumerate(union)}
    # The cache bin of every store frequency, for the masked variants' full TF maps.
    tf_bins = None
    # The band-swap TF maps are run_corrca_band_swap.py's, except under another
    # baseline, which that script does not read.
    if (
        args.masked_tf_maps
        and masked_signals
        and (band_signal is None or args.baseline is not None)
    ):
        grid = np.asarray(cache_freqs, dtype=float)
        tf_bins = np.array([int(np.argmin(np.abs(grid - f))) for f in results.freqs])
        if not np.allclose(grid[tf_bins], results.freqs, rtol=1e-6):
            raise ValueError("The store's frequencies are not on the cache's grid.")

    def _has_row(participant: str, condition: str) -> bool:
        if per_recording:
            return bool(results.rows(participant, condition))
        return bool(results.rows(participant))

    store_participants = list(dict.fromkeys(results.participants))
    participants = [
        p
        for p in store_participants
        if all(_has_row(p, c) and p in cached[c] for c in conditions)
    ]
    if not participants:
        raise ValueError(
            "No participant is present in the store under every condition and every cache."
        )
    dropped = [p for p in store_participants if p not in participants]
    if dropped:
        _logger.warning(f"dropped (not paired / not cached): {dropped}")
    n_participants = len(participants)

    # ---- 4. project both conditions (one stream each) ---------------------
    # project_condition takes per-row operators; CorrCA has one, so every row is a view
    # of it.
    projected = {
        condition: project_condition(
            wavelet_sets[condition],
            condition,
            participants,
            resolve,
            binary,
            np.broadcast_to(ops["pca_row"], (n_rows, n_channels)),
            union,
            read_channels,
            args.n_times,
            np.flatnonzero(electrode_mask),
            np.broadcast_to(ops["masked"], (n_rows, n_components, n_channels)),
            outside_filters=(
                np.broadcast_to(ops["outside"], (n_rows, n_components, n_channels))
                if "non_roi_masked" in signals
                else None
            ),
            swapped_filters=(
                np.broadcast_to(ops["swapped"], (n_rows, n_components, n_channels))
                if "swapped" in ops
                else None
            ),
            tf_freq_indices=tf_bins,
            log_power=log_power,
        )
        for condition in conditions
    }

    # ---- 5. epoch geometry (shared window) --------------------------------
    sfreq = results.sfreq
    geometry = {
        c: at.epoch_geometry(onsets[c], projected[c].reference_raw.shape[-1], sfreq)
        for c in conditions
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)
    baseline_mask = baseline_window(times, args.baseline)
    baseline_token = baseline_label(args.baseline)
    if baseline_token:
        _logger.info(
            f"baseline {args.baseline[0]:g} to {args.baseline[1]:g} s: "
            f"{int(baseline_mask.sum())} sample(s), "
            f"{times[baseline_mask][0]:.3f} to {times[baseline_mask][-1]:.3f} s"
        )

    # The stored sources, per condition, on that condition's own time axis.
    if per_recording:
        stored = {
            c: [results.tf_maps[resolve(p, c)] for p in participants]
            for c in conditions
        }
    else:
        stored = {}
        for c in conditions:
            segment = results.condition_track(results.tf_maps, c)
            stored[c] = [segment[resolve(p, c)] for p in participants]
    for c in conditions:
        keep = projected[c].reference_raw.shape[-1]
        if any(src.shape[-1] < keep for src in stored[c]):
            raise ValueError(
                f"[{c}] a stored source is shorter than the projected cache; the store "
                "and the cache disagree on the time axis."
            )
        stored[c] = [src[..., :keep] for src in stored[c]]

    # Each stored source's TF map in units of its own pre-stimulus SD, per recording:
    # the courses' reading at every frequency. condition -> (K, P, F, W). A band-swap
    # run reads only its own filters, so not there.
    tf_baseline = {
        c: np.stack(
            [
                at.baseline_tf_map(src, geometry[c][0], pre, post, baseline_mask)
                for src in stored[c]
            ],
            axis=1,
        ).astype(np.float32)
        for c in conditions
        if band_signal is None
    }
    # The masked variants' maps, read the same way: signal -> condition -> (K, P, F, W).
    masked_tf = {}
    if tf_bins is not None:
        for signal in masked_signals:
            projection = masked_of[signal][2]
            masked_tf[signal] = {
                c: np.stack(
                    [
                        at.baseline_tf_map(
                            src, geometry[c][0], pre, post, baseline_mask
                        )
                        for src in getattr(projected[c], projection)
                    ],
                    axis=1,
                ).astype(np.float32)
                for c in conditions
            }

    polarity_note = (
        f"shared CorrCA filter: one sign per component, anchored at fit time "
        f"('{anchor}'); no per-recording flip"
    )
    baseline_note = (
        f", baseline {args.baseline[0]:g} to {args.baseline[1]:g} s"
        if baseline_token
        else ""
    )
    signs = np.ones((n_participants, len(labels)))

    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if args.band is None
        else SpectrumTypeVariants.BANDS.value
    )
    # Inside the fit's own directory, in a subdirectory of its own; under another
    # baseline, inside baseline_<lo>_<hi>ms/, which mirrors the fit's directory.
    fit_dir = corrca_plots_dir(
        variant,
        store_condition,
        music_type,
        band=args.band,
        n_pca=args.n_pca,
        root=args.save_dir,
    )
    plots_dir = (
        fit_dir / baseline_token if baseline_token else fit_dir
    ) / _PLOTS_SUBDIR
    plot_prefix = "" if args.band is None else f"{args.band}_"
    # The statistical figures here; the TF maps and topographies are the fit's, in its
    # own directory (where step 8 adds the swapped_filter ones).
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- 6. the grid: selection x window ----------------------------------
    rows_out: list[dict] = []
    snr_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    snr_courses: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for sel_name, bins in store_selections.items():
        take = [union_position[int(b)] for b in cache_selections[sel_name]]
        cut: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in ("full", *masked_signals)
        }
        roi_snr: dict[str, np.ndarray] = {}
        roi_scale: dict[str, dict[str, float]] = {}
        for condition in conditions:
            onsets_inside = geometry[condition][0]
            proj = projected[condition]
            # The reference rows, identical in every variant: the masks on the RAW
            # cache, then the PCA mask on the z-scored cache.
            reference_band = np.concatenate(
                [
                    proj.reference_raw[:, :, take, :].mean(axis=2),
                    proj.pca_z[:, :, take, :].mean(axis=2),
                    *(
                        []
                        if proj.pca_log_z is None
                        else [proj.pca_log_z[:, :, take, :].mean(axis=2)]
                    ),
                ],
                axis=1,
            )[:, reference_rows]  # (P, references, T)
            bands = {
                "full": np.stack(
                    [src[:, bins, :].mean(axis=1) for src in stored[condition]]
                ),
                **{
                    s: getattr(proj, masked_of[s][1])[:, :, take, :].mean(axis=2)
                    for s in masked_signals
                },
            }  # learned rows, each (P, K, T)
            for name, learned in bands.items():
                cut[name][condition], _ = at.cut_trials(
                    np.concatenate([learned, reference_band], axis=1),
                    onsets_inside,
                    pre,
                    post,
                )
            roi_band = proj.roi_raw[:, :, take, :].mean(axis=2)
            roi_cut, _ = at.cut_trials(roi_band, onsets_inside, pre, post)
            roi_snr[condition], roi_scale[condition] = at.roi_channelwise_snr(
                roi_cut, baseline_mask
            )

        # One ROI scale for both conditions, so a participant's two conditions are not
        # rescaled differently (the stage-06 rule; exact, not a re-fit).
        shared_roi_scale = float(
            np.median([d["roi_baseline_sd"] for d in roi_scale.values()])
        )
        for condition in conditions:
            roi_snr[condition] = roi_snr[condition] * (
                roi_scale[condition]["roi_baseline_sd"] / shared_roi_scale
            )

        for lo, hi in args.stimulus_intervals:
            window_label, window = _stimulus_window(times, lo, hi)

            for signal in signals:
                value, course = {}, {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+3, N, W)
                    normed, _rel, _pos = at.baseline_normalise(trials, baseline_mask)
                    normed = normed.copy()
                    normed[:, full_mask_column] = roi_snr[condition]
                    per_trial = normed[..., window].mean(axis=-1)
                    value[condition] = np.median(per_trial, axis=2) * signs
                    course[condition] = np.median(normed, axis=2) * signs[:, :, None]

                cell = {
                    "variant": signal,
                    "selection": sel_name,
                    "window": window_label,
                }
                # The course does not depend on the window; every window rewrites it.
                snr_courses[(signal, sel_name)] = course
                snr_values[(signal, sel_name, window_label)] = value
                _accumulate_tests(
                    rows_out,
                    cell,
                    value,
                    conditions,
                    labels,
                    comp_labels,
                    args.contrast_alternative,
                    args.snr_alternative,
                    mask_labels=mask_labels,
                )
                if not args.skip_plots:
                    _write_cell_plots(
                        plots_dir,
                        plot_prefix,
                        cell,
                        value,
                        course,
                        times,
                        (lo, hi),
                        conditions,
                        labels,
                        comp_labels,
                        args.alpha,
                        rows_out,
                        n_participants,
                        polarity_note,
                        mask_labels,
                        participants,
                        title_note=baseline_note,
                    )

    # ---- 7. one CSV for the whole grid ------------------------------------
    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    store_token = (
        f"{variant.value}__{store_condition.value}"
        if variant in CORRCA_INDIVIDUAL_VARIANTS
        else variant.value
    )
    # The band-swap reading gets files of its own, under a name the notebooks'
    # assr_snr_grid__* globs never pick up.
    # Another baseline likewise: assr_snr_grid_baseline_<lo>_<hi>ms__*.
    stem = "assr_snr_grid" if band_signal is None else "corrca_band_swap_snr"
    if baseline_token:
        stem = f"{stem}_{baseline_token}"
    csv_name = (
        f"{stem}__{store_token}__{spectrum}"
        if band_signal is None
        else f"{stem}__{store_token}__{args.swap_band}"
    )
    csv_path = results_root / f"{csv_name}__pca{args.n_pca}.csv"
    frame = pd.DataFrame(rows_out)
    frame.to_csv(csv_path, index=False)
    courses_path = csv_path.with_name(f"{csv_path.stem}__courses.npz")
    save_snr_courses(courses_path, snr_courses, participants, labels, times)
    values_path = csv_path.with_name(f"{csv_path.stem}__values.npz")
    save_snr_values(values_path, snr_values, participants, labels)
    if band_signal is not None or baseline_token:
        # The stored sources' TF maps are the default grid's; not written twice. Under
        # another baseline they are drawn instead (step 9).
        _logger.info(
            f"saved {csv_path}, {courses_path.name} and {values_path.name} "
            f"({len(frame)} test rows, n = {n_participants})"
        )
    else:
        tf_path = csv_path.with_name(f"{csv_path.stem}__tf_baseline.npz")
        save_npz_atomic(
            tf_path,
            participants=np.asarray(participants),
            conditions=np.asarray(conditions),
            sources=np.asarray(comp_labels),
            epoch_times=times,
            freqs=np.asarray(results.freqs),
            **{f"tf_baseline__{c}": tf_baseline[c] for c in conditions},
            **{
                f"tf_baseline__{signal}__{c}": maps[c]
                for signal, maps in masked_tf.items()
                for c in conditions
            },
        )
        _logger.info(
            f"saved {csv_path}, {courses_path.name}, {values_path.name} and "
            f"{tf_path.name} "
            f"({len(frame)} test rows, "
            f"n = {n_participants}, floor p = "
            f"{at.p_floor(n_participants):.5f} two-sided)"
        )

    # ---- 7b. how far the contrast rests on a few participants --------------
    if (
        len(conditions) == 2
        and band_signal is None
        and "full" in signals
        and not args.skip_robustness
    ):
        _contrast_robustness(
            args,
            refit_path=corrca_refit_path(results.path),
            store_filters=np.asarray(results.extras["corrca_filters"], dtype=float),
            store_channels=store_channels,
            read_channels=read_channels,
            wavelet_sets=wavelet_sets,
            conditions=conditions,
            participants=participants,
            labels=labels,
            comp_labels=comp_labels,
            snr_values=snr_values,
            selections=cache_selections,
            geometry=geometry,
            times=times,
            pre=pre,
            post=post,
            baseline_mask=baseline_mask,
            log_power=log_power,
            csv_path=csv_path,
            plots_dir=None if args.skip_plots else plots_dir,
            plot_prefix=plot_prefix,
            tag=f"{variant.value}{baseline_note}",
        )

    # ---- 8. the swapped filters' TF maps, in the fit's own directory --------
    # Last, so the statistics above are saved whatever happens here.
    if swapped_r is not None and not args.skip_plots and not baseline_token:
        fit_dir.mkdir(parents=True, exist_ok=True)
        reading = CORRCA_SWAPPED_READING
        swapped_r.to_csv(
            fit_dir / f"{plot_prefix}{reading}_topomap_correlation.csv", index=False
        )
        if tf_bins is None:
            _logger.info(
                "no swapped_filter TF maps: they need the full-frequency projection "
                "(--masked_tf_maps)."
            )
        elif args.swapped_components > 0:
            twin = CORRCA_SWAPPED_OF[variant]
            scale = "log-power" if twin in CORRCA_LOG_VARIANTS else "raw-power"
            shown = list(range(min(args.swapped_components, n_components)))
            _draw_swapped_tf_maps(
                {c: projected[c].swapped_tf_z for c in conditions},
                {c: onsets[c] for c in conditions},
                participants,
                conditions,
                np.asarray(results.freqs),
                sfreq,
                shown,
                fit_dir,
                f"{plot_prefix}{reading}_",
                label=f"{store_condition.value}_{music_type.value}"
                + ("_log" if log_power else ""),
                note=f"filters of the {scale} fit ({twin.value}) on this fit's input; "
                "topomap r "
                + ", ".join(
                    f"{COMPONENT_PREFIX} {k + 1} {swapped_r['pattern_r'][k]:+.2f}"
                    for k in shown
                ),
            )

    # ---- 9. another baseline: the stimulus-averaged TF maps under it --------
    if baseline_token and not args.skip_plots:
        lo, hi = args.baseline
        out_dir = fit_dir / baseline_token
        shown_swapped = list(range(min(args.swapped_components, n_components)))
        # File prefix -> (condition -> (K, P, F, W) maps, components drawn, what reads).
        readings = {}
        if tf_baseline:
            readings[plot_prefix] = (tf_baseline, list(range(n_components)), "")
        for signal, maps in masked_tf.items():
            shown = (
                list(range(n_components))
                if signal in ("roi_masked", "non_roi_masked")
                else shown_swapped
            )
            readings[f"{plot_prefix}{signal}_"] = (maps, shown, f"{signal}; ")
        if not masked_tf and masked_signals:
            _logger.info(
                f"no {', '.join(masked_signals)} TF maps under this baseline: they "
                "need the full-frequency projection (--masked_tf_maps)."
            )
        freq_marks = list(_TF_FREQ_MARKS)
        if band_signal is not None:
            freq_marks += _band_edges(args, variant, store_condition, results.freqs)
        epoch_marks = [0.0, min(AssrEpoch.STIMULUS_DURATION_S, float(times[-1]))]
        # The fit's sign note, after the reading, as on the fit's masked maps.
        sign_note = (
            CORRCA_ALIGNMENT_NOTE if anchor != "none" else CORRCA_ALIGNMENT_NOTE_NONE
        )
        for prefix, (maps, shown, what) in readings.items():
            if not shown:
                continue
            onset_stacked, rows_participants, rows_conditions = (
                stack_conditions_on_subject_axis(
                    {c: maps[c].transpose(1, 0, 2, 3) for c in conditions},
                    conditions,
                    participants,
                )
            )
            _draw_onset_tf_maps(
                onset_stacked,
                rows_participants,
                rows_conditions,
                conditions,
                np.asarray(results.freqs),
                times,
                epoch_marks,
                shown,
                out_dir,
                prefix,
                shared={
                    "label": f"{store_condition.value}_{music_type.value}"
                    + ("_log" if log_power else ""),
                    "alignment_note": f"{what}baseline {lo * 1e3:g} to "
                    f"{hi * 1e3:g} ms; {sign_note}",
                    "component_prefix": COMPONENT_PREFIX,
                    "freq_marks": freq_marks,
                },
            )


def _stimulus_window(times: np.ndarray, lo: float, hi: float) -> tuple[str, np.ndarray]:
    """``(label, epoch-sample mask)`` of one ``--stimulus_intervals`` entry."""
    if (lo, hi) == (0.0, AssrEpoch.STIMULUS_DURATION_S):
        window = AssrEpoch.stimulus_mask(times)
    else:
        window = (times >= lo) & (times <= hi)
    if not window.any():
        raise ValueError(f"window {lo}-{hi}s selects no epoch sample.")
    return f"{lo:g}-{hi:g}s", window


def _load_subset_refits(
    path: Path, store_filters: np.ndarray
) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray, np.ndarray] | None:
    """The fit's leave-subset-out refits, or ``None`` with a warning.

    :param store_filters: ``(C, K)`` the store's filters, which the refits' own copy
        must match, so a refit file left over from another fit is never read.
    :return: ``(left-out participant labels per refit, (R, C, K) filters, (R, K)
        pattern_r, (R, K) filter_r)``: the refit's topography and filter against the
        full fit's.
    """
    if not path.is_file():
        _logger.warning(
            f"no refit filters at {path}: the robustness rows use the stored filters "
            "only (run the fit with --store_components and the influence check)."
        )
        return None
    with np.load(path, allow_pickle=False) as stored:
        names = [str(n) for n in stored["participants"]]
        subsets = stored["subsets"]
        filters = np.asarray(stored["filters"], dtype=float)
        full = np.asarray(stored["full_filters"], dtype=float)
        pattern_r = np.asarray(stored["pattern_r"], dtype=float)
        filter_r = np.asarray(stored["filter_r"], dtype=float)
    scale = float(np.abs(store_filters).max()) or 1.0
    if full.shape != store_filters.shape or not np.allclose(
        full, store_filters, rtol=1e-4, atol=1e-5 * scale
    ):
        _logger.warning(
            f"{path.name} belongs to another fit than the store (its filters differ): "
            "not read."
        )
        return None
    dropped = [tuple(names[g] for g in row if g >= 0) for row in subsets]
    return dropped, filters, pattern_r, filter_r


def _contrast_robustness(
    args: argparse.Namespace,
    *,
    refit_path: Path,
    store_filters: np.ndarray,
    store_channels: list[str],
    read_channels: list[str],
    wavelet_sets: dict,
    conditions: list[str],
    participants: list[str],
    labels: list[str],
    comp_labels: list[str],
    snr_values: dict[tuple[str, str, str], dict[str, np.ndarray]],
    selections: dict[str, np.ndarray],
    geometry: dict,
    times: np.ndarray,
    pre: int,
    post: int,
    baseline_mask: np.ndarray,
    log_power: bool,
    csv_path: Path,
    plots_dir: Path | None,
    plot_prefix: str,
    tag: str,
) -> None:
    """Whether the ``full`` contrast of every grid cell rests on a few participants.

    Per cell and source, all on the ``full`` reading:

    - ``full`` — the contrast test itself, with the exact sign test beside it;
    - ``fixed_drop`` — rerun with every subset of ``--drop_sizes`` participants left
      out, the filters unchanged: does the verdict hang on whose differences are in;
    - ``refit_drop`` (components only) — CorrCA refitted without each subset the fit
      wrote (``--refit_subset_sizes``), every remaining participant read through the
      refit's filters: does it hang on the participants that shaped the filters;
    - ``held_out`` (components only) — each participant read through the refit that
      left them alone out, so no value comes from a filter its own data helped choose.

    The refits' values come from the cache, the stored ``full`` values from the store;
    the full fit is read both ways and their agreement logged.

    Writes ``<csv stem>__robustness.csv`` (every replicate) and
    ``__robustness_summary.csv`` (per subset size), and a ``robustness__full.png`` per
    grid cell.
    """
    alt = args.contrast_alternative
    first, second = conditions
    n_components = len(comp_labels)
    refits = _load_subset_refits(refit_path, store_filters)
    replicate_values = None
    dropped_sets: list[tuple[str, ...]] = []
    if refits is not None:
        dropped_sets, refit_filters, refit_pattern_r, refit_filter_r = refits
        # Replicate 0 is the full fit, read through the cache like the refits.
        stack = np.concatenate([store_filters[None], refit_filters], axis=0)
        flat = stack.transpose(0, 2, 1).reshape(-1, stack.shape[1])  # (M, C)
        if read_channels != store_channels:
            flat = _embed(flat, store_channels, read_channels)
        windows = dict(
            _stimulus_window(times, lo, hi) for lo, hi in args.stimulus_intervals
        )
        _logger.info(
            f"robustness: {len(dropped_sets)} refit(s) from {refit_path.name}, "
            f"{flat.shape[0]} filter(s) read per participant"
        )
        replicate_values = {
            c: replicate_snr_values(
                wavelet_sets[c],
                c,
                participants,
                read_channels,
                args.n_times,
                flat,
                selections,
                geometry[c][0],
                pre,
                post,
                baseline_mask,
                windows,
                log_power=log_power,
            )
            for c in conditions
        }
    single = {d[0]: r for r, d in enumerate(dropped_sets) if len(d) == 1}

    rows: list[dict] = []
    summaries: list[pd.DataFrame] = []
    for (signal, selection, window), value in snr_values.items():
        if signal != "full":
            continue
        cell = {"selection": selection, "window": window}
        reps = None
        if replicate_values is not None:
            reps = {
                c: replicate_values[c][(selection, window)].reshape(
                    len(participants), -1, n_components
                )
                for c in conditions
            }
            gap = max(
                float(np.nanmax(np.abs(reps[c][:, 0] - value[c][:, :n_components])))
                for c in conditions
            )
            _logger.info(
                f"robustness {selection} {window}: the full fit read from the cache "
                f"vs from the store, largest |difference| {gap:.2g}"
            )
        cell_rows: list[pd.DataFrame] = []
        for s, source in enumerate(labels):
            d = at.condition_contrast(value, conditions, s)
            full = {
                "k": 0,
                "dropped": "",
                "n": int(np.isfinite(d).sum()),
                **at.paired_test(d, alt),
                "sign_p": at.sign_test(d, alt),
            }
            families = {
                "full": pd.DataFrame([full]),
                "fixed_drop": at.replicate_tests(
                    at.drop_participants(d, participants, args.drop_sizes), alt
                ),
            }
            if reps is not None and source in comp_labels:
                k = comp_labels.index(source)
                refit_reps = []
                for r, left_out in enumerate(dropped_sets):
                    keep = np.array([p not in left_out for p in participants])
                    refit_reps.append(
                        (
                            left_out,
                            reps[first][keep, r + 1, k] - reps[second][keep, r + 1, k],
                        )
                    )
                families["refit_drop"] = at.replicate_tests(refit_reps, alt).assign(
                    pattern_r=refit_pattern_r[:, k], filter_r=refit_filter_r[:, k]
                )
                if all(p in single for p in participants):
                    held = np.array(
                        [
                            reps[first][i, single[p] + 1, k]
                            - reps[second][i, single[p] + 1, k]
                            for i, p in enumerate(participants)
                        ]
                    )
                    families["held_out"] = pd.DataFrame(
                        [
                            {
                                "k": 1,
                                "dropped": "",
                                "n": int(np.isfinite(held).sum()),
                                **at.paired_test(held, alt),
                                "sign_p": at.sign_test(held, alt),
                            }
                        ]
                    )
            for test, frame in families.items():
                frame = frame.assign(**cell, source=source, test=test)
                cell_rows.append(frame)
                if test in ("fixed_drop", "refit_drop"):
                    summary = at.replicate_summary(frame, full["p"], args.alpha)
                    summary.insert(0, "test", test)
                    summary.insert(0, "source", source)
                    summary.insert(0, "window", window)
                    summary.insert(0, "selection", selection)
                    summary["full_p"] = full["p"]
                    summary["fragility"] = at.fragility(frame, full["p"], args.alpha)
                    summaries.append(summary)
            if source in comp_labels:
                _log_robustness(source, cell, full, families, args.alpha)
        cell_table = pd.concat(cell_rows, ignore_index=True)
        rows.append(cell_table)
        if plots_dir is not None:
            cell_dir = plots_dir / selection / window
            cell_dir.mkdir(parents=True, exist_ok=True)
            plot_contrast_robustness(
                cell_table,
                comp_labels[:STABILITY_COMPONENTS],
                participants,
                args.alpha,
                f"Robustness of {first} − {second} ({alt}) to the participants — "
                f"full__{selection}__{window}, {tag}, n={len(participants)}",
                cell_dir / f"{plot_prefix}robustness__full.png",
            )
    if not rows:
        return
    table = pd.concat(rows, ignore_index=True)
    lead = ["selection", "window", "source", "test", "k", "dropped", "n"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table["significant"] = table["p"] < args.alpha
    out = csv_path.with_name(f"{csv_path.stem}__robustness.csv")
    table.to_csv(out, index=False)
    summary_path = csv_path.with_name(f"{csv_path.stem}__robustness_summary.csv")
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    _logger.info(f"saved {out.name} ({len(table)} rows) and {summary_path.name}")


def _log_robustness(
    source: str, cell: dict, full: dict, families: dict, alpha: float
) -> None:
    """One log line per component and grid cell."""
    parts = [f"p {full['p']:.3g} ({full['same sign']}), sign test {full['sign_p']:.3g}"]
    for test in ("fixed_drop", "refit_drop"):
        if test not in families:
            continue
        frame = families[test]
        per_k = ", ".join(
            f"-{int(k)}: {int((g['p'] < alpha).sum())}/{len(g)} p<{alpha:g}"
            for k, g in frame.groupby("k")
        )
        fragile = at.fragility(frame, full["p"], alpha)
        parts.append(
            f"{test} [{per_k}; verdict changes from k={fragile}]"
            if fragile
            else f"{test} [{per_k}; verdict never changes]"
        )
    if "held_out" in families:
        held = families["held_out"].iloc[0]
        parts.append(f"held-out p {held['p']:.3g} ({held['same sign']})")
    _logger.info(
        f"robustness {cell['selection']} {cell['window']} {source}: " + "; ".join(parts)
    )


def _band_edges(
    args: argparse.Namespace,
    variant: IvaVariants,
    store_condition: ConditionVariants,
    broadband_freqs: np.ndarray,
) -> list[float]:
    """The band twin's edges inside the broadband grid, marked like the band swap's."""
    path = iva_results_path(
        ExperimentNames(args.experiment),
        store_condition,
        variant,
        MusicTypeVariants(args.music_type),
        args.swap_band,
        args.n_pca,
        processed_data_dir=args.store_dir,
    )
    with np.load(path, allow_pickle=False) as stored:
        band = np.asarray(stored["freqs"], dtype=float)
    grid = np.asarray(broadband_freqs, dtype=float)
    return [
        float(edge)
        for edge in (band.min(), band.max())
        if grid.min() < edge < grid.max()
    ]


def _draw_onset_tf_maps(
    onset_stacked: np.ndarray,
    rows_participants: list[str],
    rows_conditions: list[str],
    conditions: list[str],
    freqs: np.ndarray,
    epoch_times: np.ndarray,
    epoch_marks: list[float],
    components: list[int],
    out_dir: Path,
    prefix: str,
    *,
    shared: dict,
) -> None:
    """The fit scripts' stimulus-averaged TF figures, group and ``participants/``.

    :param onset_stacked: ``(rows, K, F, W)`` trial-normalised maps, conditions stacked
        on the row axis.
    """
    plot_condition_mean_tf_maps(
        onset_stacked,
        rows_participants,
        rows_conditions,
        conditions,
        freqs,
        epoch_times,
        components,
        epoch_marks=epoch_marks,
        equalize=False,
        save_path=out_dir / f"{prefix}condition_mean_tf_maps_onset.png",
        **shared,
    )
    plt.close("all")
    plot_participant_condition_tf_maps(
        onset_stacked,
        rows_participants,
        rows_conditions,
        conditions,
        freqs,
        epoch_times,
        components,
        root_dir=out_dir / "participants",
        prefix=f"{prefix}onset_",
        epoch_marks=epoch_marks,
        equalize=False,
        **shared,
    )
    plt.close("all")


def _draw_swapped_tf_maps(
    by_condition: dict[str, np.ndarray],
    onsets: dict[str, np.ndarray],
    participants: list[str],
    conditions: list[str],
    freqs: np.ndarray,
    sfreq: float,
    components: list[int],
    out_dir: Path,
    prefix: str,
    *,
    label: str,
    note: str,
    freq_marks: list[float] | None = None,
) -> None:
    """The fit scripts' TF figures for the ``swapped_filter`` reading.

    The same plotting calls and the same stimulus average as the fit's own
    ``condition_mean_tf_maps`` (+ ``_onset``, + ``participants/``), so the two sit side
    by side in the fit's directory and differ only in the filters.

    :param by_condition: Condition -> ``(P, K, F, T)`` projection, z-scored like the
        fit's input.
    :param onsets: Condition -> onset samples on that condition's own time axis.
    :param freq_marks: Frequencies (Hz) to mark; default the ASSR frequency.
    """
    shared = {
        "label": label,
        "alignment_note": note,
        "component_prefix": COMPONENT_PREFIX,
        "freq_marks": _TF_FREQ_MARKS if freq_marks is None else freq_marks,
    }
    participants_dir = out_dir / "participants"
    stacked, rows_participants, rows_conditions = stack_conditions_on_subject_axis(
        by_condition, conditions, participants
    )
    n_split = stacked.shape[-1]
    reference = onsets[conditions[0]]
    _logger.info(f"[{label}] writing {prefix}* TF maps to {out_dir}")
    plot_condition_mean_tf_maps(
        stacked,
        rows_participants,
        rows_conditions,
        conditions,
        freqs,
        np.arange(n_split) / sfreq,
        components,
        time_marks=reference[reference < n_split] / sfreq,
        save_path=out_dir / f"{prefix}condition_mean_tf_maps.png",
        **shared,
    )
    plt.close("all")
    plot_participant_condition_tf_maps(
        stacked,
        rows_participants,
        rows_conditions,
        conditions,
        freqs,
        np.arange(n_split) / sfreq,
        components,
        root_dir=participants_dir,
        prefix=prefix,
        time_marks=reference[reference < n_split] / sfreq,
        **shared,
    )
    del stacked
    averaged = _onset_average_per_condition(
        by_condition,
        onsets,
        participants,
        conditions,
        label=label,
        sfreq=sfreq,
        min_onsets=_MIN_ONSETS,
    )
    if averaged is None:
        return
    onset_stacked, onset_participants, onset_conditions, epoch_times, marks = averaged
    _draw_onset_tf_maps(
        onset_stacked,
        onset_participants,
        onset_conditions,
        conditions,
        freqs,
        epoch_times,
        marks,
        components,
        out_dir,
        prefix,
        shared=shared,
    )


def _write_cell_plots(
    plots_dir,
    plot_prefix,
    cell,
    value,
    course,
    times,
    window_interval,
    conditions,
    labels,
    comp_labels,
    alpha,
    rows_out,
    n_participants,
    polarity_note,
    mask_labels,
    participants,
    *,
    title_note: str = "",
) -> None:
    """The stage-06 three figures for one grid cell, with components named ``CC``.

    Plus the per-participant trial courses of the first two components beside the
    ``(PCA, z)`` reference (``(full)`` in a non-ROI store, which has no other). A single
    condition has no contrast, so it gets no p-value summary and no contrast box on its
    courses. *mask_labels* are the store's reference rows; *title_note* is appended to
    every title (the baseline, when not the default).
    """
    tag = f"{cell['variant']}__{cell['selection']}__{cell['window']}{title_note}"
    variant = cell["variant"]
    cell_dir = plots_dir / cell["selection"] / cell["window"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in rows_out if all(r[k] == cell[k] for k in cell)])
    paired = len(conditions) > 1
    cell_tables = {
        "contrast": frame[frame["family"] == "contrast"] if paired else None,
        **{
            f"discrimination:{m}": frame[
                (frame["family"] == "discrimination") & (frame["reference"] == m)
            ]
            for m in (mask_labels if paired else ())
        },
        **{
            f"snr:{m}:{c}": frame[
                (frame["family"] == "snr")
                & (frame["reference"] == m)
                & (frame["condition"] == c)
            ]
            for m in mask_labels
            for c in conditions
        },
    }
    if paired:
        plot_cell(
            cell_tables,
            value,
            conditions,
            labels,
            comp_labels,
            alpha,
            f"Condition contrast & discrimination — {tag}, n={n_participants}",
            cell_dir / f"{plot_prefix}pvalue_summary__{variant}.png",
            component_prefix=COMPONENT_PREFIX,
            mask_labels=mask_labels,
        )
    plot_snr(
        cell_tables,
        value,
        conditions,
        comp_labels,
        alpha,
        f"{COMPONENT_PREFIX} SNR vs reference, per condition — {tag}, "
        f"n={n_participants}",
        cell_dir / f"{plot_prefix}snr_by_condition__{variant}.png",
        component_prefix=COMPONENT_PREFIX,
        mask_labels=mask_labels,
    )
    plot_courses(
        course,
        times,
        window_interval,
        cell_tables["contrast"],
        labels,
        conditions,
        alpha,
        f"Trial course per spatial filter — {tag}, n={n_participants}\n{polarity_note}",
        cell_dir / f"{plot_prefix}trial_course_by_source__{variant}.png",
        mask_labels=mask_labels,
    )
    # The PCA mask on the z-scored cache: what the fit was handed. A non-ROI store
    # keeps only the full-mask reference, so it stands in there.
    participant_reference = (
        at.PCA_Z_MASK_LABEL if at.PCA_Z_MASK_LABEL in mask_labels else _FULL_LABEL
    )
    plot_participant_courses(
        course,
        times,
        window_interval,
        labels,
        participants,
        conditions,
        [*comp_labels[:_PARTICIPANT_COURSE_COMPONENTS], participant_reference],
        f"Trial course per participant — {tag}\n{polarity_note}",
        cell_dir / f"{plot_prefix}trial_course_by_participant__{variant}.png",
        mask_labels=mask_labels,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
