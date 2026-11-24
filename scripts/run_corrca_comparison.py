"""
CLI script setting stage-06b CorrCA fits side by side: one run across ``--n_pca``, or
across frequency bands, or across the joins. Read-only: every fit, its figures' CSVs
and its SNR grid must already exist; nothing is fitted.

Three comparisons (``--compare``), each over the **full-montage, raw-power** fit
only (never ``_log``, ``_roi`` or ``_non_roi``):

* ``pca`` — one run (``--run``, default the time-axis join) and band (``--band``)
  at every ``--n_pca``: does a larger subspace find more synchrony, or only fit
  more noise?
* ``band`` — one run at one ``--n_pca``, broadband against its band-restricted
  twins (``--bands``, default ``broadband 1-30hz 30-50hz``);
* ``join`` — one band and ``--n_pca``, the runs against each other (``--runs``,
  default the time-axis join, the subject-axis join and the Placebo-fitted join; the
  first is the topomaps' reference row).
  An individual-conditions run holds two fits, one per condition, with different
  filters: its rows are drawn per condition and it has no paired test.

Components are compared **by index** (CC 1 of one fit against CC 1 of the next),
never re-matched; the topography figure gives each map's r with the first fit's same
component, so a re-ordering shows.

Per fit, the leading ``--components`` (default 3), read from what the fit already
wrote:

* **ISC** — the fit's own (in-sample) ISC, and the ISC per condition on the whole
  recording and on the onset-locked maps (``corrca_condition_isc.csv``,
  ``corrca_onset_condition_isc.csv``);
* **held out** — from the leave-one-participant-out influence
  (``influence/corrca_influence.csv``): the median held-out ISC per condition, the
  median in-sample − held-out gap (the fit's optimism), and the |r| of the
  topography refitted without each participant against the full one (median,
  worst);
* **Placebo vs Psilocybin p-values** — the paired LOO-ISC tests (whole, onset) and
  the SNR grid's ``contrast`` family on the stored sources (``full``) at
  ``--snr_selection`` for every ``--snr_windows`` (uncorrected, as written);
* **SNR courses** — the grid's trial-normalised courses per condition
  (``__courses.npz``), mean ± SEM over participants;
* **topographies** — the shared filters ``W`` (what the fit draws) and the shared
  forward patterns (what the held-out |r| compares), one row per fit;
* **TF maps** — each condition's group-mean TF map, as the fit's
  ``condition_mean_tf_maps`` draws it without the difference row: the whole
  recording (participants equal-weighted) and the onset-locked average (each trial
  in its pre-onset SD). One row per (fit, condition): a fit's Placebo above its
  Psilocybin, the next fit's pair below, one colour scale per fit and component;
* **the two terms of the ISC** — ``R_w``'s (the variance the component carries) and
  ``R_b``'s (its covariance with the other participants) per condition and per
  participant, from the stored sources, each per unit filter norm so fits of other
  ``n_pca``, bands or joins share one scale
  (:func:`~src.analysis.corrca.condition_variance_terms`), with paired tests of
  both; and how both move when one participant is left out, from the influence
  CSV's ``within_*`` / ``between_*`` columns (fits run before those columns existed
  are skipped with a warning; rerun the fit job to add them).

The stores are read with only the compared components, one recording at a time, so
a 30-component broadband store costs a tenth of its size in memory. Anything missing
(a store, a CSV, a grid) is a warning and a gap in the figures, never an error. A
1-30 Hz fit has no 40 Hz bin and so no SNR grid of its own: its SNR rows (p-values
and courses) are its filters read on the broadband input, the band-swap grid
(``run_corrca_snr_grid.py --swap_band 1-30hz``, job ``run_band_swap.pbs``), when that
has run.

The script takes no ``--n_subjects`` / ``--n_channels`` / ``--n_times``: it reads
finished fits, so a small check runs on small fits (see the notebook).

Output::

    plots/06b-corrca-condition-comparison/Comparison_<MusicType>/
        <spectrum>/pca__<variant>/              # --compare pca
        band__<variant>/pca_<n>/                # --compare band
        <spectrum>/join__<run>-<run>.../pca_<n>/  # --compare join
            corrca_comparison_metrics.png / .csv
            corrca_comparison_topomap_filter.png, corrca_comparison_topomap_pattern.png
            corrca_comparison_tf_maps.png, corrca_comparison_tf_maps_onset.png
            corrca_comparison_snr_course.png
            corrca_comparison_variance_terms.png / .csv (+ _tests.csv)
            corrca_comparison_variance_shift.png / .csv

Examples::

    python scripts/run_corrca_comparison.py --compare pca --run separate \\
        --n_pca 3 5 10 20

    python scripts/run_corrca_comparison.py --compare band --run tracks --n_pca 5

    python scripts/run_corrca_comparison.py --compare join --n_pca 5 --band 30-50hz

    python scripts/run_corrca_comparison.py --compare join --n_pca 10 \\
        --runs separate placebo_fit tracks
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    CORRCA_INDIVIDUAL_VARIANTS,
    CORRCA_STAGE_DIR,
    INFLUENCE_SUBDIR,
    MAX_DRAWN_COMPONENTS,
    corrca_plots_dir,
)

# The fits' own stimulus average, so the maps are the fits' condition_mean_tf_maps.
from scripts.run_iva_condition_comparison import _onset_average  # noqa: E402
from scripts.run_corrca_snr_grid import (  # noqa: E402
    _STORE_CONDITIONS,
    BAND_SWAPPED_READING,
)
from src.analysis.corrca import (  # noqa: E402
    condition_variance_terms,
    matched_component_correlation,
    paired_condition_test,
)
from src.analysis.iva_quality import (  # noqa: E402
    ASSR_FREQ,
    equalize_subject_influence,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.definitions.frequency import resolve_band_range  # noqa: E402
from src.io.iva_store import (  # noqa: E402
    IvaComponentResults,
    iva_results_path,
    load_iva_components,
)
from src.io.loading import load_snr_courses  # noqa: E402
from src.visualization.corrca_plots import (  # noqa: E402
    plot_comparison_courses,
    plot_comparison_metrics,
    plot_comparison_tf_maps,
    plot_comparison_topomaps,
    plot_comparison_variance_shift,
    plot_comparison_variance_terms,
)

_logger = logging.getLogger(__name__)

#: ``--run`` token -> the run's full-montage, raw-power fit.
RUNS = {
    "separate": IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE,
    "tracks": IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS,
    "individual": IvaVariants.CORRCA_CHANNEL_INDIVIDUAL_CONDITIONS,
    "placebo_fit": IvaVariants.CORRCA_CHANNEL_PLACEBO_FIT,
}
COMPARISONS = ("pca", "band", "join")
BROADBAND = "broadband"
DEFAULT_PCA_SWEEP = (3, 5, 10, 20)
#: Condition label of a reading pooled over both conditions (the time-axis join's
#: influence rows are participants, not recordings).
POOLED = "both"
STEM = "corrca_comparison"
#: The ``within_*`` / ``between_*`` influence columns the shift figure reads.
SHIFT_COLUMNS = tuple(
    f"{term}_{reading}"
    for term in ("within", "between")
    for reading in ("full", "fixed", "without")
)


class Fit(NamedTuple):
    """One stored fit and where its products are."""

    entry: str
    #: Row label: the entry, plus the condition for an individual-condition fit.
    label: str
    variant: IvaVariants
    #: Conditions this fit's rows hold.
    conditions: list[str]
    store_path: Path
    plot_dir: Path
    #: ``"<band>_"`` for a band store's files, else empty.
    prefix: str
    snr_csv: Path
    #: The band-swap grid of a band fit: its filters on the broadband input,
    #: ``None`` for a broadband fit.
    swap_csv: Path | None


class Entry(NamedTuple):
    """One position on the comparison's x axis."""

    label: str
    band: str | None
    fits: list[Fit]


@dataclass
class FitReading:
    """What one fit contributes to the figures."""

    fit: Fit
    metrics: list[dict] = field(default_factory=list)
    variance: pd.DataFrame | None = None
    variance_tests: pd.DataFrame | None = None
    shift: pd.DataFrame | None = None
    #: ``"filter"`` / ``"pattern"`` -> ``(K, C)`` shared maps.
    maps: dict[str, np.ndarray] | None = None
    info: object = None
    channels: tuple[str, ...] | None = None
    fit_isc: np.ndarray | None = None
    #: Condition -> ``(participants, n, W)``.
    courses: dict[str, np.ndarray] = field(default_factory=dict)
    course_times: np.ndarray | None = None
    #: ``(n,)`` contrast p at the first window.
    course_p: np.ndarray | None = None
    #: How the SNR rows were read, when not from the fit's own grid.
    snr_note: str = ""
    #: Condition -> ``((n, F, T) mean, times)``: the whole-recording TF map, each
    #: participant equal-weighted, and the onset-locked one, trial-normalised.
    tf_whole: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    tf_onset: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    tf_freqs: np.ndarray | None = None
    epoch_marks: list[float] | None = None


def _band_token(band: str) -> str | None:
    return None if band == BROADBAND else band


def _band_range(band: str | None) -> tuple[float, float] | None:
    """``(low, high)`` Hz of a store band token (``30-50hz`` or a named band)."""
    if band is None:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)hz", band)
    if match:
        return float(match.group(1)), float(match.group(2))
    return resolve_band_range(band)


def _window_bounds(window: str) -> tuple[float, float]:
    """``(lo, hi)`` seconds of a grid window label such as ``0-0.5s``."""
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)s", window)
    if not match:
        raise ValueError(f"Cannot read the window label {window!r}; expected 'lo-his'.")
    return float(match.group(1)), float(match.group(2))


def _fits_of(
    entry: str,
    run: str,
    band: str | None,
    n_pca: int,
    args: argparse.Namespace,
) -> list[Fit]:
    variant = RUNS[run]
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if band is None
        else SpectrumTypeVariants.BANDS.value
    )
    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    if variant in CORRCA_INDIVIDUAL_VARIANTS:
        stores = [
            (ConditionVariants(c), [c], f"{entry} · {c}", f"{variant.value}__{c}")
            for c in args.conditions
        ]
    else:
        stores = [
            (_STORE_CONDITIONS[variant], list(args.conditions), entry, variant.value)
        ]
    return [
        Fit(
            entry=entry,
            label=label,
            variant=variant,
            conditions=conditions,
            store_path=iva_results_path(
                experiment,
                store_condition,
                variant,
                music_type,
                band,
                n_pca,
                processed_data_dir=args.store_dir,
            ),
            plot_dir=corrca_plots_dir(
                variant,
                store_condition,
                music_type,
                band=band,
                n_pca=n_pca,
                root=args.fit_plots_dir,
            ),
            prefix="" if band is None else f"{band}_",
            snr_csv=results_dir / f"assr_snr_grid__{token}__{spectrum}__pca{n_pca}.csv",
            swap_csv=(
                None
                if band is None
                else results_dir
                / f"corrca_band_swap_snr__{token}__{band}__pca{n_pca}.csv"
            ),
        )
        for store_condition, conditions, label, token in stores
    ]


def build_entries(args: argparse.Namespace) -> list[Entry]:
    """The compared fits, in x order, from ``--compare`` and its options."""
    if args.compare == "pca":
        band = _band_token(args.band)
        return [
            Entry(f"pca {n}", band, _fits_of(f"pca {n}", args.run, band, n, args))
            for n in args.n_pca
        ]
    (n_pca,) = args.n_pca
    if args.compare == "band":
        return [
            Entry(b, _band_token(b), _fits_of(b, args.run, _band_token(b), n_pca, args))
            for b in args.bands
        ]
    band = _band_token(args.band)
    return [
        Entry(run, band, _fits_of(run, run, band, n_pca, args)) for run in args.runs
    ]


def output_dir(args: argparse.Namespace) -> Path:
    """``Comparison_<music>/[<spectrum>/]<analysis>/[pca_<n>/]`` of the comparison."""
    label = f"Comparison_{args.music_type}"
    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if args.band == BROADBAND
        else SpectrumTypeVariants.BANDS.value
    )
    if args.compare == "pca":
        return ProjectPaths.analysis_plots_dir(
            CORRCA_STAGE_DIR,
            label,
            spectrum=spectrum,
            analysis_type=f"pca__{RUNS[args.run].value}",
            root=args.save_dir,
        )
    if args.compare == "band":
        return ProjectPaths.analysis_plots_dir(
            CORRCA_STAGE_DIR,
            label,
            analysis_type=f"band__{RUNS[args.run].value}",
            n_pca=args.n_pca[0],
            root=args.save_dir,
        )
    return ProjectPaths.analysis_plots_dir(
        CORRCA_STAGE_DIR,
        label,
        spectrum=spectrum,
        # The runs in the name, so another set never overwrites this one.
        analysis_type=f"join__{'-'.join(args.runs)}",
        n_pca=args.n_pca[0],
        root=args.save_dir,
    )


def _metric(
    reading: FitReading, component: int, condition: str, metric: str, value: float
) -> None:
    reading.metrics.append(
        {
            "entry": reading.fit.entry,
            "fit": reading.fit.label,
            "component": component,
            "condition": condition,
            "metric": metric,
            "value": float(value),
        }
    )


def _read_csv(path: Path, what: str, label: str) -> pd.DataFrame | None:
    if not path.is_file():
        _logger.warning(f"[{label}] no {what} at {path}.")
        return None
    return pd.read_csv(path, dtype={"participant": str})


def _condition_rows(
    results: IvaComponentResults, sources: np.ndarray, conditions: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Each condition's sources and participants: a segment of a time-axis store,
    the condition's rows of a subject-axis one."""
    by_condition, participants = {}, {}
    for condition in conditions:
        if results.segment_conditions:
            if condition not in results.segment_conditions:
                continue
            by_condition[condition] = results.condition_track(sources, condition)
            participants[condition] = list(results.participants)
        else:
            rows = results.rows(condition=condition)
            if not rows:
                continue
            by_condition[condition] = sources[rows]
            participants[condition] = [results.participants[i] for i in rows]
    return by_condition, participants


def read_store(reading: FitReading, n: int) -> None:
    """The fit's ISC, maps and topography layout, and the two ISC terms per
    condition and participant from its stored sources."""
    fit = reading.fit
    if not fit.store_path.is_file():
        _logger.warning(
            f"[{fit.label}] no store at {fit.store_path}: no maps, no ISC terms."
        )
        return
    results = load_iva_components(fit.store_path, components=n)
    filters = np.asarray(results.extras["corrca_filters"], dtype=float)  # (C, K)
    reading.fit_isc = np.asarray(results.extras["corrca_fit_isc"], dtype=float)
    reading.maps = {
        "filter": filters.T,
        "pattern": np.asarray(results.extras["corrca_shared_patterns"], dtype=float),
    }
    reading.channels = results.channel_names
    reading.info = results.topo_info()
    fit_condition = fit.conditions[0] if len(fit.conditions) == 1 else ""
    for k in range(min(n, reading.fit_isc.size)):
        _metric(reading, k + 1, fit_condition, "fit_isc", reading.fit_isc[k])

    by_condition, participants = _condition_rows(
        results, results.tf_maps, fit.conditions
    )
    if not by_condition:
        _logger.warning(f"[{fit.label}] the store holds none of {fit.conditions}.")
        return
    norms = (filters[:, : results.tf_maps.shape[1]] ** 2).sum(axis=0)
    group, per_participant = condition_variance_terms(by_condition, participants, norms)
    for row in group.itertuples():
        _metric(reading, row.component, row.condition, "within", row.within)
        _metric(reading, row.component, row.condition, "between", row.between)
    reading.variance = per_participant.assign(entry=fit.entry, fit=fit.label)
    if len(by_condition) == 2:
        tests = []
        for term in ("within", "between"):
            test = paired_condition_test(per_participant, list(by_condition), term)
            tests.append(test.assign(term=term))
            for row in test.itertuples():
                _metric(reading, row.component, "", f"p_{term}", row.p_value)
        reading.variance_tests = pd.concat(tests).assign(entry=fit.entry, fit=fit.label)
    for row in group.itertuples():
        _logger.info(
            f"[{fit.label}] CC {row.component} {row.condition}: R_w term "
            f"{row.within:.4g}, R_b term {row.between:.4g}, ISC {row.isc:.4f}"
        )
    read_tf_maps(reading, results, by_condition)


#: Stimulus epochs a condition needs before its onset average is drawn (the fits').
MIN_ONSETS = 5


def read_tf_maps(
    reading: FitReading,
    results: IvaComponentResults,
    by_condition: dict[str, np.ndarray],
) -> None:
    """Each condition's group-mean TF maps, as the fit's ``condition_mean_tf_maps``
    draws them: the whole recording with every participant equal-weighted, and the
    onset-locked average, each trial in its pre-onset SD, median over trials."""
    reading.tf_freqs = np.asarray(results.freqs, dtype=float)
    for condition, sources in by_condition.items():
        times = (
            results.times_for(condition)
            if results.segment_conditions
            else np.asarray(results.times, dtype=float)
        )
        reading.tf_whole[condition] = (
            equalize_subject_influence(sources).mean(axis=0),
            times,
        )
        averaged = _onset_average(
            sources,
            results.stimulus_onsets(condition),
            label=f"{reading.fit.label}, {condition}",
            n_times=sources.shape[-1],
            sfreq=results.sfreq,
            min_onsets=MIN_ONSETS,
        )
        if averaged is not None:
            onset_tf, epoch_times, reading.epoch_marks = averaged
            reading.tf_onset[condition] = (onset_tf.mean(axis=0), epoch_times)


def read_isc_csvs(reading: FitReading, n: int) -> None:
    """The per-condition ISC (whole, onset) and their paired tests."""
    fit = reading.fit
    for stem, metric in (("corrca", "whole"), ("corrca_onset", "onset")):
        group = _read_csv(
            fit.plot_dir / f"{fit.prefix}{stem}_condition_isc.csv",
            f"{metric} ISC",
            fit.label,
        )
        if group is not None:
            for row in group[group["component"] <= n].itertuples():
                _metric(reading, row.component, row.condition, f"isc_{metric}", row.isc)
        test_path = fit.plot_dir / f"{fit.prefix}{stem}_condition_isc_test.csv"
        if len(fit.conditions) == 2 and test_path.is_file():
            test = pd.read_csv(test_path)
            for row in test[test["component"] <= n].itertuples():
                _metric(reading, row.component, "", f"p_isc_{metric}", row.p_value)


def read_influence(reading: FitReading, n: int) -> None:
    """Held-out ISC, the in-sample optimism, the refit topography |r| and the ISC
    terms' leave-one-out shift."""
    fit = reading.fit
    influence = _read_csv(
        fit.plot_dir / INFLUENCE_SUBDIR / f"{fit.prefix}corrca_influence.csv",
        "influence table",
        fit.label,
    )
    if influence is None:
        return
    influence = influence[influence["component"] <= n].copy()
    default = fit.conditions[0] if len(fit.conditions) == 1 else POOLED
    influence["condition"] = influence["condition"].fillna(default).replace("", default)
    influence["gap"] = influence["insample_isc"] - influence["heldout_isc"]
    for (component, condition), rows in influence.groupby(["component", "condition"]):
        _metric(
            reading, component, condition, "heldout_isc", rows["heldout_isc"].median()
        )
        _metric(
            reading,
            component,
            condition,
            "insample_minus_heldout",
            rows["gap"].median(),
        )
    per_group = influence.drop_duplicates(["group", "component"])
    topo_condition = fit.conditions[0] if len(fit.conditions) == 1 else ""
    for component, rows in per_group.groupby("component"):
        _metric(
            reading,
            component,
            topo_condition,
            "topography_r",
            rows["pattern_r"].median(),
        )
        _metric(
            reading,
            component,
            topo_condition,
            "topography_r_min",
            rows["pattern_r"].min(),
        )
    if not set(SHIFT_COLUMNS) <= set(influence.columns):
        _logger.warning(
            f"[{fit.label}] {fit.prefix}corrca_influence.csv predates the R_w / R_b "
            "columns: no shift for this fit. Rerun its fit job to add them."
        )
        return
    reading.shift = per_group[
        ["participant", "group", "component", *SHIFT_COLUMNS]
    ].assign(entry=fit.entry, fit=fit.label)


def read_snr(
    reading: FitReading, n: int, args: argparse.Namespace, band: str | None
) -> None:
    """The SNR grid's condition-contrast p-values and its trial courses.

    A band without the 40 Hz bin (1-30 Hz) has no grid of its own; its filters are
    read on the broadband input instead, by the band-swap grid
    (``run_corrca_snr_grid.py --swap_band <band>``), when that has run.
    """
    fit = reading.fit
    if reading.maps is None:
        # The grid reads the store; a grid whose store is gone is from another fit.
        _logger.warning(
            f"[{fit.label}] no store, so no SNR rows (a grid left is stale)."
        )
        return
    bounds = _band_range(band)
    csv_path, signal = fit.snr_csv, "full"
    if bounds is not None and not bounds[0] <= ASSR_FREQ <= bounds[1]:
        if fit.swap_csv is None or not fit.swap_csv.is_file():
            _logger.info(
                f"[{fit.label}] {band} holds no {ASSR_FREQ:g} Hz and no band-swap "
                f"grid is at {fit.swap_csv}: no SNR rows (run_band_swap.pbs adds them)."
            )
            return
        csv_path, signal = fit.swap_csv, f"{BAND_SWAPPED_READING}_{band}"
        reading.snr_note = "its filters on broadband"
        _logger.info(
            f"[{fit.label}] {band} holds no {ASSR_FREQ:g} Hz: SNR rows from its "
            f"filters on the broadband input ({csv_path.name})."
        )
    grid = _read_csv(csv_path, "SNR grid", fit.label)
    if grid is None:
        return
    courses_path = csv_path.with_name(f"{csv_path.stem}__courses.npz")
    if not courses_path.is_file():
        _logger.warning(f"[{fit.label}] no SNR courses at {courses_path}.")
        return
    _participants, labels, times, courses = load_snr_courses(courses_path)
    if not np.isclose(times[0], -AssrEpoch.PRE_ONSET_S, atol=0.5 / ASSR_FREQ):
        _logger.warning(
            f"[{fit.label}] {csv_path.name} was cut from {times[0]:+.3f} s, not the "
            f"current {-AssrEpoch.PRE_ONSET_S:+.3f} s pre-onset window: a stale grid, "
            "skipped. Rerun the fit job's SNR grid."
        )
        return
    sources = [f"CC {k + 1}" for k in range(n)]
    contrast = grid[
        (grid["family"] == "contrast")
        & (grid["variant"] == signal)
        & (grid["selection"] == args.snr_selection)
    ]
    for window in args.snr_windows:
        rows = contrast[contrast["window"] == window].set_index("source")
        p = rows["p"].reindex(sources).to_numpy(dtype=float)
        for k, value in enumerate(p):
            _metric(reading, k + 1, "", f"p_snr_contrast__{window}", value)
        if window == args.snr_windows[0] and len(fit.conditions) == 2:
            reading.course_p = p
    by_condition = courses.get((signal, args.snr_selection))
    if by_condition is None:
        _logger.warning(f"[{fit.label}] no {signal}/{args.snr_selection} courses.")
        return
    keep = [labels.index(s) for s in sources if s in labels]
    reading.courses = {c: data[:, keep] for c, data in by_condition.items()}
    reading.course_times = times


def read_fit(
    fit: Fit, n: int, args: argparse.Namespace, band: str | None
) -> FitReading:
    reading = FitReading(fit)
    _logger.info(f"[{fit.label}] reading {fit.store_path.name} and {fit.plot_dir}")
    read_store(reading, n)
    read_isc_csvs(reading, n)
    read_influence(reading, n)
    read_snr(reading, n, args, band)
    return reading


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Through a temporary name, so a full disk never leaves a truncated table."""
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _metric_panels(args: argparse.Namespace) -> list[tuple[str, str, bool]]:
    return [
        ("fit_isc", "ISC of the fit (in-sample)", False),
        ("isc_whole", "ISC per condition, whole recording", False),
        ("isc_onset", "ISC per condition, onset-locked maps", False),
        ("heldout_isc", "Held-out ISC, LOO refit (median)", False),
        ("insample_minus_heldout", "In-sample − held-out ISC (median)", False),
        ("topography_r", "Topography |r|, full vs LOO refit (median)", False),
        ("topography_r_min", "Topography |r|, full vs LOO refit (worst)", False),
        ("within", "R_w term: variance per unit filter", False),
        ("between", "R_b term: shared covariance per unit filter", False),
        ("p_isc_whole", "p: LOO-ISC Psilocybin − Placebo, whole", True),
        ("p_isc_onset", "p: LOO-ISC Psilocybin − Placebo, onset", True),
        ("p_within", "p: R_w term Psilocybin − Placebo", True),
        ("p_between", "p: R_b term Psilocybin − Placebo", True),
        *(
            (
                f"p_snr_contrast__{w}",
                f"p: SNR Placebo > Psilocybin, {args.snr_selection} {w}",
                True,
            )
            for w in args.snr_windows
        ),
    ]


def _log_summary(metrics: pd.DataFrame, entries: list[str], n: int) -> None:
    wanted = ("fit_isc", "heldout_isc", "insample_minus_heldout", "topography_r")
    for entry in entries:
        rows = metrics[(metrics["entry"] == entry) & metrics["metric"].isin(wanted)]
        for k in range(1, n + 1):
            parts = [
                f"{r.metric}{'' if not r.condition else f' ({r.condition})'} {r.value:.4f}"
                for r in rows[rows["component"] == k].itertuples()
            ]
            if parts:
                _logger.info(f"[{entry}] CC {k}: " + ", ".join(parts))


def run(args: argparse.Namespace) -> None:
    """Read every compared fit and draw the comparison.

    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    """
    if args.n_pca is None:
        args.n_pca = list(DEFAULT_PCA_SWEEP) if args.compare == "pca" else [5]
    if args.compare != "pca" and len(args.n_pca) != 1:
        raise ValueError(
            f"--compare {args.compare} compares at one --n_pca; got {args.n_pca}."
        )
    n = min(args.components, MAX_DRAWN_COMPONENTS)
    if n < args.components:
        _logger.warning(f"--components {args.components}: comparing the first {n}.")
    entries = build_entries(args)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{args.band}_" if args.compare != "band" and args.band != BROADBAND else ""
    )
    _logger.info(
        f"CorrCA comparison ({args.compare}): {[e.label for e in entries]}, CC 1-{n}, "
        f"writing to {out_dir}"
    )
    readings = [read_fit(fit, n, args, e.band) for e in entries for fit in e.fits]
    labels = [e.label for e in entries]
    title = f"CorrCA, {args.compare} comparison — {args.music_type}"

    metrics = pd.DataFrame(
        [m for r in readings for m in r.metrics],
        columns=["entry", "fit", "component", "condition", "metric", "value"],
    )
    _write_csv(metrics, out_dir / f"{prefix}{STEM}_metrics.csv")
    _log_summary(metrics, labels, n)
    swapped = [r.fit.label for r in readings if r.snr_note]
    plot_comparison_metrics(
        metrics,
        _metric_panels(args),
        labels,
        n,
        connect=args.compare == "pca",
        alpha=args.alpha,
        title=f"{title}: synchrony, held-out stability and condition tests "
        "(uncorrected p)"
        + (
            f"\nSNR p of {', '.join(swapped)}: its filters on the broadband input "
            "(no 40 Hz bin of its own)"
            if swapped
            else ""
        ),
        save_path=out_dir / f"{prefix}{STEM}_metrics.png",
    )
    plt.close("all")

    stored = [r for r in readings if r.maps is not None]
    for kind, what in (
        ("filter", "shared filters W"),
        ("pattern", "shared forward patterns"),
    ):
        first = stored[0] if stored else None
        plot_comparison_topomaps(
            [r.maps[kind] for r in stored],
            [r.info for r in stored],
            [r.fit.label for r in stored],
            n,
            isc=[r.fit_isc for r in stored],
            reference_r=[
                None
                if r is first or r.channels != first.channels
                else matched_component_correlation(first.maps[kind], r.maps[kind])
                for r in stored
            ],
            title=f"{title}: {what}, each panel on its own scale",
            save_path=out_dir / f"{prefix}{STEM}_topomap_{kind}.png",
        )
        plt.close("all")

    # One row per (entry, condition), each entry's conditions on consecutive rows.
    marks = next((r.epoch_marks for r in readings if r.epoch_marks), None)
    for attr, suffix, what, epoch_marks in (
        ("tf_whole", "", "whole recording, participants equal-weighted", None),
        (
            "tf_onset",
            "_onset",
            "onset-locked, each trial in its pre-onset SD, median over trials",
            marks,
        ),
    ):
        tf_rows = [
            (r.fit.entry, c, mean, r.tf_freqs, times)
            for r in readings
            for c in args.conditions
            if c in getattr(r, attr)
            for mean, times in [getattr(r, attr)[c]]
        ]
        plot_comparison_tf_maps(
            tf_rows,
            n,
            title=f"{title}: condition-mean TF maps, {what}; one colour scale per "
            "fit and component",
            freq_marks=[ASSR_FREQ],
            epoch_marks=epoch_marks,
            save_path=out_dir / f"{prefix}{STEM}_tf_maps{suffix}.png",
        )
        plt.close("all")

    # One row per entry: an individual-conditions entry joins its two fits' courses.
    course_rows, course_p, course_labels, course_times = [], [], [], []
    for entry in entries:
        merged, p, note, times = {}, None, "", None
        for r in readings:
            if r.fit.entry == entry.label and r.courses:
                merged.update(r.courses)
                times = r.course_times
                p = r.course_p if r.course_p is not None else p
                note = r.snr_note or note
        if merged:
            course_rows.append(merged)
            course_times.append(times)
            course_p.append(p)
            course_labels.append(f"{entry.label} ({note})" if note else entry.label)
    if course_rows:
        plot_comparison_courses(
            course_rows,
            course_times,
            course_labels,
            n,
            window=_window_bounds(args.snr_windows[0]),
            p_values=course_p,
            title=f"{title}: trial-normalised SNR course, {args.snr_selection} "
            f"(shaded: the contrast window {args.snr_windows[0]})",
            save_path=out_dir / f"{prefix}{STEM}_snr_course.png",
        )
        plt.close("all")

    variance = [r.variance for r in readings if r.variance is not None]
    if variance:
        per_participant = pd.concat(variance, ignore_index=True)
        tests = pd.concat(
            [r.variance_tests for r in readings if r.variance_tests is not None]
            or [pd.DataFrame(columns=["entry", "component", "term", "p_value"])],
            ignore_index=True,
        )
        _write_csv(per_participant, out_dir / f"{prefix}{STEM}_variance_terms.csv")
        _write_csv(tests, out_dir / f"{prefix}{STEM}_variance_terms_tests.csv")
        plot_comparison_variance_terms(
            per_participant,
            tests,
            labels,
            n,
            title=f"{title}: the two terms of each ISC per condition (ISC = R_b / R_w; "
            "per unit filter norm, in units of one channel's variance)",
            save_path=out_dir / f"{prefix}{STEM}_variance_terms.png",
        )
        plt.close("all")

    shift = [r.shift for r in readings if r.shift is not None]
    if shift:
        shift_frame = pd.concat(shift, ignore_index=True)
        _write_csv(shift_frame, out_dir / f"{prefix}{STEM}_variance_shift.csv")
        plot_comparison_variance_shift(
            shift_frame,
            labels,
            n,
            title=f"{title}: how the ISC terms move with one participant left out",
            save_path=out_dir / f"{prefix}{STEM}_variance_shift.png",
        )
        plt.close("all")
    _logger.info(f"done: {out_dir}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set stage-06b CorrCA fits side by side across n_pca, bands or "
        "joins (read-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    what = parser.add_argument_group("what is compared")
    what.add_argument("--compare", choices=COMPARISONS, required=True)
    what.add_argument(
        "--run",
        choices=list(RUNS),
        default="tracks",
        help="The run compared across n_pca (pca) or bands (band).",
    )
    what.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUNS),
        default=["tracks", "separate", "placebo_fit"],
        help="The runs compared (join); the first is the topomaps' reference row.",
    )
    what.add_argument(
        "--band",
        default=BROADBAND,
        help=f"Store band token of the pca and join comparisons; '{BROADBAND}' is the "
        "store without one.",
    )
    what.add_argument(
        "--bands",
        nargs="+",
        default=[BROADBAND, "1-30hz", "30-50hz"],
        help="Store band tokens compared (band).",
    )
    what.add_argument(
        "--n_pca",
        nargs="+",
        type=int,
        default=None,
        help=f"The sweep (pca; default {' '.join(map(str, DEFAULT_PCA_SWEEP))}), or "
        "the one component count of the band and join comparisons (default 5).",
    )
    what.add_argument(
        "--components",
        type=int,
        default=3,
        help=f"Leading CCs compared (at most {MAX_DRAWN_COMPONENTS}).",
    )
    what.add_argument(
        "--conditions",
        nargs="+",
        default=[
            c.value for c in (ConditionVariants.PLACEBO, ConditionVariants.PSILOCYBIN)
        ],
    )
    what.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    what.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)

    snr = parser.add_argument_group("the SNR grid read")
    snr.add_argument("--snr_selection", default="40hz")
    snr.add_argument(
        "--snr_windows",
        nargs="+",
        default=["0-0.5s", "0.2-0.5s"],
        help="Grid windows whose contrast p is compared; the first is shaded in the "
        "course figure.",
    )
    snr.add_argument("--alpha", type=float, default=0.05)

    paths = parser.add_argument_group("where the fits are, and where to write")
    paths.add_argument(
        "--store_dir",
        type=Path,
        default=None,
        help="Processed-data root of the stores.",
    )
    paths.add_argument(
        "--fit_plots_dir",
        type=Path,
        default=None,
        help="Plot root the fits drew into (their CSVs are read there).",
    )
    paths.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Where the SNR grids wrote; default results/<experiment>.",
    )
    paths.add_argument(
        "--save_dir", type=Path, default=None, help="Plot root to write to."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging and run.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
