"""
CLI script reproducing Steps 6-11 of
``notebooks/06-iva-condition-comparison/iva_component_analysis_joined_tracks.ipynb``
on the **full** recording: the per-trial 40 Hz ASSR extraction through a stored
decomposition's spatial filters and the fixed ASSR-electrode mask, followed by the
participant-level paired tests and the summary figures.

**Why this exists rather than the notebook.** The notebook reads the stage-03 *subset*
caches, which are trimmed for interactive work — the ASSR one is 3000 samples, so only
9 of the 148 stimulus onsets have a whole epoch inside it. This reads the
source-of-truth cache under ``data/processed/<experiment>/wavelets/`` and gets every
trial. That cache is ~50 GB of DEFLATE-compressed float64 per condition, so it is
neither loadable nor memory-mappable; :func:`~src.io.loading.stream_wavelet_cache_subjects`
decompresses it once, subject by subject, keeping only the frequency bins this analysis
needs. Expect roughly 13 minutes per condition and a peak of a few hundred MB — the run
is I/O bound, not memory bound, which is why the job asks for modest resources.

What it does, in order:

1. Load the stored IVA components and invert the forward patterns back into the
   ``(components x channels)`` spatial filters. Append the checked-in ASSR-electrode
   mask as one more filter row, so both kinds of operator share one source axis.
2. Stream each condition's wavelet cache, projecting every participant's channels
   through their own filters at the wanted frequency bins only.
3. Cut onset-locked trials, using the onsets stored beside the concatenated array and
   the epoch geometry the paradigm defines.
4. Normalise every trial against its own pre-stimulus window, subtractively — a
   projected IVA source is a *signed* combination of channel powers, so any ratio
   against a baseline flips sign wherever that baseline is negative.
5. Anchor each participant's component polarity to the ASSR electrodes, so a higher
   value means more 40 Hz power over that area for everyone and a directional test can
   be stated at all.
6. Test, with participants as the unit and an exact Wilcoxon signed-rank:

   * **contrast** - is Psilocybin lower than Placebo, per source;
   * **discrimination** - does a component separate the conditions *better than the
     mask*, as an interaction, all components tested and none selected by its own
     effect size.

Outputs:

* ``data/processed/<experiment>/assr_trials/assr_trials__<variant>__<music>__<sel>__pca<N>.npz``
  - the trials with both normalisations and every axis needed to interpret them.
* ``results/<experiment>/assr_trial_stats__<sel>__pca<N>.csv`` - the test tables.
* ``plots/06-iva-condition-comparison/<Condition>_<Music>/broadband/assr_trial_stats/``
  - the time-course grid and the p-value summary.

Examples::

    # Full run, both frequency selections, figures and CSV
    python scripts/run_assr_trial_stats.py --experiment assr --n_pca 5

    # A quick shape check without decompressing the whole cache
    python scripts/run_assr_trial_stats.py --experiment assr --n_pca 5 \\
        --n_times 3000 --skip_plots

No correction for multiplicity is applied: the two families are 6 and 5 tests, and the
tables carry ``effect`` and ``same sign`` so a row can be read against its neighbours
rather than against a threshold alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import assr_trials as at  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
    PreprocessedDataVariants,
    SpectrumTypeVariants,
)
from src.io.iva_store import load_iva_components  # noqa: E402
from src.io.loading import (  # noqa: E402
    assr_electrode_mask,
    read_wavelet_cache_header,
    stream_wavelet_cache_subjects,
)

_logger = logging.getLogger(__name__)

#: Stage directory under ``plots/``, shared with the other stage-06 products.
_STAGE_DIR = "06-iva-condition-comparison"

#: Analysis-type subdirectory, so these figures never collide with the decomposition's.
_ANALYSIS_DIR = "assr_trial_stats"

#: Subdirectory of an experiment's processed data holding the extracted trials.
_TRIALS_DIR_NAME = "assr_trials"

#: Colour per condition, used by every panel.
_CONDITION_COLORS = {
    ConditionVariants.PLACEBO.value: "#0F6E8C",
    ConditionVariants.PSILOCYBIN.value: "#A6357F",
}

#: Colour of a marker whose test cleared alpha, and of one that did not.
_SIGNIFICANT = "#1B5E20"
_NULL = "0.45"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :return: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Per-trial 40 Hz ASSR extraction through stored IVA filters and the "
            "fixed electrode mask, plus the participant-level paired tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    stored = parser.add_argument_group("the stored decomposition to read")
    stored.add_argument(
        "--experiment",
        default=ExperimentNames.ASSR.value,
        choices=[e.value for e in ExperimentNames],
        help="Experiment whose component store and wavelet cache are read.",
    )
    stored.add_argument(
        "--variant",
        default=IvaVariants.CHANNEL_JOINED_TRACKS.value,
        choices=[v.value for v in IvaVariants],
        help="IVA variant the spatial filters come from.",
    )
    stored.add_argument(
        "--store_condition",
        default=ConditionVariants.JOINED_TRACKS.value,
        help="Condition subdirectory of the component store.",
    )
    stored.add_argument(
        "--music_type",
        default=MusicTypeVariants.ASSR.value,
        help="Music type the store was written under.",
    )
    stored.add_argument(
        "--band",
        default=None,
        help="Band the stored run was restricted to; omit for a broadband run.",
    )
    stored.add_argument(
        "--n_pca",
        type=int,
        default=5,
        help="The stored run's --n_pca, which is also its component count.",
    )
    stored.add_argument(
        "--store_dir",
        type=Path,
        default=None,
        help="Processed-data root the store resolves against.",
    )

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument(
        "--conditions",
        nargs=2,
        default=[c.value for c in REAL_CONDITIONS],
        help="The two conditions, in subtraction order (first minus second).",
    )
    data.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help="Processed-data root holding <experiment>/wavelets/.",
    )
    data.add_argument(
        "--n_times",
        type=int,
        default=None,
        help=(
            "Keep only the first N samples of each recording. Only for a fast shape "
            "check — it does NOT shorten the decompression, which is sequential."
        ),
    )
    data.add_argument(
        "--coordinate_system",
        default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value,
        help="Coordinate system whose ASSR electrode list to use.",
    )
    data.add_argument(
        "--mask_sum",
        action="store_true",
        help="Sum over the ASSR electrodes instead of averaging over them.",
    )
    data.add_argument(
        "--lenient_mask",
        action="store_true",
        help="Accept the intersection when a listed ASSR electrode is absent.",
    )

    analysis = parser.add_argument_group("the extraction and the tests")
    analysis.add_argument(
        "--center_freq",
        type=float,
        default=iva_quality.ASSR_FREQ,
        help="Centre frequency of the response, in Hz.",
    )
    analysis.add_argument(
        "--halfwidths",
        type=float,
        nargs="+",
        default=[0.0, iva_quality.TF_ANCHOR_HALFWIDTH_HZ],
        help=(
            "One extraction per half-width, in Hz. 0 takes the single nearest bin; a "
            "positive value averages every bin in the closed interval."
        ),
    )
    analysis.add_argument(
        "--test_selection",
        default=None,
        help=(
            "Which selection label the tests and figures run on. Defaults to the "
            "first --halfwidths entry. One only: testing both doubles every family "
            "for no new question."
        ),
    )
    analysis.add_argument(
        "--response_measure",
        default="stimulus",
        choices=["stimulus", "stimulus_minus_rest"],
        help=(
            "How a trial's time course becomes one number. Fix this before looking "
            "at any p-value."
        ),
    )
    analysis.add_argument(
        "--contrast_alternative",
        default="greater",
        choices=["greater", "less", "two-sided"],
        help=(
            "Direction of the condition contrast, computed as the first condition "
            "minus the second. 'greater' encodes 'psilocybin lowers the response'. "
            "Legitimate only if fixed in advance."
        ),
    )
    analysis.add_argument(
        "--no_polarity_anchor",
        action="store_true",
        help=(
            "Skip anchoring component polarity to the ASSR electrodes. The "
            "directional contrast is then NOT interpretable."
        ),
    )
    analysis.add_argument(
        "--alpha", type=float, default=0.05, help="Alpha for marking."
    )

    output = parser.add_argument_group("output")
    output.add_argument("--save_dir", type=Path, default=None, help="Plots root.")
    output.add_argument("--results_dir", type=Path, default=None, help="Results root.")
    output.add_argument(
        "--trials_dir",
        type=Path,
        default=None,
        help="Where the extracted trial files go.",
    )
    output.add_argument("--skip_plots", action="store_true", help="Write no figures.")
    output.add_argument(
        "--skip_trial_files",
        action="store_true",
        help="Run the tests without writing the trial arrays.",
    )
    output.add_argument("--log_level", default="INFO", help="Logging level.")
    return parser


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _concatenated_dir(experiment: ExperimentNames, root: Path | None) -> Path:
    """Directory holding the concatenated arrays and their sidecars.

    :param experiment: Experiment being analysed.
    :param root: Processed-data root, or ``None`` for the project default.
    :return: The directory.
    """
    base = ProjectPaths.PROCESSED_DATA_DIR if root is None else Path(root)
    return base / experiment.value / PreprocessedDataVariants.CONCATENATED.value


def _load_onsets(concat_dir: Path, label: str) -> np.ndarray:
    """Stimulus onsets stored beside a concatenated array.

    They are not in the component store — a run written before onsets were stored has
    none — but the concatenated array is what the wavelet transform was computed on, so
    its sidecar indexes exactly the time axis the projected tracks live on.

    :param concat_dir: Directory holding the concatenated products.
    :param label: ``<Condition>_<MusicType>`` product label.
    :return: Onset sample indices.
    :raises FileNotFoundError: If the sidecar is missing.
    """
    path = concat_dir / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(
            f"No stimulus onsets at {path}. They are written next to the concatenated "
            "array by the stimulus alignment; without them there are no trials to cut."
        )
    return np.load(path).astype(int)


def _cache_participants(concat_dir: Path, label: str) -> list[str]:
    """Participant label per subject index of a condition's wavelet cache.

    The cache's subject axis is in concatenation order and carries no labels, so the
    mapping comes from the metadata CSV written beside the concatenated array — which
    is authoritative, unlike re-deriving it from the dataset filters.

    :param concat_dir: Directory holding the concatenated products.
    :param label: ``<Condition>_<MusicType>`` product label.
    :return: Participant labels in subject order.
    :raises FileNotFoundError: If the metadata CSV is missing.
    :raises ValueError: If it lacks the subject-index column.
    """
    path = concat_dir / f"{label}.metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"No concatenation metadata at {path}.")
    frame = pd.read_csv(path, index_col=0)

    index_column = "SingleDataMetadata.CONCATENATED_PERSON_INDEX"
    id_column = "SingleDataMetadata.PARTICIPANT_ID"
    missing = [c for c in (index_column, id_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} has no {missing} column(s).")

    by_index = dict(zip(frame[index_column], frame[id_column].astype(str)))
    return [
        "".join(ch for ch in by_index[i] if ch.isdigit())[-3:].zfill(3)
        for i in sorted(by_index)
    ]


def _wavelet_cache_path(
    root: Path | None, experiment: ExperimentNames, label: str
) -> Path:
    """The source-of-truth wavelet cache of one condition.

    :param root: Processed-data root, or ``None`` for the project default.
    :param experiment: Experiment being analysed.
    :param label: ``<Condition>_<MusicType>`` product label.
    :return: Path to the cache.
    :raises FileNotFoundError: If no cache matches.
    """
    base = ProjectPaths.PROCESSED_DATA_DIR if root is None else Path(root)
    directory = (
        base / experiment.value / "wavelets" / SpectrumTypeVariants.BROADBAND.value
    )
    matches = sorted(directory.glob(f"{label}__wavelet_power__*__freqdim1.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No wavelet cache for {label} in {directory}. Build it with "
            "jobs/metacentrum/03-wavelet-analysis/store_assr_wavelets.pbs first."
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_condition(
    cache_path: Path,
    filters: np.ndarray,
    freq_indices: np.ndarray,
    cache_labels: list[str],
    participants: list[str],
    store_channels: list[str],
    n_times: int | None,
    label: str,
) -> np.ndarray:
    """Stream one condition's cache and project every wanted participant.

    Only the participants the store has a filter for are projected; the rest are
    decompressed and discarded, because a DEFLATE stream cannot be seeked.

    :param cache_path: The condition's wavelet cache.
    :param filters: ``(participants, sources, channels)`` stacked filters, in
        *participants* order.
    :param freq_indices: Frequency bins to keep.
    :param cache_labels: Participant label per subject index of the cache.
    :param participants: Participants to project, in output-row order.
    :param store_channels: The stored run's channel names, which the filters index.
    :param n_times: Optional trim of the time axis.
    :param label: Product label, for logging.
    :return: ``(participants, sources, freqs, times)``.
    :raises ValueError: If the cache's channel axis does not match the stored run's.
    """
    header = read_wavelet_cache_header(cache_path)
    if header.channel_names != store_channels:
        raise ValueError(
            f"{cache_path.name}: the cache's channel axis does not match the stored "
            f"run's.\n  cache: {header.channel_names[:5]}...\n"
            f"  store: {store_channels[:5]}...\n"
            "The spatial filters are indexed by the stored channels, and dropping "
            "columns from a filter makes a different operator rather than restricting "
            "it."
        )

    wanted = {name: row for row, name in enumerate(participants)}
    keep_times = header.n_times if n_times is None else min(n_times, header.n_times)
    projected = np.empty(
        (len(participants), filters.shape[1], freq_indices.size, keep_times),
        dtype=np.float64,
    )
    seen = np.zeros(len(participants), dtype=bool)

    _logger.info(
        f"[{label}] streaming {cache_path.name} "
        f"({cache_path.stat().st_size / 1e9:.1f} GB, {header.n_subjects} subjects, "
        f"{freq_indices.size}/{header.n_freqs} frequency bins kept)"
    )
    started = time.time()
    for subject, block in stream_wavelet_cache_subjects(
        cache_path, freq_indices=freq_indices, n_times=keep_times, header=header
    ):
        name = cache_labels[subject] if subject < len(cache_labels) else None
        if name is None or name not in wanted:
            continue
        row = wanted[name]
        # (S, C) x (C, F, T) -> (S, F, T): contracts channels only, so frequency and
        # time pass through untouched.
        projected[row] = at.project_channels(filters[row], block)
        seen[row] = True
        _logger.info(
            f"[{label}]   participant {name} "
            f"({int(seen.sum())}/{len(participants)}, {time.time() - started:.0f}s)"
        )

    if not seen.all():
        absent = [p for p, ok in zip(participants, seen) if not ok]
        raise ValueError(
            f"[{label}] participants {absent} never appeared in the cache."
        )
    _logger.info(f"[{label}] projected in {time.time() - started:.0f}s")
    return projected


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_time_courses(
    course: dict[str, np.ndarray],
    times: np.ndarray,
    labels: list[str],
    contrast: pd.DataFrame,
    conditions: list[str],
    alpha: float,
    title: str,
    path: Path,
) -> None:
    """Grid of per-source time courses, both conditions overlaid.

    The band is the spread **across participants**, never across trials: trials within
    a participant are correlated, so a band drawn from them would look tight while
    saying nothing about how well the effect generalises to a new person.

    :param course: Per condition, ``(participants, sources, samples)``.
    :param times: Epoch time base in seconds.
    :param labels: Source label per row of the source axis.
    :param contrast: The 4b table, indexed by source.
    :param conditions: Condition order.
    :param alpha: Threshold for emphasising a panel's annotation.
    :param title: Figure suptitle.
    :param path: Destination file.
    """
    n_sources = len(labels)
    ncols = min(3, n_sources)
    nrows = int(np.ceil(n_sources / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.7 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_source = contrast.set_index("source")

    for s, source in enumerate(labels):
        ax = axes[s // ncols][s % ncols]
        ax.axvspan(
            0.0, AssrEpoch.STIMULUS_DURATION_S, color="0.55", alpha=0.11, lw=0, zorder=0
        )
        ax.axhline(0.0, color="0.45", lw=0.8, zorder=1)
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
        for condition in conditions:
            values = course[condition][:, s]
            centre = values.mean(axis=0)
            half = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            colour = _CONDITION_COLORS.get(condition, "0.3")
            ax.fill_between(
                times,
                centre - half,
                centre + half,
                color=colour,
                alpha=0.18,
                lw=0,
                zorder=2,
            )
            ax.plot(times, centre, color=colour, lw=1.9, zorder=3, label=condition)

        is_reference = source == at.BINARY_FILTER_LABEL
        row = by_source.loc[source]
        ax.set_title(
            f"{source}{'  (reference)' if is_reference else ''}",
            fontsize=12,
            fontweight="bold" if is_reference else "normal",
            loc="left",
        )
        ax.text(
            0.985,
            0.955,
            f"{conditions[0]} > {conditions[1]}\np = {row['p']:.3f}   {row['same sign']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            color="0.25" if row["p"] > alpha else "black",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor="0.75" if row["p"] > alpha else "black",
                alpha=0.88,
                lw=1.2 if row["p"] <= alpha else 0.8,
            ),
        )
        if s // ncols == nrows - 1:
            ax.set_xlabel("Time from stimulus onset (s)")
        if s % ncols == 0:
            ax.set_ylabel("40 Hz power\n(pre-stimulus SD)")

    for extra in range(n_sources, nrows * ncols):
        axes[extra // ncols][extra % ncols].axis("off")
    axes[0][0].legend(loc="lower right", frameon=True, fontsize=10)
    fig.suptitle(title, y=1.0, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"saved {path}")


def _forest(
    ax,
    rows: list[tuple[str, np.ndarray, tuple[float, float], pd.Series]],
    alpha: float,
    xlabel: str,
    title: str,
    reference_label: str | None,
) -> None:
    """One forest panel: per-participant points, a bootstrap interval and the p.

    The x range is set by the estimates and their intervals rather than by the raw
    scatter, because one outlying participant otherwise stretches the axis and squashes
    every interval; points outside are drawn as carets on the edge so none is silently
    dropped.

    :param ax: Axes to draw on.
    :param rows: ``(label, per-participant values, interval, test row)`` per row.
    :param alpha: Threshold for emphasis.
    :param xlabel: Axis label.
    :param title: Panel title.
    :param reference_label: Row to mark as the fixed reference, if any.
    """
    edges = [bound for _lab, _d, ci, _row in rows for bound in ci]
    edges += [float(row["median"]) for _lab, _d, _ci, row in rows]
    edges += list(
        np.percentile(np.concatenate([d for _lab, d, _ci, _row in rows]), [8, 92])
    )
    span = max(edges) - min(edges)
    lo, hi = min(edges) - 0.16 * span, max(edges) + 0.16 * span

    off_axis = 0
    for i, (label, values, (low, high), row) in enumerate(rows):
        y = len(rows) - 1 - i
        significant = row["p"] <= alpha
        colour = _SIGNIFICANT if significant else _NULL
        inside = (values >= lo) & (values <= hi)
        ax.scatter(
            values[inside],
            np.full(int(inside.sum()), y),
            s=13,
            color=colour,
            alpha=0.3,
            zorder=2,
            lw=0,
        )
        for value in values[~inside]:
            off_axis += 1
            ax.scatter(
                [hi if value > hi else lo],
                [y],
                s=26,
                color=colour,
                alpha=0.55,
                marker=">" if value > hi else "<",
                zorder=2,
                lw=0,
            )
        ax.plot(
            [low, high], [y, y], color=colour, lw=2.4, zorder=3, solid_capstyle="round"
        )
        ax.scatter(
            [row["median"]],
            [y],
            s=95,
            color=colour,
            zorder=4,
            marker="D" if label == reference_label else "o",
            edgecolor="white",
            linewidth=1.1,
        )
        ax.text(
            1.005,
            y,
            f"p={row['p']:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9.5,
            family="monospace",
            fontweight="bold" if significant else "normal",
            color=colour,
        )

    if off_axis:
        ax.text(
            0.5,
            -0.205,
            f"{off_axis} participant point(s) beyond the axis, drawn as carets on the edge",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="0.45",
            style="italic",
        )
    ax.set_xlim(lo, hi)
    ax.axvline(0.0, color="0.3", lw=1.0 if reference_label else 1.6, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _d, _ci, _row in reversed(rows)])
    for tick, (label, *_rest) in zip(ax.get_yticklabels(), reversed(rows)):
        if label == reference_label:
            tick.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=12)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Run the whole extraction and test pipeline for one stored decomposition.

    :param args: Parsed CLI arguments.
    """
    experiment = ExperimentNames(args.experiment)
    variant = IvaVariants(args.variant)
    store_condition = ConditionVariants(args.store_condition)
    music_type = MusicTypeVariants(args.music_type)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    conditions = at.resolve_conditions(args.conditions)

    # ---- 1. the stored decomposition and its filters --------------------
    results = load_iva_components(
        experiment=experiment,
        condition=store_condition,
        variant=variant,
        music_type=music_type,
        band=args.band,
        n_pca=args.n_pca,
        processed_data_dir=args.store_dir,
    )
    _logger.info(f"loaded {results.path}")

    store_channels = list(results.channel_names)
    spatial_filters = at.recover_spatial_filters(results.channel_patterns)
    electrode_mask = assr_electrode_mask(
        store_channels, coordinate_system, strict=not args.lenient_mask
    )
    binary = at.binary_filter_weights(electrode_mask, normalize=not args.mask_sum)
    filters = at.stack_filters(spatial_filters, binary)
    labels = at.source_labels(results.n_components)
    binary_channels = [n for n, keep in zip(store_channels, electrode_mask) if keep]
    _logger.info(
        f"filters {filters.shape}; {int(electrode_mask.sum())} ASSR electrode(s) "
        f"({'sum' if args.mask_sum else 'mean'})"
    )

    # ---- 2. frequency selections ----------------------------------------
    selections = {
        at.selection_label(args.center_freq, half): at.frequency_selection(
            results.freqs, args.center_freq, half
        )
        for half in args.halfwidths
    }
    union = np.unique(np.concatenate(list(selections.values())))
    for name, bins in selections.items():
        _logger.info(
            f"selection {name}: bins {bins.tolist()} = {results.freqs[bins]} Hz"
        )

    test_selection = args.test_selection or next(iter(selections))
    if test_selection not in selections:
        raise ValueError(
            f"--test_selection {test_selection!r} is not among {list(selections)}."
        )

    # ---- 3. project each condition from the source cache ----------------
    concat_dir = _concatenated_dir(experiment, args.wavelet_data_dir)
    store_participants = list(results.participants)
    onsets, cache_labels, cache_paths = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        cache_labels[condition] = _cache_participants(concat_dir, label)
        cache_paths[condition] = _wavelet_cache_path(
            args.wavelet_data_dir, experiment, label
        )

    participants = [
        p for p in store_participants if all(p in cache_labels[c] for c in conditions)
    ]
    if not participants:
        raise ValueError("No stored participant appears in every condition's cache.")
    dropped = [p for p in store_participants if p not in participants]
    if dropped:
        _logger.warning(f"not in every cache, dropped: {dropped}")

    rows = [store_participants.index(p) for p in participants]
    filters = filters[rows]
    flip, strength = at.polarity_flip(results.channel_patterns[rows], electrode_mask)
    determined = at.polarity_is_determined(strength)

    projected = {
        condition: _project_condition(
            cache_paths[condition],
            filters,
            union,
            cache_labels[condition],
            participants,
            store_channels,
            args.n_times,
            condition,
        )
        for condition in conditions
    }

    # ---- 4. cut trials, per selection ------------------------------------
    sfreq = results.sfreq
    geometry = {
        condition: at.epoch_geometry(
            onsets[condition], projected[condition].shape[-1], sfreq
        )
        for condition in conditions
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)
    _logger.info(
        f"epoch {pre} pre + {post} post = {pre + post} samples = "
        f"[{times[0]:.3f}, {times[-1]:.3f}] s"
    )

    union_position = {int(b): i for i, b in enumerate(union)}
    trial_sets: dict[str, at.AssrTrialSet] = {}
    for name, bins in selections.items():
        take = [union_position[int(b)] for b in bins]
        trials, trials_z, trials_rel, positive, kept = {}, {}, {}, {}, {}
        for condition in conditions:
            band = projected[condition][:, :, take, :].mean(axis=2)  # (P, S, T)
            cut, onsets_kept = at.cut_trials(band, geometry[condition][0], pre, post)
            trials[condition] = cut
            trials_z[condition], trials_rel[condition], positive[condition] = (
                at.baseline_normalise(cut, times < 0.0)
            )
            kept[condition] = onsets_kept
            _logger.info(
                f"[{condition}] {name}: {cut.shape} — "
                f"{onsets_kept.size} of {onsets[condition].size} onset(s) kept"
            )
        trial_sets[name] = at.AssrTrialSet(
            trials=trials,
            trials_z=trials_z,
            trials_rel=trials_rel,
            baseline_positive=positive,
            onsets=kept,
            participants=tuple(participants),
            labels=tuple(labels),
            times=times,
            sfreq=sfreq,
            freqs=results.freqs[bins],
            selection=name,
            binary_channels=tuple(binary_channels),
            metadata={
                "store_path": str(results.path),
                "variant": variant.value,
                "n_pca": str(args.n_pca),
                "source": "full wavelet cache",
                "signal_zscore": "none at extraction",
                "baseline_correction": "per trial, subtractive (see trials_z)",
                "units": "wavelet power",
                "binary_filter": "sum" if args.mask_sum else "mean",
            },
        )

    # ---- 5. write the trial files ----------------------------------------
    trials_root = (
        Path(args.trials_dir)
        if args.trials_dir
        else ProjectPaths.PROCESSED_DATA_DIR / experiment.value / _TRIALS_DIR_NAME
    )
    if not args.skip_trial_files:
        for name, trial_set in trial_sets.items():
            path = at.save_assr_trials(
                trials_root
                / at.trials_filename(variant.value, music_type.value, name, args.n_pca),
                trial_set,
            )
            _logger.info(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB)")

    # ---- 6. collapse and test --------------------------------------------
    tested = trial_sets[test_selection]
    stimulus = tested.stimulus_mask()

    def _reduce(values: np.ndarray) -> np.ndarray:
        """Time axis to one number per trial."""
        if args.response_measure == "stimulus":
            return values[..., stimulus].mean(axis=-1)
        return values[..., stimulus].mean(axis=-1) - values[..., ~stimulus].mean(
            axis=-1
        )

    anchor = (
        np.concatenate([flip, np.ones((len(participants), 1))], axis=1)
        if not args.no_polarity_anchor
        else np.ones((len(participants), len(labels)))
    )
    value = {
        condition: np.median(_reduce(tested.trials_z[condition]), axis=2) * anchor
        for condition in conditions
    }
    n_participants = len(participants)

    _logger.info("polarity anchor over the ASSR electrodes:")
    for k in range(results.n_components):
        _logger.info(
            f"  {labels[k]}: flipped {int((flip[:, k] < 0).sum())}/{n_participants}, "
            f"median |w_mask|/|w_all| {np.median(strength[:, k]):.3f} "
            f"({'well determined' if determined[k] else 'WEAK'})"
        )

    reference = labels.index(at.BINARY_FILTER_LABEL)
    contrast = pd.DataFrame(
        [
            {
                "family": "contrast",
                "source": source,
                **at.paired_test(
                    at.condition_contrast(value, conditions, s),
                    args.contrast_alternative,
                ),
            }
            for s, source in enumerate(labels)
        ]
    )
    discrimination = pd.DataFrame(
        [
            {
                "family": "discrimination",
                "source": source,
                "IC contrast": float(
                    np.median(at.condition_contrast(value, conditions, s))
                ),
                # Two-sided: nothing predicts that a learned component should beat a
                # fixed electrode selection.
                **at.paired_test(
                    at.discrimination_gain(value, conditions, s, reference), "two-sided"
                ),
            }
            for s, source in enumerate(labels)
            if s != reference
        ]
    )

    print(f"\n=== {conditions[0]} minus {conditions[1]}, n = {n_participants} ===")
    print(contrast.drop(columns="family").to_string(index=False))
    print(
        f"\n=== does a component separate the conditions better than "
        f"{at.BINARY_FILTER_LABEL}? ==="
    )
    print(discrimination.drop(columns="family").to_string(index=False))
    print(
        f"\nExact Wilcoxon, participants are the unit. Floor p = "
        f"{at.p_floor(n_participants):.5f} two-sided, "
        f"{at.p_floor(n_participants, one_sided=True):.5f} one-sided. "
        f"Uncorrected over {len(contrast) + len(discrimination)} tests."
    )

    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / f"assr_trial_stats__{test_selection}__pca{args.n_pca}.csv"
    pd.concat([contrast, discrimination]).to_csv(csv_path, index=False)
    _logger.info(f"saved {csv_path}")

    if args.skip_plots:
        return

    # ---- 7. figures -------------------------------------------------------
    plots_dir = (
        (Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH)
        / _STAGE_DIR
        / f"{store_condition.value}_{music_type.value}"
        / SpectrumTypeVariants.BROADBAND.value
        / _ANALYSIS_DIR
        / f"pca_{args.n_pca}"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    course = {
        condition: np.median(tested.trials_z[condition], axis=2) * anchor[:, :, None]
        for condition in conditions
    }
    n_trials = tested.onsets[conditions[0]].size
    _plot_time_courses(
        course,
        times,
        labels,
        contrast,
        conditions,
        args.alpha,
        f"40 Hz response per spatial filter — mean +/- SEM across participants, "
        f"{n_participants} participants x {n_trials} trials "
        f"({test_selection}, polarity-anchored to the ASSR electrodes)",
        plots_dir / f"trial_course_by_source_{test_selection}.png",
    )

    rng = np.random.default_rng(42)

    def _ci(values: np.ndarray) -> tuple[float, float]:
        """Percentile bootstrap interval for the median, over participants."""
        finite = values[np.isfinite(values)]
        draws = np.median(
            finite[rng.integers(0, finite.size, size=(10_000, finite.size))], axis=1
        )
        return tuple(
            np.percentile(draws, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)])
        )

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    contrast_rows = [
        (
            source,
            at.condition_contrast(value, conditions, s),
            _ci(at.condition_contrast(value, conditions, s)),
            contrast.set_index("source").loc[source],
        )
        for s, source in enumerate(labels)
    ]
    _forest(
        left,
        contrast_rows,
        args.alpha,
        f"{conditions[0]} - {conditions[1]}  (pre-stimulus SD)",
        f"Is {conditions[1]} lower than {conditions[0]}?\n"
        f"{args.contrast_alternative}, positive = predicted direction",
        at.BINARY_FILTER_LABEL,
    )

    gain_rows = [
        (
            source,
            at.discrimination_gain(value, conditions, labels.index(source), reference),
            _ci(
                at.discrimination_gain(
                    value, conditions, labels.index(source), reference
                )
            ),
            discrimination.set_index("source").loc[source],
        )
        for source in labels
        if source != at.BINARY_FILTER_LABEL
    ]
    _forest(
        right,
        gain_rows,
        args.alpha,
        f"(IC contrast) - ({at.BINARY_FILTER_LABEL} contrast)   (pre-stimulus SD)",
        f"Does any component separate the conditions better than "
        f"{at.BINARY_FILTER_LABEL}?\ntwo-sided; 0 = exactly as good as the fixed "
        "electrode selection",
        None,
    )

    better = int((discrimination["median"] > 0).sum())
    fig.suptitle(
        f"Exact Wilcoxon signed-rank, n = {n_participants} participants "
        f"(floor p = {at.p_floor(n_participants):.5f} two-sided) — uncorrected over "
        f"{len(contrast) + len(discrimination)} tests; "
        f"{better}/{len(discrimination)} IC(s) out-separate the reference",
        y=1.0,
        fontsize=12.5,
    )
    fig.tight_layout()
    summary_path = plots_dir / f"pvalue_summary_{test_selection}.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"saved {summary_path}")


def main() -> None:
    """Parse arguments and run."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
