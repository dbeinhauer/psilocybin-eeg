"""
Full-recording ASSR SNR / condition-contrast grid, for either IVA condition-comparison
variant.

This is the CLI counterpart of
``notebooks/06-iva-condition-comparison/iva_component_analysis_joined.ipynb`` and its
``_tracks`` sibling, run on the **full** wavelet cache rather than the 12 s notebook
subset. It reproduces the whole current workflow — both ASSR references, both signal
variants, and all three test families — and sweeps the frequency-selection x
stimulus-window grid in one pass, which is exactly what a job wants:

* **Two references.** ``ASSR-mask (full)`` is the fixed fronto-central electrode average
  on the whole channel space; ``ASSR-mask (PCA)`` is the same average read only through
  each recording's PCA subspace (``mask @ P^T P``), the apples-to-apples reference for
  the learned components.
* **Two signal variants.** ``zscored`` reads the stored per-recording z-scored sources
  for the IC rows (full recording, every onset) and the masks on the z-scored cache;
  ``prestim`` projects the raw cache and references every trial to its own pre-stimulus
  baseline.
* **Three test families**, participants as the unit, exact Wilcoxon:
  ``contrast`` (Placebo - Psilocybin per source), ``discrimination`` (does an IC separate
  the conditions better than a reference, the interaction, vs each reference) and
  ``snr`` (is an IC's response magnitude *lower* than a reference's, per condition).
* **The grid**: every ``--halfwidths`` x ``--stimulus_intervals`` combination, so a
  single run covers e.g. 40 Hz and 35-45 Hz crossed with 0-500 ms and 200-500 ms.

**Both decomposition variants.** ``--variant channel_joined_tracks`` shares one filter
per participant across the conditions; ``--variant channel_joined`` has a separate
topography per recording, so every projection and polarity anchor is resolved per
(participant, condition) through ``results.row``. Nothing else differs.

Outputs (per run, all grid cells in one file):

* ``results/<experiment>/assr_snr_grid__<variant>__pca<N>.csv`` — every test row, tagged
  with its signal variant, frequency selection and stimulus window.
* ``plots/06-iva-condition-comparison/<Condition>_<Music>/broadband/assr_snr_grid/
  pca_<N>/`` — a p-value summary and an SNR-by-condition figure per grid cell.

Examples::

    # The job grid: both frequencies x both windows, both variants of signal.
    python scripts/run_assr_snr_grid.py --experiment assr --variant channel_joined \\
        --n_pca 5 --halfwidths 0 5 --stimulus_intervals 0:0.5 0.2:0.5

    # A quick shape check without decompressing the whole cache.
    python scripts/run_assr_snr_grid.py --experiment assr --n_pca 5 --n_times 3000 \\
        --skip_plots

No multiplicity correction is applied; the CSV carries ``effect`` and ``same sign`` so a
row can be weighed against its neighbours rather than a threshold alone.
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

_STAGE_DIR = "06-iva-condition-comparison"
_ANALYSIS_DIR = "assr_snr_grid"
_FULL_LABEL = "ASSR-mask (full)"
_PCA_LABEL = "ASSR-mask (PCA)"
_MASK_LABELS = (_FULL_LABEL, _PCA_LABEL)
_SIGNAL_VARIANTS = ("zscored", "prestim")
_CONDITION_COLORS = {
    ConditionVariants.PLACEBO.value: "#0F6E8C",
    ConditionVariants.PSILOCYBIN.value: "#A6357F",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _stimulus_interval(text: str) -> tuple[float, float]:
    """Parse ``"lo:hi"`` seconds into a ``(lo, hi)`` window."""
    try:
        lo, hi = (float(x) for x in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"stimulus interval must be 'lo:hi' seconds; got {text!r}."
        ) from exc
    if not hi > lo:
        raise argparse.ArgumentTypeError(f"need hi > lo; got {text!r}.")
    return lo, hi


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Full-recording ASSR SNR / contrast grid for an IVA variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stored = parser.add_argument_group("the stored decomposition to read")
    stored.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    stored.add_argument(
        "--variant",
        default=IvaVariants.CHANNEL_JOINED.value,
        choices=[
            IvaVariants.CHANNEL_JOINED.value,
            IvaVariants.CHANNEL_JOINED_TRACKS.value,
        ],
        help="channel_joined = per-recording topography; _tracks = shared per participant.",
    )
    stored.add_argument(
        "--store_condition",
        default=None,
        help="Store condition; defaults per variant (Joined / JoinedTracks).",
    )
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument("--band", default=None)
    stored.add_argument("--n_pca", type=int, default=5)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument("--wavelet_data_dir", type=Path, default=None)
    data.add_argument(
        "--coordinate_system", default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value
    )
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Two conditions, contrast order. Default Placebo Psilocybin.",
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
        "--signal_variants",
        nargs="+",
        default=list(_SIGNAL_VARIANTS),
        choices=list(_SIGNAL_VARIANTS),
        help="Which normalisations to test side by side.",
    )
    analysis.add_argument(
        "--contrast_alternative",
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="Direction of Placebo - Psilocybin (4b). 'greater' = psilocybin lowers it.",
    )
    analysis.add_argument(
        "--snr_alternative",
        default="less",
        choices=["less", "greater", "two-sided"],
        help="Direction of (IC SNR - reference SNR) in the SNR family.",
    )
    analysis.add_argument("--no_polarity_anchor", action="store_true")
    analysis.add_argument("--alpha", type=float, default=0.05)

    output = parser.add_argument_group("output")
    output.add_argument("--save_dir", type=Path, default=None)
    output.add_argument("--results_dir", type=Path, default=None)
    output.add_argument("--skip_plots", action="store_true")
    output.add_argument("--log_level", default="INFO")
    return parser


# ---------------------------------------------------------------------------
# Inputs (mirrored from run_assr_trial_stats.py — same cache/onset conventions)
# ---------------------------------------------------------------------------


def _concatenated_dir(experiment: ExperimentNames, root: Path | None) -> Path:
    base = ProjectPaths.PROCESSED_DATA_DIR if root is None else Path(root)
    return base / experiment.value / PreprocessedDataVariants.CONCATENATED.value


def _load_onsets(concat_dir: Path, label: str) -> np.ndarray:
    path = concat_dir / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(
            f"No stimulus onsets at {path}; the stimulus alignment writes them next to "
            "the concatenated array."
        )
    return np.load(path).astype(int)


def _cache_participants(concat_dir: Path, label: str) -> list[str]:
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


def _default_store_condition(variant: IvaVariants) -> ConditionVariants:
    """The store condition each variant was written under."""
    if variant == IvaVariants.CHANNEL_JOINED_TRACKS:
        return ConditionVariants.JOINED_TRACKS
    return ConditionVariants.JOINED


def _zscore_time(block: np.ndarray) -> np.ndarray:
    """Z-score a ``(..., time)`` block along its last axis (per channel/frequency)."""
    mean = block.mean(axis=-1, keepdims=True)
    std = block.std(axis=-1, keepdims=True)
    return (block - mean) / np.where(std == 0, 1.0, std)


# ---------------------------------------------------------------------------
# Row bookkeeping — the one thing the two variants differ on
# ---------------------------------------------------------------------------


def _row_resolver(results, variant: IvaVariants):
    """A ``(participant, condition) -> store row`` function for either variant.

    The subject-axis join (``channel_joined``) has one row per recording, so a filter is
    condition-specific and ``results.row(p, c)`` picks it. The time-axis join
    (``channel_joined_tracks``) has one row per participant, shared by both conditions,
    so the condition is ignored.
    """
    per_recording = variant == IvaVariants.CHANNEL_JOINED

    def resolve(participant: str, condition: str) -> int:
        if per_recording:
            return results.row(participant, condition)
        return results.row(participant)

    return resolve, per_recording


# ---------------------------------------------------------------------------
# Projection — one streaming pass produces both signal variants' inputs
# ---------------------------------------------------------------------------


def _project_condition(
    cache_path: Path,
    condition: str,
    participants: list[str],
    resolve,
    filters: np.ndarray,
    binary: np.ndarray,
    pca_rows: np.ndarray,
    freq_indices: np.ndarray,
    cache_labels: list[str],
    store_channels: list[str],
    n_times: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream one condition once; return the prestim and z-scored-mask projections.

    :return: ``(prestim, zmask)`` where ``prestim`` is
        ``(participants, components + 2, freqs, times)`` (IC filters, then the full and
        PCA masks, on the RAW cache) and ``zmask`` is ``(participants, 2, freqs, times)``
        (the two masks on the z-scored cache). The z-scored IC rows are read straight
        from the store, so they are not produced here.
    """
    header = read_wavelet_cache_header(cache_path)
    if header.channel_names != store_channels:
        raise ValueError(
            f"{cache_path.name}: channel axis does not match the stored run's; the "
            "filters are indexed by the stored channels."
        )
    keep_times = header.n_times if n_times is None else min(n_times, header.n_times)
    wanted = {name: row for row, name in enumerate(participants)}
    n_comp = filters.shape[1]
    prestim = np.empty(
        (len(participants), n_comp + 2, freq_indices.size, keep_times), np.float32
    )
    zmask = np.empty((len(participants), 2, freq_indices.size, keep_times), np.float32)
    seen = np.zeros(len(participants), bool)

    _logger.info(
        f"[{condition}] streaming {cache_path.name} "
        f"({cache_path.stat().st_size / 1e9:.1f} GB, {header.n_subjects} subjects, "
        f"{freq_indices.size}/{header.n_freqs} bins kept)"
    )
    started = time.time()
    for subject, block in stream_wavelet_cache_subjects(
        cache_path, freq_indices=freq_indices, n_times=keep_times, header=header
    ):
        name = cache_labels[subject] if subject < len(cache_labels) else None
        if name is None or name not in wanted:
            continue
        row = wanted[name]
        store = resolve(name, condition)
        # per-recording spatial operators for this (participant, condition)
        full_row = binary[None, :]
        pca_row = pca_rows[store][None, :]
        stacked = np.concatenate(
            [filters[store], full_row, pca_row], axis=0
        )  # (K+2, C)
        prestim[row] = at.project_channels(stacked, block)
        zmask[row] = at.project_channels(
            np.concatenate([full_row, pca_row], axis=0), _zscore_time(block)
        )
        seen[row] = True
        _logger.info(
            f"[{condition}]   {name} ({int(seen.sum())}/{len(participants)}, "
            f"{time.time() - started:.0f}s)"
        )
    if not seen.all():
        absent = [p for p, ok in zip(participants, seen) if not ok]
        raise ValueError(
            f"[{condition}] participants {absent} never appeared in the cache."
        )
    _logger.info(f"[{condition}] projected in {time.time() - started:.0f}s")
    return prestim, zmask


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _bootstrap_ci(values: np.ndarray, alpha: float, rng) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (np.nan, np.nan)
    draws = np.median(
        finite[rng.integers(0, finite.size, size=(10_000, finite.size))], axis=1
    )
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def _forest(ax, rows, alpha, xlabel, title, rng):
    """A minimal forest: (label, per-participant diffs, table row) tuples."""
    medians = [r[2]["median"] for r in rows]
    cis = [_bootstrap_ci(r[1], alpha, rng) for r in rows]
    edges = [b for ci in cis for b in ci] + medians
    span = (np.nanmax(edges) - np.nanmin(edges)) or 1.0
    lo, hi = np.nanmin(edges) - 0.18 * span, np.nanmax(edges) + 0.22 * span
    for i, (label, diffs, row) in enumerate(rows):
        y = len(rows) - 1 - i
        sig = row["p"] <= alpha
        colour = "#1B5E20" if sig else "0.45"
        low, high = cis[i]
        ax.plot(
            [low, high], [y, y], color=colour, lw=2.3, solid_capstyle="round", zorder=3
        )
        ax.scatter(
            [row["median"]], [y], s=80, color=colour, edgecolor="white", zorder=4
        )
        ax.text(
            0.99,
            y,
            f"p={row['p']:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=8.5,
            family="monospace",
            fontweight="bold" if sig else "normal",
            color=colour,
        )
    ax.set_xlim(lo, hi)
    ax.axvline(0.0, color="0.3", lw=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)


def _plot_cell(cell_tables, value, conditions, labels, ic_labels, alpha, title, path):
    """The p-value summary (contrast + one versus panel per reference) for one grid cell."""
    rng = np.random.default_rng(42)
    contrast = cell_tables["contrast"].set_index("source")
    fig, axes = plt.subplots(
        1, 1 + len(_MASK_LABELS), figsize=(7.6 + 5.6 * len(_MASK_LABELS), 5.0)
    )
    rows = [
        (
            src,
            at.condition_contrast(value, conditions, labels.index(src)),
            contrast.loc[src],
        )
        for src in labels
    ]
    _forest(
        axes[0],
        rows,
        alpha,
        f"{conditions[0]} - {conditions[1]}",
        f"Is {conditions[1]} lower than {conditions[0]}?",
        rng,
    )
    for ax, mask in zip(axes[1:], _MASK_LABELS):
        table = cell_tables[f"discrimination:{mask}"].set_index("source")
        ref = labels.index(mask)
        rows = [
            (
                src,
                at.discrimination_gain(value, conditions, labels.index(src), ref),
                table.loc[src],
            )
            for src in ic_labels
        ]
        _forest(
            ax,
            rows,
            alpha,
            f"(IC - {mask}) contrast",
            f"Better than {mask}?\ntwo-sided; 0 = as good",
            rng,
        )
    fig.suptitle(title, y=1.02, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_snr(cell_tables, value, conditions, ic_labels, alpha, title, path):
    """SNR-by-condition: one panel per reference, both conditions overlaid.

    Source columns of ``value[cond]`` are the IC rows first, then the two masks, so an
    IC row *i* is column *i* and a mask is ``len(ic_labels) + its offset``.
    """
    rng = np.random.default_rng(42)
    offsets = np.linspace(0.2, -0.2, len(conditions))
    n_ic = len(ic_labels)
    fig, axes = plt.subplots(
        1,
        len(_MASK_LABELS),
        figsize=(6.6 * len(_MASK_LABELS), 0.8 + 0.7 * n_ic),
        squeeze=False,
    )
    for ax, mask in zip(axes[0], _MASK_LABELS):
        ref_col = n_ic + _MASK_LABELS.index(mask)
        drawn, edges = [], []
        for ci, cond in enumerate(conditions):
            table = cell_tables[f"snr:{mask}:{cond}"].set_index("source")
            m = value[cond]
            for i, src in enumerate(ic_labels):
                interval = _bootstrap_ci(m[:, i] - m[:, ref_col], alpha, rng)
                edges += list(interval)
                drawn.append((ci, i, interval, table.loc[src]))
        span = (np.nanmax(edges) - np.nanmin(edges)) or 1.0
        lo, hi = np.nanmin(edges) - 0.18 * span, np.nanmax(edges) + 0.18 * span
        for ci, i, (low, high), row in drawn:
            y = n_ic - 1 - i + offsets[ci]
            colour = _CONDITION_COLORS[conditions[ci]]
            sig = row["p"] <= alpha
            ax.plot(
                [low, high],
                [y, y],
                color=colour,
                lw=2.1,
                solid_capstyle="round",
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                [row["median"]],
                [y],
                s=64,
                marker="o",
                color=colour if sig else "white",
                edgecolor=colour,
                linewidth=1.5,
                zorder=4,
            )
            ax.text(
                0.985,
                y,
                f"p={row['p']:.3f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=8,
                family="monospace",
                fontweight="bold" if sig else "normal",
                color=colour,
            )
        ax.set_xlim(lo, hi)
        ax.axvline(0.0, color="0.3", lw=1.3, zorder=1)
        ax.set_yticks(range(n_ic))
        ax.set_yticklabels(list(reversed(ic_labels)))
        ax.set_xlabel(f"IC SNR - {mask} SNR")
        ax.set_title(f"vs {mask}", loc="left", fontsize=11)
    handles = [
        plt.Line2D([0], [0], marker="o", color=_CONDITION_COLORS[c], lw=0, label=c)
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(conditions),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(title, y=1.03, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Run the full grid for one stored decomposition."""
    experiment = ExperimentNames(args.experiment)
    variant = IvaVariants(args.variant)
    store_condition = (
        ConditionVariants(args.store_condition)
        if args.store_condition
        else _default_store_condition(variant)
    )
    music_type = MusicTypeVariants(args.music_type)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    conditions = at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))

    # ---- 1. store, filters, both references ------------------------------
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
    patterns = results.channel_patterns
    filters = at.recover_spatial_filters(patterns)  # (rows, K, C)
    electrode_mask = assr_electrode_mask(
        store_channels, coordinate_system, strict=not args.lenient_mask
    )
    binary = at.binary_filter_weights(electrode_mask, normalize=not args.mask_sum)
    projectors = at.pca_reconstruction_projectors(patterns, filters)
    pca_rows = at.pca_mask_rows(binary, projectors)  # (rows, C), per recording
    n_components = results.n_components
    ic_labels = [at.COMPONENT_LABEL.format(k=k + 1) for k in range(n_components)]
    labels = ic_labels + list(_MASK_LABELS)
    resolve, per_recording = _row_resolver(results, variant)
    _logger.info(
        f"variant {variant.value} ({'per-recording' if per_recording else 'shared'} "
        f"filters); {int(electrode_mask.sum())} ASSR electrode(s)"
    )

    # ---- 2. frequency selections (union streamed once) -------------------
    selections = {
        at.selection_label(args.center_freq, h): at.frequency_selection(
            results.freqs, args.center_freq, h
        )
        for h in args.halfwidths
    }
    union = np.unique(np.concatenate(list(selections.values())))
    union_position = {int(b): i for i, b in enumerate(union)}

    # ---- 3. onsets, caches, paired cohort --------------------------------
    concat_dir = _concatenated_dir(experiment, args.wavelet_data_dir)
    onsets, cache_labels, cache_paths = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        cache_labels[condition] = _cache_participants(concat_dir, label)
        cache_paths[condition] = _wavelet_cache_path(
            args.wavelet_data_dir, experiment, label
        )

    store_participants = list(dict.fromkeys(results.participants))
    participants = [
        p
        for p in store_participants
        if all(resolve_ok(results, p, c, per_recording) for c in conditions)
        and all(p in cache_labels[c] for c in conditions)
    ]
    if not participants:
        raise ValueError(
            "No participant is present in the store under every condition and every cache."
        )
    dropped = [p for p in store_participants if p not in participants]
    if dropped:
        _logger.warning(f"dropped (not paired / not cached): {dropped}")
    n_participants = len(participants)

    # ---- 4. project both conditions (one stream each) --------------------
    prestim_proj, zmask_proj = {}, {}
    for condition in conditions:
        prestim_proj[condition], zmask_proj[condition] = _project_condition(
            cache_paths[condition],
            condition,
            participants,
            resolve,
            filters,
            binary,
            pca_rows,
            union,
            cache_labels[condition],
            store_channels,
            args.n_times,
        )

    # ---- 5. epoch geometry (shared window) -------------------------------
    sfreq = results.sfreq
    geometry = {
        c: at.epoch_geometry(onsets[c], prestim_proj[c].shape[-1], sfreq)
        for c in conditions
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)
    baseline_mask = times < 0.0
    _logger.info(
        f"epoch {pre}+{post}={pre + post} samples = [{times[0]:.3f}, {times[-1]:.3f}] s"
    )

    # per-(participant, condition) polarity flip, IC rows only (masks stay +1)
    flip = {}
    for condition in conditions:
        rows_c = [resolve(p, condition) for p in participants]
        flip_ic, _strength = at.polarity_flip(patterns[rows_c], electrode_mask)
        if args.no_polarity_anchor:
            flip_ic = np.ones_like(flip_ic)
        flip[condition] = np.concatenate(
            [flip_ic, np.ones((n_participants, len(_MASK_LABELS)))], axis=1
        )

    # z-scored IC source band, per (selection, condition): straight from the store
    def _z_ic_band(bins: np.ndarray, condition: str) -> np.ndarray:
        # tf_maps[row] is (K, F, T); average over the SELECTED FREQUENCY bins (axis 1),
        # keeping every component, to match the raw-projected band collapse.
        return np.stack(
            [
                results.tf_maps[resolve(p, condition)][:, bins, :].mean(axis=1)
                for p in participants
            ]
        )  # (P, K, T_full)

    # ---- 6. the grid: selection x window ---------------------------------
    rows_out: list[dict] = []
    plots_dir = (
        (Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH)
        / _STAGE_DIR
        / f"{store_condition.value}_{music_type.value}"
        / SpectrumTypeVariants.BROADBAND.value
        / _ANALYSIS_DIR
        / f"pca_{args.n_pca}"
    )
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for sel_name, bins in selections.items():
        take = [union_position[int(b)] for b in bins]
        # cut trials once per condition per variant input
        cut = {"prestim": {}, "zscored": {}}
        for condition in conditions:
            onsets_inside = geometry[condition][0]
            prestim_band = prestim_proj[condition][:, :, take, :].mean(
                axis=2
            )  # (P, K+2, T)
            cut["prestim"][condition], _ = at.cut_trials(
                prestim_band, onsets_inside, pre, post
            )
            zmask_band = zmask_proj[condition][:, :, take, :].mean(axis=2)  # (P, 2, T)
            z_ic = _z_ic_band(bins, condition)  # (P, K, T)
            zscored_band = np.concatenate([z_ic, zmask_band], axis=1)  # (P, K+2, T)
            cut["zscored"][condition], _ = at.cut_trials(
                zscored_band, onsets_inside, pre, post
            )

        for lo, hi in args.stimulus_intervals:
            window = (times >= lo) & (times <= hi)
            if not window.any():
                raise ValueError(f"window {lo}-{hi}s selects no epoch sample.")
            window_label = f"{lo:g}-{hi:g}s"

            for signal in args.signal_variants:
                # value per condition: median over trials of the window mean, anchored
                value = {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+2, N, W)
                    if signal == "prestim":
                        z, _rel, _pos = at.baseline_normalise(trials, baseline_mask)
                        per_trial = z[..., window].mean(axis=-1)
                    else:
                        per_trial = trials[..., window].mean(axis=-1)
                    value[condition] = np.median(per_trial, axis=2) * flip[condition]

                cell = {
                    "variant": signal,
                    "selection": sel_name,
                    "window": window_label,
                }
                _accumulate_tests(
                    rows_out,
                    cell,
                    value,
                    conditions,
                    labels,
                    ic_labels,
                    args.contrast_alternative,
                    args.snr_alternative,
                )

                if not args.skip_plots:
                    _write_cell_plots(
                        plots_dir,
                        cell,
                        value,
                        conditions,
                        labels,
                        ic_labels,
                        args.alpha,
                        rows_out,
                        n_participants,
                    )

    # ---- 7. one CSV for the whole grid -----------------------------------
    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / f"assr_snr_grid__{variant.value}__pca{args.n_pca}.csv"
    frame = pd.DataFrame(rows_out)
    frame.to_csv(csv_path, index=False)
    _logger.info(f"saved {csv_path} ({len(frame)} test rows)")
    print(
        f"\n{len(frame)} tests over {len(selections)} selection(s) x "
        f"{len(args.stimulus_intervals)} window(s) x {len(args.signal_variants)} "
        f"variant(s), n = {n_participants}. Floor p = "
        f"{at.p_floor(n_participants):.5f} two-sided."
    )
    print(frame.to_string(index=False))


def resolve_ok(results, participant: str, condition: str, per_recording: bool) -> bool:
    """Whether the store has a row for this participant under this condition."""
    if per_recording:
        return bool(results.rows(participant, condition))
    return bool(results.rows(participant))


def _accumulate_tests(
    rows_out, cell, value, conditions, labels, ic_labels, contrast_alt, snr_alt
) -> None:
    """Append every test row for one grid cell to *rows_out*."""
    # 4b — condition contrast per source
    for s, source in enumerate(labels):
        rows_out.append(
            {
                **cell,
                "family": "contrast",
                "source": source,
                "reference": "",
                "condition": "",
                **at.paired_test(
                    at.condition_contrast(value, conditions, s), contrast_alt
                ),
            }
        )
    # 4c — discrimination gain vs each reference
    for mask in _MASK_LABELS:
        ref = labels.index(mask)
        for source in ic_labels:
            s = labels.index(source)
            rows_out.append(
                {
                    **cell,
                    "family": "discrimination",
                    "source": source,
                    "reference": mask,
                    "condition": "",
                    "IC contrast": float(
                        np.median(at.condition_contrast(value, conditions, s))
                    ),
                    **at.paired_test(
                        at.discrimination_gain(value, conditions, s, ref), "two-sided"
                    ),
                }
            )
    # Step 12 — SNR magnitude vs each reference, PER CONDITION
    for mask in _MASK_LABELS:
        ref = labels.index(mask)
        for condition in conditions:
            m = value[condition]
            for source in ic_labels:
                s = labels.index(source)
                d = m[:, s] - m[:, ref]
                rows_out.append(
                    {
                        **cell,
                        "family": "snr",
                        "source": source,
                        "reference": mask,
                        "condition": condition,
                        "IC SNR": float(np.nanmedian(m[:, s])),
                        "ref SNR": float(np.nanmedian(m[:, ref])),
                        **at.paired_test(d, snr_alt),
                    }
                )


def _write_cell_plots(
    plots_dir,
    cell,
    value,
    conditions,
    labels,
    ic_labels,
    alpha,
    rows_out,
    n_participants,
) -> None:
    """Two figures for one grid cell, built from the rows just accumulated."""
    tag = f"{cell['variant']}__{cell['selection']}__{cell['window']}"
    frame = pd.DataFrame([r for r in rows_out if all(r[k] == cell[k] for k in cell)])
    cell_tables = {
        "contrast": frame[frame["family"] == "contrast"],
        **{
            f"discrimination:{m}": frame[
                (frame["family"] == "discrimination") & (frame["reference"] == m)
            ]
            for m in _MASK_LABELS
        },
        **{
            f"snr:{m}:{c}": frame[
                (frame["family"] == "snr")
                & (frame["reference"] == m)
                & (frame["condition"] == c)
            ]
            for m in _MASK_LABELS
            for c in conditions
        },
    }
    _plot_cell(
        cell_tables,
        value,
        conditions,
        labels,
        ic_labels,
        alpha,
        f"Condition contrast & discrimination — {tag}, n={n_participants}",
        plots_dir / f"pvalue_summary__{tag}.png",
    )
    _plot_snr(
        cell_tables,
        value,
        conditions,
        ic_labels,
        alpha,
        f"IC SNR vs reference, per condition — {tag}, n={n_participants}",
        plots_dir / f"snr_by_condition__{tag}.png",
    )


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
