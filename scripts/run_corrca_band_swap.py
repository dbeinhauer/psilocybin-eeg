"""
Band-swapped CorrCA filters (stage 06b): each broadband fit read through its
band-restricted twin's filters, and the band fit through the broadband one's.

Read-only: every store must already exist, written by a broadband and a banded run
(e.g. ``--band_range 30 50``, token ``--band 30-50hz``, or ``--band_range 1 30``,
token ``--band 1-30hz``) of the same fit script and ``--n_pca``. Nothing is fitted. For the run's full fit (and, with
``--log_power`` / ``--roi_fit`` / ``--non_roi_fit``, its ``_log``, ``_roi``, ``_non_roi``
fits and their log twins), the broadband store and its band twin
(same variant and electrode cut) are paired. The pairing and outputs mirror the raw <-> log
``swapped_filter`` reading of ``run_corrca_snr_grid.py``:

  - ``band_swapped_filter_topomap_correlation.csv``: Pearson r over channels of the
    two fits' shared forward patterns (``pattern_r``, the topomaps) and of their
    filters (``filter_r``), component *k* against component *k*, never re-matched.
  - TF maps of the leading ``--swapped_components``. The band fit's filters are
    applied to the broadband fit's input (every cache frequency) and drawn into the
    broadband fit's directory. The broadband fit's filters are applied to the band
    fit's input (the band's bins only) and drawn into the band fit's directory. Both
    use the fit's own ``condition_mean_tf_maps`` calls (+ ``_onset``,
    + ``participants/``), so within one directory the data stay fixed and only the
    topography changes. The broadband maps mark the band's inner edge (30 Hz for
    both ``30-50hz`` and ``1-30hz``) beside the 40 Hz ASSR line: everything across it
    is what the band filters pick up outside the band they were fitted on.

The input is what each fit was handed: the cache on its frequencies, ``log10`` first
for a ``*_log*`` fit, z-scored along time per (channel, frequency) and per
condition. Because the z-score is per frequency, the band fit's input is exactly the
broadband input at the band's bins. So the broadband filters on the band input
reproduce the broadband fit's own sources there; only the colour scale is the band's.
The informative direction is the band filters read at every frequency: for
``1-30hz``, whether a topography chosen below 30 Hz also carries the 40 Hz response.

The cache is read on the full montage once per condition, and every fit's filters are
embedded with zeros off its own electrodes. Because the z-score is per channel, that
reads exactly what an ROI or non-ROI fit saw. One pass then serves all six pairs.

Outputs::

    plots/06b-corrca-condition-comparison/<Join>_<Music>/broadband/<variant>/pca_<N>/
        band_swapped_filter_<band>_condition_mean_tf_maps{,_onset}.png
        band_swapped_filter_<band>_topomap_correlation.csv
        participants/band_swapped_filter_<band>_*.png
    plots/06b-corrca-condition-comparison/<Join>_<Music>/bands/<variant>/pca_<N>/
        <band>_band_swapped_filter_condition_mean_tf_maps{,_onset}.png
        <band>_band_swapped_filter_topomap_correlation.csv
        participants/<band>_band_swapped_filter_*.png
    results/<experiment>/corrca_band_swap__<variant>__<band>__pca<N>.csv
        # every pair of the run; an individual condition's carries <variant>__<Condition>

Examples::

    python scripts/run_corrca_band_swap.py --experiment assr \\
        --variant corrca_channel_joined_separate --band 30-50hz --n_pca 5

    python scripts/run_corrca_band_swap.py --variant corrca_channel_joined_tracks \\
        --band 1-30hz --n_pca 5 --n_subjects 2 --n_times 3000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    CORRCA_INDIVIDUAL_VARIANTS,
    CORRCA_LOG_OF,
    MAX_DRAWN_COMPONENTS,
    STABILITY_COMPONENTS,
    add_corrca_fit_selection_args,
    corrca_plots_dir,
    corrca_run_plan,
)
from scripts.run_assr_snr_grid import (  # noqa: E402
    _cache_frequency_grid,
    _load_onsets,
)

# The swapped_filter figures, so both swaps are drawn by the same calls.
from scripts.run_corrca_snr_grid import (  # noqa: E402
    _STORE_CONDITIONS,
    _TF_FREQ_MARKS,
    _draw_swapped_tf_maps,
    _embed,
)
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis.assr_variants import zscore_time  # noqa: E402
from src.analysis.corrca import matched_component_correlation  # noqa: E402
from src.analysis.wavelet_ica import log10_power  # noqa: E402
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    ExperimentNames,
    IvaVariants,
    MusicTypeVariants,
)
from src.io.iva_store import iva_results_path  # noqa: E402
from src.io.loading import concatenated_dir as _concatenated_dir  # noqa: E402
from src.io.wavelet_store import (  # noqa: E402
    find_wavelet_set,
    iter_wavelet_subjects,
    wavelet_label_dir,
)
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402

_logger = logging.getLogger(__name__)

#: Label of the reading: figure and CSV prefix.
READING = "band_swapped_filter"
#: The two inputs a pair is read on.
_BROADBAND, _BAND = "broadband", "band"


class StoreHead(NamedTuple):
    """What the swap needs from a stage-06b store, without its sources."""

    path: Path
    channels: list[str]
    filters: np.ndarray  # (K, C)
    patterns: np.ndarray  # (K, C)
    freqs: np.ndarray
    sfreq: float
    participants: list[str]


class BandPair(NamedTuple):
    variant: IvaVariants
    log_power: bool
    broadband: StoreHead
    band: StoreHead


def read_store_head(path: Path) -> StoreHead:
    """Read a store's filters, patterns and axes; the ``tf_map`` is never loaded."""
    with np.load(path, allow_pickle=False) as stored:
        if "extra__corrca_filters" not in stored.files:
            raise ValueError(
                f"{path} holds no 'corrca_filters': not a stage-06b store."
            )
        return StoreHead(
            path=path,
            channels=[str(name) for name in stored["channel_names"]],
            filters=np.asarray(stored["extra__corrca_filters"], dtype=float).T,
            patterns=np.asarray(stored["extra__corrca_shared_patterns"], dtype=float),
            freqs=np.asarray(stored["freqs"], dtype=float),
            sfreq=float(stored["sfreq"]),
            participants=list(dict.fromkeys(str(p) for p in stored["participants"])),
        )


def pair_correlation(pair: BandPair) -> pd.DataFrame:
    """One row per component: ``pattern_r`` (the topomaps) and ``filter_r``."""
    pattern_r = matched_component_correlation(
        pair.broadband.patterns, pair.band.patterns
    )
    filter_r = matched_component_correlation(pair.broadband.filters, pair.band.filters)
    return pd.DataFrame(
        {
            "component": np.arange(1, pattern_r.size + 1),
            "variant": pair.variant.value,
            "pattern_r": pattern_r,
            "filter_r": filter_r,
        }
    )


def cache_bins(cache_freqs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Cache bin of every store frequency.

    :raises ValueError: If a store frequency is not on the cache's grid.
    """
    grid = np.asarray(cache_freqs, dtype=float)
    bins = np.array([int(np.argmin(np.abs(grid - f))) for f in freqs])
    if not np.allclose(grid[bins], freqs, rtol=1e-6):
        raise ValueError("The store's frequencies are not on the cache's grid.")
    return bins


def _as_index(positions: np.ndarray) -> slice | np.ndarray:
    """A contiguous run as a slice, so indexing a block views it instead of copying."""
    if positions.size and np.array_equal(
        positions, np.arange(positions[0], positions[0] + positions.size)
    ):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions


def project_swaps(
    block: np.ndarray,
    readers: dict[tuple[IvaVariants, str], tuple[np.ndarray, bool, slice | np.ndarray]],
) -> dict[tuple[IvaVariants, str], np.ndarray]:
    """One subject's block read through every swapped filter set.

    :param block: ``(channels, freqs, times)`` raw wavelet power on the full montage.
    :param readers: ``(variant, input) -> (filters (K, C), log_power, bins)``: the
        twin's filters, the input's power scale and the block bins of the input's
        frequencies.
    :return: ``(variant, input) -> (K, len(bins), times)`` float32.
    """
    inputs: dict[bool, np.ndarray] = {}
    for log_power in sorted({log for _, log, _ in readers.values()}):
        inputs[log_power] = zscore_time(log10_power(block) if log_power else block)
    return {
        key: at.project_channels(filters, inputs[log_power][:, bins]).astype(np.float32)
        for key, (filters, log_power, bins) in readers.items()
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read each stage-06b broadband fit through its band twin's "
        "filters and vice versa, from stored fits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stored = parser.add_argument_group("the stored decompositions to pair")
    stored.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    stored.add_argument(
        "--variant",
        default=IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS.value,
        choices=[v.value for v in CORRCA_LOG_OF],
        help="Which stage-06b run: its full fit (plus the fits the flags below "
        "select) is paired with its band twin.",
    )
    add_corrca_fit_selection_args(stored)
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument(
        "--band",
        default="30-50hz",
        help="Band token of the band-restricted stores (the broadband ones have "
        "none), e.g. 30-50hz or 1-30hz.",
    )
    stored.add_argument("--n_pca", type=int, default=5)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument("--wavelet_data_dir", type=Path, default=None)
    data.add_argument("--concat_dir", type=Path, default=None)
    data.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions to read; for the individual-condition stores, each on its "
        "own. Default Placebo Psilocybin.",
    )
    data.add_argument(
        "--n_subjects",
        type=int,
        default=None,
        help="Keep the first N paired participants (a quick check).",
    )
    data.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Trim the time axis (a quick check; not for real runs).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--swapped_components",
        type=int,
        default=STABILITY_COMPONENTS,
        help="Project and draw this many leading components. The correlation CSVs "
        "always cover every component.",
    )
    output.add_argument("--save_dir", type=Path, default=None)
    output.add_argument("--results_dir", type=Path, default=None)
    output.add_argument("--skip_plots", action="store_true")
    output.add_argument("--log_level", default="INFO")
    return parser


def run(args: argparse.Namespace) -> None:
    """Pair and read every fit of the run; an individual-condition run per condition."""
    variant = IvaVariants(args.variant)
    plan = corrca_run_plan(variant, args)
    if variant in CORRCA_INDIVIDUAL_VARIANTS:
        stores = [
            (ConditionVariants(c), [c])
            for c in at.resolve_conditions(args.conditions or list(REAL_CONDITIONS))
        ]
    else:
        stores = [(_STORE_CONDITIONS[variant], args.conditions)]
    for store_condition, conditions in stores:
        run_join(args, plan, store_condition, conditions)


def _pairs(
    args: argparse.Namespace,
    plan: list[tuple[IvaVariants, str, bool]],
    store_condition: ConditionVariants,
) -> list[BandPair]:
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    pairs = []
    for fit, _cut, log_power in plan:
        heads = {}
        for spectrum, band in ((_BROADBAND, None), (_BAND, args.band)):
            path = iva_results_path(
                experiment,
                store_condition,
                fit,
                music_type,
                band,
                args.n_pca,
                processed_data_dir=args.store_dir,
            )
            if not path.is_file():
                _logger.warning(f"no {fit.value} store at {path}: pair skipped.")
                break
            heads[spectrum] = read_store_head(path)
        else:
            broadband, band = heads[_BROADBAND], heads[_BAND]
            if (
                broadband.channels != band.channels
                or broadband.filters.shape != band.filters.shape
                or broadband.sfreq != band.sfreq
            ):
                _logger.warning(
                    f"{fit.value}: the broadband and {args.band} stores differ in "
                    "channels, component count or sampling rate: pair skipped."
                )
                continue
            pairs.append(BandPair(fit, log_power, broadband, band))
    return pairs


def run_join(
    args: argparse.Namespace,
    plan: list[tuple[IvaVariants, str, bool]],
    store_condition: ConditionVariants,
    conditions: list[str] | None,
) -> None:
    """Every pair of one join (or one individual condition): CSVs, one read, figures."""
    experiment = ExperimentNames(args.experiment)
    music_type = MusicTypeVariants(args.music_type)
    conditions = at.resolve_conditions(conditions or list(REAL_CONDITIONS))
    pairs = _pairs(args, plan, store_condition)
    if not pairs:
        _logger.warning(f"[{store_condition.value}] no complete pair: nothing to do.")
        return

    # ---- 1. topomap / filter correlations ----------------------------------
    tables = {}
    for pair in pairs:
        table = pair_correlation(pair)
        tables[pair.variant] = table
        _logger.info(
            f"{pair.variant.value}: broadband vs {args.band}, matched-component r "
            + "; ".join(
                f"{COMPONENT_PREFIX} {row.component}: pattern {row.pattern_r:+.3f}, "
                f"filter {row.filter_r:+.3f}"
                for row in table.itertuples()
            )
        )
    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    full = IvaVariants(args.variant)
    store_token = (
        f"{full.value}__{store_condition.value}"
        if full in CORRCA_INDIVIDUAL_VARIANTS
        else full.value
    )
    csv_path = results_root / (
        f"corrca_band_swap__{store_token}__{args.band}__pca{args.n_pca}.csv"
    )
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.insert(2, "store_condition", store_condition.value)
    combined.insert(3, "band", args.band)
    combined.to_csv(csv_path, index=False)
    _logger.info(f"wrote {csv_path}")

    def _fit_dir(variant: IvaVariants, spectrum: str) -> Path:
        return corrca_plots_dir(
            variant,
            store_condition,
            music_type,
            band=None if spectrum == _BROADBAND else args.band,
            n_pca=args.n_pca,
            root=args.save_dir,
        )

    # One broadband directory holds the reading of every band twin, so its files carry
    # the band; the band directory's carry it as their usual plot prefix.
    def _prefix(spectrum: str) -> str:
        if spectrum == _BROADBAND:
            return f"{READING}_{args.band}_"
        return f"{args.band}_{READING}_"

    for pair in pairs:
        for spectrum in (_BROADBAND, _BAND):
            fit_dir = _fit_dir(pair.variant, spectrum)
            fit_dir.mkdir(parents=True, exist_ok=True)
            twin = args.band if spectrum == _BROADBAND else "broadband"
            tables[pair.variant].assign(spectrum=spectrum, twin_spectrum=twin).to_csv(
                fit_dir / f"{_prefix(spectrum)}topomap_correlation.csv", index=False
            )
    n_components = min(
        args.swapped_components,
        pairs[0].broadband.filters.shape[0],
        MAX_DRAWN_COMPONENTS,
    )
    if args.skip_plots or n_components <= 0:
        return

    # ---- 2. the cache, the cohort, the bins ---------------------------------
    concat_dir = _concatenated_dir(experiment, args.concat_dir)
    wavelet_sets, onsets = {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        wavelet_sets[condition] = find_wavelet_set(
            wavelet_label_dir(args.wavelet_data_dir, experiment, label)
        )
    read_channels = list(wavelet_sets[conditions[0]].channel_names)
    if any(list(w.channel_names) != read_channels for w in wavelet_sets.values()):
        raise ValueError("The conditions' caches differ in their channels.")
    cache_freqs = _cache_frequency_grid(wavelet_sets, conditions)
    broadband_bins = cache_bins(cache_freqs, pairs[0].broadband.freqs)
    band_bins = cache_bins(cache_freqs, pairs[0].band.freqs)
    if any(
        not np.array_equal(p.broadband.freqs, pairs[0].broadband.freqs)
        or not np.array_equal(p.band.freqs, pairs[0].band.freqs)
        for p in pairs
    ):
        raise ValueError("The stores of one run differ in their frequency grids.")
    read = np.union1d(broadband_bins, band_bins)
    positions = {
        _BROADBAND: _as_index(np.searchsorted(read, broadband_bins)),
        _BAND: _as_index(np.searchsorted(read, band_bins)),
    }
    participants = [
        p
        for p in pairs[0].broadband.participants
        if all(
            p in head.participants
            for pair in pairs
            for head in (pair.broadband, pair.band)
        )
        and all(p in w.participants for w in wavelet_sets.values())
    ]
    if args.n_subjects is not None:
        participants = participants[: args.n_subjects]
    if not participants:
        raise ValueError("No participant is in every store and every cache.")
    sfreq = pairs[0].broadband.sfreq

    # The broadband input reads the band twin's filters, and the band input the
    # broadband fit's: (variant, input) -> (filters, log_power, bins).
    readers = {}
    for pair in pairs:
        for spectrum, twin in ((_BROADBAND, pair.band), (_BAND, pair.broadband)):
            readers[(pair.variant, spectrum)] = (
                _embed(twin.filters[:n_components], twin.channels, read_channels),
                pair.log_power,
                positions[spectrum],
            )

    # ---- 3. one pass over each condition's cache ----------------------------
    projected = {}  # condition -> (variant, input) -> (P, K, F, T)
    for condition in conditions:
        wavelet_set = wavelet_sets[condition]
        available = wavelet_set.open(participants[0]).shape[-1]
        keep = available if args.n_times is None else min(args.n_times, available)
        _logger.info(
            f"[{condition}] reading {len(participants)} participant(s) from "
            f"{wavelet_set.directory}: {len(pairs)} pair(s), {n_components} "
            "component(s) each way"
        )
        started = time.time()
        n_bins = {_BROADBAND: broadband_bins.size, _BAND: band_bins.size}
        out = {
            key: np.empty(
                (len(participants), n_components, n_bins[key[1]], keep), np.float32
            )
            for key in readers
        }
        for row, (name, block) in enumerate(
            iter_wavelet_subjects(
                wavelet_set, participants, freq_indices=read, n_times=keep
            )
        ):
            for key, value in project_swaps(block, readers).items():
                out[key][row] = value
            del block
            _logger.info(f"[{condition}]   {name} ({time.time() - started:.0f}s)")
        projected[condition] = out

    # ---- 4. figures, in each fit's own directory ----------------------------
    shown = list(range(n_components))
    band_freqs = pairs[0].band.freqs
    broadband_freqs = pairs[0].broadband.freqs
    # Only an edge inside the broadband grid separates in-band from out-of-band rows.
    band_edges = [
        float(edge)
        for edge in (band_freqs.min(), band_freqs.max())
        if broadband_freqs.min() < edge < broadband_freqs.max()
    ]
    freq_marks = {
        _BROADBAND: [*_TF_FREQ_MARKS, *band_edges],
        _BAND: list(_TF_FREQ_MARKS),
    }
    edge_note = (
        f"; band edge {', '.join(f'{e:g}' for e in band_edges)} Hz marked"
        if band_edges
        else ""
    )
    for pair in pairs:
        table = tables[pair.variant]
        topomap_r = ", ".join(
            f"{COMPONENT_PREFIX} {k + 1} {table['pattern_r'][k]:+.2f}" for k in shown
        )
        for spectrum, head, twin_name in (
            (_BROADBAND, pair.broadband, args.band),
            (_BAND, pair.band, "broadband"),
        ):
            this_name = "broadband" if spectrum == _BROADBAND else args.band
            _draw_swapped_tf_maps(
                {c: projected[c][(pair.variant, spectrum)] for c in conditions},
                onsets,
                participants,
                conditions,
                head.freqs,
                sfreq,
                shown,
                _fit_dir(pair.variant, spectrum),
                _prefix(spectrum),
                label=f"{store_condition.value}_{music_type.value}"
                + ("_log" if pair.log_power else "")
                + f" [{this_name}]",
                note=f"filters of the {twin_name} fit on this {this_name} fit's "
                f"input; topomap r {topomap_r}"
                + (edge_note if spectrum == _BROADBAND else ""),
                freq_marks=freq_marks[spectrum],
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
