"""
Full-recording ASSR SNR / condition-contrast grid, for either IVA condition-comparison
variant.

This is the CLI counterpart of
``notebooks/06-iva-condition-comparison/iva_component_analysis_joined.ipynb`` and its
``_tracks`` sibling, run on the **full** wavelet cache rather than the 12 s notebook
subset. It reproduces the whole current workflow — both ASSR references, every signal
variant, and all three test families — and sweeps the frequency-selection x
stimulus-window grid in one pass, which is exactly what a job wants:

* **Three references**, in increasing proximity to what the IVA actually saw.
  ``ASSR-mask (full)`` is the fixed fronto-central electrode average on the whole
  channel space, built the equal-weight way: every electrode is referenced to its own
  pre-stimulus mean per trial and divided by its baseline SD pooled over trials BEFORE
  the ROI is averaged, then the average is referenced to its own again (see
  :func:`~src.analysis.assr_trials.roi_channelwise_snr`), because averaging raw power
  first would weight each electrode by its own power level, and a per-trial divisor
  would let a momentarily quiet channel plateau the whole course.
  ``ASSR-mask (PCA)`` is the same average read only through each recording's PCA
  subspace (``mask @ P^T P``) on the RAW cache. ``ASSR-mask (PCA, z)`` is that same
  operator on the Z-SCORED cache — the signal the decomposition was handed, after the
  channel reduction and before the unmixing — so the pair separates the reduction from
  the per-(channel, frequency) rescaling z-scoring adds. In ``zscored`` the last two
  coincide by construction; in the baseline variants they differ.
* **Seven signal variants**, differing in how the LEARNED rows are read — which spatial
  filter, on which signal — and in nothing else. **The references never move**: every
  variant but ``zscored`` carries ``prestim``'s three reference columns verbatim, which
  is what makes the whole set one comparison rather than seven separate ones.

  - ``zscored`` reads the stored per-recording z-scored sources for the IC rows (full
    recording, every onset) and the masks on the z-scored cache. No per-trial baseline.
  - ``prestim`` projects the raw cache through the recovered spatial filters and
    references every trial to its own pre-stimulus baseline.
  - ``stored_prestim`` takes the IC rows from the stored sources — the decomposition's
    own output, which the projection only approximates, since the model was fitted on
    time-z-scored data and ``prestim`` applies the same filter to un-z-scored power —
    and gives them the same per-trial baseline.
  - ``mean_prestim`` is ``prestim`` with a single thing changed: the **cohort-mean
    filter** (:func:`~src.analysis.assr_trials.cohort_mean_pattern`) in place of each
    recording's own. Anything that differs between the two rows is the per-recording
    topography, and nothing else.
  - ``mean_stored_prestim`` applies that shared filter to the IVA's **own input**,
    rebuilt in channel space as ``P^T P (z)`` — z-scored along time, then reduced onto
    that recording's PCA subspace. No stored array is the shared filter on that signal,
    so it is reconstructed rather than read back.
  - ``masked_prestim`` is each recording's own filter with every weight **outside the
    ASSR electrodes zeroed**, on the z-scored cache. It sits between the two things the
    grid already has — the binary reference reads those electrodes with equal weight,
    an unrestricted component reads the whole head with learned weights, this reads only
    those electrodes with learned weights — and so separates a better weighting *inside*
    the anchor area from access to signal *outside* it, which no other row can tell
    apart.
  - ``mean_masked_prestim`` is the same restriction on the cohort-mean filter.

  **Polarity.** The three ``mean_*`` rows read ONE shared operator whose sign was
  anchored once when the cohort mean was built, so no per-recording flip is applied to
  them — doing so would scramble a quantity that does not carry that sign.
  ``masked_prestim`` uses each recording's own filter and keeps the flip; zeroing
  channels does not change a component's sign.

  **The restricted pair reads the z-scored wavelet, not raw power**, because the filters
  were estimated on data z-scored along time and their weights presume that scaling — a
  weighted average of raw power is additionally weighted by each electrode's own power
  level, the same artefact ``roi_channelwise_snr`` exists to remove from the binary
  reference. And no PCA for them: ``P^T P`` mixes every channel back into the retained
  subspace, which would defeat "read only these electrodes" entirely.

  Because this script streams the **full** cache, every variant sees identical onsets
  and identical reference rows; the only thing that differs is how the component itself
  was read. (In the notebooks, whose caches are a 12 s subset, the stored IC rows carry
  many more trials than the mask rows — that asymmetry does not exist here.)
* **Three test families**, participants as the unit, exact Wilcoxon:
  ``contrast`` (Placebo - Psilocybin per source), ``discrimination`` (does an IC separate
  the conditions better than a reference, the interaction, vs each reference) and
  ``snr`` (is an IC's response magnitude *lower* than a reference's, per condition).
* **The grid**: every ``--halfwidths`` x ``--stimulus_intervals`` combination, so a
  single run covers e.g. 40 Hz and 35-45 Hz crossed with 0-500 ms and 200-500 ms.

**Both decomposition variants.** ``--variant iva_channel_joined_tracks`` shares one
filter per participant across the conditions; ``--variant iva_channel_joined`` has a
separate
topography per recording, so every projection and polarity anchor is resolved per
(participant, condition) through ``results.row``. Nothing else differs.

Outputs (per run, all grid cells in one file):

* ``results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv`` — every test
  row, tagged with its signal variant, frequency selection and stimulus window.
* ``plots/06-iva-condition-comparison/<Condition>_<Music>/<spectrum>/assr_snr_grid/
  pca_<N>/<selection>/<window>/`` — one subdirectory per frequency selection (band range)
  and stimulus window (time interval), each holding a p-value summary, an SNR-by-condition
  figure and a per-source trial-course figure for every signal variant.
* ``.../pca_<N>/{mean,mean_masked}_filter_topomaps.png`` — the two spatial-filter check
  figures, at the top of the run's directory because the operators they draw depend on
  neither the selection nor the window. The first asks whether the cohort agreed on the
  map the ``mean_*`` rows average; the second, how much of each filter's weight survived
  the ASSR-electrode restriction the ``masked_*`` rows apply.

``<spectrum>`` is ``broadband`` by default and ``bands`` when ``--band`` names a
band-restricted store (with the band as the first token of every filename), so a run
over a band-restricted decomposition never lands on top of the broadband one.

Examples::

    # The job grid: both frequencies x both windows, every variant of signal.
    python scripts/run_assr_snr_grid.py --experiment assr --variant iva_channel_joined \\
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
from dataclasses import dataclass
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
from src.visualization.assr_filter_plots import (  # noqa: E402
    plot_cohort_mean_topomaps,
    plot_masked_filter_topomaps,
)

_logger = logging.getLogger(__name__)

_STAGE_DIR = "06-iva-condition-comparison"
_ANALYSIS_DIR = "assr_snr_grid"
# The row labels come from `assr_trials` rather than from this file: stage-07 emits the
# same three references, and a row can only be compared across the two stages if both
# spell it the same way.
_FULL_LABEL = at.FULL_MASK_LABEL
_PCA_LABEL = at.PCA_MASK_LABEL
#: The same subspace-projected electrode average, but read off the Z-SCORED wavelet:
#: literally the signal the decomposition was handed, after the channel PCA and before
#: the unmixing. It is the reference-side analogue of the "stored_prestim" IC rows and
#: fills the same missing cell — z-scored input WITH a per-trial baseline — so reading
#: it beside "ASSR-mask (PCA)" separates the channel reduction from the
#: per-(channel, frequency) rescaling that z-scoring adds. In the "zscored" variant it
#: necessarily coincides with "ASSR-mask (PCA)", which is a free check that the two
#: paths agree; in the baseline variants the two genuinely differ.
_PCA_Z_LABEL = at.PCA_Z_MASK_LABEL
_MASK_LABELS = at.MASK_LABELS
_SIGNAL_VARIANTS = (
    "zscored",
    "prestim",
    "stored_prestim",
    "mean_prestim",
    "mean_stored_prestim",
    "masked_prestim",
    "mean_masked_prestim",
)
# The variants whose trials are referenced to their own pre-stimulus window. "zscored"
# is the odd one out: its z-score along time IS the normalisation.
_BASELINE_VARIANTS = tuple(v for v in _SIGNAL_VARIANTS if v != "zscored")
#: The variants that read the ONE cohort-mean filter rather than each recording's own.
#: A shared operator carries a single sign, anchored once when it was built
#: (:func:`~src.analysis.assr_trials.cohort_mean_pattern`), so applying the
#: per-recording flip to these rows would scramble a quantity that does not carry it.
#: ``masked_prestim`` is deliberately NOT here: it uses each recording's own filter, and
#: zeroing channels does not change a component's sign.
_SHARED_FILTER_VARIANTS = ("mean_prestim", "mean_stored_prestim", "mean_masked_prestim")
#: Filenames of the two spatial-filter check figures, written once per run (the
#: operators do not depend on the frequency selection or the stimulus window).
_MEAN_FILTER_FIGURE = "mean_filter_topomaps.png"
_MEAN_MASKED_FILTER_FIGURE = "mean_masked_filter_topomaps.png"
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
        help=(
            "Which readings to test side by side. Every variant keeps 'prestim's "
            "reference rows unchanged, so only the learned rows move. 'stored_prestim' "
            "reads the IC rows off the decomposition; 'mean_*' swap each recording's "
            "own filter for the one cohort-mean filter; 'masked_*' zero every filter "
            "weight outside the ASSR electrodes, which separates a better weighting "
            "INSIDE that area from access to signal outside it."
        ),
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


def _cache_frequency_grid(cache_paths: dict, conditions: list[str]) -> np.ndarray:
    """The wavelet cache's own frequency axis, checked identical across conditions.

    Separate from the store's axis on purpose: a band-restricted store holds only its
    band's bins, while the cache is always the broadband one, so an index resolved on
    one grid is meaningless on the other. Reading the header is cheap — it decompresses
    nothing.

    :param cache_paths: Condition → cache path.
    :param conditions: Conditions to read, in order.
    :return: The shared ``(F,)`` frequency grid in Hz.
    :raises ValueError: If the conditions' caches were built on different grids, which
        would make one bin index mean two frequencies.
    """
    grids = {
        condition: np.asarray(
            read_wavelet_cache_header(cache_paths[condition]).freqs, dtype=float
        )
        for condition in conditions
    }
    reference = grids[conditions[0]]
    for condition, grid in grids.items():
        if grid.shape != reference.shape or not np.allclose(grid, reference):
            raise ValueError(
                f"The {conditions[0]} and {condition} wavelet caches were built on "
                f"different frequency grids ({reference.size} vs {grid.size} bins); a "
                "bin index cannot mean the same frequency in both. Rebuild them with "
                "the same wavelet settings."
            )
    return reference


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
# Projection — one streaming pass produces every signal variant's input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Projected:
    """One condition's projections, one array per group of source rows.

    Kept as named arrays rather than one wide stack because the groups are read on
    different cadences: the reference rows are shared by every variant, the learned rows
    are what each variant swaps out, and ``roi_raw`` skips the normalisation the others
    get. Every array is ``(participants, rows, kept freqs, times)``.
    """

    #: ``components + 2`` rows on the RAW cache: each recording's own IC filters, then
    #: the full and PCA masks. The reference rows every variant borrows come from here.
    prestim: np.ndarray
    #: The two masks on the Z-SCORED cache.
    zmask: np.ndarray
    #: The mask's electrodes kept UNCOMBINED, so
    #: :func:`~src.analysis.assr_trials.roi_channelwise_snr` can normalise each one
    #: before they are averaged.
    roi_raw: np.ndarray
    #: ``components`` rows: the cohort-mean filter on the RAW cache — "prestim" with a
    #: single thing changed, the filter.
    mean_raw: np.ndarray
    #: The cohort-mean filter on the IVA's own input (z-scored, then this recording's
    #: PCA subspace), rebuilt here because no stored array is ``U_mean`` on that signal.
    mean_stored_z: np.ndarray
    #: Each recording's OWN filter, restricted to the anchor electrodes, on the z-scored
    #: cache.
    masked_z: np.ndarray
    #: The cohort-mean filter under the same restriction.
    mean_masked_z: np.ndarray


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
    roi_index: np.ndarray,
    mean_filter: np.ndarray,
    masked_filters: np.ndarray,
    mean_masked_filter: np.ndarray,
    projectors: np.ndarray,
) -> _Projected:
    """Stream one condition once; return every per-condition array the grid needs.

    Everything is read off the **same** decompressed block, so the four operators the
    ``mean_*`` / ``masked_*`` variants add cost one more tensordot each rather than
    another pass over a ~50 GB cache. The z-scored copy of a block is built once and
    shared by every operator that needs it, for the same reason.

    The z-scored IC rows of ``stored_prestim`` are read straight from the store, so they
    are not produced here.

    :param mean_filter: ``(components, channels)`` cohort-mean filter, shared by every
        recording and both conditions.
    :param masked_filters: ``(rows, components, channels)`` per-recording filters with
        every weight outside the anchor electrodes zeroed.
    :param mean_masked_filter: ``(components, channels)`` the same restriction on the
        cohort-mean filter.
    :param projectors: ``(rows, channels, channels)`` PCA reconstruction projectors, so
        the IVA's own input can be rebuilt as ``P^T P (z)`` in channel space.
    :return: A :class:`_Projected`.
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

    def _empty(rows: int) -> np.ndarray:
        return np.empty(
            (len(participants), rows, freq_indices.size, keep_times), np.float32
        )

    prestim = _empty(n_comp + 2)
    zmask = _empty(2)
    roi_raw = _empty(roi_index.size)
    mean_raw = _empty(n_comp)
    mean_stored_z = _empty(n_comp)
    masked_z = _empty(n_comp)
    mean_masked_z = _empty(n_comp)
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
        # The cohort-mean filter on the RAW cache: "prestim" with the filter swapped and
        # nothing else, so the difference between the two rows IS the topography.
        mean_raw[row] = at.project_channels(mean_filter, block)
        roi_raw[row] = block[roi_index]

        # The z-scored copy is the expensive intermediate; build it once and hand it to
        # every operator that reads the standardisation the decomposition was fitted on.
        block_z = _zscore_time(block)
        zmask[row] = at.project_channels(
            np.concatenate([full_row, pca_row], axis=0), block_z
        )
        # The IVA's own input, rebuilt: z-scored, then this recording's PCA subspace.
        # Composing the two channel operators first keeps the (C, F, T) intermediate
        # from being materialised twice.
        mean_stored_z[row] = at.project_channels(
            mean_filter @ projectors[store], block_z
        )
        # The restricted operators read the z-scored wavelet and NOT raw power: the
        # weights presume that scaling, and a weighted average of raw power is
        # additionally weighted by each electrode's own power level — the very artefact
        # roi_channelwise_snr exists to remove from the binary reference. And no PCA:
        # P^T P mixes every channel back in, which would defeat "read only these
        # electrodes" entirely.
        masked_z[row] = at.project_channels(masked_filters[store], block_z)
        mean_masked_z[row] = at.project_channels(mean_masked_filter, block_z)
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
    return _Projected(
        prestim=prestim,
        zmask=zmask,
        roi_raw=roi_raw,
        mean_raw=mean_raw,
        mean_stored_z=mean_stored_z,
        masked_z=masked_z,
        mean_masked_z=mean_masked_z,
    )


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


def _plot_courses(
    course, times, window_interval, contrast, labels, conditions, alpha, title, path
):
    """Per-source epoch trial course, both conditions overlaid.

    The line is the mean across participants of each participant's median-over-trials
    course; the band is +/- SEM ACROSS PARTICIPANTS (never across trials, which are
    correlated within a participant). The paradigm's stimulus interval is shaded grey and
    the cell's test window gold, and each panel carries its 4b p-value.
    """
    by_source = contrast.set_index("source")
    n_src = len(labels)
    ncols = min(4, n_src)
    nrows = int(np.ceil(n_src / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axflat = axes.flat
    for s, source in enumerate(labels):
        ax = axflat[s]
        ax.axvspan(
            0.0, AssrEpoch.STIMULUS_DURATION_S, color="0.55", alpha=0.10, lw=0, zorder=0
        )
        ax.axvspan(
            window_interval[0],
            window_interval[1],
            color="#B8860B",
            alpha=0.13,
            lw=0,
            zorder=0,
        )
        ax.axhline(0.0, color="0.45", lw=0.8, zorder=1)
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
        for cond in conditions:
            vals = course[cond][:, s]  # (P, W)
            centre = np.nanmean(vals, axis=0)
            n = np.sum(~np.isnan(vals), axis=0)
            half = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            colour = _CONDITION_COLORS.get(cond, "0.3")
            ax.fill_between(
                times,
                centre - half,
                centre + half,
                color=colour,
                alpha=0.18,
                lw=0,
                zorder=2,
            )
            ax.plot(times, centre, color=colour, lw=1.8, zorder=3, label=cond)
        row = by_source.loc[source]
        is_ref = source in _MASK_LABELS
        ax.set_title(
            f"{source}{'  (reference)' if is_ref else ''}",
            loc="left",
            fontsize=11,
            fontweight="bold" if is_ref else "normal",
        )
        ax.text(
            0.985,
            0.955,
            f"{conditions[0]} > {conditions[1]}\np = {row['p']:.3f}   {row['same sign']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            family="monospace",
            color="0.25" if row["p"] > alpha else "black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.75" if row["p"] > alpha else "black",
                alpha=0.85,
            ),
        )
        if s + ncols >= n_src:
            ax.set_xlabel("Time from onset (s)")
        if s % ncols == 0:
            ax.set_ylabel("response (a.u.)")
    for j in range(n_src, nrows * ncols):
        axflat[j].axis("off")
    axflat[0].legend(loc="lower right", frameon=True, fontsize=9)
    fig.suptitle(title, y=1.0, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    full_mask_column = labels.index(_FULL_LABEL)
    _logger.info(
        f"variant {variant.value} ({'per-recording' if per_recording else 'shared'} "
        f"filters); {int(electrode_mask.sum())} ASSR electrode(s)"
    )

    # ---- 1b. the two extra operators the mean_* / masked_* variants read ----
    # Built once, from the store alone — no cache access — and used by every condition,
    # which is exactly what makes the shared filter unable to favour either of them.
    masked_filters = at.restrict_filters_to_mask(filters, electrode_mask)
    masked_share = at.mask_weight_share(filters, electrode_mask)

    # ---- 2. onsets, caches, paired cohort --------------------------------
    concat_dir = _concatenated_dir(experiment, args.wavelet_data_dir)
    onsets, cache_labels, cache_paths = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        cache_labels[condition] = _cache_participants(concat_dir, label)
        cache_paths[condition] = _wavelet_cache_path(
            args.wavelet_data_dir, experiment, label
        )

    # ---- 3. frequency selections, resolved on BOTH grids -----------------
    # The store and the cache do NOT share a frequency axis, and a bin index only means
    # something against the grid it was resolved on. `--band` reads a band-restricted
    # store whose grid holds only that band's bins, while the cache streamed here is
    # always the broadband one: on a 30-50 Hz store bin 10 is 40 Hz, but bin 10 of the
    # cache is 11 Hz. Resolving once and using the indices on both sides therefore read
    # the references and every projected row out of the wrong part of the spectrum for
    # any band-restricted run, while leaving broadband runs correct because there the
    # two grids coincide. Each side now gets the selection resolved on its OWN grid, so
    # both read the same HERTZ.
    cache_freqs = _cache_frequency_grid(cache_paths, conditions)
    store_selections, cache_selections = {}, {}
    for halfwidth in args.halfwidths:
        name = at.selection_label(args.center_freq, halfwidth)
        store_selections[name] = at.frequency_selection(
            results.freqs, args.center_freq, halfwidth
        )
        cache_selections[name] = at.frequency_selection(
            cache_freqs, args.center_freq, halfwidth
        )
        store_hz = np.asarray(results.freqs)[store_selections[name]]
        cache_hz = cache_freqs[cache_selections[name]]
        _logger.info(
            f"[{name}] store bins {store_hz.min():g}-{store_hz.max():g} Hz "
            f"({store_hz.size}), cache bins {cache_hz.min():g}-{cache_hz.max():g} Hz "
            f"({cache_hz.size})"
        )
        # A band-restricted store can hold only part of the requested window, in which
        # case the stored IC rows average over less of it than the projected rows do.
        # That is not an error — each side averages what it has — but it makes "the same
        # selection" mean two slightly different things, which belongs in the log.
        if (store_hz.min(), store_hz.max()) != (cache_hz.min(), cache_hz.max()):
            _logger.warning(
                f"[{name}] the store covers {store_hz.min():g}-{store_hz.max():g} Hz of "
                f"this selection but the cache covers "
                f"{cache_hz.min():g}-{cache_hz.max():g} Hz, so the stored IC rows "
                "(zscored, stored_prestim) and the projected rows average over "
                "different parts of it. Narrow --halfwidths, or use a store whose band "
                "contains the whole window."
            )
    # The union is over CACHE bins, because it is what the streaming reader indexes.
    union = np.unique(np.concatenate(list(cache_selections.values())))
    union_position = {int(b): i for i, b in enumerate(union)}

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

    # ---- 3b. the cohort-mean filter, over the ANALYSED rows ---------------
    # Every recording of the paired cohort pooled into ONE mean, never a mean per
    # condition: a condition-specific filter would be estimated from the very data the
    # paired contrast tests. Built from the analysed rows rather than the whole store so
    # it describes the cohort the CSV reports on.
    # Deduplicated, because the two variants differ in what a "recording" is: the
    # subject-axis join resolves a distinct row per (participant, condition), while the
    # time-axis join shares one mixing matrix across the conditions and resolves the
    # SAME row twice. Averaging that row twice would not move the mean, but it would
    # report a cohort of 2P where there are P topographies and halve every diagnostic's
    # apparent independence.
    mean_rows = list(
        dict.fromkeys(resolve(p, c) for p in participants for c in conditions)
    )
    cohort = at.cohort_mean_pattern(patterns[mean_rows], electrode_mask)
    mean_masked_filter = at.restrict_filters_to_mask(
        cohort.spatial_filter, electrode_mask
    )
    mean_masked_share = at.mask_weight_share(cohort.spatial_filter, electrode_mask)
    weak_mean, total_mean = at.polarity_weak_count(cohort.input_strength)
    _logger.info(
        f"cohort-mean filter over {cohort.n_recordings} recording(s): "
        f"|U_mean A_mean - I| = {cohort.identity_error:.2e}; "
        f"{weak_mean}/{total_mean} input anchor(s) below "
        f"{at.POLARITY_CORR_FLOOR}"
    )
    for k in range(n_components):
        _logger.info(
            f"  {ic_labels[k]}: corr(mean topo, mask) {cohort.mask_corr[k]:+.3f}, "
            f"median cos to mean {np.median(cohort.cosine_to_mean[:, k]):.3f}, "
            f"|w| kept by the ROI restriction "
            f"{mean_masked_share[k]:.1%} shared / "
            f"{np.median(masked_share[mean_rows][:, k]):.1%} median per-recording"
        )
    low_cosine = [
        ic_labels[k]
        for k in range(n_components)
        if np.median(cohort.cosine_to_mean[:, k]) < 0.3
    ]
    if low_cosine:
        _logger.warning(
            f"the cohort does not agree on the map of {low_cosine} (median cosine to "
            "the mean < 0.3), so those components' mean_* rows describe an average "
            "topography nobody resembles rather than a shared response."
        )

    # ---- 4. project both conditions (one stream each) --------------------
    roi_index = np.flatnonzero(electrode_mask)
    projected: dict[str, _Projected] = {}
    for condition in conditions:
        projected[condition] = _project_condition(
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
            roi_index,
            cohort.spatial_filter,
            masked_filters,
            mean_masked_filter,
            projectors,
        )
    prestim_proj = {c: projected[c].prestim for c in conditions}
    zmask_proj = {c: projected[c].zmask for c in conditions}
    roi_proj = {c: projected[c].roi_raw for c in conditions}

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

    # Per-(participant, component) polarity flip, IC rows only (masks stay +1). The
    # anchor is the correlation between the component's topography and the 0/1 ASSR
    # mask, so it asks whether the pattern is more positive over that area than over the
    # rest of the head — a pattern riding on a global offset cannot flip it. Every pair
    # is flipped on the sign of that correlation, however small; declining to flip a
    # weak one keeps the run's arbitrary sign rather than avoiding a choice. What the
    # weak ones do earn is a count, carried onto every figure whose direction rests on
    # them.
    flip, polarity_weak = {}, {}
    for condition in conditions:
        rows_c = [resolve(p, condition) for p in participants]
        flip_ic, strength = at.polarity_flip(patterns[rows_c], electrode_mask)
        weak, total = at.polarity_weak_count(strength)
        if args.no_polarity_anchor:
            flip_ic = np.ones_like(flip_ic)
            weak = 0
        polarity_weak[condition] = (weak, total)
        _logger.info(
            f"[{condition}] polarity anchor (mask_corr): "
            f"{int((flip_ic < 0).sum())}/{flip_ic.size} pair(s) flipped, "
            f"{weak}/{total} with |corr| < {at.POLARITY_CORR_FLOOR} "
            f"(median |corr| {np.median(strength):.3f})"
        )
        flip[condition] = np.concatenate(
            [flip_ic, np.ones((n_participants, len(_MASK_LABELS)))], axis=1
        )
    # The shared-filter variants read ONE operator whose sign was anchored once when the
    # cohort mean was built, so there is no per-recording sign for them to carry and
    # applying the flip above would scramble a quantity that does not have it.
    shared_flip = np.ones((n_participants, len(labels)))

    # The stored z-scored IC sources, per condition, on that condition's OWN time axis:
    #   * subject-axis join (per_recording): tf_maps[row] IS the single recording.
    #   * time-axis join (tracks): tf_maps[row] is BOTH conditions concatenated, so the
    #     condition's segment is sliced out (a last-axis view) to match the per-condition
    #     cache and onsets — without this the axes disagree (2*T vs T).
    if per_recording:
        _ic_source = {
            c: [results.tf_maps[resolve(p, c)] for p in participants]
            for c in conditions
        }
    else:
        _ic_source = {}
        for c in conditions:
            segment = results.condition_track(results.tf_maps, c)  # (rows, K, F, T_seg)
            _ic_source[c] = [segment[resolve(p, c)] for p in participants]

    # The stored sources always span the whole recording, while the projection can be
    # trimmed (--n_times), so put the stored axis on the projected one before anything
    # concatenates the two. Only a trim is legitimate: a stored track SHORTER than what
    # was projected means the store and the cache do not describe the same recording.
    for c in conditions:
        keep = prestim_proj[c].shape[-1]
        short = [
            p for p, src in zip(participants, _ic_source[c]) if src.shape[-1] < keep
        ]
        if short:
            raise ValueError(
                f"[{c}] the stored source is shorter than the projected cache for "
                f"{short}; the store and the cache disagree on the time axis."
            )
        _ic_source[c] = [src[..., :keep] for src in _ic_source[c]]

    def _z_ic_band(bins: np.ndarray, condition: str) -> np.ndarray:
        # (K, F, T) per participant -> average over the SELECTED FREQUENCY bins (axis 1),
        # keeping every component, to match the raw-projected band collapse.
        return np.stack(
            [src[:, bins, :].mean(axis=1) for src in _ic_source[condition]]
        )  # (P, K, T_condition)

    # One line for every figure that depends on the sign anchor, so a reader never has
    # to go back to the log to find how much of a group mean rests on a coin toss.
    if args.no_polarity_anchor:
        polarity_note = "polarity anchor OFF — component signs are the run's own"
    else:
        polarity_note = (
            "sign anchor corr(topography, ASSR mask): "
            + ", ".join(
                f"{c} {polarity_weak[c][0]}/{polarity_weak[c][1]} weak"
                for c in conditions
            )
            + f" (|corr| < {at.POLARITY_CORR_FLOOR})"
        )
    # The shared-filter figures get their own line: their direction rests on the ONE
    # anchor the cohort mean carries, not on n_participants x n_components of them.
    shared_polarity_note = (
        f"shared filter, sign anchored once when the cohort mean was built over "
        f"{cohort.n_recordings} recording(s); no per-recording flip"
    )

    # ---- 6. the grid: selection x window ---------------------------------
    rows_out: list[dict] = []
    # The spectrum directory follows the STORE that was read, not the cache: a
    # band-restricted decomposition is a different product, so its figures must not
    # land on top of the broadband ones. `--band` names the store, and the canonical
    # layout puts anything but broadband under `bands/` with the band as the first
    # filename token (which `_plot_prefix` supplies below).
    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if args.band is None
        else SpectrumTypeVariants.BANDS.value
    )
    plots_dir = (
        (Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH)
        / _STAGE_DIR
        / f"{store_condition.value}_{music_type.value}"
        / spectrum
        / _ANALYSIS_DIR
        / f"pca_{args.n_pca}"
    )
    # Band as the first token of every filename, per the canonical plot layout.
    plot_prefix = "" if args.band is None else f"{args.band}_"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        # The spatial-filter check figures, at the top of the pca_<N> directory rather
        # than inside a <selection>/<window>/ cell: the operators do not depend on
        # either, so one copy per run is the honest placement. They are what says
        # whether the mean_* / masked_* rows below are worth reading at all.
        _write_filter_plots(
            plots_dir,
            plot_prefix,
            results,
            cohort,
            mean_masked_filter,
            mean_masked_share,
            electrode_mask,
            coordinate_system,
        )

    for sel_name, bins in store_selections.items():
        # `bins` indexes the STORE (it slices the stored IC sources); `take` indexes the
        # projected arrays, whose frequency axis is the streamed union of CACHE bins.
        # The two are resolved on different grids and must never be swapped.
        take = [union_position[int(b)] for b in cache_selections[sel_name]]
        # cut trials once per condition per variant input
        cut: dict[str, dict[str, np.ndarray]] = {name: {} for name in _SIGNAL_VARIANTS}
        roi_snr: dict[str, np.ndarray] = {}
        roi_scale: dict[str, dict[str, float]] = {}
        for condition in conditions:
            onsets_inside = geometry[condition][0]
            prestim_band = prestim_proj[condition][:, :, take, :].mean(
                axis=2
            )  # (P, K+2, T)
            zmask_band = zmask_proj[condition][:, :, take, :].mean(axis=2)  # (P, 2, T)
            z_ic = _z_ic_band(bins, condition)  # (P, K, T)
            # The pre-IVA reference: the PCA mask on the z-scored cache, which is what
            # zmask's second row already is. Appended to EVERY variant's band so the
            # source axis is the same everywhere; only the baseline variants then give
            # it a per-trial reference, which is what makes it differ there from the raw
            # "(PCA)" row. In "zscored" it duplicates that row by construction.
            pca_z_band = zmask_band[:, 1:2]  # (P, 1, T)

            prestim_band = np.concatenate(
                [prestim_band, pca_z_band], axis=1
            )  # (P, K+3, T)
            cut["prestim"][condition], _ = at.cut_trials(
                prestim_band, onsets_inside, pre, post
            )
            zscored_band = np.concatenate(
                [z_ic, zmask_band, pca_z_band], axis=1
            )  # (P, K+3, T)
            cut["zscored"][condition], _ = at.cut_trials(
                zscored_band, onsets_inside, pre, post
            )
            # The stored IC rows on the RAW mask rows: the learned rows come off the
            # decomposition, the references are "prestim"'s own columns rather than
            # anything recomputed, so the reference a component is judged against is
            # identical in the two baseline variants and only the component moves.
            stored_band = np.concatenate(
                [z_ic, prestim_band[:, n_components:]], axis=1
            )  # (P, K+3, T)
            cut["stored_prestim"][condition], _ = at.cut_trials(
                stored_band, onsets_inside, pre, post
            )
            # The four filter variants. Each swaps ONLY the learned rows and keeps
            # "prestim"'s reference columns verbatim, exactly as stored_prestim does, so
            # the fixed quantity every variant is judged against never moves and a
            # difference between two rows is a difference of spatial filters alone.
            reference_band = prestim_band[:, n_components:]  # (P, 3, T)
            for name, learned in (
                ("mean_prestim", projected[condition].mean_raw),
                ("mean_stored_prestim", projected[condition].mean_stored_z),
                ("masked_prestim", projected[condition].masked_z),
                ("mean_masked_prestim", projected[condition].mean_masked_z),
            ):
                learned_band = learned[:, :, take, :].mean(axis=2)  # (P, K, T)
                cut[name][condition], _ = at.cut_trials(
                    np.concatenate([learned_band, reference_band], axis=1),
                    onsets_inside,
                    pre,
                    post,
                )
            # The binary-ROI row, built the equal-weight way: each electrode is
            # referenced to its OWN pre-stimulus window before the ROI is averaged, and
            # the average is then referenced to its own again so the row lands back in
            # units of its own baseline SD. Averaging raw power first, as the columns
            # above still do for the PCA reference, silently weights each electrode by
            # its own power level. Kept separate from `cut` because it is already
            # normalised and must skip baseline_normalise below.
            roi_band = roi_proj[condition][:, :, take, :].mean(axis=2)  # (P, R, T)
            roi_cut, _ = at.cut_trials(roi_band, onsets_inside, pre, post)
            roi_snr[condition], roi_scale[condition] = at.roi_channelwise_snr(
                roi_cut, baseline_mask
            )

        # Put every condition's ROI row on ONE scale. Each call derived its own from its
        # own participants, and those differ between conditions — leaving them apart
        # would rescale a participant's two conditions differently and corrupt the very
        # paired difference the contrast tests. Rescaling is exact, not a re-fit:
        # snr / shared == (snr / own) * (own / shared).
        shared_roi_scale = float(
            np.median([d["roi_baseline_sd"] for d in roi_scale.values()])
        )
        for condition in conditions:
            roi_snr[condition] = roi_snr[condition] * (
                roi_scale[condition]["roi_baseline_sd"] / shared_roi_scale
            )
        _logger.info(
            f"[{sel_name}] equal-weight ROI over {roi_index.size} electrode(s): shared "
            f"baseline SD {shared_roi_scale:.3f} "
            f"(~{1.0 / shared_roi_scale**2:.1f} effective channels); per-condition "
            + ", ".join(
                f"{c} {roi_scale[c]['roi_baseline_sd']:.3f} "
                f"(spread {roi_scale[c]['participant_sd_spread']:.2f}x across "
                f"participants, NOT applied)"
                for c in conditions
            )
            + f"; max |per-channel z| "
            f"{max(d['max_abs_z1'] for d in roi_scale.values()):.0f}"
        )

        for lo, hi in args.stimulus_intervals:
            # The paradigm's own interval is read HALF-OPEN, as AssrEpoch.stimulus_mask
            # defines it — 500 ms at 250 Hz is exactly 125 samples, and `<= hi` would
            # add the first sample after the stimulus ended. It is also how the stage-07
            # read-out reads its default window, so the two stages' default cells cover
            # the same samples and their shared reference row is comparable to the digit
            # rather than to the third decimal. Any other interval is taken as typed.
            if (lo, hi) == (0.0, AssrEpoch.STIMULUS_DURATION_S):
                window = AssrEpoch.stimulus_mask(times)
            else:
                window = (times >= lo) & (times <= hi)
            if not window.any():
                raise ValueError(f"window {lo}-{hi}s selects no epoch sample.")
            window_label = f"{lo:g}-{hi:g}s"

            for signal in args.signal_variants:
                # Which sign convention this variant's learned rows carry. A shared
                # operator was anchored once when it was built, so it takes no
                # per-recording flip; everything else does.
                shared_filter = signal in _SHARED_FILTER_VARIANTS
                # value (window-reduced) and course (full epoch) per condition, both from
                # the SAME normalised trials, anchored by that variant's flip.
                value, course = {}, {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+3, N, W)
                    signs = shared_flip if shared_filter else flip[condition]
                    if signal in _BASELINE_VARIANTS:
                        normed, _rel, _pos = at.baseline_normalise(
                            trials, baseline_mask
                        )
                        # Swap in the equal-weight ROI row. Only the baseline variants
                        # get it: "zscored" has no per-trial baseline to build it on.
                        normed = normed.copy()
                        normed[:, full_mask_column] = roi_snr[condition]
                    else:
                        normed = trials
                    per_trial = normed[..., window].mean(axis=-1)  # (P, K+3, N)
                    value[condition] = np.median(per_trial, axis=2) * signs
                    course[condition] = np.median(normed, axis=2) * signs[:, :, None]

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
                        plot_prefix,
                        cell,
                        value,
                        course,
                        times,
                        (lo, hi),
                        conditions,
                        labels,
                        ic_labels,
                        args.alpha,
                        rows_out,
                        n_participants,
                        shared_polarity_note if shared_filter else polarity_note,
                    )

    # ---- 7. one CSV for the whole grid -----------------------------------
    results_root = (
        Path(args.results_dir)
        if args.results_dir
        else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
    )
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / (
        f"assr_snr_grid__{variant.value}__{spectrum}__pca{args.n_pca}.csv"
    )
    frame = pd.DataFrame(rows_out)
    frame.to_csv(csv_path, index=False)
    _logger.info(f"saved {csv_path} ({len(frame)} test rows)")
    print(
        f"\n{len(frame)} tests over {len(store_selections)} selection(s) x "
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


def _write_filter_plots(
    plots_dir,
    plot_prefix,
    results,
    cohort,
    mean_masked_filter,
    mean_masked_share,
    electrode_mask,
    coordinate_system,
) -> None:
    """The two spatial-filter check figures, written once per run.

    They sit at the top of the ``pca_<N>`` directory rather than inside a grid cell,
    because the operators they draw depend on neither the frequency selection nor the
    stimulus window. What they are for: a ``mean_*`` row is only readable if the cohort
    agreed on the map that was averaged, and a ``masked_*`` row only means "a cleaner
    view of the same component" if the component had weight on the anchor electrodes to
    begin with. Both figures answer their question by eye, in one page.

    A store written without channel names cannot place a value on a scalp, so the
    figures are skipped with a warning rather than failing the run — the CSV, which is
    the product, does not depend on them.

    :param plots_dir: The run's ``pca_<N>`` directory.
    :param plot_prefix: Band token plus underscore, or empty for a broadband run.
    :param results: The loaded store, for :meth:`topo_info`.
    :param cohort: The :class:`~src.analysis.assr_trials.CohortMeanFilter`.
    :param mean_masked_filter: ``(components, channels)`` restricted shared filter.
    :param mean_masked_share: ``(components,)`` share of its L2 weight kept.
    :param electrode_mask: Boolean ``(channels,)`` anchor mask.
    :param coordinate_system: Montage the stored channel names belong to.
    """
    try:
        info = results.topo_info(coordinate_system)
    except (ValueError, FileNotFoundError) as exc:
        _logger.warning(
            f"no topography layout for this store ({exc}); skipping the spatial-filter "
            "check figures. The grid itself is unaffected."
        )
        return
    plot_cohort_mean_topomaps(
        cohort.pattern,
        info,
        electrode_mask,
        mask_corr=cohort.mask_corr,
        cosine_to_mean=cohort.cosine_to_mean,
        n_recordings=cohort.n_recordings,
        save_path=plots_dir / f"{plot_prefix}{_MEAN_FILTER_FIGURE}",
    )
    plot_masked_filter_topomaps(
        mean_masked_filter,
        info,
        weight_share=mean_masked_share,
        n_mask_channels=int(np.asarray(electrode_mask).sum()),
        variant="mean_masked_prestim",
        save_path=plots_dir / f"{plot_prefix}{_MEAN_MASKED_FILTER_FIGURE}",
    )
    plt.close("all")
    _logger.info(f"wrote the spatial-filter check figures to {plots_dir}")


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
    ic_labels,
    alpha,
    rows_out,
    n_participants,
    polarity_note,
) -> None:
    """Three figures for one grid cell, built from the rows just accumulated.

    Each frequency selection (band range) and stimulus window (time interval) gets its
    own subdirectory, ``<selection>/<window>/``, so the two signal variants of a cell sit
    together and the grid is browsable by band and window; the variant stays in the
    filename. The figure titles keep the full ``variant__selection__window`` context.

    *plot_prefix* is the band token plus an underscore for a band-restricted store, and
    empty for a broadband one -- the canonical layout wants the band as the first token
    of every filename under ``bands/``.

    *polarity_note* names how many (participant, component) sign anchors were decided
    on a weak topography correlation. It rides on the trial-course figure because that
    is the one whose shape a wrongly flipped participant would visibly cancel.
    """
    tag = f"{cell['variant']}__{cell['selection']}__{cell['window']}"
    variant = cell["variant"]
    cell_dir = plots_dir / cell["selection"] / cell["window"]
    cell_dir.mkdir(parents=True, exist_ok=True)
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
        cell_dir / f"{plot_prefix}pvalue_summary__{variant}.png",
    )
    _plot_snr(
        cell_tables,
        value,
        conditions,
        ic_labels,
        alpha,
        f"IC SNR vs reference, per condition — {tag}, n={n_participants}",
        cell_dir / f"{plot_prefix}snr_by_condition__{variant}.png",
    )
    _plot_courses(
        course,
        times,
        window_interval,
        cell_tables["contrast"],
        labels,
        conditions,
        alpha,
        f"Trial course per spatial filter — {tag}, n={n_participants}\n{polarity_note}",
        cell_dir / f"{plot_prefix}trial_course_by_source__{variant}.png",
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
