"""
Full-recording ASSR SNR / condition-contrast grid, for either IVA condition-comparison
variant.

This is the CLI counterpart of
``notebooks/06-iva-condition-comparison/iva_component_analysis_joined.ipynb`` and its
``_tracks`` sibling, run on the **full** wavelet cache rather than the 12 s notebook
subset. It reproduces the whole current workflow — the ASSR references, both signal
variants, and all three test families — and sweeps the frequency-selection x
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
  the per-(channel, frequency) rescaling z-scoring adds.
* **Two signal variants**, differing in how the LEARNED rows are read and in nothing
  else. **The references never move**: both variants carry the same three reference
  columns (the masks projected through the RAW cache, each trial referenced to its own
  pre-stimulus window, the full mask swapped for the equal-weight ROI row), which is
  what makes the pair one comparison rather than two.

  - ``full`` takes the IC rows from the stored sources — the decomposition's own
    output on the time-z-scored data it was fitted on — and gives every trial a
    pre-stimulus baseline.
  - ``roi_masked`` is each recording's own filter with every weight **outside the
    ASSR electrodes zeroed**, on the z-scored cache. It sits between the two things the
    grid already has — the binary reference reads those electrodes with equal weight,
    ``full`` reads the whole head with learned weights, this reads only those
    electrodes with learned weights — and so separates a better weighting *inside* the
    anchor area from access to signal *outside* it, which no other row can tell apart.
    It reads the z-scored wavelet, not raw power, because the filters were estimated on
    it; and no PCA, because ``P^T P`` would mix every channel back in. Zeroing channels
    does not change a component's sign, so it keeps the per-recording flip.

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
filter per participant across the conditions; ``--variant iva_channel_joined_separate`` has a
separate
topography per recording, so every projection and polarity anchor is resolved per
(participant, condition) through ``results.row``. Nothing else differs.

Outputs (per run, all grid cells in one file):

* ``results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv`` — every test
  row, tagged with its signal variant, frequency selection and stimulus window.
* ``results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>__values.npz`` —
  the per-participant values those tests compare.
* ``plots/06-iva-condition-comparison/<Condition>_<Music>/<spectrum>/assr_snr_grid/
  pca_<N>/<selection>/<window>/`` — one subdirectory per frequency selection (band range)
  and stimulus window (time interval), each holding a p-value summary, an SNR-by-condition
  figure and a per-source trial-course figure for each signal variant.

``<spectrum>`` is ``broadband`` by default and ``bands`` when ``--band`` names a
band-restricted store (with the band as the first token of every filename), so a run
over a band-restricted decomposition never lands on top of the broadband one.

Examples::

    # The job grid: both frequencies x both windows, both signal variants.
    python scripts/run_assr_snr_grid.py --experiment assr --variant iva_channel_joined_separate \\
        --n_pca 5 --halfwidths 0 5 --stimulus_intervals 0:0.5 0.2:0.5

    # A quick shape check on the first 12 s.
    python scripts/run_assr_snr_grid.py --experiment assr --n_pca 5 --n_times 3000 \\
        --skip_plots

No multiplicity correction is applied; the CSV carries ``effect`` and ``same sign`` so a
row can be weighed against its neighbours rather than a threshold alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import assr_trials as at  # noqa: E402
from src.analysis.assr_variants import (  # noqa: E402
    ProjectedVariants as _Projected,
    project_condition as _project_condition,
    row_resolver as _row_resolver,
)
from src.analysis import iva_quality  # noqa: E402
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
from src.io.iva_store import load_iva_components  # noqa: E402
from src.io.loading import (  # noqa: E402
    assr_electrode_mask,
    concatenated_dir as _concatenated_dir,
)
from src.io.saving import save_snr_values  # noqa: E402
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.assr_snr_plots import (  # noqa: E402
    plot_cell as _plot_cell,
    plot_courses as _plot_courses,
    plot_snr as _plot_snr,
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
#: the unmixing. It is the reference-side analogue of the "full" IC rows — z-scored
#: input WITH a per-trial baseline — so reading it beside "ASSR-mask (PCA)" separates
#: the channel reduction from the per-(channel, frequency) rescaling z-scoring adds.
_PCA_Z_LABEL = at.PCA_Z_MASK_LABEL
_MASK_LABELS = at.MASK_LABELS
#: ``full`` reads the stored sources, ``roi_masked`` each filter restricted to the ASSR
#: electrodes; both reference every trial to its own pre-stimulus window.
_SIGNAL_VARIANTS = ("full", "roi_masked")


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
        default=IvaVariants.CHANNEL_JOINED_SEPARATE.value,
        choices=[
            IvaVariants.CHANNEL_JOINED_SEPARATE.value,
            IvaVariants.CHANNEL_JOINED_TRACKS.value,
        ],
        help="channel_joined = per-recording topography; _tracks = shared per participant.",
    )
    stored.add_argument(
        "--store_condition",
        default=None,
        help="Store condition; defaults per variant (JoinedSeparate / JoinedTracks).",
    )
    stored.add_argument("--music_type", default=MusicTypeVariants.ASSR.value)
    stored.add_argument("--band", default=None)
    stored.add_argument("--n_pca", type=int, default=5)
    stored.add_argument("--store_dir", type=Path, default=None)

    data = parser.add_argument_group("the wavelet cache and the cohort")
    data.add_argument("--wavelet_data_dir", type=Path, default=None)
    data.add_argument(
        "--concat_dir",
        type=Path,
        default=None,
        help=(
            "Processed-data root the concatenated arrays and their onset sidecars are "
            "read from. Separate from --wavelet_data_dir, so a small wavelet strip can "
            "be read without moving the onsets."
        ),
    )
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
            "Which readings to test side by side; both keep the same reference rows, "
            "so only the learned rows move. 'full' reads the IC rows off the "
            "decomposition; 'roi_masked' zeroes every filter weight outside the ASSR "
            "electrodes, which separates a better weighting INSIDE that area from "
            "access to signal outside it."
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
# Inputs (cache/onset conventions shared with the rest of the ASSR scripts)
# ---------------------------------------------------------------------------


def _load_onsets(concat_dir: Path, label: str) -> np.ndarray:
    path = concat_dir / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(
            f"No stimulus onsets at {path}; the stimulus alignment writes them next to "
            "the concatenated array."
        )
    return np.load(path).astype(int)


def _cache_frequency_grid(wavelet_sets: dict, conditions: list[str]) -> np.ndarray:
    """The wavelet cache's own frequency axis, checked identical across conditions.

    Separate from the store's axis on purpose: a band-restricted store holds only its
    band's bins, while the cache is always the broadband one, so an index resolved on
    one grid is meaningless on the other.

    :param wavelet_sets: Condition → :class:`~src.io.wavelet_store.WaveletSet`.
    :param conditions: Conditions to read, in order.
    :return: The shared ``(F,)`` frequency grid in Hz.
    :raises ValueError: If the conditions' caches were built on different grids, which
        would make one bin index mean two frequencies.
    """
    grids = {condition: wavelet_sets[condition].freqs for condition in conditions}
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
    return ConditionVariants.JOINED_SEPARATE


# ---------------------------------------------------------------------------
# Row bookkeeping — the one thing the two variants differ on
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Projection — one streaming pass produces every signal variant's input
# ---------------------------------------------------------------------------


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

    # ---- 1b. the restricted operator the roi_masked variant reads ----------
    masked_filters = at.restrict_filters_to_mask(filters, electrode_mask)
    masked_share = at.mask_weight_share(filters, electrode_mask)

    # ---- 2. onsets, caches, paired cohort --------------------------------
    concat_dir = _concatenated_dir(experiment, args.concat_dir)
    onsets, wavelet_sets, cached = {}, {}, {}
    for condition in conditions:
        label = f"{condition}_{music_type.value}"
        onsets[condition] = _load_onsets(concat_dir, label)
        wavelet_sets[condition] = find_wavelet_set(
            wavelet_label_dir(args.wavelet_data_dir, experiment, label)
        )
        cached[condition] = set(wavelet_sets[condition].participants)

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
                "('full') and the projected rows average over "
                "different parts of it. Narrow --halfwidths, or use a store whose band "
                "contains the whole window."
            )
    # The union is over CACHE bins, because it is what the wavelet reader indexes.
    union = np.unique(np.concatenate(list(cache_selections.values())))
    union_position = {int(b): i for i, b in enumerate(union)}

    store_participants = list(dict.fromkeys(results.participants))
    participants = [
        p
        for p in store_participants
        if all(resolve_ok(results, p, c, per_recording) for c in conditions)
        and all(p in cached[c] for c in conditions)
    ]
    if not participants:
        raise ValueError(
            "No participant is present in the store under every condition and every cache."
        )
    dropped = [p for p in store_participants if p not in participants]
    if dropped:
        _logger.warning(f"dropped (not paired / not cached): {dropped}")
    n_participants = len(participants)

    analysed_rows = list(
        dict.fromkeys(resolve(p, c) for p in participants for c in conditions)
    )
    for k in range(n_components):
        _logger.info(
            f"  {ic_labels[k]}: |w| kept by the ROI restriction, median per-recording "
            f"{np.median(masked_share[analysed_rows][:, k]):.1%}"
        )

    # ---- 4. project both conditions (one stream each) --------------------
    roi_index = np.flatnonzero(electrode_mask)
    projected: dict[str, _Projected] = {}
    for condition in conditions:
        projected[condition] = _project_condition(
            wavelet_sets[condition],
            condition,
            participants,
            resolve,
            binary,
            pca_rows,
            union,
            store_channels,
            args.n_times,
            roi_index,
            masked_filters,
        )

    # ---- 5. epoch geometry (shared window) -------------------------------
    sfreq = results.sfreq
    geometry = {
        c: at.epoch_geometry(onsets[c], projected[c].reference_raw.shape[-1], sfreq)
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
        keep = projected[c].reference_raw.shape[-1]
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

    # ---- 6. the grid: selection x window ---------------------------------
    rows_out: list[dict] = []
    snr_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
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
    plots_dir = ProjectPaths.analysis_plots_dir(
        _STAGE_DIR,
        f"{store_condition.value}_{music_type.value}",
        spectrum=spectrum,
        analysis_type=_ANALYSIS_DIR,
        n_pca=args.n_pca,
        root=args.save_dir,
    )
    # Band as the first token of every filename, per the canonical plot layout.
    plot_prefix = "" if args.band is None else f"{args.band}_"
    if not args.skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

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
            proj = projected[condition]
            # The reference rows, identical in both variants: the masks on the RAW
            # cache, then the PCA mask on the z-scored cache.
            reference_band = np.concatenate(
                [
                    proj.reference_raw[:, :, take, :].mean(axis=2),
                    proj.pca_z[:, :, take, :].mean(axis=2),
                ],
                axis=1,
            )  # (P, 3, T)
            learned = {
                "full": _z_ic_band(bins, condition),
                "roi_masked": proj.masked_z[:, :, take, :].mean(axis=2),
            }  # each (P, K, T)
            for name, learned_band in learned.items():
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
            roi_band = proj.roi_raw[:, :, take, :].mean(axis=2)  # (P, R, T)
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
                # value (window-reduced) and course (full epoch) per condition, both from
                # the SAME normalised trials, anchored by the per-recording flip.
                value, course = {}, {}
                for condition in conditions:
                    trials = cut[signal][condition]  # (P, K+3, N, W)
                    normed, _rel, _pos = at.baseline_normalise(trials, baseline_mask)
                    # Swap in the equal-weight ROI row.
                    normed = normed.copy()
                    normed[:, full_mask_column] = roi_snr[condition]
                    per_trial = normed[..., window].mean(axis=-1)  # (P, K+3, N)
                    signs = flip[condition]
                    value[condition] = np.median(per_trial, axis=2) * signs
                    course[condition] = np.median(normed, axis=2) * signs[:, :, None]

                cell = {
                    "variant": signal,
                    "selection": sel_name,
                    "window": window_label,
                }
                snr_values[(signal, sel_name, window_label)] = value
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
                        polarity_note,
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
    values_path = csv_path.with_name(f"{csv_path.stem}__values.npz")
    save_snr_values(values_path, snr_values, participants, labels)
    _logger.info(f"saved {csv_path} and {values_path.name} ({len(frame)} test rows)")
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
    rows_out,
    cell,
    value,
    conditions,
    labels,
    ic_labels,
    contrast_alt,
    snr_alt,
    mask_labels=_MASK_LABELS,
) -> None:
    """Append every test row for one grid cell to *rows_out*.

    *mask_labels* are the reference rows of *labels* the learned rows are tested against.
    """
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
    for mask in mask_labels:
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
    for mask in mask_labels:
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
