"""
CLI script reproducing
``notebooks/06-iva-condition-comparison/wavelet_iva_channel_joined.ipynb``:
the channel-as-independent IVA-G decomposition of wavelet power run on the
**joined** condition (:attr:`~src.definitions.fields.ConditionVariants.JOINED`),
with Placebo and Psilocybin pooled on the **subject** axis, followed by the
Placebo/Psilocybin comparison of the recovered components.

Pipeline, identical to the single-condition
``run_wavelet_iva_channel.py`` except for what sits on the subject axis::

    datasets = recordings          # participant × condition (2P entries)
    samples  = time × frequency    # the axis IVA aligns across ALL recordings
    mixing   = channel             # free per recording

Every *recording* is one IVA dataset, so each participant appears twice and a
participant's two recordings get their own mixing matrices. The recovered
spectro-temporal sources ``(F, T)`` are therefore shared by both conditions —
one aligned set describes the whole pooled cohort — while the channel
topographies ``(C,)`` stay per recording, which is what makes a per-condition
contrast of the patterns meaningful.

**No new wavelet cache is written.** The pooled tensor is assembled in memory
from the existing per-condition caches by
:func:`~scripts.notebook_helpers.load_joined_condition_wavelets`, which is
possible because the alignment was fitted over every recording of both
conditions, so they already share one time base. Asking for a ``Joined_*``
label directly would instead write a second, redundant copy of the cache.

Pass ``--subset_cache`` to additionally keep a **per-extent** copy of each
condition's trimmed tensor under
``notebooks/03-wavelet-analysis/wavelet_cache/<experiment>/``: the first run at
a given extent writes it, every later run reads it back and never decompresses
the multi-GB source cache. Worth it here, since a joined load reads two of them.

Sign handling, in the order it happens:

1. ``iva_g`` fixes each component's sign only **per recording**.
2. :func:`~src.analysis.wavelet_ica.align_iva_component_signs` resolves it from
   IVA's own ``Sigma_N`` cross-dataset covariance.
3. :func:`~src.analysis.assr_trials.polarity_flip` re-decides it from the
   correlation between each component's **topography** and the binary ASSR
   electrode mask, and has the final say — so a positive value means more power
   over that area for every recording, which is what the contrasts need. Both
   criteria are per-``(component, recording)`` flips, so composing them is well
   defined.

Resolving the sign is not cosmetic on a pooled dataset: the flips fall
arbitrarily across the two condition blocks, so an unresolved sign
*manufactures* a condition difference. The signs are applied to the channel
patterns as well as the maps, so a component's topography and its TF map never
disagree about which way is up.

Figures, in three views (see
:mod:`src.visualization.iva_condition_plots` for the scaling rules):

  - ``condition_mean_topomaps`` — condition-mean channel topographies, rows =
    conditions plus their difference, columns = components
  - ``condition_mean_tf_maps`` — the same grid for the whole-recording TF maps
  - ``condition_mean_tf_maps_onset`` — the same grid **averaged over stimuli**
    (ASSR only): a fixed epoch cut around every stimulus onset, so the contrast
    becomes a contrast of stimulus responses rather than of whole recordings
  - ``participants/condition_topomap_ic<k>_participants_*`` — one figure per
    component: Placebo on the first row, Psilocybin on the second, one column
    per participant ordered by ID, so a participant's two recordings are
    vertically paired
  - ``participants/condition_tf_ic<k>_participants_*`` and its ``onset_``
    counterpart — the TF versions of the same layout

Output (canonical per-condition layout)::

    plots/06-iva-condition-comparison/Joined_<MusicType>/
        broadband/iva_channel_joined/pca_<n_pca>/     # default (no --band)
            condition_mean_topomaps.png
            condition_mean_tf_maps.png
            condition_mean_tf_maps_onset.png
            participants/*.png
        bands/iva_channel_joined/pca_<n_pca>/          # when --band <name>
            <band>_condition_mean_topomaps.png
            ...

``--n_pca`` reduces the **channel** axis, so it must be ≤ the number of
channels (≈195 after preprocessing).

Examples::

    # ASSR, all matched participants, ten components
    python scripts/run_iva_condition_comparison.py \\
        --experiment assr --n_pca 10 --reuse_wavelets --subset_cache

    # Fast exploration on a channel/time subset, cached for the next run
    python scripts/run_iva_condition_comparison.py \\
        --experiment assr --n_pca 10 --n_channels 32 --n_times 3000 \\
        --n_pairs 5 --reuse_wavelets --subset_cache

Pass ``--store_components`` to additionally keep the recovered components under
``data/processed/<experiment>/iva_results/<Condition>/``: the sign-aligned TF maps
and channel topographies, with the participant and condition of every row, so the
comparison can be redone (or taken further) without repeating the decomposition.
Read them back with :func:`src.io.iva_store.load_iva_components`.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    add_iva_store_args,
    add_wavelet_grid_args,
    resolve_wavelet_dir,
)
from scripts.notebook_helpers import (  # noqa: E402
    load_joined_condition_wavelets,
    resolve_notebook_wavelet_cache_dir,
)
from src.analysis import iva_quality  # noqa: E402
from src.analysis.iva_condition_comparison import (  # noqa: E402
    apply_component_signs,
    decompose_channel_iva,
    slice_to_band,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    CoordinateSystems,
    ExclusionCategories,
    ExperimentNames,
    FrequencyBandNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.io.iva_store import save_iva_components  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_tf_maps,
    plot_condition_mean_topomaps,
    plot_participant_condition_tf_maps,
    plot_participant_condition_topomaps,
)
from src.analysis import assr_trials as at  # noqa: E402
from src.io.loading import assr_electrode_mask  # noqa: E402
from src.visualization.iva_quality_plots import topo_info_subset  # noqa: E402

_logger = logging.getLogger(__name__)

#: Stage directory under ``plots/``.
_STAGE_DIR = "06-iva-condition-comparison"

#: Analysis-type subdirectory, naming the independent (mixing) axis and the join.
_ANALYSIS_DIR = IvaVariants.CHANNEL_JOINED.value

#: What the final sign alignment was, printed on every figure.
_ALIGNMENT_NOTE = "corr(topography, ASSR electrode mask), per (recording, component)"
#: What the figures say when the montage carries no anchor electrode and the
#: signs were therefore stored as iva_g returned them. Naming it matters: an
#: unaligned group mean partly cancels, and a reader must not take a flat map
#: for an absent effect.
_ALIGNMENT_NOTE_NONE = "NOT sign-aligned (no ASSR anchor electrode present)"

#: Frequency (Hz) marked on every TF panel — the ASSR stimulation frequency.
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]

#: Minimum onsets that must fit the time window for the stimulus average to be drawn.
_MIN_ONSETS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the channel-as-independent IVA decomposition on the joined "
            "condition (Placebo + Psilocybin pooled on the subject axis) and draw "
            "the per-component condition comparison. Reproduces "
            "wavelet_iva_channel_joined.ipynb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentNames],
        default=ExperimentNames.PSILO_MUSIC.value,
        help="Which experiment dataset to analyse.",
    )
    parser.add_argument(
        "--music_type",
        nargs="+",
        choices=[mt.value for mt in MusicTypeVariants],
        default=None,
        help=(
            "One or more music types, each analysed in its own joined run. When "
            "omitted, defaults to CLASSIC + PSYTRANCE for the psilo_music "
            "experiment and ASSR for the assr experiment."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[c.value for c in REAL_CONDITIONS],
        default=[c.value for c in REAL_CONDITIONS],
        help=(
            "Conditions to pool, in the order their blocks appear on the subject "
            "axis and their rows appear in every figure."
        ),
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=10,
        help=(
            "Per-recording PCA dim over CHANNELS before IVA-G (square mixing "
            "requirement), i.e. the number of IVA components. Must be ≤ the number "
            "of channels."
        ),
    )
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=None,
        help=(
            "1-based component numbers to draw (matching the 'IC <k>' labels). "
            "Defaults to every component."
        ),
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=None,
        help=(
            "Keep only the first N matched participants, by ID. Counts "
            "PARTICIPANTS, not recordings: each one brings a recording per "
            "condition, so the subject axis is 2N. Defaults to every matched "
            "participant."
        ),
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=None,
        help="Keep only the first N channels. Defaults to all of them.",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=None,
        help=(
            "Keep only the first N time samples. The IVA sample axis is "
            "n_freqs × this, so it drives peak memory. Defaults to the whole "
            "aligned time axis."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for per-recording PCA and IVA W_init.",
    )
    parser.add_argument(
        "--iva_opt_approach",
        choices=["gradient", "newton", "quasi"],
        default="newton",
        help="IVA-G optimisation method.",
    )
    parser.add_argument(
        "--iva_max_iter",
        type=int,
        default=1024,
        help="Maximum IVA-G iterations.",
    )
    parser.add_argument(
        "--iva_w_diff_stop",
        type=float,
        default=1e-6,
        help="IVA-G convergence threshold on |ΔW|.",
    )
    add_wavelet_grid_args(parser)
    parser.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help=(
            "Base data directory holding the per-condition wavelet caches. The "
            "'<experiment>/wavelets' suffix is appended automatically."
        ),
    )
    parser.add_argument(
        "--reuse_wavelets",
        action="store_true",
        help="Reuse the cached per-condition transforms when available.",
    )
    parser.add_argument(
        "--subset_cache",
        action="store_true",
        help=(
            "Keep a per-extent copy of each condition's trimmed wavelet tensor "
            "under notebooks/03-wavelet-analysis/wavelet_cache/<experiment>/, so "
            "later runs at the same extent skip decompressing the source cache. "
            "Shared with every other wavelet workflow."
        ),
    )
    parser.add_argument(
        "--no_reuse_subset_cache",
        action="store_true",
        help=(
            "With --subset_cache, recompute from the source cache and overwrite "
            "the subset entry instead of reading it — the escape hatch for a stale "
            "cache."
        ),
    )
    parser.add_argument(
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the pooled broadband tensor is "
            "sliced to the band's frequency range before IVA, and plots are "
            "written to 'bands/' with a '<band>_' prefix."
        ),
    )
    parser.add_argument(
        "--skip_onset_average",
        action="store_true",
        help=(
            "Skip the stimulus-averaged TF comparison. It is skipped anyway for "
            "experiments without stimulus annotations."
        ),
    )
    parser.add_argument(
        "--min_onsets",
        type=int,
        default=_MIN_ONSETS_DEFAULT,
        help=(
            "Minimum stimulus epochs that must fit the time window before the "
            "stimulus-averaged comparison is drawn."
        ),
    )
    parser.add_argument(
        "--no_difference_row",
        action="store_true",
        help="Omit the between-condition difference row from the condition-mean grids.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Base directory for output plots. Defaults to the project plots/ root.",
    )
    add_iva_store_args(parser)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for data loading.",
    )
    parser.add_argument(
        "--coordinate_system",
        default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value,
        help="Montage the ASSR electrode mask is read for.",
    )
    parser.add_argument(
        "--lenient_mask",
        action="store_true",
        help="Accept the intersection when anchor electrodes are missing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _onset_average(
    sources: np.ndarray,
    onsets: np.ndarray | None,
    *,
    label: str,
    n_times: int,
    sfreq: float,
    min_onsets: int,
) -> tuple[np.ndarray, np.ndarray, list[float]] | None:
    """Average a fixed epoch around every stimulus onset.

    The epoch is the paradigm window from
    :class:`~src.definitions.constants.AssrEpoch` via
    :func:`~src.analysis.iva_quality.onset_window` — the same call the notebook and
    the ``--quality`` path make, so no onset-locked ASSR figure is cut on a different
    window than another. No baseline subtraction: ``zscore_by_time`` already zeroed
    each series' time-mean, so the pre-onset interval reads ≈ 0 by construction.

    :param sources: ``(S, K, F, T)`` whole-recording component maps.
    :param onsets: Stimulus onset sample indices, or ``None`` when the experiment has
        no annotations.
    :param label: Dataset label, for logging.
    :param n_times: Length of the usable time axis.
    :param sfreq: Sampling frequency in Hz.
    :param min_onsets: Minimum epochs that must fit before averaging is worthwhile.
    :return: ``(onset_tf, epoch_times, epoch_marks)`` — the ``(S, K, F, W)`` average,
        its time axis in seconds with ``0`` at the onset, and the onset/offset times to
        mark. ``None`` when there is nothing to average.
    """
    if onsets is None or len(onsets) == 0:
        _logger.info(f"[{label}] No stimulus onsets; skipping the stimulus average.")
        return None

    onsets_in = np.asarray(onsets)[np.asarray(onsets) < n_times].astype(int)
    if onsets_in.size == 0:
        _logger.info(f"[{label}] No onset inside the time window.")
        return None

    pre, post = iva_quality.onset_window(onsets_in, n_times, sfreq)
    n_fitting = int(((onsets_in - pre >= 0) & (onsets_in + post <= n_times)).sum())
    if n_fitting < min_onsets:
        _logger.warning(
            f"[{label}] Only {n_fitting} stimulus epoch(s) fit the {n_times}-sample "
            f"window, need {min_onsets}; skipping the stimulus average. Raise "
            "--n_times or lower --min_onsets."
        )
        return None

    onset_tf, n_used = iva_quality.epoch_average(sources, onsets_in, pre, post)
    epoch_times = np.arange(-pre, post) / sfreq
    # Each frequency referenced to its own pre-onset mean. The z-scoring puts the
    # pre-onset interval near 0 over the WHOLE recording but not within any one epoch,
    # so this is what makes the map a change rather than a level. Mean only — see
    # subtract_epoch_baseline for why no divisor.
    onset_tf = iva_quality.subtract_epoch_baseline(onset_tf, epoch_times < 0.0)
    # Onset (dashed) then the stimulus offset (dotted), clamped into the epoch.
    marks = [0.0, min(AssrEpoch.STIMULUS_DURATION_S, float(epoch_times[-1]))]
    _logger.info(
        f"[{label}] Onset-averaged {onset_tf.shape} from {n_used} epoch(s); window "
        f"[{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s, each frequency referenced "
        "to its own pre-onset mean"
    )
    if post < int(round(AssrEpoch.POST_ONSET_S * sfreq)):
        _logger.info(
            f"[{label}] post-onset span trimmed from {AssrEpoch.POST_ONSET_S} s to "
            f"{post / sfreq:.3f} s by the shortest inter-onset gap."
        )
    return onset_tf, epoch_times, marks


def run_condition_comparison(
    music_type: MusicTypeVariants,
    *,
    experiment_name: ExperimentNames,
    conditions: list[ConditionVariants],
    exclusion_categories: list[ExclusionCategories],
    freqs_full: np.ndarray,
    wavelet_dir: Path,
    reuse_wavelets: bool,
    subset_cache_dir: Path | None,
    reuse_subset_cache: bool,
    n_pca: int,
    components: list[int] | None,
    n_pairs: int | None,
    n_channels: int | None,
    n_times: int | None,
    random_state: int,
    iva_opt_approach: str,
    iva_max_iter: int,
    iva_w_diff_stop: float,
    band: str | None,
    show_difference: bool,
    onset_average: bool,
    min_onsets: int,
    save_root: Path,
    n_jobs: int,
    store_root: Path | None = None,
    store_dtype: str = "float32",
    coordinate_system: CoordinateSystems = CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
    lenient_mask: bool = False,
) -> None:
    """Run the joined decomposition for one music type and write every figure.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param experiment_name: Experiment being analysed.
    :param conditions: Conditions to pool, in subject-axis block and figure-row order.
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs_full: Morlet frequency grid the caches were written with.
    :param wavelet_dir: Per-condition wavelet cache directory (band subdirectory
        appended here).
    :param reuse_wavelets: Reuse the cached transforms.
    :param subset_cache_dir: Notebook-level subset cache, or ``None`` to disable.
    :param reuse_subset_cache: Read an existing subset cache entry when present.
    :param n_pca: Per-recording channel-PCA dimension (= number of components).
    :param components: 0-based component indices to draw, or ``None`` for all.
    :param n_pairs: Keep only the first N matched participants, or ``None`` for all.
    :param n_channels: Keep only the first N channels, or ``None`` for all.
    :param n_times: Keep only the first N time samples, or ``None`` for all.
    :param random_state: Seed for the PCA and ``W_init``.
    :param iva_opt_approach: ``iva_g`` optimisation method.
    :param iva_max_iter: Maximum ``iva_g`` iterations.
    :param iva_w_diff_stop: ``iva_g`` convergence threshold.
    :param band: Optional frequency band to restrict to.
    :param show_difference: Draw the between-condition difference row.
    :param onset_average: Draw the stimulus-averaged TF comparison.
    :param min_onsets: Minimum epochs required for that average.
    :param save_root: Base ``plots/`` directory.
    :param n_jobs: Parallel jobs for loading.
    :param store_root: Processed-data root to store the recovered components under,
        or ``None`` to store nothing.
    :param store_dtype: Floating precision of the stored component arrays.
    :param coordinate_system: Montage the electrode mask is read for.
    :param lenient_mask: Accept the intersection when anchor electrodes are missing.
    """
    pooled, condition_analyzers = load_joined_condition_wavelets(
        music_type,
        exclusion_categories,
        freqs_full,
        wavelet_dir=wavelet_dir / SpectrumTypeVariants.BROADBAND.value,
        experiment_name=experiment_name,
        conditions=conditions,
        n_channels=n_channels,
        n_times=n_times,
        reuse_wavelets=reuse_wavelets,
        subset_cache_dir=subset_cache_dir,
        reuse_subset_cache=reuse_subset_cache,
        n_jobs=n_jobs,
    )

    if n_pairs is not None:
        keep = sorted(set(pooled.participants))[:n_pairs]
        pooled = pooled.select_participants(keep)
        _logger.info(f"Restricted to {len(keep)} participant(s): {keep}")

    # The canonical JOINED product name, as in run_iva_condition_tracks.py.
    # Deliberately NOT pooled.data.label: that is a *data* label, and the wavelet
    # transform appends the representation to it ("Joined_ASSR (wavelet power
    # 1-50 Hz)"). Using it here would put spaces, parentheses and a frequency range
    # into the plot directory and into every figure filename, and would disagree
    # with the component store, which names the condition from the enum.
    label = f"{ConditionVariants.JOINED.value}_{music_type.value}"
    sfreq = pooled.data.sfreq
    data_4d, iva_freqs = slice_to_band(pooled.data.data, freqs_full, band)
    n_subjects, n_ch, n_freqs, n_t = data_4d.shape
    _logger.info(
        f"[{label}] pooled {data_4d.shape} (recordings × channels × freqs × times), "
        f"{pooled.n_pairs} participant(s) × {len(pooled.conditions)} condition(s), "
        f"sfreq={sfreq} Hz"
    )

    sources, patterns = decompose_channel_iva(
        data_4d,
        label=label,
        n_pca=n_pca,
        random_state=random_state,
        iva_opt_approach=iva_opt_approach,
        iva_max_iter=iva_max_iter,
        iva_w_diff_stop=iva_w_diff_stop,
    )

    # Final orientation, from PC1 of the maps the figures draw. Applied to the patterns
    # too, so topography and map agree about which way is up.
    # ── Orient every (recording, component) sign to the ASSR electrodes ──
    # The only sign alignment there is. IVA fixes a component only up to a per-dataset
    # sign, and what every downstream figure and contrast needs is not merely a
    # CONSISTENT sign but a MEANINGFUL one: positive = more power over the ASSR area.
    # The anchor is the correlation between the component's topography and the 0/1
    # electrode mask — more positive over that area than over the rest of the head — so
    # a pattern riding on a global offset cannot flip it. It reads the topography alone
    # and never the tested response, so it stays symmetric in the conditions and cannot
    # manufacture a contrast.
    #
    # Applied HERE rather than only at analysis time, so the stored sources and
    # topographies are already oriented and every figure drawn from the file agrees with
    # every test run on it. Re-applying the same anchor downstream is then a no-op.
    channel_names = list(
        topo_info_subset(condition_analyzers[pooled.conditions[0]].info, n_ch).ch_names
    )
    # A montage that carries none of the anchor electrodes has no ROI to anchor to —
    # a different situation from a recording that dropped a few of them, which
    # --lenient_mask governs. Say so and store the signs unaligned rather than either
    # crashing or pretending an anchor was applied.
    try:
        present = assr_electrode_mask(channel_names, coordinate_system, strict=False)
    except FileNotFoundError:
        present = np.zeros(len(channel_names), dtype=bool)
    if not present.any():
        _logger.warning(
            f"[{label}] no anchor electrode of {coordinate_system.value} is present in "
            "this montage, so the component signs are stored UNALIGNED — every "
            "downstream group mean will partly cancel. Check --coordinate_system."
        )
        mask_anchor = {"polarity_anchor": np.asarray("none")}
        alignment_note = _ALIGNMENT_NOTE_NONE
    else:
        electrode_mask = assr_electrode_mask(
            channel_names, coordinate_system, strict=not lenient_mask
        )
        mask_flip, mask_strength = at.polarity_flip(patterns, electrode_mask)
        sources = apply_component_signs(sources, mask_flip)
        patterns = apply_component_signs(patterns, mask_flip)
        n_weak, n_anchors = at.polarity_weak_count(mask_strength)
        alignment_note = _ALIGNMENT_NOTE
        mask_anchor = {
            "polarity_anchor": np.asarray("assr_mask_corr"),
            "polarity_anchor_flip": mask_flip,
            "polarity_anchor_strength": mask_strength,
        }
        _logger.info(
            f"[{label}] ASSR-mask sign anchor: {int((mask_flip < 0).sum())}/"
            f"{mask_flip.size} (recording, component) pair(s) flipped; {n_weak}/{n_anchors} "
            f"decided on |corr| < {at.POLARITY_CORR_FLOOR} "
            f"(median |corr| {np.median(mask_strength):.3f})"
        )
        if n_weak:
            _logger.warning(
                f"[{label}] {n_weak}/{n_anchors} sign anchor(s) rest on a weak topography "
                "correlation; a wrongly flipped recording CANCELS signal in a group mean "
                "rather than merely adding variance."
            )
    # Store the sign-aligned components before drawing anything: the arrays are the
    # expensive product of the run, the figures are lossy summaries of them, and the
    # per-recording bookkeeping this join carries — one row per (participant,
    # condition) — is exactly what makes a stored subject axis readable again.
    if store_root is not None:
        # Per-condition stimulus onsets, trimmed to the stored time axis, so an
        # onset-locked read-out is possible from the file alone. Kept per condition
        # rather than as one array because a shared time base still leaves the
        # per-condition indices free to differ by a sample or two of rounding.
        stored_onsets: dict[str, np.ndarray] = {}
        for pooled_condition in pooled.conditions:
            condition_onsets = pooled.condition_onsets(pooled_condition)
            if condition_onsets is None:
                continue
            stored_onsets[f"stimulus_onsets_{pooled_condition.value}"] = (
                condition_onsets[condition_onsets < n_t].astype(np.int64)
            )
        save_iva_components(
            experiment=experiment_name,
            condition=ConditionVariants.JOINED,
            variant=IvaVariants.CHANNEL_JOINED,
            music_type=music_type,
            band=band,
            n_pca=n_pca,
            sfreq=sfreq,
            participants=list(pooled.participants),
            subject_conditions=[c.value for c in pooled.subject_conditions],
            arrays={
                IvaComponentArrays.TF_MAP: sources,
                IvaComponentArrays.CHANNEL_PATTERN: patterns,
            },
            freqs=iva_freqs,
            times=np.arange(n_t) / sfreq,
            channel_names=list(
                topo_info_subset(
                    condition_analyzers[pooled.conditions[0]].info, n_ch
                ).ch_names
            ),
            extras={
                **mask_anchor,
                **stored_onsets,
            },
            dtype=store_dtype,
            processed_data_dir=store_root,
            logger=_logger,
        )

    comp_indices = list(range(n_pca)) if components is None else components
    out_of_range = [k + 1 for k in comp_indices if not 0 <= k < n_pca]
    if out_of_range:
        raise ValueError(
            f"--components {out_of_range} are outside 1..{n_pca} (--n_pca)."
        )

    # Canonical layout: plots/<stage>/<Condition>_<MusicType>/<spectrum>/<type>/pca_N/
    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if band is None
        else SpectrumTypeVariants.BANDS.value
    )
    out_dir = save_root / _STAGE_DIR / label / spectrum / _ANALYSIS_DIR / f"pca_{n_pca}"
    out_dir.mkdir(parents=True, exist_ok=True)
    participants_dir = out_dir / "participants"
    prefix = "" if band is None else f"{band}_"
    _logger.info(f"[{label}] writing figures to {out_dir}")

    condition_rows = [c.value for c in pooled.conditions]
    subject_participants = list(pooled.participants)
    subject_conditions = [c.value for c in pooled.subject_conditions]
    times = np.arange(n_t) / sfreq
    onsets = pooled.condition_onsets(pooled.conditions[0])
    onset_times = np.array([]) if onsets is None else onsets[onsets < n_t] / sfreq

    topo_info = topo_info_subset(condition_analyzers[pooled.conditions[0]].info, n_ch)

    # ── condition means ────────────────────────────────────────────────
    plot_condition_mean_topomaps(
        patterns,
        subject_participants,
        subject_conditions,
        condition_rows,
        topo_info,
        n_ch,
        comp_indices,
        label=label,
        show_difference=show_difference,
        alignment_note=alignment_note,
        save_path=out_dir / f"{prefix}condition_mean_topomaps.png",
    )
    plt.close("all")

    plot_condition_mean_tf_maps(
        sources,
        subject_participants,
        subject_conditions,
        condition_rows,
        iva_freqs,
        times,
        comp_indices,
        label=label,
        show_difference=show_difference,
        time_marks=onset_times,
        freq_marks=_TF_FREQ_MARKS,
        alignment_note=alignment_note,
        save_path=out_dir / f"{prefix}condition_mean_tf_maps.png",
    )
    plt.close("all")

    # ── per participant ────────────────────────────────────────────────
    written = plot_participant_condition_topomaps(
        patterns,
        subject_participants,
        subject_conditions,
        condition_rows,
        topo_info,
        n_ch,
        comp_indices,
        label=label,
        root_dir=participants_dir,
        prefix=prefix,
        alignment_note=alignment_note,
    )
    written += plot_participant_condition_tf_maps(
        sources,
        subject_participants,
        subject_conditions,
        condition_rows,
        iva_freqs,
        times,
        comp_indices,
        label=label,
        root_dir=participants_dir,
        prefix=prefix,
        time_marks=onset_times,
        freq_marks=_TF_FREQ_MARKS,
        alignment_note=alignment_note,
    )

    # ── stimulus-averaged ──────────────────────────────────────────────
    if onset_average:
        averaged = _onset_average(
            sources,
            onsets,
            label=label,
            n_times=n_t,
            sfreq=sfreq,
            min_onsets=min_onsets,
        )
        if averaged is not None:
            onset_tf, epoch_times, epoch_marks = averaged
            plot_condition_mean_tf_maps(
                onset_tf,
                subject_participants,
                subject_conditions,
                condition_rows,
                iva_freqs,
                epoch_times,
                comp_indices,
                label=label,
                show_difference=show_difference,
                freq_marks=_TF_FREQ_MARKS,
                epoch_marks=epoch_marks,
                alignment_note=alignment_note,
                save_path=out_dir / f"{prefix}condition_mean_tf_maps_onset.png",
            )
            plt.close("all")
            written += plot_participant_condition_tf_maps(
                onset_tf,
                subject_participants,
                subject_conditions,
                condition_rows,
                iva_freqs,
                epoch_times,
                comp_indices,
                label=label,
                root_dir=participants_dir,
                prefix=f"{prefix}onset_",
                freq_marks=_TF_FREQ_MARKS,
                epoch_marks=epoch_marks,
                alignment_note=alignment_note,
            )

    _logger.info(
        f"[{label}] done: {len(written)} per-participant figure(s) in "
        f"{participants_dir}, condition-mean grids in {out_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the joined comparison for every music type.

    Each music type is a separate joined run: the pooled cohort and the
    decomposition are per music type, so nothing is shared between them.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    experiment_name = ExperimentNames(args.experiment)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        # ASSR has no music dimension; uses a single placeholder "music type".
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]

    conditions = [ConditionVariants(c) for c in args.conditions]
    if len(set(conditions)) < 2:
        raise ValueError(
            "--conditions must name at least two distinct conditions; a joined run "
            f"pools them, got {[c.value for c in conditions]}."
        )

    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    wavelet_dir = resolve_wavelet_dir(args.wavelet_data_dir, experiment_name)
    # None disables storing entirely; a path (or the project default) enables it.
    store_root = (
        None
        if not args.store_components
        else (
            Path(args.store_dir)
            if args.store_dir is not None
            else ProjectPaths.PROCESSED_DATA_DIR
        )
    )
    subset_cache_dir = (
        resolve_notebook_wavelet_cache_dir(experiment_name)
        / SpectrumTypeVariants.BROADBAND.value
        if args.subset_cache
        else None
    )
    freqs_full = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )
    # The CLI takes 1-based component numbers, matching the "IC <k>" figure labels.
    components = None if args.components is None else [k - 1 for k in args.components]

    _logger.info(
        f"IVA condition comparison: experiment={experiment_name.value}, "
        f"conditions={[c.value for c in conditions]}, "
        f"music_types={[mt.value for mt in music_types]}, n_pca={args.n_pca}, "
        f"band={args.band or SpectrumTypeVariants.BROADBAND.value}, "
        f"iva_opt={args.iva_opt_approach}"
    )
    _logger.info(f"Wavelet source cache : {wavelet_dir}")
    _logger.info(f"Wavelet subset cache : {subset_cache_dir or 'disabled'}")

    for music_type in music_types:
        _logger.info(f"=== {music_type.value} ===")
        run_condition_comparison(
            music_type,
            experiment_name=experiment_name,
            conditions=conditions,
            exclusion_categories=exclusion_categories,
            freqs_full=freqs_full,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=args.reuse_wavelets,
            subset_cache_dir=subset_cache_dir,
            reuse_subset_cache=not args.no_reuse_subset_cache,
            n_pca=args.n_pca,
            components=components,
            n_pairs=args.n_pairs,
            n_channels=args.n_channels,
            n_times=args.n_times,
            random_state=args.random_state,
            iva_opt_approach=args.iva_opt_approach,
            iva_max_iter=args.iva_max_iter,
            iva_w_diff_stop=args.iva_w_diff_stop,
            band=args.band,
            show_difference=not args.no_difference_row,
            onset_average=not args.skip_onset_average,
            min_onsets=args.min_onsets,
            save_root=save_root,
            n_jobs=args.n_jobs,
            store_root=store_root,
            store_dtype=args.store_dtype,
            coordinate_system=CoordinateSystems(args.coordinate_system),
            lenient_mask=args.lenient_mask,
        )


if __name__ == "__main__":
    main()
