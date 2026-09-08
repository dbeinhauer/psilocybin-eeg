"""
CLI script reproducing
``notebooks/06-iva-condition-comparison/wavelet_iva_channel_joined_tracks.ipynb``:
the channel-as-independent IVA-G decomposition of wavelet power run on the
**time-concatenated** join (:attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS`),
followed by the Placebo/Psilocybin comparison of the recovered components.

The companion of ``run_iva_condition_comparison.py``. Both run the identical
pipeline on the identical participants and share every numerical step
(:mod:`src.analysis.iva_condition_comparison`); they differ only in **which axis
the conditions are pooled along**, and that one choice changes what the
comparison can say::

    subject k  ->  [ ---- Placebo track ---- | ---- Psilocybin track ---- ]
                   0                     T_pl                  T_pl + T_ps

    datasets = participants        # P entries, not 2P
    samples  = time × frequency    # time spans BOTH conditions
    mixing   = channel             # one matrix per participant, both conditions

**What this buys.** The subject axis holds one entry per participant and the
mixing is estimated over the whole concatenated recording, so IVA returns a
*single* set of components covering both conditions. The contrast is read off by
slicing the sources back into the two time segments, so the components are the
same in both conditions **by construction** — no component-matching step, and no
risk of comparing a Placebo component against a different Psilocybin one.

**What it costs.** One mixing matrix per participant means **one channel
topography per (participant, component), shared by both conditions**. There is
no per-condition topography to compare, so the topography figures are emitted
with a **single row**, labelled as shared, rather than as two identical rows and
an all-zero difference — a figure that would read as a null result while really
being a statement about the model. For a genuine topography contrast use
``run_iva_condition_comparison.py``, where each recording has its own mixing
matrix.

**Z-scoring happens before the concatenation** (``--zscore_mode``, default
``per_condition``): each condition is standardised on its own and the
standardised tracks are then laid end to end, so each enters the decomposition
exactly as in a single-condition run. The consequence is the same trade-off the
subject-axis variant makes — an overall power difference between the conditions
is normalised away, so what is compared is temporal and spectral *structure*, not
amplitude. Pass ``--zscore_mode joint`` to keep the amplitude contrast.

**Nothing upstream changes and no new cache is written.** Each condition keeps
its own alignment, used only within that condition, so the two tracks need not
share a time base and need not even have the same length. Only participants
recorded under **both** conditions are kept, so the design is balanced. Pass
``--subset_cache`` to additionally keep a per-extent copy of each condition's
trimmed tensor under
``notebooks/03-wavelet-analysis/wavelet_cache/<experiment>/``, shared with every
other wavelet workflow.

Sign handling, in order:

1. ``iva_g`` fixes each component's sign only per participant.
2. :func:`~src.analysis.wavelet_ica.align_iva_component_signs` resolves it from
   ``Sigma_N``.
3. :func:`~src.analysis.assr_trials.polarity_flip` re-decides it from the
   correlation between each component's topography and the binary ASSR
   electrode mask, and has the final say — so a positive value means more power
   over that area for everyone, which is what the contrasts actually need.

All three run **before** the split, deliberately: a sign belongs to a
``(participant, component)`` pair, not to a condition. Flipping a participant's
two segments differently would destroy the one property this variant is built
on, so aligning first guarantees both halves inherit the same sign. Note what a
"dataset" is here — one participant, both tracks — so a flip can never invert a
participant's Placebo segment relative to their Psilocybin one.

Figures:

  - ``shared_mean_topomaps`` — the shared channel topographies, **one row**,
    columns = components
  - ``condition_mean_tf_maps`` — rows = conditions plus their difference,
    columns = components, on the split TF maps
  - ``condition_mean_tf_maps_onset`` — the same, **averaged over stimuli**
    (ASSR only), each condition epoched with its own onsets on its own segment
  - ``participants/shared_condition_topomap_ic<k>_participants_*`` — one figure
    per component, the shared topography per participant (one row)
  - ``participants/condition_tf_ic<k>_participants_*`` and its ``onset_``
    counterpart — Placebo on the first row, Psilocybin on the second, one column
    per participant ordered by ID, so a participant's two segments are
    vertically paired

Output (canonical per-condition layout)::

    plots/06-iva-condition-comparison/JoinedTracks_<MusicType>/
        broadband/iva_channel_joined_tracks/pca_<n_pca>/    # default (no --band)
            shared_mean_topomaps.png
            condition_mean_tf_maps.png
            condition_mean_tf_maps_onset.png
            participants/*.png
        bands/iva_channel_joined_tracks/pca_<n_pca>/         # when --band <name>
            <band>_shared_mean_topomaps.png
            ...

``--n_pca`` reduces the **channel** axis, so it must be ≤ the number of channels
(≈195 after preprocessing). ``--n_times`` is applied **per condition**, i.e. per
segment, so the concatenated sample axis is ``n_freqs × 2 × n_times``.

Examples::

    # ASSR, all matched participants, ten components
    python scripts/run_iva_condition_tracks.py \\
        --experiment assr --n_pca 10 --reuse_wavelets --subset_cache

    # Fast exploration on a channel/time subset, cached for the next run
    python scripts/run_iva_condition_tracks.py \\
        --experiment assr --n_pca 10 --n_channels 32 --n_times 3000 \\
        --n_pairs 5 --reuse_wavelets --subset_cache

Pass ``--store_components`` to additionally keep the recovered components under
``data/processed/<experiment>/iva_results/JoinedTracks/``. They go in **unsplit**, on
the concatenated time axis, with the segment order and lengths, so the stored file
answers both questions — the whole recording, and either condition via
:meth:`~src.io.iva_store.IvaComponentResults.condition_track`. The shared topography
is stored once per participant, not duplicated per condition. Read them back with
:func:`src.io.iva_store.load_iva_components`.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    load_paired_condition_wavelets,
    resolve_notebook_wavelet_cache_dir,
)
from src.analysis import iva_quality  # noqa: E402
from src.analysis.condition_tracks import ZSCORE_MODES  # noqa: E402
from src.analysis.iva_condition_comparison import (  # noqa: E402
    apply_component_signs,
    decompose_channel_iva,
    slice_to_band,
    stack_conditions_on_subject_axis,
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
from src.analysis import assr_trials as at  # noqa: E402
from src.io.iva_store import save_iva_components  # noqa: E402
from src.io.loading import assr_electrode_mask  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_tf_maps,
    plot_condition_mean_topomaps,
    plot_participant_condition_tf_maps,
    plot_participant_condition_topomaps,
)
from src.visualization.iva_quality_plots import topo_info_subset  # noqa: E402

_logger = logging.getLogger(__name__)

#: Stage directory under ``plots/`` — shared with the subject-axis variant.
_STAGE_DIR = "06-iva-condition-comparison"

#: Analysis-type subdirectory, naming the independent axis and the join, so the two
#: variants' figures never collide.
_ANALYSIS_DIR = IvaVariants.CHANNEL_JOINED_TRACKS.value

#: What the final sign alignment was, printed on every figure.
_ALIGNMENT_NOTE = "corr(topography, ASSR electrode mask), per (participant, component)"
#: What the figures say when the montage carries no anchor electrode and the
#: signs were therefore stored as iva_g returned them. Naming it matters: an
#: unaligned group mean partly cancels, and a reader must not take a flat map
#: for an absent effect.
_ALIGNMENT_NOTE_NONE = "NOT sign-aligned (no ASSR anchor electrode present)"

#: Frequency (Hz) marked on every TF panel — the ASSR stimulation frequency.
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]

#: Minimum onsets that must fit for the stimulus average to be drawn.
_MIN_ONSETS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the channel-as-independent IVA decomposition on the time-concatenated "
            "join (each participant's Placebo track followed by their Psilocybin one) "
            "and draw the per-component condition comparison. Reproduces "
            "wavelet_iva_channel_joined_tracks.ipynb."
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
            "One or more music types, each analysed in its own run. When omitted, "
            "defaults to CLASSIC + PSYTRANCE for the psilo_music experiment and ASSR "
            "for the assr experiment."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[c.value for c in REAL_CONDITIONS],
        default=[c.value for c in REAL_CONDITIONS],
        help=(
            "Conditions to concatenate, in the order their segments appear along the "
            "time axis and their rows appear in every figure."
        ),
    )
    parser.add_argument(
        "--zscore_mode",
        choices=list(ZSCORE_MODES),
        default="per_condition",
        help=(
            "How the tracks are standardised before being concatenated. "
            "'per_condition' standardises each on its own (so each enters the "
            "decomposition as in a single-condition run, and an overall power "
            "difference between conditions is normalised away); 'joint' z-scores over "
            "the whole concatenated recording, keeping that difference; 'none' assumes "
            "the caller already standardised."
        ),
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=10,
        help=(
            "Per-participant PCA dim over CHANNELS before IVA-G (square mixing "
            "requirement), i.e. the number of IVA components. Must be ≤ the number of "
            "channels."
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
            "Keep only the first N matched participants, by ID. Here that is also the "
            "subject count: each participant is ONE subject carrying both tracks. "
            "Defaults to every matched participant."
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
            "Keep only the first N time samples OF EACH CONDITION. The IVA sample axis "
            "is n_freqs × the concatenated length, i.e. n_freqs × 2 × this, so it "
            "drives peak memory. Defaults to each condition's whole time axis."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for per-participant PCA and IVA W_init.",
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
            "Keep a per-extent copy of each condition's trimmed wavelet tensor under "
            "notebooks/03-wavelet-analysis/wavelet_cache/<experiment>/, so later runs "
            "at the same extent skip decompressing the source cache. Shared with every "
            "other wavelet workflow."
        ),
    )
    parser.add_argument(
        "--no_reuse_subset_cache",
        action="store_true",
        help=(
            "With --subset_cache, recompute from the source cache and overwrite the "
            "subset entry instead of reading it."
        ),
    )
    parser.add_argument(
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the concatenated tensor is sliced to "
            "the band's frequency range before IVA, and plots are written to 'bands/' "
            "with a '<band>_' prefix."
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
            "Minimum stimulus epochs that must fit EACH condition's segment before the "
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


def _onset_average_per_condition(
    sources_by_condition: dict[str, np.ndarray],
    onsets_by_condition: dict[str, np.ndarray | None],
    participants: list[str],
    conditions: list[str],
    *,
    label: str,
    sfreq: float,
    min_onsets: int,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, list[float]] | None:
    """Epoch each condition on **its own** onsets and stack the averages.

    The one place the stimulus average genuinely differs from the subject-axis variant.
    The tracks keep their own time bases here, so there is no single onset list covering
    both conditions: each split track is averaged against the onsets of its own segment,
    which is what ``condition_onsets`` returns (local, unshifted).

    Both are cut on the same paradigm window from
    :class:`~src.definitions.constants.AssrEpoch` via
    :func:`~src.analysis.iva_quality.onset_window`, taking the **shorter** post-onset
    span of the two so the two averages share one epoch axis and can be stacked.

    Each frequency is then referenced to its own pre-onset mean
    (:func:`~src.analysis.iva_quality.subtract_epoch_baseline`). The z-scoring puts the
    pre-onset interval near 0 over the WHOLE recording, but not within any one epoch —
    the local level still drifts — so this is what makes the map a change rather than a
    level. Only the mean is removed; see that function for why no divisor is applied.

    :param sources_by_condition: Condition name → ``(P, K, F, T_c)`` split sources.
    :param onsets_by_condition: Condition name → onset samples local to that segment,
        or ``None``.
    :param participants: Participant label per row, in row order.
    :param conditions: Conditions in the block order wanted.
    :param label: Dataset label, for logging.
    :param sfreq: Sampling frequency in Hz.
    :param min_onsets: Minimum epochs that must fit *each* condition.
    :return: ``(stacked, subject_participants, subject_conditions, epoch_times,
        epoch_marks)``, or ``None`` when any condition cannot be averaged.
    """
    geometry: dict[str, tuple[np.ndarray, int, int]] = {}
    for condition in conditions:
        onsets = onsets_by_condition.get(condition)
        n_times = sources_by_condition[condition].shape[-1]
        if onsets is None or len(onsets) == 0:
            _logger.info(f"[{label}] {condition}: no stimulus onsets.")
            return None
        onsets_in = np.asarray(onsets)[np.asarray(onsets) < n_times].astype(int)
        if onsets_in.size == 0:
            _logger.info(f"[{label}] {condition}: no onset inside the segment.")
            return None
        pre, post = iva_quality.onset_window(onsets_in, n_times, sfreq)
        n_fitting = int(((onsets_in - pre >= 0) & (onsets_in + post <= n_times)).sum())
        if n_fitting < min_onsets:
            _logger.warning(
                f"[{label}] {condition}: only {n_fitting} epoch(s) fit its "
                f"{n_times}-sample segment, need {min_onsets}; skipping the stimulus "
                "average. Raise --n_times or lower --min_onsets."
            )
            return None
        geometry[condition] = (onsets_in, pre, post)

    # One epoch geometry for both, so the averages can be stacked.
    pre = min(g[1] for g in geometry.values())
    post = min(g[2] for g in geometry.values())
    epoch_times = np.arange(-pre, post) / sfreq
    marks = [0.0, min(AssrEpoch.STIMULUS_DURATION_S, float(epoch_times[-1]))]
    baseline_mask = epoch_times < 0.0

    averaged: dict[str, np.ndarray] = {}
    for condition in conditions:
        onsets_in, _pre, _post = geometry[condition]
        averaged[condition], n_used = iva_quality.epoch_average(
            sources_by_condition[condition], onsets_in, pre, post
        )
        averaged[condition] = iva_quality.subtract_epoch_baseline(
            averaged[condition], baseline_mask
        )
        _logger.info(
            f"[{label}] {condition}: averaged {n_used} epoch(s) -> "
            f"{averaged[condition].shape}, each frequency referenced to its own "
            "pre-onset mean"
        )

    stacked, subject_participants, subject_conditions = (
        stack_conditions_on_subject_axis(averaged, conditions, participants)
    )
    _logger.info(
        f"[{label}] onset averages stacked {stacked.shape}; window "
        f"[{epoch_times[0]:.3f}, {epoch_times[-1]:.3f}] s"
    )
    if post < int(round(AssrEpoch.POST_ONSET_S * sfreq)):
        _logger.info(
            f"[{label}] post-onset span trimmed from {AssrEpoch.POST_ONSET_S} s to "
            f"{post / sfreq:.3f} s by the shortest inter-onset gap."
        )
    return stacked, subject_participants, subject_conditions, epoch_times, marks


def run_condition_tracks(
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
    zscore_mode: str,
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
    """Run the time-concatenated decomposition for one music type and write every figure.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param experiment_name: Experiment being analysed.
    :param conditions: Conditions to concatenate, in time-axis segment order.
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs_full: Morlet frequency grid the caches were written with.
    :param wavelet_dir: Per-condition wavelet cache directory.
    :param reuse_wavelets: Reuse the cached transforms.
    :param subset_cache_dir: Notebook-level subset cache, or ``None`` to disable.
    :param reuse_subset_cache: Read an existing subset cache entry when present.
    :param zscore_mode: One of :data:`~src.analysis.condition_tracks.ZSCORE_MODES`.
    :param n_pca: Per-participant channel-PCA dimension (= number of components).
    :param components: 0-based component indices to draw, or ``None`` for all.
    :param n_pairs: Keep only the first N matched participants, or ``None`` for all.
    :param n_channels: Keep only the first N channels, or ``None`` for all.
    :param n_times: Keep only the first N samples of each condition, or ``None``.
    :param random_state: Seed for the PCA and ``W_init``.
    :param iva_opt_approach: ``iva_g`` optimisation method.
    :param iva_max_iter: Maximum ``iva_g`` iterations.
    :param iva_w_diff_stop: ``iva_g`` convergence threshold.
    :param band: Optional frequency band to restrict to.
    :param show_difference: Draw the between-condition difference row.
    :param onset_average: Draw the stimulus-averaged TF comparison.
    :param min_onsets: Minimum epochs required per condition for that average.
    :param save_root: Base ``plots/`` directory.
    :param n_jobs: Parallel jobs for loading.
    :param store_root: Processed-data root to store the recovered components under,
        or ``None`` to store nothing.
    :param store_dtype: Floating precision of the stored component arrays.
    :param coordinate_system: Montage the electrode mask is read for.
    :param lenient_mask: Accept the intersection when the recording is missing some of
        the listed anchor electrodes, instead of refusing to under-select.
    """
    paired, condition_analyzers = load_paired_condition_wavelets(
        music_type,
        exclusion_categories,
        freqs_full,
        wavelet_dir=wavelet_dir / SpectrumTypeVariants.BROADBAND.value,
        experiment_name=experiment_name,
        conditions=conditions,
        zscore_mode=zscore_mode,
        n_channels=n_channels,
        n_times=n_times,
        reuse_wavelets=reuse_wavelets,
        subset_cache_dir=subset_cache_dir,
        reuse_subset_cache=reuse_subset_cache,
        n_jobs=n_jobs,
    )

    if n_pairs is not None and n_pairs < paired.n_pairs:
        keep = sorted(paired.participants)[:n_pairs]
        rows = [paired.participants.index(p) for p in keep]
        paired = dataclasses.replace(
            paired,
            data=dataclasses.replace(paired.data, data=paired.data.data[rows]),
            participants=tuple(keep),
        )
        _logger.info(f"Restricted to {len(keep)} participant(s): {keep}")

    # The canonical JOINED_TRACKS product name. Deliberately not paired.data.label,
    # which is "Placebo_X+Psilocybin_X" — a '+' in a directory and in every figure
    # filename is awkward to glob and to quote.
    label = f"{ConditionVariants.JOINED_TRACKS.value}_{music_type.value}"

    sfreq = paired.data.sfreq
    data_4d, iva_freqs = slice_to_band(paired.data.data, freqs_full, band)
    _n_subjects, n_ch, n_freqs, n_total = data_4d.shape
    _logger.info(
        f"[{label}] concatenated {data_4d.shape} (participants × channels × freqs × "
        f"times), {paired.n_pairs} participant(s), segments {paired.segment_lengths}, "
        f"sfreq={sfreq} Hz, zscore_mode={zscore_mode}"
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

    # Orient BEFORE the split: a sign belongs to a (participant, component) pair, so
    # flipping a participant's two segments differently would break the one property
    # this variant is built on — that the component is the same in both conditions.
    # ── Orient every (participant, component) sign to the ASSR electrodes ──
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
        topo_info_subset(condition_analyzers[paired.conditions[0]].info, n_ch).ch_names
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
            f"{mask_flip.size} (participant, component) pair(s) flipped; {n_weak}/{n_anchors} "
            f"decided on |corr| < {at.POLARITY_CORR_FLOOR} "
            f"(median |corr| {np.median(mask_strength):.3f})"
        )
        if n_weak:
            _logger.warning(
                f"[{label}] {n_weak}/{n_anchors} sign anchor(s) rest on a weak topography "
                "correlation; a wrongly flipped participant CANCELS signal in a group mean "
                "rather than merely adding variance."
            )
    # Store the sign-aligned components before splitting or drawing anything. The
    # arrays go in **unsplit**, on the concatenated time axis, together with the
    # segment order and lengths: that is the model's own layout, and it keeps the
    # stored file able to answer both questions — the whole recording, and either
    # condition on its own (:meth:`~src.io.iva_store.IvaComponentResults.condition_track`).
    # The channel patterns have no condition axis at all in this variant, by
    # construction: one mixing matrix per participant covers both tracks.
    if store_root is not None:
        # Per-condition stimulus onsets, LOCAL to each condition's segment (the same
        # frame condition_track returns), trimmed to that segment, so an
        # onset-locked read-out is possible from the file alone. The two conditions
        # keep their own alignments here, so the onset counts need not match.
        stored_onsets: dict[str, np.ndarray] = {}
        for segment_condition, segment_length in zip(
            paired.conditions, paired.segment_lengths
        ):
            segment_onsets = paired.condition_onsets(segment_condition, absolute=False)
            if segment_onsets is None:
                continue
            stored_onsets[f"stimulus_onsets_{segment_condition.value}"] = (
                segment_onsets[segment_onsets < segment_length].astype(np.int64)
            )
        save_iva_components(
            experiment=experiment_name,
            condition=ConditionVariants.JOINED_TRACKS,
            variant=IvaVariants.CHANNEL_JOINED_TRACKS,
            music_type=music_type,
            band=band,
            n_pca=n_pca,
            sfreq=sfreq,
            participants=list(paired.participants),
            arrays={
                IvaComponentArrays.TF_MAP: sources,
                IvaComponentArrays.CHANNEL_PATTERN: patterns,
            },
            freqs=iva_freqs,
            times=np.arange(n_total) / sfreq,
            channel_names=list(
                topo_info_subset(
                    condition_analyzers[paired.conditions[0]].info, n_ch
                ).ch_names
            ),
            segment_conditions=[c.value for c in paired.conditions],
            segment_lengths=list(paired.segment_lengths),
            extras={
                "zscore_mode": np.asarray(zscore_mode),
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

    # Split the time axis back into the two conditions, then restack onto the subject
    # axis the comparison figures index by.
    condition_rows = [c.value for c in paired.conditions]
    participants = list(paired.participants)
    sources_by_condition = {
        c.value: paired.condition_track(sources, c) for c in paired.conditions
    }
    onsets_by_condition = {
        c.value: paired.condition_onsets(c, absolute=False) for c in paired.conditions
    }
    sources_stacked, subject_participants, subject_conditions = (
        stack_conditions_on_subject_axis(
            sources_by_condition, condition_rows, participants
        )
    )
    n_split = sources_stacked.shape[-1]
    times = np.arange(n_split) / sfreq
    lengths = {name: a.shape[-1] for name, a in sources_by_condition.items()}
    if len(set(lengths.values())) > 1:
        _logger.info(
            f"[{label}] segments differ in length {lengths}; the stack was trimmed to "
            f"{n_split} samples."
        )
    ref_onsets = onsets_by_condition[condition_rows[0]]
    onset_times = (
        np.array([]) if ref_onsets is None else ref_onsets[ref_onsets < n_split] / sfreq
    )

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

    topo_info = topo_info_subset(condition_analyzers[paired.conditions[0]].info, n_ch)

    # ── the shared topographies: ONE row, not a fabricated comparison ──
    shared_row = " + ".join(condition_rows) + " (shared)"
    plot_condition_mean_topomaps(
        patterns,
        participants,
        [shared_row] * len(participants),
        [shared_row],
        topo_info,
        n_ch,
        comp_indices,
        label=label,
        alignment_note=alignment_note,
        save_path=out_dir / f"{prefix}shared_mean_topomaps.png",
    )
    plt.close("all")

    # ── condition means, on the split TF maps ──────────────────────────
    plot_condition_mean_tf_maps(
        sources_stacked,
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
        participants,
        [shared_row] * len(participants),
        [shared_row],
        topo_info,
        n_ch,
        comp_indices,
        label=label,
        root_dir=participants_dir,
        prefix=f"{prefix}shared_",
        alignment_note=alignment_note,
    )
    written += plot_participant_condition_tf_maps(
        sources_stacked,
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
        averaged = _onset_average_per_condition(
            sources_by_condition,
            onsets_by_condition,
            participants,
            condition_rows,
            label=label,
            sfreq=sfreq,
            min_onsets=min_onsets,
        )
        if averaged is not None:
            onset_stacked, onset_parts, onset_conds, epoch_times, marks = averaged
            plot_condition_mean_tf_maps(
                onset_stacked,
                onset_parts,
                onset_conds,
                condition_rows,
                iva_freqs,
                epoch_times,
                comp_indices,
                label=label,
                show_difference=show_difference,
                freq_marks=_TF_FREQ_MARKS,
                epoch_marks=marks,
                alignment_note=alignment_note,
                save_path=out_dir / f"{prefix}condition_mean_tf_maps_onset.png",
            )
            plt.close("all")
            written += plot_participant_condition_tf_maps(
                onset_stacked,
                onset_parts,
                onset_conds,
                condition_rows,
                iva_freqs,
                epoch_times,
                comp_indices,
                label=label,
                root_dir=participants_dir,
                prefix=f"{prefix}onset_",
                freq_marks=_TF_FREQ_MARKS,
                epoch_marks=marks,
                alignment_note=alignment_note,
            )

    _logger.info(
        f"[{label}] done: {len(written)} per-participant figure(s) in "
        f"{participants_dir}, condition-level grids in {out_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the time-concatenated join for every music type.

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
            "--conditions must name at least two distinct conditions; a concatenated "
            f"run lays them end to end, got {[c.value for c in conditions]}."
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
        f"IVA condition tracks: experiment={experiment_name.value}, "
        f"conditions={[c.value for c in conditions]}, "
        f"music_types={[mt.value for mt in music_types]}, n_pca={args.n_pca}, "
        f"zscore_mode={args.zscore_mode}, "
        f"band={args.band or SpectrumTypeVariants.BROADBAND.value}, "
        f"iva_opt={args.iva_opt_approach}"
    )
    _logger.info(f"Wavelet source cache : {wavelet_dir}")
    _logger.info(f"Wavelet subset cache : {subset_cache_dir or 'disabled'}")

    for music_type in music_types:
        _logger.info(f"=== {music_type.value} ===")
        run_condition_tracks(
            music_type,
            experiment_name=experiment_name,
            conditions=conditions,
            exclusion_categories=exclusion_categories,
            freqs_full=freqs_full,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=args.reuse_wavelets,
            subset_cache_dir=subset_cache_dir,
            reuse_subset_cache=not args.no_reuse_subset_cache,
            zscore_mode=args.zscore_mode,
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
