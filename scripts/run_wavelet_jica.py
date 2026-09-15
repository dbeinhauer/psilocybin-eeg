"""
CLI script reproducing the stage-07 joint-ICA (jICA) notebooks headlessly.

Runs both halves of ``notebooks/07-ica-condition-comparison/`` in one pass:

1. **The decomposition** (``wavelet_ica_channel_joined.ipynb`` /
   ``wavelet_ica_channel_joined_tracks.ipynb``) — the component time-frequency maps,
   the per-recording channel topographies and the per-recording loading.
2. **The 40 Hz read-out** (``ica_component_analysis.ipynb``) — the recovered filters
   projected onto the raw wavelet power, cut into stimulus-locked trials, drawn as
   the response over time per condition, and tested against the fixed
   ASSR-electrode reference.

The ICA counterpart of ``run_iva_condition_tracks.py`` +
``run_assr_snr_grid.py``, and it shares their machinery rather than repeating it:
the join assembly, the FastICA fit and the per-block filter recovery come from
:mod:`src.analysis.wavelet_jica`, and **everything downstream is
:mod:`src.analysis.assr_trials`** — the frequency selection, the projection, the
epoch geometry, the trial cutting, the per-trial normalisation, the polarity
anchoring and the paired tests are decomposition-agnostic and are used unchanged.

**Same independence geometry as the channel-IVA variants**, different way of tying
the recordings together: mixing = channels, samples = the joint (frequency x time)
plane, but instead of one dataset per recording coupled through IVA's
source-component vector, every recording's channels are laid side by side into one
feature axis and a single FastICA is fitted to the lot. So a component is one
**shared** spectro-temporal source plus a mixing column that splits into
per-recording blocks::

    --join joined_tracks   one block per participant, sample axis spans both tracks
                           -> topography and loading SHARED, contrast in the TF maps
    --join joined          one block per recording
                           -> TF map shared, contrast in the topographies/loadings

The two are complements: between them they cover both sides of the decomposition and
neither covers both alone. ``joined_tracks`` is the default for the read-out because
the same operator reads both conditions there, so a condition difference cannot be an
artefact of the filter having changed; run ``joined`` as the check that the conclusion
survives when each recording gets its own filter.

**What jICA buys over IVA**, and it is the reason this stage exists: FastICA's
objective is invariant under flipping an *entire* unmixing row and nothing else, so a
component's sign flips its map and its whole mixing column together and there is **no
per-recording sign ambiguity** to resolve. IVA-G's ``sum_k log det Sigma_k`` *is*
invariant under per-dataset flips, so the IVA path must spend alignment passes on
signs that are genuinely unidentifiable, carrying the risk that arbitrary flips
manufacture a condition difference. Here the only free sign is one per component, and
this script pins it once — anchored to the cohort topography, identical for every
recording and both conditions. **A per-recording flip is therefore NOT applied**: it
would overwrite a determined quantity, and under ``--join joined`` could flip one
condition of a pair and not the other, turning the paired difference into a sum.

**What it costs**: a per-block filter is that recording's *contribution* to a shared
source rather than its own copy of the component, so a participant with a near-zero
loading has a near-noise time course and still carries full weight in every group
statistic. The loading figures are what say which those are.

The read-out runs a **(spatial filter x signal) grid** (``--signal_variants``), six
cells by default, every one of them per-trial baselined so all six are in the same
units and directly comparable — the IVA stage's grid cell for cell, so a row of either
stage's CSV can be read against the other::

                          | raw wavelet power | z-scored (what the fit saw)
    ----------------------+-------------------+-----------------------------
    that block's filter   | prestim           | zscored_prestim
    cohort-mean filter    | mean_prestim      | mean_zscored_prestim
    ... ASSR electrodes   | -                 | masked_prestim
    cohort mean, ASSR     | -                 | mean_masked_prestim

Down one axis: applying the filter to raw power drops the per-channel ``1/sd``
weighting the fit folded in, so it *approximates* the component, while the z-scored
half reads exactly the signal the fit saw and so IS the component. Down the other: the
cohort-mean filter is one operator shared by everybody, built by averaging the
unit-normalised per-block topographies over both conditions and inverting
(:func:`~src.analysis.wavelet_jica.cohort_mean_pattern`,
:func:`~src.analysis.wavelet_jica.cohort_mean_filter`). It exists because a
per-recording filter spends ``C`` parameters rediscovering a stereotyped topography and
the variance of that estimate can exceed the bias it removes — and because it is the
like-for-like learned comparison to the fixed electrode mask, which is itself a single
group-level spatial weighting. The **masked** pair is either of those with every weight
outside the ASSR electrodes zeroed, which separates a better weighting *inside* the
anchor area from access to signal *outside* it; both read the z-scored signal only,
because a restricted filter's weights presume the scaling they were fitted on. Every
baselined variant borrows its reference rows from ``prestim``, so the whole grid is
judged against one set of numbers. See
:data:`~src.analysis.wavelet_jica.SIGNAL_VARIANTS`.

**Three fixed reference rows**, the same three ``run_assr_snr_grid.py`` emits for the
IVA stage and named identically (:data:`~src.analysis.assr_trials.MASK_LABELS`), in
increasing proximity to what the decomposition actually saw:

* ``ASSR-mask (full)`` — the fronto-central electrode average on the whole channel
  space, built with **equal weight per electrode**: each ROI electrode is referenced to
  its own pre-stimulus window and divided by its baseline SD *pooled over trials*
  **before** the ROI is averaged, and the average is referenced to its own pre-stimulus
  again — against ONE constant shared by both conditions, so the paired contrast is
  never rescaled per participant
  (:func:`~src.analysis.assr_trials.roi_channelwise_snr`). Averaging raw power first
  would weight each electrode by its own power level, which is uncorrelated with whether
  that electrode carries any 40 Hz response, and would leave the reference off the units
  every component row is in — which
  :func:`~src.analysis.assr_trials.reference_snr_tests` subtracts from directly. This is
  the row the two stages compute identically, so it is where they can be checked against
  each other.
* ``ASSR-mask (PCA)`` — the same average read only through the retained subspace, on RAW
  power (:func:`~src.analysis.wavelet_jica.subspace_mask_rows`), so a component is judged
  against a reference that lost the same directions it did.
* ``ASSR-mask (PCA, z)`` — that operator on the Z-SCORED signal, i.e. exactly what the
  fit was handed. Read beside the row above it separates the channel reduction from the
  per-(channel, frequency) rescaling z-scoring adds; in ``zscored`` the two coincide by
  construction, which is a free check that both paths agree.

**The subspace rows are joint**, and that is the one place this stage's reference
differs in kind from the IVA stage's: jICA reduces the *stacked* channel axis, so what
it kept for one recording genuinely depends on the others at the same sample, whereas
the IVA store's PCA is fitted per recording. Both families of tests run against all
three references, and every test row names the one it was judged against.

**There is no per-participant test on the global TF maps**, and that is structural
rather than an omission: jICA's ``tf_maps`` is ``(K, F, T)`` — one map per component,
shared by every recording — so cutting trials from it yields one number per (component,
condition) with no participant axis. A paired test over participants on that would
report ``n = P`` for what is a single measurement. The per-participant quantity jICA
does offer is the per-block term ``u_{b,k} . x_b^z`` of the sum that *forms* those maps,
which is what ``zscored`` and ``zscored_prestim`` test.

Two extents are **not** free knobs, and the defaults are the working ones:

* ``--n_channels`` defaults to every channel. The ASSR-electrode reference is a
  fronto-central selection spread over the montage, so a leading channel slice does
  not contain it and the comparison the read-out is built on becomes meaningless.
  The run refuses rather than quietly averaging over two electrodes.
* ``--n_pairs`` defaults to every matched participant. The participant is the unit of
  every test and an exact Wilcoxon over ``P`` participants cannot return a *p* below
  ``2/2**P``: 0.0005 at 12, but 0.06 at 5. Subsetting the cohort does not just weaken
  the test, it makes it unable to reject at all.

* ``--n_times`` defaults to each condition's whole time axis. It is a **leading
  crop**, not a subsample, so any value keeps only the first ``N/sfreq`` seconds of
  every recording and drops the later trials wholesale — which excludes habituation
  and time-on-task from the components *and* from the tests, on top of costing
  trials. Treat it as a debugging flag, not a setting.

Peak memory is roughly ``2-3 x (blocks x channels x freqs x times) x itemsize`` — the
raw tensor, the standardised copy and the stacked matrix. At the full ASSR extent (12
participants, 195 channels, 50 frequencies, ~47000 samples per condition, float32)
that is ~44 GB per array and a ~90-130 GB peak. Nothing proportional to the data is
allocated on top of that: the channel reduction goes through the
``(features, features)`` covariance — 2340 x 2340 here — rather than an SVD of the
data matrix, which is both far cheaper and the only route that *works* at this size.
See :func:`~src.analysis.wavelet_jica.fit_joint_ica` for the 32-bit LAPACK ceiling
that rules the direct SVD out.

Figures (canonical per-product layout)::

    plots/07-ica-condition-comparison/<Product>_<MusicType>/broadband/
        ica_channel_joined_tracks/ica_<n>/       # the decomposition
            condition_tf_maps.png                # or shared_tf_maps.png for --join joined
            condition_tf_maps_onset.png
            shared_mean_topomaps.png             # or condition_mean_topomaps.png
            loading_bars_absolute.png
            loading_bars_within_component.png
            participants/*.png                   # with --participant_grids
        ica_component_analysis/ica_<n>/          # the 40 Hz read-out
            trial_course_by_source_<selection>_<variant>.png
            pvalue_summary_<selection>_<variant>_<reference>.png
            snr_vs_reference_<selection>_<reference>.png

``<reference>`` is ``full``, ``pca`` or ``pca_z``
(:data:`~src.analysis.assr_trials.MASK_SLUGS`): "better than the reference?" is a
different question for each of the three, so each gets its own figure. The course
figure is drawn once per cell and carries all three reference rows as panels.

Tests are written to one CSV per run, with a ``reference`` column naming which row each
``vs_reference`` / ``snr`` test was judged against (empty for ``contrast``)::

    results/<experiment>/jica_tests__<join>__<MusicType>__ica<n>.csv

Examples::

    # The standard pass: the decomposition figures and the read-out, full extent
    python scripts/run_wavelet_jica.py --experiment assr --join joined_tracks \\
        --n_ica 5 --reuse_wavelets

    # The companion join, as the check that the conclusion survives when each
    # recording gets its own filter
    python scripts/run_wavelet_jica.py --experiment assr --join joined \\
        --n_ica 5 --reuse_wavelets
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    add_wavelet_grid_args,
    resolve_wavelet_dir,
)
from scripts.notebook_helpers import (  # noqa: E402
    load_joined_condition_wavelets,
    load_paired_condition_wavelets,
    resolve_notebook_wavelet_cache_dir,
)
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.analysis.wavelet_ica import zscore_by_time  # noqa: E402
from src.analysis.wavelet_jica import (  # noqa: E402
    DEFAULT_SIGNAL_VARIANTS,
    IDENTITY_TOLERANCE,
    SIGNAL_VARIANTS,
    VARIANT_RECIPE,
    VARIANT_REFERENCE_FROM,
    VARIANT_UNITS,
    JointIcaResult,
    JoinLayout,
    assemble_join,
    bootstrap_median_ci,
    cohort_mean_filter,
    cohort_mean_pattern,
    component_polarity,
    fit_joint_ica,
    subspace_mask_rows,
    subspace_reference_rows,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    ConditionVariants,
    CoordinateSystems,
    ExclusionCategories,
    ExperimentNames,
    JicaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)
from src.io.loading import assr_electrode_mask  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_topomaps,
    plot_participant_condition_topomaps,
)
from src.visualization.iva_quality_plots import (  # noqa: E402
    participant_sort_key,
    topo_info_subset,
)
from src.visualization.jica_plots import (  # noqa: E402
    DIFFERENCE_ROW,
    plot_global_tf_grid,
    plot_loading_bars,
    plot_pvalue_summary,
    plot_response_courses,
    plot_snr_vs_reference,
)

_logger = logging.getLogger(__name__)

#: Stage directory under ``plots/``.
_STAGE_DIR = "07-ica-condition-comparison"

#: What pinned the component sign on the DECOMPOSITION figures: the hypothesis-free
#: rule `orient_components` applies at fit time. One sign per component, not per
#: recording — jICA has no per-recording ambiguity to resolve.
_SIGN_NOTE = "largest |TF| excursion positive (one sign per component)"

#: What pinned it on the READ-OUT figures, where a directional test needs "higher =
#: more power over the reference electrodes" to be true. Still one sign per component,
#: re-anchored to the cohort topography by `component_polarity`.
_READOUT_SIGN_NOTE = "corr(cohort topography, ASSR mask), one sign per component"

#: Frequency (Hz) marked on every TF panel — the ASSR stimulation frequency.
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]

#: Below this many present reference electrodes the run refuses: it is no longer the
#: fronto-central selection the reference is meant to be, which is what a leading
#: channel subset produces.
_MIN_REFERENCE_ELECTRODES = 5

#: One colour per condition, shared by every figure.
_CONDITION_COLORS = {
    ConditionVariants.PLACEBO.value: "#0F6E8C",
    ConditionVariants.PSILOCYBIN.value: "#A6357F",
}

#: How the two joins map onto their plot subdirectory and product label.
_JOIN_CHOICES = {
    "joined": (ConditionVariants.JOINED, JicaVariants.CHANNEL_JOINED),
    "joined_tracks": (
        ConditionVariants.JOINED_TRACKS,
        JicaVariants.CHANNEL_JOINED_TRACKS,
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the joint-ICA decomposition of wavelet power on one of the two "
            "condition joins and the 40 Hz response read-out built on it. Reproduces "
            "the stage-07 notebooks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentNames],
        default=ExperimentNames.ASSR.value,
        help="Which experiment dataset to analyse.",
    )
    parser.add_argument(
        "--music_type",
        nargs="+",
        choices=[mt.value for mt in MusicTypeVariants],
        default=None,
        help=(
            "One or more music types, each analysed in its own run. When omitted, "
            "defaults to CLASSIC + PSYTRANCE for psilo_music and ASSR for assr."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[c.value for c in REAL_CONDITIONS],
        default=[c.value for c in REAL_CONDITIONS],
        help=(
            "Conditions to join, in the order their blocks (or time segments) appear "
            "and their rows appear in every figure. The contrast is the first minus "
            "the second."
        ),
    )
    parser.add_argument(
        "--join",
        choices=sorted(_JOIN_CHOICES),
        default="joined_tracks",
        help=(
            "'joined_tracks' gives one channel block per participant whose sample axis "
            "spans both tracks, so the topography and loading are SHARED and the "
            "contrast lives in the TF maps. 'joined' gives one block per recording, so "
            "each condition has its own topography and loading and the contrast lives "
            "there instead."
        ),
    )
    parser.add_argument(
        "--n_ica",
        type=int,
        default=5,
        help=(
            "Independent components to extract, and the dimension the stacked "
            "channel axis is reduced to before FastICA. NOT capped by the channel "
            "count (the feature axis is blocks x channels) but capped in practice by "
            "convergence, which the run reports. Fewer than the notebooks' 10 by "
            "default: every component adds an uncorrected test to two families."
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
        "--ica_algorithm",
        choices=["parallel", "deflation"],
        default="parallel",
        help=(
            "FastICA update. 'deflation' extracts one component at a time and often "
            "converges where the symmetric 'parallel' update oscillates."
        ),
    )
    parser.add_argument(
        "--ica_fun",
        choices=["logcosh", "exp", "cube"],
        default="logcosh",
        help="FastICA contrast function ('cube' favours spiky sources).",
    )
    parser.add_argument(
        "--ica_max_iter", type=int, default=2000, help="Maximum FastICA iterations."
    )
    parser.add_argument(
        "--ica_tol", type=float, default=1e-4, help="FastICA convergence tolerance."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for FastICA's initial unmixing.",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=None,
        help=(
            "Keep only the first N matched participants, by ID. Defaults to every "
            "matched participant, and should stay there: the participant is the unit "
            "of every test and the exact Wilcoxon floor is 2/2**N, so a small cohort "
            "cannot reject at all."
        ),
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=None,
        help=(
            "Keep only the first N channels. Defaults to all of them, and should stay "
            "there whenever --skip_analysis is not passed: the ASSR-electrode "
            "reference is spread over the montage, so a leading slice does not contain "
            "it."
        ),
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=None,
        help=(
            "Keep only the first N time samples OF EACH CONDITION. Bounds how many "
            "stimulus onsets fit the window and drives peak memory. Defaults to each "
            "condition's whole time axis."
        ),
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        default=True,
        help="Cast the loaded wavelet power to float32, halving peak memory.",
    )
    parser.add_argument(
        "--float64",
        dest="float32",
        action="store_false",
        help="Keep the cache's own dtype instead of casting to float32.",
    )
    add_wavelet_grid_args(parser)
    parser.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help=(
            "Base data directory holding the per-condition wavelet caches. The "
            "'<experiment>/wavelets/broadband' suffix is appended automatically."
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
            "notebooks/03-wavelet-analysis/wavelet_cache/<experiment>/, shared with "
            "every other wavelet workflow. Worth it only for a channel/time subset."
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

    # ---- the 40 Hz read-out ------------------------------------------------
    parser.add_argument(
        "--assr_freq",
        type=float,
        default=iva_quality.ASSR_FREQ,
        help="Stimulation frequency the response is read at.",
    )
    parser.add_argument(
        "--halfwidths",
        type=float,
        nargs="+",
        default=[0.0, iva_quality.TF_ANCHOR_HALFWIDTH_HZ],
        help=(
            "Frequency half-widths (Hz) to extract, each its own selection. 0 takes "
            "the single nearest bin — on the 1 Hz grid that IS the 40 Hz row; a "
            "positive one averages the closed interval, which survives wavelet "
            "smearing and stimulator drift at the cost of off-frequency power."
        ),
    )
    parser.add_argument(
        "--test_halfwidth",
        type=float,
        default=0.0,
        help=(
            "Which of --halfwidths the tests and figures reduce over. Must be one of "
            "them."
        ),
    )
    parser.add_argument(
        "--stimulus_interval",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help=(
            "Custom response window in seconds, e.g. 0.2 0.5 to skip the onset "
            "transient and read only the sustained steady state. Defaults to the "
            "paradigm's full driven interval."
        ),
    )
    parser.add_argument(
        "--response_measure",
        choices=["stimulus", "stimulus_minus_rest"],
        default="stimulus",
        help=(
            "How a trial's time course becomes one number. 'stimulus_minus_rest' "
            "cancels whatever offset the per-trial normalisation left behind. Fix this "
            "BEFORE looking at any p-value."
        ),
    )
    parser.add_argument(
        "--signal_variants",
        nargs="+",
        choices=list(SIGNAL_VARIANTS),
        default=list(DEFAULT_SIGNAL_VARIANTS),
        help=(
            "Read-out cells to run side by side, as a (spatial filter x signal) grid. "
            "The default is the six per-trial-baselined cells — the IVA stage's grid "
            "cell for cell — so every row is in units of its own pre-stimulus SD and "
            "all six are comparable: {'prestim', 'zscored_prestim'} use each block's "
            "OWN filter, {'mean_prestim', 'mean_zscored_prestim'} one cohort-mean "
            "filter shared by everybody, and {'masked_prestim', 'mean_masked_prestim'} "
            "either of those with every weight outside the ASSR electrodes zeroed, "
            "which separates a better weighting INSIDE the anchor area from access to "
            "signal outside it. The 'zscored*' and 'masked*' rows apply the filter to "
            "exactly the signal the fit saw (so the per-block term IS the component), "
            "the rest to RAW power (which approximates it, dropping the per-channel "
            "1/sd weighting the fit folded in). Every baselined variant borrows its "
            "reference rows from 'prestim', so they require it. 'zscored' (no "
            "baseline) is kept for back-compatibility and is off by default: without a "
            "baseline it is not in the reference's units, which is what the grid is for."
        ),
    )
    parser.add_argument(
        "--mean_filter_align",
        action="store_true",
        help=(
            "Flip each block's pattern toward the reference electrodes BEFORE "
            "averaging them into the cohort filter — the stage-06 IVA convention. Off "
            "by default here, and that is deliberate: IVA-G's objective is invariant "
            "under per-dataset sign flips so those signs are arbitrary and must be "
            "aligned, whereas a jICA block's sign is a RESULT, so aligning first would "
            "manufacture a coherent cohort topography where the cohort has none. The "
            "per-block cosine-to-mean reported by the run is what says whether it "
            "would have mattered."
        ),
    )
    parser.add_argument(
        "--mean_filter_raw_patterns",
        action="store_true",
        help=(
            "Average the cohort filter's patterns at their native scale instead of "
            "normalising each to unit L2 norm first. Not recommended: jICA's per-block "
            "loadings are unconstrained, so one high-loading block would set the "
            "cohort topography by itself."
        ),
    )
    parser.add_argument(
        "--contrast_alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help=(
            "Direction of the condition contrast, computed as first minus second "
            "condition. 'greater' encodes the prior that psilocybin LOWERS the 40 Hz "
            "response; legitimate only because the polarity anchor makes 'higher = "
            "more power over the reference electrodes' true of every row."
        ),
    )
    parser.add_argument(
        "--snr_alternative",
        choices=["greater", "less", "two-sided"],
        default="less",
        help=(
            "Direction of the per-condition component-vs-reference magnitude test. "
            "'less' encodes the expectation that a per-block filter — a contribution "
            "to a shared source — recovers less power than a fixed selection aimed at "
            "the response."
        ),
    )
    parser.add_argument(
        "--no_polarity_anchor",
        action="store_true",
        help=(
            "Skip anchoring each component's polarity to the reference electrodes. "
            "The contrast direction is then NOT interpretable."
        ),
    )
    parser.add_argument(
        "--mask_sum",
        action="store_true",
        help=(
            "Sum the reference electrodes instead of averaging them. The mean is the "
            "default: it keeps the output in the input's power units and does not "
            "scale with how many electrodes the list contains."
        ),
    )
    parser.add_argument(
        "--mask_not_strict",
        action="store_true",
        help=(
            "Accept the intersection when a listed reference electrode is absent "
            "(preprocessing drops the boundary electrodes) instead of refusing. Relax "
            "this only after reading which electrodes the log says are missing."
        ),
    )
    parser.add_argument(
        "--coordinate_system",
        choices=[c.value for c in CoordinateSystems],
        default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value,
        help="Coordinate system whose ASSR electrode list to use.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level for the figures."
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10_000,
        help=(
            "Percentile-bootstrap resamples over participants for the forest "
            "intervals. Spread only — the p-values are the exact Wilcoxon ones."
        ),
    )
    parser.add_argument(
        "--course_spread",
        choices=["sem", "iqr"],
        default="sem",
        help="Band drawn around each mean time course, ACROSS PARTICIPANTS.",
    )
    parser.add_argument(
        "--min_trials",
        type=int,
        default=5,
        help=(
            "Minimum stimulus epochs that must fit each condition's window before the "
            "read-out is attempted."
        ),
    )

    # ---- what to run and where it goes -------------------------------------
    parser.add_argument(
        "--skip_decomposition_plots",
        action="store_true",
        help="Fit the decomposition but draw none of its figures.",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Stop after the decomposition figures; run no trials and no tests.",
    )
    parser.add_argument(
        "--skip_onset_average",
        action="store_true",
        help=(
            "Skip the stimulus-averaged TF grid. Skipped anyway for experiments "
            "without stimulus annotations."
        ),
    )
    parser.add_argument(
        "--participant_grids",
        action="store_true",
        help=(
            "Also write the per-participant topography grids — one figure PER "
            "COMPONENT, so they are opt-in."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Base directory for output plots. Defaults to the project plots/ root.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help=(
            "Directory for the test CSV. Defaults to results/<experiment>/ under the "
            "project root."
        ),
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Parallel jobs for data loading."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LoadedCohort:
    """One join's raw wavelet power plus everything needed to interpret its axes.

    :param raw_by_condition: Condition name → ``(P, C, F, T)`` **un-z-scored** power,
        participant-matched row for row. What the trials are cut from.
    :param participants: Participant label per row.
    :param channel_names: Channel name per channel, so the electrode mask can be
        aligned to the data.
    :param onsets_by_condition: Condition name → stimulus onsets local to that
        condition's own time axis, or ``None`` when the experiment has none.
    :param info: MNE ``Info`` restricted to the channel axis, for the topomaps.
    :param label: Product label, e.g. ``"JoinedTracks_ASSR"``.
    :param sfreq: Sampling frequency in Hz.
    """

    raw_by_condition: dict[str, np.ndarray]
    participants: list[str]
    channel_names: list[str]
    onsets_by_condition: dict[str, np.ndarray | None]
    info: object
    label: str
    sfreq: float


def _load_cohort(
    music_type: MusicTypeVariants,
    *,
    join: ConditionVariants,
    experiment_name: ExperimentNames,
    conditions: list[ConditionVariants],
    exclusion_categories: list[ExclusionCategories],
    freqs: np.ndarray,
    wavelet_dir: Path,
    reuse_wavelets: bool,
    subset_cache_dir: Path | None,
    reuse_subset_cache: bool,
    n_pairs: int | None,
    n_channels: int | None,
    n_times: int | None,
    as_float32: bool,
    n_jobs: int,
) -> LoadedCohort:
    """Load one join's raw wavelet power, whichever loader that join needs.

    The two joins have genuinely different requirements, which is why the loader
    differs: the recording-axis join stacks recordings into one subject axis, so it
    needs both conditions on one time base; the time-axis join lays them end to end
    and does not. The latter is asked for ``zscore_mode="none"`` because the trials
    must be cut from **raw** power — the decomposition's own standardisation is
    applied later, by :func:`~src.analysis.wavelet_jica.assemble_join`, where it
    belongs to the decomposition rather than to the signal.

    :return: The loaded cohort. See :class:`LoadedCohort` for the fields.
    :raises ValueError: If the channel names cannot be aligned to the data.
    """
    common = dict(
        wavelet_dir=wavelet_dir,
        experiment_name=experiment_name,
        conditions=conditions,
        n_channels=n_channels,
        n_times=n_times,
        reuse_wavelets=reuse_wavelets,
        subset_cache_dir=subset_cache_dir,
        reuse_subset_cache=reuse_subset_cache,
        n_jobs=n_jobs,
    )
    if join is ConditionVariants.JOINED:
        pooled, analyzers = load_joined_condition_wavelets(
            music_type, exclusion_categories, freqs, **common
        )
        if n_pairs is not None:
            keep = sorted(set(pooled.participants), key=participant_sort_key)[:n_pairs]
            pooled = pooled.select_participants(keep)
        data = pooled.data
        # Row order WITHIN a condition block; condition_subjects returns both
        # conditions' rows in this same order, so they are participant-matched.
        participants = [
            p
            for p, c in zip(pooled.participants, pooled.subject_conditions)
            if c is pooled.conditions[0]
        ]
        raw_by_condition = {
            c.value: pooled.condition_subjects(data.data, c) for c in pooled.conditions
        }
        onsets_by_condition = {
            c.value: pooled.condition_onsets(c) for c in pooled.conditions
        }
        analyzer = analyzers[pooled.conditions[0]]
    else:
        paired, analyzers = load_paired_condition_wavelets(
            music_type, exclusion_categories, freqs, zscore_mode="none", **common
        )
        if n_pairs is not None and n_pairs < paired.n_pairs:
            keep = sorted(paired.participants, key=participant_sort_key)[:n_pairs]
            rows = [paired.participants.index(p) for p in keep]
            paired = dataclasses.replace(
                paired,
                data=dataclasses.replace(paired.data, data=paired.data.data[rows]),
                participants=tuple(keep),
            )
        data = paired.data
        participants = list(paired.participants)
        # Views into the concatenated tensor — deliberately not copies, which would
        # double the largest array in the run for nothing.
        raw_by_condition = {
            c.value: paired.condition_track(data.data, c) for c in paired.conditions
        }
        onsets_by_condition = {
            c.value: paired.condition_onsets(c, absolute=False)
            for c in paired.conditions
        }
        analyzer = analyzers[paired.conditions[0]]

    if as_float32:
        raw_by_condition = {
            name: np.asarray(array, dtype=np.float32)
            for name, array in raw_by_condition.items()
        }

    n_data_channels = raw_by_condition[conditions[0].value].shape[1]
    info = topo_info_subset(analyzer.info, n_data_channels)
    if data.feature_names is not None and len(data.feature_names) == n_data_channels:
        channel_names = list(data.feature_names)
    else:
        channel_names = list(info["ch_names"])
    if len(channel_names) != n_data_channels:
        raise ValueError(
            f"{len(channel_names)} channel name(s) for a {n_data_channels}-channel "
            "axis; the electrode mask could not be aligned."
        )

    _logger.info(
        f"[{data.label}] {len(participants)} participant(s), {n_data_channels} "
        f"channel(s), "
        + ", ".join(f"{name} {array.shape}" for name, array in raw_by_condition.items())
    )
    return LoadedCohort(
        raw_by_condition=raw_by_condition,
        participants=participants,
        channel_names=channel_names,
        onsets_by_condition=onsets_by_condition,
        info=info,
        label=data.label,
        sfreq=data.sfreq,
    )


# ---------------------------------------------------------------------------
# The decomposition figures
# ---------------------------------------------------------------------------


def _write_decomposition_plots(
    result: JointIcaResult,
    layout: JoinLayout,
    cohort: LoadedCohort,
    *,
    freqs: np.ndarray,
    comp_indices: list[int],
    out_dir: Path,
    onset_average: bool,
    participant_grids: bool,
    min_trials: int,
) -> None:
    """Draw the component maps, the topographies and the loading.

    Which side carries the condition contrast is exactly what the join decides, so
    the figures differ between them rather than being drawn identically:

    * ``JOINED_TRACKS`` — the sample axis spans both tracks, so the TF maps split
      into a genuine per-condition contrast with a difference row, while the
      topography and the loading are **shared** and get a single row.
    * ``JOINED`` — the sample axis is common to both conditions, so the TF maps get a
      single shared row while the topographies and loadings carry the contrast.

    Drawing a shared quantity as a two-row comparison would produce two identical rows
    and an all-zero difference — a figure that looks like a null result but is really
    a statement about the model.

    :param result: The fitted decomposition.
    :param layout: The join it was fitted on.
    :param cohort: The loaded cohort, for the onsets and the topomap ``Info``.
    :param freqs: Frequency grid the maps are on.
    :param comp_indices: 0-based components to draw.
    :param out_dir: Directory the figures go in.
    :param onset_average: Also draw the stimulus-averaged grid.
    :param participant_grids: Also write the per-participant topography grids.
    :param min_trials: Minimum epochs per condition for the stimulus average.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(layout.conditions)
    shared_row = " + ".join(conditions) + " (shared)"
    tracks = layout.join is ConditionVariants.JOINED_TRACKS
    lengths = {c: cohort.raw_by_condition[c].shape[-1] for c in conditions}

    # ---- the component TF maps -------------------------------------------
    if tracks:
        edges = np.cumsum([0] + [lengths[c] for c in conditions])
        by_condition = {
            c: result.tf_maps[..., edges[i] : edges[i + 1]]
            for i, c in enumerate(conditions)
        }
        n_split = min(array.shape[-1] for array in by_condition.values())
        rows = {c: array[..., :n_split] for c, array in by_condition.items()}
        rows[DIFFERENCE_ROW] = rows[conditions[1]] - rows[conditions[0]]
        row_order = [*conditions, DIFFERENCE_ROW]
        times = np.arange(n_split) / cohort.sfreq
        tf_name = "condition_tf_maps"
        tf_title = "Component TF maps, split by condition"
    else:
        by_condition = None
        rows = {shared_row: result.tf_maps}
        row_order = [shared_row]
        times = np.arange(result.tf_maps.shape[-1]) / cohort.sfreq
        tf_name = "shared_tf_maps"
        tf_title = "Component TF maps (one per component, shared by both conditions)"

    reference_onsets = cohort.onsets_by_condition.get(conditions[0])
    time_marks = (
        None
        if reference_onsets is None
        else np.asarray(reference_onsets)[np.asarray(reference_onsets) < times.size]
        / cohort.sfreq
    )
    fig = plot_global_tf_grid(
        rows,
        row_order,
        comp_indices,
        freqs,
        times,
        label=cohort.label,
        title=tf_title,
        time_marks=time_marks,
        freq_marks=_TF_FREQ_MARKS,
        sign_note=_SIGN_NOTE,
        save_path=out_dir / f"{tf_name}.png",
    )
    plt.close(fig)

    # ---- the same, averaged over stimuli ---------------------------------
    if onset_average:
        _write_onset_tf_grid(
            result,
            layout,
            cohort,
            freqs=freqs,
            comp_indices=comp_indices,
            out_dir=out_dir,
            by_condition=by_condition,
            shared_row=shared_row,
            min_trials=min_trials,
        )

    # ---- the topographies -------------------------------------------------
    n_channels = layout.n_channels
    if tracks:
        # One pattern per participant, shared: a single row, one entry per participant.
        patterns = result.recording_patterns(layout, conditions[0])
        row_participants = list(layout.participants)
        row_conditions = [shared_row] * len(row_participants)
        topo_rows = [shared_row]
        topo_name, topo_prefix = "shared_mean_topomaps", "shared_"
    else:
        patterns = np.concatenate(
            [result.recording_patterns(layout, c) for c in conditions]
        )
        row_participants = [p for _c in conditions for p in layout.participants]
        row_conditions = [c for c in conditions for _p in layout.participants]
        topo_rows = list(conditions)
        topo_name, topo_prefix = "condition_mean_topomaps", ""
    fig = plot_condition_mean_topomaps(
        patterns,
        row_participants,
        row_conditions,
        topo_rows,
        cohort.info,
        n_channels,
        comp_indices,
        label=cohort.label,
        alignment_note=_SIGN_NOTE,
        save_path=out_dir / f"{topo_name}.png",
    )
    plt.close(fig)
    if participant_grids:
        paths = plot_participant_condition_topomaps(
            patterns,
            row_participants,
            row_conditions,
            topo_rows,
            cohort.info,
            n_channels,
            comp_indices,
            label=cohort.label,
            root_dir=out_dir / "participants",
            prefix=topo_prefix,
            alignment_note=_SIGN_NOTE,
        )
        _logger.info(f"wrote {len(paths)} per-participant topography figure(s)")

    # ---- the loading ------------------------------------------------------
    # Rows of block_ic_energy are the channel blocks, so the bookkeeping is the join's
    # own: one row per participant for a time join, one per recording otherwise.
    loading_participants = [block[0] for block in layout.blocks]
    loading_conditions = [
        shared_row if block[1] is None else block[1] for block in layout.blocks
    ]
    loading_groups = [shared_row] if tracks else list(conditions)
    energy = result.block_ic_energy
    fig = plot_loading_bars(
        energy * 100.0,
        loading_participants,
        loading_conditions,
        loading_groups,
        comp_indices,
        label=cohort.label,
        title=(
            "Per-participant component loading (shared by both conditions)"
            if tracks
            else "Per-recording component loading"
        ),
        ylabel="% of joint channel-space energy",
        colors=None if tracks else _CONDITION_COLORS,
        save_path=out_dir / "loading_bars_absolute.png",
    )
    plt.close(fig)
    within = energy / energy.sum(axis=0, keepdims=True)
    fig = plot_loading_bars(
        within * 100.0,
        loading_participants,
        loading_conditions,
        loading_groups,
        comp_indices,
        label=cohort.label,
        title=(
            "Share of each component carried by each block "
            f"(even split = {100 / layout.n_blocks:.1f}%)"
        ),
        ylabel="% of this IC's energy",
        colors=None if tracks else _CONDITION_COLORS,
        save_path=out_dir / "loading_bars_within_component.png",
    )
    plt.close(fig)
    _logger.info(f"decomposition figures -> {out_dir}")


def _write_onset_tf_grid(
    result: JointIcaResult,
    layout: JoinLayout,
    cohort: LoadedCohort,
    *,
    freqs: np.ndarray,
    comp_indices: list[int],
    out_dir: Path,
    by_condition: dict[str, np.ndarray] | None,
    shared_row: str,
    min_trials: int,
) -> None:
    """The component TF maps averaged over stimuli.

    The whole-recording map answers "what does this component do" but cannot answer
    "what does it do *to a stimulus*": the ASSR is a train of short stimuli, and a
    response locked to their onsets is smeared across a map that spans every stimulus
    at once. Each condition is epoched with **its own** onsets on its own segment,
    because a time join leaves the tracks on their own time bases.

    Each frequency is then referenced to its own pre-onset mean
    (:func:`~src.analysis.iva_quality.subtract_epoch_baseline`), exactly as the stage-06
    decomposition does. The z-scoring puts the pre-onset interval near 0 over the
    **whole** recording, but not within any one epoch — the local level still drifts —
    so this is what makes the map a change rather than a level. Only the mean is
    removed; see that function for why no divisor is applied. Each condition is
    referenced to its own baseline before the difference row is taken, so the difference
    is between two changes rather than between two levels.
    """
    conditions = list(layout.conditions)
    geometries: dict[str, tuple] = {}
    for condition in conditions:
        onsets = cohort.onsets_by_condition.get(condition)
        track = result.tf_maps if by_condition is None else by_condition[condition]
        if onsets is None or len(onsets) == 0:
            _logger.info(
                f"[{cohort.label}] no stimulus onsets for {condition}; the "
                "stimulus-averaged grid is skipped."
            )
            return
        try:
            geometries[condition] = at.epoch_geometry(
                np.asarray(onsets), track.shape[-1], cohort.sfreq
            )
        except ValueError as error:
            _logger.info(f"[{cohort.label}] {condition}: {error} Grid skipped.")
            return
    pre, post = at.common_epoch_window(geometries)

    epoch_times = at.epoch_time_base(pre, post, cohort.sfreq)
    epoch_marks = [0.0, min(AssrEpoch.STIMULUS_DURATION_S, float(epoch_times[-1]))]
    baseline_mask = epoch_times < 0.0

    rows: dict[str, np.ndarray] = {}
    for condition in conditions:
        onsets = geometries[condition][0]
        track = result.tf_maps if by_condition is None else by_condition[condition]
        fitting = int(((onsets - pre >= 0) & (onsets + post <= track.shape[-1])).sum())
        if fitting < min_trials:
            _logger.info(
                f"[{cohort.label}] {condition}: {fitting} epoch(s) fit, need "
                f"{min_trials}. Stimulus-averaged grid skipped."
            )
            return
        averaged, n_used = iva_quality.epoch_average(track, onsets, pre, post)
        # Per FREQUENCY, so the delta row and the gamma row each get their own baseline.
        rows[condition] = iva_quality.subtract_epoch_baseline(averaged, baseline_mask)
        _logger.info(
            f"[{cohort.label}] {condition}: averaged {n_used} epoch(s), each frequency "
            "referenced to its own pre-onset mean"
        )
        if by_condition is None:
            # One shared map: both conditions read the same onsets, so one row.
            break

    if by_condition is None:
        row_order = [shared_row]
        rows = {shared_row: rows[conditions[0]]}
        name = "shared_tf_maps_onset"
    else:
        rows[DIFFERENCE_ROW] = rows[conditions[1]] - rows[conditions[0]]
        row_order = [*conditions, DIFFERENCE_ROW]
        name = "condition_tf_maps_onset"

    fig = plot_global_tf_grid(
        rows,
        row_order,
        comp_indices,
        freqs,
        epoch_times,
        label=cohort.label,
        title="Onset-averaged component TF maps",
        freq_marks=_TF_FREQ_MARKS,
        epoch_marks=epoch_marks,
        sign_note=_SIGN_NOTE,
        save_path=out_dir / f"{name}.png",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# The 40 Hz read-out
# ---------------------------------------------------------------------------


def _run_analysis(
    result: JointIcaResult,
    layout: JoinLayout,
    cohort: LoadedCohort,
    *,
    freqs: np.ndarray,
    args: argparse.Namespace,
    product: str,
    out_dir: Path,
) -> list[dict]:
    """Project, cut trials, test, and draw the read-out. Returns the test rows.

    Every step from the frequency selection onwards is
    :mod:`src.analysis.assr_trials`, unchanged: it is decomposition-agnostic, and
    running the two decompositions through the same code is what makes their results
    comparable at all.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(layout.conditions)
    participants = list(layout.participants)
    n_participants = len(participants)

    # ---- the fixed reference ---------------------------------------------
    mask = assr_electrode_mask(
        cohort.channel_names,
        CoordinateSystems(args.coordinate_system),
        strict=not args.mask_not_strict,
    )
    present = int(mask.sum())
    reference_channels = [
        name for name, keep in zip(cohort.channel_names, mask) if keep
    ]
    _logger.info(
        f"reference electrodes: {present} of {mask.size} "
        f"({'sum' if args.mask_sum else 'mean'}) {reference_channels}"
    )
    if present < _MIN_REFERENCE_ELECTRODES:
        raise ValueError(
            f"Only {present} ASSR electrode(s) are present on this {mask.size}-channel "
            "axis, so the reference would not be the fronto-central selection it is "
            "meant to be. Drop --n_channels: the selection is spread over the montage, "
            "so a leading channel slice does not contain it."
        )
    reference_filter = at.binary_filter_weights(mask, normalize=not args.mask_sum)
    # Kept UNCOMBINED alongside the weighting above so the reference row can be built
    # the equal-weight way below: each electrode normalised before the ROI is averaged,
    # which `reference_filter` — a mean of raw power — cannot express.
    roi_index = np.flatnonzero(mask)

    # THREE reference rows, the same three the IVA stage emits and named identically
    # (`at.MASK_LABELS`), in increasing proximity to what the fit actually saw: the
    # electrode average on the whole channel space, the same average read only through
    # the retained subspace on RAW power, and that operator on the Z-SCORED signal —
    # the signal the decomposition was handed. Reading the first beside the last says
    # how much of the reference's advantage is access to directions the reduction
    # dropped rather than a better weighting.
    labels = [
        at.COMPONENT_LABEL.format(k=k + 1) for k in range(result.n_components)
    ] + list(at.MASK_LABELS)
    ic_labels = labels[: result.n_components]
    reference_columns = {label: labels.index(label) for label in at.MASK_LABELS}
    mask_columns = list(reference_columns.values())
    # The equal-weight row replaces this column in every baselined variant; it is also
    # the reference the figures lead with, and the only one the IVA stage and this one
    # compute identically (the PCA rows read each stage's own reduction).
    full_index = reference_columns[at.FULL_MASK_LABEL]
    # The mask carried into the JOINT reduction — one row per channel block, spanning
    # the whole stacked feature axis. Independent of the selection and of the variant,
    # so it is built once and applied to the data per selection below.
    subspace_rows = subspace_mask_rows(result, reference_filter)

    # ---- the epoch window every condition can supply ---------------------
    geometries = {}
    for condition in conditions:
        onsets = cohort.onsets_by_condition.get(condition)
        if onsets is None or len(onsets) == 0:
            _logger.warning(
                f"[{cohort.label}] no stimulus onsets for {condition}; the read-out "
                "needs them and is skipped."
            )
            return []
        geometries[condition] = at.epoch_geometry(
            np.asarray(onsets),
            cohort.raw_by_condition[condition].shape[-1],
            cohort.sfreq,
        )
    pre, post = at.common_epoch_window(geometries)
    epoch_times = at.epoch_time_base(pre, post, cohort.sfreq)
    baseline_mask = epoch_times < 0.0
    if args.stimulus_interval is None:
        window_mask = AssrEpoch.stimulus_mask(epoch_times)
        window_desc = f"0-{AssrEpoch.STIMULUS_DURATION_S:.3f} s (paradigm default)"
    elif tuple(args.stimulus_interval) == (0.0, AssrEpoch.STIMULUS_DURATION_S):
        # Typing the paradigm's own interval means the paradigm's own window, read
        # half-open like the default rather than a sample wider — so the row is the
        # same whichever way it was asked for, and the same one the IVA stage reads.
        window_mask = AssrEpoch.stimulus_mask(epoch_times)
        window_desc = f"0-{AssrEpoch.STIMULUS_DURATION_S:.3f} s (paradigm default)"
    else:
        low, high = args.stimulus_interval
        window_mask = (epoch_times >= low) & (epoch_times <= high)
        window_desc = f"{low:.3f}-{high:.3f} s (custom)"
        if not window_mask.any():
            raise ValueError(
                f"--stimulus_interval {args.stimulus_interval} selects no epoch "
                f"sample; the epoch spans [{epoch_times[0]:.3f}, "
                f"{epoch_times[-1]:.3f}] s."
            )

    # ---- the selections, and which one the tests read --------------------
    selections = {
        at.selection_label(args.assr_freq, hw): at.frequency_selection(
            freqs, args.assr_freq, hw
        )
        for hw in args.halfwidths
    }
    test_selection = at.selection_label(args.assr_freq, args.test_halfwidth)
    if test_selection not in selections:
        raise ValueError(
            f"--test_halfwidth {args.test_halfwidth} names selection "
            f"{test_selection!r}, which is not among --halfwidths "
            f"{sorted(selections)}. Add it."
        )
    band_bins = np.array(sorted({int(b) for bins in selections.values() for b in bins}))
    local_bins = {
        name: np.array([int(np.flatnonzero(band_bins == b)[0]) for b in bins])
        for name, bins in selections.items()
    }

    # ---- the cohort-mean filter, and the one sign per component ----------
    # Both come off the SAME object: the cohort forward model, averaged over every
    # block with both conditions pooled. Pooling is what keeps the operator symmetric
    # in the conditions, so neither the `mean_*` rows nor the sign can manufacture a
    # condition difference.
    cohort_topo = cohort_mean_pattern(
        result.patterns,
        normalize=not args.mean_filter_raw_patterns,
        align_to=mask if args.mean_filter_align else None,
    )
    mean_filter = cohort_mean_filter(cohort_topo.pattern)
    flip_k, strength_k = component_polarity(cohort_topo.pattern, mask)

    # The per-block correlations are NOT applied — that is the whole point — but they
    # are what says whether the cohort expresses a component coherently, so they are
    # reported. A component whose blocks disagree has a weak cohort topography however
    # large its loadings.
    block_flip, block_strength = at.polarity_flip(result.patterns, mask)
    disagreeing = cohort_topo.disagreeing_blocks()
    n_blocks = block_flip.shape[0]
    _logger.info(
        f"[{cohort.label}] polarity: ONE sign per component, anchored to "
        "corr(cohort topography, ASSR mask). jICA leaves no per-recording sign to "
        "resolve, so no per-recording flip is applied."
    )
    for k in range(result.n_components):
        against = int((block_flip[:, k] != flip_k[k]).sum())
        _logger.info(
            f"[{cohort.label}]   IC {k + 1}: flip {flip_k[k]:+.0f}, "
            f"|corr(cohort topo, mask)| {strength_k[k]:.3f}"
            f"{' (WEAK)' if strength_k[k] < at.POLARITY_CORR_FLOOR else ''}; "
            f"{against}/{n_blocks} block(s) anchor the other way, "
            f"{int(disagreeing[k])}/{n_blocks} point away from the cohort mean "
            f"(median |corr| {np.median(block_strength[:, k]):.3f}, median cosine "
            f"{np.median(cohort_topo.cosine[:, k]):+.3f})"
        )
    weak = [
        k + 1
        for k in range(result.n_components)
        if strength_k[k] < at.POLARITY_CORR_FLOOR
    ]
    if weak:
        _logger.warning(
            f"[{cohort.label}] IC {weak}: the cohort topography barely projects onto "
            "the reference electrodes, so the component's sign — and with it the "
            "direction of every one-sided test on that row — is decided on noise."
        )

    # One vector over components, broadcast over participants and identical for both
    # conditions. The reference column is never flipped: it is a non-negative electrode
    # average, so "higher = more power there" already holds for it.
    source_flip = np.ones(len(labels))
    if not args.no_polarity_anchor:
        source_flip[: result.n_components] = flip_k
    flip_by_condition = {condition: source_flip for condition in conditions}

    # ---- project, cut, normalise, reduce ---------------------------------
    # Every selection is carried through, so the CSV says whether a result survives
    # the other frequency window; the figures are drawn for --test_halfwidth only.
    value_by_cell: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    course_by_cell: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    n_trials: dict[str, int] = {}
    for selection, bins in local_bins.items():
        # ---- the fixed reference row, built the equal-weight way ----------
        # Each ROI electrode is referenced to its own pre-stimulus window and divided by
        # its baseline SD (pooled over trials) BEFORE the ROI is averaged, and the
        # average is then referenced to its own again so the row lands back in units of
        # its own baseline SD. Averaging raw power first — which is what a plain
        # `reference_filter` projection does — silently weights each electrode by its
        # own power level, and that level is uncorrelated with whether the electrode
        # carries any 40 Hz response. The mask's whole intent is equal weight, so the
        # normalisation has to come first. It also matters for the comparison: one
        # normalised channel has baseline SD 1, but the MEAN of correlated channels does
        # not, so without this the reference sits systematically off every component row
        # — which `at.reference_snr_tests` subtracts from directly.
        #
        # Advanced-indexed on both the ROI channels and the selection's frequency bins
        # at once, so only that corner of the (possibly ~44 GB) raw tensor is ever
        # materialised.
        roi_snr: dict[str, np.ndarray] = {}
        roi_scale: dict[str, dict[str, float]] = {}
        for condition in conditions:
            roi_band = cohort.raw_by_condition[condition][
                :, roi_index[:, None], selections[selection][None, :], :
            ].mean(axis=2)  # (P, R, T)
            roi_cut, _kept = at.cut_trials(
                roi_band, geometries[condition][0], pre, post
            )
            roi_snr[condition], roi_scale[condition] = at.roi_channelwise_snr(
                roi_cut, baseline_mask
            )
        # Put every condition on ONE scale. Each call derived its own from its own
        # participants and those differ between conditions — leaving them apart would
        # rescale a participant's two conditions differently and corrupt the very paired
        # difference the contrast tests. The rescale is exact, not a re-fit:
        # snr / shared == (snr / own) * (own / shared).
        shared_roi_scale = float(
            np.median([d["roi_baseline_sd"] for d in roi_scale.values()])
        )
        for condition in conditions:
            roi_snr[condition] = roi_snr[condition] * (
                roi_scale[condition]["roi_baseline_sd"] / shared_roi_scale
            )
        _logger.info(
            f"[{cohort.label}] {selection}: equal-weight ROI over {roi_index.size} "
            f"electrode(s), shared baseline SD {shared_roi_scale:.3f} "
            f"(~{1.0 / shared_roi_scale**2:.1f} effective channels); per-condition "
            + ", ".join(
                f"{c} {roi_scale[c]['roi_baseline_sd']:.3f} (participant spread "
                f"{roi_scale[c]['participant_sd_spread']:.2f}x, NOT applied)"
                for c in conditions
            )
            + "; max |per-channel z| "
            f"{max(d['max_abs_z1'] for d in roi_scale.values()):.0f}"
        )

        # ---- the two subspace reference rows ------------------------------
        # The same electrode average read only through what the reduction kept, on RAW
        # power and on the z-scored signal the fit was handed. Neither depends on the
        # variant, so both are built once per selection and carried into every cell —
        # the same discipline the equal-weight row above follows.
        pca_reference = {
            zscore: subspace_reference_rows(
                cohort.raw_by_condition,
                layout,
                subspace_rows,
                selections[selection],
                zscore=zscore,
            )
            for zscore in (False, True)
        }
        # What the reduction left of the mask, as a share of its norm — scale-free, so
        # it says something the raw medians cannot: those are in raw power units and
        # look like zero however well the row is doing. A small share is expected and
        # is the joint reduction's doing (K directions out of blocks x channels, chosen
        # on z-scored data); it is why the "(PCA)" row is the weak one here and the
        # "(PCA, z)" row the informative one. See `subspace_reference_rows`.
        surviving = np.linalg.norm(subspace_rows, axis=1) / np.linalg.norm(
            reference_filter
        )
        _logger.info(
            f"[{cohort.label}] {selection}: subspace reference over "
            f"{layout.n_blocks} block(s) x {layout.n_channels} channel(s), "
            f"{result.retained * 100:.1f}% of the joint channel space retained; "
            f"the mask keeps {np.median(surviving) * 100:.2f}% of its norm "
            f"(min {surviving.min() * 100:.2f}%, max {surviving.max() * 100:.2f}%)"
        )

        for variant in args.signal_variants:
            cell_value: dict[str, np.ndarray] = {}
            cell_course: dict[str, np.ndarray] = {}
            recipe = VARIANT_RECIPE[variant]
            for condition in conditions:
                # Slicing the frequency axis before z-scoring is exact: zscore_by_time
                # standardises each frequency independently. Averaging over it is not,
                # so the band mean comes after.
                sliced = np.ascontiguousarray(
                    cohort.raw_by_condition[condition][:, :, band_bins, :]
                )
                if recipe["zscore"]:
                    # Apply the filter to exactly the signal the fit saw, so the
                    # per-block terms are the component itself rather than an
                    # approximation of it.
                    sliced = zscore_by_time(sliced)
                band = sliced[:, :, bins, :].mean(axis=2)  # (P, C, T)

                # The filter axis of the grid. "own" gives each recording its own
                # unmixing row — C free parameters per recording; "mean" gives everybody
                # the one cohort operator, which is the same KIND of thing the fixed
                # electrode mask is (a single group-level spatial weighting) and so is
                # the like-for-like learned comparison to it. The "masked" pair is
                # either of those with every weight outside the ASSR electrodes zeroed,
                # which separates a better weighting INSIDE the anchor area from access
                # to signal outside it.
                filter_kind = str(recipe["filter"])
                if filter_kind.startswith("mean"):
                    learned = np.broadcast_to(
                        mean_filter, (n_participants,) + mean_filter.shape
                    )
                else:
                    learned = result.recording_filters(layout, condition)
                if filter_kind.endswith("masked"):
                    learned = at.restrict_filters_to_mask(learned, mask)
                stacked = at.stack_filters(learned, reference_filter)
                projected = np.stack(
                    [
                        at.project_channels(stacked[i], band[i])
                        for i in range(n_participants)
                    ]
                )  # (P, K + 1, T)
                # The two subspace rows are not per-block channel filters — the joint
                # reduction mixes the blocks — so they are computed outside this
                # projection and appended here. In a variant with no per-trial baseline
                # the "(PCA)" row reads the z-scored signal too, which makes it coincide
                # with "(PCA, z)" by construction: a free check that the two agree.
                subspace_source = pca_reference[bool(recipe["zscore"])]
                projected = np.concatenate(
                    [
                        projected,
                        subspace_source[condition][:, np.newaxis, :],
                        pca_reference[True][condition][:, np.newaxis, :],
                    ],
                    axis=1,
                )  # (P, K + 3, T)
                trials, kept = at.cut_trials(
                    projected, geometries[condition][0], pre, post
                )
                n_trials[condition] = int(kept.size)
                if kept.size < args.min_trials:
                    _logger.warning(
                        f"[{cohort.label}] {condition}: {kept.size} trial(s) fit, "
                        f"need {args.min_trials}."
                    )
                if recipe["baseline"]:
                    # Units of each trial's own pre-stimulus SD — an SNR, and the unit
                    # the fixed reference is already in. The mean is per trial (drift is
                    # local), the SD pooled across trials; see `at.baseline_normalise`.
                    signal, _rel, _positive = at.baseline_normalise(
                        trials, baseline_mask
                    )
                    # Swap in the equal-weight reference row. Only the baseline variants
                    # get it: `zscored` has no per-trial baseline to build it on, and
                    # there its reference is the mask on exactly the signal the fit saw.
                    # Safe in place: `baseline_normalise` returns a fresh array, never a
                    # view of `trials`. The two subspace rows keep the plain per-trial
                    # baseline: they are projections, not electrode averages, so there is
                    # no per-electrode normalisation for them to skip.
                    signal[:, full_index] = roi_snr[condition]
                else:
                    # Already dimensionless: z-scored along time before projection.
                    signal = trials

                reduced = signal[..., window_mask].mean(axis=-1)
                if args.response_measure == "stimulus_minus_rest":
                    reduced = reduced - signal[..., ~window_mask].mean(axis=-1)
                # One sign per SOURCE, broadcast over participants: identical for every
                # recording and both conditions, so it cannot touch the paired contrast.
                flip = flip_by_condition[condition]
                cell_value[condition] = np.nanmedian(reduced, axis=2) * flip
                cell_course[condition] = np.nanmedian(signal, axis=2) * flip[:, None]
            value_by_cell[(selection, variant)] = cell_value
            course_by_cell[(selection, variant)] = cell_course

        # The reference rows are the quantity a component is judged against, so a
        # variant that borrows them is compared against exactly the same numbers as the
        # variant it borrows from — which is what makes the comparison isolate how the
        # COMPONENT was derived, and nothing else. ALL THREE move together: a grid whose
        # cells were judged against three references two of which had shifted would not
        # be one comparison.
        for variant, source_variant in VARIANT_REFERENCE_FROM.items():
            if variant not in args.signal_variants:
                continue
            if (selection, source_variant) not in value_by_cell:
                # Only reachable for a raw-signal variant run without `prestim`, which
                # `main` allows because its own rows come out identical anyway. A
                # z-scored one is refused there rather than silently left unborrowed.
                continue
            for condition in conditions:
                for store in (value_by_cell, course_by_cell):
                    store[(selection, variant)][condition][:, mask_columns] = store[
                        (selection, source_variant)
                    ][condition][:, mask_columns]

    floor = at.p_floor(n_participants)
    _logger.info(
        f"[{cohort.label}] read-out on {test_selection} @ {window_desc}, "
        f"{n_participants} participant(s), trials {n_trials}; exact Wilcoxon floor "
        f"p = {floor:.5f} two-sided"
    )
    if floor > 0.01:
        _logger.warning(
            f"With {n_participants} participants no p can fall below {floor:.4f}, so a "
            "null result is a statement about the cohort size as much as the effect. "
            "Drop --n_pairs."
        )

    # ---- the tests --------------------------------------------------------
    rows_out: list[dict] = []
    # variant -> reference -> condition -> records
    figure_snr_rows: dict[str, dict[str, dict[str, list[dict]]]] = {}
    figure_values: dict[str, dict[str, np.ndarray]] = {}
    for (selection, variant), value in value_by_cell.items():
        contrast_rows = [
            {
                "source": source,
                **at.paired_test(
                    at.condition_contrast(value, conditions, s),
                    args.contrast_alternative,
                ),
            }
            for s, source in enumerate(labels)
        ]
        # Both families run against EVERY reference row, which is the IVA stage's
        # structure and the reason the extra rows are worth their runtime: a component
        # that beats the whole-head electrode average but not the same average
        # restricted to the subspace the fit kept has not earned much, and only the
        # pair of numbers says which of the two it is. Learned rows only — a reference
        # judged against another reference is neither family's question.
        versus_rows = {
            reference: [
                {
                    "source": source,
                    "reference": reference,
                    "IC contrast": float(
                        np.nanmedian(at.condition_contrast(value, conditions, s))
                    ),
                    **at.paired_test(
                        at.discrimination_gain(value, conditions, s, column),
                        "two-sided",
                    ),
                }
                for s, source in enumerate(ic_labels)
            ]
            for reference, column in reference_columns.items()
        }
        snr_rows = {
            reference: {
                condition: [
                    {**row, "reference": reference}
                    for row in at.reference_snr_tests(
                        value[condition],
                        labels,
                        condition=condition,
                        reference_index=column,
                        alternative=args.snr_alternative,
                    )
                    if row["source"] in set(ic_labels)
                ]
                for condition in conditions
            }
            for reference, column in reference_columns.items()
        }

        common = {
            "join": layout.join.value,
            "product": product,
            "n_ica": result.n_components,
            "variant": variant,
            "selection": selection,
            "window": window_desc,
            "response": args.response_measure,
            "n_participants": n_participants,
        }
        for family, records in (
            ("contrast", contrast_rows),
            *(("vs_reference", rows) for rows in versus_rows.values()),
            *(
                ("snr", rows)
                for by_condition in snr_rows.values()
                for rows in by_condition.values()
            ),
        ):
            for row in records:
                d = _difference_for(
                    family, row, value, labels, conditions, reference_columns
                )
                low, high = bootstrap_median_ci(
                    d,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.random_state,
                    alpha=args.alpha,
                )
                rows_out.append(
                    {**common, "family": family, **row, "ci_low": low, "ci_high": high}
                )

        # Figures for the tested selection only; the other selections are in the CSV.
        if selection != test_selection:
            continue
        figure_values[variant] = value
        figure_snr_rows[variant] = snr_rows
        note = (
            f"{n_participants} participants, variant {variant}, {selection} @ "
            f"{window_desc} — "
            + (
                _READOUT_SIGN_NOTE
                if not args.no_polarity_anchor
                else "polarity anchor OFF, direction NOT interpretable"
            )
        )
        fig = plot_response_courses(
            course_by_cell[(selection, variant)],
            epoch_times,
            labels,
            conditions,
            contrast_rows,
            label=cohort.label,
            units=VARIANT_UNITS.get(variant, variant),
            selection=selection,
            reference_label=at.FULL_MASK_LABEL,
            mask_labels=at.MASK_LABELS,
            condition_colors=_CONDITION_COLORS,
            spread_mode=args.course_spread,
            test_interval=(
                None
                if args.stimulus_interval is None
                else tuple(args.stimulus_interval)
            ),
            alpha=args.alpha,
            note=note,
            save_path=out_dir / f"trial_course_by_source_{selection}_{variant}.png",
        )
        plt.close(fig)
        # One p-value summary per reference: the left panel (the contrast per source)
        # is the same in all of them, the right panel — "better than the reference?" —
        # is the whole point of having three.
        for reference, slug in at.MASK_SLUGS.items():
            fig = plot_pvalue_summary(
                value,
                labels,
                conditions,
                contrast_rows,
                versus_rows[reference],
                label=cohort.label,
                units=VARIANT_UNITS.get(variant, variant),
                reference_label=reference,
                mask_labels=at.MASK_LABELS,
                alpha=args.alpha,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.random_state,
                note=f"floor p = {floor:.5f} — {note}",
                save_path=out_dir / f"pvalue_summary_{selection}_{variant}_{slug}.png",
            )
            plt.close(fig)

    for reference, slug in at.MASK_SLUGS.items():
        fig = plot_snr_vs_reference(
            figure_values,
            {variant: rows[reference] for variant, rows in figure_snr_rows.items()},
            labels,
            conditions,
            label=cohort.label,
            units_by_variant=VARIANT_UNITS,
            reference_label=reference,
            mask_labels=at.MASK_LABELS,
            condition_colors=_CONDITION_COLORS,
            alternative=args.snr_alternative,
            alpha=args.alpha,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.random_state,
            note=f"n = {n_participants}, {test_selection} @ {window_desc}",
            save_path=out_dir / f"snr_vs_reference_{test_selection}_{slug}.png",
        )
        plt.close(fig)
    _logger.info(f"read-out figures -> {out_dir}")
    return rows_out


def _difference_for(
    family: str,
    row: dict,
    value: dict[str, np.ndarray],
    labels: list[str],
    conditions: list[str],
    reference_columns: dict[str, int],
) -> np.ndarray:
    """The per-participant differences a test row was computed from.

    Recovered rather than carried alongside the row so the CSV's interval is
    guaranteed to describe the same quantity the p-value came from.

    :param family: ``"contrast"``, ``"vs_reference"`` or ``"snr"``.
    :param row: The test record, naming its source and (for ``"snr"``) condition.
    :param value: Condition → ``(P, S)`` per-participant response.
    :param labels: Source label per column.
    :param conditions: The two conditions, in contrast order.
    :param reference_columns: Reference label → column, for the families that name one
        (every row of those carries a ``"reference"`` entry saying which).
    :return: ``(P,)`` differences.
    :raises ValueError: On an unknown *family*, or a row naming an unknown reference.
    """
    source = labels.index(row["source"])
    if family == "contrast":
        return at.condition_contrast(value, conditions, source)
    if row.get("reference") not in reference_columns:
        raise ValueError(
            f"A {family!r} row must name one of {sorted(reference_columns)}; got "
            f"{row.get('reference')!r}."
        )
    reference_index = reference_columns[row["reference"]]
    if family == "vs_reference":
        return at.discrimination_gain(value, conditions, source, reference_index)
    if family == "snr":
        condition = row["condition"]
        return value[condition][:, source] - value[condition][:, reference_index]
    raise ValueError(f"Unknown test family {family!r}.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_jica(
    music_type: MusicTypeVariants,
    *,
    args: argparse.Namespace,
    experiment_name: ExperimentNames,
    conditions: list[ConditionVariants],
    exclusion_categories: list[ExclusionCategories],
    freqs: np.ndarray,
    wavelet_dir: Path,
    subset_cache_dir: Path | None,
    save_root: Path,
    results_root: Path,
) -> list[dict]:
    """Load, decompose, draw and test one music type. Returns the test rows.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param args: Parsed CLI arguments.
    :param experiment_name: Experiment being analysed.
    :param conditions: Conditions to join, in block/segment order.
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs: Morlet frequency grid the caches were written with.
    :param wavelet_dir: Per-condition wavelet cache directory, band subdirectory
        included (``data/processed/<experiment>/wavelets/broadband``). A path without
        it misses the stage-03 caches and silently recomputes them.
    :param subset_cache_dir: Notebook-level subset cache, or ``None`` to disable.
    :param save_root: Base ``plots/`` directory.
    :param results_root: Directory the test CSV goes in.
    :return: One record per test, empty when the read-out was skipped.
    :raises ValueError: If ``--components`` names a component outside the run.
    """
    join, variant = _JOIN_CHOICES[args.join]
    cohort = _load_cohort(
        music_type,
        join=join,
        experiment_name=experiment_name,
        conditions=conditions,
        exclusion_categories=exclusion_categories,
        freqs=freqs,
        wavelet_dir=wavelet_dir,
        reuse_wavelets=args.reuse_wavelets,
        subset_cache_dir=subset_cache_dir,
        reuse_subset_cache=not args.no_reuse_subset_cache,
        n_pairs=args.n_pairs,
        n_channels=args.n_channels,
        n_times=args.n_times,
        as_float32=args.float32,
        n_jobs=args.n_jobs,
    )

    layout = assemble_join(
        cohort.raw_by_condition,
        cohort.participants,
        [c.value for c in conditions],
        join,
    )
    result = fit_joint_ica(
        layout,
        n_ica=args.n_ica,
        random_state=args.random_state,
        algorithm=args.ica_algorithm,
        fun=args.ica_fun,
        max_iter=args.ica_max_iter,
        tol=args.ica_tol,
    )
    identity_error = result.identity_error()
    _logger.info(
        f"|U A - I| = {identity_error:.2e} (tolerance {IDENTITY_TOLERANCE:.0e})"
    )
    if identity_error > IDENTITY_TOLERANCE:
        raise ValueError(
            f"components_ @ mixing_ is not the identity (max residual "
            f"{identity_error:.2e}); the unmixing is rank-deficient and the per-block "
            "filters are not the operator the fit used."
        )
    if not result.converged:
        _logger.warning(
            "FastICA did not converge, so the unmixing is wherever the solver stopped. "
            f"Lower --n_ica (now {args.n_ica}) or try --ica_algorithm deflation before "
            "reading anything below."
        )

    comp_indices = (
        list(range(result.n_components))
        if args.components is None
        else [k - 1 for k in args.components]
    )
    out_of_range = [k + 1 for k in comp_indices if not 0 <= k < result.n_components]
    if out_of_range:
        raise ValueError(
            f"--components names IC {out_of_range}, outside 1..{result.n_components}."
        )

    # The product directory is named from the ENUM, never from the data label: that
    # label carries the wavelet representation suffix and would end up in a path.
    product = f"{join.value}_{music_type.value}"
    spectrum = SpectrumTypeVariants.BROADBAND.value
    product_dir = save_root / _STAGE_DIR / product / spectrum
    if not args.skip_decomposition_plots:
        _write_decomposition_plots(
            result,
            layout,
            cohort,
            freqs=freqs,
            comp_indices=comp_indices,
            out_dir=product_dir / variant.value / f"ica_{args.n_ica}",
            onset_average=not args.skip_onset_average,
            participant_grids=args.participant_grids,
            min_trials=args.min_trials,
        )

    if args.skip_analysis:
        return []
    rows = _run_analysis(
        result,
        layout,
        cohort,
        freqs=freqs,
        args=args,
        product=product,
        out_dir=product_dir
        / JicaVariants.COMPONENT_ANALYSIS.value
        / f"ica_{args.n_ica}",
    )
    if rows:
        results_root.mkdir(parents=True, exist_ok=True)
        csv_path = results_root / (
            f"jica_tests__{args.join}__{music_type.value}__ica{args.n_ica}.csv"
        )
        frame = pd.DataFrame(rows)
        frame.to_csv(csv_path, index=False)
        _logger.info(f"saved {csv_path} ({len(frame)} test rows)")
        print(f"\n{len(frame)} tests, {cohort.label}, join {args.join}:")
        print(frame.to_string(index=False))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the selected join for every music type.

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

    # Only a variant that reads the Z-SCORED signal is REFUSED without its source: its
    # own reference rows would be the mask on z-scored data, a different quantity that
    # nothing puts back in the others' units. A raw-signal variant borrows too — the
    # invariant is stated once, in VARIANT_REFERENCE_FROM — but it recomputes the
    # identical numbers, so running it alone is legitimate and is allowed.
    for variant, source_variant in VARIANT_REFERENCE_FROM.items():
        if (
            variant in args.signal_variants
            and VARIANT_RECIPE[variant]["zscore"]
            and source_variant not in args.signal_variants
        ):
            raise ValueError(
                f"--signal_variants {variant!r} takes its fixed-reference row from "
                f"{source_variant!r}, so {source_variant!r} has to be requested too. "
                "Holding the reference fixed is what makes the comparison isolate how "
                "the component was derived."
            )

    conditions = [ConditionVariants(c) for c in args.conditions]
    if len(set(conditions)) != 2:
        raise ValueError(
            "--conditions must name exactly two distinct conditions; the join pairs "
            f"them and every test contrasts them, got {[c.value for c in conditions]}."
        )

    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    results_root = (
        Path(args.results_dir)
        if args.results_dir is not None
        else ProjectPaths.PROJECT_ROOT / "results" / experiment_name.value
    )
    # The BAND SUBDIRECTORY is part of the path, not an optional decoration: the
    # per-condition caches stage-03 writes live in `<experiment>/wavelets/broadband/`,
    # and a path one level above them is simply a cache MISS — on which the loader
    # recomputes the whole transform (tens of GB and ~40 min per condition) from the
    # SPLICED time-domain signal, which carries edge artefacts at every splice point.
    # Every other wavelet script and every stage-06/07 notebook appends it the same way.
    wavelet_dir = (
        resolve_wavelet_dir(args.wavelet_data_dir, experiment_name)
        / SpectrumTypeVariants.BROADBAND.value
    )
    subset_cache_dir = (
        resolve_notebook_wavelet_cache_dir(experiment_name)
        / SpectrumTypeVariants.BROADBAND.value
        if args.subset_cache
        else None
    )
    freqs = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )

    _logger.info(
        f"jICA: experiment={experiment_name.value}, join={args.join}, "
        f"conditions={[c.value for c in conditions]}, "
        f"music_types={[mt.value for mt in music_types]}, n_ica={args.n_ica}, "
        f"ica={args.ica_algorithm}/{args.ica_fun}"
    )
    _logger.info(f"Wavelet source cache : {wavelet_dir}")
    _logger.info(f"Wavelet subset cache : {subset_cache_dir or 'disabled'}")
    if args.n_channels is not None and not args.skip_analysis:
        _logger.warning(
            f"--n_channels {args.n_channels} with the read-out enabled: the "
            "ASSR-electrode reference is spread over the montage, so a leading channel "
            "slice may not contain it. The run will refuse if too few are present."
        )

    for music_type in music_types:
        _logger.info(f"=== {music_type.value} ===")
        run_jica(
            music_type,
            args=args,
            experiment_name=experiment_name,
            conditions=conditions,
            exclusion_categories=exclusion_categories,
            freqs=freqs,
            wavelet_dir=wavelet_dir,
            subset_cache_dir=subset_cache_dir,
            save_root=save_root,
            results_root=results_root,
        )


if __name__ == "__main__":
    main()
