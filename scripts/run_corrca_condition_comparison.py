"""
CLI script reproducing
``notebooks/06b-corrca-condition-comparison/corrca_channel_joined_separate.ipynb``:
Correlated Component Analysis (CorrCA) of wavelet power on the **joined** condition
(:attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE`), Placebo and
Psilocybin pooled on the **subject** axis, followed by the Placebo/Psilocybin
comparison of the recovered components.

The CorrCA companion of ``run_iva_condition_comparison.py`` (stage 06): the same
pooled tensor, the same figures, a different decomposition::

    subjects = recordings          # participant × condition (2P entries)
    samples  = time × frequency    # z-scored per (recording, channel, frequency)
    features = channel             # ONE filter per component, shared by all 2P

What changes with CorrCA (:mod:`src.analysis.corrca`, method in
``docs/analyses/06b-corrca-condition-comparison/``):

* **One shared filter.** Every recording is projected through the same channel
  weights, chosen to maximise the inter-recording correlation of the projected
  power. ``--n_pca`` is the dimension of the pooled within-recording covariance
  subspace those filters are sought in, which is also the component count.
* **One sign per component.** The filter is shared, so a component has the same
  polarity in every recording by construction. The one free sign is fixed from the
  **shared** pattern against the ASSR electrode mask — never per recording, which
  would invert part of a component the model fitted as one.
* **Topographies.** Each recording still gets its own forward pattern
  ``R_ss W (Wᵀ R_ss W)⁻¹`` — the shared component seen through that recording's
  covariance — which is what the per-condition topography grids contrast. The
  model's own group topography is drawn separately.
* **Synchrony per condition.** The read-out IVA has no counterpart for: each
  component's ISC within Placebo and within Psilocybin, from the same filters, plus
  every participant's leave-one-out ISC paired across conditions and a Wilcoxon
  signed-rank test per component (uncorrected).

The fit pools both conditions, so the filters are symmetric in them. Its
between-recording term includes each participant's own Placebo-Psilocybin pair.

**Filters fitted on one condition** (``--fit_condition Placebo``, variant
``corrca_channel_placebo_fit``; ``--fit_condition Psilocybin``, variant
``corrca_channel_psilocybin_fit``). The same join, loaded and drawn the same way, but
CorrCA is fitted on that condition's recordings alone and every recording of both
conditions is then projected through those filters. The other condition's recordings
never shape the filters, so their ISC is an out-of-sample read-out of the fitted
condition's components, while the fitted condition's ISC stays in-sample
(optimistic). The fit's own ISC and shared patterns are the fitted condition's, and
the leave-one-participant-out influence and the leave-subset-out refits are refits on
that condition alone, read on its recordings. The figures go under
``JoinedPlaceboFitted_<MusicType>/`` or ``JoinedPsilocybinFitted_<MusicType>/``; the
store sits beside the pooled fit's under ``JoinedSeparate/``, as its own variant.

Figures (panels labelled ``CC <k>``):

  - ``condition_mean_tf_maps`` /
    ``condition_mean_tf_maps_onset`` — rows = conditions plus their difference,
    columns = components (drawn by :mod:`src.visualization.iva_condition_plots`)
  - ``participants/*`` — per component, Placebo row above Psilocybin row, one column
    per participant
  - ``shared_spatial_topomap_filter`` — the shared spatial filter ``W`` itself, the
    channel weights that, applied to the z-scored power, give the TF maps
  - ``corrca_condition_isc`` — per-component synchrony per condition, paired
  - ``corrca_condition_isc.csv`` / ``corrca_participant_isc.csv`` /
    ``corrca_condition_isc_test.csv`` — the numbers behind it
  - ``corrca_onset_condition_isc`` (+ its three CSVs) — the same synchrony scored on
    the onset-locked, trial-normalised maps instead of the whole recording
  - ``influence/corrca_influence`` / ``influence/corrca_heldout_isc`` (+
    ``influence/corrca_influence.csv``) — the leave-one-participant-out check: one refit per participant, the ISC and topography
    each component loses without them, and every recording's in-sample, held-out and
    own-pair ISC (figures for CC 1-3); ``influence/corrca_onset_*`` the same on the
    onset-locked, trial-averaged maps. With ``--store_components`` and two conditions
    the refits leaving out every subset of ``--refit_subset_sizes`` participants are
    stored beside the store, for the SNR grid's robustness rows.
    ``--skip_influence`` drops all of it.
  - ``roi_masked_condition_mean_tf_maps`` (+ ``_onset``, + ``participants/roi_masked_*``)
    — the same TF maps with the fitted filter restricted to the ASSR electrodes (every
    other weight zeroed) on the z-scored signal: what the component carries from the
    ROI alone
  - ``non_roi_masked_*`` — the same with the ASSR electrodes zeroed instead: what the
    rest of the head carries. The two add up to the full maps. Only with
    ``--roi_masked``.
  - ``swapped_filter_*`` TF maps (the raw <-> log twin's filters on this fit's input)
    are drawn into this directory by ``run_corrca_snr_grid.py``, after every fit is
    stored, not here.

Only the full montage on raw power is fitted by default. ``--roi_fit`` and
``--non_roi_fit`` add fits from the same loaded tensor, the channel axis cut
BEFORE the z-scoring: on the binary ASSR ROI alone (ROI → z-score → PCA → CorrCA),
stored and drawn as ``corrca_channel_joined_separate_roi``, then on every other electrode,
as ``corrca_channel_joined_separate_non_roi``. A cut is skipped, with a warning, when the
montage holds no ASSR electrode or fewer than ``--n_pca`` kept channels;
``--log_power`` adds the log-power twin of every selected fit. The non-ROI fit has no
anchor electrode, so it keeps the fit's own orientation.

With ``--skip_stored --store_components`` a fit whose store exists is read back
instead of refit and every figure is drawn from it, so a rerun redraws without
recomputing; a store fitted on other data (cohort, channels, time extent) is refused.

Output::

    plots/06b-corrca-condition-comparison/JoinedSeparate_<MusicType>/
        broadband/corrca_channel_joined_separate/pca_<n_pca>/
        bands/corrca_channel_joined_separate/pca_<n_pca>/      # --band / --band_range
        broadband/corrca_channel_joined_separate_roi/pca_<n_pca>/  # the ROI fit
        broadband/corrca_channel_joined_separate_non_roi/pca_<n_pca>/  # the rest
    plots/06b-corrca-condition-comparison/JoinedPlaceboFitted_<MusicType>/
        broadband/corrca_channel_placebo_fit*/pca_<n_pca>/   # --fit_condition Placebo
    plots/06b-corrca-condition-comparison/JoinedPsilocybinFitted_<MusicType>/
        broadband/corrca_channel_psilocybin_fit*/pca_<n_pca>/  # --fit_condition Psilocybin

Examples::

    python scripts/run_corrca_condition_comparison.py \\
        --experiment assr --n_pca 10 --reuse_wavelets

    python scripts/run_corrca_condition_comparison.py \\
        --experiment assr --n_pca 10 --n_channels 32 --n_times 3000 \\
        --n_pairs 5 --lenient_mask --reuse_wavelets

    python scripts/run_corrca_condition_comparison.py \\
        --experiment assr --n_pca 5 --fit_condition Placebo --reuse_wavelets

``--store_components`` keeps the oriented TF maps and per-recording topographies, plus
the filters, shared patterns and ISCs, under
``data/processed/<experiment>/iva_results/JoinedSeparate/`` (the shared component
store, variant ``corrca_channel_joined_separate``); read them back with
:func:`src.io.iva_store.load_iva_components`.
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
    CORRCA_MASKED_READINGS,
    CORRCA_FIT_CONDITION_LABELS,
    CORRCA_SEPARATE_FIT_RUNS,
    CorrcaFit,
    drawn_components,
    add_corrca_condition_args,
    anchor_corrca_signs,
    corrca_plots_dir,
    corrca_refit_path,
    corrca_store_extras,
    log_corrca_conditions,
    resolve_wavelet_dir,
    project_corrca_sources,
    roi_masked_corrca_sources,
    run_corrca_fits,
    stored_corrca_fit,
    write_condition_isc,
    write_corrca_condition_report,
    write_corrca_influence,
)
from scripts.notebook_helpers import load_joined_condition_wavelets  # noqa: E402

# The stimulus average is the stage-06 one, so both stages epoch on one window.
from scripts.run_iva_condition_comparison import _onset_average  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.analysis.corrca import (  # noqa: E402
    condition_isc,
    decompose_channel_corrca,
    paired_condition_test,
)
from src.analysis.iva_condition_comparison import slice_to_band  # noqa: E402
from src.analysis.wavelet_ica import log10_power  # noqa: E402
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    CoordinateSystems,
    ExclusionCategories,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
)
from src.definitions.frequency import band_token  # noqa: E402
from src.io.iva_store import iva_results_path, save_iva_components  # noqa: E402
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_tf_maps,
    plot_participant_condition_tf_maps,
    plot_participant_condition_topomaps,
)
from src.visualization.iva_quality_plots import topo_info_subset  # noqa: E402

_logger = logging.getLogger(__name__)

#: Store variant of the pooled fit, and the analysis-type subdirectory named after
#: it; ``--fit_condition`` picks another run of the same join
#: (:data:`~scripts.analysis_common.CORRCA_SEPARATE_FIT_RUNS`).
_VARIANT = IvaVariants.CORRCA_CHANNEL_JOINED_SEPARATE
_ANALYSIS_DIR = _VARIANT.value

#: Frequency (Hz) marked on every TF panel — the ASSR stimulation frequency.
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CorrCA on the joined condition (Placebo + Psilocybin pooled on the "
            "subject axis) and draw the per-component condition comparison, "
            "including each component's synchrony per condition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_corrca_condition_args(parser)
    parser.add_argument(
        "--fit_condition",
        choices=[c.value for c in CORRCA_SEPARATE_FIT_RUNS if c is not None],
        default=None,
        help=(
            "Fit the CorrCA filters on this condition's recordings alone, then project "
            "every recording of both conditions through them (the "
            "'corrca_channel_placebo_fit' / 'corrca_channel_psilocybin_fit' run). "
            "Default: fit on every recording."
        ),
    )
    return parser


def _row_view(data: np.ndarray, rows: list[int]) -> np.ndarray:
    """*data*'s *rows*: a view when they are contiguous, as the join stacks the
    conditions in blocks, else a copy."""
    if rows == list(range(rows[0], rows[-1] + 1)):
        return data[rows[0] : rows[-1] + 1]
    return data[rows]


def run_corrca_condition_comparison(
    music_type: MusicTypeVariants, args: argparse.Namespace
) -> None:
    """Fit, orient, score and draw the subject-axis join for one music type.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    """
    experiment_name = ExperimentNames(args.experiment)
    conditions = [ConditionVariants(c) for c in args.conditions]
    fit_condition = (
        None if args.fit_condition is None else ConditionVariants(args.fit_condition)
    )
    run_variant = CORRCA_SEPARATE_FIT_RUNS[fit_condition]
    run_label = f"{ConditionVariants.JOINED_SEPARATE.value}_{music_type.value}"
    if fit_condition is not None:
        run_label = f"{CORRCA_FIT_CONDITION_LABELS[fit_condition]}_{music_type.value}"
    band = tuple(args.band_range) if args.band_range is not None else args.band
    band_name = band_token(band)
    freqs_full = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )
    pooled, condition_analyzers = load_joined_condition_wavelets(
        music_type,
        [ExclusionCategories.BAD_MUSIC, ExclusionCategories.ARTIFACTS],
        freqs_full,
        wavelet_dir=resolve_wavelet_dir(args.wavelet_data_dir, experiment_name),
        experiment_name=experiment_name,
        conditions=conditions,
        n_channels=args.n_channels,
        n_times=args.n_times,
        reuse_wavelets=args.reuse_wavelets,
        n_jobs=args.n_jobs,
    )
    if args.n_pairs is not None:
        keep = sorted(set(pooled.participants))[: args.n_pairs]
        pooled = pooled.select_participants(keep)
        _logger.info(f"Restricted to {len(keep)} participant(s): {keep}")

    sfreq = pooled.data.sfreq
    data_4d, freqs = slice_to_band(pooled.data.data, freqs_full, band)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    topo_info = topo_info_subset(
        condition_analyzers[pooled.conditions[0]].info, data_4d.shape[1]
    )
    condition_rows = [c.value for c in pooled.conditions]
    subject_participants = list(pooled.participants)
    subject_conditions = [c.value for c in pooled.subject_conditions]
    # The recordings the filters are fitted on; None = every one.
    fit_rows = (
        None
        if fit_condition is None
        else [i for i, c in enumerate(subject_conditions) if c == fit_condition.value]
    )

    # Full montage, on raw then log power, then the ASSR ROI and the rest of the head,
    # raw then log — one load, every fit stored and drawn as its own variant. The fit
    # z-scores each recording itself, so preparing is only the optional log.
    def _run_fit(fit: CorrcaFit) -> None:
        variant, data_4d, topo_info = fit.variant, fit.data, fit.info
        label = f"{run_label}_log" if fit.log_power else run_label
        _logger.info(f"[{label}] === {variant.value} ===")
        _n_rec, n_ch, _n_freqs, n_t = data_4d.shape
        _logger.info(
            f"[{label}] pooled {data_4d.shape} (recordings × channels × freqs × times), "
            f"{pooled.n_pairs} participant(s) × {len(pooled.conditions)} condition(s)"
        )

        channel_names = list(topo_info.ch_names)
        reused = stored_corrca_fit(
            args,
            ConditionVariants.JOINED_SEPARATE,
            variant,
            music_type,
            band_name,
            participants=subject_participants,
            subject_conditions=subject_conditions,
            channel_names=channel_names,
            n_times=n_t,
            label=label,
            logger=_logger,
        )
        if reused is None:
            corrca = decompose_channel_corrca(
                data_4d,
                label=label,
                n_pca=args.n_pca,
                gamma=args.gamma,
                fit_rows=fit_rows,
            )
            corrca, alignment_note, extras = anchor_corrca_signs(
                corrca,
                channel_names,
                coordinate_system,
                lenient_mask=args.lenient_mask,
                label=label,
                logger=_logger,
                expect_anchor=fit.channels != "non_roi",
            )
            if fit_condition is not None:
                extras["corrca_fit_condition"] = np.asarray(fit_condition.value)
        else:
            corrca, alignment_note = reused
        sources, patterns = corrca.sources, corrca.patterns

        rows_of = {
            c: [i for i, rc in enumerate(subject_conditions) if rc == c]
            for c in condition_rows
        }
        group_isc, participant_isc = condition_isc(
            {c: sources[rows] for c, rows in rows_of.items()},
            {c: [subject_participants[i] for i in rows] for c, rows in rows_of.items()},
        )
        tests = (
            paired_condition_test(participant_isc, condition_rows)
            if len(condition_rows) == 2
            else None
        )
        log_corrca_conditions(group_isc, tests, label=label, logger=_logger)

        if args.store_components and reused is None:
            stored_onsets: dict[str, np.ndarray] = {}
            for condition in pooled.conditions:
                onsets_c = pooled.condition_onsets(condition)
                if onsets_c is not None:
                    stored_onsets[f"stimulus_onsets_{condition.value}"] = onsets_c[
                        onsets_c < n_t
                    ].astype(np.int64)
            save_iva_components(
                experiment=experiment_name,
                condition=ConditionVariants.JOINED_SEPARATE,
                variant=variant,
                music_type=music_type,
                band=band_name,
                n_pca=args.n_pca,
                sfreq=sfreq,
                participants=subject_participants,
                subject_conditions=subject_conditions,
                arrays={
                    IvaComponentArrays.TF_MAP: sources,
                    IvaComponentArrays.CHANNEL_PATTERN: patterns,
                },
                freqs=freqs,
                times=np.arange(n_t) / sfreq,
                channel_names=channel_names,
                extras={
                    **extras,
                    **corrca_store_extras(corrca.model, group_isc, condition_rows),
                    **stored_onsets,
                },
                dtype=args.store_dtype,
                processed_data_dir=(
                    Path(args.store_dir)
                    if args.store_dir is not None
                    else ProjectPaths.PROCESSED_DATA_DIR
                ),
                logger=_logger,
            )

        n_components = corrca.model.n_components
        # Every figure draws at most MAX_DRAWN_COMPONENTS; the tables keep all.
        comp_indices = drawn_components(n_components, args.components, logger=_logger)

        out_dir = corrca_plots_dir(
            variant,
            ConditionVariants.JOINED_SEPARATE,
            music_type,
            band=band_name,
            n_pca=args.n_pca,
            root=args.save_dir,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        participants_dir = out_dir / "participants"
        prefix = "" if band_name is None else f"{band_name}_"
        show_difference = not args.no_difference_row
        times = np.arange(n_t) / sfreq
        onsets = pooled.condition_onsets(pooled.conditions[0])
        onset_times = np.array([]) if onsets is None else onsets[onsets < n_t] / sfreq
        shared = {
            "label": label,
            "alignment_note": alignment_note,
            "component_prefix": COMPONENT_PREFIX,
        }
        _logger.info(f"[{label}] writing figures to {out_dir}")

        written = plot_participant_condition_topomaps(
            patterns,
            subject_participants,
            subject_conditions,
            condition_rows,
            topo_info,
            n_ch,
            comp_indices,
            root_dir=participants_dir,
            prefix=prefix,
            **shared,
        )

        def _draw_tf_maps(
            tf_sources: np.ndarray,
            tf_prefix: str,
            tf_shared: dict,
        ) -> list:
            """The whole-recording and stimulus-averaged TF grids, group and participant."""
            plot_condition_mean_tf_maps(
                tf_sources,
                subject_participants,
                subject_conditions,
                condition_rows,
                freqs,
                times,
                comp_indices,
                show_difference=show_difference,
                time_marks=onset_times,
                freq_marks=_TF_FREQ_MARKS,
                save_path=out_dir / f"{tf_prefix}condition_mean_tf_maps.png",
                **tf_shared,
            )
            plt.close("all")
            drawn = plot_participant_condition_tf_maps(
                tf_sources,
                subject_participants,
                subject_conditions,
                condition_rows,
                freqs,
                times,
                comp_indices,
                root_dir=participants_dir,
                prefix=tf_prefix,
                time_marks=onset_times,
                freq_marks=_TF_FREQ_MARKS,
                **tf_shared,
            )
            averaged = (
                None
                if args.skip_onset_average
                else _onset_average(
                    tf_sources,
                    onsets,
                    label=label,
                    n_times=n_t,
                    sfreq=sfreq,
                    min_onsets=args.min_onsets,
                )
            )
            if averaged is not None:
                onset_tf, epoch_times, epoch_marks = averaged
                plot_condition_mean_tf_maps(
                    onset_tf,
                    subject_participants,
                    subject_conditions,
                    condition_rows,
                    freqs,
                    epoch_times,
                    comp_indices,
                    show_difference=show_difference,
                    freq_marks=_TF_FREQ_MARKS,
                    epoch_marks=epoch_marks,
                    save_path=out_dir / f"{tf_prefix}condition_mean_tf_maps_onset.png",
                    equalize=False,
                    **tf_shared,
                )
                plt.close("all")
                drawn += plot_participant_condition_tf_maps(
                    onset_tf,
                    subject_participants,
                    subject_conditions,
                    condition_rows,
                    freqs,
                    epoch_times,
                    comp_indices,
                    root_dir=participants_dir,
                    prefix=f"{tf_prefix}onset_",
                    equalize=False,
                    freq_marks=_TF_FREQ_MARKS,
                    epoch_marks=epoch_marks,
                    **tf_shared,
                )
                # Synchrony of the onset-locked maps: the evoked TF response's shape,
                # baseline included, rather than the whole-recording time course.
                write_condition_isc(
                    {c: onset_tf[rows] for c, rows in rows_of.items()},
                    {
                        c: [subject_participants[i] for i in rows]
                        for c, rows in rows_of.items()
                    },
                    comp_indices,
                    stem="corrca_onset",
                    component_prefix=COMPONENT_PREFIX,
                    label=f"{label}, onset-locked TF maps",
                    out_dir=out_dir,
                    prefix=tf_prefix,
                    logger=_logger,
                )
            return drawn

        written += _draw_tf_maps(sources, prefix, shared)
        # The full fit read through the ROI alone, then through the rest of the head. A
        # cut fit has only one of the two electrode sets, so it has no masked reading.
        masked_readings = (
            {}
            if fit.channels != "full" or not args.roi_masked
            else CORRCA_MASKED_READINGS
        )
        for reading, note in masked_readings.items():
            masked_sources = roi_masked_corrca_sources(
                data_4d,
                corrca.model,
                channel_names,
                coordinate_system,
                lenient_mask=args.lenient_mask,
                label=label,
                logger=_logger,
                complement=reading == "non_roi_masked",
            )
            if masked_sources is not None:
                written += _draw_tf_maps(
                    masked_sources,
                    f"{prefix}{reading}_",
                    {**shared, "alignment_note": f"{note}; {alignment_note}"},
                )
            del masked_sources

        write_corrca_condition_report(
            corrca.model,
            group_isc,
            participant_isc,
            tests,
            condition_rows,
            topo_info,
            n_ch,
            comp_indices,
            label=label,
            alignment_note=alignment_note,
            out_dir=out_dir,
            prefix=prefix,
        )
        if not args.skip_influence:
            # The refits leave participants out of the data the filters were fitted
            # on, so a condition-fitted run refits, and reads, that condition alone.
            if fit_rows is None:
                influence_data = data_4d
                influence_participants = subject_participants
                influence_conditions = subject_conditions
            else:
                influence_data = _row_view(data_4d, fit_rows)
                influence_participants = [subject_participants[i] for i in fit_rows]
                influence_conditions = [subject_conditions[i] for i in fit_rows]

            def _onset_maps(filters: np.ndarray):
                # The onset average of the TF maps above, through any filters.
                if args.skip_onset_average:
                    return None
                averaged = _onset_average(
                    project_corrca_sources(influence_data, filters, dtype=np.float32),
                    onsets,
                    label=label,
                    n_times=n_t,
                    sfreq=sfreq,
                    min_onsets=args.min_onsets,
                )
                if averaged is None:
                    return None
                return averaged[0], influence_participants, influence_conditions

            # condition_rows, not the fit's one condition: the refits are written
            # for the SNR grid's contrast, which reads both conditions through them.
            write_corrca_influence(
                influence_data,
                corrca.model,
                influence_participants,
                influence_conditions,
                condition_rows,
                onset_maps=_onset_maps,
                n_pca=args.n_pca,
                label=label,
                out_dir=out_dir,
                prefix=prefix,
                logger=_logger,
                refit_path=(
                    corrca_refit_path(
                        iva_results_path(
                            experiment_name,
                            ConditionVariants.JOINED_SEPARATE,
                            variant,
                            music_type,
                            band_name,
                            args.n_pca,
                            processed_data_dir=(
                                None if args.store_dir is None else Path(args.store_dir)
                            ),
                        )
                    )
                    if args.store_components
                    else None
                ),
                refit_subset_sizes=args.refit_subset_sizes,
            )
        _logger.info(
            f"[{label}] done: {len(written)} per-participant figure(s), condition grids "
            f"and the synchrony report in {out_dir}"
        )

    run_corrca_fits(
        data_4d,
        topo_info,
        coordinate_system,
        run_variant,
        args,
        prepare=lambda data, log: log10_power(data) if log else data,
        run_fit=_run_fit,
        label=run_label,
        logger=_logger,
    )


def run(args: argparse.Namespace) -> None:
    """Run the analysis for an already-parsed argument namespace.

    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    """
    experiment_name = ExperimentNames(args.experiment)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]
    if len(set(args.conditions)) < 2:
        raise ValueError(
            "--conditions must name at least two distinct conditions; a joined run "
            f"pools them, got {args.conditions}."
        )
    if args.fit_condition is not None and args.fit_condition not in args.conditions:
        raise ValueError(
            f"--fit_condition {args.fit_condition} is not among --conditions "
            f"{args.conditions}; the filters are fitted on recordings of the join."
        )
    _logger.info(
        f"CorrCA condition comparison: experiment={experiment_name.value}, "
        f"conditions={args.conditions}, music_types={[m.value for m in music_types]}, "
        f"n_pca={args.n_pca}, gamma={args.gamma}, filters fitted on "
        f"{args.fit_condition or 'every recording'}"
    )
    for music_type in music_types:
        _logger.info(f"=== {music_type.value} ===")
        run_corrca_condition_comparison(music_type, args)


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
