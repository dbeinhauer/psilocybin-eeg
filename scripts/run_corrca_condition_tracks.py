"""
CLI script reproducing
``notebooks/06b-corrca-condition-comparison/corrca_channel_joined_tracks.ipynb``:
Correlated Component Analysis (CorrCA) of wavelet power on the **time-concatenated**
join (:attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS`), followed by the
Placebo/Psilocybin comparison of the recovered components.

The CorrCA companion of ``run_iva_condition_tracks.py`` (stage 06) and the sibling of
``run_corrca_condition_comparison.py``; the two differ only in which axis the
conditions are pooled along::

    subject k  ->  [ ---- Placebo track ---- | ---- Psilocybin track ---- ]

    subjects = participants        # P entries
    samples  = time × frequency    # time spans BOTH conditions
    features = channel             # ONE filter per component, shared by everyone

**What this buys.** The filters are fitted on both conditions at once and are the
same in both, so a component *is* the same spatial combination under Placebo and
Psilocybin by construction. Splitting the projected sources back into the two
segments then gives the cleanest inter-condition contrast CorrCA offers: each
component's ISC within each condition, measured through one fixed filter, with every
participant's leave-one-out ISC paired across conditions and a Wilcoxon signed-rank
test per component (uncorrected).

**What it costs.** The per-participant topography is estimated over the whole
concatenated recording, so it is shared by the conditions and drawn as a single row,
exactly as in the stage-06 tracks run.

**Z-scoring happens before the concatenation** (``--zscore_mode``, default
``per_condition``): each condition is standardised on its own, so an overall power
difference between the conditions is normalised away and what is compared is
temporal and spectral structure. ``--zscore_mode joint`` keeps the amplitude contrast.

The sign is fixed once per component from the shared pattern against the ASSR
electrode mask (see ``run_corrca_condition_comparison.py``); the method is documented
in ``docs/analyses/06b-corrca-condition-comparison/``.

Figures (panels labelled ``CC <k>``):

  - ``shared_mean_topomaps`` — per-participant topographies, one row (shared)
  - ``condition_mean_tf_maps`` / ``condition_mean_tf_maps_onset`` — rows =
    conditions plus their difference, on the split TF maps
  - ``participants/*`` — per component, one column per participant
  - ``corrca_shared_topomaps`` — the model's shared forward pattern per component
  - ``corrca_condition_isc`` and the three ``corrca_*.csv`` tables — synchrony per
    condition
  - ``roi_masked_condition_mean_tf_maps`` (+ ``_onset``, + ``participants/roi_masked_*``)
    — the same TF maps with the fitted filter restricted to the ASSR electrodes (every
    other weight zeroed) on the z-scored signal
  - ``non_roi_masked_*`` — the same with the ASSR electrodes zeroed instead; the two
    add up to the full maps. ``--skip_roi_masked`` drops both.

Every run then fits twice more from the same loaded tensor, the channel axis cut
BEFORE the z-scoring: on the binary ASSR ROI alone (ROI → z-score → PCA → CorrCA),
stored and drawn as ``corrca_channel_joined_tracks_roi``, then on every other electrode,
as ``corrca_channel_joined_tracks_non_roi``. A cut is skipped, with a warning, when the
montage holds no ASSR electrode or fewer than ``--n_pca`` kept channels;
``--skip_roi_fit`` / ``--skip_non_roi_fit`` skip it outright. The non-ROI fit has no
anchor electrode, so it keeps the fit's own orientation.

With ``--skip_stored --store_components`` a fit whose store exists is read back
instead of refit and every figure is drawn from it, so a rerun redraws without
recomputing; a store fitted on other data (cohort, channels, time extent) is refused.

Output::

    plots/06b-corrca-condition-comparison/JoinedTracks_<MusicType>/
        broadband/corrca_channel_joined_tracks/pca_<n_pca>/
        bands/corrca_channel_joined_tracks/pca_<n_pca>/      # --band / --band_range
        broadband/corrca_channel_joined_tracks_roi/pca_<n_pca>/  # the ROI fit
        broadband/corrca_channel_joined_tracks_non_roi/pca_<n_pca>/  # the rest

``--n_times`` is applied per condition, so the concatenated sample axis is
``n_freqs × 2 × n_times``. ``--store_components`` keeps the components **unsplit**,
with the segment order and lengths, under
``data/processed/<experiment>/iva_results/JoinedTracks/`` (variant
``corrca_channel_joined_tracks``).

Examples::

    python scripts/run_corrca_condition_tracks.py \\
        --experiment assr --n_pca 10 --reuse_wavelets

    python scripts/run_corrca_condition_tracks.py \\
        --experiment assr --n_pca 10 --n_channels 32 --n_times 3000 \\
        --n_pairs 5 --lenient_mask --reuse_wavelets
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    CORRCA_MASKED_READINGS,
    CorrcaFit,
    add_corrca_condition_args,
    anchor_corrca_signs,
    corrca_plots_dir,
    standardise_joined_tracks,
    corrca_store_extras,
    log_corrca_conditions,
    resolve_wavelet_dir,
    roi_masked_corrca_sources,
    run_corrca_fits,
    stored_corrca_fit,
    write_corrca_condition_report,
)
from scripts.notebook_helpers import load_paired_condition_wavelets  # noqa: E402

# The per-condition stimulus average is the stage-06 one, so both stages epoch alike.
from scripts.run_iva_condition_tracks import (  # noqa: E402
    _onset_average_per_condition,
)
from src.analysis import iva_quality  # noqa: E402
from src.analysis.condition_tracks import ZSCORE_MODES  # noqa: E402
from src.analysis.corrca import (  # noqa: E402
    condition_isc,
    decompose_channel_corrca,
    paired_condition_test,
)
from src.analysis.iva_condition_comparison import (  # noqa: E402
    slice_to_band,
    stack_conditions_on_subject_axis,
)
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
from src.io.iva_store import save_iva_components  # noqa: E402
from src.visualization.corrca_plots import COMPONENT_PREFIX  # noqa: E402
from src.visualization.iva_condition_plots import (  # noqa: E402
    plot_condition_mean_tf_maps,
    plot_condition_mean_topomaps,
    plot_participant_condition_tf_maps,
    plot_participant_condition_topomaps,
)
from src.visualization.iva_quality_plots import topo_info_subset  # noqa: E402

_logger = logging.getLogger(__name__)

#: Store variant, and the analysis-type subdirectory named after it.
_VARIANT = IvaVariants.CORRCA_CHANNEL_JOINED_TRACKS
_ANALYSIS_DIR = _VARIANT.value

#: Frequency (Hz) marked on every TF panel — the ASSR stimulation frequency.
_TF_FREQ_MARKS = [iva_quality.ASSR_FREQ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CorrCA on the time-concatenated join (each participant's Placebo "
            "track followed by their Psilocybin one) and draw the per-component "
            "condition comparison, including each component's synchrony per condition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_corrca_condition_args(parser)
    parser.add_argument(
        "--zscore_mode",
        choices=list(ZSCORE_MODES),
        default="per_condition",
        help=(
            "How the tracks are standardised before being concatenated: "
            "'per_condition' (an overall power difference is normalised away), "
            "'joint' (kept), or 'none' (already standardised)."
        ),
    )
    return parser


def run_corrca_condition_tracks(
    music_type: MusicTypeVariants,
    args: argparse.Namespace,
    *,
    conditions: Sequence[ConditionVariants] | None = None,
    full: IvaVariants = _VARIANT,
) -> None:
    """Fit, orient, score and draw the time-axis join for one music type.

    A single condition is the join reduced to one track, which is what
    ``run_corrca_individual_conditions.py`` fits: stored under that condition rather
    than ``JoinedTracks``, with no between-condition test.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    :param conditions: Conditions to lay end to end; ``None`` reads ``--conditions``.
    :param full: The run's raw-power full fit; its log-power, ROI and non-ROI fits
        follow (:func:`~scripts.analysis_common.corrca_run_plan`).
    """
    experiment_name = ExperimentNames(args.experiment)
    if conditions is None:
        conditions = [ConditionVariants(c) for c in args.conditions]
    store_condition = (
        ConditionVariants.JOINED_TRACKS if len(conditions) > 1 else conditions[0]
    )
    run_label = f"{store_condition.value}_{music_type.value}"
    band = tuple(args.band_range) if args.band_range is not None else args.band
    band_name = band_token(band)
    freqs_full = np.linspace(
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )
    paired, condition_analyzers = load_paired_condition_wavelets(
        music_type,
        [ExclusionCategories.BAD_MUSIC, ExclusionCategories.ARTIFACTS],
        freqs_full,
        wavelet_dir=resolve_wavelet_dir(args.wavelet_data_dir, experiment_name),
        experiment_name=experiment_name,
        conditions=conditions,
        # Joined raw: each fit standardises its own input (after the log, for the
        # log-power fits), exactly as --zscore_mode would have at the join.
        zscore_mode="none",
        n_channels=args.n_channels,
        n_times=args.n_times,
        reuse_wavelets=args.reuse_wavelets,
        n_jobs=args.n_jobs,
    )
    if args.n_pairs is not None and args.n_pairs < paired.n_pairs:
        keep = sorted(paired.participants)[: args.n_pairs]
        rows = [paired.participants.index(p) for p in keep]
        paired = dataclasses.replace(
            paired,
            data=dataclasses.replace(paired.data, data=paired.data.data[rows]),
            participants=tuple(keep),
        )
        _logger.info(f"Restricted to {len(keep)} participant(s): {keep}")

    sfreq = paired.data.sfreq
    data_4d, freqs = slice_to_band(paired.data.data, freqs_full, band)
    coordinate_system = CoordinateSystems(args.coordinate_system)
    topo_info = topo_info_subset(
        condition_analyzers[paired.conditions[0]].info, data_4d.shape[1]
    )
    condition_rows = [c.value for c in paired.conditions]
    participants = list(paired.participants)
    boundaries = paired.boundaries

    def _prepare(data: np.ndarray, log: bool) -> np.ndarray:
        return standardise_joined_tracks(
            log10_power(data) if log else data, boundaries, args.zscore_mode
        )

    # Full montage, on raw then log power, then the ASSR ROI and the rest of the head,
    # raw then log — one load, every fit stored and drawn as its own variant.
    def _run_fit(fit: CorrcaFit) -> None:
        variant, data_4d, topo_info = fit.variant, fit.data, fit.info
        label = f"{run_label}_log" if fit.log_power else run_label
        _logger.info(f"[{label}] === {variant.value} ===")
        _n_subj, n_ch, _n_freqs, n_total = data_4d.shape
        _logger.info(
            f"[{label}] concatenated {data_4d.shape} (participants × channels × freqs × "
            f"times), segments {paired.segment_lengths}, zscore_mode={args.zscore_mode}"
        )

        channel_names = list(topo_info.ch_names)
        reused = stored_corrca_fit(
            args,
            store_condition,
            variant,
            music_type,
            band_name,
            participants=participants,
            subject_conditions=[store_condition.value] * len(participants),
            channel_names=channel_names,
            n_times=n_total,
            label=label,
            logger=_logger,
        )
        if reused is None:
            corrca = decompose_channel_corrca(
                data_4d, label=label, n_pca=args.n_pca, gamma=args.gamma
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
        else:
            corrca, alignment_note = reused
        sources, patterns = corrca.sources, corrca.patterns

        # Split the projected sources back into the conditions — the filters are shared,
        # so each segment is the same component seen under one condition.
        sources_by_condition = {
            c.value: paired.condition_track(sources, c) for c in paired.conditions
        }
        group_isc, participant_isc = condition_isc(
            sources_by_condition, {c: participants for c in condition_rows}
        )
        tests = (
            paired_condition_test(participant_isc, condition_rows)
            if len(condition_rows) == 2
            else None
        )
        log_corrca_conditions(group_isc, tests, label=label, logger=_logger)

        if args.store_components and reused is None:
            stored_onsets: dict[str, np.ndarray] = {}
            for condition, length in zip(paired.conditions, paired.segment_lengths):
                onsets_c = paired.condition_onsets(condition, absolute=False)
                if onsets_c is not None:
                    stored_onsets[f"stimulus_onsets_{condition.value}"] = onsets_c[
                        onsets_c < length
                    ].astype(np.int64)
            save_iva_components(
                experiment=experiment_name,
                condition=store_condition,
                variant=variant,
                music_type=music_type,
                band=band_name,
                n_pca=args.n_pca,
                sfreq=sfreq,
                participants=participants,
                arrays={
                    IvaComponentArrays.TF_MAP: sources,
                    IvaComponentArrays.CHANNEL_PATTERN: patterns,
                },
                freqs=freqs,
                times=np.arange(n_total) / sfreq,
                channel_names=channel_names,
                segment_conditions=condition_rows,
                segment_lengths=list(paired.segment_lengths),
                extras={
                    "zscore_mode": np.asarray(args.zscore_mode),
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
        comp_indices = (
            list(range(n_components))
            if args.components is None
            else [k - 1 for k in args.components]
        )
        out_of_range = [k + 1 for k in comp_indices if not 0 <= k < n_components]
        if out_of_range:
            raise ValueError(
                f"--components {out_of_range} are outside 1..{n_components}."
            )

        # The length stack_conditions_on_subject_axis trims every condition to.
        n_split = min(track.shape[-1] for track in sources_by_condition.values())
        condition_onsets = {
            c.value: paired.condition_onsets(c, absolute=False)
            for c in paired.conditions
        }
        times = np.arange(n_split) / sfreq
        ref_onsets = paired.condition_onsets(paired.conditions[0], absolute=False)
        onset_times = (
            np.array([])
            if ref_onsets is None
            else ref_onsets[ref_onsets < n_split] / sfreq
        )

        out_dir = corrca_plots_dir(
            variant,
            store_condition,
            music_type,
            band=band_name,
            n_pca=args.n_pca,
            root=args.save_dir,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        participants_dir = out_dir / "participants"
        prefix = "" if band_name is None else f"{band_name}_"
        show_difference = not args.no_difference_row
        shared = {
            "label": label,
            "alignment_note": alignment_note,
            "component_prefix": COMPONENT_PREFIX,
        }
        _logger.info(f"[{label}] writing figures to {out_dir}")

        # The per-participant topographies span both tracks: ONE row, not a fabricated
        # comparison. A single track is simply that condition's row.
        if len(condition_rows) > 1:
            shared_row = " + ".join(condition_rows) + " (shared)"
            topomaps_name = "shared_mean_topomaps.png"
        else:
            shared_row = condition_rows[0]
            topomaps_name = "condition_mean_topomaps.png"
        plot_condition_mean_topomaps(
            patterns,
            participants,
            [shared_row] * len(participants),
            [shared_row],
            topo_info,
            n_ch,
            comp_indices,
            save_path=out_dir / f"{prefix}{topomaps_name}",
            **shared,
        )
        plt.close("all")
        written = plot_participant_condition_topomaps(
            patterns,
            participants,
            [shared_row] * len(participants),
            [shared_row],
            topo_info,
            n_ch,
            comp_indices,
            root_dir=participants_dir,
            prefix=f"{prefix}shared_" if len(condition_rows) > 1 else prefix,
            **shared,
        )

        def _draw_tf_maps(
            tf_by_condition: dict[str, np.ndarray], tf_prefix: str, tf_shared: dict
        ) -> list:
            """The whole-recording and stimulus-averaged TF grids, group and participant."""
            tf_stacked, tf_participants, tf_conditions = (
                stack_conditions_on_subject_axis(
                    tf_by_condition, condition_rows, participants
                )
            )
            plot_condition_mean_tf_maps(
                tf_stacked,
                tf_participants,
                tf_conditions,
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
                tf_stacked,
                tf_participants,
                tf_conditions,
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
                else _onset_average_per_condition(
                    tf_by_condition,
                    condition_onsets,
                    participants,
                    condition_rows,
                    label=label,
                    sfreq=sfreq,
                    min_onsets=args.min_onsets,
                )
            )
            if averaged is not None:
                onset_stacked, onset_parts, onset_conds, epoch_times, marks = averaged
                plot_condition_mean_tf_maps(
                    onset_stacked,
                    onset_parts,
                    onset_conds,
                    condition_rows,
                    freqs,
                    epoch_times,
                    comp_indices,
                    show_difference=show_difference,
                    freq_marks=_TF_FREQ_MARKS,
                    epoch_marks=marks,
                    save_path=out_dir / f"{tf_prefix}condition_mean_tf_maps_onset.png",
                    equalize=False,
                    **tf_shared,
                )
                plt.close("all")
                drawn += plot_participant_condition_tf_maps(
                    onset_stacked,
                    onset_parts,
                    onset_conds,
                    condition_rows,
                    freqs,
                    epoch_times,
                    comp_indices,
                    root_dir=participants_dir,
                    prefix=f"{tf_prefix}onset_",
                    equalize=False,
                    freq_marks=_TF_FREQ_MARKS,
                    epoch_marks=marks,
                    **tf_shared,
                )
            return drawn

        written += _draw_tf_maps(sources_by_condition, prefix, shared)
        # The full fit read through the ROI alone, then through the rest of the head. A
        # cut fit has only one of the two electrode sets, so it has no masked reading.
        masked_readings = (
            {}
            if fit.channels != "full" or args.skip_roi_masked
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
                    {
                        c.value: paired.condition_track(masked_sources, c)
                        for c in paired.conditions
                    },
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
        _logger.info(
            f"[{label}] done: {len(written)} per-participant figure(s), condition grids "
            f"and the synchrony report in {out_dir}"
        )

    run_corrca_fits(
        data_4d,
        topo_info,
        coordinate_system,
        full,
        args,
        prepare=_prepare,
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
            "--conditions must name at least two distinct conditions; a concatenated "
            f"run lays them end to end, got {args.conditions}."
        )
    _logger.info(
        f"CorrCA condition tracks: experiment={experiment_name.value}, "
        f"conditions={args.conditions}, music_types={[m.value for m in music_types]}, "
        f"n_pca={args.n_pca}, gamma={args.gamma}, zscore_mode={args.zscore_mode}"
    )
    for music_type in music_types:
        _logger.info(f"=== {music_type.value} ===")
        run_corrca_condition_tracks(music_type, args)


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
