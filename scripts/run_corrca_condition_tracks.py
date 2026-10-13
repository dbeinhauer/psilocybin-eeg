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

Output::

    plots/06b-corrca-condition-comparison/JoinedTracks_<MusicType>/
        broadband/corrca_channel_joined_tracks/pca_<n_pca>/
        bands/corrca_channel_joined_tracks/pca_<n_pca>/      # --band / --band_range

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    add_corrca_condition_args,
    anchor_corrca_signs,
    corrca_store_extras,
    log_corrca_conditions,
    resolve_wavelet_dir,
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
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    CoordinateSystems,
    ExclusionCategories,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
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

#: Stage directory under ``plots/`` — shared with the subject-axis variant.
_STAGE_DIR = "06b-corrca-condition-comparison"

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
    music_type: MusicTypeVariants, args: argparse.Namespace
) -> None:
    """Fit, orient, score and draw the time-axis join for one music type.

    :param music_type: Music type to analyse (``ASSR`` for the ASSR experiment).
    :param args: Parsed arguments, as :func:`_build_arg_parser` produces.
    """
    experiment_name = ExperimentNames(args.experiment)
    conditions = [ConditionVariants(c) for c in args.conditions]
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
        zscore_mode=args.zscore_mode,
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

    label = f"{ConditionVariants.JOINED_TRACKS.value}_{music_type.value}"
    sfreq = paired.data.sfreq
    band = tuple(args.band_range) if args.band_range is not None else args.band
    band_name = band_token(band)
    data_4d, freqs = slice_to_band(paired.data.data, freqs_full, band)
    _n_subj, n_ch, _n_freqs, n_total = data_4d.shape
    _logger.info(
        f"[{label}] concatenated {data_4d.shape} (participants × channels × freqs × "
        f"times), segments {paired.segment_lengths}, zscore_mode={args.zscore_mode}"
    )

    corrca = decompose_channel_corrca(
        data_4d, label=label, n_pca=args.n_pca, gamma=args.gamma
    )
    topo_info = topo_info_subset(condition_analyzers[paired.conditions[0]].info, n_ch)
    channel_names = list(topo_info.ch_names)
    corrca, alignment_note, extras = anchor_corrca_signs(
        corrca,
        channel_names,
        CoordinateSystems(args.coordinate_system),
        lenient_mask=args.lenient_mask,
        label=label,
        logger=_logger,
    )
    sources, patterns = corrca.sources, corrca.patterns

    # Split the projected sources back into the conditions — the filters are shared,
    # so each segment is the same component seen under one condition.
    condition_rows = [c.value for c in paired.conditions]
    participants = list(paired.participants)
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

    if args.store_components:
        stored_onsets: dict[str, np.ndarray] = {}
        for condition, length in zip(paired.conditions, paired.segment_lengths):
            onsets_c = paired.condition_onsets(condition, absolute=False)
            if onsets_c is not None:
                stored_onsets[f"stimulus_onsets_{condition.value}"] = onsets_c[
                    onsets_c < length
                ].astype(np.int64)
        save_iva_components(
            experiment=experiment_name,
            condition=ConditionVariants.JOINED_TRACKS,
            variant=_VARIANT,
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
        raise ValueError(f"--components {out_of_range} are outside 1..{n_components}.")

    sources_stacked, subject_participants, subject_conditions = (
        stack_conditions_on_subject_axis(
            sources_by_condition, condition_rows, participants
        )
    )
    n_split = sources_stacked.shape[-1]
    times = np.arange(n_split) / sfreq
    ref_onsets = paired.condition_onsets(paired.conditions[0], absolute=False)
    onset_times = (
        np.array([]) if ref_onsets is None else ref_onsets[ref_onsets < n_split] / sfreq
    )

    spectrum = (
        SpectrumTypeVariants.BROADBAND.value
        if band_name is None
        else SpectrumTypeVariants.BANDS.value
    )
    out_dir = ProjectPaths.analysis_plots_dir(
        _STAGE_DIR,
        label,
        spectrum=spectrum,
        analysis_type=_ANALYSIS_DIR,
        n_pca=args.n_pca,
        root=args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH,
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
    # comparison.
    shared_row = " + ".join(condition_rows) + " (shared)"
    plot_condition_mean_topomaps(
        patterns,
        participants,
        [shared_row] * len(participants),
        [shared_row],
        topo_info,
        n_ch,
        comp_indices,
        save_path=out_dir / f"{prefix}shared_mean_topomaps.png",
        **shared,
    )
    plt.close("all")
    plot_condition_mean_tf_maps(
        sources_stacked,
        subject_participants,
        subject_conditions,
        condition_rows,
        freqs,
        times,
        comp_indices,
        show_difference=show_difference,
        time_marks=onset_times,
        freq_marks=_TF_FREQ_MARKS,
        save_path=out_dir / f"{prefix}condition_mean_tf_maps.png",
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
        prefix=f"{prefix}shared_",
        **shared,
    )
    written += plot_participant_condition_tf_maps(
        sources_stacked,
        subject_participants,
        subject_conditions,
        condition_rows,
        freqs,
        times,
        comp_indices,
        root_dir=participants_dir,
        prefix=prefix,
        time_marks=onset_times,
        freq_marks=_TF_FREQ_MARKS,
        **shared,
    )

    averaged = (
        None
        if args.skip_onset_average
        else _onset_average_per_condition(
            sources_by_condition,
            {
                c.value: paired.condition_onsets(c, absolute=False)
                for c in paired.conditions
            },
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
            save_path=out_dir / f"{prefix}condition_mean_tf_maps_onset.png",
            **shared,
        )
        plt.close("all")
        written += plot_participant_condition_tf_maps(
            onset_stacked,
            onset_parts,
            onset_conds,
            condition_rows,
            freqs,
            epoch_times,
            comp_indices,
            root_dir=participants_dir,
            prefix=f"{prefix}onset_",
            freq_marks=_TF_FREQ_MARKS,
            epoch_marks=marks,
            **shared,
        )

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
