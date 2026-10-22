"""
ASSR channel-PCA time-frequency maps before any joint decomposition, Placebo against
Psilocybin, by four orders of the same three steps.

The question is what the recordings show around the ``fam+`` onsets before IVA or CorrCA
touch them. Each pipeline reduces one recording to its first ``--n_components`` channel
PCs, and every pipeline runs on Placebo and Psilocybin **separately**:

1. ``1_zscore_pca`` — every channel/frequency z-scored against the whole recording, then a
   channel PCA, as ``run_assr_wavelet_pca.py``.
2. ``2_roi_zscore_pca`` — the same, on the binary ASSR ROI only (the electrodes of
   ``config/assr_electrodes/``, 06b's ``ASSR-mask (full)``).
3. ``3_pca_zscore`` — the channel PCA first, on raw power, then each component's power
   z-scored against the whole recording.
4. ``4_roi_pca_zscore`` — the ROI, then the PCA, then the z-score.

The PCA is fit as stage 00 fits it: on the onset-averaged ``(channels, freqs, time)``
stack, each ``(freq, time)`` bin one observation. Its loadings are then applied to the
continuous recording, which gives each component a continuous ``(freqs, times)``
**source** — z-scored before (1, 2) or after (3, 4) the projection. A component's TF map
is its source averaged over the onsets, so zero is the recording's mean level in every
pipeline (in 1 that differs from ``run_assr_wavelet_pca.py``'s map by one constant).
Signs are aligned per component across participants AND both conditions in one pass, so
the Placebo − Psilocybin row is not a sign artefact.

**SNR grid.** Each pipeline's sources are read the way 06b reads its ``full``
variant: the selection's bins averaged, trials cut, every trial referenced to its own
pre-stimulus window, the median over trials. The reference row is the binary ROI built
exactly as 06b builds ``ASSR-mask (full)``
(:func:`~src.analysis.assr_trials.roi_channelwise_snr` on raw power, one scale shared by
both conditions), so on the same cohort its numbers are 06b's. Tests are 06b's
(participants as the unit, exact Wilcoxon, uncorrected): ``contrast`` (Placebo −
Psilocybin per source), ``discrimination`` and ``snr`` (each PC against the ROI).

Outputs::

    results/<experiment>/assr_roi_tf_variants/ASSR/snr_grid__full__pca<N>.csv
    results/<experiment>/assr_roi_tf_variants/ASSR/snr_values__full__pca<N>.npz
    results/<experiment>/assr_roi_tf_variants/ASSR/snr_courses__full__pca<N>.npz
    results/<experiment>/assr_roi_tf_variants/ASSR/tf_maps__pca<N>.npz  # --store_tf_maps
    plots/00-preprocessing/assr_roi_tf_variants/ASSR/pca_<N>/
        tf_maps__<pipeline>.png                  # Placebo / Psilocybin / diff x PCs
        tf_baseline__<pipeline>.png              # the same, trial-normalised
        subjects/PSI<id>/{tf_maps,tf_baseline}__<pipeline>.png
        snr/<selection>/<window>/
            pvalue_summary__<pipeline>.png
            snr_by_condition__<pipeline>.png
            trial_course_by_source__<pipeline>.png

Example::

    python scripts/run_assr_roi_tf_variants.py --n_subjects 3 --n_times 8000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stage 06's trial windows and test bookkeeping, so the grid writes 06b's CSV schema.
from scripts.run_assr_snr_grid import (  # noqa: E402
    _accumulate_tests,
    _stimulus_interval,
)

# Stage 00's onset average and channel PCA, so pipeline 1 is the same computation.
from scripts.run_assr_wavelet_pca import (  # noqa: E402
    _PID_COL,
    channel_pca_tf_maps,
    epoch_average,
)
from src.analysis import assr_trials as at  # noqa: E402
from src.analysis import iva_quality  # noqa: E402
from src.analysis.assr_variants import zscore_time  # noqa: E402
from src.analysis.pca_polarity import (  # noqa: E402
    align_pc1_signs,
    topography_consistency,
)
from src.definitions.constants import AssrEpoch, ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    REAL_CONDITIONS,
    CoordinateSystems,
    ExperimentNames,
    MusicTypeVariants,
)
from src.io.loading import assr_electrode_mask, concatenated_dir  # noqa: E402
from src.io.saving import (  # noqa: E402
    save_npz_atomic,
    save_snr_courses,
    save_snr_values,
)
from src.io.wavelet_store import find_wavelet_set, wavelet_label_dir  # noqa: E402
from src.visualization.assr_snr_plots import (  # noqa: E402
    plot_cell,
    plot_condition_tf_grid,
    plot_courses,
    plot_snr,
)

_STAGE_DIR = "00-preprocessing"
_ANALYSIS_DIR = "assr_roi_tf_variants"
_SIGNAL_VARIANT = "full"
#: The binary ROI is the one reference row, under 06b's name for it.
_ROI_LABEL = at.FULL_MASK_LABEL
_REFERENCES = (_ROI_LABEL,)
_PC_PREFIX = "PC"


@dataclass(frozen=True)
class Pipeline:
    """One order of ROI restriction, z-scoring and channel PCA."""

    name: str
    roi_first: bool
    zscore_first: bool
    title: str


PIPELINES = (
    Pipeline("1_zscore_pca", False, True, "z-score → channel PCA (all channels)"),
    Pipeline("2_roi_zscore_pca", True, True, "binary ROI → z-score → channel PCA"),
    Pipeline("3_pca_zscore", False, False, "channel PCA → z-score (all channels)"),
    Pipeline("4_roi_pca_zscore", True, False, "binary ROI → channel PCA → z-score"),
)
_BY_NAME = {p.name: p for p in PIPELINES}


@dataclass
class PipelineResult:
    """One (pipeline, condition) over the paired cohort.

    :param tf_maps: ``(K, P, F, W)`` onset-averaged sources.
    :param tf_baseline: ``(K, P, F, W)`` the same epochs, every trial referenced to its
        own pre-stimulus window (:func:`~src.analysis.assr_trials.baseline_tf_map`).
    :param loadings: ``(K, P, C)`` — ``C`` is the pipeline's channel count.
    :param explained: ``(K, P)`` explained-variance ratios.
    :param course: selection → ``(P, K, W)`` trial-median course.
    :param value: ``(selection, window)`` → ``(P, K)`` trial-median window mean.
    """

    tf_maps: np.ndarray
    tf_baseline: np.ndarray
    loadings: np.ndarray
    explained: np.ndarray
    course: dict[str, np.ndarray]
    value: dict[tuple[str, str], np.ndarray]


@dataclass(frozen=True)
class Grid:
    """The epoch, frequency selections and test windows every recording is cut on."""

    pre: int
    post: int
    baseline_mask: np.ndarray
    selections: dict[str, np.ndarray]
    windows: dict[str, np.ndarray]


def pipeline_sources(
    data: np.ndarray,
    onsets: np.ndarray,
    pre: int,
    post: int,
    n_components: int,
    zscore_first: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One recording's channel PCs, z-scored before or after the projection.

    :param data: ``(C, F, T)`` raw wavelet power of the channels in scope.
    :param onsets: Onset samples on this time axis.
    :param pre: Samples kept before each onset.
    :param post: Samples kept after each onset.
    :param zscore_first: z-score each channel/frequency, then fit and project (1, 2);
        otherwise fit and project raw power, then z-score each component (3, 4).
    :return: ``(tf_maps (K, F, W), loadings (K, C), explained (K,),
        sources (K, F, T))`` — *tf_maps* is *sources* averaged over the onsets.
    """
    signal = zscore_time(data) if zscore_first else data
    stack, _ = epoch_average(signal, onsets, pre, post)
    _, loadings, explained = channel_pca_tf_maps(stack, n_components)
    # In the signal's dtype: a float64 loading would upcast the whole (C, F, T) block.
    sources = np.tensordot(loadings.astype(signal.dtype), signal, axes=(1, 0))
    if not zscore_first:
        sources = zscore_time(sources)
    tf_maps, _ = epoch_average(sources, onsets, pre, post)
    return tf_maps, loadings, explained, sources


def full_reading(
    sources: np.ndarray, onsets: np.ndarray, grid: Grid
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """06b's ``full`` reading of continuous ``(K, F, T)`` sources.

    :return: ``(course, value)`` — selection → ``(K, W)`` trial-median course and
        ``(selection, window)`` → ``(K,)`` trial-median window mean.
    """
    course, value = {}, {}
    for sel, bins in grid.selections.items():
        band = sources[:, bins].mean(axis=1)  # (K, T)
        trials, _ = at.cut_trials(band, onsets, grid.pre, grid.post)  # (K, N, W)
        normed, _rel, _positive = at.baseline_normalise(trials, grid.baseline_mask)
        course[sel] = np.median(normed, axis=1)
        for win, window in grid.windows.items():
            value[(sel, win)] = np.median(normed[..., window].mean(axis=-1), axis=1)
    return course, value


def roi_reference(
    roi_raw: np.ndarray, onsets: np.ndarray, grid: Grid
) -> tuple[dict, dict, dict]:
    """06b's ``ASSR-mask (full)`` row for one recording, on its own baseline scale.

    :param roi_raw: ``(R, F, T)`` raw power of the ROI electrodes.
    :return: ``(course, value, scale)`` — selection → ``(W,)``, ``(selection,
        window)`` → float, and selection → this recording's ROI baseline SD; the
        caller moves course and value onto the shared scale by ``* scale / shared``.
    """
    course, value, scale = {}, {}, {}
    for sel, bins in grid.selections.items():
        band = roi_raw[:, bins].mean(axis=1)  # (R, T)
        trials, _ = at.cut_trials(band, onsets, grid.pre, grid.post)  # (R, N, W)
        snr, diagnostics = at.roi_channelwise_snr(trials[None], grid.baseline_mask)
        course[sel] = np.median(snr[0], axis=0)
        scale[sel] = diagnostics["roi_baseline_sd"]
        for win, window in grid.windows.items():
            value[(sel, win)] = float(np.median(snr[0][:, window].mean(axis=-1)))
    return course, value, scale


def align_signs(
    by_condition: dict[str, PipelineResult], channel_names: list[str]
) -> dict[str, list]:
    """Stage 00's per-component sign alignment over participants AND conditions.

    Every recording was fit on its own, so each is flipped toward one template built
    from both conditions. Signs are applied in place.

    :return: condition → per-component
        :class:`~src.analysis.pca_polarity.SignConsistency`.
    """
    conditions = list(by_condition)
    n_components = by_condition[conditions[0]].loadings.shape[0]
    consistency: dict[str, list] = {c: [] for c in conditions}
    for k in range(n_components):
        stacked = np.concatenate([by_condition[c].loadings[k] for c in conditions])
        signs, _anchor = align_pc1_signs(stacked, channel_names=channel_names)
        for c, sign in zip(conditions, np.split(signs, len(conditions))):
            result = by_condition[c]
            result.loadings[k] *= sign[:, None]
            result.tf_maps[k] *= sign[:, None, None]
            result.tf_baseline[k] *= sign[:, None, None]
            for course in result.course.values():
                course[:, k] *= sign[:, None]
            for value in result.value.values():
                value[:, k] *= sign
            consistency[c].append(topography_consistency(result.loadings[k]))
    return consistency


def store_tf_maps(
    path: Path,
    results: dict[str, dict[str, PipelineResult]],
    conditions: list[str],
    participants: list[str],
    epoch_times: np.ndarray,
    freqs: np.ndarray,
) -> None:
    """Write every pipeline's sign-aligned onset-averaged TF maps to one ``.npz``.

    Keys ``tf_maps__<pipeline>__<condition>`` and
    ``tf_baseline__<pipeline>__<condition>`` hold ``(K, P, F, W)`` and
    ``explained__<pipeline>__<condition>`` hold ``(K, P)``; ``participants``,
    ``conditions``, ``pipelines``, ``epoch_times`` and ``freqs`` are the axes. Written
    to a temporary name and renamed, so a full disk never leaves a truncated file.
    """
    arrays: dict[str, np.ndarray] = {
        "participants": np.asarray(participants),
        "conditions": np.asarray(conditions),
        "pipelines": np.asarray(list(results)),
        "epoch_times": epoch_times,
        "freqs": freqs,
    }
    for pipeline, by_condition in results.items():
        for c in conditions:
            arrays[f"tf_maps__{pipeline}__{c}"] = by_condition[c].tf_maps.astype(
                np.float32
            )
            arrays[f"tf_baseline__{pipeline}__{c}"] = by_condition[
                c
            ].tf_baseline.astype(np.float32)
            arrays[f"explained__{pipeline}__{c}"] = by_condition[c].explained
    save_npz_atomic(path, **arrays)


def _write_cell_plots(
    cell_dir: Path,
    cell: dict,
    rows_out: list[dict],
    value: dict,
    course: dict,
    times: np.ndarray,
    window_interval: tuple[float, float],
    conditions: list[str],
    labels: list[str],
    pc_labels: list[str],
    alpha: float,
    n_participants: int,
    note: str,
) -> None:
    """06b's three SNR-grid figures for one cell, the ROI as the one reference."""
    pipeline = cell["pipeline"]
    tag = f"{pipeline}__{cell['variant']}__{cell['selection']}__{cell['window']}"
    frame = pd.DataFrame([r for r in rows_out if all(r[k] == cell[k] for k in cell)])
    tables = {
        "contrast": frame[frame["family"] == "contrast"],
        f"discrimination:{_ROI_LABEL}": frame[frame["family"] == "discrimination"],
        **{
            f"snr:{_ROI_LABEL}:{c}": frame[
                (frame["family"] == "snr") & (frame["condition"] == c)
            ]
            for c in conditions
        },
    }
    plot_cell(
        tables,
        value,
        conditions,
        labels,
        pc_labels,
        alpha,
        f"Condition contrast & discrimination — {tag}, n={n_participants}",
        cell_dir / f"pvalue_summary__{pipeline}.png",
        component_prefix=_PC_PREFIX,
        mask_labels=_REFERENCES,
    )
    plot_snr(
        tables,
        value,
        conditions,
        pc_labels,
        alpha,
        f"PC SNR vs the binary ROI, per condition — {tag}, n={n_participants}",
        cell_dir / f"snr_by_condition__{pipeline}.png",
        component_prefix=_PC_PREFIX,
        mask_labels=_REFERENCES,
    )
    plot_courses(
        course,
        times,
        window_interval,
        tables["contrast"],
        labels,
        conditions,
        alpha,
        f"Trial course per source — {tag}, n={n_participants}\n{note}",
        cell_dir / f"trial_course_by_source__{pipeline}.png",
        mask_labels=_REFERENCES,
    )


def _load_inputs(experiment: ExperimentNames, conditions: list[str]) -> dict[str, dict]:
    concat = concatenated_dir(experiment)
    inputs = {}
    for condition in conditions:
        label = f"{condition}_{MusicTypeVariants.ASSR.value}"
        onsets_path = concat / f"{label}{ProjectPaths.STIMULUS_ONSETS_SUFFIX}"
        meta_path = concat / f"{label}.metadata.csv"
        wavelet_set = find_wavelet_set(wavelet_label_dir(None, experiment, label))
        for path in (onsets_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"{label}: no {path}.")
        pids = set(
            pd.read_csv(meta_path, index_col=0)[_PID_COL].astype(str).str.zfill(3)
        )
        inputs[condition] = {
            "wavelet_set": wavelet_set,
            "participants": pids - set(wavelet_set.missing(sorted(pids))),
            "onsets": np.load(onsets_path).astype(int),
        }
    return inputs


def run(args: argparse.Namespace) -> None:
    """Every pipeline over the paired cohort, then the CSV and the figures."""
    experiment = ExperimentNames(args.experiment)
    conditions = at.resolve_conditions(
        args.conditions or [c.value for c in REAL_CONDITIONS]
    )
    if len(conditions) != 2:
        raise ValueError(f"Need exactly two conditions to contrast; got {conditions}.")
    pipelines = [_BY_NAME[name] for name in args.pipelines]
    inputs = _load_inputs(experiment, conditions)

    # ---- paired cohort, shared axes, ROI ------------------------------------
    paired = set.intersection(*(item["participants"] for item in inputs.values()))
    dropped = set.union(*(item["participants"] for item in inputs.values())) - paired
    participants = sorted(paired, key=int)[: args.n_subjects]
    if len(participants) < 2:
        raise ValueError(f"Only {len(participants)} paired participant(s); need 2.")
    if dropped:
        print(f"Not in every condition (dropped): {sorted(dropped)}", flush=True)
    sets = [item["wavelet_set"] for item in inputs.values()]
    channel_names, freqs, sfreq = sets[0].channel_names, sets[0].freqs, sets[0].sfreq
    for ws in sets[1:]:
        if ws.channel_names != channel_names or not np.allclose(ws.freqs, freqs):
            raise ValueError("The conditions' wavelet caches disagree on their axes.")
    roi_index = np.flatnonzero(
        assr_electrode_mask(
            channel_names, CoordinateSystems(args.coordinate_system), strict=True
        )
    )
    roi_names = [channel_names[i] for i in roi_index]
    n_components = min(args.n_components, roi_index.size)
    pc_labels = [f"{_PC_PREFIX} {k + 1}" for k in range(n_components)]
    labels = pc_labels + [_ROI_LABEL]

    # ---- epoch window (06b's: common across conditions), selections, windows --
    n_times = {
        c: min(
            item["wavelet_set"].open(participants[0]).shape[-1],
            args.n_times or np.iinfo(int).max,
        )
        for c, item in inputs.items()
    }
    geometry = {
        c: at.epoch_geometry(item["onsets"], n_times[c], sfreq)
        for c, item in inputs.items()
    }
    pre, post = at.common_epoch_window(geometry)
    times = at.epoch_time_base(pre, post, sfreq)
    windows, intervals = {}, {}
    for lo, hi in args.stimulus_intervals:
        if (lo, hi) == (0.0, AssrEpoch.STIMULUS_DURATION_S):
            window = AssrEpoch.stimulus_mask(times)
        else:
            window = (times >= lo) & (times <= hi)
        if not window.any():
            raise ValueError(f"window {lo}-{hi}s selects no epoch sample.")
        windows[f"{lo:g}-{hi:g}s"], intervals[f"{lo:g}-{hi:g}s"] = window, (lo, hi)
    grid = Grid(
        pre=pre,
        post=post,
        baseline_mask=times < 0.0,
        selections={
            at.selection_label(args.center_freq, hw): at.frequency_selection(
                freqs, args.center_freq, hw
            )
            for hw in args.halfwidths
        },
        windows=windows,
    )
    print(
        f"{len(participants)} paired participants; {len(channel_names)} channels, ROI "
        f"{roi_index.size}; {freqs.size} bins {freqs[0]:.0f}-{freqs[-1]:.0f} Hz; epoch "
        f"[{times[0]:.3f}, {times[-1]:.3f}] s; {n_components} PC(s); pipelines "
        f"{[p.name for p in pipelines]}",
        flush=True,
    )

    # ---- one read per recording, every pipeline from it ---------------------
    n_subj, n_freqs, win = len(participants), freqs.size, pre + post

    def _empty(*shape: int) -> np.ndarray:
        return np.full(shape, np.nan)

    results = {
        p.name: {
            c: PipelineResult(
                _empty(n_components, n_subj, n_freqs, win),
                _empty(n_components, n_subj, n_freqs, win),
                _empty(
                    n_components,
                    n_subj,
                    roi_index.size if p.roi_first else len(channel_names),
                ),
                _empty(n_components, n_subj),
                {s: _empty(n_subj, n_components, win) for s in grid.selections},
                {
                    (s, w): _empty(n_subj, n_components)
                    for s in grid.selections
                    for w in grid.windows
                },
            )
            for c in conditions
        }
        for p in pipelines
    }
    roi_course = {
        c: {s: _empty(n_subj, win) for s in grid.selections} for c in conditions
    }
    roi_value = {
        c: {(s, w): _empty(n_subj) for s in grid.selections for w in grid.windows}
        for c in conditions
    }
    roi_scale = {c: {s: _empty(n_subj) for s in grid.selections} for c in conditions}

    for i, participant in enumerate(participants):
        for c in conditions:
            onsets = geometry[c][0]
            # One read into memory: every pipeline passes over the block several times,
            # which on the memory map would re-read it from storage each time.
            block = np.array(
                inputs[c]["wavelet_set"].open(participant)[..., : n_times[c]],
                dtype=np.float32,
            )
            roi_block = block[roi_index]
            course, value, scale = roi_reference(roi_block, onsets, grid)
            for s in grid.selections:
                roi_course[c][s][i] = course[s]
                roi_scale[c][s][i] = scale[s]
            for key, v in value.items():
                roi_value[c][key][i] = v
            for p in pipelines:
                tf_maps, loadings, explained, sources = pipeline_sources(
                    roi_block if p.roi_first else block,
                    onsets,
                    pre,
                    post,
                    n_components,
                    p.zscore_first,
                )
                result = results[p.name][c]
                result.tf_maps[:, i] = tf_maps
                result.tf_baseline[:, i] = at.baseline_tf_map(
                    sources, onsets, pre, post, grid.baseline_mask
                )
                result.loadings[:, i] = loadings
                result.explained[:, i] = explained
                course, value = full_reading(sources, onsets, grid)
                for s in grid.selections:
                    result.course[s][i] = course[s]
                for key, v in value.items():
                    result.value[key][i] = v
            del block, roi_block
        print(f"  reduced {i + 1}/{n_subj} ({participant})", flush=True)

    # 06b's shared ROI scale: per selection, each condition's median recording ROI
    # baseline SD, then the median over conditions — one constant for both.
    for s in grid.selections:
        shared = float(np.median([np.nanmedian(roi_scale[c][s]) for c in conditions]))
        for c in conditions:
            factor = roi_scale[c][s] / shared
            roi_course[c][s] *= factor[:, None]
            for w in grid.windows:
                roi_value[c][(s, w)] *= factor
    consistency = {
        p.name: align_signs(
            results[p.name], roi_names if p.roi_first else channel_names
        )
        for p in pipelines
    }

    # ---- the SNR / contrast grid --------------------------------------------
    def _values(p: str, s: str, w: str) -> dict[str, np.ndarray]:
        return {
            c: np.column_stack([results[p][c].value[(s, w)], roi_value[c][(s, w)]])
            for c in conditions
        }

    rows_out: list[dict] = []
    for p in pipelines:
        for s in grid.selections:
            for w in grid.windows:
                _accumulate_tests(
                    rows_out,
                    {
                        "pipeline": p.name,
                        "variant": _SIGNAL_VARIANT,
                        "selection": s,
                        "window": w,
                    },
                    _values(p.name, s, w),
                    conditions,
                    labels,
                    pc_labels,
                    args.contrast_alternative,
                    args.snr_alternative,
                    mask_labels=_REFERENCES,
                )
    dataset = MusicTypeVariants.ASSR.value
    results_dir = (
        (
            Path(args.results_dir)
            if args.results_dir
            else ProjectPaths.PROJECT_ROOT / "results" / experiment.value
        )
        / _ANALYSIS_DIR
        / dataset
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"snr_grid__{_SIGNAL_VARIANT}__pca{n_components}.csv"
    frame = pd.DataFrame(rows_out)
    frame.to_csv(csv_path, index=False)
    # Each pipeline's PC courses with the ROI course as the last source, as the course
    # figure reads them.
    courses = {
        (p.name, s): {
            c: np.concatenate(
                [results[p.name][c].course[s], roi_course[c][s][:, None]], axis=1
            )
            for c in conditions
        }
        for p in pipelines
        for s in grid.selections
    }
    courses_path = (
        results_dir / f"snr_courses__{_SIGNAL_VARIANT}__pca{n_components}.npz"
    )
    save_snr_courses(courses_path, courses, participants, labels, times)
    values_path = results_dir / f"snr_values__{_SIGNAL_VARIANT}__pca{n_components}.npz"
    save_snr_values(
        values_path,
        {
            (p.name, s, w): _values(p.name, s, w)
            for p in pipelines
            for s in grid.selections
            for w in grid.windows
        },
        participants,
        labels,
    )
    print(
        f"saved {csv_path}, {courses_path.name} and {values_path.name} "
        f"({len(frame)} test rows, floor p = "
        f"{at.p_floor(n_subj):.5f} two-sided). Contrast {conditions[0]} - "
        f"{conditions[1]} ({args.contrast_alternative}), p:",
        flush=True,
    )
    print(
        frame[frame["family"] == "contrast"]
        .pivot_table(
            index=["pipeline", "source"], columns=["selection", "window"], values="p"
        )
        .round(4)
        .to_string(),
        flush=True,
    )
    if args.store_tf_maps:
        tf_path = results_dir / f"tf_maps__pca{n_components}.npz"
        store_tf_maps(tf_path, results, conditions, participants, times, freqs)
        print(f"saved {tf_path}", flush=True)
    if args.skip_plots:
        return

    # ---- figures ------------------------------------------------------------
    # The analysis first, then the dataset: one tree for both conditions.
    plots_dir = (
        (Path(args.save_dir) if args.save_dir else ProjectPaths.PLOTS_PATH)
        / _STAGE_DIR
        / _ANALYSIS_DIR
        / dataset
        / f"pca_{n_components}"
    )
    for p in pipelines:
        by_condition = results[p.name]
        group_notes = {
            c: [
                f" — EV {by_condition[c].explained[k].mean() * 100:.0f}%, agreeing "
                f"{consistency[p.name][c][k].n_agreeing}/"
                f"{consistency[p.name][c][k].n_subjects}"
                for k in range(n_components)
            ]
            for c in conditions
        }
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Each map twice: z vs the whole recording (the onset average of the sources),
        # and trial-normalised (every trial against its own pre-stimulus window, in
        # that condition's baseline SD), which compares across methods and conditions.
        readings = {
            "tf_maps": ("tf_maps", "z vs recording", ""),
            "tf_baseline": ("tf_baseline", "baseline SD", ", trial-normalised"),
        }
        for stem, (field, unit, note) in readings.items():
            plot_condition_tf_grid(
                {c: getattr(by_condition[c], field).mean(axis=1) for c in conditions},
                group_notes,
                conditions,
                times,
                freqs,
                args.center_freq,
                f"{p.name}: {p.title}{note} — group mean (n={n_subj})",
                plots_dir / f"{stem}__{p.name}.png",
                colorbar_label=unit,
            )
        for i, participant in enumerate(participants):
            subject_dir = plots_dir / "subjects" / f"PSI{participant}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject_notes = {
                c: [
                    f" — EV {by_condition[c].explained[k, i] * 100:.0f}%"
                    for k in range(n_components)
                ]
                for c in conditions
            }
            for stem, (field, unit, note) in readings.items():
                plot_condition_tf_grid(
                    {c: getattr(by_condition[c], field)[:, i] for c in conditions},
                    subject_notes,
                    conditions,
                    times,
                    freqs,
                    args.center_freq,
                    f"{p.name}: {p.title}{note} — PSI{participant}",
                    subject_dir / f"{stem}__{p.name}.png",
                    colorbar_label=unit,
                )

        for s in grid.selections:
            for w in grid.windows:
                cell_dir = plots_dir / "snr" / s / w
                cell_dir.mkdir(parents=True, exist_ok=True)
                _write_cell_plots(
                    cell_dir,
                    {
                        "pipeline": p.name,
                        "variant": _SIGNAL_VARIANT,
                        "selection": s,
                        "window": w,
                    },
                    rows_out,
                    _values(p.name, s, w),
                    courses[(p.name, s)],
                    times,
                    intervals[w],
                    conditions,
                    labels,
                    pc_labels,
                    args.alpha,
                    n_subj,
                    f"{p.title}; PCs as 06b 'full', the ROI as 06b '{_ROI_LABEL}'",
                )
    print(f"[DONE] plots written to {plots_dir}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASSR channel-PCA TF maps by four orders of ROI / z-score / PCA, "
        "Placebo vs Psilocybin, with 06b's 'full' SNR grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", default=ExperimentNames.ASSR.value)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Two conditions, contrast order. Default Placebo Psilocybin.",
    )
    parser.add_argument(
        "--coordinate_system",
        default=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS.value,
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=[p.name for p in PIPELINES],
        choices=[p.name for p in PIPELINES],
        help="Which orders of ROI / z-score / PCA to run.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="Leading channel-PCA components (clamped to the ROI size).",
    )
    parser.add_argument("--center_freq", type=float, default=iva_quality.ASSR_FREQ)
    parser.add_argument(
        "--halfwidths",
        type=float,
        nargs="+",
        default=[0.0, iva_quality.TF_ANCHOR_HALFWIDTH_HZ],
        help="One SNR-grid frequency selection per half-width in Hz (0 = one bin).",
    )
    parser.add_argument(
        "--stimulus_intervals",
        type=_stimulus_interval,
        nargs="+",
        default=[
            (0.0, AssrEpoch.STIMULUS_DURATION_S),
            (0.2, AssrEpoch.STIMULUS_DURATION_S),
        ],
        help="One SNR-grid test window per 'lo:hi' seconds, e.g. 0:0.5 0.2:0.5.",
    )
    parser.add_argument(
        "--contrast_alternative",
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="Direction of Placebo - Psilocybin. 'greater' = psilocybin lowers it.",
    )
    parser.add_argument(
        "--snr_alternative",
        default="less",
        choices=["less", "greater", "two-sided"],
        help="Direction of (PC SNR - ROI SNR) in the SNR family.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--n_subjects", type=int, default=None, help="Keep the first N (small runs)."
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Trim the time axis (a quick shape check; not for real runs).",
    )
    parser.add_argument(
        "--save_dir", default=None, help="Plot root; default the project's plots/."
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Results root; default results/<experiment>/.",
    )
    parser.add_argument(
        "--store_tf_maps",
        action="store_true",
        help="Also write the group TF maps of every pipeline to "
        "results/.../tf_maps__pca<N>.npz (small), for redrawing without a rerun.",
    )
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    run(_build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
