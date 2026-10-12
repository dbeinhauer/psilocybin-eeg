"""
Stage 08: train the contrastive synchrony model and score group synchrony with it.

An encoder learns to embed a short window of a subject's wavelet power so that windows
from **different subjects at the same moment of the same track** land close together,
and windows from different moments land apart. The group synchrony of a window is then
the mean pairwise cosine similarity of all subjects' embeddings there, compared with
circularly time-shifted surrogates; windows above the null threshold are the candidate
synchronisation intervals.

**Cross-fitting.** A contrastive model makes the moments it trained on synchronous by
construction, so a window is only ever scored by a model that never saw it. Each track
is cut into ``training.n_folds`` time blocks; fold *k* trains on every block but *k*
and scores block *k*. Each block gets its own null, because each fold model has its own
embedding space.

Every setting comes from ``config/models/contrastive_synchrony.yaml``; ``--config``
merges a variant file over it and ``--set`` overrides single keys. The method is
documented in ``docs/analyses/08-contrastive-synchrony/README.md``.

Input: the stage-03 wavelet store, ``data/processed/<experiment>/wavelets/<label>/``.

Output::

    results/<experiment>/contrastive_synchrony/<run_name>/
        config.yaml              the resolved config the run used
        model_fold<k>.pt         encoder weights of each fold
        history.csv              loss and retrieval accuracy per fold and epoch
        synchrony.csv            group synchrony per window and track
        candidates.csv           merged candidate intervals
        subject_synchrony.csv    each participant's similarity to the group
    plots/08-contrastive-synchrony/<run_name>/
        training_history.png
        <label>_synchrony.png
        <label>_subject_synchrony.png

Examples::

    python scripts/run_contrastive_synchrony.py
    python scripts/run_contrastive_synchrony.py --run_name mlp --set model.name=mlp
    python scripts/run_contrastive_synchrony.py --config my_variant.yaml --skip_training
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import add_subset_arguments  # noqa: E402
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.models.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SynchronyConfig,
    load_config,
    save_config,
)
from src.models.data import (  # noqa: E402
    SynchronyDataset,
    build_dataset,
    cross_fit_split,
    window_frames,
    window_starts,
)
from src.models.encoders import build_encoder, n_parameters  # noqa: E402
from src.models.synchrony import (  # noqa: E402
    candidate_intervals,
    score_track,
    subject_synchrony,
)
from src.models.training import (  # noqa: E402
    embed_windows,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_model,
)
from src.visualization.contrastive_plots import (  # noqa: E402
    plot_subject_synchrony,
    plot_synchrony_timecourse,
    plot_training_history,
)

_logger = logging.getLogger(__name__)

STAGE = "08-contrastive-synchrony"
RESULTS_SUBDIR = "contrastive_synchrony"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Variant YAML merged over the default config. Default: the default.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config keys, e.g. --set training.epochs=5 model.name=mlp.",
    )
    parser.add_argument(
        "--run_name", default=None, help="Shortcut for --set run_name=<name>."
    )
    parser.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help="Processed-data root holding <experiment>/wavelets. Default: "
        "data/processed; notebooks/cache for the notebook strips.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Threads reading participants' wavelets in parallel (-1 = all CPUs).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Results root; the run writes to <root>/contrastive_synchrony/<run_name>/. "
        "Default: results/<experiment>/.",
    )
    parser.add_argument(
        "--save_dir", type=Path, default=None, help="Plot root. Default: plots/."
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Load the run's saved fold models and only score and plot.",
    )
    add_subset_arguments(parser)  # --n_times counts frames at data.frame_rate here
    parser.add_argument("--verbose", action="store_true")
    return parser


def train_fold(
    fold: int,
    datasets: list[SynchronyDataset],
    config: SynchronyConfig,
    window: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Train one fold's encoder on every track with block *fold* held out."""
    n_folds = config.training.n_folds
    splits = [cross_fit_split(ds, fold, n_folds, window) for ds in datasets]
    torch.manual_seed(config.training.seed + fold)
    model = build_encoder(
        config.model.name,
        datasets[0].n_features,
        window,
        config.model.embedding_dim,
        config.model.params,
    ).to(device)
    _logger.info(
        f"Fold {fold + 1}/{n_folds}: encoder '{config.model.name}', "
        f"{n_parameters(model):,} parameters, input "
        f"({datasets[0].n_features} features x {window} frames)"
    )
    history = train_model(
        model,
        [segment for segments, _ in splits for segment in segments],
        [held_out for _, held_out in splits],
        config.training,
        window,
        datasets[0].frame_rate,
        device,
    )
    return model, history.assign(fold=fold)


def score_block(
    model: torch.nn.Module,
    block: SynchronyDataset,
    config: SynchronyConfig,
    window: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window scores and per-subject similarity of one held-out block.

    Times are in seconds from the start of the whole track.
    """
    stride = max(1, int(round(config.scoring.stride_s * block.frame_rate)))
    starts = window_starts(block.n_frames, window, stride)
    starts_s = (starts + block.frame_offset) / block.frame_rate
    embeddings = embed_windows(model, block.data, window, starts, device)
    scores = score_track(
        embeddings,
        starts_s,
        window / block.frame_rate,
        n_null=config.scoring.n_null,
        null_quantile=config.scoring.null_quantile,
        min_shift=math.ceil(window / stride),
        seed=config.training.seed,
    )
    subjects = pd.DataFrame(
        {
            "participant": [p for p in block.participants for _ in starts],
            "start_s": list(starts_s) * block.n_subjects,
            "similarity": subject_synchrony(embeddings).ravel(),
        }
    )
    return scores, subjects


def run(args: argparse.Namespace) -> None:
    """Train (unless skipped), score and plot for an already-parsed namespace."""
    overrides = list(args.set) + (
        [f"run_name={args.run_name}"] if args.run_name else []
    )
    config = load_config(args.config, overrides)
    data_cfg = config.data
    n_folds = config.training.n_folds
    device = resolve_device(config.training.device)
    results_root = (
        Path(args.results_dir)
        if args.results_dir is not None
        else ProjectPaths.PROJECT_ROOT / "results" / data_cfg.experiment_name.value
    )
    run_dir = results_root / RESULTS_SUBDIR / config.run_name
    plots_dir = ProjectPaths.analysis_plots_dir(
        STAGE, config.run_name, root=args.save_dir
    )
    _logger.info(
        f"Run '{config.run_name}' on {device}: results {run_dir}, plots {plots_dir}"
    )

    datasets = [
        build_dataset(
            data_cfg,
            condition,
            music_type,
            root=args.wavelet_data_dir,
            n_subjects=args.n_subjects,
            n_channels=args.n_channels,
            n_times=args.n_times,
            n_jobs=args.n_jobs,
        )
        for condition, music_type in data_cfg.recordings
    ]
    window = window_frames(data_cfg.window_s, datasets[0].frame_rate)

    histories, scores, subjects = [], [], []
    for fold in range(n_folds):
        model_path = run_dir / f"model_fold{fold}.pt"
        if args.skip_training:
            model = load_checkpoint(model_path, device)
        else:
            model, history = train_fold(fold, datasets, config, window, device)
            save_checkpoint(
                model_path, model, config.model, datasets[0].n_features, window
            )
            histories.append(history)
        for dataset in datasets:
            _, block = cross_fit_split(dataset, fold, n_folds, window)
            block_scores, block_subjects = score_block(
                model, block, config, window, device
            )
            keys = {
                "label": dataset.label,
                "condition": dataset.condition.value,
                "music_type": dataset.music_type.value,
                "fold": fold,
            }
            scores.append(block_scores.assign(**keys))
            subjects.append(block_subjects.assign(**keys))

    scores_df = pd.concat(scores, ignore_index=True)
    subjects_df = pd.concat(subjects, ignore_index=True)
    # Per block: a run of candidates never spans two folds, whose nulls differ.
    intervals_df = pd.concat(
        [
            candidate_intervals(block).assign(label=label, fold=fold)
            for (label, fold), block in scores_df.groupby(["label", "fold"], sort=False)
        ],
        ignore_index=True,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    if histories:  # the config describes the trained models; re-scoring keeps it
        save_config(config, run_dir / "config.yaml")
        history_df = pd.concat(histories, ignore_index=True)
        history_df.to_csv(run_dir / "history.csv", index=False)
        plt.close(plot_training_history(history_df, plots_dir / "training_history.png"))
    scores_df.to_csv(run_dir / "synchrony.csv", index=False)
    intervals_df.to_csv(run_dir / "candidates.csv", index=False)
    subjects_df.to_csv(run_dir / "subject_synchrony.csv", index=False)

    for dataset in datasets:
        label = dataset.label
        track = scores_df[scores_df["label"] == label]
        _logger.info(
            f"[{label}] {len(track)} windows, mean synchrony "
            f"{track['synchrony'].mean():.3f}, "
            f"{int((intervals_df['label'] == label).sum())} candidate interval(s)"
        )
        plt.close(
            plot_synchrony_timecourse(
                track,
                intervals_df[intervals_df["label"] == label],
                f"{label} — cross-fitted group synchrony ({config.run_name})",
                plots_dir / f"{label}_synchrony.png",
            )
        )
        plt.close(
            plot_subject_synchrony(
                subjects_df[subjects_df["label"] == label],
                f"{label} — similarity of each participant to the group",
                plots_dir / f"{label}_subject_synchrony.png",
            )
        )
    _logger.info(f"Done: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
