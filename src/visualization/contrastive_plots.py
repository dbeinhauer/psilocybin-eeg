"""
Figures of the stage-08 contrastive synchrony model.

Each function takes the tables :mod:`src.models.training` and
:mod:`src.models.synchrony` produce, writes one figure through an optional
``save_path`` and returns it without showing it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.visualization.iva_quality_plots import save_fig

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
REFERENCE_COLOR = "#8a8984"  # chance level and null threshold: context, not data
CANDIDATE_COLOR = "#cde2fb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "similarity", ["#cde2fb", "#2a78d6", "#0d366b"]
)


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_training_history(
    history: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """Loss and positive-retrieval accuracy per epoch, training against held-out.

    One thin line per cross-fitting fold. Accuracy is the share of windows whose most
    similar other window in the batch was recorded at the same moment; the dashed line
    is what an untrained encoder scores. A held-out curve near chance while the
    training curve climbs is memorisation, not synchrony.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in (
        (ax_loss, "loss", "Contrastive loss"),
        (ax_acc, "accuracy", "Same-moment retrieval accuracy"),
    ):
        for split, color, name in (
            ("train", TRAIN_COLOR, "training"),
            ("val", VAL_COLOR, "held-out block"),
        ):
            for i, (_, fold) in enumerate(history.groupby("fold")):
                ax.plot(
                    fold["epoch"],
                    fold[f"{split}_{metric}"],
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=name if i == 0 else None,
                )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(ax)

    chance = history.groupby("epoch")["val_chance"].mean()
    ax_acc.plot(
        chance.index,
        chance.to_numpy(),
        color=REFERENCE_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="chance",
    )
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(frameon=False, loc="center right")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_synchrony_timecourse(
    scores: pd.DataFrame,
    intervals: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """Cross-fitted group synchrony over one track, with nulls and candidates.

    Each window is drawn at its centre, one line segment per fold block — every block
    was scored by a different model, so the trace is not joined across blocks. Each
    block has its own null threshold (dashed); shaded spans are candidate intervals.
    """
    fig, ax = plt.subplots(figsize=(12, 3.6))
    handles: list[Artist] = []
    for row in intervals.itertuples():
        ax.axvspan(row.start_s, row.end_s, color=CANDIDATE_COLOR, zorder=0.5)
    if len(intervals):
        handles.append(Patch(color=CANDIDATE_COLOR, label="candidate interval"))

    for _, block in scores.groupby("fold"):
        centre = (block["start_s"] + block["end_s"]) / 2
        ax.plot(centre, block["synchrony"], color=TRAIN_COLOR, linewidth=1.5)
        ax.hlines(
            block["null_threshold"].iloc[0],
            centre.min(),
            centre.max(),
            color=REFERENCE_COLOR,
            linewidth=1.2,
            linestyle="--",
        )
        ax.axvline(block["start_s"].min(), color="#e4e3df", linewidth=0.8, zorder=0)
    centre = (scores["start_s"] + scores["end_s"]) / 2
    handles.append(
        plt.Line2D(
            [],
            [],
            color=REFERENCE_COLOR,
            linestyle="--",
            label="null threshold (per block)",
        )
    )

    ax.set_xlim(centre.min(), centre.max())
    ax.set_xlabel("Time in track (s)")
    ax.set_ylabel("Mean pairwise cosine")
    ax.set_title(title, loc="left")
    # Above the axes, right-aligned against the left-aligned title: never over the data.
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(handles),
    )
    _style(ax)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_subject_synchrony(
    subject_scores: pd.DataFrame, title: str, save_path: Path | None = None
) -> Figure:
    """Each participant's similarity to the rest of the group, over time.

    Rows are participants; a vertical stripe across all rows is a moment the whole
    group shares, a short one carried by a few rows is not.
    """
    table = subject_scores.pivot(
        index="participant", columns="start_s", values="similarity"
    ).sort_index()
    starts = table.columns.to_numpy(dtype=float)
    step = float(np.diff(starts).min()) if len(starts) > 1 else 1.0
    # Onto a regular grid, so the unscored gaps at fold boundaries stay blank instead
    # of being squeezed out of the time axis.
    grid = np.arange(starts[0], starts[-1] + step / 2, step)
    table = table.reindex(columns=grid, method="nearest", tolerance=step / 4)
    values = table.to_numpy()
    starts, half = grid, step / 2
    fig, ax = plt.subplots(figsize=(12, 0.25 * len(table) + 1.6))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=SEQUENTIAL,
        vmin=float(np.nanpercentile(values, 1)),
        vmax=float(np.nanpercentile(values, 99)),
        interpolation="nearest",
        extent=(starts[0] - half, starts[-1] + half, len(table) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(table)), table.index)
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Participant")
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, label="Cosine to the others' mean", pad=0.01)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
