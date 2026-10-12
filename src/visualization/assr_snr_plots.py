"""
Figures for the stage-06 ASSR SNR / condition-contrast grid.

Four figures, each drawn once per grid cell (frequency selection x stimulus window):
the per-source trial courses, the p-value summary, the SNR-versus-reference panel, and
the forest panel the first two are built from.

Extracted from ``scripts/run_assr_snr_grid.py`` so the notebook and the CLI draw the
same figures rather than two implementations of them. Behaviour is unchanged from the
script's originals.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import assr_trials as at
from src.definitions.constants import AssrEpoch
from src.definitions.fields import ConditionVariants

_logger = logging.getLogger(__name__)

#: The reference rows every figure keeps distinct from the learned ones.
MASK_LABELS = at.MASK_LABELS

#: One colour per condition, so a course and its forest marker agree across figures.
CONDITION_COLORS = {
    ConditionVariants.PLACEBO.value: "#0F6E8C",
    ConditionVariants.PSILOCYBIN.value: "#A6357F",
}


def bootstrap_ci(values: np.ndarray, alpha: float, rng) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (np.nan, np.nan)
    draws = np.median(
        finite[rng.integers(0, finite.size, size=(10_000, finite.size))], axis=1
    )
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def forest(ax, rows, alpha, xlabel, title, rng):
    """A minimal forest: (label, per-participant diffs, table row) tuples."""
    medians = [r[2]["median"] for r in rows]
    cis = [bootstrap_ci(r[1], alpha, rng) for r in rows]
    edges = [b for ci in cis for b in ci] + medians
    span = (np.nanmax(edges) - np.nanmin(edges)) or 1.0
    lo, hi = np.nanmin(edges) - 0.18 * span, np.nanmax(edges) + 0.22 * span
    for i, (label, diffs, row) in enumerate(rows):
        y = len(rows) - 1 - i
        sig = row["p"] <= alpha
        colour = "#1B5E20" if sig else "0.45"
        low, high = cis[i]
        ax.plot(
            [low, high], [y, y], color=colour, lw=2.3, solid_capstyle="round", zorder=3
        )
        ax.scatter(
            [row["median"]], [y], s=80, color=colour, edgecolor="white", zorder=4
        )
        ax.text(
            0.99,
            y,
            f"p={row['p']:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=8.5,
            family="monospace",
            fontweight="bold" if sig else "normal",
            color=colour,
        )
    ax.set_xlim(lo, hi)
    ax.axvline(0.0, color="0.3", lw=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)


def plot_courses(
    course, times, window_interval, contrast, labels, conditions, alpha, title, path
):
    """Per-source epoch trial course, both conditions overlaid.

    The line is the mean across participants of each participant's median-over-trials
    course; the band is +/- SEM ACROSS PARTICIPANTS (never across trials, which are
    correlated within a participant). The paradigm's stimulus interval is shaded grey and
    the cell's test window gold, and each panel carries its 4b p-value.
    """
    by_source = contrast.set_index("source")
    n_src = len(labels)
    ncols = min(4, n_src)
    nrows = int(np.ceil(n_src / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axflat = axes.flat
    for s, source in enumerate(labels):
        ax = axflat[s]
        ax.axvspan(
            0.0, AssrEpoch.STIMULUS_DURATION_S, color="0.55", alpha=0.10, lw=0, zorder=0
        )
        ax.axvspan(
            window_interval[0],
            window_interval[1],
            color="#B8860B",
            alpha=0.13,
            lw=0,
            zorder=0,
        )
        ax.axhline(0.0, color="0.45", lw=0.8, zorder=1)
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
        for cond in conditions:
            vals = course[cond][:, s]  # (P, W)
            centre = np.nanmean(vals, axis=0)
            n = np.sum(~np.isnan(vals), axis=0)
            half = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            colour = CONDITION_COLORS.get(cond, "0.3")
            ax.fill_between(
                times,
                centre - half,
                centre + half,
                color=colour,
                alpha=0.18,
                lw=0,
                zorder=2,
            )
            ax.plot(times, centre, color=colour, lw=1.8, zorder=3, label=cond)
        row = by_source.loc[source]
        is_ref = source in MASK_LABELS
        ax.set_title(
            f"{source}{'  (reference)' if is_ref else ''}",
            loc="left",
            fontsize=11,
            fontweight="bold" if is_ref else "normal",
        )
        ax.text(
            0.985,
            0.955,
            f"{conditions[0]} > {conditions[1]}\np = {row['p']:.3f}   {row['same sign']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            family="monospace",
            color="0.25" if row["p"] > alpha else "black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.75" if row["p"] > alpha else "black",
                alpha=0.85,
            ),
        )
        if s + ncols >= n_src:
            ax.set_xlabel("Time from onset (s)")
        if s % ncols == 0:
            ax.set_ylabel("response (a.u.)")
    for j in range(n_src, nrows * ncols):
        axflat[j].axis("off")
    axflat[0].legend(loc="lower right", frameon=True, fontsize=9)
    fig.suptitle(title, y=1.0, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cell(cell_tables, value, conditions, labels, ic_labels, alpha, title, path):
    """The p-value summary (contrast + one versus panel per reference) for one grid cell."""
    rng = np.random.default_rng(42)
    contrast = cell_tables["contrast"].set_index("source")
    fig, axes = plt.subplots(
        1, 1 + len(MASK_LABELS), figsize=(7.6 + 5.6 * len(MASK_LABELS), 5.0)
    )
    rows = [
        (
            src,
            at.condition_contrast(value, conditions, labels.index(src)),
            contrast.loc[src],
        )
        for src in labels
    ]
    forest(
        axes[0],
        rows,
        alpha,
        f"{conditions[0]} - {conditions[1]}",
        f"Is {conditions[1]} lower than {conditions[0]}?",
        rng,
    )
    for ax, mask in zip(axes[1:], MASK_LABELS):
        table = cell_tables[f"discrimination:{mask}"].set_index("source")
        ref = labels.index(mask)
        rows = [
            (
                src,
                at.discrimination_gain(value, conditions, labels.index(src), ref),
                table.loc[src],
            )
            for src in ic_labels
        ]
        forest(
            ax,
            rows,
            alpha,
            f"(IC - {mask}) contrast",
            f"Better than {mask}?\ntwo-sided; 0 = as good",
            rng,
        )
    fig.suptitle(title, y=1.02, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snr(cell_tables, value, conditions, ic_labels, alpha, title, path):
    """SNR-by-condition: one panel per reference, both conditions overlaid.

    Source columns of ``value[cond]`` are the IC rows first, then the two masks, so an
    IC row *i* is column *i* and a mask is ``len(ic_labels) + its offset``.
    """
    rng = np.random.default_rng(42)
    offsets = np.linspace(0.2, -0.2, len(conditions))
    n_ic = len(ic_labels)
    fig, axes = plt.subplots(
        1,
        len(MASK_LABELS),
        figsize=(6.6 * len(MASK_LABELS), 0.8 + 0.7 * n_ic),
        squeeze=False,
    )
    for ax, mask in zip(axes[0], MASK_LABELS):
        ref_col = n_ic + MASK_LABELS.index(mask)
        drawn, edges = [], []
        for ci, cond in enumerate(conditions):
            table = cell_tables[f"snr:{mask}:{cond}"].set_index("source")
            m = value[cond]
            for i, src in enumerate(ic_labels):
                interval = bootstrap_ci(m[:, i] - m[:, ref_col], alpha, rng)
                edges += list(interval)
                drawn.append((ci, i, interval, table.loc[src]))
        span = (np.nanmax(edges) - np.nanmin(edges)) or 1.0
        lo, hi = np.nanmin(edges) - 0.18 * span, np.nanmax(edges) + 0.18 * span
        for ci, i, (low, high), row in drawn:
            y = n_ic - 1 - i + offsets[ci]
            colour = CONDITION_COLORS[conditions[ci]]
            sig = row["p"] <= alpha
            ax.plot(
                [low, high],
                [y, y],
                color=colour,
                lw=2.1,
                solid_capstyle="round",
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                [row["median"]],
                [y],
                s=64,
                marker="o",
                color=colour if sig else "white",
                edgecolor=colour,
                linewidth=1.5,
                zorder=4,
            )
            ax.text(
                0.985,
                y,
                f"p={row['p']:.3f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=8,
                family="monospace",
                fontweight="bold" if sig else "normal",
                color=colour,
            )
        ax.set_xlim(lo, hi)
        ax.axvline(0.0, color="0.3", lw=1.3, zorder=1)
        ax.set_yticks(range(n_ic))
        ax.set_yticklabels(list(reversed(ic_labels)))
        ax.set_xlabel(f"IC SNR - {mask} SNR")
        ax.set_title(f"vs {mask}", loc="left", fontsize=11)
    handles = [
        plt.Line2D([0], [0], marker="o", color=CONDITION_COLORS[c], lw=0, label=c)
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(conditions),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(title, y=1.03, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
