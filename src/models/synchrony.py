"""
Group synchrony read off window embeddings.

Given every subject's embedding of the same sequence of windows of one track, the
synchrony of a window is the **mean pairwise cosine similarity between subjects** at
that window. It is judged against circularly time-shifted surrogates: shifting each
subject by its own random lag keeps every subject's embeddings intact but destroys
their alignment in time, which is the only thing synchrony is about. The null is pooled
over all windows and surrogates, so one threshold holds for the whole track.

Model-agnostic and numpy-only: anything that maps windows to vectors can be scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _unit(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norm, 1e-12)


def group_synchrony(embeddings: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine similarity across subjects, per window.

    Uses ``sum_{i != j} e_i . e_j = |sum_i e_i|^2 - sum_i |e_i|^2``, so the cost is
    linear rather than quadratic in the number of subjects.

    :param embeddings: ``(subjects, windows, dim)``, at least two subjects.
    :return: ``(windows,)`` in ``[-1, 1]``.
    """
    unit = _unit(embeddings)
    n_subjects = unit.shape[0]
    total = unit.sum(axis=0)
    pair_sum = (total**2).sum(axis=-1) - (unit**2).sum(axis=(0, -1))
    return pair_sum / (n_subjects * (n_subjects - 1))


def subject_synchrony(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity of each subject with the mean of all the others, per window.

    Shows who carries a synchronous moment: a high group value that rests on a few
    subjects looks different here from one shared by everybody.

    :param embeddings: ``(subjects, windows, dim)``.
    :return: ``(subjects, windows)``.
    """
    unit = _unit(embeddings)
    others = _unit(unit.sum(axis=0, keepdims=True) - unit)
    return (unit * others).sum(axis=-1)


def shifted_null(
    embeddings: np.ndarray, n_null: int, min_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """Group synchrony of circularly time-shifted surrogates.

    :param min_shift: Smallest lag in windows; set it to the window overlap so a
        shifted window never overlaps its own original position.
    :return: ``(n_null, windows)``.
    :raises ValueError: If the track is too short for any admissible lag.
    """
    n_subjects, n_windows = embeddings.shape[:2]
    if n_windows <= 2 * min_shift:
        raise ValueError(
            f"{n_windows} windows are too few for circular shifts of at least "
            f"{min_shift} in both directions."
        )
    null = np.empty((n_null, n_windows))
    rows = np.arange(n_subjects)[:, None]
    for i in range(n_null):
        lags = rng.integers(min_shift, n_windows - min_shift + 1, size=n_subjects)
        columns = (np.arange(n_windows)[None, :] - lags[:, None]) % n_windows
        null[i] = group_synchrony(embeddings[rows, columns])
    return null


def score_track(
    embeddings: np.ndarray,
    starts_s: np.ndarray,
    window_s: float,
    *,
    n_null: int,
    null_quantile: float,
    min_shift: int,
    seed: int,
) -> pd.DataFrame:
    """Synchrony of every window of one track, with its null comparison.

    :param embeddings: ``(subjects, windows, dim)``.
    :param starts_s: ``(windows,)`` window start times in seconds.
    :param window_s: Window length in seconds.
    :return: One row per window: ``start_s``, ``end_s``, ``synchrony``, ``z`` (against
        the pooled null), ``null_threshold`` and ``candidate`` (above the threshold).
    """
    synchrony = group_synchrony(embeddings)
    null = shifted_null(embeddings, n_null, min_shift, np.random.default_rng(seed))
    threshold = float(np.quantile(null, null_quantile))
    return pd.DataFrame(
        {
            "start_s": starts_s,
            "end_s": starts_s + window_s,
            "synchrony": synchrony,
            "z": (synchrony - null.mean()) / max(float(null.std()), 1e-12),
            "null_threshold": threshold,
            "candidate": synchrony > threshold,
        }
    )


def candidate_intervals(scores: pd.DataFrame) -> pd.DataFrame:
    """Merge candidate windows whose time spans overlap or touch into intervals.

    Windows overlap whenever the stride is shorter than the window, so merging by span
    rather than by consecutive index keeps one non-candidate window from splitting what
    is in time a single interval.

    :param scores: One block's output of :func:`score_track`.
    :return: One row per interval: ``start_s``, ``end_s``, ``n_windows``,
        ``peak_synchrony``, ``peak_z``.
    """
    candidates = scores[scores["candidate"]].sort_values("start_s")
    end = candidates["end_s"].to_numpy()
    # A window opens a new interval when it starts after every earlier one has ended.
    reach = np.r_[-np.inf, np.maximum.accumulate(end)[:-1]]
    run_id = np.cumsum(candidates["start_s"].to_numpy() > reach)
    runs = candidates.groupby(run_id)
    return pd.DataFrame(
        {
            "start_s": runs["start_s"].min(),
            "end_s": runs["end_s"].max(),
            "n_windows": runs.size(),
            "peak_synchrony": runs["synchrony"].max(),
            "peak_z": runs["z"].max(),
        }
    ).reset_index(drop=True)
