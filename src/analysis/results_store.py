"""
Results store — export precomputed analysis results to CSV for interactive
visualisation.

This module provides functions to save analysis results (intersubject
statistics, windowed statistics, LOO-ISC scores, pairwise ISC matrices) as
CSV files under a canonical directory layout.  The resulting *results database*
is consumed by the **Interactive Explorer** page in the ``viz_catalog/``
Streamlit app, which renders interactive plots directly from the stored data.

Canonical directory structure::

    <results_root>/
    └── <Condition>_<MusicType>/
        ├── broadband/
        │   ├── intersubject_timeseries.csv
        │   ├── windowed_stats.csv
        │   ├── loo_isc.csv
        │   └── pairwise_isc.csv
        └── bands/
            └── <band>/
                ├── intersubject_timeseries.csv
                ├── windowed_stats.csv
                ├── loo_isc.csv
                └── pairwise_isc.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

__all__ = [
    "save_intersubject_timeseries",
    "save_windowed_stats",
    "save_loo_isc",
    "save_pairwise_isc",
    "load_intersubject_timeseries",
    "load_windowed_stats",
    "load_loo_isc",
    "load_pairwise_isc",
    "scan_results_db",
]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Create parent directories for *path* if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_intersubject_timeseries(
    stats: dict[str, np.ndarray],
    sfreq: float,
    out_dir: Path,
    *,
    condition: str = "",
    music_type: str = "",
    band: str = "broadband",
) -> Path:
    """Save intersubject mean and variance time-series to CSV.

    :param stats: Output of
        :func:`~src.analysis.mean_variance.compute_intersubject_stats`.
    :param sfreq: Sampling frequency (Hz).
    :param out_dir: Directory in which to create the CSV.
    :param condition: Condition label (e.g. ``"Placebo"``).
    :param music_type: Music-type label (e.g. ``"CLASSIC"``).
    :param band: Frequency band name or ``"broadband"``.
    :return: Path to the written CSV file.
    """
    n_times = stats["mean_t"].shape[0]
    time = np.arange(n_times) / sfreq
    df = pd.DataFrame(
        {
            "time": time,
            "mean_signal": stats["mean_t"],
            "variance": stats["var_t"],
            "std": stats["std_t"],
        }
    )
    df["condition"] = condition
    df["music_type"] = music_type
    df["band"] = band

    dest = out_dir / "intersubject_timeseries.csv"
    _ensure_dir(dest)
    df.to_csv(dest, index=False)
    _logger.info("Saved intersubject timeseries → %s", dest)
    return dest


def save_windowed_stats(
    windowed_df: pd.DataFrame,
    out_dir: Path,
    *,
    condition: str = "",
    music_type: str = "",
    band: str = "broadband",
) -> Path:
    """Save windowed statistics DataFrame to CSV.

    :param windowed_df: Output of
        :func:`~src.analysis.mean_variance.compute_windowed_stats`.
    :param out_dir: Target directory.
    :param condition: Condition label.
    :param music_type: Music-type label.
    :param band: Frequency band name or ``"broadband"``.
    :return: Path to the written CSV file.
    """
    df = windowed_df.copy()
    df["condition"] = condition
    df["music_type"] = music_type
    df["band"] = band

    dest = out_dir / "windowed_stats.csv"
    _ensure_dir(dest)
    df.to_csv(dest, index=False)
    _logger.info("Saved windowed stats → %s", dest)
    return dest


def save_loo_isc(
    loo_isc: np.ndarray,
    mean_isc: np.ndarray,
    out_dir: Path,
    *,
    condition: str = "",
    music_type: str = "",
    band: str = "broadband",
    method: str = "pearson",
) -> Path:
    """Save leave-one-out ISC scores to CSV.

    :param loo_isc: ``(n_subjects, n_channels)`` LOO-ISC array.
    :param mean_isc: ``(n_channels,)`` mean LOO-ISC across subjects.
    :param out_dir: Target directory.
    :param condition: Condition label.
    :param music_type: Music-type label.
    :param band: Frequency band name or ``"broadband"``.
    :param method: Correlation method (``"pearson"`` or ``"spearman"``).
    :return: Path to the written CSV file.
    """
    n_subjects, n_channels = loo_isc.shape
    records: list[dict] = []
    for s in range(n_subjects):
        for ch in range(n_channels):
            records.append(
                {
                    "subject": s,
                    "channel": ch,
                    "isc": float(loo_isc[s, ch]),
                }
            )
    # Append per-channel means as a summary row (subject = -1)
    for ch in range(n_channels):
        records.append(
            {
                "subject": -1,
                "channel": ch,
                "isc": float(mean_isc[ch]),
            }
        )

    df = pd.DataFrame(records)
    df["condition"] = condition
    df["music_type"] = music_type
    df["band"] = band
    df["method"] = method

    dest = out_dir / "loo_isc.csv"
    _ensure_dir(dest)
    df.to_csv(dest, index=False)
    _logger.info("Saved LOO-ISC (%s) → %s", method, dest)
    return dest


def save_pairwise_isc(
    matrix: np.ndarray,
    out_dir: Path,
    *,
    condition: str = "",
    music_type: str = "",
    band: str = "broadband",
) -> Path:
    """Save a pairwise ISC matrix to CSV.

    :param matrix: ``(n_subjects, n_subjects)`` symmetric ISC matrix.
    :param out_dir: Target directory.
    :param condition: Condition label.
    :param music_type: Music-type label.
    :param band: Frequency band name or ``"broadband"``.
    :return: Path to the written CSV file.
    """
    n = matrix.shape[0]
    records: list[dict] = []
    for i in range(n):
        for j in range(i, n):
            records.append(
                {
                    "subject_i": i,
                    "subject_j": j,
                    "isc": float(matrix[i, j]),
                }
            )

    df = pd.DataFrame(records)
    df["condition"] = condition
    df["music_type"] = music_type
    df["band"] = band

    dest = out_dir / "pairwise_isc.csv"
    _ensure_dir(dest)
    df.to_csv(dest, index=False)
    _logger.info("Saved pairwise ISC → %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_intersubject_timeseries(csv_path: Path) -> pd.DataFrame:
    """Load an intersubject timeseries CSV into a DataFrame.

    :param csv_path: Path to ``intersubject_timeseries.csv``.
    :return: DataFrame with columns ``time``, ``mean_signal``, ``variance``,
        ``std``, ``condition``, ``music_type``, ``band``.
    """
    return pd.read_csv(csv_path)


def load_windowed_stats(csv_path: Path) -> pd.DataFrame:
    """Load a windowed-stats CSV into a DataFrame.

    :param csv_path: Path to ``windowed_stats.csv``.
    :return: DataFrame with windowed stat columns plus metadata.
    """
    return pd.read_csv(csv_path)


def load_loo_isc(csv_path: Path) -> pd.DataFrame:
    """Load a LOO-ISC CSV into a DataFrame.

    :param csv_path: Path to ``loo_isc.csv``.
    :return: DataFrame with columns ``subject``, ``channel``, ``isc``,
        ``condition``, ``music_type``, ``band``, ``method``.
    """
    return pd.read_csv(csv_path)


def load_pairwise_isc(csv_path: Path) -> pd.DataFrame:
    """Load a pairwise-ISC CSV into a DataFrame.

    :param csv_path: Path to ``pairwise_isc.csv``.
    :return: DataFrame with columns ``subject_i``, ``subject_j``, ``isc``,
        ``condition``, ``music_type``, ``band``.
    """
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

#: CSV basenames that are recognized as results-database entries.
_KNOWN_CSV_FILES: frozenset[str] = frozenset(
    {
        "intersubject_timeseries.csv",
        "windowed_stats.csv",
        "loo_isc.csv",
        "pairwise_isc.csv",
    }
)


def scan_results_db(root: Path) -> list[dict[str, str]]:
    """Scan *root* for CSV result files and return a metadata list.

    Each record contains:

    - ``path`` — absolute path to the CSV file.
    - ``filename`` — CSV basename.
    - ``condition_music`` — e.g. ``"Placebo_CLASSIC"``.
    - ``condition`` — e.g. ``"Placebo"``.
    - ``music_type`` — e.g. ``"CLASSIC"``.
    - ``spectrum_type`` — ``"broadband"`` or ``"bands"``.
    - ``band`` — band name or ``"broadband"``.
    - ``analysis_type`` — one of ``intersubject_timeseries``,
      ``windowed_stats``, ``loo_isc``, ``pairwise_isc``.

    :param root: Root directory of the results database.
    :return: List of metadata dicts, one per recognized CSV file.
    """
    records: list[dict[str, str]] = []
    if not root.exists():
        return records

    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path.name not in _KNOWN_CSV_FILES:
            continue

        rel = csv_path.relative_to(root)
        parts = list(rel.parts)

        # Expected layouts:
        #   <Condition>_<Music>/broadband/<file.csv>          → 3 parts
        #   <Condition>_<Music>/bands/<band>/<file.csv>       → 4 parts
        if len(parts) < 3:
            continue
        condition_music = parts[0]
        if "_" not in condition_music:
            continue

        spectrum_type = parts[1]
        if spectrum_type not in ("broadband", "bands"):
            continue

        condition, music_type = condition_music.split("_", 1)
        analysis_type = csv_path.stem  # e.g. "loo_isc"

        if spectrum_type == "broadband":
            band = "broadband"
        elif len(parts) >= 4:
            band = parts[2]
        else:
            continue

        records.append(
            {
                "path": str(csv_path),
                "filename": csv_path.name,
                "condition_music": condition_music,
                "condition": condition,
                "music_type": music_type,
                "spectrum_type": spectrum_type,
                "band": band,
                "analysis_type": analysis_type,
            }
        )

    return records
