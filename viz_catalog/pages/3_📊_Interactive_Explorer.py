"""Interactive Explorer — visualise precomputed CSV results interactively."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Interactive Explorer", page_icon="📊", layout="wide"
)

st.title("📊 Interactive Explorer")
st.markdown(
    "Browse and interactively visualise precomputed analysis results "
    "(CSV files) produced by the analysis pipeline.  "
    "Point the explorer at your local **results database** directory."
)

# ---------------------------------------------------------------------------
# Results-database scanning (self-contained, no src/ imports)
# ---------------------------------------------------------------------------

#: CSV basenames recognised as results-database entries.
#: NOTE: Must be kept in sync with ``_KNOWN_CSV_FILES`` in
#: ``src/analysis/results_store.py``.  Duplicated here because viz_catalog
#: must remain self-contained (no imports from ``src/``).
_KNOWN_CSV: frozenset[str] = frozenset(
    {
        "intersubject_timeseries.csv",
        "windowed_stats.csv",
        "loo_isc.csv",
        "pairwise_isc.csv",
    }
)


@st.cache_data(ttl=30)
def scan_results_db(directory: str) -> list[dict[str, str]]:
    """Scan *directory* for recognised CSV result files."""
    root = Path(directory)
    records: list[dict[str, str]] = []
    if not root.exists():
        return records
    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path.name not in _KNOWN_CSV:
            continue
        rel = csv_path.relative_to(root)
        parts = list(rel.parts)
        if len(parts) < 3:
            continue
        condition_music = parts[0]
        if "_" not in condition_music:
            continue
        spectrum_type = parts[1]
        if spectrum_type not in ("broadband", "bands"):
            continue
        condition, music_type = condition_music.split("_", 1)
        analysis_type = csv_path.stem
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


# ---------------------------------------------------------------------------
# Directory input
# ---------------------------------------------------------------------------
results_dir_str = st.text_input(
    "Results database directory",
    value="./results_db",
    help=(
        "Path to the results-database folder containing CSV files. "
        "The expected layout is: "
        "<Condition>_<MusicType>/<broadband|bands/band>/<analysis>.csv"
    ),
)
results_dir = Path(results_dir_str).expanduser().resolve()

if not results_dir.exists():
    st.warning(
        f"Directory **{results_dir}** does not exist. "
        "Please enter a valid path to your results database."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
with st.spinner("Scanning results database…"):
    records = scan_results_db(str(results_dir))

if not records:
    st.info(
        "No recognised CSV result files found. "
        "Expected files: `intersubject_timeseries.csv`, "
        "`windowed_stats.csv`, `loo_isc.csv`, `pairwise_isc.csv`."
    )
    st.stop()

st.success(f"Found **{len(records)}** result file(s) in `{results_dir}`.")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

all_conditions = sorted({r["condition"] for r in records})
all_music = sorted({r["music_type"] for r in records})
all_spectrum = sorted({r["spectrum_type"] for r in records})
all_bands = sorted({r["band"] for r in records})
all_analysis = sorted({r["analysis_type"] for r in records})

sel_conditions = st.sidebar.multiselect("Condition", all_conditions, default=[])
sel_music = st.sidebar.multiselect("Music type", all_music, default=[])
sel_spectrum = st.sidebar.multiselect("Spectrum type", all_spectrum, default=[])
sel_bands = st.sidebar.multiselect("Frequency band", all_bands, default=[])
sel_analysis = st.sidebar.multiselect("Analysis type", all_analysis, default=[])

any_filter = sel_conditions or sel_music or sel_spectrum or sel_bands or sel_analysis
if not any_filter:
    st.sidebar.markdown(f"**0** / {len(records)} results shown")
    st.info("👆 Select at least one filter in the sidebar to explore results.")
    st.stop()

filtered = [
    r
    for r in records
    if (not sel_conditions or r["condition"] in sel_conditions)
    and (not sel_music or r["music_type"] in sel_music)
    and (not sel_spectrum or r["spectrum_type"] in sel_spectrum)
    and (not sel_bands or r["band"] in sel_bands)
    and (not sel_analysis or r["analysis_type"] in sel_analysis)
]

st.sidebar.markdown(f"**{len(filtered)}** / {len(records)} results shown")

if not filtered:
    st.warning("No results match the current filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Visualisation tools — each analysis_type gets a dedicated renderer
# ---------------------------------------------------------------------------


def _label(rec: dict[str, str]) -> str:
    """Build a human-readable label for a record."""
    parts = [rec["condition"], rec["music_type"]]
    if rec["band"] != "broadband":
        parts.append(rec["band"])
    return " / ".join(parts)


def _off_diagonal(mat: np.ndarray) -> np.ndarray:
    """Return the off-diagonal elements of a square matrix."""
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def viz_intersubject_timeseries(rec: dict[str, str]) -> None:
    """Interactive time-series plot: mean signal ± std envelope."""
    df = pd.read_csv(rec["path"])
    label = _label(rec)

    st.subheader(f"Intersubject Time-Series — {label}")

    # Allow the user to downsample for faster rendering
    max_points = st.slider(
        "Max display points",
        min_value=200,
        max_value=len(df),
        value=min(2000, len(df)),
        step=100,
        key=f"ts_pts_{rec['path']}",
    )
    step = max(1, len(df) // max_points)
    ds = df.iloc[::step].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean signal (channel-averaged)**")
        chart_df = ds[["time", "mean_signal"]].set_index("time")
        st.line_chart(chart_df, use_container_width=True)

    with col2:
        st.markdown("**Intersubject variance**")
        chart_df = ds[["time", "variance"]].set_index("time")
        st.area_chart(chart_df, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(df, use_container_width=True)


def viz_windowed_stats(rec: dict[str, str]) -> None:
    """Interactive windowed statistics: bar chart + sync-candidate markers."""
    df = pd.read_csv(rec["path"])
    label = _label(rec)

    st.subheader(f"Windowed Statistics — {label}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean variance per window**")
        chart_df = df[["center", "mean_variance"]].rename(
            columns={"center": "time (s)"}
        )
        st.bar_chart(chart_df.set_index("time (s)"), use_container_width=True)

    with col2:
        st.markdown("**Variance of signal per window**")
        chart_df = df[["center", "var_signal"]].rename(
            columns={"center": "time (s)"}
        )
        st.bar_chart(chart_df.set_index("time (s)"), use_container_width=True)

    # Synchrony candidate summary
    if "sync_candidate" in df.columns:
        n_sync = int(df["sync_candidate"].sum())
        st.metric("Synchrony candidate windows", n_sync, f"/ {len(df)} total")

    with st.expander("Raw data table"):
        st.dataframe(df, use_container_width=True)


def viz_loo_isc(rec: dict[str, str]) -> None:
    """Interactive LOO-ISC distribution: histogram + per-subject view."""
    df = pd.read_csv(rec["path"])
    label = _label(rec)

    st.subheader(f"LOO-ISC Distribution — {label}")

    per_subject = df[df["subject"] >= 0].copy()
    means = df[df["subject"] == -1].copy()

    # Summary metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Mean ISC (all)", f"{per_subject['isc'].mean():.4f}")
    with col_m2:
        st.metric("Median ISC", f"{per_subject['isc'].median():.4f}")
    with col_m3:
        st.metric(
            "Std ISC",
            f"{per_subject['isc'].std():.4f}",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ISC distribution (all subject × channel pairs)**")
        # Use numpy to create histogram data for st.bar_chart
        values = per_subject["isc"].values
        n_bins = st.slider(
            "Number of bins",
            min_value=10,
            max_value=100,
            value=30,
            key=f"loo_bins_{rec['path']}",
        )
        counts, edges = np.histogram(values, bins=n_bins)
        centres = (edges[:-1] + edges[1:]) / 2
        hist_df = pd.DataFrame({"ISC": centres, "count": counts}).set_index("ISC")
        st.bar_chart(hist_df, use_container_width=True)

    with col2:
        st.markdown("**Mean ISC per channel**")
        if not means.empty:
            chart_df = means[["channel", "isc"]].copy()
            chart_df = chart_df.rename(columns={"isc": "mean ISC"})
            chart_df = chart_df.set_index("channel")
            st.bar_chart(chart_df, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(df, use_container_width=True)


def viz_pairwise_isc(rec: dict[str, str]) -> None:
    """Interactive pairwise ISC: heatmap-like table + distribution."""
    df = pd.read_csv(rec["path"])
    label = _label(rec)

    st.subheader(f"Pairwise ISC — {label}")

    # Reconstruct the matrix
    max_idx = max(df["subject_i"].max(), df["subject_j"].max())
    n = int(max_idx) + 1
    mat = np.zeros((n, n))
    for _, row in df.iterrows():
        i, j = int(row["subject_i"]), int(row["subject_j"])
        mat[i, j] = row["isc"]
        mat[j, i] = row["isc"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pairwise ISC matrix**")
        mat_df = pd.DataFrame(
            mat,
            index=[f"S{i}" for i in range(n)],
            columns=[f"S{i}" for i in range(n)],
        )
        st.dataframe(
            mat_df.style.background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.5),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Off-diagonal ISC distribution**")
        off_diag = _off_diagonal(mat)
        n_bins = st.slider(
            "Number of bins",
            min_value=5,
            max_value=50,
            value=15,
            key=f"pair_bins_{rec['path']}",
        )
        counts, edges = np.histogram(off_diag, bins=n_bins)
        centres = (edges[:-1] + edges[1:]) / 2
        hist_df = pd.DataFrame({"ISC": centres, "count": counts}).set_index("ISC")
        st.bar_chart(hist_df, use_container_width=True)

    # Summary
    off_diag = _off_diagonal(mat)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Mean off-diagonal ISC", f"{off_diag.mean():.4f}")
    with col_m2:
        st.metric("Std off-diagonal ISC", f"{off_diag.std():.4f}")

    with st.expander("Raw data table"):
        st.dataframe(df, use_container_width=True)


# Map analysis_type → renderer
_VIZ_MAP: dict[str, Callable[[dict[str, str]], None]] = {
    "intersubject_timeseries": viz_intersubject_timeseries,
    "windowed_stats": viz_windowed_stats,
    "loo_isc": viz_loo_isc,
    "pairwise_isc": viz_pairwise_isc,
}

# ---------------------------------------------------------------------------
# Render selected results
# ---------------------------------------------------------------------------
for rec in filtered:
    with st.container(border=True):
        renderer = _VIZ_MAP.get(rec["analysis_type"])
        if renderer is not None:
            renderer(rec)
        else:
            st.warning(
                f"No interactive visualisation available for "
                f"analysis type **{rec['analysis_type']}**."
            )
            st.json(rec)
