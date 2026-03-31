"""Results Browser page — scan a local results directory and browse/compare plots."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Results Browser", page_icon="🔬", layout="wide")

st.title("🔬 Results Browser")
st.markdown(
    "Browse actual plot files produced by the analysis pipeline. "
    "Enter the path to your local results folder below."
)

# ---------------------------------------------------------------------------
# Catalog loading — used for info boxes in each result image
# ---------------------------------------------------------------------------
_CATALOG_PATH = Path(__file__).parent.parent / "catalog.yaml"
GITHUB_BASE = "https://github.com/dbeinhauer/psilocybin-eeg/blob/develop"


@st.cache_data
def load_catalog() -> dict:
    """Load catalog.yaml for info-box lookup."""
    if not _CATALOG_PATH.exists():
        return {}
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f)


_catalog = load_catalog()


def find_catalog_entry(record: dict) -> tuple[dict | None, dict | None]:
    """Return (analysis, plot) from catalog matching *record*, or (None, None).

    Matching strategy (highest priority first):
    1. Exact stage-directory → catalog-id match + filename pattern match.
    2. Numeric-prefix match + filename pattern match (handles minor name variations).
    3. Exact stage-directory → catalog-id match (stage-level fallback, no plot).
    4. First numeric-prefix match (coarse fallback, no plot).
    """
    stage = record.get("stage", "")
    filename = record.get("filename", "")

    if not stage:
        return None, None

    stage_num = stage.split("-")[0] if "-" in stage else stage

    exact_fallback: dict | None = None  # exact ID match but no plot pattern match
    prefix_fallback: dict | None = None  # first numeric-prefix match

    for analysis in _catalog.get("analyses", []):
        aid = analysis.get("id", "")
        aid_num = aid.split("-")[0] if "-" in aid else aid
        if aid_num != stage_num:
            continue

        exact_id = aid == stage  # full directory name matches catalog id

        # Try to match filename against each plot's filename_pattern
        for plot in analysis.get("plots", []):
            pattern = plot.get("filename_pattern", "")
            if pattern and fnmatch.fnmatch(filename, pattern):
                return analysis, plot

        # Track fallbacks: prefer exact ID match over coarse numeric match
        if exact_id:
            exact_fallback = analysis
        elif prefix_fallback is None:
            prefix_fallback = analysis

    return (exact_fallback or prefix_fallback), None


# ---------------------------------------------------------------------------
# Band keyword extraction — whole-token matching to avoid false positives
# ---------------------------------------------------------------------------
_SPECTRUM_TYPES: frozenset[str] = frozenset({"broadband", "bands"})

_BAND_CANONICAL: dict[str, str] = {
    "delta": "delta",
    "theta": "theta",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "δ": "delta",
    "θ": "theta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
}
_TOKEN_SEP = re.compile(r"[_\-\s.]+")


def _extract_band(text: str) -> str:
    """Return the canonical band name found as a whole token in *text*, or '' if none."""
    for token in _TOKEN_SEP.split(text.lower()):
        if token in _BAND_CANONICAL:
            return _BAND_CANONICAL[token]
    return ""


# ---------------------------------------------------------------------------
# Results directory input
# ---------------------------------------------------------------------------
results_dir_str = st.text_input(
    "Results directory",
    value="./results",
    help="Absolute or relative path to the folder containing .png / .jpg plot files.",
)
results_dir = Path(results_dir_str).expanduser().resolve()

if not results_dir.exists():
    st.warning(
        f"Directory **{results_dir}** does not exist. "
        "Please enter a valid path to your results folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Scan for image files
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@st.cache_data(ttl=30)
def scan_images(directory: str) -> list[dict]:
    """Recursively scan *directory* for image files and return a metadata list.

    Only files matching the canonical pipeline output structure are included::

        <stage>/<Condition>_<MusicType>/broadband/<analysis_type>/<filename.ext>
        <stage>/<Condition>_<MusicType>/bands/<analysis_type>/<filename.ext>

    Files that do not conform to this layout (wrong depth, missing
    ``<Condition>_<MusicType>`` separator, or unknown spectrum type) are
    silently skipped.
    """
    root = Path(directory)
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        parts = list(rel.parts)

        # Require exactly: stage / condition_music / spectrum_type / analysis_type / filename
        if len(parts) < 5:
            continue
        # Require the second token to look like <Condition>_<MusicType>
        if "_" not in parts[1]:
            continue
        # Require the third token to be a known spectrum type
        if parts[2] not in _SPECTRUM_TYPES:
            continue

        stage = parts[0]
        condition_music = parts[1]
        spectrum_type = parts[2]  # "broadband" or "bands"
        analysis_type = parts[3]  # e.g. "loo_isc", "pairwise_isc", "timeseries"

        condition, music_type = condition_music.split("_", 1)

        # Band: all broadband files map to "broadband"; per-band files get
        # the band extracted from the filename stem.
        if spectrum_type == "broadband":
            band = "broadband"
        else:
            band = _extract_band(path.stem)

        records.append(
            {
                "path": str(path),
                "relative": str(rel),
                "filename": path.name,
                "stem": path.stem,
                "stage": stage,
                "condition_music": condition_music,
                "condition": condition,
                "music_type": music_type,
                "spectrum_type": spectrum_type,
                "analysis_type": analysis_type,
                "band": band,
            }
        )
    return records


with st.spinner("Scanning directory…"):
    images = scan_images(str(results_dir))

if not images:
    st.info("No PNG / JPG files found matching the expected directory structure.")
    st.stop()

st.success(f"Found **{len(images)}** image(s) in `{results_dir}`.")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

all_stages = sorted({r["stage"] for r in images if r["stage"]})
all_conditions = sorted({r["condition"] for r in images if r["condition"]})
all_music_types = sorted({r["music_type"] for r in images if r["music_type"]})
all_spectrum_types = sorted({r["spectrum_type"] for r in images if r["spectrum_type"]})
all_analysis_types = sorted({r["analysis_type"] for r in images if r["analysis_type"]})
all_bands = sorted({r["band"] for r in images if r["band"]})

# All filters default to empty — nothing is shown until the user selects something.
sel_stages = st.sidebar.multiselect("Analysis stage", all_stages, default=[])
sel_conditions = st.sidebar.multiselect("Condition", all_conditions, default=[])
sel_music = st.sidebar.multiselect("Music type", all_music_types, default=[])
sel_spectrum = st.sidebar.multiselect("Spectrum type", all_spectrum_types, default=[])
sel_analysis_types = st.sidebar.multiselect(
    "Analysis type", all_analysis_types, default=[]
)
if all_bands:
    sel_bands = st.sidebar.multiselect("Frequency band", all_bands, default=[])
else:
    sel_bands = []
free_text = st.sidebar.text_input("Filename contains", "")

# Guard: show nothing until at least one filter is active
any_filter_active = (
    sel_stages
    or sel_conditions
    or sel_music
    or sel_spectrum
    or sel_analysis_types
    or sel_bands
    or free_text
)
if not any_filter_active:
    st.sidebar.markdown(f"**0** / {len(images)} images shown")
    st.info("👆 Select at least one filter in the sidebar to start browsing images.")
    st.stop()

# Apply filters
filtered = [
    r
    for r in images
    if (not sel_stages or r["stage"] in sel_stages)
    and (not sel_conditions or r["condition"] in sel_conditions)
    and (not sel_music or r["music_type"] in sel_music)
    and (not sel_spectrum or r["spectrum_type"] in sel_spectrum)
    and (not sel_analysis_types or r["analysis_type"] in sel_analysis_types)
    and (not sel_bands or r["band"] in sel_bands)
    and (not free_text or free_text.lower() in r["filename"].lower())
]

st.sidebar.markdown(f"**{len(filtered)}** / {len(images)} images shown")

if not filtered:
    st.warning("No images match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# View mode + Compare mode
# ---------------------------------------------------------------------------
st.sidebar.header("Display")
view_mode = st.sidebar.radio("View mode", ["Grid", "List"], horizontal=True)
compare_mode = st.sidebar.radio(
    "Compare mode",
    ["None", "Manual (select 2–4)", "By condition / music type"],
    index=0,
)


# ---------------------------------------------------------------------------
# Helper — render info expander for one image record
# ---------------------------------------------------------------------------
def render_info(rec: dict) -> None:
    """Show a collapsed expander with catalog summary and links for *rec*."""
    analysis, plot = find_catalog_entry(rec)
    with st.expander("ℹ️ About this plot"):
        if analysis:
            st.markdown(f"**Stage:** {analysis['title']}")
            desc = analysis.get("description", "").strip()
            if desc:
                st.caption(desc)
        if plot:
            st.markdown(f"**Plot:** {plot['title']}")
            interp = plot.get("interpretation", "").strip()
            if interp:
                st.info(f"💡 {interp}")
            nb = plot.get("notebook", "")
            if nb:
                st.markdown(f"📓 [Open notebook on GitHub]({GITHUB_BASE}/{nb})")
        elif not analysis:
            st.caption("No catalog entry found for this image.")
        st.page_link(
            "pages/1_📋_Catalog.py",
            label="📋 Open Analysis Catalog for full details",
            icon="📋",
        )


# ---------------------------------------------------------------------------
# Compare mode — By condition / music type
# ---------------------------------------------------------------------------
if compare_mode == "By condition / music type":
    st.markdown("### Compare by condition / music type")
    st.caption(
        "Select a plot by its filename stem. The browser will find every "
        "condition / music-type variant of that plot and show them side by side."
    )

    all_stems = sorted({r["stem"] for r in filtered})
    ref_stem = st.selectbox("Reference plot (filename without extension)", all_stems)

    matches = [r for r in filtered if r["stem"] == ref_stem]
    if not matches:
        st.warning("No images found for the selected reference.")
    else:
        # Group by condition_music label; keep one representative per group
        groups: dict[str, dict] = {}
        for r in matches:
            lbl = (
                f"{r['condition']}_{r['music_type']}"
                if (r["condition"] or r["music_type"])
                else r["relative"]
            )
            groups[lbl] = r

        n_groups = len(groups)
        if n_groups < 2:
            st.info(
                "Only one condition / music-type found for this plot. "
                "Adjust the sidebar filters or choose a different plot."
            )
        cols = st.columns(n_groups)
        for col, (lbl, rec) in zip(cols, sorted(groups.items())):
            with col:
                st.markdown(f"**{lbl}**")
                st.image(rec["path"], use_container_width=True)
                render_info(rec)
    st.stop()

# ---------------------------------------------------------------------------
# Compare mode — Manual (select 2–4)
# ---------------------------------------------------------------------------
if compare_mode == "Manual (select 2–4)":
    st.markdown("### Compare images")
    st.caption(
        "Tick the checkboxes below to select 2–4 images to compare side by side."
    )

    selected_for_compare: list[str] = []
    n_cols = 4
    cols = st.columns(n_cols)
    for i, rec in enumerate(filtered):
        col = cols[i % n_cols]
        with col:
            checked = st.checkbox(
                rec["filename"], key=f"cmp_{rec['path']}", value=False
            )
            if checked:
                selected_for_compare.append(rec["path"])
            st.image(rec["path"], use_container_width=True)

    if 2 <= len(selected_for_compare) <= 4:
        st.divider()
        st.subheader("Side-by-side comparison")
        cmp_cols = st.columns(len(selected_for_compare))
        for col, img_path in zip(cmp_cols, selected_for_compare):
            with col:
                st.image(img_path, use_container_width=True)
                st.caption(Path(img_path).name)
    elif len(selected_for_compare) > 4:
        st.warning("Select at most 4 images for comparison.")
    st.stop()

# ---------------------------------------------------------------------------
# Normal view — grid or list
# ---------------------------------------------------------------------------
if view_mode == "Grid":
    n_cols = st.sidebar.slider("Columns", min_value=1, max_value=6, value=3)
    cols = st.columns(n_cols)
    for i, rec in enumerate(filtered):
        col = cols[i % n_cols]
        with col:
            st.image(rec["path"], use_container_width=True)
            st.caption(rec["relative"])
            render_info(rec)
else:
    # List view — full-width with metadata
    for rec in filtered:
        with st.container(border=True):
            col_img, col_meta = st.columns([2, 3])
            with col_img:
                st.image(rec["path"], use_container_width=True)
                render_info(rec)
            with col_meta:
                st.markdown(f"**{rec['filename']}**")
                st.markdown(f"- **Path**: `{rec['relative']}`")
                if rec["stage"]:
                    st.markdown(f"- **Stage**: `{rec['stage']}`")
                if rec["condition"]:
                    st.markdown(f"- **Condition**: `{rec['condition']}`")
                if rec["music_type"]:
                    st.markdown(f"- **Music type**: `{rec['music_type']}`")
                if rec["spectrum_type"]:
                    st.markdown(f"- **Spectrum type**: `{rec['spectrum_type']}`")
                if rec["analysis_type"]:
                    st.markdown(f"- **Analysis type**: `{rec['analysis_type']}`")
                if rec.get("band"):
                    st.markdown(f"- **Frequency band**: `{rec['band']}`")
                file_size = os.path.getsize(rec["path"])
                st.markdown(f"- **Size**: {file_size / 1024:.1f} KB")
