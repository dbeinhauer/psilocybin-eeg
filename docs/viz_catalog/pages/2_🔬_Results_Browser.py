"""Results Browser page — scan a local results directory and browse/compare plots."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

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


@st.cache_data
def scan_images(directory: str) -> list[dict]:
    """Recursively scan *directory* for image files and return a metadata list."""
    root = Path(directory)
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        parts = list(rel.parts)  # e.g. ["01-mean-variance", "Placebo_CLASSIC", "raw", "foo.png"]
        record = {
            "path": str(path),
            "relative": str(rel),
            "filename": path.name,
            "stem": path.stem,
            # Heuristic token extraction from directory structure
            "stage": parts[0] if len(parts) >= 2 else "",
            "condition_music": parts[1] if len(parts) >= 3 else "",
            "subdir": str(Path(*parts[2:-1])) if len(parts) >= 4 else "",
        }
        # Try to split condition_music into condition + music_type
        cm = record["condition_music"]
        if "_" in cm:
            split = cm.split("_", 1)
            record["condition"] = split[0]
            record["music_type"] = split[1]
        else:
            record["condition"] = cm
            record["music_type"] = ""
        records.append(record)
    return records


with st.spinner("Scanning directory…"):
    images = scan_images(str(results_dir))

if not images:
    st.info("No PNG / JPG files found in the selected directory.")
    st.stop()

st.success(f"Found **{len(images)}** image(s) in `{results_dir}`.")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

all_stages = sorted({r["stage"] for r in images if r["stage"]})
all_conditions = sorted({r["condition"] for r in images if r["condition"]})
all_music_types = sorted({r["music_type"] for r in images if r["music_type"]})

sel_stages = st.sidebar.multiselect("Analysis stage", all_stages, default=all_stages)
sel_conditions = st.sidebar.multiselect("Condition", all_conditions, default=all_conditions)
sel_music = st.sidebar.multiselect("Music type", all_music_types, default=all_music_types)
free_text = st.sidebar.text_input("Filename contains", "")

# Apply filters
filtered = [
    r
    for r in images
    if (not sel_stages or r["stage"] in sel_stages)
    and (not sel_conditions or r["condition"] in sel_conditions)
    and (not sel_music or r["music_type"] in sel_music)
    and (not free_text or free_text.lower() in r["filename"].lower())
]

st.sidebar.markdown(f"**{len(filtered)}** / {len(images)} images shown")

if not filtered:
    st.warning("No images match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# View mode + Compare mode toggle
# ---------------------------------------------------------------------------
st.sidebar.header("Display")
view_mode = st.sidebar.radio("View mode", ["Grid", "List"], horizontal=True)
compare_mode = st.sidebar.checkbox("Compare mode (select 2–4 images)")

# ---------------------------------------------------------------------------
# Compare mode — selection
# ---------------------------------------------------------------------------
if compare_mode:
    st.markdown("### Compare images")
    st.caption("Tick the checkboxes below to select 2–4 images to compare side by side.")

    selected_for_compare: list[str] = []
    n_cols = 4
    cols = st.columns(n_cols)
    for i, rec in enumerate(filtered):
        col = cols[i % n_cols]
        with col:
            checked = st.checkbox(rec["filename"], key=f"cmp_{rec['path']}", value=False)
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
else:
    # List view — full-width with metadata
    for rec in filtered:
        with st.container(border=True):
            col_img, col_meta = st.columns([2, 3])
            with col_img:
                st.image(rec["path"], use_container_width=True)
            with col_meta:
                st.markdown(f"**{rec['filename']}**")
                st.markdown(f"- **Path**: `{rec['relative']}`")
                if rec["stage"]:
                    st.markdown(f"- **Stage**: `{rec['stage']}`")
                if rec["condition"]:
                    st.markdown(f"- **Condition**: `{rec['condition']}`")
                if rec["music_type"]:
                    st.markdown(f"- **Music type**: `{rec['music_type']}`")
                if rec["subdir"]:
                    st.markdown(f"- **Subdir**: `{rec['subdir']}`")
                file_size = os.path.getsize(rec["path"])
                st.markdown(f"- **Size**: {file_size / 1024:.1f} KB")
