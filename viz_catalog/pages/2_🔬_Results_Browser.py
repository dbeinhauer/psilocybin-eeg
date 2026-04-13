"""Results Browser page — scan a local results directory and browse/compare plots."""

from __future__ import annotations

import base64
import fnmatch
import io
import os
import re
from pathlib import Path

from PIL import Image
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
# Catalog loading — used for info boxes and human-readable labels
# ---------------------------------------------------------------------------
_CATALOG_PATH = Path(__file__).parent.parent / "catalog.yaml"
GITHUB_BASE = "https://github.com/dbeinhauer/psilocybin-eeg/blob/develop"


@st.cache_data
def load_catalog() -> dict:
    """Load catalog.yaml for info-box lookup and analysis-type label mapping."""
    if not _CATALOG_PATH.exists():
        return {}
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f)


_catalog = load_catalog()


def build_analysis_type_label_map(catalog: dict) -> dict[str, str]:
    """Return a flat {slug: display_name} mapping from all analyses in catalog.

    Each analysis entry may carry an ``analysis_type_labels`` dict that maps
    directory-name slugs (e.g. ``loo_isc``) to human-readable display names
    (e.g. ``"Leave-one-out ISC"``).  Labels from later entries override
    earlier ones for the same slug (last writer wins).
    """
    mapping: dict[str, str] = {}
    for analysis in catalog.get("analyses", []):
        for slug, label in analysis.get("analysis_type_labels", {}).items():
            mapping[slug] = label
    return mapping


_analysis_type_labels: dict[str, str] = build_analysis_type_label_map(_catalog)


def build_catalog_sort_key_map(
    catalog: dict,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Return a {(stage_id, analysis_type_slug): (analysis_idx, type_idx)} map.

    Used to sort Results Browser sections in the same order as catalog.yaml.
    """
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for i, analysis in enumerate(catalog.get("analyses", [])):
        aid = analysis.get("id", "")
        for j, slug in enumerate(analysis.get("analysis_type_labels", {})):
            result[(aid, slug)] = (i, j)
    return result


_catalog_sort_key_map: dict[tuple[str, str], tuple[int, int]] = (
    build_catalog_sort_key_map(_catalog)
)
_CATALOG_LEN = len(_catalog.get("analyses", []))


def _section_sort_key(stage: str, analysis_type: str) -> tuple[int, int]:
    """Return (analysis_idx, type_idx) for catalog-order sorting.

    Falls back to numeric-prefix matching when the full stage name does not
    appear in the catalog (e.g. minor directory-name variations).  Sections
    not found in the catalog are placed after all known entries.
    """
    if (stage, analysis_type) in _catalog_sort_key_map:
        return _catalog_sort_key_map[(stage, analysis_type)]
    # Numeric-prefix fallback: "02-isc-broadband" → prefix "02"
    stage_num = stage.split("-")[0] if "-" in stage else stage
    for (s, at), key in _catalog_sort_key_map.items():
        s_num = s.split("-")[0] if "-" in s else s
        if s_num == stage_num and at == analysis_type:
            return key
    return (_CATALOG_LEN, 0)


def slug_to_display(slug: str) -> str:
    """Return a human-readable display name for an analysis-type slug.

    Falls back to a title-cased version of the slug if no catalog entry exists.
    """
    return _analysis_type_labels.get(slug, slug.replace("_", " ").title())


_THUMBNAIL_MAX_PX = 600  # longest edge of the displayed thumbnail
_THUMBNAIL_QUALITY = 72  # JPEG quality for thumbnails


@st.cache_data(max_entries=400)
def _image_b64(path: str) -> tuple[str, str]:
    """Return (base64-encoded image data, mime type) for *path*, cached.

    Used for the link target (full-size PNG opened on click).
    """
    suffix = Path(path).suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return data, mime


@st.cache_data(max_entries=400)
def _thumbnail_b64(path: str) -> str:
    """Return base64-encoded JPEG thumbnail for *path*, cached.

    Downscales the image so its longest edge is at most *_THUMBNAIL_MAX_PX*,
    then encodes as JPEG at *_THUMBNAIL_QUALITY*.  Much smaller than the
    original, so the browser renders the grid quickly.
    """
    img = Image.open(path)
    img.thumbnail((_THUMBNAIL_MAX_PX, _THUMBNAIL_MAX_PX), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_THUMBNAIL_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def render_clickable_image(path: str, use_container_width: bool = True) -> None:
    """Render a thumbnail that opens the full-size image in a new tab on click.

    The displayed ``<img>`` uses a small JPEG thumbnail so the page renders
    quickly.  The ``<a href>`` embeds the original full-size file, which the
    browser opens only when the user actually clicks.
    """
    thumb_data = _thumbnail_b64(path)
    thumb_src = f"data:image/jpeg;base64,{thumb_data}"
    full_data, full_mime = _image_b64(path)
    full_src = f"data:{full_mime};base64,{full_data}"
    width_style = "width:100%;" if use_container_width else ""
    st.markdown(
        f'<a href="{full_src}" target="_blank">'
        f'<img src="{thumb_src}" style="{width_style}cursor:pointer;" />'
        f"</a>",
        unsafe_allow_html=True,
    )


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

# Stage-03 analysis_type directory names, classified by domain.
# Slugs in neither set default to "power".
_STAGE03_PHASE_SLUGS: frozenset[str] = frozenset(
    {
        "band_itpc_tc",
        "itpc_spectrum",
        "itpc_vs_isc",
        "phase_distribution",
        "phase_loo_isc",
        "tf_itpc_map",
        "topomap_phase_isc",
    }
)
# Slugs that belong to both domains (shown in both sections).
_STAGE03_BOTH_SLUGS: frozenset[str] = frozenset(
    {
        "power_phase_joint",
    }
)

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

        # virtual_stages: the display stage name(s) used in filters and section
        # headers.  For stage 03 we split into "03-wavelet-power" /
        # "03-wavelet-phase" (or both for joint analyses); other stages keep
        # the raw directory name.
        stage_num = stage.split("-")[0] if "-" in stage else stage
        if stage_num == "03":
            if analysis_type in _STAGE03_BOTH_SLUGS:
                virtual_stages: list[str] = ["03-wavelet-power", "03-wavelet-phase"]
            elif analysis_type in _STAGE03_PHASE_SLUGS:
                virtual_stages = ["03-wavelet-phase"]
            else:
                virtual_stages = ["03-wavelet-power"]
        else:
            virtual_stages = [stage]

        records.append(
            {
                "path": str(path),
                "relative": str(rel),
                "filename": path.name,
                "stem": path.stem,
                "stage": stage,
                "virtual_stages": virtual_stages,
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

all_stages = sorted({vs for r in images for vs in r["virtual_stages"]})
all_conditions = sorted({r["condition"] for r in images if r["condition"]})
all_music_types = sorted({r["music_type"] for r in images if r["music_type"]})
all_spectrum_types = sorted({r["spectrum_type"] for r in images if r["spectrum_type"]})
all_bands = sorted({r["band"] for r in images if r["band"]})

# All filters default to empty — nothing is shown until the user selects something.
sel_stages = st.sidebar.multiselect("Analysis stage", all_stages, default=[])
sel_conditions = st.sidebar.multiselect("Condition", all_conditions, default=[])
sel_music = st.sidebar.multiselect("Music type", all_music_types, default=[])
sel_spectrum = st.sidebar.multiselect("Spectrum type", all_spectrum_types, default=[])

# Build slug→label and label→slug mappings for the Analysis type filter.
# Options are restricted to analysis types available in the selected stage(s).
_stage_filtered = (
    images
    if not sel_stages
    else [r for r in images if any(vs in sel_stages for vs in r["virtual_stages"])]
)
_available_analysis_type_slugs = sorted(
    {r["analysis_type"] for r in _stage_filtered if r["analysis_type"]}
)
_slug_to_label: dict[str, str] = {
    s: slug_to_display(s) for s in _available_analysis_type_slugs
}
_label_to_slug: dict[str, str] = {v: k for k, v in _slug_to_label.items()}
_available_analysis_type_display = sorted(_slug_to_label.values())

sel_analysis_type_display = st.sidebar.multiselect(
    "Analysis type", _available_analysis_type_display, default=[]
)
# Convert display names back to slugs for filtering
sel_analysis_types = [_label_to_slug[lbl] for lbl in sel_analysis_type_display]

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
    or sel_analysis_type_display
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
    if (not sel_stages or any(vs in sel_stages for vs in r["virtual_stages"]))
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

# Grid column count — shown only in Grid + None mode, but defined here so it
# is available before the compare-mode early-returns below.
n_cols = st.sidebar.slider("Columns", min_value=1, max_value=6, value=3)


# ---------------------------------------------------------------------------
# Helper — render info expander for one image record
# ---------------------------------------------------------------------------
def render_info(rec: dict, stage_override: str | None = None) -> None:
    """Show a collapsed expander with the full catalog entry for *rec*.

    *stage_override* replaces ``rec["stage"]`` for catalog lookup; used when
    rendering a record inside a virtual-stage section (e.g. ``03-wavelet-power``).
    """
    lookup_rec = rec if stage_override is None else {**rec, "stage": stage_override}
    analysis, plot = find_catalog_entry(lookup_rec)
    with st.expander("ℹ️ About this plot"):
        if not analysis and not plot:
            st.caption("No catalog entry found for this image.")
            return

        # ── Analysis-level ────────────────────────────────────────────────
        if analysis:
            st.markdown(f"#### {analysis['title']}")
            desc = analysis.get("description", "").strip()
            if desc:
                st.markdown(desc)
            nb_list = analysis.get("notebooks", [])
            if nb_list:
                st.markdown("**Notebooks**")
                for nb in nb_list:
                    st.markdown(f"- [{nb['label']}]({GITHUB_BASE}/{nb['path']})")

        # ── Plot-level ────────────────────────────────────────────────────
        if plot:
            st.divider()
            st.markdown(f"**{plot['title']}**")

            section = plot.get("section", "")
            if section:
                st.caption(f"Section: {section}")

            plot_desc = plot.get("description", "").strip()
            if plot_desc:
                st.markdown(plot_desc)

            ds = plot.get("data_shape", {})
            if ds:
                st.markdown("**Data shape**")
                st.table(
                    {
                        "": ["Input", "Output"],
                        "Shape": [ds.get("input", ""), ds.get("output", "")],
                    }
                )

            ops = plot.get("operation_order", [])
            if ops:
                st.markdown("**Order of operations**")
                for i, op in enumerate(ops, 1):
                    st.markdown(f"{i}. {op}")

            interp = plot.get("interpretation", "").strip()
            if interp:
                st.info(f"💡 {interp}")

            nb_path = plot.get("notebook", "")
            if nb_path:
                st.markdown(f"📓 [Open notebook on GitHub]({GITHUB_BASE}/{nb_path})")

        # ── Link back to catalog (pre-selecting the right analysis) ───────
        if analysis:
            aid = analysis.get("id", "")
            st.markdown(f"[📋 Open in Analysis Catalog](/Catalog?analysis_id={aid})")


# ---------------------------------------------------------------------------
# Compare mode — By condition / music type
# ---------------------------------------------------------------------------
if compare_mode == "By condition / music type":
    st.markdown("### Compare by condition / music type")
    st.caption(
        "Select a plot by its canonical name. The browser will find every "
        "condition / music-type variant of that plot and show them side by side."
    )

    # Canonical stem: strip trailing _<condition>_<music_type>, _<music_type>,
    # or _<condition> suffixes so that e.g. "sw_isc_bar_beta_CLASSIC" and
    # "sw_isc_bar_beta_PSYTRANCE" both map to "sw_isc_bar_beta".
    _cm_suffixes: list[str] = (
        [f"_{c}_{m}" for c in all_conditions for m in all_music_types]
        + [f"_{m}" for m in all_music_types]
        + [f"_{c}" for c in all_conditions]
    )

    def _canonical(stem: str) -> str:
        for sfx in _cm_suffixes:
            if stem.endswith(sfx):
                return stem[: -len(sfx)]
        return stem

    all_canonical = sorted({_canonical(r["stem"]) for r in filtered})
    ref_canonical = st.selectbox(
        "Reference plot (filename without extension)", all_canonical
    )

    matches = [r for r in filtered if _canonical(r["stem"]) == ref_canonical]
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
                render_clickable_image(rec["path"])
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
    cmp_cols = st.columns(n_cols)
    for i, rec in enumerate(filtered):
        col = cmp_cols[i % n_cols]
        with col:
            checked = st.checkbox(
                rec["filename"], key=f"cmp_{rec['path']}", value=False
            )
            if checked:
                selected_for_compare.append(rec["path"])
            render_clickable_image(rec["path"])

    if 2 <= len(selected_for_compare) <= 4:
        st.divider()
        st.subheader("Side-by-side comparison")
        side_cols = st.columns(len(selected_for_compare))
        for col, img_path in zip(side_cols, selected_for_compare):
            with col:
                render_clickable_image(img_path)
                st.caption(Path(img_path).name)
    elif len(selected_for_compare) > 4:
        st.warning("Select at most 4 images for comparison.")
    st.stop()

# ---------------------------------------------------------------------------
# Normal view — results grouped by analysis type, then by condition/music/band
# ---------------------------------------------------------------------------
# Band sort: broadband always first, then alphabetical.
_BAND_SORT_FIRST = "broadband"


def _band_sort_key(band: str) -> tuple[int, str]:
    return (0, band) if band == _BAND_SORT_FIRST else (1, band)


# Group records by (virtual_stage, analysis_type), then sort by catalog order.
# Records with multiple virtual_stages (e.g. power_phase_joint) appear in each.
section_records: dict[tuple[str, str], list[dict]] = {}
for rec in filtered:
    active_vstages = (
        [vs for vs in rec["virtual_stages"] if vs in sel_stages]
        if sel_stages
        else rec["virtual_stages"]
    )
    for vs in active_vstages:
        key = (vs, rec["analysis_type"])
        section_records.setdefault(key, []).append(rec)

section_order: list[tuple[str, str]] = sorted(
    section_records,
    key=lambda k: _section_sort_key(k[0], k[1]),
)

for vstage, analysis_type in section_order:
    group = section_records[(vstage, analysis_type)]
    display_name = slug_to_display(analysis_type)

    # Section header — show human-readable analysis-type name.
    # Include stage prefix when multiple stages are present so the user can
    # tell sections apart at a glance.
    if len(section_order) > 1 and len({s for s, _ in section_order}) > 1:
        st.subheader(f"{vstage} — {display_name}")
    else:
        st.subheader(display_name)

    # Sub-group by (condition, music_type, band) and sort:
    # condition → music_type → band (broadband first, then alpha).
    subgroup_records: dict[tuple[str, str, str], list[dict]] = {}
    for rec in group:
        subkey = (rec["condition"], rec["music_type"], rec["band"])
        subgroup_records.setdefault(subkey, []).append(rec)

    sorted_subkeys = sorted(
        subgroup_records,
        key=lambda t: (t[0], t[1], _band_sort_key(t[2])),
    )

    for condition, music_type, band in sorted_subkeys:
        sub_records = subgroup_records[(condition, music_type, band)]

        # Subsection header
        parts: list[str] = []
        if condition:
            parts.append(f"Condition: **{condition}**")
        if music_type:
            parts.append(f"Music: **{music_type}**")
        if band:
            parts.append(f"Band: **{band}**")
        if parts:
            st.markdown("##### " + " | ".join(parts))

        if view_mode == "Grid":
            cols = st.columns(n_cols)
            for i, rec in enumerate(sub_records):
                col = cols[i % n_cols]
                with col:
                    render_clickable_image(rec["path"])
                    st.caption(rec["relative"])
                    render_info(rec, stage_override=vstage)
        else:
            # List view — full-width with metadata
            for rec in sub_records:
                with st.container(border=True):
                    col_img, col_meta = st.columns([2, 3])
                    with col_img:
                        render_clickable_image(rec["path"])
                        render_info(rec, stage_override=vstage)
                    with col_meta:
                        st.markdown(f"**{rec['filename']}**")
                        st.markdown(f"- **Path**: `{rec['relative']}`")
                        if vstage:
                            st.markdown(f"- **Stage**: `{vstage}`")
                        if rec["condition"]:
                            st.markdown(f"- **Condition**: `{rec['condition']}`")
                        if rec["music_type"]:
                            st.markdown(f"- **Music type**: `{rec['music_type']}`")
                        if rec["spectrum_type"]:
                            st.markdown(
                                f"- **Spectrum type**: `{rec['spectrum_type']}`"
                            )
                        if rec["analysis_type"]:
                            st.markdown(
                                f"- **Analysis type**: {slug_to_display(rec['analysis_type'])}"
                            )
                        if rec.get("band"):
                            st.markdown(f"- **Frequency band**: `{rec['band']}`")
                        file_size = os.path.getsize(rec["path"])
                        st.markdown(f"- **Size**: {file_size / 1024:.1f} KB")
