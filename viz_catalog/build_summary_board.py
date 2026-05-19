"""Build static HTML summary boards for Placebo results.

Walks ``results/`` and ``plots/`` for existing PNGs and emits one tabbed HTML
page per music type. Layout and slot definitions match
``notes/board_layout.md``. Re-run after each pipeline change.

Usage:
    python viz_catalog/build_summary_board.py
    python viz_catalog/build_summary_board.py --music-type CLASSIC
    python viz_catalog/build_summary_board.py --output-dir /tmp/board
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {"slug": "channel_time", "label": "channel-time", "tint": "#cfe8ff"},
    {"slug": "subject_frequency_channel", "label": "subj-freq-chan", "tint": "#d4f0d4"},
    {"slug": "subject_time", "label": "subj-time", "tint": "#ffe5cc"},
    {"slug": "time", "label": "time", "tint": "#e8d4f0"},
]

TAB_A_STEMS = {
    "channel_time": ["ica_subject_loadings", "ica_loo_isc_bar", "isc_component_matrix"],
    "subject_frequency_channel": ["ica_loo_isc_bar", "isc_component_matrix"],
    "subject_time": ["ica_subject_loadings", "ica_loo_isc_bar", "isc_component_matrix"],
    "time": ["ica_subject_loadings", "ica_loo_isc_bar", "isc_component_matrix"],
}

TAB_B_STEMS = {
    "channel_time": [
        "pca_scree",
        "ica_component_timecourses",
        "ica_channel_time_heatmap",
        "ica_subject_frequency_heatmap",
        "ica_subject_time_heatmap",
        "ica_freq_channel_heatmap",
    ],
    "subject_frequency_channel": [
        "pca_scree",
        "ica_pairwise_frequency_channel",
        "ica_pairwise_subject_channel",
        "ica_pairwise_subject_frequency",
    ],
    "subject_time": [
        "pca_scree",
        "ica_freq_channel_heatmap",
        "ica_subject_time_heatmap",
    ],
    "time": [
        "pca_scree",
        "ica_component_timecourses",
        "ica_channel_frequency_heatmap",
        "ica_subject_channel_heatmap",
        "ica_subject_frequency_heatmap",
        "ica_subject_time_heatmap",
    ],
}

TAB_C_STEMS = {
    "channel_time": ["ica_topomap_mean_variance"],
    "subject_frequency_channel": [],
    "subject_time": ["ica_topomap_mean"],
    "time": ["ica_topomap_mean_variance"],
}

TAB_D_STEMS = {
    "channel_time": ["ica_time_frequency"],
    "subject_frequency_channel": ["ica_time_frequency"],
    "subject_time": [
        "ica_time_frequency",
        "ica_sliding_window_loo_isc",
        "ica_mean_variance_over_time",
    ],
    "time": ["ica_time_frequency"],
}

TAB_LABELS = [
    ("a", "A. Summary"),
    ("b", "B. Controls"),
    ("c", "C. Topomaps"),
    ("d", "D. Time"),
    ("e", "E. Sanity"),
]


def variant_dir(slug: str, music_type: str) -> Path:
    return (
        REPO_ROOT
        / "results"
        / f"04-{slug.replace('_', '-')}-wavelet-ica-analysis"
        / f"Placebo_{music_type}"
        / "broadband"
        / slug
    )


def resolve_variant_file(slug: str, music_type: str, stem: str) -> Path | None:
    path = variant_dir(slug, music_type) / f"{stem}_Placebo_{music_type}.png"
    return path if path.exists() else None


def cross_decomp_files(music_type: str) -> list[tuple[str, Path | None]]:
    """Tab D's cross-decomposition row (pre-ICA references + broadband ISC)."""
    candidates: list[tuple[str, Path]] = [
        (
            "tf_map (pre-ICA, 03-wavelet)",
            REPO_ROOT
            / "results/03-wavelet-analysis"
            / f"Placebo_{music_type}/broadband/tf_map"
            / f"tf_map_Placebo_{music_type}.png",
        ),
        (
            "sw_isc_overlay (pre-ICA, 03-wavelet)",
            REPO_ROOT
            / "results/03-wavelet-analysis"
            / f"Placebo_{music_type}/broadband/sliding_window"
            / f"sw_isc_overlay_Placebo_{music_type}.png",
        ),
        (
            "sw_isc_overlay (broadband, 02-isc)",
            REPO_ROOT
            / "results/02-isc-broadband-analysis"
            / f"Placebo_{music_type}/broadband/sliding_window"
            / f"sw_isc_overlay_{music_type}.png",
        ),
    ]
    return [(label, p if p.exists() else None) for label, p in candidates]


def sanity_sections(music_type: str) -> dict[str, list[Path]]:
    """Sanity tab — grouped file lists. Globs are forgiving: empty matches OK."""

    def glob_all(rel_pattern: str) -> list[Path]:
        return sorted(REPO_ROOT.glob(rel_pattern))

    return {
        "Raw mean/variance (01-raw-mean-variance)": glob_all(
            f"results/01-raw-mean-variance-analysis/Placebo_{music_type}/broadband/*/*.png"
        ),
        "ISC distributions (02-isc)": glob_all(
            f"results/02-isc-broadband-analysis/Placebo_{music_type}/broadband/loo_isc/*.png"
        )
        + glob_all(
            f"results/02-isc-broadband-analysis/Placebo_{music_type}/broadband/mean_field/*.png"
        ),
        "Wavelet baselines (03-wavelet)": glob_all(
            f"results/03-wavelet-analysis/Placebo_{music_type}/broadband/intersubject_variance/*.png"
        )
        + glob_all(
            f"results/03-wavelet-analysis/Placebo_{music_type}/broadband/spectral_profile/*.png"
        )
        + glob_all(
            f"results/03-wavelet-analysis/Placebo_{music_type}/broadband/wavelet_loo_isc/*.png"
        ),
    }


def rel_src(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def render_card(
    path: Path | None,
    label: str,
    output_dir: Path,
    *,
    variant: dict | None = None,
    na_text: str | None = None,
) -> str:
    """Render one card.

    If ``variant`` is given, the card is tinted with the variant's color and
    its label shows the variant name (intended for plot-type clusters where
    cards are compared across variants). Otherwise neutral background and the
    given ``label`` is shown.

    ``na_text`` forces the card into placeholder mode regardless of ``path``.
    """
    if variant is not None:
        title = variant["label"]
        style = f' style="--tint:{variant["tint"]}"'
        tinted_class = " variant-tinted"
    else:
        title = label
        style = ""
        tinted_class = ""

    if na_text is not None or path is None:
        msg = na_text or "not available"
        return (
            f'<div class="card{tinted_class} placeholder"{style}>'
            f'<div class="card-label">{html.escape(title)}</div>'
            f'<div class="placeholder-text">{html.escape(msg)}</div>'
            "</div>"
        )
    src = rel_src(path, output_dir)
    return (
        f'<div class="card{tinted_class}"{style}>'
        f'<div class="card-label" title="{html.escape(src)}">{html.escape(title)}</div>'
        f'<a href="{src}" target="_blank">'
        f'<img src="{src}" loading="lazy" alt="{html.escape(title)}"/>'
        "</a></div>"
    )


def render_na_card(message: str) -> str:
    return (
        '<div class="card placeholder">'
        f'<div class="placeholder-text">{html.escape(message)}</div>'
        "</div>"
    )


def render_typed_cluster(label: str, cards_html: str, *, two_col: bool = False) -> str:
    """A plot-type cluster: neutral header + grid of (typically tinted) cards."""
    cards_class = "cards" + (" two-col" if two_col else "")
    return (
        '<section class="cluster">'
        f'<header class="cluster-header">{html.escape(label)}</header>'
        f'<div class="{cards_class}">{cards_html}</div>'
        "</section>"
    )


def render_variant_block(variant: dict, cards_html: str, *, cross: bool = False) -> str:
    classes = "variant-block" + (" cross" if cross else "")
    style = f'style="--tint:{variant["tint"]}"' if not cross else ""
    return (
        f'<section class="{classes}" {style}>'
        f'<header class="variant-header">{html.escape(variant["label"])}</header>'
        f'<div class="cards">{cards_html}</div>'
        "</section>"
    )


def render_tab_a(music_type: str, output_dir: Path) -> str:
    blocks = []
    for v in VARIANTS:
        cards = []
        if v["slug"] == "subject_frequency_channel":
            cards.append(render_na_card("subject_loadings: not applicable (3-way decomp)"))
        for stem in TAB_A_STEMS[v["slug"]]:
            cards.append(render_card(resolve_variant_file(v["slug"], music_type, stem), stem, output_dir))
        blocks.append(render_variant_block(v, "".join(cards)))
    return f'<div class="variant-grid two-col">{"".join(blocks)}</div>'


def render_tab_b(music_type: str, output_dir: Path) -> str:
    blocks = []
    for v in VARIANTS:
        cards = [
            render_card(resolve_variant_file(v["slug"], music_type, stem), stem, output_dir)
            for stem in TAB_B_STEMS[v["slug"]]
        ]
        blocks.append(render_variant_block(v, "".join(cards)))
    return f'<div class="variant-grid one-col">{"".join(blocks)}</div>'


def render_tab_c(music_type: str, output_dir: Path) -> str:
    blocks = []
    for v in VARIANTS:
        stems = TAB_C_STEMS[v["slug"]]
        if not stems:
            cards = render_na_card("topomap: not applicable for this variant")
        else:
            cards = "".join(
                render_card(resolve_variant_file(v["slug"], music_type, s), s, output_dir)
                for s in stems
            )
        blocks.append(render_variant_block(v, cards))
    return f'<div class="variant-grid one-col">{"".join(blocks)}</div>'


def render_tab_d(music_type: str, output_dir: Path) -> str:
    cross_cards = "".join(
        render_card(p, label, output_dir) for label, p in cross_decomp_files(music_type)
    )
    cross = render_variant_block(
        {"label": "Cross-decomposition (no variant)", "tint": "#dddddd"},
        cross_cards,
        cross=True,
    )
    per_variant = []
    for v in VARIANTS:
        cards = "".join(
            render_card(resolve_variant_file(v["slug"], music_type, s), s, output_dir)
            for s in TAB_D_STEMS[v["slug"]]
        )
        per_variant.append(render_variant_block(v, cards))
    return f'<div class="variant-grid one-col">{cross}{"".join(per_variant)}</div>'


def render_tab_e(music_type: str, output_dir: Path) -> str:
    sections = []
    for section_name, files in sanity_sections(music_type).items():
        if not files:
            cards = render_na_card("no files")
        else:
            cards = "".join(render_card(p, p.name, output_dir) for p in files)
        sections.append(
            f'<section class="sanity-block">'
            f'<header class="sanity-header">{html.escape(section_name)}</header>'
            f'<div class="cards small">{cards}</div>'
            "</section>"
        )
    return "".join(sections)


def render_page(music_type: str, output_dir: Path) -> str:
    legend = "".join(
        f'<span class="legend-chip" style="--tint:{v["tint"]}">{html.escape(v["label"])}</span>'
        for v in VARIANTS
    )
    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{tid}">{html.escape(label)}</button>'
        for i, (tid, label) in enumerate(TAB_LABELS)
    )
    panels = {
        "a": render_tab_a(music_type, output_dir),
        "b": render_tab_b(music_type, output_dir),
        "c": render_tab_c(music_type, output_dir),
        "d": render_tab_d(music_type, output_dir),
        "e": render_tab_e(music_type, output_dir),
    }
    tab_panels = "".join(
        f'<div class="tab-panel{" active" if tid == "a" else ""}" id="tab-{tid}">{panels[tid]}</div>'
        for tid, _ in TAB_LABELS
    )

    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(music_type)} — Placebo summary</title>"
        f"<style>{CSS}</style></head><body>"
        f'<header class="page-header">'
        f"<h1>{html.escape(music_type)} — Placebo</h1>"
        f'<div class="legend">{legend}</div>'
        f'<p class="note">Broadband only; bands accessed via Results Browser.</p>'
        f"</header>"
        f'<nav class="tabs">{tab_buttons}</nav>'
        f"<main>{tab_panels}</main>"
        f"<script>{JS}</script></body></html>"
    )


CSS = """
*,*::before,*::after{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f7f7f8;color:#1a1a1a}
.page-header{padding:14px 24px 8px;background:#fff;border-bottom:1px solid #e5e5e7}
.page-header h1{margin:0 0 8px;font-size:22px}
.legend{display:flex;gap:8px;margin-bottom:4px;flex-wrap:wrap}
.legend-chip{display:inline-flex;align-items:center;padding:3px 10px;border-radius:12px;background:var(--tint);font-size:12px}
.note{margin:4px 0;font-size:12px;color:#555}
.tabs{display:flex;gap:4px;padding:8px 24px 0;background:#fff;border-bottom:1px solid #e5e5e7;position:sticky;top:0;z-index:10}
.tab-btn{background:transparent;border:none;padding:8px 14px;cursor:pointer;font-size:14px;border-radius:6px 6px 0 0;color:#444}
.tab-btn:hover{background:#f0f0f3}
.tab-btn.active{background:#f7f7f8;border:1px solid #e5e5e7;border-bottom:1px solid #f7f7f8;margin-bottom:-1px;font-weight:600;color:#000}
main{padding:16px 24px 48px}
.tab-panel{display:none}
.tab-panel.active{display:block}
.variant-grid{display:grid;gap:14px}
.variant-grid.two-col{grid-template-columns:repeat(2,1fr)}
.variant-grid.one-col{grid-template-columns:1fr}
.variant-block{background:#fff;border:1px solid #e5e5e7;border-radius:8px;overflow:hidden}
.variant-header{background:var(--tint);padding:6px 12px;font-weight:600;font-size:13px;color:#1a1a1a}
.variant-block.cross .variant-header{background:#dddddd}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;padding:10px}
.cards.small{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}
.card{background:#fafafa;border:1px solid #ececef;border-radius:6px;padding:6px;display:flex;flex-direction:column;gap:6px}
.card-label{font-size:11px;color:#444;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
.card img{width:100%;height:auto;display:block;border-radius:4px}
.card.placeholder{align-items:center;justify-content:center;min-height:80px;color:#888;font-style:italic;text-align:center}
.placeholder-text{font-size:12px}
.sanity-block{margin-bottom:14px;background:#fff;border:1px solid #e5e5e7;border-radius:8px;padding:10px 12px}
.sanity-header{font-weight:600;font-size:13px;margin-bottom:6px}
@media print{.tab-panel{display:block!important;break-after:page}.tabs{display:none}}
"""

JS = """
document.querySelectorAll('.tab-btn').forEach(function(btn){
  btn.addEventListener('click',function(){
    var tid=btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.toggle('active',b===btn);});
    document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.toggle('active',p.id==='tab-'+tid);});
  });
});
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--music-type",
        choices=["CLASSIC", "PSYTRANCE", "both"],
        default="both",
        help="Music type(s) to build. Default: both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "summary",
        help="Where to write the HTML files. Default: ./summary/",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    music_types = (
        ["CLASSIC", "PSYTRANCE"] if args.music_type == "both" else [args.music_type]
    )
    for mt in music_types:
        page = render_page(mt, output_dir)
        out_path = output_dir / f"{mt}.html"
        out_path.write_text(page)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
