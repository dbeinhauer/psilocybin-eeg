# Psilocybin-EEG Visualization Catalog

An interactive Streamlit multi-page app that serves as:

1. **Analysis Catalog** — reference guide listing every analysis type with data shapes,
   order of operations, interpretation notes, notebook links, and matplotlib sketches.
2. **Results Browser** — local file browser for actual computed plot images.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r docs/viz_catalog/requirements.txt
```

### 2. Launch the app

From the **repository root**:

```bash
streamlit run docs/viz_catalog/app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Pages

### 📋 Catalog

- Select an **Analysis Group** from the sidebar (e.g. "Stage 00 — Preprocessing").
- Each group shows its description, linked notebooks, and a grid of **plot cards**.
- Every plot card includes:
  - Data shape (input → output)
  - Order of operations
  - Interpretation note
  - A small **matplotlib sketch** generated inline from `catalog.yaml`

### 🔬 Results Browser

- Enter a **local results directory** path (where your `.png` / `.jpg` plot files live).
- The browser recursively scans the folder and builds an index from path tokens.
- Use the **sidebar filters** (analysis stage, condition, music type, free text) to narrow results.
- Switch between **grid view** and **list view**.
- **Compare mode**: tick 2–4 images to display them side-by-side.

---

## How to Add a New Analysis or Plot Type

All content is driven by `docs/viz_catalog/catalog.yaml` — no Python changes needed for
new entries.

### Add a new analysis group

```yaml
analyses:
  - id: "03-wavelet"
    title: "Stage 03 — Wavelet Analysis"
    description: "Time-frequency decomposition using Morlet wavelets."
    notebooks:
      - path: "notebooks/03-wavelet-analysis/wavelet_power_exploration.ipynb"
        label: "Wavelet Power Exploration"
    plots:
      - id: "tf_map"
        title: "Time-frequency map"
        notebook: "notebooks/03-wavelet-analysis/wavelet_power_exploration.ipynb"
        section: "TF decomposition"
        filename_pattern: "tf_map*.png"
        data_shape:
          input: "(n_channels, n_times)"
          output: "(n_frequencies, n_times) — power per freq × time"
        operation_order:
          - "Apply complex Morlet wavelet"
          - "Compute instantaneous power"
          - "Average across channels"
        interpretation: "Bright regions = high power at that frequency and time."
        sketch_type: "timeseries_heatmap"
```

### Add a new sketch type

1. Open `docs/viz_catalog/pages/1_📋_Catalog.py`.
2. Find the `SKETCH_FUNCTIONS` dictionary near the top.
3. Add a new entry:

```python
def sketch_my_new_type() -> Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    # ... draw your sketch ...
    ax.set_title("My New Type", fontsize=8)
    fig.tight_layout()
    return fig

SKETCH_FUNCTIONS["my_new_type"] = sketch_my_new_type
```

4. Reference `sketch_type: "my_new_type"` in `catalog.yaml`.

---

## Directory Structure

```
docs/viz_catalog/
├── app.py                       ← Streamlit entry point
├── catalog.yaml                 ← Single source of truth for all analyses/plots
├── requirements.txt             ← Streamlit + pyyaml + matplotlib
├── README.md                    ← This file
└── pages/
    ├── 1_📋_Catalog.py          ← Analysis catalog view
    └── 2_🔬_Results_Browser.py  ← Real results browser
```
