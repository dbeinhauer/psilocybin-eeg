# Psilocybin-EEG Visualization Catalog

An interactive Streamlit multi-page app that serves as:

1. **Analysis Catalog** — reference guide listing every analysis type with data shapes,
   order of operations, interpretation notes, notebook links, and matplotlib sketches.
2. **Results Browser** — local file browser for actual computed plot images.
3. **Interactive Explorer** — interactive visualisation of precomputed CSV analysis
   results (time-series, ISC distributions, windowed statistics) directly in Streamlit.

---

## Running Locally

### Option A — with `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is the project's recommended Python toolchain.
Run the app in an **isolated, temporary environment** — no manual install needed:

```bash
# From the repository root:
uv run --with "streamlit>=1.32.0" --with "pyyaml>=6.0" --with "numpy>=1.24.0" --with "matplotlib>=3.7.0" \
    streamlit run viz_catalog/app.py
```

Or install into a dedicated virtual environment and reuse it:

```bash
# Create and activate a virtual environment
uv venv .venv-viz
source .venv-viz/bin/activate          # macOS / Linux
# .venv-viz\Scripts\activate            # Windows PowerShell

# Install catalog dependencies
uv pip install -r viz_catalog/requirements.txt

# Launch
streamlit run viz_catalog/app.py
```

### Option B — with plain `pip`

```bash
# (Optionally create a venv first)
python -m venv .venv-viz && source .venv-viz/bin/activate

pip install -r viz_catalog/requirements.txt
streamlit run viz_catalog/app.py
```

The app opens at **`http://localhost:8501`** in your browser.

---

## Browsing Real Results Locally

The **Results Browser** page lets you view plot files that were generated on HPC
(or a local pipeline run) once you copy them to your machine.

### 1. Copy plots from the HPC cluster

Use `rsync` (recommended — fast, incremental) or `scp`:

```bash
# rsync — copy the entire plots/ tree from HPC to your local machine
rsync -avz --progress \
    <your-username>@<hpc-hostname>:/path/to/project/plots/ \
    ~/psilocybin-eeg-results/

# scp alternative
scp -r <your-username>@<hpc-hostname>:/path/to/project/plots/ \
    ~/psilocybin-eeg-results/
```

> **Tip:** The plots directory on HPC follows the layout
> `plots/<NN-analysis-name>/<Condition>_<MUSIC_TYPE>/<subdir>/`.
> The Results Browser parses this structure automatically for its sidebar filters.

### 2. Point the Results Browser at your local folder

1. Launch the app (see above).
2. Navigate to **🔬 Results Browser** in the sidebar.
3. Paste the local path you copied plots into, e.g. `~/psilocybin-eeg-results/` or
   `C:\Users\you\psilocybin-eeg-results\`.
4. Use the sidebar filters to narrow by analysis stage, condition, music type, or
   filename pattern.

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

Only files matching the canonical pipeline output structure are indexed:

```
<stage>/<Condition>_<MusicType>/broadband/<analysis_type>/<filename.ext>
<stage>/<Condition>_<MusicType>/bands/<analysis_type>/<filename.ext>
```

> **No images are shown by default.** Select at least one filter to start browsing.

#### Sidebar filters

| Filter | Source | Example values |
|--------|--------|---------------|
| **Analysis stage** | First path component | `01-raw-mean-variance-analysis`, `02-isc-broadband-analysis` |
| **Condition** | First token of `<Condition>_<MusicType>` | `Placebo`, `Psilocybin` |
| **Music type** | Second token of `<Condition>_<MusicType>` | `CLASSIC`, `PSYTRANCE` |
| **Spectrum type** | Third path component — always `broadband` or `bands` | `broadband`, `bands` |
| **Analysis type** | Fourth path component — the specific analysis run | `loo_isc`, `pairwise_isc`, `sliding_window`, `mean_field`, `timeseries`, `variance`, `windowed`, `band_overlap` |
| **Frequency band** | Extracted from filename stem — whole-token matching only | `broadband` (all files in `broadband/`), `delta`, `theta`, `alpha`, `beta`, `gamma` |
| **Filename contains** | Free-text substring search on the filename | |

#### View modes

- **Grid view** — configurable number of columns (1–6); shows image tiles with relative path captions.
- **List view** — full-width images alongside a metadata panel (stage, condition, music type, analysis part, frequency band, file size).

#### ℹ️ Per-image info panel

Every image (in both Grid and List view) has an **"ℹ️ About this plot"** expander below it. Expand it to see:

- The **analysis stage** title and description from `catalog.yaml`.
- The matched **plot** title and interpretation note.
- A direct link to the **source notebook** on GitHub.
- A one-click link to the **📋 Analysis Catalog** page for full methodology details.

The info panel looks up `catalog.yaml` by matching the image's stage prefix and filename pattern, so it works automatically for all standard output files.

#### Compare modes

Three modes are available via the **Compare mode** radio button in the sidebar:

| Mode | When to use |
|------|-------------|
| **None** | Normal browsing (default) |
| **Manual (select 2–4)** | Tick individual images to compare any combination side-by-side |
| **By condition / music type** | Select one plot by filename stem; the browser automatically finds every condition/music-type variant of that plot and displays them in parallel columns |

**By condition / music type** is the recommended mode for comparing Placebo vs Psilocybin or CLASSIC vs PSYTRANCE results for the same analysis.

### 📊 Interactive Explorer

The Interactive Explorer lets you interactively visualise precomputed analysis
results stored as CSV files in a **results database** directory.  Unlike the
Results Browser (which shows static `.png` images), the Interactive Explorer
reads raw numeric data and renders interactive Streamlit charts — line charts,
area charts, histograms, styled DataFrames — that you can zoom, hover, and
explore.

#### Results database layout

The explorer expects CSV files in the following directory structure:

```
<results_db_root>/
└── <Condition>_<MusicType>/          ← e.g. Placebo_CLASSIC
    ├── broadband/
    │   ├── intersubject_timeseries.csv
    │   ├── windowed_stats.csv
    │   ├── loo_isc.csv
    │   └── pairwise_isc.csv
    └── bands/
        └── <band>/                   ← e.g. alpha, delta
            ├── intersubject_timeseries.csv
            ├── windowed_stats.csv
            ├── loo_isc.csv
            └── pairwise_isc.csv
```

These CSV files are produced by `src.analysis.results_store` save functions
(called from analysis scripts or notebooks).

#### Generating results

Use the save functions from `src.analysis.results_store`:

```python
from src.analysis.results_store import (
    save_intersubject_timeseries,
    save_windowed_stats,
    save_loo_isc,
    save_pairwise_isc,
)
```

Each function takes the analysis output (NumPy arrays or DataFrames) and writes
a metadata-enriched CSV to the specified directory.

#### Sidebar filters

| Filter | Source | Example values |
|--------|--------|---------------|
| **Condition** | First token of directory name | `Placebo`, `Psilocybin` |
| **Music type** | Second token of directory name | `CLASSIC`, `PSYTRANCE` |
| **Spectrum type** | `broadband` or `bands` | `broadband`, `bands` |
| **Frequency band** | Band subdirectory name | `broadband`, `delta`, `alpha` |
| **Analysis type** | CSV filename stem | `intersubject_timeseries`, `windowed_stats`, `loo_isc`, `pairwise_isc` |

#### Visualisation tools

Each CSV type gets a dedicated interactive renderer:

| Analysis type | Visualisation |
|---------------|---------------|
| `intersubject_timeseries` | Line chart (mean signal) + area chart (variance) with configurable downsampling |
| `windowed_stats` | Bar charts (mean variance, signal variance) + synchrony-candidate metric |
| `loo_isc` | Histogram (ISC distribution) + per-channel bar chart + summary metrics |
| `pairwise_isc` | Colour-graded matrix + off-diagonal distribution histogram + summary metrics |

All renderers include a collapsible **Raw data table** expander showing the full
DataFrame for detailed inspection.

---

## How to Add a New Analysis or Plot Type

All content is driven by `viz_catalog/catalog.yaml` — no Python changes needed for
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

1. Open `viz_catalog/pages/1_📋_Catalog.py`.
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
viz_catalog/
├── app.py                       ← Streamlit entry point
├── catalog.yaml                 ← Single source of truth for all analyses/plots
├── requirements.txt             ← Streamlit + pyyaml + matplotlib + pandas
├── README.md                    ← This file
└── pages/
    ├── 1_📋_Catalog.py          ← Analysis catalog view
    ├── 2_🔬_Results_Browser.py  ← Real results browser
    └── 3_📊_Interactive_Explorer.py  ← Interactive CSV results explorer
```
