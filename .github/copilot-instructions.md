# Psilocybin-EEG Project - Agent Instructions

## Project Context

This project analyzes EEG neural data from participants recorded during listening to two types of music (classical and psytrance) under two conditions (placebo and psilocybin treatment). The major goal is to find and analyze the **inter-subject synchrony (ISC)** between participants.

### Experimental Design
- **EEG System**: EGI GSN-HydroCel 257-channel montage
- **Conditions**: Placebo / Psilocybin (within-subject, counterbalanced A/B)
- **Music Types**: Classical (`CLASSIC`), Psytrance (`PSYTRANCE`)
- **Primary Analysis**: Leave-one-out ISC, pairwise ISC, sliding-window ISC per frequency band
- **File Formats**: EDF (raw), MNE `.fif` (processed), NumPy `.npy` (probabilities), CSV (metadata)

## Technology Stack
- **Python**: ≥3.12
- **Core Libraries**: MNE-Python (EEG processing), NumPy, SciPy, Pandas
- **ICA/Artifact Removal**: AutoReject, MNE-ICALabel, PyTorch (optional GPU)
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest (194 tests)
- **Linting**: Ruff (via GitHub Actions)

## Pipeline Architecture

The analysis pipeline has three main stages:

### 1. Preprocessing (`src/preprocessing/`, `scripts/run_preprocessing.py`)
1. **Channel preparation** (`channel_prep.py`): Rename channels, set types (EEG/ECG/TAG), apply montage, exclude boundary electrodes
2. **Signal filtering** (`filtering.py`): Crop, notch filter (50 Hz), bandpass (1-100 Hz), RANSAC bad-channel detection, interpolation, average reference, AutoReject
3. **ICA artifact removal** (`ica.py`): Extended-Infomax ICA, ICLabel classification (brain/muscle/eye/heart/line/channel/other), component exclusion, reconstruction

### 2. Time Alignment (`src/preprocessing/time_alignment.py`, `scripts/run_time_alignment.py`)
- Extract TAG channel stimulus markers
- Compute pairwise cross-correlation
- Select reference recording
- Shift and crop all recordings to common time window

### 3. Analysis (`src/analysis/`, `scripts/run_analysis.py`)
- **ISC computation** (`isc.py`): Leave-one-out, pairwise, sliding-window ISC
- **Frequency bands**: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-70 Hz)
- **Wavelet analysis** (`wavelet.py`): Morlet wavelet power and phase analysis
- **Summary statistics** (`summary.py`): Aggregation across participants and conditions

## Directory Structure

```
psilocybin-eeg/
├── config/                        # Electrode coordinates, excluded electrodes, participant maps
├── data/                          # Raw, interim, and processed data (gitignored)
│   ├── raw/psilo_music/           # Original EDF files
│   ├── interim/psilo_music/       # Before/after ICA, IC probabilities
│   └── processed/psilo_music/     # Cleaned EEG, cropped, concatenated
├── src/                           # Source package
│   ├── definitions/               # Enums, constants, field definitions
│   ├── io/                        # Parsing, loading, saving
│   ├── preprocessing/             # Channel prep, filtering, ICA, alignment
│   ├── analysis/                  # ISC, wavelet, mean/variance, summary
│   ├── visualization/             # Plots for preprocessing and analysis
│   ├── filtering/                 # Metadata filtering utilities
│   └── utils/                     # Logging helpers
├── scripts/                       # CLI entry points
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # pytest test suite
├── jobs/                          # HPC job scripts (Metacentrum, Umbriel)
└── docs/                          # Extended documentation
```

## Coding Standards

### Python Style
- Follow **PEP 8** and **Ruff linter** rules (enforced via GitHub Actions)
- Use **type hints** for all function parameters and return values
- Function/variable/file names: `snake_case`
- Class names: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes/methods: prefix with `_`

### Naming Conventions
- **DataFrame columns**: `snake_case` (e.g., `"condition"`, `"music_type"`, `"participant_id"`)
- **Enum values**: Use enum classes from `src/definitions/fields.py`:
  - `ConditionVariants.PLACEBO`, `ConditionVariants.PSILOCYBIN`
  - `MusicTypeVariants.CLASSICAL`, `MusicTypeVariants.PSYTRANCE`
  - `FrequencyBandNames.DELTA`, `FrequencyBandNames.THETA`, etc.
- **File paths**: Use `ProjectPaths` constants from `src/definitions/constants.py`
- **Never hardcode categorical values as strings** — always use Enums

### Documentation
- **Docstrings** for all public functions and classes (Google style)
- Include type information in docstrings for complex types
- Add comments for complex logic or non-obvious decisions
- Keep comments concise and up-to-date with code changes

### Error Handling
- Use descriptive error messages
- Validate input data dimensions and types early
- Log errors using `src/utils/logging_config.py` utilities
- Prefer explicit validation over silent failures

## Module-Specific Guidelines

### `src/definitions/`
- **Single source of truth** for all categorical values (Enums)
- Never hardcode strings; always reference Enum values
- All field names defined in `fields.py`
- Path constants in `constants.py` (use `ProjectPaths`)
- Channel mappings in `mappings.py`

### `src/io/`
- **Parsing** (`parsing.py`): Extract metadata from filenames (condition, music type, participant)
- **Loading** (`loading.py`): Load EEG data, route by data type, construct paths
- **Saving** (`saving.py`): Save data with proper naming conventions
- Always preserve metadata through load/save operations

### `src/preprocessing/`
- Follow the pipeline order: **channel_prep → filtering → ica → time_alignment**
- Use MNE-Python conventions for `Raw` and `Epochs` objects
- Always preserve channel information and metadata
- Document all preprocessing parameters (filter cutoffs, ICA thresholds, etc.)
- Use `LoggerMixin` from `src/utils/logging_config.py` for class logging

### `src/analysis/`
- **ISC functions** (`isc.py`): Validate input shape `(n_samples, n_channels, n_participants)` before computation
- Return results as **pandas DataFrames** with meaningful indices (participant, condition, music, band, channel)
- Include metadata columns for traceability
- Document all statistical assumptions and methods
- **Data representations** (`data_representations.py`): Use `AnalysisData` classes to encapsulate data + metadata

### `src/visualization/`
- Use consistent color schemes (defined in module-level constants)
- All plots should have clear titles, labels, and legends
- Save plots with descriptive filenames including metadata (condition, music, participant)
- Use Seaborn for statistical plots, Matplotlib for EEG topomaps

### `src/filtering/`
- Use `DatasetFilter` class for metadata-based filtering
- Support filtering by condition, music type, participant, frequency band
- Return filtered DataFrames with preserved indices

### `tests/`
- **Coverage target**: 90%+ for all modules
- Test file structure: `tests/test_<module>.py` for each `src/<package>/<module>.py`
- Use pytest fixtures from `conftest.py` for common test data
- All tests must pass before PR merge (`python -m pytest tests/ -v`)
- Mock file I/O operations where appropriate
- Test edge cases (empty data, missing values, invalid inputs)

## Key Enums and Constants

### From `src/definitions/fields.py`
- **Conditions**: `ConditionVariants.PLACEBO`, `ConditionVariants.PSILOCYBIN`
- **Music Types**: `MusicTypeVariants.CLASSICAL`, `MusicTypeVariants.PSYTRANCE`
- **Frequency Bands**: `FrequencyBandNames.DELTA`, `.THETA`, `.ALPHA`, `.BETA`, `.GAMMA`
- **Analysis Types**: `AnalysisVariants.ISC`, `.MEAN_VARIANCE`, `.WAVELET_POWER`, `.WAVELET_PHASE`
- **Data Types**: `DataTypeVariants.RAW`, `.BEFORE_ICA`, `.AFTER_ICA`, `.CROPPED`, `.CONCATENATED`
- **Coordinate Systems**: `CoordinateSystem.STANDARD_1020`, `.STANDARD_1005`, `.GSN_HYDRO_CEL_257`

### Frequency Band Definitions
```python
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 70),
}
```

## Common Workflows

### Adding a New Analysis Function
1. Create function in appropriate `src/analysis/` module
2. Add comprehensive docstring with input/output specs and examples
3. Use type hints for all parameters and return values
4. Validate input data dimensions and types
5. Return pandas DataFrame with proper metadata columns
6. Write tests in `tests/test_<module>.py`
7. Run tests: `python -m pytest tests/test_<module>.py -v`
8. Update module README if adding major functionality

### Modifying Preprocessing Steps
1. Edit the appropriate `src/preprocessing/` module
2. Ensure changes preserve metadata and channel information
3. Document parameter changes in docstrings
4. Update `src/preprocessing/README.md` if pipeline order changes
5. Run full preprocessing test suite: `python -m pytest tests/test_channel_prep.py tests/test_filtering_preprocessing.py tests/test_ica.py -v`
6. Verify no regressions in downstream analysis

### Working with Data
1. **Never hardcode data paths** — use `ProjectPaths` from `src/definitions/constants.py`
2. Use `src/io/parsing.py` functions to extract metadata from filenames
3. Route all data loading through `src/io/loading.py`
4. Route all data saving through `src/io/saving.py`
5. Follow naming conventions from `data/README.md`
6. Add new participants to exclusion list in `data/excluded_participants/psilo_music.csv` if needed

### Running Scripts
```bash
# Preprocessing
python scripts/run_preprocessing.py --raw_processing --plot_results

# Time alignment
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC

# Analysis (ISC)
python scripts/run_analysis.py --analysis isc --condition Placebo --music_type CLASSIC

# Analysis (mean/variance)
python scripts/run_analysis.py --analysis mean_variance --condition Psilocybin --music_type PSYTRANCE

# Wavelet analysis
python scripts/run_analysis.py --analysis wavelet_power --wavelet_cache_dir data/processed/psilo_music/wavelets
```

## Dos and Don'ts

### Do
✓ Use Enum types for all categorical values (condition, music type, frequency bands)
✓ Add type hints to all functions
✓ Write tests for new functionality (aim for 90%+ coverage)
✓ Follow existing project structure and naming conventions
✓ Use logging utilities from `src/utils/logging_config.py`
✓ Preserve metadata through all pipeline stages
✓ Validate input data dimensions early
✓ Return pandas DataFrames from analysis functions
✓ Document complex logic with clear comments
✓ Run linter and tests before committing (`ruff check .`, `pytest tests/ -v`)

### Don't
✗ Hardcode categorical values as strings (use Enums instead)
✗ Skip type hints on new functions
✗ Modify the preprocessing pipeline order without documentation
✗ Commit data files to git (they're gitignored)
✗ Remove or disable existing tests
✗ Add dependencies without updating `requirements.txt` and `pyproject.toml`
✗ Use deprecated MNE-Python functions (check MNE docs)
✗ Create circular imports between modules
✗ Mutate input data without copying first
✗ Use global state or class-level mutable defaults

## Testing Requirements

### Before Committing
```bash
# Run full test suite
python -m pytest tests/ -v

# Run tests for specific module
python -m pytest tests/test_isc.py -v

# Check code style
ruff check .

# Auto-fix formatting
ruff format .
```

### Test Coverage
- All new functions must have corresponding tests
- Test both happy path and edge cases
- Use parametrized tests for multiple input variations
- Mock file I/O to avoid dependencies on actual data files
- 194 tests currently pass; maintain or improve this count

## HPC Execution

For large-scale processing on HPC clusters (Metacentrum, Umbriel):
- Job scripts in `jobs/metacentrum/` and `jobs/umbriel/`
- Request appropriate resources (CPU, memory, GPU for ICA)
- Use `--verbose` flags for detailed logging in HPC jobs
- See `docs/hpc_guide.md` for resource requirements

## Documentation Resources

- [`CODEBASE_STRUCTURE.md`](../CODEBASE_STRUCTURE.md) — Full annotated directory tree
- [`docs/pipeline_overview.md`](../docs/pipeline_overview.md) — Detailed pipeline stages
- [`docs/data_dictionary.md`](../docs/data_dictionary.md) — All field names and enum values
- [`docs/hpc_guide.md`](../docs/hpc_guide.md) — HPC job submission guide
- [`data/README.md`](../data/README.md) — Data directory layout and naming
- [`src/preprocessing/README.md`](../src/preprocessing/README.md) — Preprocessing details

## Important Notes

- This is a **neuroscience research project** analyzing the effects of psilocybin on neural synchrony
- The primary metric is **Inter-Subject Correlation (ISC)** — the correlation of EEG signals across participants
- EEG data is high-dimensional: 257 channels × thousands of timepoints × multiple participants
- Artifact removal (ICA) is critical for data quality
- Always validate data integrity at each pipeline stage
- Metadata tracking is essential for reproducibility
