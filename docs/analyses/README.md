# Analyses

One directory per analysis stage, mirroring `notebooks/`, `plots/` and
`jobs/metacentrum/`. Each holds the deep description of what that stage computes and
why — the reasoning that is too long to sit in a notebook cell or a docstring.

| Stage | Documentation |
|---|---|
| `00-preprocessing` | [README](00-preprocessing/README.md) |
| `01-raw-mean-variance-analysis` | [README](01-raw-mean-variance-analysis/README.md) |
| `02-isc-broadband-analysis` | [README](02-isc-broadband-analysis/README.md) |
| `03-wavelet-analysis` | [README](03-wavelet-analysis/README.md) |
| `04-wavelet-ica-analysis` | [README](04-wavelet-ica-analysis/README.md) |
| `05-wavelet-iva-analysis` | [README](05-wavelet-iva-analysis/README.md) |
| `06-iva-condition-comparison` | [README](06-iva-condition-comparison/README.md) |
| `06b-corrca-condition-comparison` | [README](06b-corrca-condition-comparison/README.md) |
| `07-ica-condition-comparison` | [README](07-ica-condition-comparison/README.md) |
| `08-contrastive-synchrony` | [README](08-contrastive-synchrony/README.md) |

## What goes where

- **`docs/analyses/<stage>/`** — the method: what is computed, which variants exist, why
  each choice was made, what the caveats are.
- **The notebook** — parameter selection and a low-cost test of the workflow before it is
  run at full scale. It links here rather than restating the method.
- **The script** — the implementation, and the ground truth. If an analysis step is not in
  the script, it does not belong in the notebook either.
- **Docstrings in `src/`** — the contract of one function: arguments, return, errors.
