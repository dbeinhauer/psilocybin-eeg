# Stage 07 — Joint ICA across conditions

The same question as [stage 06](../06-iva-condition-comparison/README.md) — does a learned
spatial filter read the 40 Hz ASSR better than the fixed fronto-central electrode average,
and does it separate Placebo from Psilocybin better — asked of a **joint ICA** instead of
IVA.

Everything stage 06 documents about the three reference rows, polarity anchoring, per-trial
normalisation and the three test families applies unchanged. This page covers only what
differs.

One script does the whole stage: `scripts/run_wavelet_jica.py`
(`jobs/metacentrum/07-ica-condition-comparison/`).

---

## jICA vs IVA

IVA estimates one decomposition **per dataset** and couples component *k* across them
through `Sigma_N`. Joint ICA instead stacks the datasets into **one** matrix and estimates
a **single** unmixing for all of them, so component *k* is the same operator for everyone
by construction rather than by alignment.

That buys a simpler object and costs the per-recording topography: a jICA component has one
map, so there is no per-recording topography contrast to make. What varies between
conditions is only the signal read through the shared filter.

## The two joins

`--join` chooses what gets stacked:

- **`joined`** — the subject axis. Each recording contributes its own rows to the stacked
  matrix, so the cohort is `2P` datasets.
- **`joined_tracks`** — the time axis. Each participant's two tracks are concatenated
  first, so the cohort is `P` datasets and a participant's two conditions are guaranteed to
  share everything except the samples they contribute.

`JicaVariants` in `src/definitions/fields.py` names them.

## What the script does, and the two halves

A full run decomposes and then analyses. Either half can be skipped:

| Flag | Keeps |
|---|---|
| `--skip_analysis` | the decomposition and its component figures only |
| `--skip_decomposition_plots` | the tests and the summary figures only |

The notebooks use that split: `wavelet_ica_channel_joined_separate*.ipynb` look at the
decomposition, `ica_component_analysis.ipynb` looks at the tests.

## Polarity

ICA leaves each component's sign free exactly as IVA does, and the anchor is the same:
the correlation of the component's forward pattern with the ASSR electrode mask
(`assr_trials.polarity_flip`). Because the filter is shared, the sign is resolved **once**
per component rather than per recording — there is no per-recording flip to apply, and
applying one would scramble a quantity that does not carry it.

`wavelet_jica.orient_components` does the orientation;
`wavelet_jica.block_energy` reports how much of each component's variance each block
contributes, which is the diagnostic for whether a component is genuinely shared or is
carrying one block alone.

## Figures

`src/visualization/jica_plots.py`, shared with the notebooks:

- `plot_response_courses` — the stimulus-locked response per source, both conditions
  overlaid, each panel carrying its *p*.
- `plot_pvalue_summary` — the contrast forest, plus one panel per reference row showing
  `IC − reference`.
- `plot_snr_vs_reference` — the magnitude comparison, per condition.
- `plot_global_tf_grid` — the components' TF maps; `condition_tf_maps_onset` is the
  onset-locked one, trial-normalised as every ASSR trial-locked map is (each trial in its
  own pre-onset SD, median over trials; see stage 06, §7).

## Outputs

```
results/<experiment>/jica_tests__<join>__<music>__ica<N>.csv
plots/07-ica-condition-comparison/...
```

## Caveats

Stage 06's caveats about the notebook strip caches, the participant being the unit of every
test, and the absence of multiplicity correction all apply here too.
