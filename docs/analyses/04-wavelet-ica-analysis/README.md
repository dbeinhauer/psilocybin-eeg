# Stage 04 — ICA on wavelet tensors

PCA followed by FastICA on the `(subjects, channels, frequencies, times)` wavelet-power
tensor. The predecessor of [stage 05](../05-wavelet-iva-analysis/README.md): same data,
same question, but each reshape is decomposed on its own rather than jointly across
subjects.

| Script | Observations | Features | Notebook |
|---|---|---|---|
| `run_wavelet_ica_time.py` | time | `S x C x F` | `wavelet_ica_time.ipynb` |
| `run_wavelet_ica_channel_time.py` | `S x F` | `C x T` | `wavelet_ica_channel_time.ipynb` |
| `run_wavelet_ica_frequency_channel.py` | `S x T` | `F x C` | `wavelet_ica_frequency_channel.ipynb` |
| `run_wavelet_ica_subject_time.py` | `F x C` | `S x T` | `wavelet_ica_subject_time.ipynb` |
| `run_wavelet_ica_subject_frequency_channel.py` | `T` | `S x F x C` | `wavelet_ica_subject_frequency_channel.ipynb` |

`run_separate_assr_wavelet_ica_channel.py` is a sixth entry point: a per-recording channel
ICA on the ASSR data, used to check what a decomposition estimated *separately* per
recording recovers, against the joint ones stages 05-07 use.

---

## The five reshapes

Every variant is the same pipeline — z-score along time, reshape to 2-D, PCA to `--n_pca`,
FastICA to `--n_ica`, project back — and differs only in which axes become observations
and which become features. That choice is the whole experiment: it decides what
"independent" means.

The tensor is **z-scored along the time axis** first, so each `(subject, channel,
frequency)` slice has zero mean and unit variance. Without it the decomposition is
dominated by which channels and frequencies happen to carry the most absolute power.

## Why the components are harder to read than stage 05's

ICA here has no mechanism tying a component to the same source across subjects. Where a
subject axis is folded into the *features*, a component mixes all subjects and there is
nothing to align; where it is folded into the *observations*, each subject contributes
rows and a component may describe one subject alone.

This is precisely what IVA was adopted for in stage 05, and why the notebooks in this
stage lean on the cross-component and inter-individual correlation figures: they are the
only check that a component means the same thing for more than one person.

## Cluster agreement

The ISC matrices are thresholded at `ISC_CLUSTER_THRESHOLDS` (0.3 / 0.5 / 0.7) and the
connected components of the resulting graph are the "clusters" the figures report — a
coarse but threshold-transparent way to ask which subjects a component groups together.
Read it at more than one threshold; a cluster that only exists at 0.3 is not a finding.

## Running small

All five take `--n_subjects`, `--n_channels` and `--n_times`, cutting the leading slice of
each axis before the wavelet transform. A channel subset invalidates an electrode mask
built for the full montage.

## Outputs

```
plots/04-wavelet-ica-analysis/{Condition}_{Music}/{spectrum}/{ica_variant}/pca_{N}/
```

## Caveats

- Component identity is not comparable across variants, across subsets, or across runs
  with a different `--n_pca`; FastICA's order and signs are arbitrary within a run.
- This stage is superseded by 05-07 for the condition question. It is kept because the
  five reshapes are the evidence for why the joint methods were needed.
