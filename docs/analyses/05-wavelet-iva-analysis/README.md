# Stage 05 — IVA on wavelet tensors

Independent Vector Analysis applied to the `(subjects, channels, frequencies, times)`
wavelet-power tensor of a **single** condition. Stage 06 is the same machinery across two
conditions; this stage is where the decomposition itself is characterised.

| Script | Axis the decomposition mixes over | Notebook |
|---|---|---|
| `run_wavelet_iva_channel.py` | channels | `wavelet_iva_channel.ipynb` |
| `run_wavelet_iva_frequency_channel.py` | frequency x channel | `wavelet_iva_frequency_channel.ipynb` |
| `run_wavelet_iva_time.py` | time | `wavelet_iva_time.ipynb` |
| `run_wavelet_iva_channel.py --quality` | channels, quality mode | `wavelet_iva_channel_quality.ipynb` |

---

## What IVA buys over per-subject ICA

A per-subject ICA gives each subject their own components in their own arbitrary order
with their own arbitrary signs, and matching them up afterwards is guesswork. IVA
estimates all subjects **jointly** under the assumption that component *k* is the same
source in every one — a *source component vector* — so component *k* is comparable across
subjects by construction.

`Sigma_N[:, :, k]` is that SCV's `subjects x subjects` covariance. Normalised to a
correlation and averaged off the diagonal it answers the question the whole stage rests
on: *how much do the subjects actually agree about this component?* A value near zero
means the model was obliged to call `S` unrelated per-subject signals one component.

## The three axis variants

All three run the same `iva_g`; they differ only in which axis is treated as the mixing
dimension, i.e. what a "channel" of the unmixing is:

- **channel** — mixing over the 195 electrodes, samples are `frequency x time`. A
  component is a scalp topography plus a spectro-temporal course. The default reading.
- **frequency_channel** — mixing over `frequency x channel`, samples are time. A component
  is a spectro-spatial pattern; it can separate two sources that share a topography but
  differ in spectrum.
- **time** — mixing over time, samples are `frequency x channel`. A component is a
  temporal profile; useful when the question is *when* rather than *where*.

`IvaVariants` names them. Which one is right is an empirical question, which is why all
three exist and why the notebooks sweep the same parameters over each.

## Polarity and ordering

IVA fixes neither. Both are resolved before anything is averaged or plotted:

- **Sign** — `wavelet_ica.align_iva_component_signs` resolves the per-subject sign from
  `Sigma_N`, taking the orientation that makes the SCV's correlations positive. Without
  it a group mean cancels itself.
- **Order** — components are ranked by their mean off-diagonal SCV correlation, so
  IC 1 is the most-shared component rather than an arbitrary one. `--n_top` and
  `--n_bottom` select from each end of that ranking; together they may not exceed
  `--n_pca`, since IVA outputs exactly that many.

## Quality mode

`--quality` runs `src/analysis/iva_quality.py` instead of the standard figure set. It
scores each component against a reference by how well its per-subject topography and
onset-locked time course agree with it, and equalises subject influence first so one loud
participant cannot carry the score. See `iva_quality.quality_score` and
`equalize_subject_influence`.

## Running small

Every script takes `--n_subjects`, `--n_channels` and `--n_times`, which cut the leading
slice of each axis before the wavelet transform, so a notebook pass costs seconds. Note
that a channel subset invalidates any electrode mask built for the full montage — the
ASSR checks need `--lenient_mask` at reduced width.

## Outputs

```
plots/05-wavelet-iva-analysis/{Condition}_{Music}/{spectrum}/{iva_variant}/pca_{N}/
```

## Caveats

- The SCV correlation is the thing to read first. A component whose subjects do not agree
  has no group interpretation, whatever its topography looks like.
- Component count is `--n_pca`: the per-subject PCA dimension *is* the component count.
- Subsetting changes which subjects are in the cohort, so component identity is not
  comparable between a subset run and a full one.
