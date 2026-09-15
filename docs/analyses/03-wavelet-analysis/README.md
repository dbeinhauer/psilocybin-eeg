# Stage 03 — Wavelet power and phase

Morlet time-frequency decomposition, and the tensor every later stage consumes.
`scripts/run_analysis.py --analysis wavelet_power|wavelet_phase`.

---

## What this stage produces

A `(subjects, channels, frequencies, times)` tensor of wavelet power — the input to stages
04 through 07. Cached under `data/processed/<experiment>/wavelets/` because recomputing it
costs far more than storing it: the caches run to tens of gigabytes per condition.

The default grid is 1-50 Hz in 50 bins (`src/definitions/frequency.py`). It is
deliberately linear rather than logarithmic, so the 40 Hz ASSR line sits on its own bin.

## Power versus phase

- **`wavelet_power`** — the magnitude. What every downstream decomposition uses, and what
  "synchrony" means in stages 04-07: shared *amount* of activity at a frequency.
- **`wavelet_phase`** — **extraction only**. The phase tensor is computed and cached, and
  nothing is analysed or plotted from it. Every group measure in this project treats its
  input as a linear quantity -- means, variances, Pearson and Spearman ISC -- and phase
  angles are circular, so none of them is meaningful on phase. The extraction is kept
  because a later analysis may want the cache; the ITPC and phase-ISC figures that used to
  exist were removed.

## The cache is the product

This stage is mostly a cache-building step, which is why its notebooks are the thinnest
exploration and its job the longest. Two caches exist per condition and they are not
interchangeable:

- the **source-of-truth** cache under `data/processed/`, full length, tens of GB
- a **strip** under `notebooks/cache/`, a few subjects and seconds, built by
  `scripts/run_build_notebook_cache.py`, which is what the notebooks read

Both use the same on-disk layout — `(subjects, channels * frequencies, times)` with the
frequency axis folded into the feature axis, which is what the `freqdim1` filename token
records — so the same reader serves both and a notebook exercises the code the job runs.

## Outputs

```
plots/03-wavelet-analysis/{Condition}_{Music}/{broadband|bands}/{analysis_type}/
data/processed/{experiment}/wavelets/broadband/{label}__wavelet_power__{grid}__freqdim1.npz
```

## Caveats

- Wavelet power at a frequency is smeared in time by the wavelet's own length; a narrow
  band buys frequency resolution at the cost of temporal precision. This is why stage 06
  tests both a single 40 Hz bin and a 35-45 Hz mean.
- The sliding-window figures need a window that fits: at 250 Hz the 15 s default needs
  3,750 samples.
