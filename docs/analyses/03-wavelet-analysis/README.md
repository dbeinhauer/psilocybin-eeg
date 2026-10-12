# Stage 03 — Wavelet power and phase

Morlet time-frequency decomposition, and the tensor every later stage consumes.
`scripts/run_analysis.py --analysis wavelet_power|wavelet_phase`.

---

## What this stage produces

A `(subjects, channels, frequencies, times)` tensor of wavelet power — the input to stages
04 through 08. Stored under `data/processed/<experiment>/wavelets/` because recomputing it
costs far more than keeping it: tens of gigabytes per condition.

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

## The store is the product

This stage is mostly a store-building step, which is why its notebooks are the thinnest
exploration and its job the longest. The wavelets are kept **one file per participant**,
uncompressed float32, by `src/io/wavelet_store.py`:

```
data/processed/<experiment>/wavelets/<Condition>_<MusicType>/
    wavelet_power__<f0>_<f1>_<n>.json                 channels, freqs, n_cycles, sfreq
    <participant>__wavelet_power__<f0>_<f1>_<n>.npy   (channels, freqs, times)
```

Why this layout and not one compressed file per label:

- **Compression bought nothing.** DEFLATE shrank float wavelet power by about 5 %, and a
  compressed member cannot be read from the middle, so every read decompressed whole
  subjects in order — ~40 minutes per ASSR condition before any analysis started.
- **Uncompressed `.npy` is memory-mapped**, so a reader takes any participant, channel
  range, frequency bin or time span directly, and participants can be read in parallel.
  A notebook-sized run reads only its slice.
- **float32 halves the store**; power needs nowhere near float64's precision.
- **The participant id is in the filename**, so a row's identity never depends on an
  ordering sidecar, and a reader asks for participants by name.
- **A dataset cut on its channel or time axis is never stored.** It is transformed in
  memory only, so a notebook-sized run cannot overwrite a participant's full array with a
  fragment — the failure that once destroyed two full caches.

Each subject is written as soon as it is transformed, so an interrupted job resumes at
the first subject not yet stored. The stores are read through
`src.io.wavelet_store.find_wavelet_set` / `load_wavelet_subjects`, or through
`scripts.analysis_common.wavelet_transform(..., reuse_wavelets=True)`, which reads a
dataset's own participants at its own extent.

A **strip** under `notebooks/cache/` — a few subjects, seconds and frequency bins, built by
`scripts/run_build_notebook_cache.py` — uses the same layout, for running the notebooks
where the full store is not available.

## Outputs

```
plots/03-wavelet-analysis/{Condition}_{Music}/{broadband|bands}/{analysis_type}/
data/processed/{experiment}/wavelets/{label}/{participant}__wavelet_power__{grid}.npy
```

## Caveats

- Wavelet power at a frequency is smeared in time by the wavelet's own length; a narrow
  band buys frequency resolution at the cost of temporal precision. This is why stage 06
  tests both a single 40 Hz bin and a 35-45 Hz mean.
- The sliding-window figures need a window that fits: at 250 Hz the 15 s default needs
  3,750 samples.
