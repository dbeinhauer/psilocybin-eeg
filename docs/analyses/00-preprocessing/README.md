# Stage 00 — Preprocessing, alignment and inspection

Two different kinds of notebook live here, and they are read differently.

**Driven analyses** — `assr_raw_pca_analysis.ipynb`, `assr_wavelet_pca_analysis.ipynb` and
`assr_roi_tf_variants.ipynb` call `scripts/run_assr_raw_pca.py`,
`scripts/run_assr_wavelet_pca.py` and `scripts/run_assr_roi_tf_variants.py`, exactly as
the later stages do. Their method is documented below.

**Inspection notebooks** — everything else. They answer a question about the dataset
rather than producing a result: whether the annotations line up, what the raw recordings
contain, whether an alignment worked. They run inline on one or a few recordings, need no
job, and are kept because the questions recur.

---

## The preprocessing pipeline

`scripts/run_preprocessing.py`, in order:

1. **Channel preparation** — rename raw EDF channels to montage names, set channel types
   (EEG, ECG, TAG/stim), apply the GSN-HydroCel montage, drop the boundary electrodes
   listed in `config/excluded_electrodes/`.
2. **Filtering** — crop the first and last 10 s, 50 Hz notch, FIR bandpass 1-100 Hz,
   RANSAC bad-channel detection and interpolation, average reference, AutoReject epoch
   annotation.
3. **ICA** — extended-Infomax, ICLabel classification, threshold-based component
   exclusion, reconstruction.

See [`docs/preprocessing_steps.md`](../../preprocessing_steps.md) for the thresholds.

## Two alignment paths

The experiments recover stimulus timing differently, and this is the single most
consequential difference between them.

- **`psilo_music`** carries a **TAG channel** with the music signal. Recordings are
  aligned by cross-correlating it: `scripts/run_time_alignment.py`.
- **`assr`** has no TAG channel. Timing comes from `fam+` **annotations**, and those are
  systematically late: `scripts/run_stimulus_alignment.py`.

### Why the ASSR annotations are late

The EDF header stores the recording start only to the nearest whole second, so every
annotation comes back late by the sub-second remainder. The lag is a **per-recording
file-format artefact**, not a property of the paradigm, and it is large — measured at
372-455 ms across the 38 ASSR recordings, which is more than two cycles at 40 Hz.

`scripts/run_stimulus_shift_calibration.py` measures it per recording against the `.evt`
exports and writes `config/participant_mappings/<experiment>_time_shift.csv`. Recordings
with no entry fall back to the median, `AssrEpoch.MARKER_ONSET_OFFSET_S = -0.4245 s`, and
are then still mis-timed by up to ~50 ms.

**Changing that offset or the calibration CSV invalidates every product from the coarse
crop onwards** — preprocessing included, because the stimulus-aware crop trims around the
offset-adjusted onsets. Regenerate in order: preprocessing, cropped, the concatenated
array with its `.stimulus_onsets.npy`, then the wavelet cache.

## The two PCA analyses

Both ask the same question — what a channel PCA recovers of the 40 Hz ASSR before any
joint decomposition is attempted — on two representations:

| Script | Input | Reads |
|---|---|---|
| `run_assr_raw_pca.py` | time-domain evoked response | FFT SNR at the stimulation frequency |
| `run_assr_wavelet_pca.py` | wavelet time-frequency power | band power around it |

They exist to answer whether the later stages' machinery is needed at all: if a plain
per-recording PCA already recovers the response, IVA and jICA are buying nothing. Both
compare each PCA component against an **unreduced reference** — the same electrode
average without the reduction — which is what makes the comparison interpretable.

`run_assr_raw_pca.py` defines its own `assr_snr`: an FFT SNR with a Hann taper and a
leakage-excluded noise floor. It is a different measure from
`assr_trials.roi_channelwise_snr`, which is baseline-referenced; they are not
interchangeable.

### Channel PCA before any joint decomposition

`run_assr_roi_tf_variants.py` (job `run_assr_roi_tf_variants.pbs`) asks what the
recordings show around the onsets before IVA or CorrCA touch them, by four orders of the
same three steps, each run on Placebo and Psilocybin **separately**:

| Pipeline | Order |
|---|---|
| `1_zscore_pca` | z-score vs recording → channel PCA (all channels) — `run_assr_wavelet_pca.py` |
| `2_roi_zscore_pca` | binary ASSR ROI → z-score → channel PCA |
| `3_pca_zscore` | channel PCA (all channels) → z-score of each PC |
| `4_roi_pca_zscore` | binary ASSR ROI → channel PCA → z-score of each PC |

The PCA is fit as stage 00 fits it, on the onset-averaged stack; its loadings are then
applied to the continuous recording, giving each PC a continuous source, and a PC's TF
map is that source averaged over the onsets (zero = the recording's mean level). Signs
are aligned across participants and both conditions in one pass, so the
Placebo − Psilocybin row is not a sign artefact. Each PC is also drawn trial-normalised
(`tf_baseline__<pipeline>.png`, stored as `tf_baseline__*` in `tf_maps__pca<n>.npz`):
every trial in its own pre-onset SD, median over trials, as every ASSR trial-locked map
is (stage 06, §7). Without the z-score first (3, 4) the PCA
runs on raw power, which the low frequencies dominate: PC1 alone takes ~90% of the
variance there.

The SNR grid reads each source the way 06b reads its `full` variant (selection averaged,
every trial referenced to its own pre-stimulus window, median over trials), against the
binary ROI built exactly as 06b's `ASSR-mask (full)` — on the same 12 paired participants
that row reproduces 06b's numbers to < 1e-8.

## Outputs

```
plots/00-preprocessing/{label}/{analysis_type}/
plots/00-preprocessing/assr_roi_tf_variants/ASSR/pca_{n}/      # both conditions in one tree
results/assr/assr_roi_tf_variants/ASSR/
```

## Caveats

- The inspection notebooks read single recordings and are not cohort results.
- The ASSR marker offset above is the largest single source of timing error in the
  project; check a recording has a calibration entry before trusting its onset-locked
  figures.
