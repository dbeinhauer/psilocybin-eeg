# Stage 06 — IVA condition comparison

Does a **learned** spatial filter read the 40 Hz auditory steady-state response better
than simply averaging the fronto-central electrodes — and does it separate Placebo from
Psilocybin better?

Two steps, two scripts:

| Step | Script | Job |
|---|---|---|
| Decompose and store the components | `scripts/run_iva_condition_comparison.py` (`--store_components`) | `run_condition_comparison.pbs` |
| Analyse the stored components | `scripts/run_assr_snr_grid.py` | `run_snr_grid_joined.pbs` |

The notebooks in `notebooks/06-iva-condition-comparison/` select parameters and test the
workflow at reduced scale. They call the same code; the script is the ground truth.

---

## 1. The two joins

IVA estimates components jointly across datasets, assuming component *k* is the same
source in every dataset. What counts as "a dataset" is the choice the two variants make:

- **`iva_channel_joined_separate`** (subject axis) — one dataset per **recording**, so rows
  `0..P-1` are Placebo and `P..2P-1` Psilocybin. A participant occupies two rows with two
  mixing matrices, so topographies are per recording and a topography contrast between
  conditions is meaningful. A bare participant label addresses two rows, so rows are
  resolved through `results.row(participant, condition)`, never by position.
- **`iva_channel_joined_tracks`** (time axis) — one dataset per **participant**, both
  tracks concatenated. One filter serves both conditions and the topography is shared.

`assr_variants.row_resolver` returns the right lookup for either.

## 2. Filters, not patterns

The store keeps forward patterns (topographies). For recording *s* the filter
`U_s = W_s @ P_s` (components x channels) is the **backward** operator that extracts a
source — what a projection needs — while the stored `A_s = pinv(U_s)` is the **forward**
model that belongs on a topomap. After `iva_g(..., whiten=True)` the two are nearly
unrelated (Haufe et al., 2014), so projecting with the pattern would be the classic
filter-vs-pattern error. `assr_trials.recover_spatial_filters` inverts the stored pattern
and the round trip `U_s @ A_s = I` is checked, not assumed.

## 3. The three reference rows

Fixed by the paradigm and the montage rather than learned, so there is nothing to
estimate and nothing to sign-align. Defined once in `assr_trials` (`FULL_MASK_LABEL`,
`PCA_MASK_LABEL`, `PCA_Z_MASK_LABEL`) because both stages emit them, and a row is only
comparable across stages if it is named the same in both. In increasing proximity to
what the decomposition actually saw:

| Row | Operator | Reads |
|---|---|---|
| `ASSR-mask (full)` | the fronto-central electrode average on the whole channel space | raw power |
| `ASSR-mask (PCA)` | the same average through each recording's PCA subspace (`mask @ PᵀP`) | raw power |
| `ASSR-mask (PCA, z)` | that operator on the z-scored cache — literally the signal the decomposition was handed | z-scored power |

`(full)` shows what the channel reduction discarded; `(PCA)` is the like-for-like
reference a component can be judged against; the pair `(PCA)` / `(PCA, z)` separates the
reduction from the per-(channel, frequency) rescaling that z-scoring adds.

**Equal weighting matters.** `roi_channelwise_snr` references every electrode to its own
pre-stimulus mean and divides by its baseline SD pooled over trials *before* the ROI is
averaged. Averaging raw power first would weight each electrode by its own power level,
and a per-trial divisor would let a momentarily quiet channel plateau the whole course.

Being a projected quantity, the PCA rows are **signed** — see §6.

## 4. The seven signal variants

All differ in how the **learned** rows are read — which spatial filter, on which signal —
and in nothing else. **The references never move**: every variant but `zscored` carries
`prestim`'s reference columns verbatim, which is what makes the set one comparison rather
than seven separate ones.

| Variant | Filter | Signal |
|---|---|---|
| `zscored` | each recording's own | the stored z-scored sources; no per-trial baseline |
| `prestim` | each recording's own | raw cache, per-trial pre-stimulus baseline |
| `stored_prestim` | — | the stored sources, given the same per-trial baseline |
| `mean_prestim` | the cohort mean | raw cache |
| `mean_stored_prestim` | the cohort mean | the IVA's own input, rebuilt as `PᵀP (z)` |
| `masked_prestim` | each recording's own, restricted to the ASSR electrodes | z-scored cache |
| `mean_masked_prestim` | the cohort mean, same restriction | z-scored cache |

Why each pair exists:

- **`prestim` vs `stored_prestim`** — the model was fitted on data z-scored along time, so
  the stored source is `U_s @ z_s`. Applying the same `U_s` to un-z-scored power drops that
  per-channel rescaling, so `prestim` is an *approximation* of the source and the stored
  array is the source itself.
- **`prestim` vs `mean_prestim`** — one thing changes, the filter. Anything that differs is
  the per-recording topography and nothing else.
- **`masked_*`** — sits between the two rows the grid already has. The binary reference
  reads the ASSR electrodes with equal weight; an unrestricted component reads the whole
  head with learned weights; this reads *only* those electrodes with learned weights. It is
  the only row that separates "the component weights the anchor area better" from "the
  component sees something outside it".

The restricted pair reads the **z-scored** wavelet and **no PCA**. The filters were
estimated on z-scored data and their weights presume that scaling; and `PᵀP` mixes every
channel back into the retained subspace, which would defeat "read only these electrodes".

`assr_variants.project_condition` builds every variant's input in one streaming pass over
the cache, so the extra operators cost one tensordot each rather than another pass over a
~50 GB file.

## 5. The cohort-mean filter

Three things must happen before averaging topographies means anything, in this order
(`assr_trials.cohort_mean_pattern`):

1. **Sign** — polarity is arbitrary per recording, so averaging raw patterns drives the
   mean toward zero. Each topography is flipped to `corr(topography, ASSR mask) > 0`.
2. **Scale** — amplitude is arbitrary too, so each topography is normalised to unit L2
   norm and contributes its *shape* only.
3. **Average the pattern, then invert** — `U_mean = pinv(A_mean)`. Averaging the filters
   instead would average `Σ⁻¹`-reweighted backward operators, which is not the backward
   operator of the average topography.

Component *k* may be averaged across recordings at all because IVA aligns its sources
across datasets by construction — which a per-dataset ICA would not license.

The run writes `mean_filter_topomaps.png` as the check: a recognisable fronto-central mean
map means the average survived; a diffuse one means the recordings disagreed and the
`mean_*` rows are reading a filter that represents nobody.

## 6. Polarity

A component's sign is free, so it is anchored to the fixed spatial reference:

```
flip[p, c, k] = sign( mean forward-pattern weight of IC k over the ASSR electrodes )
```

This reads only the pattern, never the tested response. That matters: an anchor taken from
the tested quantity inflates the one-sided false-positive rate to **~0.68** under a
simulated null.

The three `mean_*` variants read one shared operator whose sign was anchored once when the
cohort mean was built, so **no** per-recording flip is applied to them — doing so would
scramble a quantity that does not carry that sign. `masked_prestim` keeps the flip; zeroing
channels does not change a component's sign. Reference rows are never flipped.

## 7. Trials and normalisation

**Onsets** come from the concatenated array's `.stimulus_onsets.npy` sidecar. That array is
what the wavelet transform ran on, so its sample indices address exactly the time axis the
maps are on.

**The window comes from the paradigm, not the recording.** `AssrEpoch` sets the baseline
and post-onset span; the recording only *caps* the latter (`iva_quality.onset_window`
shortens `post` by the shortest inter-onset gap) so an epoch can never reach the next
stimulus. Both conditions then share the shorter `post`, so trials are contrastable sample
for sample.

**Each trial is referenced to its own pre-stimulus window** (`baseline_normalise`), not to
a condition- or participant-level average. 40 Hz power drifts across the recording and
differs by participant; a per-trial baseline removes both without any group statistic
entering the correction.

Two forms come back, and only one is usable for everything:

- **`z`** — `(x − baseline) / baseline SD`. Defined for every trial of every source
  whatever its sign, dimensionless, comparable across participants and sources.
  **This is the canonical form**; the tests run on it throughout.
- **`rel`** — `x / baseline − 1`, written only where the baseline is positive and `NaN`
  elsewhere so a downstream mean cannot average a sign-flipped value.

The mask is a non-negative electrode average, so a ratio is fine for it. An IVA source is a
**signed** combination of channels: its power runs negative on roughly half the trials and
some baselines sit near zero, where a ratio is not noisy but meaningless. Normalising the
two arms differently would confound exactly the comparison this stage exists for, so both
use `z`. `ASSR-mask (PCA)` behaves like a component here, not like the full mask.

Note `baseline_normalise` returns `(z, rel, positive)` — in that order.

## 8. The three test families

Participants are the unit of every test, never trials: a participant's trials are
correlated, so treating them as independent is anticonservative. The test is an **exact**
Wilcoxon signed-rank over the ranks of `|d|`, enumerating all `2**P` sign assignments, so
the p-value is exact and distribution-free — with a hard floor of `2/2**P` two-sided.

| Family | Question | Sided |
|---|---|---|
| `contrast` | Placebo − Psilocybin, per source | one-sided `greater` |
| `discrimination` | does an IC separate the conditions better than a reference? The interaction `(IC Pla − IC Psi) − (ref Pla − ref Psi)`, against each reference | two-sided |
| `snr` | is an IC's response magnitude *lower* than a reference's, per condition? | one-sided `less` |

`discrimination` and `snr` ask different things: the first is about separating the
conditions, the second about recovering the response at all. The `*_masked_*` rows answer
the second most directly, since they read the same electrodes as the reference and so
isolate the *weighting* within the anchor area.

**Nothing is corrected for multiplicity.** The CSV carries `effect` and `same sign` so a
row can be weighed against its neighbours rather than against a threshold alone.

## 9. The grid

Every `--halfwidths` x `--stimulus_intervals` combination in one pass. A half-width of `0`
is the single nearest frequency bin (the grid is 1 Hz spaced, so no neighbour is mixed in);
a positive one averages the closed interval, deliberately wider than the response so it is
insensitive to wavelet smearing and stimulator drift, at the cost of admitting
off-frequency power. Neither is obviously right and both are cheap, so which one a result
survives under is itself informative.

## 10. Outputs

```
results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv
plots/06-iva-condition-comparison/<Condition>_<Music>/<spectrum>/assr_snr_grid/pca_<N>/
    mean_filter_topomaps.png          # did the cohort agree on the averaged map?
    mean_masked_filter_topomaps.png   # how much weight survived the restriction?
    <selection>/<window>/             # one directory per grid cell
```

## 11. Caveats

- **Notebook caches are a 12 s subset.** The full recording is 188 s with ~148 onsets per
  condition; a 3000-sample cache yields ~9 trials. Extending that means re-projecting from
  a longer cache, not re-cutting the arrays. The script streams the full cache, so every
  variant there sees identical onsets and the asymmetry below does not arise.
- **`stored_prestim` is not a like-for-like row in the notebook.** Its IC rows come from the
  whole recording (~148 trials) while the references come from the subset (~9), so its IC
  side is the less noisy of the two. Every other variant is a clean comparison.
- **A within-epoch drift survives** the per-trial baseline, which only sets each trial's
  starting level.
- **Trial counts do not buy statistical power here.** The unit is the participant
  (P ≈ 12) whatever a variant's trial count.
