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

**Synchrony per condition.** The decomposition run also writes stage 06b's ISC read-out
on the oriented sources: group ISC per condition, each participant's leave-one-out ISC
(dots, paired by line) and a Wilcoxon test of the paired difference, uncorrected
(`iva_condition_isc.png` plus three `iva_*isc*.csv` beside it). See the
[06b README](../06b-corrca-condition-comparison/README.md#3-synchrony-per-condition) for
the definitions. Two caveats specific to IVA:

- IVA-G maximises dependence *across datasets*, with a separate demixing matrix per
  dataset, so these ISCs are more inflated by the fit than CorrCA's (one shared filter).
  Compare the conditions with each other, not the absolute level with stage 06b's.
- Each recording carries its own sign (§6). A recording anchored the wrong way has a
  negative leave-one-out ISC rather than a low one, so check a negative dot against its
  anchor strength before reading it as anti-synchrony.

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

## 4. The two signal variants

Both differ only in how the **learned** rows are read. **The references never move**: both
carry the same three reference columns (the masks projected through the raw cache, every
trial referenced to its own pre-stimulus window, the full mask swapped for the
equal-weight ROI row), which is what makes the pair one comparison rather than two.

| Variant | Filter | Signal |
|---|---|---|
| `full` | — | the stored sources (the decomposition's own output), per-trial pre-stimulus baseline |
| `roi_masked` | each recording's own, restricted to the ASSR electrodes | z-scored cache, same baseline |

`roi_masked` sits between the two rows the grid already has. The binary reference reads
the ASSR electrodes with equal weight; `full` reads the whole head with learned weights;
`roi_masked` reads *only* those electrodes with learned weights. It is the only row that
separates "the component weights the anchor area better" from "the component sees
something outside it".

It reads the **z-scored** wavelet and **no PCA**. The filters were estimated on z-scored
data and their weights presume that scaling; and `PᵀP` mixes every channel back into the
retained subspace, which would defeat "read only these electrodes".

`assr_variants.project_condition` builds both variants' inputs in one streaming pass over
the cache.

The grid used to run five more variants (`zscored`, `prestim` and the three cohort-mean
`mean_*` rows); they were retired on 2026-10-18. `full` and `roi_masked` are the former
`stored_prestim` and `masked_prestim`, with identical numbers.

## 5. The cohort-mean filter (retired)

The grid no longer reads a cohort-mean filter; see §4. The construction
(`assr_trials.cohort_mean_pattern`: sign, unit scale, average the patterns, then invert)
stays in the library.

## 6. Polarity

A component's sign is free, so it is anchored to the fixed spatial reference:

```
flip[p, c, k] = sign( mean forward-pattern weight of IC k over the ASSR electrodes )
```

This reads only the pattern, never the tested response. That matters: an anchor taken from
the tested quantity inflates the one-sided false-positive rate to **~0.68** under a
simulated null.

`roi_masked` keeps the flip; zeroing channels does not change a component's sign.
Reference rows are never flipped.

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

**The onset-locked TF maps are read the same way, at every frequency**
(`assr_trials.baseline_tf_map`): each trial of each frequency minus its own pre-onset
mean, divided by that recording's pre-onset SD pooled over its trials, the median over
trials kept. Every ASSR trial-locked TF figure uses it — `condition_mean_tf_maps_onset`
and the `participants/*onset_*` grids here and in 06b, stage 07's
`condition_tf_maps_onset`, stage 00's `tf_baseline__*` and inspection maps. A map is
then an SNR in baseline SD, so it compares across conditions, components and methods;
it is drawn without the per-participant equal weighting (`equalize=False`), which would
undo that unit. The SD is per recording, hence per condition: every decomposition's
sources are z-scored per condition upstream, and only a per-condition SD cancels that
scale. A 100 ms baseline is short at the bottom of the map, but on CorrCA CC1 the pooled
within-window SD is a near-constant fraction of the total baseline SD from 1 to 50 Hz
(0.2–0.4), so the divisor does not single out the low frequencies.

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
- **`full` is not a like-for-like row in the notebook.** Its IC rows come from the
  whole recording (~148 trials) while the references come from the subset (~9), so its IC
  side is the less noisy of the two. Every other variant is a clean comparison.
- **A within-epoch drift survives** the per-trial baseline, which only sets each trial's
  starting level.
- **Trial counts do not buy statistical power here.** The unit is the participant
  (P ≈ 12) whatever a variant's trial count.
