# Stage 06b — CorrCA condition comparison

Which spatial combination of channels makes the wavelet power **most synchronised across
people**, and is that synchrony different under Psilocybin than under Placebo?

Stage 06 answers the first half with IVA, which finds statistically dependent sources
and leaves synchrony as a by-product. This stage uses **Correlated Component Analysis**
(CorrCA; Dmochowski et al. 2012, Parra et al. 2018), which maximises synchrony directly.
The data, joins and figures are the same as in stage 06, so the two stages can be
compared panel for panel.

| Step | Script | Job |
|---|---|---|
| Decompose, compare, store: subject axis (`JoinedSeparate`) | `scripts/run_corrca_condition_comparison.py` | `run_condition_separate.pbs` |
| Decompose, compare, store: time axis (`JoinedTracks`) | `scripts/run_corrca_condition_tracks.py` | `run_condition_tracks.pbs` |
| SNR / contrast grid on the stored components (§5) | `scripts/run_corrca_snr_grid.py` | `run_snr_grid_separate.pbs`, `run_snr_grid_tracks.pbs` |

The SNR jobs read the store the first two write (`--store_components`, same `--n_pca`,
full channel set), so run them second.

Every job has a `*_gamma.pbs` twin that decomposes only 30–50 Hz (`--band_range 30 50`,
stored and plotted as `30-50hz`, under `bands/`). The gamma SNR jobs read the gamma
store (`--band 30-50hz`).

The notebooks in `notebooks/06b-corrca-condition-comparison/` select parameters at
reduced scale: `corrca_channel_joined_*` for the decomposition (`BAND_RANGE = (30, 50)`
for the gamma run) and `corrca_component_analysis_joined_*` for the SNR grid
(`BAND = "30-50hz"`). They call the same `run()` function the jobs do, and the script is
the ground truth. The SNR notebooks read the strip under `notebooks/cache/`, built by
`scripts/run_build_notebook_cache.py`. The core is `src/analysis/corrca.py`.

---

## 1. What CorrCA computes

Wavelet power `(S, C, F, T)` is z-scored along time for every (subject, channel,
frequency) series, exactly as for IVA. It is then flattened to `x_s` of shape
`(C, F·T)`, so channels are the features and time-frequency bins are the samples. One
pass over the data accumulates two quantities:

```
R_w = Σ_s R_ss                     pooled within-subject covariance
R_b = (R_t − R_w) / (S − 1)        pooled between-subject cross-covariance,
                                   R_t = covariance of the subject sum
```

The filters `w` maximise the **inter-subject correlation**:

```
ISC(w) = wᵀ R_b w / wᵀ R_w w        →   R_b w = λ R_w w,   λ = ISC
```

The fit runs in three steps:

1. Optionally shrink `R_w` with `--gamma`.
2. Keep the leading `--n_pca` eigen-directions of `R_w`, and whiten within them.
3. Solve a symmetric eigenproblem in that subspace.

The components come out sorted by ISC. When every subject has equal variance, λ equals
the mean pairwise Pearson r of the projected signals.

**Why truncate.** After average referencing and ICA cleaning, `R_w` is rank-deficient,
so the problem has to be solved in a subspace that actually carries variance. `--n_pca`
is that subspace's dimension, and it is also the number of components. It plays the
role IVA's per-recording PCA plays, but it is computed once on the pooled covariance.

**One shared filter.** Every subject is projected through the same `w`. That is the
central difference from IVA, and it has three consequences:

- **Sign.** A component has the same polarity in every subject by construction, so no
  per-subject sign alignment exists or is needed. The one free sign per component is
  fixed so the **shared** forward pattern is positive on average over the ASSR
  electrodes (`corrca.anchor_component_signs`). The topomaps then read positive over
  that area, and more power there is a positive source. It is applied to every
  recording alike. Flipping a single recording would invert part of a component the
  model fitted as one.

  This is not stage 06's criterion. Stage 06 flips on the correlation with the mask
  (`assr_trials.polarity_flip`): is the area more positive *than the rest of the head*?
  That leaves a component sitting on a global negative offset blue over the mask. The
  relative test exists to keep IVA's per-recording flips from following each
  recording's own offset. With one shared sign that risk doesn't arise, so 06b uses
  the absolute one. The stored `polarity_anchor_strength` is
  `|mask mean| / RMS of the pattern`; below 0.1 the sign is close to a coin toss.
- **Topographies.** The model's own topography is the shared forward pattern
  `A = R_w W` (figure `corrca_shared_topomaps`). Each subject also gets a descriptive
  pattern, `A_s = R_ss W (Wᵀ R_ss W)⁻¹`: the shared component seen through that
  subject's covariance. Those per-subject patterns feed the condition topography grids.
  Because they share one filter, they agree with each other more than IVA's free
  per-recording topographies do. Don't read that agreement as evidence of synchrony.
- **Sources.** `wᵀ x_s` reshaped to `(F, T)` is each subject's component TF map. The
  maps are comparable across subjects in both scale and sign, so a plain group mean
  does not cancel.

A forward pattern, not a filter, belongs on a topomap (Haufe et al. 2014). See
`wavelet_ica.iva_component_patterns` for the same argument made for IVA.

## 2. The two joins

The joins are the stage-06 ones; only what they mean for a shared filter differs.

- **`corrca_channel_joined_separate`** (subject axis). Every recording is one subject,
  so there are 2P of them. The between-subject term pools every pair, including each
  participant's own Placebo–Psilocybin pair, which for ASSR is still stimulus-driven
  synchrony. Topographies are per recording, so the topography grids contrast the
  conditions.
- **`corrca_channel_joined_tracks`** (time axis). Every participant is one subject whose
  time axis carries both tracks, z-scored per condition before concatenation
  (`--zscore_mode`). The filter is fitted on, and identical in, both conditions, and
  the projected sources are split back into the segments. Topographies span both tracks
  and are drawn as one shared row. This is the cleanest inter-condition contrast here:
  one fixed spatial component, observed under two conditions.

## 3. Synchrony per condition

The read-out IVA has no counterpart for. For each condition, using the **same**
filters:

- **Group ISC.** The CorrCA ratio evaluated on that condition's sources
  (`corrca.projected_isc`).
- **Per-participant leave-one-out ISC.** The correlation of a participant's component
  signal with the mean of the others in the same condition (`corrca.leave_one_out_isc`).
- **Paired test.** A Wilcoxon signed-rank test of the participant LOO-ISC difference
  (Psilocybin − Placebo) per component, with pairs formed by participant label
  (`corrca.paired_condition_test`). The p-values are **uncorrected** across components.

The results go to `corrca_condition_isc.png` and three CSVs beside it, and the store
keeps the `(conditions, K)` group ISC.

**Caveats.**

- The filters are fitted on all recordings, including the ones being scored, so every
  ISC here is optimistically biased. The bias is shared by both conditions, so the
  *difference* is the interpretable quantity, not the absolute level.
- Wavelet power is strongly autocorrelated, so the effective sample count is far below
  `F·T`. The Wilcoxon test runs across participants, not samples, which is why it is the
  test used.
- With `--zscore_mode per_condition`, an overall power difference between conditions is
  removed before the fit. What is compared is the synchrony of fluctuations, not
  amplitude.
- One filter serves every frequency bin. If the 40 Hz steady state and the low-frequency
  onset response have different topographies, a broadband fit blends them. In that case
  restrict the band (`--band_range 30 50`).

## 4. Outputs

```
plots/06b-corrca-condition-comparison/<JoinedSeparate|JoinedTracks>_<MusicType>/
    <broadband|bands>/<variant>/pca_<n>/
        condition_mean_topomaps.png      (separate) | shared_mean_topomaps.png (tracks)
        condition_mean_tf_maps.png, condition_mean_tf_maps_onset.png
        corrca_shared_topomaps.png, corrca_condition_isc.png
        corrca_condition_isc.csv, corrca_participant_isc.csv, corrca_condition_isc_test.csv
        participants/*_cc<k>_participants_*.png
data/processed/<experiment>/iva_results/<JoinedSeparate|JoinedTracks>/
    <variant>__<MusicType>__<spectrum>__pca<n>.npz        (--store_components)
```

The component store is the stage-06 one (`src/io/iva_store.py`), keyed by the
`IvaVariants.CORRCA_*` members. Besides the TF maps and per-recording patterns it keeps
these extras:

| Extra | Contents |
|---|---|
| `corrca_filters` | `(C, K)` shared filters |
| `corrca_shared_patterns` | `(K, C)` shared forward patterns |
| `corrca_fit_isc` | ISC of the fit, per component |
| `corrca_gamma`, `corrca_n_retained` | the regularisation used |
| `corrca_condition_isc` + `corrca_condition_isc_conditions` | `(conditions, K)` group ISC, with its row order |
| `polarity_anchor_flip`, `polarity_anchor_strength` | `(K,)` sign anchor and its strength |

## 5. SNR and stimulation-interval contrasts

`run_corrca_snr_grid.py` is the CorrCA companion of the stage-06 `run_assr_snr_grid.py`
([method](../06-iva-condition-comparison/README.md), §3–§9). It uses the same trials,
the same three ASSR references and the same per-trial normalisation. It sweeps the same
grid of frequency selection (`--halfwidths`, e.g. the 40 Hz bin and 35–45 Hz) crossed
with stimulus window (`--stimulus_intervals`, e.g. 0–500 ms and 200–500 ms), and runs
the same three exact-Wilcoxon test families:

| Family | Question | Sided |
|---|---|---|
| `contrast` | Placebo − Psilocybin, per source | one-sided `greater` |
| `discrimination` | does a component separate the conditions better than a reference? | two-sided |
| `snr` | is a component's response weaker than a reference's, per condition? | one-sided `less` |

It draws the same figures per grid cell: `pvalue_summary__<variant>`,
`snr_by_condition__<variant>` and `trial_course_by_source__<variant>`.

**What CorrCA changes:**

- **The filter is read back from the store, never recovered.** The stage-06 grid
  inverts each recording's pattern to get its filter. For CorrCA that would be wrong:
  the per-recording pattern is the shared component seen through that recording's
  covariance, and its pseudo-inverse is a different operator from the filter the model
  applied. The stored `corrca_filters` are used directly.
- **No per-recording polarity flip.** The one sign per component was anchored at fit
  time, and the stored sources carry it.
- **`ASSR-mask (PCA)` reads through the CorrCA subspace.** This is the orthogonal
  projector onto the span of the filters, the `--n_pca`-dimensional subspace they were
  sought in. In stage 06 it is each recording's PCA subspace. The label and the role are
  the same: the like-for-like reference for a component that could only see that
  subspace.
- **Four signal variants:** `zscored`, `prestim`, `stored_prestim` and
  `masked_prestim`. Stage 06's `mean_*` rows exist to swap per-recording filters for one
  shared filter, which CorrCA's filter already is.

**Outputs:**

- `results/<experiment>/assr_snr_grid__corrca_channel_joined_<join>__<spectrum>__pca<N>.csv`
- `plots/06b-corrca-condition-comparison/<Condition>_<Music>/<spectrum>/corrca_snr_grid/pca_<N>/`

The plot directory also holds `masked_filter_topomaps.png`, which shows what the
`masked_prestim` rows read.

## 6. Not here yet

- **Cross-validated ISC.** Fit on all but one participant and score the held-out one.
  This would give an absolute ISC level free of the fit bias.
- **A time-shift surrogate null for the group ISC.**
- **Time-resolved ISC on the component signals.** Sliding windows, to locate *when* each
  condition is synchronised.
- **Fitting on Placebo only.** Then project Psilocybin through those filters.
