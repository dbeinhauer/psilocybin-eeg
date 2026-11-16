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
| Decompose, store: each condition alone (§2) | `scripts/run_corrca_individual_conditions.py` | `run_individual_conditions.pbs` |
| SNR / contrast grid on the stored components (§5) | `scripts/run_corrca_snr_grid.py` | the same job, second step |

Each job fits six times from one data load, in this order: the full montage, the full
montage on log power (`_log`, §5c), the ASSR ROI alone (`_roi`), every other electrode
(`_non_roi`; §5b), then `_log_roi` and `_log_non_roi`. It then runs the SNR grid on the
six stores in the same order. The fit runs with `--skip_stored`: a fit whose store for that
`--n_pca` and band already exists is read back instead of refit, and every figure is
still drawn from it. The data is loaded either way, for the masked maps and any fit
still missing. A store fitted on other data (cohort, channels, time extent: the store
name does not record `--n_times` or `--n_pairs`) is refused, not reused.
`qsub -v FORCE_FIT=1` refits all six; `qsub -v N_PCA=<n>` picks the component count
(default 5).

The six fits run in parallel, each in its own forked process that reads the one loaded
tensor without copying it. `--parallel_fits P` is a memory budget in full-montage fits:
fits start in plan order while the running ones weigh at most `P`, where a cut fit
weighs its share of the channels (ROI ~0.2, non-ROI ~0.8). A running fit holds about
three copies of its prepared input at its peak (the input, its z-scored copy and one
temporary). The broadband joins run with `P = 2`, the individual conditions and the
gamma jobs with `P = 3`; each job's header gives the memory estimate. `P = 1` (the
script default) runs the fits one after another in one process.

Every job has a `*_gamma.pbs` twin that decomposes only 30–50 Hz (`--band_range 30 50`,
stored and plotted as `30-50hz`, under `bands/`), whose grid reads the gamma store
(`--band 30-50hz`). Every job also has a `*_1_30hz.pbs` twin that decomposes only the band
below gamma, 1–30 Hz (`--band_range 1 30`, token `1-30hz`; the 30 Hz bin belongs to both
bands). It fits only and runs no SNR grid. The grid reads its `full` rows at 40 Hz on the
store's own bins, and it refuses a store whose band does not contain that frequency. These
jobs run `P = 2`, since a 1–30 Hz fit carries 30 of the 50 bins.

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

Beside the joins, **`corrca_channel_individual_conditions`** fits each condition alone:
the time-axis join reduced to one track, so every participant the condition has is one
subject (not only the matched ones), and each condition gets its **own** filters. `CC 1`
of Placebo and `CC 1` of Psilocybin are then two different spatial combinations, to be
compared by their topographies and synchrony, not row by row; there is no paired test.
The store is a one-segment tracks store under the condition itself
(`iva_results/<Condition>/`), so the SNR grid reads it as it reads the join.

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
plots/06b-corrca-condition-comparison/IndividualConditions_<MusicType>/
    <broadband|bands>/corrca_channel_individual_conditions/<Condition>/pca_<n>/
        the same, with condition_mean_topomaps.png and no _test.csv
data/processed/<experiment>/iva_results/<JoinedSeparate|JoinedTracks|Condition>/
    <variant>__<MusicType>__<spectrum>__pca<n>.npz        (--store_components)
```

The `_roi` and `_non_roi` fits (§5b) draw into `<variant>_roi/` and `<variant>_non_roi/`
beside them, and the SNR grid (§5) into a `snr_grid/` subdirectory of the fit it read.
The full fit also draws `roi_masked_*` and `non_roi_masked_*` TF maps (§5b).

`condition_mean_tf_maps_onset.png` and the `participants/*onset_*` grids are
trial-normalised (each trial in its own pre-onset SD, median over trials), as every
ASSR trial-locked map is — see stage 06, §7. `run_corrca_snr_grid.py` stores the
per-participant maps in `results/<experiment>/assr_snr_grid__*__tf_baseline.npz`; it
draws no TF map or topography itself, only its statistical figures (§5).

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
`snr_by_condition__<variant>` and `trial_course_by_source__<variant>`. It adds
`trial_course_by_participant__<variant>`: each participant's median-over-trials course
for CC 1, CC 2 and the `ASSR-mask (PCA, z)` reference (what the fit was handed;
`ASSR-mask (full)` in a non-ROI store, which has no other), with one row per
condition. A participant keeps the same line style in every panel, and the bold
line is the all-participant mean, the centre line of `trial_course_by_source`.

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
- **The signal variants are stage 06's two and their complement:** `full` (the stored
  sources), `roi_masked` (the shared filter with every weight outside the ASSR
  electrodes zeroed, on the z-scored cache) and `non_roi_masked` (the same filter with
  the ASSR electrodes zeroed instead). The two masked filters add up to the full one,
  so between them they split a component's response into what the ROI carries and what
  the rest of the head carries. Neither is renormalised; every reading is in units of
  its own pre-stimulus SD, so their scales never meet. An ROI store (§5b) has no
  electrode outside the ROI and skips `non_roi_masked`.
- **Masked TF maps** (`--masked_tf_maps`, on in every job): the masked
  variants' full trial-normalised TF maps, read the way the stored sources' are, are
  stored (not drawn) beside those in the `__tf_baseline.npz`
  (`tf_baseline__<variant>__<Condition>`). The cache is then read at every frequency
  of the store rather than only at the 40 Hz selections, several times the read time.

**Outputs:**

- `results/<experiment>/assr_snr_grid__<variant>__<spectrum>__pca<N>.csv`; an
  individual condition's files carry it, `<variant>__<Condition>__<spectrum>`
- plots in `snr_grid/` inside the fit's own directory (§4): `.../<variant>/pca_<N>/snr_grid/`,
  likewise under `<variant>_roi/` and `<variant>_non_roi/`, and under
  `.../corrca_channel_individual_conditions/<Condition>/pca_<N>/` for one condition.
  Each grid cell gets `<selection>/<window>/` with the three statistical figures and
  nothing else: the TF maps and topographies are the fit's, in its directory.

**Individual conditions.** `--variant corrca_channel_individual_conditions` runs the
grid once per condition, each on that condition alone. The filters differ between the
conditions, so there is nothing to contrast: only the `snr` family is tested, the
p-value summary is not drawn and the TF map has one row.

## 5b. The binary ROI

Two ways of asking what the CorrCA components carry from the ASSR electrodes alone:

- **ROI- and non-ROI-masked maps of the full fit.** Every fit script draws
  `roi_masked_condition_mean_tf_maps.png` (and `_onset`, and `participants/roi_masked_*`):
  the fitted shared filter with every weight outside the ROI zeroed — not renormalised —
  applied to the z-scored signal the fit saw; then `non_roi_masked_*`, the same with the
  ROI zeroed instead. The two add up to the full maps. They are the TF-map counterpart
  of the grid's `roi_masked` / `non_roi_masked` rows. `--skip_roi_masked` drops both.
- **A fit on the ROI alone.** Every fit run makes a second fit from the same loaded
  tensor, with the channel axis cut to the ROI **before** the z-scoring, so ROI → z-score
  → PCA → CorrCA. It is stored and drawn as `corrca_channel_joined_{separate,tracks}_roi`,
  beside the full-montage fit, and every grid run reads that store after the full one.
  There `roi_masked` coincides with `full` by construction, a free check that the store
  and the cache agree. The ROI fit is skipped, with a warning, when the montage holds no
  ASSR electrode or fewer than `--n_pca` of them; `--skip_roi_fit` (fit) and `--skip_roi`
  (grid) skip it outright.
- **A fit on the rest of the head.** The same cut to every electrode BUT the ROI, stored
  and drawn as `<join>_non_roi`, third. It has no anchor electrode, so it keeps the
  fit's own orientation (largest shared-pattern weight positive). The grid reads its
  store on the full montage, with the filters zero on the ROI, so its references read
  the same electrodes as every other store's; only its `full` rows are tested, and
  only against `ASSR-mask (full)`, since a reference read through its subspace is
  identically zero. `--skip_non_roi_fit` (fit) and `--skip_non_roi` (grid) skip it.

## 5c. Log-power variant

Every run also fits on log power: `log10` of the wavelet power (floored at the dtype's
smallest normal number, as in stage 08), taken **before any z-scoring**, including the
time-axis join's per-condition z-scoring. A component is then fitted on relative power
changes rather than absolute ones, which the strongest bursts dominate. Everything
else is the raw-power fit's: the same cuts, the masked maps, the figures and the SNR
grid. The products are stored and drawn as the `*_log` variants beside the raw ones
(`corrca_channel_joined_tracks_log`, `..._log_roi`, `..._log_non_roi`, and likewise for
the subject-axis join and the individual conditions). `--skip_log_power` drops the
three log-power fits (and the grid's reading of their stores).

One load serves both: each fit's input is prepared from the raw tensor when the fit
comes up — cut to its electrodes, then the log if any, then the join's
standardisation — so only one prepared copy is held at a time. For the time-axis join
the tracks are therefore joined raw (`zscore_mode none`) and standardised per fit with
`--zscore_mode`, exactly as the join would have.

In the SNR grid a `*_log*` store reads its masked rows on the log cache, z-scored like
the fit's input. It adds a fourth reference, **`ASSR-mask (log, PCA, z)`**: the PCA
mask on that same log-then-z-scored signal, the like-for-like reference for what the
decomposition was handed. The other three references are the raw-power ones every
store carries, so the two variants can be compared on them directly.

### Swapped filters: what the log does to the topography

To separate the effect of the log on the **spatial filter** from its effect on the
**signal**, every store is also read through its twin's filters. The twin is the same
run and electrode cut on the other power scale (`CORRCA_SWAPPED_OF`: raw <-> log).
So the raw fit's directory holds `swapped_filter_*` TF maps of the *log* fit's
filters on z-scored power, and the log fit's directory the raw fit's filters on
z-scored log power. Within one directory the data stay fixed and only the topography
changes, and the SNR grid's `swapped_filter` rows sit against the same references as
`full`.

- **Drawn by the SNR grid, not the fit.** The fits run in parallel, so when one fit
  finishes its twin may still be fitting or writing its store. The grid starts only
  after every fit is stored, and with `--masked_tf_maps` it already projects the twin's
  filters at every frequency, so it draws `swapped_filter_condition_mean_tf_maps`
  (+ `_onset`, + `participants/`) into the fit's own directory. It uses the fit's
  plotting calls and stimulus average, for the leading `--swapped_components`
  (default 2). A missing twin store is a warning, not an error.
- Component *k* is paired with the twin's component *k*, never re-matched. Each keeps
  its own sign anchor (§1), so the ASSR-anchored fits should agree in sign. The non-ROI
  fits are unanchored, so a negative r there can be a sign flip.
- `swapped_filter_topomap_correlation.csv` (in the fit's directory, one row per
  component) holds the Pearson r over channels of the two shared forward patterns
  (`pattern_r`, the topomaps) and of the two filters (`filter_r`). The figure titles
  carry `pattern_r`, and the grid logs both.
- The SNR rows cover every component, as for the other signal variants.

### Band-swapped filters: broadband <-> 30-50 Hz and 1-30 Hz

`scripts/run_corrca_band_swap.py` (job `run_band_swap.pbs`) does the same swap
across the frequency cut instead of the power scale. It is read-only and pairs each
broadband store with its band twin (`--band 30-50hz` or `--band 1-30hz`; the job runs
both, or `qsub -v BANDS=1-30hz` for one), with the same run, electrode cut and
`--n_pca`. The broadband fit's directory gets `band_swapped_filter_<band>_*` TF maps of
the band fit's filters on broadband input, at every frequency. The band's inner edge
(30 Hz for both bands) is marked there beside the 40 Hz line. The band fit's directory
gets `<band>_band_swapped_filter_*` maps of the broadband filters on the band input.
Each directory gets the `..._topomap_correlation.csv` (`pattern_r`, `filter_r`, as
above), and `results/assr/corrca_band_swap__*__<band>__*.csv` collects every pair of a
run.

The z-score is per (channel, frequency), so the band fit's input is the broadband
input at the band's bins exactly. The broadband filters on the band input therefore just
reproduce the broadband fit's own sources there, on the band's colour scale. The
informative direction is the band filters read at every frequency. For `30-50hz` it shows
what a topography chosen on gamma alone picks up below 30 Hz. For `1-30hz` it shows the
reverse: whether a topography chosen on delta to beta also carries the 40 Hz steady state
above the edge, and so whether the low-frequency onset response and the 40 Hz response
share one spatial map.

The same job then runs the SNR grid on that informative direction:
`run_corrca_snr_grid.py --swap_band <band>` reads every broadband store through its band
twin's filters at 40 Hz and tests the `band_swapped_filter_<band>` rows against the
store's usual references, with the same selections, windows and three test families as
the store's own grid (`qsub -v STEPS=snr` runs this part alone). Its figures sit in the
broadband fit's `snr_grid/` as `*__band_swapped_filter_<band>.png`. Its results go to
`results/assr/corrca_band_swap_snr__<variant>__<band>__pca<N>*`, kept apart so the
store's own `assr_snr_grid__*` files are never overwritten. The TF maps stay
`run_corrca_band_swap.py`'s, so this pass reads only the 40 Hz bins.

### The SNR grid and onset TF maps on another baseline

Every trial of the grid is normalised by its pre-stimulus window: its own mean is
subtracted, and it is divided by the SD pooled over trials (the ROI reference is
normalised the same way). The stimulus-averaged TF maps (`*_condition_mean_tf_maps_onset`)
are the median over trials of the same reading at every frequency. By default the window
is the whole pre-onset span, −300 to 0 ms. `run_corrca_snr_grid.py --baseline=-0.3:-0.05`
uses −300 to −50 ms instead, which keeps the samples just before the onset out of the
baseline. Write it with `=`, because an argument that starts with a minus would
otherwise be read as an option.

Its figures go to `baseline_-300_-50ms/` inside the fit's directory, which is laid out
like the fit's directory itself:

- `snr_grid/<selection>/<window>/`: the grid figures, with the default's file names.
- `condition_mean_tf_maps_onset.png` and `participants/onset_*`: the stored sources, every
  component.
- With `--masked_tf_maps`, also `roi_masked_*` and `non_roi_masked_*` (every component),
  `swapped_filter_*` and, with `--swap_band`, `band_swapped_filter_<band>_*` (CC 1-2,
  with the 30 Hz band edge marked). A band-swap run draws only its own reading.

The results go to `assr_snr_grid_baseline_-300_-50ms__*` (and
`corrca_band_swap_snr_baseline_-300_-50ms__*` with `--swap_band`), so the default
grid is not touched. No `tf_baseline` file is written, because the maps are drawn instead.

Only the per-participant medians of the normalised trials are stored, not the trials, so
this re-reads the cache. The whole-recording TF maps (`condition_mean_tf_maps.png`) do
not depend on the baseline and are not redrawn. The read-only job
`run_snr_grid_baseline.pbs` runs this for every run: on the broadband and 30-50 Hz
stores, and on both band-swap readings. Use `qsub -v BASELINE=lo:hi` for another window,
`VARIANTS=<one run>` to split the runs across jobs, and `TF_MAPS=0` to read the cache at
the 40 Hz bins only.

### Raw voltages through the topographies

`scripts/run_corrca_raw_weighted.py` (job `run_raw_weighted.pbs`) is also read-only.
It multiplies the **raw concatenated voltages** electrode by electrode with four
weightings:

- `|CC 1|` and `|CC 2|`: each fit's **shared forward pattern** (the
  `corrca_shared_topomaps`) in absolute value. The patterns were fitted on wavelet
  power, so their sign says where power co-varies, not where the voltage is positive;
  only the magnitude, "where the component lives", is used. An ROI or non-ROI fit
  weighs every electrode it did not see at zero.
- the **binary ASSR ROI**: 1 on the ASSR electrodes, 0 elsewhere.
- **uniform** over every electrode, all ones. This is the reference: its readings are
  stage 00's.

Per recording, the voltage is **z-scored** per channel along the whole recording, cut
at every `fam+` onset, and every electrode is **multiplied by its weight** (the
topography filter). The weighted trials are then read two ways:

- **PC1**: the PCA over channels of the weighted trial mean, by stage 00's own
  `channel_pca_components` (`run_assr_raw_pca.py`), gives the PC1 direction, and
  every weighted trial is projected onto it. The sign is aligned by `align_pc1_signs`
  over every recording of both conditions together, so a condition difference is never
  a sign flip.
- **all data**: the mean over all electrodes, no component separated. With uniform
  weights this is stage 00's channel mean of the z-scored voltage.

Either signal's trials are then normalised by `baseline_normalise` (each trial minus
its own pre-onset mean, divided by the pre-onset SD pooled over the recording's
trials), as the other onset figures are, and averaged. The onset normalisation acts on
the read-out signal only, after the weighting, so every electrode enters on its z-score
and the weights are the only per-electrode scaling. Normalising each channel against
its own baseline first would rescale every electrode again and override the z-score.
In pre-onset SD units neither curve depends on the weights' overall scale.

The data are average-referenced, so at every sample the electrodes sum to zero, and an
evoked response is a dipole (negative fronto-centrally, positive over the mastoid and
occipital sites). The all-data mean of a weighting covering both poles, uniform and
`|CC|` alike, cancels much of its amplitude; a single-sign region such as the binary
ROI keeps it. PC1's loading carries a sign per electrode and follows the dipole
instead. Reading both against their own pre-onset noise puts them on one scale.

Figures go to a `raw_weighted/` subdirectory of each fit's directory, for each readout
`<r>` (`pc1`, `mean`):

- `[<band>_]raw_weighted_<r>.png`: rows are the conditions, their group means and the
  paired Psilocybin − Placebo difference; columns are the weightings.
- `..._overlay.png`: the uniform group mean against each other weighting, per
  condition.
- `..._differences.png`: the paired condition difference of each weighting against
  the uniform one's, then, per condition, each weighting minus uniform.
- `participants/`: one grid per weighting.

All figures of one call share two y-ranges, over both readouts: one for the curves and
one for the differences. The job reads all three runs in one call, so this holds
across every fit, band and run. `--ylim` / `--ylim_diff` pin them by hand. Every panel
marks the stimulus onset (dashed) and offset (dotted). The uniform reference is a
dashed dark-grey line in the overlays and has a grey panel face where it has its own
panel.

## 6. Not here yet

- **Cross-validated ISC.** Fit on all but one participant and score the held-out one.
  This would give an absolute ISC level free of the fit bias.
- **A time-shift surrogate null for the group ISC.**
- **Time-resolved ISC on the component signals.** Sliding windows, to locate *when* each
  condition is synchronised.
- **Projecting one condition through the other's filters.** The individual-condition
  fits give each condition its own; scoring Psilocybin through the Placebo filters is
  not done yet.
