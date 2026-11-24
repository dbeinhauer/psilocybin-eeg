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
| Decompose on Placebo, read both, compare, store: subject axis (§2) | `scripts/run_corrca_condition_comparison.py --fit_condition Placebo` | `run_placebo_fit.pbs` |
| SNR / contrast grid on the stored components (§5) | `scripts/run_corrca_snr_grid.py` | the same job, second step |
| Compare fits across `n_pca`, bands or joins, read-only (§5d) | `scripts/run_corrca_comparison.py` | `run_comparison.pbs` |

**By default a job fits only the full montage on raw power**, and its SNR grid reads only
that store's own sources (`full`). The other fits and readings are opt-in, with the same
flags for the fit scripts and the readers (`qsub -v FITS="..."` in every job, any of the
four):

| Flag / `FITS` entry | Adds |
|---|---|
| `--log_power` / `log_power` | the log-power twin of every selected fit (`_log`, §5c); in the grid, the `swapped_filter` rows |
| `--roi_fit` / `roi_fit` | the fit on the ASSR ROI alone (`_roi`, §5b) |
| `--non_roi_fit` / `non_roi_fit` | the fit on every other electrode (`_non_roi`, §5b) |
| `--roi_masked` / `roi_masked` | the full fit's `roi_masked_*` / `non_roi_masked_*` maps; in the grid, those rows (`--signal_variants`) |

With all of them a job fits six times from one data load, in this order: the full
montage, the full montage on log power, the ASSR ROI alone, every other electrode, then
`_log_roi` and `_log_non_roi`, and the SNR grid reads the stores in the same order.

Figures draw at most five components (`MAX_DRAWN_COMPONENTS`), whatever `--n_pca`; the
tables and stores keep every one. The stability figures (influence, held-out ISC, the
grid's robustness and per-participant courses) and the swapped and band-swapped TF maps
draw the leading three (`STABILITY_COMPONENTS`), fewer only when the fit has fewer.

The fit runs with `--skip_stored`: a fit whose store for that
`--n_pca` and band already exists is read back instead of refit, and every figure is
still drawn from it. The data is loaded either way, for the masked maps and any fit
still missing. A store fitted on other data (cohort, channels, time extent: the store
name does not record `--n_times` or `--n_pairs`) is refused, not reused.
`qsub -v FORCE_FIT=1` refits every selected fit; `qsub -v N_PCA=<n>` picks the component count
(default 5).

The selected fits run in parallel, each in its own forked process that reads the one loaded
tensor without copying it. `--parallel_fits P` is a memory budget in full-montage fits:
fits start in plan order while the running ones weigh at most `P`, where a cut fit
weighs its share of the channels (ROI ~0.2, non-ROI ~0.8). A running fit holds about
three copies of its prepared input at its peak (the input, its z-scored copy and one
temporary). The broadband joins run with `P = 2`, the individual conditions and the
gamma jobs with `P = 3` (`qsub -v PARALLEL_FITS=<P>`); each job's header gives the
memory estimate. `P = 1` (the script default) runs the fits one after another in one
process. The jobs keep the memory and CPUs every fit needs, so any `FITS` selection
fits; only their walltime is sized for the default full fit, and each header names the
walltime to request with `FITS`.

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
  `A = R_w W`, stored as `corrca_shared_patterns` but not drawn. Each subject also
  gets a descriptive pattern, `A_s = R_ss W (Wᵀ R_ss W)⁻¹`: the shared component seen
  through that subject's covariance. Those per-subject patterns feed the per-participant topographies.
  Because they share one filter, they agree with each other more than IVA's free
  per-recording topographies do. Don't read that agreement as evidence of synchrony.
  The filter `W` itself — the channel weights that, applied to each subject's
  time-z-scored power, give the TF maps — is drawn as `shared_spatial_topomap_filter`.
  It is what produces the maps, not where the component sits on the scalp: a filter
  also weights electrodes negatively to cancel noise, so read the pattern for location.
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

**`corrca_channel_placebo_fit`** sits between the two: the subject-axis join, loaded
and drawn exactly as `corrca_channel_joined_separate`, but with the filters fitted on
the **Placebo recordings alone** (`--fit_condition Placebo`). `R_w` and `R_b` are summed
over the P Placebo recordings only, and every recording of both conditions is then
projected through the resulting filters (`decompose_channel_corrca(..., fit_rows=...)`).
So there is one set of filters, as in the join, but only Placebo chose it, and
`CC k` is the same spatial combination in both rows of every figure:

- **What it asks.** How do the Placebo components behave under Psilocybin? The
  Psilocybin recordings never shape the filters, so their ISC, TF maps and SNR are
  an out-of-sample read-out of the Placebo components. Each Psilocybin recording still
  gets its own forward pattern `A_s = R_ss W (Wᵀ R_ss W)⁻¹`, through its own covariance.
- **The bias is one-sided.** Placebo is read through filters fitted on it, so its ISC
  is optimistic and Psilocybin's is not. Unlike the pooled fit, the bias is not shared
  by the two conditions, and a Placebo > Psilocybin difference is expected even with
  no drug effect. The fit's own ISC (`corrca_fit_isc`), its shared patterns and the
  sign anchor are Placebo's.
- **Influence and refits on Placebo.** The leave-one-participant-out check (§3) and
  the leave-subset-out refits (§5) refit on the Placebo recordings minus the left-out
  participants, and the influence figures read the Placebo recordings. The refits are
  still written for the SNR grid, which reads both conditions through them. Its
  `held_out` rows read each participant's two recordings through a refit that saw
  neither, so they are the contrast free of the one-sided bias.
- **Same products.** The six fits (`_log`, `_roi`, `_non_roi`, ...), the masked and
  swapped-filter maps and the SNR grid are the join's. They are drawn under their
  own label, `JoinedPlaceboFitted_<MusicType>/<spectrum>/corrca_channel_placebo_fit*/`,
  and stored beside the pooled fit under `iva_results/JoinedSeparate/` (the store
  holds the join's rows, so it keeps the join's condition). The store adds the extra
  `corrca_fit_condition` (`"Placebo"`).

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

**Onset-locked synchrony.** The same three read-outs scored on the onset-locked maps
(`condition_mean_tf_maps_onset`, each trial referenced to its pre-onset window and in
units of that recording's pre-onset SD, median over trials) go to
`corrca_onset_condition_isc.png` and its three CSVs. The whole-recording score
correlates time courses: the time z-scoring leaves every frequency row zero-mean, so
it is the stimulus-locked rise and fall of power pooled over frequencies. The onset
score instead correlates the **shape of the evoked TF response**, including where in
frequency it sits and its contrast with the baseline. It answers whether participants
share the same averaged stimulus response, which is the closer question for the ASSR
paradigm. A participant whose dot sits far above the rest while the others are near 0
marks a component carried by a few people. These maps are not refitted, so the
in-sample bias below applies to them too.

**Participant influence.** Whether a component is shared by the group or carried by a
few participants. Each participant is left out in turn (both recordings in the
subject-axis join), CorrCA is refitted on the rest with the same `n_pca` and `gamma`,
and the refit's components are matched to the full fit's by |r| of the shared
patterns (Hungarian assignment). One pass over the data gives every refit in closed
form (`corrca.jackknife_statistics`): with `Σ` the sum over recordings and `y_g` the
left-out participant's, `(Σ - y_g)(Σ - y_g)ᵀ = R_t - (G + Gᵀ) + H`, built from each
recording's cross-covariance with `Σ` and with its own participant. Per component:

- **ISC lost** (`influence`): full-fit ISC minus the matched refit's, with the
  **topography |r|** (`pattern_r`) between them. A component carried by a few has those
  few losing much of its ISC and moving its topography.
- **In-sample vs held-out ISC** per recording: its correlation with the sum of every
  *other* participant, through the full filter (which it helped choose) and through
  the refit's (which never saw it). The gap is the recording's own pull on the filter.
- **Own-pair ISC** (`within_participant_isc`): the recording against the same
  participant's other condition, through the full filter. The between-recording term
  of the subject-axis join rewards this pair as well, so a component can rest on
  participants resembling *themselves* across conditions (a stable individual
  topography) while every score against others stays near zero. The per-condition ISC
  above cannot show this, because the pair spans the two conditions.

- **The two terms of the ISC** (`within_*`, `between_*`): `wᵀR_w w` and `wᵀR_b w`
  per unit filter norm (§5d), for the full fit (`_full`), the full fit's filter on the
  data without the participant (`_fixed`) and the refit (`_without`, whose ratio is
  `isc_without`). They say whether leaving someone out moves the ISC through the
  variance the component carries or through its shared covariance, and whether
  removing the data or rotating the filter does it.

A participant is flagged as an outlier when their loss has a robust z (median/MAD over
participants) above 3.5 **and** costs at least 10% of the component's ISC. The share
floor keeps a negligible loss from being flagged where every loss is near zero. On
components whose ISC is itself near zero, the flag is meaningless. The results go to
`influence/corrca_influence.png`, `influence/corrca_heldout_isc.png` and
`influence/corrca_influence.csv`, plus one log line per component. The figures and
log lines cover CC 1-3 only (`STABILITY_COMPONENTS`, as every stability
figure); the CSV keeps every component. In the held-out figure a flagged recording's
line is red and named at its held-out reading. `--skip_influence` skips all of it.
The cost is one extra chunked pass over the data and one small eigenproblem per
participant.

**Onset-locked influence.** The same read-outs on each recording's onset-locked,
trial-averaged TF map, the signal `corrca_onset_condition_isc` scores, so the
influence can be compared on the stimulus response the ASSR question is about. The
covariances cannot score a trial average, so the full fit's and every
leave-one-out refit's filters for CC 1–3 are applied to the input in one projection,
averaged around the onsets with the fit script's own call, and the ISCs taken on
the maps (`corrca.map_influence`). The ISC lost is then the full maps' group ISC
minus the group ISC of the remaining participants' maps through the refit; the
topography |r| is the refit's, as above. They go to `influence/corrca_onset_influence.png`,
`influence/corrca_onset_heldout_isc.png` and `influence/corrca_onset_influence.csv`.
The projection holds `(recordings, 3 + 3·participants, F, T)` in float32, about 9 GB
for the broadband join.

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
        condition_mean_tf_maps.png, condition_mean_tf_maps_onset.png
        shared_spatial_topomap_filter.png
        corrca_condition_isc.png
        corrca_condition_isc.csv, corrca_participant_isc.csv, corrca_condition_isc_test.csv
        corrca_onset_condition_isc.png (+ the same three CSVs, corrca_onset_ prefix)
        influence/
            corrca_influence.png, corrca_heldout_isc.png, corrca_influence.csv
            corrca_onset_influence.png, corrca_onset_heldout_isc.png,
            corrca_onset_influence.csv
        participants/*_cc<k>_participants_*.png
plots/06b-corrca-condition-comparison/IndividualConditions_<MusicType>/
    <broadband|bands>/corrca_channel_individual_conditions/<Condition>/pca_<n>/
        the same, with no _test.csv
plots/06b-corrca-condition-comparison/JoinedPlaceboFitted_<MusicType>/
    <broadband|bands>/corrca_channel_placebo_fit/pca_<n>/
        the same as the joins
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
for CC 1-3 and the `ASSR-mask (PCA, z)` reference (what the fit was handed;
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

### Robustness of the contrast to the participants

The `contrast` test asks whether Placebo − Psilocybin differs over 12 pairs. With so
few, a significant result can rest on a handful of participants in two ways: through
the **test** (a few large differences carry the Wilcoxon) or through the **filter** (a
few participants shaped the CorrCA topography that produces everyone's value). The
grid checks both on the `full` reading of every cell (`--skip_robustness` skips it):

| Test | What changes | Asks |
|---|---|---|
| `full` + `sign_p` | nothing; exact sign test beside the Wilcoxon | does the verdict rest on how many point the same way, or on how far a few do? |
| `fixed_drop` | every subset of `--drop_sizes` (1, 2, 3) participants left out, the filters kept | does the verdict hang on whose differences are in? |
| `refit_drop` | CorrCA refitted without each subset the fit wrote, every remaining participant read through the refit | does it hang on the participants that shaped the filter? |
| `held_out` | each participant read through the refit that left them alone out | is the effect there when no value comes from a filter its own data helped choose? |

The refits come from the fit script, not the grid. With `--store_components` and two
conditions, the influence check also refits CorrCA without every subset of
`--refit_subset_sizes` participants (default 1, 2 and 3: 12 + 66 + 220 refits for
12 participants). Each refit's components are matched to the full fit's (Hungarian on
|r| of the patterns) and signed so their pattern agrees with the full one. The
filters are written beside the store as `<store>__subset_refits.npz`. Leaving several
participants out at once needs the cross terms between every pair of participants
(`jackknife_statistics(..., group_cross=True)`), which about doubles the influence
pass on the subject-axis join and triples it on the time-axis one. The grid reads the
refits' values straight from the cache, as the fit's input was z-scored, and logs how
closely the full fit read that way matches its stored sources. A refit file whose full
filters differ from the store's is ignored, so a stale file is never read.

- **Fragility**: the fewest participants whose removal changes the verdict at
  `--alpha`. For a non-significant component it is the fewest whose removal makes it
  significant.
- **Flip shares**: per participant, the share of the verdict-changing size-k subsets
  that leave them out. By chance every participant has `k / P` (0.25 for 3 of 12). A
  participant far above it is among the few the verdict rests on.

Outputs, beside the grid CSV:

- `<grid csv stem>__robustness.csv`: every replicate, with columns `selection`,
  `window`, `source`, `test`, `k`, `dropped`, `n`, the `paired_test` columns,
  `sign_p` and `significant`.
- `__robustness_summary.csv`: per source, test and k, with `n_significant`,
  `n_flips`, `max_p` with `worst` (the subset at it), the median range, `full_p` and
  `fragility`.
- `snr_grid/<selection>/<window>/robustness__full.png`, for CC 1-3. From the
  top: the replicate p-values per family and size (red where the verdict changes,
  ringed where the refit's topography is flagged, held-out last). Then one row per
  subset size with the flip shares per participant, fixed and refit side by side
  against the `k / P` chance line. Last, the |r| of every refit's topography with the
  full fit's, per size. A refit is flagged (red, the lowest five per size named by
  their left-out participants) when its |r| is a robust low outlier among its size
  (modified z below −3.5) **and** at least 0.05 under their median
  (`corrca.low_topography_flags`). A flagged refit found a different spatial map, so
  its contrast reads another component.
- The `refit_drop` rows of the CSV carry `pattern_r` and `filter_r`, the refit's
  topography and filter against the full fit's.
- One log line per component and cell.

## 5b. The binary ROI

Two ways of asking what the CorrCA components carry from the ASSR electrodes alone:

- **ROI- and non-ROI-masked maps of the full fit.** With `--roi_masked`, every fit script draws
  `roi_masked_condition_mean_tf_maps.png` (and `_onset`, and `participants/roi_masked_*`):
  the fitted shared filter with every weight outside the ROI zeroed — not renormalised —
  applied to the z-scored signal the fit saw; then `non_roi_masked_*`, the same with the
  ROI zeroed instead. The two add up to the full maps. They are the TF-map counterpart
  of the grid's `roi_masked` / `non_roi_masked` rows. Only with `--roi_masked`.
- **A fit on the ROI alone.** With `--roi_fit`, a run makes a second fit from the same loaded
  tensor, with the channel axis cut to the ROI **before** the z-scoring, so ROI → z-score
  → PCA → CorrCA. It is stored and drawn as `corrca_channel_joined_{separate,tracks}_roi`,
  beside the full-montage fit, and a grid run with `--roi_fit` reads that store after the full one.
  There `roi_masked` coincides with `full` by construction, a free check that the store
  and the cache agree. The ROI fit is skipped, with a warning, when the montage holds no
  ASSR electrode or fewer than `--n_pca` of them. It runs only with `--roi_fit` (fit and
  grid alike).
- **A fit on the rest of the head.** The same cut to every electrode BUT the ROI, stored
  and drawn as `<join>_non_roi`, third. It has no anchor electrode, so it keeps the
  fit's own orientation (largest shared-pattern weight positive). The grid reads its
  store on the full montage, with the filters zero on the ROI, so its references read
  the same electrodes as every other store's; only its `full` rows are tested, and
  only against `ASSR-mask (full)`, since a reference read through its subspace is
  identically zero. It runs only with `--non_roi_fit` (fit and grid alike).

## 5c. Log-power variant

With `--log_power`, a run also fits on log power: `log10` of the wavelet power (floored at the dtype's
smallest normal number, as in stage 08), taken **before any z-scoring**, including the
time-axis join's per-condition z-scoring. A component is then fitted on relative power
changes rather than absolute ones, which the strongest bursts dominate. Everything
else is the raw-power fit's: the same cuts, the masked maps, the figures and the SNR
grid. The products are stored and drawn as the `*_log` variants beside the raw ones
(`corrca_channel_joined_tracks_log`, `..._log_roi`, `..._log_non_roi`, and likewise for
the subject-axis join and the individual conditions), one per selected electrode cut;
the grid reads their stores with the same flag.

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
  (default 3). A missing twin store is a warning, not an error.
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
both, or `qsub -v BANDS=1-30hz` for one). The job reads only the runs in `RUNS`
(default `tracks`; any of `tracks separate placebo_fit individual`) at `N_PCA`
(default 5), and of each only its full fit (full montage, raw power); the twin has the
same run, electrode cut and `--n_pca`. The broadband fit's directory gets `band_swapped_filter_<band>_*` TF maps of
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
  `swapped_filter_*` and, with `--swap_band`, `band_swapped_filter_<band>_*` (CC 1-3,
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
  stored `corrca_shared_patterns`) in absolute value. The patterns were fitted on wavelet
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

## 5d. Comparing fits: `n_pca`, bands, joins

`scripts/run_corrca_comparison.py` (job `run_comparison.pbs`, notebook
`corrca_comparison_{pca,band,join}.ipynb`, one per comparison) is read-only. It sets finished fits side by side, only the
full-montage raw-power fit of each run, along one of three axes:

| `--compare` | Fixed | Compared |
|---|---|---|
| `pca` | `--run`, `--band` | every `--n_pca` (default 3 5 10 20) |
| `band` | `--run`, `--n_pca` | `--bands` (default broadband, 1-30hz, 30-50hz) |
| `join` | `--band`, `--n_pca` | `--runs` (default separate, tracks, individual) |

The leading `--components` (default 3) are compared **by index**. CC 1 at `n_pca = 5`
and CC 1 at `n_pca = 20` are sought in different subspaces, so they need not be the
same component. The topography figure gives each map's r with the first fit's same
component, so a re-ordering shows. An individual-conditions run holds two fits with
different filters, drawn as two rows and never paired.

What is compared comes from the products each fit already wrote: its ISC (the fit's
own, and per condition on the whole recording and on the onset maps); the held-out ISC,
the in-sample − held-out gap and the refit topography |r| (median and worst) from the
influence table (§3); the uncorrected Placebo/Psilocybin p-values (LOO-ISC whole and
onset, and the grid's `contrast` family on `full` at `--snr_selection` for every
`--snr_windows`); the grid's SNR courses; and the shared filters and forward patterns.

**Reading `n_pca`.** A larger subspace always raises the in-sample ISC: the whitening
divides by `√eigenvalue`, so the filters can use directions of little variance that
correlate across people by chance. The held-out ISC and the in-sample − held-out gap
are the guard. A value of `n_pca` that raises the held-out ISC found real structure;
one that only widens the gap fits noise. With `corrca_channel_placebo_fit` the
optimism is one-sided (§2), so a widening gap there also biases the contrast.

**The two terms of the ISC.** `ISC = wᵀR_b w / wᵀR_w w` cannot tell a weaker shared
response from a stronger unshared one. The comparison therefore reads both terms per
condition from the stored sources (`corrca.condition_variance_terms`). Per participant:

- `within`: the variance of the participant's component signal, their term of `R_w`;
- `between`: its mean covariance with every other participant of the condition, their
  term of `R_b`.

Both are divided by `‖w‖²`. On power z-scored per (channel, frequency) they are then in
units of one channel's variance, independent of the filter's arbitrary scale, so fits
of another `n_pca`, band or join share an axis. A `within` far above 1 means the filter
follows a high-variance (often global) direction; far below 1, it cancels shared
variance between electrodes. The condition means' ratio is the condition's ISC
exactly. Both terms are tested Psilocybin − Placebo like the LOO-ISC (paired
Wilcoxon, uncorrected). The shift figure reads the `within_*` / `between_*` influence
columns (§3) as `Δwithin / within` and `Δbetween / within` (the latter in ISC units).
An influence table written before those columns existed is skipped with a warning,
and rerunning the fit job adds them.

The stores are read with only the compared components, one recording at a time
(`load_iva_components(..., components=)`). A missing store, CSV or grid leaves a gap
and a warning. A 1-30 Hz fit has no SNR grid. A band store's grid file is named
`__bands__` whatever its band, so the 30-50 Hz grid is the one read.

Outputs, in `plots/06b-corrca-condition-comparison/Comparison_<MusicType>/`:
`<spectrum>/pca__<variant>/`, `band__<variant>/pca_<n>/` or `<spectrum>/join__<runs>/pca_<n>/` (the runs joined by
`-`, e.g. `join__separate-placebo_fit-tracks`),
each with `corrca_comparison_metrics.{png,csv}`, `_topomap_filter.png`,
`_topomap_pattern.png`, `_snr_course.png`, `_variance_terms.{png,csv}` (+
`_variance_terms_tests.csv`) and `_variance_shift.{png,csv}`.

## 6. Not here yet

- **A time-shift surrogate null for the group ISC.**
- **Time-resolved ISC on the component signals.** Sliding windows, to locate *when* each
  condition is synchronised.
- **Projecting Placebo through the Psilocybin filters.** `corrca_channel_placebo_fit`
  scores Psilocybin through the Placebo filters (§2); the reverse direction is not
  wired (`CORRCA_SEPARATE_FIT_RUNS` has no Psilocybin run).
