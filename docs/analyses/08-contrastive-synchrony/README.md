# Stage 08 — Contrastive synchrony model

Can a model **learn** what subjects share when they hear the same music at the same
moment, and point at the moments where that sharing is strongest?

Stages 01-07 measure synchrony with fixed recipes (ISC, variance, ICA/IVA). This stage
learns the representation instead: an encoder is trained so that it cannot tell two
subjects apart at the same moment of a track, but can tell two moments apart. Whatever it
keeps is, by construction, the part of the signal the group shares in time.

One script, one job: `scripts/run_contrastive_synchrony.py`
(`jobs/metacentrum/08-contrastive-synchrony/train.pbs`) loads, trains, scores and plots.

The code lives in `src/models/`; the figures in `src/visualization/contrastive_plots.py`.
`notebooks/08-contrastive-synchrony/contrastive_synchrony.ipynb` runs the same code on a
few subjects and minutes. This is a **baseline, built to be changed** — see §6.

---

## 1. Data

Input is the stage-03 wavelet **power** of the music recordings (1-50 Hz, 50 bins,
195 channels, 250 Hz). Within one `<condition>_<music type>` track every subject heard the
same stimulus on one time base (stage-00 alignment), so sample *t* is the same moment of
the music for everybody — the fact the whole method rests on.

**ASSR is never trained on.** It is the held-out validation set (§7), and
`data.experiment: assr` is refused by the config.

### Load (every run, from the config)

The data are read straight from the stage-03 per-subject store
(`data/processed/psilo_music/wavelets/<label>/`, one float32 file per participant; see
[stage 03](../03-wavelet-analysis/README.md)). Nothing is precomputed for this stage.
`src.models.data.build_dataset` applies, per label:

1. **Exclusions** — `data.exclusions` through `DatasetFilter`, exactly as other stages.
2. **Channels** — all, or a named list.
3. **Frames** — power is averaged into frames of `data.frame_rate` Hz (25 Hz by default,
   10 samples per frame) and then `log10`, so a frame is the log of its mean power. Stage
   03 uses `n_cycles = freq / 2`, which gives every frequency the same ~80 ms time
   resolution and limits power fluctuations to below ~8-9 Hz, so 25 Hz frames keep
   essentially all of them; 10 Hz would not. Only the bins inside `data.freq_range` are
   read, one channel at a time, participants in parallel threads (`--n_jobs`).
4. **Frequencies** — either averaged into the five `FREQUENCY_BANDS`
   (`freq_pooling: bands`, default; mean of log power = geometric mean of power) or kept
   bin by bin (`none`).
5. **Normalisation** — `normalize: subject` z-scores each subject's feature over time.
   This removes each subject's own baseline and gain, which would otherwise be the easiest
   thing for that subject's windows to agree on.
6. Channel × frequency is folded into one feature axis: `(subjects, features, frames)`.

A **window** is `data.window_s` seconds of that array (4 s = 100 frames by default) — the
unit the encoder embeds and the unit synchrony is scored on.

## 2. The self-supervised task

A batch holds `times_per_batch` moments, drawn from all tracks in proportion to their
length. Each moment is seen through `subjects_per_time` different subjects:

- **positives** — windows of different subjects at the same moment of the same track;
- **negatives** — every other window in the batch, **including the same subject at a
  different moment**. That negative is what stops subject identity from solving the task.
- **ignored** — windows of one track whose starts are closer than `exclusion_s`. Overlapping
  windows share most of their samples, so pushing them apart would punish the model for
  being right. They are left out of the softmax denominator.

The loss is the multi-positive InfoNCE of Khosla et al. (2020, *Supervised Contrastive
Learning*) with the moment as the class, on L2-normalised embeddings at
temperature `training.temperature`.

Two diagnostics are logged per epoch: **retrieval accuracy** (how often a window's most
similar other window comes from the same moment) and **chance** (the accuracy a random
encoder would reach on the same batches). Accuracy above chance on the held-out block is
the evidence that there is shared, time-locked structure to find.

## 3. Cross-fitting

**A contrastive model makes the moments it trained on synchronous by construction.** It
is trained to map every subject's window at moment *t* to one point, so scoring *t*
afterwards measures how well it fitted, not how much the subjects share. This is not a
theoretical worry. In a first 3-subject test the encoder reached training accuracy 1.0
with held-out accuracy at chance. Scored in-sample, the whole training part came out as
a single "candidate interval" at synchrony 0.997.

So no window is ever scored by a model that saw it. Each track is cut into
`training.n_folds` contiguous time blocks (5 by default, ~70-100 s each). Fold *k* trains
on all other blocks and scores block *k*:

- the training part is passed as separate segments, so no window spans the gap the
  held-out block leaves;
- blocks are cut at the same fractions of every track, so fold *k* holds out roughly
  the same stretch of a piece under both conditions (exactly only if the two tracks
  share a time base, see §7);
- each block is scored against **its own null**, because each fold model has its own
  embedding space and similarity baseline.

The held-out block also gives the diagnostic curves. Their batches are the same every
epoch, so the curve moves only because the model does. The curves are for reading, not
for choosing: the **last epoch is kept**, because selecting the epoch on the block that
is then scored would leak. Choose `training.epochs` from the curves, in the notebook.

## 4. Encoders

`model.name` selects a registered encoder; `model.encoders.<name>` holds its arguments.

- **`conv1d`** (default) — a 1×1 convolution mixes all channel × frequency features into
  `hidden` learned combinations (a spatial-spectral filter bank). `n_layers` temporal
  convolutions then read their dynamics, and global average pooling over time makes the
  embedding insensitive to where in the window an event falls. ~0.4 M parameters.
- **`mlp`** — flattens the window, so frame order is kept, at the cost of a large first
  layer.

## 5. Scoring synchrony

Each fold model embeds its held-out block, every subject, with windows `scoring.stride_s`
apart that lie wholly inside the block. For each window:

- **group synchrony** — mean pairwise cosine similarity of all subjects' embeddings;
- **null** — the same quantity after circularly shifting each subject by its own random
  lag within the block (at least one window length, so a shifted window never overlaps
  its original position), `scoring.n_null` times. Shifting keeps every subject's
  embeddings intact and destroys only their alignment in time. The null is pooled over
  the block's windows and surrogates;
- **candidate** — group synchrony above the block's `null_quantile`. Runs of candidate
  windows are merged into **candidate intervals**; a run never spans two blocks.

The per-subject table gives each participant's similarity to the mean of the others. It
shows whether a synchronous moment is shared by everybody or carried by a few people.

Outputs (`results/psilo_music/contrastive_synchrony/<run_name>/`):

| File | Holds |
|---|---|
| `config.yaml` | the fully resolved config of the run |
| `model_fold<k>.pt` | each fold's encoder weights + what is needed to rebuild it |
| `history.csv` | loss, accuracy and chance per fold and epoch |
| `synchrony.csv` | per window: synchrony, z against its block's null, threshold, candidate, fold |
| `candidates.csv` | merged candidate intervals with peak synchrony and z |
| `subject_synchrony.csv` | per participant and window: similarity to the group |

Figures go to `plots/08-contrastive-synchrony/<run_name>/`: `training_history.png` and,
per track, `<label>_synchrony.png` and `<label>_subject_synchrony.png`.

## 6. Changing the setup

Everything is in `config/models/contrastive_synchrony.yaml`, which is also the single
source of defaults. A variant file lists only what it changes and is merged on top.
`--set key=value` overrides single keys, and an unknown key is an error, not a no-op:

```bash
python scripts/run_contrastive_synchrony.py --run_name mlp --set model.name=mlp
python scripts/run_contrastive_synchrony.py --config config/models/my_variant.yaml
```

The run name is the output directory, so give each variant its own.

**A new encoder** is a class in `src/models/encoders.py`, decorated with
`@register_encoder("name")`, whose constructor takes `(n_features, n_frames,
embedding_dim, **its own arguments)` and whose `forward` maps `(batch, n_features,
n_frames)` to `(batch, embedding_dim)`. Add its arguments under `model.encoders.name` and
select it with `model.name`. Nothing else changes.

**A new representation** (frame rate, bands, channels, normalisation, window length) is a
config change; the wavelet store is read afresh every run.

**Scoring a new condition with a trained model**: `--skip_training` with
`--set "data.conditions=[Placebo, Psilocybin]"` scores the Psilocybin tracks with the
Placebo-trained encoder, without retraining.

## 7. Caveats and next steps

- **Per-track time base.** Positives are only formed within one track. The same music
  type under the two conditions is not assumed to share a time base (their lengths
  differ), so the same moment in Placebo and Psilocybin is never treated as a positive.
- **Stationarity of the null.** The pooled null assumes the embedding statistics do not
  drift along the track. A slow drift shared by everybody (fatigue, a global power trend)
  would read as synchrony. Per-subject z-scoring removes the mean, not a trend.
- **Overlapping windows** make neighbouring scores correlated, so the per-window "p-values"
  are not independent. Read candidate intervals, not individual windows.
- **ASSR validation (next).** ASSR stimuli are known, time-locked and identical across
  subjects. A music-trained encoder should therefore show higher synchrony during the
  stimulus trains than between them. The ASSR store uses the same 195 channels, so the
  trained model applies as is. What is missing is scoring a dataset that is not a training
  track, and a window short enough for the ~1 s ASSR epochs.
