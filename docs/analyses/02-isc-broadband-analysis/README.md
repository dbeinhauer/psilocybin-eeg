# Stage 02 — Inter-subject correlation

How much of a listener's response is shared with the rest of the cohort, measured on the
time-domain recordings. `scripts/run_isc.py`.

---

## The three measures

All operate on `(subjects, channels, samples)` and all are computed per channel.

- **Leave-one-out ISC** — each subject's series against the mean of the others. One value
  per subject per channel. The default reading, because the comparison is a subject
  against *the group* rather than against one other person.
- **Pairwise ISC** — every subject against every other. A `subjects x subjects` matrix per
  channel; the diagonal is meaningless and the matrix is symmetric, so only the upper
  triangle carries information. Useful for spotting a subject who correlates with nobody.
- **Sliding-window ISC** — LOO-ISC recomputed in overlapping windows, giving a time course
  rather than one number, so a response that is shared only during part of the stimulus is
  visible.

`src/analysis/isc.py` holds all three; they are pure functions over arrays.

## Broadband and per band

The broadband pass uses the recording as it is. The band pass filters to each of delta,
theta, alpha, beta and gamma (`src/definitions/frequency.py`) and repeats every measure,
because synchrony is not band-flat — a cohort can be tightly coupled in alpha and
uncorrelated in gamma.

Band figures carry the band as the first token of the filename, which is what lets the
plot tree hold both passes side by side.

## Windows

`--window_sec` sets the sliding window and `--window_fine_sec` a shorter one for a
finer-grained view; windows overlap by half their length. A window must fit inside the
recording: at 250 Hz the 15 s default needs 3,750 samples, so a subset run needs
`--n_times` well above that or a shorter window.

## The mean-field reading

Averaging across electrodes *before* correlating gives one series per subject, and the ISC
of those. It answers a different question from the channel-wise mean — whether the
cohort's global response is shared, rather than whether any particular location is — and
the two can disagree.

## Outputs

```
plots/02-isc-broadband-analysis/{Condition}_{Music}/{broadband|bands}/{loo_isc,pairwise_isc,sliding_window,...}/
```

## Caveats

- ISC needs a cohort. With six subjects a LOO mean is four other people, and the estimate
  is noisy; subset runs are for checking the workflow, not for reading a result.
- The measures assume the recordings are aligned sample for sample. That is what stage 00
  guarantees; an ISC computed on unaligned data measures the misalignment.
