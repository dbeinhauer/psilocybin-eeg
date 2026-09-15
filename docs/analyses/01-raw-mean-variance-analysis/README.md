# Stage 01 — Intersubject mean-variance synchrony

The simplest group measure in the project, and the baseline the correlation-based ones are
read against. `scripts/run_mean_variance.py`.

---

## What it measures

At each time point, take the across-subject **mean** and **variance** of the signal. A
stimulus-locked response that every listener shares raises the mean and lowers the
variance; idiosyncratic activity does the opposite. The pair is therefore a crude but
assumption-free synchrony index: no correlation, no windowing, no reference subject.

`src/analysis/mean_variance.py` computes it, per channel and averaged over channels, for
the broadband signal and for each frequency band.

## Why keep it alongside ISC

It fails differently from [stage 02](../02-isc-broadband-analysis/README.md). ISC is blind
to a response shared in *shape* but differing in amplitude; mean-variance is blind to a
response shared in amplitude but differing in *timing*. Where the two agree, the
conclusion is robust to that choice; where they disagree, the disagreement is the finding.

## Outputs

```
plots/01-raw-mean-variance-analysis/{Condition}_{Music}/{broadband|bands}/{analysis_type}/
```

## Caveats

- The variance term is dominated by amplitude differences between subjects unless the data
  is normalised first; the analyser normalises by default.
- Like every group measure here, it assumes sample-for-sample alignment from stage 00.
