#!/bin/bash
# Stage-06b job index: one line per job. Not a PBS job; run it locally:
#   bash jobs/metacentrum/06b-corrca-condition-comparison/help.sh
# and `bash <job>.pbs --help` for a job's parameters and example submissions.
cd "$(dirname "$0")" || exit 1
echo "Stage-06b CorrCA jobs (qsub [-v NAME=value,...] <job>):"
echo
for job in run_*.pbs; do
    # The header's third line: what the job does.
    printf '  %-38s %s\n' "$job" "$(sed -n '/^# ====/{n;n;s/^#   //p;q}' "$job")"
done
echo
echo "Parameters and examples: bash <job>.pbs --help"
echo "Common ones: N_PCA (default 5); FITS (fit jobs: log_power roi_fit non_roi_fit"
echo "roi_masked); RUNS (readers: tracks separate placebo_fit individual)."
