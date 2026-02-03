#!/bin/bash  
#PBS -N data-preprocessing-psilo-music (name of PBS job)
#PBS -o outputfile.log (name of the output log file)
#PBS -q qprodu (selects which queue to submit the job to)
#PBS -l walltime=18:00:00 (Maximum runtime in hh:mm:ss format. The job will be automatically terminated if it exceeds this time)
#PBS -l nodes=1:ppn=64 (Request one node with 64 processing cores)
#PBS -j oe (joins the standard output and the standard error files into a single log file)
#PBS -V (Exports all environment variables from your current shell to the job environment)
#PBS -m abe (emails you in case of beginning of the script (b), end of the script (e), abortion of script (a))
#PBS -M beinhauer@cs.cas.cz (Where the email will be sent)

cd $PBS_O_WORKDIR

singularity exec /home/dbeinhauer/psilocybin-eeg python scripts/run_dataset_preprocessing.py