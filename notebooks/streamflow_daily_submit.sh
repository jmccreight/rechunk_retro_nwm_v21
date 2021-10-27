#!/bin/bash -l
#PBS -N qday
#PBS -A NRAL0017
#PBS -l select=1:ncpus=1:mem=10GB
#PBS -l walltime=05:00:00
#PBS -q casper
#PBS -j oe
#PBS -J 0-4

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

source /glade/work/jamesmcc/python_envs/379zr/bin/activate

cd /glade/u/home/jamesmcc/WRF_Hydro/rechunk_retro_nwm_v21/notebooks

team_id=$PBS_ARRAY_INDEX

python3 streamflow_daily.py $team_id

exit $?
