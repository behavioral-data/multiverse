#!/bin/bash
NJOBS=`grep -c . $1` 
echo "Running experiments in $1"
sbatch --gres=gpu:"$NJOBS" --export=specs=$1  experiments/experiment_runner.sh 
