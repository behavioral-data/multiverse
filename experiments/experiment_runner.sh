#!/bin/bash
#
#SBATCH --job-name=experiments
#SBATCH --account=bdata
#SBATCH --partition=bdata-gpu
#
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=100:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikeam@cs.washington.edu

#SBATCH --chdir=/gscratch/bdata/mikeam/RobustDataScience
#SBATCH --error=/gscratch/bdata/mikeam/RobustDataScience/logs/experiments.err
#SBATCH --output=/gscratch/bdata/mikeam/RobustDataScience/logs/experiments.log
#SBATCH --export=all

export PATH=$PATH:/gscratch/bdata/mikeam/anaconda3/bin
source ~/.bashrc
conda activate RobustDataScience
export WANDB_MODE=dryrun
NJOBS=`grep -c .  $specs` 
module load parallel-20170722
parallel -j  $NJOBS  --colsep '\t' 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) eval python src/models/CORAL_BART/finetune.py {=2 uq=} &> ./logs/baseline_{1}.out' :::: $specs 
