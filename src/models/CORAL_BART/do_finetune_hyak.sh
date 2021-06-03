#!/bin/bash
#
#SBATCH --job-name=CORAL-Multitask
#SBATCH --account=bdata
#SBATCH --partition=cse-gpu
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikeam@cs.washington.edu

#SBATCH --chdir=/gscratch/bdata/mikeam/RobustDataScience
#SBATCH --error=/gscratch/bdata/mikeam/RobustDataScience/logs/multitask.err
#SBATCH --error=/gscratch/bdata/mikeam/RobustDataScience/logs/multitask.log
#SBATCH --export=all

conda activate RobustDataScience
export WANDB_MODE=dryrun
python src/models/CORAL_BART/finetune.py\
        ./data/processed/mixed.jsonl\
        --learning_rate 3e-5\
        --task multi\
        --encoder_attention_heads 8\
        --decoder_attention_heads 8\
        --encoder_layers 12\
        --decoder_layers 12\
        --max_length 512\
        --per_device_train_batch_size 8\
        --per_device_eval_batch_size 10\
        --eval_fraction 0.1\
        --logging_steps 500\
        --eval_steps 5000\
        --save_steps 10000\
        --logging_dir ./wandb\
        --logging_steps 3000\
        --num_train_epochs 5\
        --predict_spans
        --wandb
