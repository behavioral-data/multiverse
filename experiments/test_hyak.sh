#!/bin/bash
#
#SBATCH --job-name=test-hyak
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
#SBATCH --error=/gscratch/bdata/mikeam/RobustDataScience/logs/test.err
#SBATCH --error=/gscratch/bdata/mikeam/RobustDataScience/logs/test.log
#SBATCH --export=all
export PATH=$PATH:/gscratch/bdata/mikeam/anaconda3/bin
source ~/.bashrc
conda activate RobustDataScience
export WANDB_MODE=dryrun
python src/models/CORAL_BART/finetune.py\
        ./data/processed/mixed.jsonl\
        --learning_rate 3e-5\
        --task multi\
     	--max_size 100\
        --encoder_attention_heads 8\
        --decoder_attention_heads 8\
        --encoder_layers 12\
        --decoder_layers 12\
        --max_length 512\
        --per_device_train_batch_size 8\
        --per_device_eval_batch_size 10\
        --eval_fraction 0.1\
        --logging_steps 500\
        --eval_steps 50\
        --save_steps 50\
        --logging_dir ./wandb\
        --logging_steps 3000\
        --num_train_epochs 30\
        --predict_spans \
	--wandb
