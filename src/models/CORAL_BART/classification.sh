export CUDA_VISIBLE_DEVICES=0
python src/models/CORAL_BART/finetune.py \
                /homes/gws/mikeam/RobustDataScience/data/processed/filtered_less_than_5_lines.jsonl\
                --task classification\
                --cuda_device 0\
                --encoder_attention_heads 4 \
                --decoder_attention_heads 4 \
                --encoder_layers 4 \
                --decoder_layers 4 \
                --max_length 128 \
                --per_device_train_batch_size 30 \
                --per_device_eval_batch_size 150 \
                --eval_steps 10000 \
                --eval_fraction 0.05 \
                --output_dir ./results/finetuned_only \
                --save_eval \
                --wandb \
                --predict_spans \
                --deterministically_shuffle_dataset\
                --num_train_epochs 20