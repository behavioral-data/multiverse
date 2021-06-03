conda activate RobustDataScience
export CUDA_VISIBLE_DEVICES=1
python src/models/CORAL_BART/finetune.py ./data/processed/filtered_less_than_5_lines.jsonl\
        --notes "coral_bart" \
        --learning_rate 3e-5 \
        --task multi \
        --encoder_attention_heads 4 \
        --decoder_attention_heads 4 \
        --encoder_layers 4 \
        --decoder_layers 4 \
        --max_length 128 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 30 \
        --eval_fraction 0.05 \
        --logging_steps 500 \
        --eval_steps 3000 \
        --save_steps 10000 \
        --logging_dir ./wandb \
        --logging_steps 500 \
        --num_train_epochs 20 \
        --predict_spans\
         --wandb \
        --classification_threshold 0.4 \
        --library_graph \
        ./data/processed/top_10_lib_tree_no_args \
        --span_aware_decoding \
        --graph_loss_weight 0.1 \
        --graph_loss_burn_in_epochs 1.0 \
        --pos_class_weight 10.0\
        --classification_loss_weight 5.0\
        --hidden_dropout_prob 0.5
