
export CUDA_VISIBLE_DEVICES=3
python src/models/CORAL_BART/pretrain.py \
                --output_dir ./models/CORAL_BART/pretrained-lr-backup\
                --model_type bart\
                --mlm \
                --mlm_probability 0.3\
                --tokenizer_name /projects/bdata/tmp/tokenizer/\
                --do_predict \
                --big_file_path_kaggle /projects/bdata/jupyter/gezhang_backup/RobustDataScience/dataset/kaggle_cells_parsable.json\
                --max_length 512\
                --encoder_attention_heads 8\
                --decoder_attention_heads 8\
                --encoder_layers 12 \
                --decoder_layers 12 \
                --learning_rate 1e-5\
                --num_train_epochs 2000\
                --save_total_limit 2\
                --save_steps 2000\
                --per_device_train_batch_size 1\
                --evaluate_during_training\
                --overwrite_output_dir\
                --logging_dir ./logs/\
                --eval_steps 1000\
                --model_name_or_path /projects/bdata/jupyter/gezhang_backup/RobustDataScience/models/CORAL_BART/pretrained-lr-backup/checkpoint-20000

                # --logging_first_step
                # --eval_steps
