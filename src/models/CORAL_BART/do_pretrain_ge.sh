
export CUDA_VISIBLE_DEVICES=0,1
python src/models/CORAL_BART/pretrain.py \
                --output_dir ./models/CORAL_BART/pretrained-lr\
                --model_type bart\
                --mlm \
                --mlm_probability 0.3\
                --tokenizer_name /projects/bdata/tmp/tokenizer/\
                --do_train \
                --do_eval \
                --big_file_path_kaggle /projects/bdata/jupyter/gezhang_backup/RobustDataScience/dataset/kaggle_cells_parsable.json\
                --max_length 512\
                --encoder_attention_heads 8\
                --decoder_attention_heads 8\
                --encoder_layers 12\
                --decoder_layers 12\
                --learning_rate 1e-4\
                --num_train_epochs 20\
                --save_total_limit 2\
                --save_steps 4000\
                --per_device_train_batch_size 6\
                --evaluate_during_training\
                --overwrite_output_dir\
                --logging_dir ./logs/\
                --eval_steps 4000\
                --logging_steps 2000\
                # --model_name_or_path ./models/CORAL_BART/pretrained-lr/checkpoint-400
                # --weight_decay 0.01
                # --logging_first_step
                # --eval_steps

                # --output_dir ./models/CORAL_BART/pretrained-lr\
