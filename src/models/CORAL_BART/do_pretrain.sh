
export CUDA_VISIBLE_DEVICES=0,1
python src/models/CORAL_BART/pretrain.py \
                --output_dir ./models/CORAL_BART/pretrained\
                --model_type bart\
                --mlm \
                --mlm_probability 0.3\
                --tokenizer_name /projects/bdata/tmp/tokenizer/\
                --do_train \
                --do_eval \
                --data_dir /projects/bdata/datasets/ucsd-jupyter/processed/notebooks_as_scripts_with_sep/\
                --max_length 256\
                --learning_rate 1e-4\
                --num_train_epochs 5\
                --save_total_limit 2\
                --save_steps 2000\
                --per_device_train_batch_size 4\
                --evaluate_during_training\
                --overwrite_output_dir\
                --logging_dir ./logs/\
                # --logging_first_step
                # --eval_steps
