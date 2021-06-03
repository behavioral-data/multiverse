python finetune.py\
        /projects/bdata/jupyter/gezhang_backup/RobustDataScience/dataset/filtered_diffs_8_23.jsonl\
        --path_to_tokenizer /projects/bdata/tmp/tokenizer/\
        --output_dir /projects/bdata/jupyter/gezhang_backup/RobustDataScience/src/models/CORAL_BART/full-coral-unmasked \
        --task seq2seq\
        --cuda_device 1\
        --encoder_attention_heads 4\
        --decoder_attention_heads 4\
        --encoder_layers 6\
        --decoder_layers 6\
        --max_length 256\
        --per_device_train_batch_size 10\
        --per_device_eval_batch_size 20\
        --eval_fraction 0.1\
        --logging_steps 500\
        --eval_steps 5000\
        --save_steps 5000\
        --num_train_epochs 10\
        --coral
        # /projects/bdata/datasets/kaggle-competitions/processed/diffs_new.jsonl\
