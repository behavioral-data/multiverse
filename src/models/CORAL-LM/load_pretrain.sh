python3 coral/pretrain.py \
	--dataset="/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/cell_with_func_python23_1_27.txt" \
	--output_path='/projects/bdata/jupyter/gezhang_backup/BERT-pytorch/pretrain' \
	--test_path='/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/test.txt' \
	--model_path='/projects/bdata/jupyter/gezhang_backup/BERT-pytorch/pretrain/model.ep9' \
	--vocab_path='/projects/bdata/jupyter/gezhang_backup/BERT-pytorch/pretrain/vocab.txt' \
	--cuda_devices='0' \
	--log_freq=10000 \
	--epochs=50 \
	--layers=4 \
	--attn_heads=4 \
	--lr=0.00003 \
	--batch_size=16 \
	--num_workers=1 \
	--duplicate=1 \
	--dropout=0 \
	--min_occur=1 \
	--weak_supervise \
	--seq_len=160 \
	--max_graph_num=1000000 \
	--markdown \
	--hinge_loss_start_point=50 \
	--entropy_start_point=50


