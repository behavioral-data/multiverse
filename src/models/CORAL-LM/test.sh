python3 coral/test.py \
	--dataset="./examples/train.txt" \
	--output_path='output.model' \
	--model_path='output/output.model.ep14' \
	--test_path='./examples/test.txt' \
	--vocab_path='./output/vocab.txt' \
	--cuda_devices='0' \
	--log_freq=10000 \
	--layers=4 \
	--attn_heads=4 \
	--lr=0.00003 \
	--batch_size=16 \
	--num_workers=1 \
	--duplicate=5 \
	--dropout=0 \
	--min_occur=5 \
	--use_sub_token \
	--seq_len=160 \
	--max_graph_num=1000000 \
	--markdown \
	--hinge_loss_start_point=1 \
	--entropy_start_point=6


