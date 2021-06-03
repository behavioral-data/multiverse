import argparse

from torch.utils.data import DataLoader

from .model import BERT
from .trainer import BERTTrainer
from .dataset import DataReader, UnitedVocab, CORALDataset, my_collate, key_lib
import pdb
import os
import json

import torch


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--dataset", required=True,
                        type=str, help="dataset")

    parser.add_argument("-o", "--output_path", required=True,
                        type=str, help="ex)output/bert.model")
    parser.add_argument("-t", "--test_path", required=True,
                        type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int,
                        default=256, help="hidden size of transformer model")
    parser.add_argument("-me", "--markdown_emb_size", type=int,
                        default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int,
                        default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int,
                        default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int,
                        default=64, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int,
                        default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int,
                        default=5, help="dataloader worker size")
    parser.add_argument("--duplicate", type=int,
                        default=5, help="dataloader worker size")
    parser.add_argument("--model_path",
                        type=str, help="ex)output/bert.model")
    parser.add_argument("--hinge_loss_start_point", type=int, default=20)
    parser.add_argument("--entropy_start_point", type=int, default=30)
    parser.add_argument("--with_cuda", type=bool, default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int,
                        default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=str,
                        default='0', help="CUDA device ids")
    parser.add_argument("--max_graph_num", type=int, default=3000000,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--vocab_path",
                        type=str, help="vocab path")

    parser.add_argument("--on_memory", type=bool, default=True,
                        help="Loading on memory: true or false")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, help="adam first beta value")
    parser.add_argument("--dropout", type=float,
                        default=0.2, help="dropout value")
    parser.add_argument("--weak_supervise", action="store_true")
    parser.add_argument("--neighbor", action="store_true",
                        help="force topic distribution over neighbor nodes to be close")
    parser.add_argument("--min_occur", type=int,
                        default=3, help="minimum of occurrence")

    parser.add_argument("--use_sub_token", action="store_true")
    parser.add_argument("--context", action="store_true",
                        help="use information from neighbor cells")
    parser.add_argument("--markdown", action="store_true", help="use markdown")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    output_folder = args.output_path
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print("Load Data", args.dataset)
    data_reader = DataReader(
        args.dataset, use_sub_token=args.use_sub_token, max_graph_num=args.max_graph_num, code_filter=key_lib)

    labeled_data_reader = DataReader(
        args.test_path, use_sub_token=args.use_sub_token, code_filter=key_lib)

    print("Loading Vocab")
    if args.markdown:
        vocab = UnitedVocab(data_reader.graphs, min_occur=args.min_occur,
                            use_sub_token=args.use_sub_token, path=args.vocab_path)

    else:
        vocab = SNAPVocab(data_reader.graphs, min_occur=args.min_occur,
                          use_sub_token=args.use_sub_token)

    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.dataset)
    train_dataset = CORALDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=args.seq_len,
                                 n_neg=args.duplicate, use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)
    print(len(train_dataset))

    print("Loading Dev Dataset", args.dataset)
    test_dataset = CORALDataset(data_reader.graphs[int(len(data_reader) * 0.8):], vocab, seq_len=args.seq_len,
                                n_neg=args.duplicate, use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)  # \
    print(len(test_dataset))

    labeled_dataset = CORALDataset(labeled_data_reader.graphs, vocab, seq_len=args.seq_len,
                                   use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)  # \

    labeled_data_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden,
                n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

    print("Creating BERT Trainer")

    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(
        args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, pad_index=vocab.pad_index)

    print("Training Start")

    with open(os.path.join(output_folder, './setting.json'), 'w') as fout:
        json.dump(args.__dict__, fout, ensure_ascii=False, indent=2)

    best_loss = None

    for epoch in range(args.epochs):
        train_loss = trainer.train(epoch)

        test_loss = trainer.test(epoch)
        results, topk = trainer.api(test_data_loader)

        trainer.save(epoch, os.path.join(output_folder, "model.ep%d" % epoch))
        with open(os.path.join(output_folder, './results.txt'), 'a') as fout:
            json.dump({"epoch": epoch,
                       "loss": test_loss}, fout)
            fout.write('\n')

        torch.cuda.empty_cache()


def load_model():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--dataset", required=True,
                        type=str, help="dataset")

    parser.add_argument("-o", "--output_path", required=True,
                        type=str, help="ex)output/bert.model")
    parser.add_argument("-t", "--test_path", required=True,
                        type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int,
                        default=256, help="hidden size of transformer model")
    parser.add_argument("-me", "--markdown_emb_size", type=int,
                        default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int,
                        default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int,
                        default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int,
                        default=64, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int,
                        default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int,
                        default=5, help="dataloader worker size")
    parser.add_argument("--duplicate", type=int,
                        default=5, help="dataloader worker size")
    parser.add_argument("--model_path",
                        type=str, help="ex)output/bert.model")
    parser.add_argument("--hinge_loss_start_point", type=int, default=20)
    parser.add_argument("--entropy_start_point", type=int, default=30)
    parser.add_argument("--with_cuda", type=bool, default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int,
                        default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=str,
                        default='0', help="CUDA device ids")
    parser.add_argument("--max_graph_num", type=int, default=3000000,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--vocab_path",
                        type=str, help="vocab path")

    parser.add_argument("--on_memory", type=bool, default=True,
                        help="Loading on memory: true or false")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, help="adam first beta value")
    parser.add_argument("--dropout", type=float,
                        default=0.2, help="dropout value")
    parser.add_argument("--weak_supervise", action="store_true")
    parser.add_argument("--neighbor", action="store_true",
                        help="force topic distribution over neighbor nodes to be close")
    parser.add_argument("--min_occur", type=int,
                        default=3, help="minimum of occurrence")

    parser.add_argument("--use_sub_token", action="store_true")
    parser.add_argument("--context", action="store_true",
                        help="use information from neighbor cells")
    parser.add_argument("--markdown", action="store_true", help="use markdown")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    output_folder = args.output_path
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print("Load Data", args.dataset)
    data_reader = DataReader(
        args.dataset, use_sub_token=args.use_sub_token, max_graph_num=args.max_graph_num, code_filter=key_lib)

    labeled_data_reader = DataReader(
        args.test_path, use_sub_token=args.use_sub_token, code_filter=key_lib)

    print("Loading Vocab")
    if args.markdown:
        vocab = UnitedVocab(data_reader.graphs, min_occur=args.min_occur,
                            use_sub_token=args.use_sub_token, path=args.vocab_path)

    else:
        vocab = SNAPVocab(data_reader.graphs, min_occur=args.min_occur,
                          use_sub_token=args.use_sub_token)

    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.dataset)
    train_dataset = CORALDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=args.seq_len,
                                 n_neg=args.duplicate, use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)

    print(len(train_dataset))

    print("Loading Dev Dataset", args.dataset)
    test_dataset = CORALDataset(data_reader.graphs[int(len(data_reader) * 0.8):], vocab, seq_len=args.seq_len,
                                n_neg=args.duplicate, use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)  # \
    print(len(test_dataset))

    labeled_dataset = CORALDataset(labeled_data_reader.graphs, vocab, seq_len=args.seq_len,
                                   use_sub_token=args.use_sub_token, markdown=args.markdown, masked=True)

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)  # \

    labeled_data_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden,
                n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

    print("Creating BERT Trainer")

    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(
        args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, pad_index=vocab.pad_index, model_path=args.model_path)
    print("Loading Start")

    # results, topk = trainer.api(test_data_loader)
    results, topk = trainer.api(labeled_data_loader)
    mask_results = []
    ii = 0
    # for (d, _), r, tk in zip(test_dataset, results, topk):
    for (d, _), r, tk in zip(labeled_dataset, results, topk):

        # g = test_dataset.graphs[ii]
        g = labeled_dataset.graphs[ii]

        ii += 1
        bert_input = d["bert_input"]
        target_label = d["target_label"]
        flag = False
        for i, t in enumerate(bert_input):

            if t == 2:
                if not flag:
                    print('=' * 20)
                    print(g["context"])
                    flag = True
                mask_results.append((target_label[i], r[i]))

                print(vocab.idx2word[target_label[i]], [
                      vocab.idx2word[tkk] for tkk in tk[i]])

    counter = 0
    for t, r in mask_results:
        if t == r:
            counter += 1

    tokens = []
    for t, r in mask_results:

        if vocab.idx2word[r] not in tokens:
            tokens.append(vocab.idx2word[r])

    pdb.set_trace()


load_model()
# train()
