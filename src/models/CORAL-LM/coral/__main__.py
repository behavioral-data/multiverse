# coding=utf-8
# created by Ge Zhang, Jan 29, 2020
#
# train CORAL

import argparse

from torch.utils.data import DataLoader

from model import BERT
from trainer import CORALTrainer
from dataset import DataReader, my_collate, UnitedVocab, CORALDataset

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
                        default=5, help="number of negative examples")
    parser.add_argument("--model_path",
                        type=str, help="ex)output/bert.model")
    parser.add_argument("--hinge_loss_start_point", type=int, default=20)
    parser.add_argument("--entropy_start_point", type=int, default=30)
    parser.add_argument("--with_cuda", type=bool, default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="printing loss every n iter: setting n")

    parser.add_argument("--cuda_devices", type=str,
                        default='0', help="CUDA device ids")
    parser.add_argument("--max_graph_num", type=int, default=3000000,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--vocab_path",
                        type=str, help="vocab path")

    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of adam")
    parser.add_argument("--dropout", type=float,
                        default=0.2, help="dropout value")
    parser.add_argument("--min_occur", type=int,
                        default=3, help="minimum of occurrence")
    parser.add_argument("--use_sub_token", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="use markdown")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    output_folder = args.output_path.split('.')[0]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print("Read Train Data from", args.dataset)
    data_reader = DataReader(
        args.dataset, max_graph_num=args.max_graph_num)
    print("Read Test Data from", args.test_path)
    labeled_data_reader = DataReader(
        args.test_path)

    print("Loading Vocab")
    vocab = UnitedVocab(data_reader.graphs, min_occur=args.min_occur,
                        use_sub_token=args.use_sub_token, path=args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.dataset)
    train_dataset = CORALDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=args.seq_len,
                                 n_neg=args.duplicate, markdown=args.markdown)
    print(len(train_dataset))
    print("Loading Dev Dataset", args.dataset)
    test_dataset = CORALDataset(data_reader.graphs[int(len(data_reader) * 0.8):], vocab, seq_len=args.seq_len,
                                n_neg=args.duplicate, markdown=args.markdown)  # \
    print(len(test_dataset))
    print("Loading Test Dataset", args.dataset)
    labeled_dataset = CORALDataset(labeled_data_reader.graphs, vocab, seq_len=args.seq_len,
                                   markdown=args.markdown)
    print(len(labeled_dataset))

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)  # \

    labeled_data_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)

    print("Building CORAL model")
    bert = BERT(len(vocab), hidden=args.hidden,
                n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)

    print("Creating CORAL Trainer")

    trainer = CORALTrainer(bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                           lr=args.lr,
                           with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, model_path=args.model_path, n_topics=args.n_topics, hinge_loss_start_point=args.hinge_loss_start_point, entropy_start_point=args.entropy_start_point)

    print("Parameters are in {}".format(
        os.path.join(output_folder, './setting.json')))
    with open(os.path.join(output_folder, './setting.json'), 'w') as fout:
        json.dump(args.__dict__, fout, ensure_ascii=False, indent=2)

    print("Training Start")
    best_loss = None
    tolerance = 3
    for epoch in range(args.epochs):
        train_loss = trainer.train(epoch)

        test_loss = trainer.test(epoch)

        if epoch >= args.entropy_start_point:
            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                tolerance = 3

            else:
                tolerance -= 1

        trainer.save(epoch, os.path.join(
            output_folder, args.output_path))

        with open(os.path.join(output_folder, './results.txt'), 'a') as fout:
            json.dump({"epoch": epoch,
                       "loss": test_loss}, fout)
            fout.write('\n')

        torch.cuda.empty_cache()
        if tolerance == 0:
            break
    print('Best loss:', best_loss)


train()
