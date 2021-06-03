import argparse

from torch.utils.data import DataLoader

from .model import BERT
from .trainer import BERTTrainer
from .dataset import DataReader, UnitedVocab, CORALDataset, my_collate, key_lib
import pdb
import os
import json

import torch

class Session():
    def __init__(self, dataset, 
                        model_path, 
                        vocab_path,
                        cuda_devices = "1",
                        duplicate = 1,
                        log_freq =10000,
                        batch_size = 16,
                        markdown = True,
                        max_graph_num = 1000000,
                        seq_len= 160,
                        num_workers = 1, 
                        min_occur = 1, 
                        weak_supervise = True,
                        use_sub_token=False, 
                        adam_beta1 = 0.9, 
                        adam_beta2 = 0.99, 
                        adam_weight_decay = 0.1, 
                        lr = 0.0003,
                        hidden = 256,
                        layers = 4,
                        attn_heads = 4,
                        with_cuda = True,
                        dropout = 0.2):

    
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        
        self.dataset = dataset
        self.vocab_path = vocab_path
        self.model_path = model_path
        self.cuda_devices = cuda_devices
        self.log_freq = log_freq
        self.max_graph_num = max_graph_num
        self.seq_len = seq_len
        self.min_occur = min_occur
        self.weak_supervise = weak_supervise
        self.duplicate = duplicate
        self.adam_weight_decay = adam_weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.use_sub_token = use_sub_token
        self.markdown = markdown
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_cuda = with_cuda
        self.hidden= hidden
        self.attn_heads = attn_heads
        self.layers = layers
        self.dropout = dropout
        self.lr = lr


        print("Load Data", self.dataset)
        data_reader = DataReader(
            self.dataset, use_sub_token=self.use_sub_token, max_graph_num=self.max_graph_num, code_filter=key_lib)


        print("Loading Vocab")
        if self.markdown:
            self.vocab = UnitedVocab(data_reader.graphs, min_occur=self.min_occur,
                                use_sub_token=self.use_sub_token, path=self.vocab_path)

        else:
            self.vocab = SNAPVocab(data_reader.graphs, min_occur=self.min_occur,
                            use_sub_token=self.use_sub_token)

        print("Vocab Size: ", len(self.vocab))

        print("Loading Train Dataset", self.dataset)
        self.train_dataset = CORALDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], self.vocab, seq_len=self.seq_len,
                                    n_neg=self.duplicate, use_sub_token=self.use_sub_token, markdown=self.markdown, masked=True)

        print(len(self.train_dataset))

        print("Loading Dev Dataset", self.dataset)
        self.test_dataset = CORALDataset(data_reader.graphs[int(len(data_reader) * 0.8):], self.vocab, seq_len=self.seq_len,
                                    n_neg=self.duplicate, use_sub_token=self.use_sub_token, markdown=self.markdown, masked=True)  # \
        print(len(self.test_dataset))

    

        print("Creating Dataloaders")
        self.train_data_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=my_collate)

        self.test_data_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=my_collate)  # \


        print("Building BERT model")
        self.bert = BERT(len(self.vocab), hidden=self.hidden,
                    n_layers=self.layers, attn_heads=self.attn_heads, dropout=self.dropout)
        print("Creating BERT Trainer")

        self.trainer = BERTTrainer(self.bert, len(self.vocab), train_dataloader=self.train_data_loader, test_dataloader=self.test_data_loader,
                          lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), 
                          weight_decay=self.adam_weight_decay,
                          with_cuda=self.with_cuda, cuda_devices=self.cuda_devices, 
                          log_freq=self.log_freq, pad_index=self.vocab.pad_index, model_path=self.model_path)
        print("Trainer Complete")

def main():
    pass

if __name__ == '__main__':
    main()