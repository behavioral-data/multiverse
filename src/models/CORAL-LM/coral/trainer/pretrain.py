import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from coral.model import BERT, BERT_MLM
# from .optim_schedule import ScheduledOptim

import tqdm
import pdb

import os


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, pad_index=0, model_path=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERT_MLM(bert, vocab_size).to(self.device)
        if model_path:
    
            state_dict = torch.load(model_path,self.device)["model_state_dict"]
            # pdb.set_trace()
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(state_dict)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.pad_index = pad_index
        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr,
        #                   betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(
        #     self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        self.optim = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=self.pad_index)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement()
                                        for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.test_data, train=False)

    def api(self, data_loader=None):
        self.model.eval()

        # str_code = "train" if train else "test"
        if not data_loader:
            data_loader = self.test_data

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              # desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        # for (i, data), (ni, ndata) in data_iter, neg_data_iter:
        phases = []
        stages = []
        stage_vecs = []
        results = []
        topk = []
        correct = 0
        elements = 0
        with torch.no_grad():
            for i, data in data_iter:

                data = data[0]
                data = {key: value.to(self.device)
                        for key, value in data.items()}

                mask_lm_output = self.model.forward(
                    data["bert_input"], data["segment_label"], data["adj_mat"], train=False)
                # pdb.set_trace()
                topk += torch.topk(mask_lm_output, k=5)[1].tolist()
                results += mask_lm_output.argmax(dim=-1).tolist()
                for labels, t_labels in zip(mask_lm_output.argmax(dim=-1), data["target_label"]):
                    correct += sum([1 if l == t and t !=
                                    self.pad_index else 0 for l, t in zip(labels, t_labels)])
                    elements += sum([1 for t in t_labels if t !=
                                     self.pad_index])

        # pdb.set_trace()
        # pdb.set_trace()
        return results, topk

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        # pdb.set_trace()
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        def calculate_iter(data):
            mask_lm_output = self.model.forward(
                data["bert_input"], data["segment_label"], data["adj_mat"], train)
            # next_sent_output, mask_lm_output = self.model.forward(
            #     data["bert_input"], data["segment_label"], data["adj_mat"], train)
            mask_loss = self.criterion(
                mask_lm_output.transpose(1, 2), data["target_label"])
            # pdb.set_trace()
            loss = mask_loss
            return loss, mask_lm_output

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # pdb.set_trace()
            data = data[0]
            data = {key: value.to(self.device) for key, value in data.items()}

            if train:
                loss, mask_lm_output = calculate_iter(data)
            else:
                with torch.no_grad():
                    loss, mask_lm_output = calculate_iter(data)
            if train:
                self.optim.zero_grad()
                loss.backward()
                # self.optim.step_and_update_lr()
                self.optim.step()

            correct = 0
            elements = 0
            for labels, t_labels in zip(mask_lm_output.argmax(dim=-1), data["target_label"]):
                correct += sum([1 if l == t and t !=
                                self.pad_index else 0 for l, t in zip(labels, t_labels)])
                elements += sum([1 for t in t_labels if t != self.pad_index])
            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(
            #     dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            # total_element += data["is_next"].nelement()
            total_element += elements

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0 and i != 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)
        return avg_loss / len(data_iter)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path
        # output_path = file_path + ".ep%d" % epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict()
        }, output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)

        return output_path
