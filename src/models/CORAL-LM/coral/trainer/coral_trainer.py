# coding=utf-8
# created by Ge Zhang, Jan 20, 2020
#
# CORAL trainer


import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from coral.model import BERT, CORAL


import tqdm
import pdb
torch.manual_seed(0)


def my_loss(reconstructed_pos, origin_pos, origin_neg):
    duplicate = int(origin_neg.shape[0] / reconstructed_pos.shape[0])

    hid_size = origin_neg.shape[-1]

    pos_sim = torch.bmm(reconstructed_pos.unsqueeze(
        1), origin_pos.unsqueeze(2)).repeat(1, duplicate, 1).view(-1)
    neg_sim = torch.bmm(reconstructed_pos.repeat(
        1, duplicate).view(-1, hid_size).unsqueeze(1), origin_neg.unsqueeze(2)).view(-1)
    diff = neg_sim - pos_sim + 1

    diff = torch.max(diff, torch.zeros_like(diff))
    loss = torch.sum(diff)
    return loss


class CORALTrainer:
    """
    CORALTrainer
    """

    def __init__(self, bert: BERT,
                 train_dataloader: DataLoader, test_dataloader: DataLoader,
                 lr: float = 1e-4,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, loss_lambda=1, model_path=None, n_topics=50, hinge_loss_start_point=20, entropy_start_point=30):
        """
        :param bert: code representation encoder
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for CORAL training, argument -c, --cuda should be true
        self.loss_lambda = loss_lambda
        self.n_topics = n_topics

        self.hinge_loss_start_point = hinge_loss_start_point
        self.entropy_start_point = entropy_start_point
        cuda_condition = torch.cuda.is_available() and with_cuda

        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert = bert
        self.model = CORAL(bert, n_topics=n_topics).to(self.device)

        print(model_path)
        if model_path:
            state_dict = torch.load(model_path)["model_state_dict"]

            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(state_dict)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the SGD optimizer with hyper-param
        self.optim = SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Using Cross Entropy Loss function for weak supervision
        self.best_loss = None
        self.updated = False
        self.log_freq = log_freq
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

        print("Total Parameters:", sum([p.nelement()
                                        for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def api(self, data_loader=None):
        self.model.eval()

        if not data_loader:
            data_loader = self.test_data

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        stages = []
        stage_vecs = []
        with torch.no_grad():
            for i, item in data_iter:
                data = item[0]
                ndata = item[1]
                data = {key: value.to(self.device)
                        for key, value in data.items()}
                ndata = {key: value.to(self.device)
                         for key, value in ndata.items()}

                data = {key: value.to(self.device)
                        for key, value in data.items()}
                ndata = {key: value.to(self.device)
                         for key, value in ndata.items()}

                reconstructed_vec, graph_vec, origin_neg, topic_dist, stage_vec = self.model.forward(
                    data["bert_input"], ndata["bert_input"], data["segment_label"], ndata["segment_label"], data["adj_mat"], ndata["adj_mat"])

                stages += torch.max(stage_vec, 1)[-1].tolist()
                stage_vecs += stage_vec.tolist()

        return stages, stage_vecs

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: average loss
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        for i, item in data_iter:

            data = item[0]
            ndata = item[1]

            data = {key: value.to(self.device) for key, value in data.items()}
            ndata = {key: value.to(self.device)
                     for key, value in ndata.items()}

            reconstructed_vec, graph_vec, origin_neg, topic_dist, stage_vec = self.model.forward(
                data["bert_input"], ndata["bert_input"], data["segment_label"], ndata["segment_label"], data["adj_mat"], ndata["adj_mat"])

            bs, _ = reconstructed_vec.shape
            nbs, _ = origin_neg.shape
            duplicate = int(nbs / bs)

            hinge_loss = my_loss(reconstructed_vec, graph_vec, origin_neg)
            weight_loss = torch.norm(torch.mm(
                self.model.reconstruction.weight.T, self.model.reconstruction.weight) - torch.eye(self.n_topics).cuda())
            c_entropy = self.cross_entropy(stage_vec, data['stage'])
            entropy = -1 * (F.softmax(stage_vec, dim=1) *
                            F.log_softmax(stage_vec, dim=1)).sum()

            if epoch < self.hinge_loss_start_point:
                loss = c_entropy

            elif epoch < self.entropy_start_point:
                loss = c_entropy + self.loss_lambda * weight_loss + hinge_loss
            else:
                loss = c_entropy + entropy + self.loss_lambda * weight_loss + hinge_loss

            if epoch == self.hinge_loss_start_point:
                self.optim = SGD(self.model.parameters(),
                                 lr=0.00001, momentum=0.9)

            if train:
                self.optim.zero_grad()
                loss.backward()

                self.optim.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
                "cross_entropy": c_entropy.item(),
                "entropy": entropy.item(),
                "hinge_loss": hinge_loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" %
              (epoch, str_code), avg_loss / len(data_iter))
        return avg_loss / len(data_iter)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict()
        }, output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)

        return output_path
