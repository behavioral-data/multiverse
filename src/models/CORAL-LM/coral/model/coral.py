# from .bert_graph import BERTGraph

import torch.nn as nn
from .bert import BERT
import pdb
import torch


class CORAL(nn.Module):
    """
    CORAL: Code representation learning
    """

    def __init__(self, bert: BERT, n_topics=5):
        super(CORAL, self).__init__()

        self.n_topics = n_topics

        # self.bert_graph = BERTGraph(bert)
        self.bert_graph = bert

        self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

        self.reconstruction = nn.Linear(n_topics, bert.hidden, bias=False)

        self.spv_stage_label = nn.Linear(n_topics, 6)

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat):
        vecs = self.bert_graph(x, segment_label, adj_mat)
        graph_vec = vecs[:, 0]

        topic_dist = self.dim_reduction(graph_vec)

        stage_vec = self.spv_stage_label(topic_dist)

        topic_dist = nn.Softmax(dim=1)(topic_dist)
        reconstructed_vec = self.reconstruction(topic_dist)

        neg_graph_vec = self.bert_graph(
            neg_x, neg_segment_label, neg_adj_mat)
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec
