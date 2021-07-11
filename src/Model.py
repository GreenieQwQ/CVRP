import torch.nn as nn
import torch
import numpy as np
from GCN import GCN
from SequentialDecoder import SequentialDecoder
from ClassificationDecoder import ClassificationDecoder

class Model(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers,
                 decode_type, k, device):
        super(Model, self).__init__()
        self.GCN = GCN(node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k, device)
        self.sequential = SequentialDecoder(node_hidden_dim, decode_type, device)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)

    def _seq_forward(self, x, strategy):
        # para
        batch_size = x.size(0)
        node_num = x.size(1)
        node_hidden_dim = x.size(2)

        last_node = torch.zeros((batch_size, 1))  # first is depot
        logprob, solution = None, torch.zeros((batch_size, 1))
        hidden = torch.zeros_like(x)  # zero init
        for i in range(node_num - 1):
            # TODO: refresh mask
            mask = x == np.inf
            # forward
            ind, probability, hidden = self.sequential(x, last_node, hidden, mask, strategy=strategy)
            # record
            last_node = ind
            solution = torch.cat([solution, ind], dim=1)
            if logprob is None:
                logprob = probability
            else:
                logprob = torch.cat([logprob, probability], dim=1)
        # end
        return logprob, solution

    def countDist(self, solution, dis):
        # TODO: finish count Distance
        return torch.zeros((solution.size(0), 1))

    def forward(self, env):
        node, demand, timewin, dis, timedis = env
        x, e = self.GCN(node, demand, timewin, dis, timedis)
        # sample
        sample_logprob, sample_solution = self._seq_forward(x, strategy='sample')
        sample_distance = self.countDist(sample_solution, dis)
        # greedy
        _, greedy_solution = self._seq_forward(x, strategy='greedy')
        greedy_distance = self.countDist(greedy_solution, dis)
        predict_matrix = self.classificationDecoder(e)

        return sample_logprob, sample_distance, greedy_distance, predict_matrix, greedy_solution