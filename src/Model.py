import torch.nn as nn
import torch
import numpy as np
from GCN import GCN
from SequentialDecoder import SequentialDecoder
from ClassificationDecoder import ClassificationDecoder

class Model(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers,
                 decode_type, k, device):
        super(Model, self).__init__()
        self.device = device
        self.GCN = GCN(node_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k, device)
        self.sequential = SequentialDecoder(node_hidden_dim, decode_type, device)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)

    def _seq_forward(self, x, strategy):
        # para
        batch_size = x.size(0)
        node_num = x.size(1)
        node_hidden_dim = x.size(2)
        device = self.device

        last_node = torch.zeros((batch_size, 1)).long()  # first is depot
        logprob, solution = None, torch.zeros((batch_size, 1)).long().to(device)
        hidden = torch.zeros((2, batch_size, node_hidden_dim)).to(device)  # zero init
        for i in range(node_num - 1):
            # TODO: refresh mask
            mask = torch.zeros((batch_size, 1, node_num)).bool().to(device)
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
        # TODO: Test Validity
        device = self.device
        batch_size = solution.size(0)
        node_num = solution.size(1)
        result = torch.zeros((batch_size, 1)).to(device)
        for i in range(batch_size):
            for j in range(node_num-1):
                result[i] += dis[i][solution[i, j]][solution[i, j+1]]

        return result

    def forward(self, env):
        device = self.device
        node, demand, dis = env
        x, e = self.GCN(node, demand, dis)
        # sample
        sample_logprob, sample_solution = self._seq_forward(x, strategy='sample')
        sample_distance = self.countDist(sample_solution, dis)
        # greedy
        _, greedy_solution = self._seq_forward(x, strategy='greedy')
        greedy_distance = self.countDist(greedy_solution, dis)
        predict_matrix = self.classificationDecoder(e)
        # edge matrix
        batch_size = x.size(0)
        node_num = x.size(1)
        greedy_solution_matrix = torch.zeros((batch_size, node_num, node_num)).long().to(device)
        for i in range(batch_size):
            for j in range(node_num-1):
                greedy_solution_matrix[i][greedy_solution[i][j]][greedy_solution[i][j+1]] = 1

        return sample_logprob, sample_distance, greedy_distance, predict_matrix, greedy_solution_matrix