import torch
import torch.nn as nn
import numpy as np

class SequentialDecoder(nn.Module):
    def __init__(self, hidden_dim, decode_type, device):
        super(SequentialDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decode_type = decode_type
        self.device = device
        self.sm = nn.Softmax(1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2)
        self.tahn = nn.Tanh()
        self.h = nn.Linear(hidden_dim, 1)
        self.W = nn.Linear(2, 1)
        self.pointer = nn.MultiheadAttention(hidden_dim, num_heads=1)

    def forward(self, x, last_node, hidden, mask, strategy):
        # x: 用于获取lastx的emb
        # 仅forward lastnode
        # 之前的隐状态使用hidden存储
        device = self.device
        batch_size = x.size(0)
        batch_idx = torch.arange(0, batch_size).unsqueeze(1).to(device)
        last_x = x[batch_idx, last_node]
        last_x = last_x.permute(1, 0, 2)
        _, hidden = self.gru(last_x, hidden)
        z = hidden[-1]

        _, u = self.pointer(z, x.permute(1,0,2))
        u = u.masked_fill_(~mask, -np.inf)
        probs = self.sm(u)
        if strategy == "sample":
            ind = torch.multinomial(probs, num_samples=1)
        else:
            ind = torch.max(probs, dim=1)[1].unsqueeze(1)
        probability = probs[batch_idx, ind].squeeze(1)
        # ind->下一个index
        # prob 这个步骤选取index的prob
        # hidden 用于传递给下一级的隐状态
        return ind, probability, hidden
