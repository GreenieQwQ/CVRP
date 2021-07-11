import torch.nn as nn
import torch

class ClassificationDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.sm = nn.Softmax(-1)

    def forward(self, e):
        a = self.MLP(e)
        a = a.squeeze(-1)
        out = self.sm(a)
        return out
