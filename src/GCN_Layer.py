import torch.nn as nn
import torch

class GCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        """
        hidden_dim:ivar embedding dim of the node
        """
        super(GCNLayer, self).__init__()
        # node GCN layers
        self.W_node = nn.Linear(hidden_dim, hidden_dim)
        self.V_node_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_node = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.Relu = nn.ReLU()
        self.ln1_node = nn.LayerNorm(hidden_dim)
        self.ln2_node = nn.LayerNorm(hidden_dim)

        # edge
        self.W_edge = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W1_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W3_edge = nn.Linear(hidden_dim, hidden_dim)
        self.ln1_edge = nn.LayerNorm(hidden_dim)
        self.ln2_edge = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x, e, neighbor_index):
        # node embedding
        batch_size = x.size(0)
        node_num = x.size(1)
        node_hidden_dim = x.size(2)
        t = x.unsqueeze(1).repeat(1, node_num, 1, 1)

        neighbor_index = neighbor_index.unsqueeze(3).repeat(1, 1, 1, node_hidden_dim)
        # torch.gather(t, dim=2, index=neighbor_index)
        neighbor = t.gather(2, neighbor_index)
        neighbor = neighbor.view(batch_size, node_num, -1, node_hidden_dim)

        h_nb_node = self.ln1_node(x + self.Relu(self.W_node(self.attn(x, neighbor, neighbor))))
        h_node = self.ln2_node(h_nb_node + self.Relu(self.V_node(torch.cat([self.V_node_in(x), h_nb_node], dim=-1))))

        # edge
        x_from = x.unsqueeze(2).repeat(1,1,node_num,1)
        x_to = x.unsqueeze(1).repeat(1,node_num,1,1)
        # TODO: 换用attn会不会更好？
        h_nb_edge = self.ln1_edge(e + self.Relu(self.W_edge(self.W1_edge(e) + self.W2_edge(x_from) + self.W3_edge(x_to))))
        h_edge = self.ln2_edge(h_nb_edge + self.Relu(self.V_edge(torch.cat([self.V_edge_in(x), h_nb_edge], dim=-1))))

        return h_node, h_edge
