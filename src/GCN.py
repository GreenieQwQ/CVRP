import torch
import torch.nn as nn
from GCN_Layer import GCNLayer


class GCN(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k, device):
        super(GCN, self).__init__()
        self.k = k
        self.node_input_dim = node_input_dim
        # self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.gcn_num_layers = gcn_num_layers
        self.device = device

        self.node_W1 = nn.Linear(self.node_input_dim, self.node_hidden_dim)
        self.node_W2 = nn.Linear(self.node_input_dim, self.node_hidden_dim // 2)
        self.node_W3 = nn.Linear(1, self.node_hidden_dim // 2)
        self.edge_W4 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.edge_W5 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.nodes_embedding = nn.Linear(self.node_hidden_dim, self.node_hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim, bias=False)
        self.gcn_layers = nn.ModuleList([GCNLayer(self.node_hidden_dim) for i in range(self.gcn_num_layers)])
        self.Relu = nn.ReLU()

    # def forward(self, node, demand, timewin, dis, timedis):
    def forward(self, node, demand, dis):
        device = self.device
        batch_size = node.size(0)
        node_num = node.size(1)
        # node = torch.cat([node, timewin], dim=2)
        # edge = torch.cat([dis.unsqueeze(3), timedis.unsqueeze(3)], dim=3)
        edge = dis.unsqueeze(3)
        self_edge = (torch.arange(0, node_num).unsqueeze(0)).T.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        order = dis.sort(2)[1]
        neighbor_index = order[:, :, 1:self.k+1]
        a = torch.zeros_like(dis)
        a = torch.scatter(a, 2, neighbor_index, 1)
        a = torch.scatter(a, 2, self_edge, -1).to(device)

        # 仓库默认下标0
        depot = node[:, 0, :]
        demand = demand[:, 1:].unsqueeze(2)
        customer = node[:, 1:, ]

        # Node and edge embedding
        depot_embedding = self.Relu(self.node_W1(depot))
        customer_embedding = self.Relu(torch.cat([self.node_W2(customer), self.node_W3(demand)], dim=2))
        x = torch.cat([depot_embedding.unsqueeze(1), customer_embedding], dim=1)
        e = self.Relu(torch.cat([self.edge_W4(edge), self.edge_W5(a.unsqueeze(3))], dim=3))
        x = self.nodes_embedding(x)
        e = self.edges_embedding(e)

        for layer in self.gcn_layers:
            # x: BatchSize * V * Hid
            # e: B * V * V * H
            x, e = layer(x, e, neighbor_index)
        return x, e
