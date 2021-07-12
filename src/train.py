import numpy as np
import os
import torch
import torch.nn as nn
from Model import Model
from torch.optim import Adam
from tensorboardX import SummaryWriter
import time

if __name__ == '__main__':
    # init model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(2, node_hidden_dim=256, edge_hidden_dim=256, gcn_num_layers=4,
                  decode_type="what", device=device, k=5).to(device)
    # init optimizer
    lr = 0.005
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # read data
    dataDir = '../data'
    prefix = 'R-20'
    dataName = '{prefix}-training.npz'.format(prefix=prefix)
    # read 100 sample
    vehicle_capacities = 30
    dataset = np.load(os.path.join(dataDir, dataName))
    demand_set = (dataset['demand'] * vehicle_capacities).astype(int)
    dis_set = (dataset['dis']).astype(int)
    graph = dataset['graph']

    instaces_num = 100

    # recorder
    id = 'model_' + time.strftime("%D_%H_%M", time.localtime()).replace("/", "_") + "_lr{lr}".format(lr=lr)
    train_dir = '../model/' + id
    log_dir = train_dir + '/record'
    recorder = SummaryWriter(log_dir)
    for i in range(0, instaces_num):
        node = torch.FloatTensor(graph[i]).unsqueeze(0).to(device)
        demand = torch.FloatTensor(demand_set[i]).unsqueeze(0).to(device)
        dis = torch.FloatTensor(dis_set[i]).unsqueeze(0).to(device)
        env = (node, demand, dis)
        sample_logprob, sample_distance, greedy_distance, predict_matrix, solution_matrix = model(env)
        # loss
        predict_matrix = predict_matrix.view(-1, 2)
        solution_matrix = solution_matrix.view(-1).long()
        classification_loss = criterion(predict_matrix, solution_matrix)
        advantage = (sample_distance - greedy_distance).detach()
        reinforce = advantage * sample_logprob
        sequential_loss = reinforce.sum()
        loss = sequential_loss + classification_loss
        # train
        optimizer.zero_grad()
        loss.backward()
        # record
        iteration = i
        recorder.add_scalar('loss', loss, iteration)
