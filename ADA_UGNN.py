# %%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
import os 
import random
import time 
import configparser

# %%
import networkx as nx 
from torch_geometric.datasets import Planetoid

# %%
from sklearn.metrics import confusion_matrix


#%%
conf = configparser.ConfigParser()

conf.read("config.ini", encoding = "utf-8")

# %%
# freeze
Seed = int(conf["hyperparameter"]["Seed"])
torch.manual_seed(Seed) # set seed for CPU
torch.cuda.manual_seed(Seed) # set seed for current GPU
torch.cuda.manual_seed_all(Seed) # set seed for all GPUs
np.random.seed(Seed)  # Numpy module.
random.seed(Seed)  # Python random module.	
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# %%
graphs = Planetoid("datasets", name = conf["datasets"]["dataset"])
graph = graphs[0]
graph

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
a = torch.tensor(graph.x, device = device)
y = torch.tensor(graph.y, device = device)

S = F.one_hot(y)

# %%
num_nodes = graph.x.shape[0]

input_features = graph.x.shape[1]
output_features = graphs.num_classes

num_train_nodes = (graph.train_mask | graph.val_mask).sum().item()
num_test_nodes = graph.test_mask.sum().item()

num_nodes, input_features, output_features, num_train_nodes, num_test_nodes

# %%
# graph information
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(graph.edge_index.T.numpy())
G.add_edges_from([(x, x) for x in range(num_nodes)]) # add self loop
G.number_of_nodes()

A = np.array(nx.adjacency_matrix(G).todense())
D = np.diag(A.sum(axis = 1))

A = torch.tensor(A).to(device)
D = torch.tensor(D).to(device)

zero_vector = torch.zeros(num_nodes, num_nodes).to(device)
zeros_vector = torch.zeros(num_nodes, num_nodes, output_features).to(device)

A_hat = A.unsqueeze(2).repeat(1, 1, output_features)

D_sqrt = D.sqrt()
normalized_D = (1.0 / D_sqrt).diag().reshape(num_nodes, 1).to(device)

# %%
W_1 = nn.Parameter(torch.rand(input_features, 1000, device = device))
b_1 = nn.Parameter(torch.rand(1, 1000, device = device))

W_2 = nn.Parameter(torch.rand(1000, 20, device = device))
b_2 = nn.Parameter(torch.rand(1, 20, device = device))

W_3 = nn.Parameter(torch.rand(20, output_features, device = device))
b_3 = nn.Parameter(torch.rand(1, output_features, device = device))

W_c_1 = nn.Parameter(torch.rand(output_features, 20, device = device))
b_c_1 = nn.Parameter(torch.rand(1, 20, device = device))

W_c_2 = nn.Parameter(torch.rand(20, 1, device = device))
b_c_2 = nn.Parameter(torch.rand(1, 1, device = device))

#%%
def init():
    nn.init.xavier_uniform_(W_1.data)
    nn.init.xavier_uniform_(b_1.data)

    nn.init.xavier_uniform_(W_2.data)
    nn.init.xavier_uniform_(b_2.data)

    nn.init.xavier_uniform_(W_3.data)
    nn.init.xavier_uniform_(b_3.data)

    nn.init.xavier_uniform_(W_c_1.data)
    nn.init.xavier_uniform_(b_c_1.data)

    nn.init.xavier_uniform_(W_c_2.data)
    nn.init.xavier_uniform_(b_c_2.data)

# %%
init()
W_1.is_leaf, W_2.is_leaf, W_3.is_leaf, b_1.is_leaf, b_2.is_leaf, b_3.is_leaf


# %%
def train():
    h = torch.mm(a, W_1).add(b_1)
    h = F.relu(h, 0.2)
    h = F.dropout(h, p = 0.5)

    h = torch.mm(h, W_2).add(b_2)
    h = F.relu(h, 0.2)
    h = F.dropout(h, p = 0.5)

    h = torch.mm(h, W_3).add(b_3)

    h = F.softmax(h, dim = 1)
    
    first_term = torch.pow(h - S, 2).sum(dim = 1) # first_term: [num_nodes]
    
    h_normalized = normalized_D * h

    diff = (torch.pow(h_normalized.repeat(1, num_nodes).view(-1, output_features) - h_normalized.repeat(num_nodes, 1), 2)).view(num_nodes, num_nodes, output_features).sum(dim = 2)

    # # consider the local smoothness C_i
    h_prime = h.repeat(num_nodes, 1).view(num_nodes, num_nodes, output_features)

    C = torch.where(A_hat > 0, h_prime, zeros_vector)
    C = C.sum(dim = 1) * (1.0 / D.diag()).view(num_nodes, -1)
    C = s * torch.sigmoid(torch.mm(F.relu(torch.mm(C, W_c_1).add(b_c_1)), W_c_2).add(b_c_2)) # shape: [num_nodes, 3] -> [num_nodes, 1]

    diff = torch.where(A > 0, diff, zero_vector) # shape: [num_nodes, num_nodes]

    diff = diff * C # broadcast
    
    second_term = 0.5 * diff.sum(dim = 1) # second_term: [num_nodes]
    
    total_loss = (first_term[graph.val_mask | graph.train_mask].sum() + second_term[graph.val_mask | graph.train_mask].sum()) / num_train_nodes

    # total_loss = (first_term[graph.val_mask | graph.train_mask].sum()) / num_train_nodes
    
    # print(first_term[graph.val_mask | graph.train_mask].sum().item())

    # print(second_term[graph.val_mask | graph.train_mask].sum().item())
    
    # print("total_loss: ", total_loss.item())

    total_loss.backward()

    # Update Parameters
    W_1.data = W_1.data - learning_rate * W_1.grad.data
    W_2.data = W_2.data - learning_rate * W_2.grad.data
    W_3.data = W_3.data - learning_rate * W_3.grad.data
    W_c_1.data = W_c_1.data - learning_rate * W_c_1.grad.data
    W_c_2.data = W_c_2.data - learning_rate * W_c_2.grad.data

    b_1.data = b_1.data - learning_rate * b_1.grad.data
    b_2.data = b_2.data - learning_rate * b_2.grad.data
    b_3.data = b_3.data - learning_rate * b_3.grad.data
    b_c_1.data = b_c_1.data - learning_rate * 0.5 * b_c_1.grad.data
    b_c_2.data = b_c_2.data - learning_rate * 0.5 * b_c_2.grad.data

    W_1.grad.data.zero_()
    W_2.grad.data.zero_()
    W_3.grad.data.zero_()
    W_c_1.grad.data.zero_()
    W_c_2.grad.data.zero_()

    b_1.grad.data.zero_()
    b_2.grad.data.zero_()
    b_3.grad.data.zero_()
    b_c_1.grad.data.zero_()
    b_c_2.grad.data.zero_()


# %%
def test():

    with torch.no_grad():
        h = torch.mm(a, W_1).add(b_1)
        h = F.relu(h)
        
        h = torch.mm(h, W_2).add(b_2)
        h = F.relu(h)
        
        h = torch.mm(h, W_3).add(b_3)

        # h = F.softmax(h, dim = 1)

        values, indices = h.max(dim = 1)

        C = confusion_matrix(graph.y, indices.cpu().numpy())
        print(C)

        acc = (y == indices).sum() / num_nodes
        print("epoch: ", i, " acc rate: ", acc)

# %%
epochs = int(conf["hyperparameter"]["epochs"])
logging_steps = int(conf["hyperparameter"]["logging_steps"])

learning_rate = float(conf["hyperparameter"]["learning_rate"])
s = float(conf["hyperparameter"]["s"]) # Hyperparameter S: Upper Boader in Eq.(30)

# %%
for i in range(epochs):
    train()

    if (i + 1) % logging_steps == 0:
        test()

    if (i + 1) % 1000 == 0:
        learning_rate = learning_rate / 2

# %%
import pickle

model = [W_1, b_1, W_2, b_2, W_3, b_3]
with open("models/model" + "_" + conf["datasets"]["dataset"] + ".pkl", mode = "wb") as f:
    pickle.dump(model, f)
    print("model has been saved to models/model" + "_" + conf["datasets"]["dataset"] + ".pkl")

#%%
