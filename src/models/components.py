import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Helper class to reuse saving/loading model code
    """
    def __init__(self) -> None:
        super().__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        if map_location is not None:
            if map_location == 'cpu':
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.load_state_dict(torch.load(path))


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, predict=False) -> None:
        """
        Wrapper for PyTorch implementation of LSTM to take advantage of save/load of BaseModel
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.predict = predict
        if predict:
            self.fc = nn.Linear(10*hidden_size, 14) # hard coded to sequences of length 10

    def initialize_hidden_state(self, batch_size):
        self.batch_size = batch_size
        self.hidden_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden_state=None):
        if hidden_state is not None:
            self.hidden_state = hidden_state
        output, hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        if self.predict:
            return self.fc( output.view(self.batch_size, -1) ), self.hidden_state
        else:
            return output[:, -1, :], self.hidden_state # just output of last in sequence


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation, adj=None) -> None:
        """
        Single layer of graph convolution that takes in node features and adjacency matrix and outputs some value

        Args:
            in_dim: int, number of features for node vector
            out_dim: int, number of values to predict for this node
            activation: str, activation function to use
            adj: np.ndarray, optional adjacency matrix if fixed for every iteration
        """
        super().__init__()
        if adj is not None:
            # constant adjacency convolutions
            if adj[0,0] == 0:
                self.a = torch.tensor(adj + np.eye(adj.shape[0]), requires_grad=False)
            else:
                self.a = torch.tensor(adj, requires_grad=False)
            
            self.d_inv = np.linalg.inv( torch.sum(self.a, axis=1) )
            self.sqrt_d_inv = sqrtm(self.d_inv)

        # kinda shitty initializations but works for now
        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.float))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, a=None):
        if a is None:
            a = self.a
            sqrt_d_inv = self.sqrt_d_inv
        else:
            if a[0,0] == 0:
                a += torch.eye(a.shape[0])
            
            a.type(self.weight.dtype)
            
            # traditionally have D^(-1/2)AD^(-1/2) but correlation matrix has negative values so D^(-1/2) becomes ill-defined
            # d_inv = np.linalg.inv( torch.diag(torch.sum(a, axis=1)) )
            # sqrt_d_inv = torch.tensor(sqrtm(d_inv), dtype=self.weight.dtype)
            # sqrt_d_inv = torch.diag( torch.sqrt(1 / torch.sum(a, axis=1)) ) # simple since inverse of diagonal matrix

        x = torch.matmul(x, self.weight)
        # adj_inter = torch.matmul(torch.matmul(sqrt_d_inv, a), sqrt_d_inv) # intermediate step in adjacency matrix calculations
        # output = torch.matmul(torch.matmul(), x)
        output = torch.matmul(a, x)
        return self.activation(output) + self.bias


class GCN(BaseModel):
    def __init__(self, n_features, n_pred_per_node, predict=False) -> None:
        """
        Graph convoluation model

        Args:
            n_features: int, number of features per node (asset)
            n_pred_per_node: int, number of values to predict for each node (asset)
            predict: bool, does this model directly predict out or is it handled later? If True, creates fully connected layer for prediction
        """
        super().__init__()
        self.gc1 = GraphConv(n_features, n_pred_per_node, 'relu')
        self.predict = predict
        if predict:
            self.fc = nn.Linear(14*n_pred_per_node, 14)

    def forward(self, x, adj=None):
        x = x.view(1, 14, -1) # reshape so that each node's (asset's) features is own row
        if self.predict:
            return self.fc( self.gc1(x, adj).view(1, -1) )
        else:
            return self.gc1(x, adj)