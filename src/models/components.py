import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class LSTM(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, num_layers=2, batch_first=True)

    def forward(self, x):
        return self.lstm(x)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation, adj=None) -> None:
        super().__init__()
        if adj is not None:
            # constant adjacency convolutions
            if adj[0,0] == 0:
                self.a = torch.Tensor(adj + np.eye(adj.shape[0]), requires_grad=False)
            else:
                self.a = torch.Tensor(adj, requires_grad=False)
            
            self.d_inv = np.linalg.inv( torch.sum(self.a, axis=1) )
            self.sqrt_d_inv = sqrtm(self.d_inv)

        self.weight = torch.FloatTensor(in_dim, out_dim)
        self.bias = torch.FloatTensor(out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, adj=None):
        if adj is None:
            a = self.a
            sqrt_d_inv = self.sqrt_d_inv
        else:
            if adj[0,0] == 0:
                a = torch.Tensor(adj + np.eye(adj.shape[0]), requires_grad=False)
            else:
                a = torch.Tensor(adj, requires_grad=False)
            
            d_inv = np.linalg.inv( torch.sum(a, axis=1) )
            sqrt_d_inv = sqrtm(d_inv)

        x = torch.mm(x, self.weight)
        output = sqrt_d_inv.dot(a).dot(sqrt_d_inv).dot(x)
        return self.activation(output) + self.bias


class GCN(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass