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
                self.a = torch.tensor(adj + np.eye(adj.shape[0]), requires_grad=False)
            else:
                self.a = torch.tensor(adj, requires_grad=False)
            
            self.d_inv = np.linalg.inv( torch.sum(self.a, axis=1) )
            self.sqrt_d_inv = sqrtm(self.d_inv)

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

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
            
            d_inv = np.linalg.inv( torch.diag(torch.sum(a, axis=1)) )
            sqrt_d_inv = torch.tensor(sqrtm(d_inv), dtype=self.weight.dtype)
    
        x = torch.matmul(x, self.weight)
        output = torch.matmul(torch.mm(torch.mm(sqrt_d_inv, a), sqrt_d_inv), x)
        return self.activation(output) + self.bias


class GCN(BaseModel):
    def __init__(self, n_features, n_pred_per_node) -> None:
        super().__init__()
        self.gc1 = GraphConv(n_features, n_pred_per_node, 'relu')

    def forward(self, x, adj=None):
        x = x.view(1, 14, -1) # reshape so that each node's (asset's) features is own row
        return self.gc1(x, adj).view(-1, 14) # hacky way of getting the right output shape (n, 14, 1) -> (n, 14)