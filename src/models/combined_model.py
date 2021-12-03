from typing import ForwardRef
import torch
import torch.nn as nn

from .components import *


class GraphLSTM(BaseModel):
    def __init__(self, n_features, lstm_hidden_dim=14, lstm_n_layers=1, gcn_pred_per_node=1, gcn_n_layers=1, gcn_hidden_dim=1) -> None:
        """
        Combines LSTM and GCN models into single e2e trainable model 

        Args:
            n_features: int, number of features per node (asset)
            lstm_hidden_dim: int, dimensionality of LSTM hidden layer(s)
            lstm_n_layers: int, number of layers in LSTM model
            gcn_pred_per_node: int, number of outputs each node fed into gcn is mapped to i.e. with n nodes output is size n * gcn_pred_per_node
            gcn_n_layers: int, (not supported) number of layers in GCN model
            gcn_hidden_dim: int, (not supported) number of outputs in hidden dim each node in input is mapped to i.e. with n nodes, hidden layer has dimension n * gcn_hidden_dim
        """
        super().__init__()
        self.lstm = LSTM(input_size=14*n_features, hidden_size=lstm_hidden_dim, num_layers=lstm_n_layers, batch_first=True)
        self.gcn = GCN(n_features, gcn_pred_per_node)

        self.model_weights = nn.Parameter(torch.ones(2))

    def initialize_hidden_state(self, batch_size):
        self.lstm_hidden_state = (torch.zeros(1, batch_size, 14), torch.zeros(1, batch_size, 14))

    def forward(self, x, adj):
        # feed through lstm
        lstm_output, self.hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())

        # feed through gcn
        gcn_output = self.gcn(x[-1], adj) # -1 to get latest state since gcn rn takes in just one day's price data

        # combine using learnable weights
        self.model_weights /= self.model_weights.sum() # normalize to sum to 1
        final_output = self.model_weights[0]*lstm_output + self.model_weights[1]*gcn_output

        return final_output