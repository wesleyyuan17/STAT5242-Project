from typing import ForwardRef
import torch
import torch.nn as nn

from .components import *


class AdditiveGraphLSTM(BaseModel):
    def __init__(self, n_features, lstm_hidden_dim=14, lstm_n_layers=1, gcn_pred_per_node=1, gcn_n_layers=1, gcn_hidden_dim=1) -> None:
        """
        Combines LSTM and GCN models into single e2e trainable model where each model predicts separately then averaged as final output

        Args:
            n_features: int, number of features per node (asset)
            lstm_hidden_dim: int, dimensionality of LSTM hidden layer(s)
            lstm_n_layers: int, number of layers in LSTM model
            gcn_pred_per_node: int, number of outputs each node fed into gcn is mapped to i.e. with n nodes output is size n * gcn_pred_per_node
            gcn_n_layers: int, (not supported) number of layers in GCN model
            gcn_hidden_dim: int, (not supported) number of outputs in hidden dim each node in input is mapped to i.e. with n nodes, hidden layer has dimension n * gcn_hidden_dim
        """
        super().__init__()
        self.n_features = n_features
        self.lstm_input_dim = 14 * n_features
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_n_layers = lstm_n_layers

        self.lstm = LSTM(input_size=14*n_features, hidden_size=lstm_hidden_dim, num_layers=lstm_n_layers, batch_first=True, predict=True)
        self.gcn = GCN(n_features, gcn_pred_per_node, predict=True)

        self.model_weights = nn.Parameter(torch.ones(2))

    def initialize_hidden_state(self, batch_size):
        self.batch_size = batch_size
        self.lstm.initialize_hidden_state(batch_size)
        self.lstm_hidden_state = (torch.zeros(self.lstm_n_layers, batch_size, 14), torch.zeros(self.lstm_n_layers, batch_size, 14))

    def forward(self, x, adj):
        # feed through lstm
        lstm_output, hidden_state = self.lstm(x, self.lstm_hidden_state) # lstm wrapper only returns last output, don't need to index later
        self.lstm_hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        # feed through gcn
        gcn_output = self.gcn(x[:, -1, :], adj).view(self.batch_size, -1) # get latest state since gcn rn takes in just one day's price data, flatten before fc

        # combine using learnable weights
        self.model_weights = nn.Parameter(self.model_weights / self.model_weights.sum()) # normalize to sum to 1, wrap to be parameter
        final_output = self.model_weights[0]*lstm_output + self.model_weights[1]*gcn_output

        return final_output


class SequentialGraphLSTM(BaseModel):
    def __init__(self, n_features, lstm_hidden_dim=14, lstm_n_layers=1, gcn_pred_per_node=1) -> None:
        """
        Combines LSTM and GCN models into single e2e trainable model where LSTM provides embedding for GCN and final FC layer does predictions

        Args:
            n_features: int, number of features per node (asset)
            lstm_hidden_dim: int, dimensionality of LSTM hidden layer(s)
            lstm_n_layers: int, number of layers in LSTM model
            gcn_pred_per_node: int, number of outputs each node fed into gcn is mapped to i.e. with n nodes output is size n * gcn_pred_per_node
            gcn_n_layers: int, (not supported) number of layers in GCN model
            gcn_hidden_dim: int, (not supported) number of outputs in hidden dim each node in input is mapped to i.e. with n nodes, hidden layer has dimension n * gcn_hidden_dim
        """
        super().__init__()
        self.n_features = n_features
        self.lstm_input_dim = 14 * n_features
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_n_layers = lstm_n_layers

        self.lstm = LSTM(input_size=n_features, hidden_size=lstm_hidden_dim, num_layers=lstm_n_layers, batch_first=True)
        self.gcn = GCN(lstm_hidden_dim, gcn_pred_per_node)
        self.fc = nn.Linear(14*gcn_pred_per_node, 14)

    def initialize_hidden_state(self, batch_size):
        self.batch_size = batch_size
        self.lstm_hidden_state = (torch.zeros(self.lstm_n_layers, batch_size, 14), torch.zeros(self.lstm_n_layers, batch_size, 14))

    def forward(self, x, adj=None):
        x = x.view(self.batch_size, 10, 14, -1).permute(2, 0, 1, 3) # reshape so that each node's (asset's) features is own row, have assets first
        seq_embeddings = []
        for features in x:
            self.initialize_hidden_state(self.batch_size)
            lstm_output, hidden_state = self.lstm(features, self.lstm_hidden_state)
            self.hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            seq_embeddings.append(lstm_output) # lstm wrapper only returns last output
        gcn_input = torch.cat(seq_embeddings) # should be 14xlstm_hidden_dim here
        gcn_output = self.gcn(gcn_input, adj)
        final_output = self.fc( gcn_output.view(self.batch_size, -1) )
        return final_output



        