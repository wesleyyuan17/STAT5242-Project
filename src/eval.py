import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.data import get_crypto_dataset
from preprocessing.utils import *
from models.components import GCN, LSTM
from models.combined_model import AdditiveGraphLSTM, SequentialGraphLSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # where to perform training


def evaluate(model, dataset, criterion, batch_size=1, dl_kws={}, mode='additive'):
    """
    Function that trains a given model on a given dataset using user-defined optimizer/criterion

    Args:
        model: nn.Module, the model to be trained
        dataset: torch Dataset object, contains data for training the model
        criterion: function, some loss function to minimize
        dl_kws: dict, any arguments to pass to DataLoader object
        mode: str, whether training is on lstm, gcn, or a combined model (additive or sequential)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    steps_per_epoch = len(dataloader)
    model.to(device) # send model to desired training device

    model.eval()
    losses = []
    for features, target, adj in tqdm(dataloader):
        # any casting to correct datatypes here, send to device 
        features, target, adj = features.float().to(device), target.float().to(device), adj.float().to(device)

        if mode != 'gcn':
            # need to initialize hidden state
            model.initialize_hidden_state(batch_size)

        if mode == 'lstm':
            # lstm only takes in sequence of features
            output, hidden_state = model(features)
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        else:
            # gcn and combined model both use adjacency matrix
            output = model(features, adj.squeeze())
        loss = criterion(output, target) # make sure this is right order
        losses.append(loss.item())
    
    return losses


def main(eval_model, technicals, model_name):
    print('Creating model...')
    if eval_model == 'lstm':
        model = LSTM(input_size=98+14*len(technicals), hidden_size=14, batch_first=True, predict=True)
    elif eval_model == 'gcn':
        model = GCN(n_features=7+len(technicals), n_pred_per_node=1, predict=True) # 7 pre-existing features
    elif eval_model == 'additive':
        model = AdditiveGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    else:
        model = SequentialGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    model.load('model checkpoints/{}.pth'.format(model_name))
    model.float()
    print('Model created.\n')

    print('Creating dataset...')
    if model == 'gcn':
        # gcn only takes one market state at a time for now
        dataset = get_crypto_dataset(seq_len=1, technicals=technicals, evaluation=True)
    else:
        dataset = get_crypto_dataset(seq_len=10, technicals=technicals, evaluation=True) # length 10 window was good in other papers, could tune if desired
    print('Dataset created.\n')

    criterion = nn.MSELoss()
    losses = evaluate(model, dataset, criterion, mode=eval_model)
    with open('results/{}_loss.txt'.format(model_name), 'w') as f:
        # output losses to file for later
        for l in losses:
            f.write('{}\n'.format(l))
    
    print('Average MSE for {}: {:.8f}'.format(model_name, np.mean(losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', dest='eval_model', required=True, 
                        choices=['lstm', 'gcn', 'additive', 'sequential'], 
                        help='Which model is going to be evaluated')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True, 
                        help='json file with mapping of names of features to functions that create feature')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Name for loading model to local directory')
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        config = json.load(file)

    for k, v in config.items():
        config[k] = eval(v)

    main(args.eval_model, config, args.model_name)
