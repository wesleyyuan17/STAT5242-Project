import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing.data import get_crypto_dataset
from preprocessing.utils import *
from models.components import GCN, LSTM
from models.combined_model import AdditiveGraphLSTM, SequentialGraphLSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # where to perform training


def train(model, dataset, optimizer, criterion, epochs=2, batch_size=1, dl_kws={}, return_all=False, mode='additive'):
    """
    Function that trains a given model on a given dataset using user-defined optimizer/criterion

    Args:
        model: nn.Module, the model to be trained
        dataset: torch Dataset object, contains data for training the model
        optimizer: torch.optim object, controls learning algorithm used for parameter updates
        criterion: function, some loss function to minimize
        epochs: int, number of epochs to train for
        dl_kws: dict, any arguments to pass to DataLoader object
        return_all: bool, for debugging purposes - if True will return all objects to help observe states
        mode: str, whether training is on lstm, gcn, or a combined model (additive or sequential)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    steps_per_epoch = len(dataloader)
    model.to(device) # send model to desired training device
    
    model.train()
    epoch_losses = []
    for e in range(epochs):

        print('Starting epoch {}...'.format(e))

        if mode != 'gcn':
            # need to initialize hidden state
            # hidden_state = (torch.zeros(1, 1, 14), torch.zeros(1, 1, 14))
            model.initialize_hidden_state(batch_size)

        # add tqdm for progress tracking if desired
        epoch_avg_loss = 0
        for features, target, adj in tqdm(dataloader):
            # any casting to correct datatypes here, send to device 
            features, target, adj = features.float().to(device), target.float().to(device), adj.float().to(device)

            if mode == 'lstm':
                # lstm only takes in sequence of features
                output, hidden_state = model(features)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                output = output[-1]
            else:
                # gcn and combined model both use adjacency matrix
                output = model(features, adj.squeeze())
            loss = criterion(output, target) # make sure this is right order
            
            optimizer.zero_grad()
            loss.backward()
            epoch_avg_loss += loss.item()
            # gradient clipping here if desired
            optimizer.step()
        epoch_losses.append( epoch_avg_loss / steps_per_epoch )
        # lr_scheduler here if desired

        print('Epoch {} completed. Avg epoch loss: {:.4f}'.format(e, epoch_losses[-1]))
    
    if return_all:
        return model, epoch_losses, optimizer # add any other state-based objects like lr_scheduler here for debugging
    else:
        return model, epoch_losses


def plot_loss(losses):
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    fig.savefig('figures/loss.png') # can be a variable command line arg or something


def main(mode, technicals):
    print('Creating model...')
    if mode == 'lstm':
        model = LSTM(input_size=98+14*len(technicals), hidden_size=14, batch_first=True)
    elif mode == 'gcn':
        model = GCN(n_features=7+len(technicals), n_pred_per_node=1) # 7 pre-existing features
    elif mode == 'additive':
        model = AdditiveGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    else:
        model = SequentialGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    model.float()
    print('Model created.\n')
    print('Creating dataset...')
    if model == 'gcn':
        # gcn only takes one market state at a time for now
        dataset = get_crypto_dataset(seq_len=1, technicals=technicals)
    else:
        dataset = get_crypto_dataset(seq_len=10, technicals=technicals)
    print('Dataset created.\n')
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # can play around with this one
    criterion = nn.MSELoss() # regression problem, could just be MSE?

    print('Starting training...')
    model, losses = train(model, dataset, optimizer, criterion, epochs=1, mode=mode) # change up number of epochs depending on loss plot
    print('Model trained. Saving model...')
    model.save('model checkpoints/trained_model.pth') # replace this probably with command line arg or something, hard coded to fill out skeleton
    print('Model saved.')

    plot_loss(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', dest='mode', required=True, choices=['lstm', 'gcn', 'additive', 'sequential'], 
                        help='Which model is going to be trained')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True, 
                        help='json file with mapping of names of features to functions that create feature')  
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        config = json.load(file)

    for k, v in config.items():
        config[k] = eval(v)

    main(args.mode, config)