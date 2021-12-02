import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing.data import CryptoFeed, get_crypto_dataset
from models.components import GCN


def train(model, dataset, optimizer, criterion, epochs=2, dl_kws={}, return_all=False):
    dataloader = DataLoader(dataset, batch_size=1, **dl_kws)
    steps_per_epoch = len(dataloader)

    model.train()
    epoch_losses = []
    for e in range(epochs):

        print('Starting epoch {}...'.format(e))

        # add tqdm for progress tracking if desired
        epoch_avg_loss = 0
        for features, target, adj in tqdm(dataloader):
            # any casting to correct datatypes here
            features, target, adj = features.float(), target.float(), adj.float()

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


def main():
    print('Creating model...')
    model = GCN(n_features=7, n_pred_per_node=1).float() # should be combined end-to-end model that combines LSTM and GCN
    print('Model created.\n')
    print('Creating dataset...')
    dataset = get_crypto_dataset(seq_len=1)
    print('Dataset created.\n')
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # can play around with this one
    criterion = nn.MSELoss() # regression problem, could just be MSE?

    print('Starting training...')
    model, losses = train(model, dataset, optimizer, criterion, epochs=1) # change up number of epochs depending on loss plot
    print('Model trained. Saving model...')
    model.save('model checkpoints/trained_model.pth') # replace this probably with command line arg or something, hard coded to fill out skeleton
    print('Model saved.')

    plot_loss(losses)


if __name__ == '__main__':
    main()