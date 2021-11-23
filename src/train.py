import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing.data import CryptoFeed, get_crypto_dataset
from preprocessing import *
from models import *


def train(model, dataset, optimizer, criterion, epochs=2, dl_kws=None, return_all=False):
    dataloader = DataLoader(dataset, batch_size=1, **dl_kws)
    steps_per_epoch = len(dataloader)

    model.train()
    epoch_losses = []
    for e in range(epochs):
        # add tqdm for progress tracking if desired
        epoch_avg_loss = 0
        for features, target in dataloader:
            # any casting to correct datatypes here

            output = model(x)
            loss = criterion(output, target) # make sure this is right order
            
            optimizer.zero_grad()
            loss.backward()
            epoch_avg_loss += loss.item()
            # gradient clipping here if desired
            optimizer.step()
        epoch_losses.append( epoch_avg_loss / steps_per_epoch )
        # lr_schedulerer here if desired
    
    if return_all:
        return model, epoch_losses, optimizer # add any other state-based objects like lr_scheduler here for debugging
    else:
        return model, epoch_losses


def plot_loss(losses):
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(losses), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    fig.savefig('loss.png') # can be a variable command line arg or something


def main():
    model = _ # should be combined end-to-end model that combines LSTM and GCN
    dataset = get_crypto_dataset()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # can play around with this one
    criterion = _ # regression problem, could just be MSE?

    model, losses = train(model, dataset, optimizer, criterion, epochs=5) # change up number of epochs depending on loss plot
    model.save('trained_model.pth') # replace this probably with command line arg or something, hard coded to fill out skeleton

    plot_loss(losses)


if __name__ == '__main__':
    main()