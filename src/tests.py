import pytest
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.data import CryptoFeed, get_crypto_dataset
from src.models.components import GCN


@pytest.fixture()
def dataset():
    return get_crypto_dataset()


def test_dataset_loading(dataset):
    assert isinstance(dataset, Dataset)
    x, y = next(iter(dataset))
    assert len(y) == 14 # number of targets, shape of x changes depending on which features


def test_gcn():
    gcn = GCN(4, 2).float()
    x = np.array([4, 2, 1, 1], dtype=np.float32)
    adj = np.array([[1, 1], 
                    [1, 1]])
    x = torch.tensor(x)
    gcn(x, adj)
