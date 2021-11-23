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

    def forward(self, x):
        pass


class GCN(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass