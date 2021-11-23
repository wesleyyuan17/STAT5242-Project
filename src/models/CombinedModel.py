from typing import ForwardRef
import torch
import torch.nn as nn

from components import *


class GraphLSTM(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass