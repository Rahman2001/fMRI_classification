
# Stage 1: Basic Structure and Imports

import os
from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')  # Initialize with a very high loss value
        self.best_accuracy = 0         # Initialize best accuracy to 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device  # Returns the device on which the model is located
