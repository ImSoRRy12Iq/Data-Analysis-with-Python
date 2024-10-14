import numpy as np
import torch
from torch import nn

def create_model():
    NN = nn.Sequential(nn.Linear(784, 256, bias=True),
                   nn.ReLU(),
                   nn.Linear(256, 16, bias=True),
                   nn.ReLU(),
                   nn.Linear(16, 10, bias=True))
    return NN

def count_parameters(model):
    s = 0
    for param in model.parameters():
      if len(param.shape) == 2:
        s += param.shape[0]*param.shape[1]
      else: s += param.shape[0]
    return s
