import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(nn.Linear(784, 256, bias = True),
                          nn.ReLU(),
                          nn.Linear(256, 16, bias = True),
                          nn.ReLU(),
                          nn.Linear(16, 10, bias = True),
                          nn.ReLU())
    return model

def count_parameters(model):
    s = 0
    for param in model.parameters():
      k = torch.tensor(param.size()).size()[0]
      if k == 1:
        s += param.size()[0]
      else:
        s += param.size()[0]*param.size()[1]
    return s
    
