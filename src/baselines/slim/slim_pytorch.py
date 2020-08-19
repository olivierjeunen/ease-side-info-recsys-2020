import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
from torch import nn

# SLIM/EASE-like item-based model

class SLIM_PyTorch(nn.Module):
    def __init__(self, args):
        super(SLIM_PyTorch, self).__init__()
        self.num_items = args['num_items']
        self.W = torch.nn.Linear(self.num_items, self.num_items, bias = True)

    def forward(self, x, dropout_rate):
        x = torch.nn.functional.dropout(x, p = dropout_rate)
        return self.W(x)

    def clear_diag(self):
        self.W.weight.fill_diagonal_(0)
