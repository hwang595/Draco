import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

# we use LeNet here for our simple case
class FC_NN(nn.Module):
    def __init__(self):
        super(FC_NN, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    def name(self):
        return 'fc_nn'