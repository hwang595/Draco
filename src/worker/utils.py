from __future__ import print_function
from mpi4py import MPI
import numpy as np

import sys
sys.path.append("..")
from nn_ops import NN_Trainer
from compress_gradient import compress
from datasets.utils import get_batch
from util import *

import torch
from torch.autograd import Variable

import time
from datetime import datetime
import copy
from sys import getsizeof

STEP_START_ = 1

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ModelBuffer(object):
    def __init__(self, network):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        self.recv_buf = []
        self.layer_cur_step = []
        # consider we don't want to update the param of `BatchNorm` layer right now
        # we temporirially deprecate the foregoing version and only update the model
        # parameters
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(np.zeros(param.size()))
            self.layer_cur_step.append(0)