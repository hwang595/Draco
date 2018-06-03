from __future__ import print_function
import time
import copy
from sys import getsizeof

from mpi4py import MPI
import numpy as np
import hdmedians as hd
from scipy import linalg as LA
from scipy import fftpack as FT
from scipy.optimize import lsq_linear
import torch

import sys
sys.path.append("..")
from nn_ops import NN_Trainer
from optim.sgd_modified import SGDModified
from compress_gradient import decompress
import c_coding
from util import *


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

class GradientAccumulator(object):
    '''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
    def __init__(self, module, num_worker, mode='None'):
        # we will update this counter dynamically during the training process
        # the length of this counter should be number of fc layers in the network
        # we used list to contain gradients of layers
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = []
        self._mode = mode
        
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(num_worker):
                if self._mode == 'None':
                    tmp_aggregator.append(np.zeros((param.size())))
                elif self._mode == 'compress':
                    _shape = param.size()
                    if len(_shape) == 1:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0],)))*2))
                    else:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros(_shape))*2))
            # initialize the gradient aggragator
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)

    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    def _meset_grad_aggregator(self):
        '''
        reset the buffers in grad accumulator, not sure if this is necessary
        '''
        if self._mode == 'compress':
            pass
        else:
            for i, tmp_aggregator in enumerate(self.gradient_aggregator):
                for j, buf in enumerate(tmp_aggregator):
                    self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)