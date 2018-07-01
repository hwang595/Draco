from __future__ import print_function

import sys
import math
import threading
import argparse
import time

from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

from coding import search_w
from util import *


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--mode', type=str, default='normal', metavar='N',
                        help='determine if we use normal averaged gradients or geometric median (in normal mode)\
                         or whether we use normal/majority vote in coded mode to udpate the model')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--err-mode', type=str, default='rev_grad', metavar='N',
                        help='which type of byzantine err we are going to simulate rev_grad/constant/random are supported')
    parser.add_argument('--approach', type=str, default='maj_vote', metavar='N',
                        help='method used to achieve byzantine tolerence, currently majority vote is supported set to normal will return to normal mode')
    parser.add_argument('--num-aggregate', type=int, default=5, metavar='N',
                        help='how many number of gradients we wish to gather at each iteration')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--adversarial', type=int, default=1, metavar='N',
                        help='how much adversary we want to add to a certain worker')
    parser.add_argument('--worker-fail', type=int, default=2, metavar='N',
                        help='how many number of worker nodes we want to simulate byzantine error on')
    parser.add_argument('--group-size', type=int, default=5, metavar='N',
                        help='in majority vote how many worker nodes are in a certain group')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/none indicate if we compress the gradient matrix before communication')
    parser.add_argument('--checkpoint-step', type=int, default=0, metavar='N',
                        help='which step to proceed the training process')  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description='Draco'))

    datum, kwargs_master, kwargs_worker = prepare(args, rank, world_size)
    if args.approach == "baseline":
        train_loader, _, test_loader = datum
        if rank == 0:
            master_fc_nn = baseline_master.SyncReplicasMaster_NN(comm=comm, **kwargs_master)
            master_fc_nn.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
            master_fc_nn.start()
            print("Done sending messages to workers!")
        else:
            worker_fc_nn = baseline_worker.DistributedWorker(comm=comm, **kwargs_worker)
            worker_fc_nn.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
            worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)
            print("Now the next step is: {}".format(worker_fc_nn.next_step))
    # majority vote
    elif args.approach == "maj_vote":
        train_loader, _, test_loader = datum
        if rank == 0:
            coded_master = rep_master.CodedMaster(comm=comm, **kwargs_master)
            coded_master.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(coded_master.world_size, coded_master.cur_step))
            coded_master.start()
            print("Done sending messages to workers!")
        else:
            coded_worker = rep_worker.CodedWorker(comm=comm, **kwargs_worker)
            coded_worker.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(coded_worker.rank, coded_worker.world_size-1, coded_worker.next_step))
            coded_worker.train(train_loader=train_loader, test_loader=test_loader)
            print("Now the next step is: {}".format(coded_worker.next_step))
    # cyclic code
    elif args.approach == "cyclic":
        W, fake_W, W_perp, S, C_1 = search_w(world_size-1, args.worker_fail)
        # for debug print
        #np.set_printoptions(precision=4,linewidth=200.0)
        _, training_set, test_loader = datum
        if rank == 0:
            cyclic_master = cyclic_master.CyclicMaster(comm=comm, **kwargs_master)
            cyclic_master.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(cyclic_master.world_size, cyclic_master.cur_step))
            cyclic_master.start()
            print("Done sending messages to workers!")
        else:
            cyclic_worker = cyclic_worker.CyclicWorker(comm=comm, **kwargs_worker)
            cyclic_worker.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(cyclic_worker.rank, cyclic_worker.world_size-1, cyclic_worker.next_step))
            cyclic_worker.train(training_set=training_set, test_loader=test_loader)
            print("Now the next step is: {}".format(cyclic_worker.next_step))