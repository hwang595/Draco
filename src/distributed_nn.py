from __future__ import print_function

import sys
import math
import threading
import argparse
import time
import random

import numpy as np
from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn.functional as F

from torchvision import datasets, transforms

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

from distributed_worker import *
from sync_replicas_master_nn import *
from coding import search_w

#for tmp solution
from datasets import MNISTDataset
from datasets import Cifar10Dataset

SEED_ = 428

def _group_assign(world_size, group_size, rank):
    '''
    split N worker nodes into k=N/S groups
    '''
    # sanity check we assume world size is divisable by group size
    assert world_size % group_size == 0
    np.random.seed(SEED_)
    ret_group_dict={}
    k = world_size/group_size
    group_list=[[j+i*group_size+1 for j in range(group_size)] for i in range(k)]
    for i, l in enumerate(group_list):
        ret_group_dict[i]=l
    group_seeds = [0]*k
    if rank == 0:
        return ret_group_dict, -1, group_seeds
    for i,group in enumerate(group_list):
        group_seeds[i] = np.random.randint(0, 20000)
        if rank in group:
            group_num = i
    return ret_group_dict, group_num, group_seeds


def _load_data(dataset, seed):
    if seed:
        # in normal method we do not implement random seed here
        # same group should share the same shuffling result
        torch.manual_seed(seed)
        random.seed(seed)
    if dataset == "MNIST":
        training_set = datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
    elif dataset == "Cifar10":
        '''
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        '''
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # data prep for training set
        # note that the key point to reach convergence performance reported in this paper (https://arxiv.org/abs/1512.03385)
        # is to implement data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform_train)
        #training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
        #                                        download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)
    return train_loader, training_set, test_loader

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
    parser.add_argument('--kill-threshold', type=float, default=7.0, metavar='KT',
                        help='timeout threshold which triggers the killing process (default: 7s)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--err-mode', type=str, default='rev_grad', metavar='N',
                        help='which type of byzantine err we are going to simulate rev_grad/constant/random are supported')
    parser.add_argument('--coding-method', type=str, default='maj_vote', metavar='N',
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
    parser.add_argument('--err-case', type=str, default='best_case', metavar='N',
                        help='best_case or worst_case will affect the time cost for majority vote in adversarial coding')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/none indicate if we compress the gradient matrix before communication')
    parser.add_argument('--checkpoint-step', type=int, default=0, metavar='N',
                        help='which step to proceed the training process')  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

    if args.coding_method == "baseline":
        train_loader, _, test_loader = _load_data(dataset=args.dataset, seed=None)
        kwargs_master = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'max_steps':args.max_steps, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'kill_threshold': args.num_aggregate, 'timeout_threshold':args.kill_threshold,
                    'eval_freq':args.eval_freq, 'train_dir':args.train_dir, 'update_mode':args.mode, 'compress_grad':args.compress_grad, 'checkpoint_step':args.checkpoint_step}
        kwargs_worker = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'kill_threshold':args.kill_threshold, 'adversery':args.adversarial, 'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 'compress_grad':args.compress_grad, 'eval_freq':args.eval_freq, 'train_dir':args.train_dir, 'checkpoint_step':args.checkpoint_step}
        if rank == 0:
            master_fc_nn = SyncReplicasMaster_NN(comm=comm, **kwargs_master)
            master_fc_nn.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
            master_fc_nn.start()
            print("Done sending messages to workers!")
        else:
            worker_fc_nn = DistributedWorker(comm=comm, **kwargs_worker)
            worker_fc_nn.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
            worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)
    # majority vote
    elif args.coding_method == "maj_vote":
        group_list, group_num, group_seeds=_group_assign(world_size-1, args.group_size, rank)
        kwargs_master = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'max_steps':args.max_steps, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'kill_threshold': args.num_aggregate, 'timeout_threshold':args.kill_threshold,
                    'eval_freq':args.eval_freq, 'train_dir':args.train_dir, 'group_list':group_list, 'update_mode':args.mode, 'compress_grad':args.compress_grad, 'checkpoint_step':args.checkpoint_step}
        kwargs_worker = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'kill_threshold':args.kill_threshold, 'adversery':args.adversarial, 'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 'group_list':group_list, 'group_seeds':group_seeds, 'group_num':group_num,
                    'err_case':args.err_case, 'compress_grad':args.compress_grad, 'eval_freq':args.eval_freq, 'train_dir':args.train_dir}
        if rank == 0:
            coded_master = CodedMaster(comm=comm, **kwargs_master)
            coded_master.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(coded_master.world_size, coded_master.cur_step))
            coded_master.start()
        else:
            train_loader, _, test_loader = _load_data(dataset=args.dataset, seed=group_seeds[group_num])
            coded_worker = CodedWorker(comm=comm, **kwargs_worker)
            coded_worker.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(coded_worker.rank, coded_worker.world_size-1, coded_worker.next_step))
            coded_worker.train(train_loader=train_loader, test_loader=test_loader)
    # cyclic code
    elif args.coding_method == "cyclic":
        W, fake_W, W_perp, S, C_1 = search_w(world_size-1, args.worker_fail)
        # for debug print
        #np.set_printoptions(precision=4,linewidth=200.0)
        kwargs_master = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'max_steps':args.max_steps, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'eval_freq':args.eval_freq, 'train_dir':args.train_dir, 'compress_grad':args.compress_grad, 'W_perp':W_perp, 'W':W, 'worker_fail':args.worker_fail,
                    'decoding_S':S, 'C_1':C_1}
        kwargs_worker = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum, 'network':args.network,
                    'comm_method':args.comm_type, 'adversery':args.adversarial, 'worker_fail':args.worker_fail, 'err_mode':args.err_mode, 'compress_grad':args.compress_grad,
                     'encoding_matrix':W, 'seed':SEED_, 'fake_W':fake_W}
        if rank == 0:
            new_master = CyclicMaster(comm=comm, **kwargs_master)
            new_master.build_model()
            print("I am the master: the world size is {}, cur step: {}".format(new_master.world_size, new_master.cur_step))
            new_master.start()
        else:
            _, training_set, _ = _load_data(dataset=args.dataset, seed=SEED_)
            new_worker = CyclicWorker(comm=comm, **kwargs_worker)
            new_worker.build_model()
            print("I am worker: {} in all {} workers, next step: {}".format(new_worker.rank, new_worker.world_size-1, new_worker.next_step))
            new_worker.train(training_set=training_set)