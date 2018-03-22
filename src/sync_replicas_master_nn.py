from __future__ import print_function
import time
import copy

from mpi4py import MPI
import numpy as np
from scipy import linalg as LA
from scipy import fftpack as FT
from scipy.optimize import lsq_linear
from sys import getsizeof

from nn_ops import NN_Trainer
from optim.sgd_modified import SGDModified

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.vgg import *
from model_ops.fc_nn import FC_NN, FC_NN_Split
import hdmedians as hd
from compress_gradient import decompress
import c_coding

import torch

STEP_START_ = 1

def update_params_dist_version(param, avg_grad, learning_rate):
    '''
    update the network layer by layer
    '''
    avg_grad = avg_grad.reshape(param.shape)
    assert param.shape == avg_grad.shape
    param -= learning_rate * avg_grad
    return param

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


class SyncReplicasMaster_NN(NN_Trainer):
    def __init__(self, comm, **kwargs):
        '''
        master node here, no rank needed since the rank will always be 0 for master node
        '''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._timeout_threshold = kwargs['timeout_threshold']

        self._num_grad_to_collect = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._expected_grad_to_recv = kwargs['kill_threshold']
        self._max_steps = kwargs['max_steps']
        self._update_mode = kwargs['update_mode']
        self._compress_grad = kwargs['compress_grad']
        self._checkpoint_step = kwargs['checkpoint_step']

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            #self.network=LeNetSplit()
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNetSplit18()
        elif self.network_config == "ResNet34":
            self.network=ResNetSplit34()
        elif self.network_config == "ResNet50":
            self.network=ResNetSplit50()
        elif self.network_config == "ResNet101":
            self.network=ResNetSplit101()
        elif self.network_config == "ResNet152":
            self.network=ResNetSplit152()
        elif self.network_config == "FC":
            self.network=FC_NN_Split()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn()
        elif self.network_config == "VGG13":
            self.network=vgg13_bn()
        elif self.network_config == "VGG16":
            self.network=vgg16_bn()

        if self._checkpoint_step != 0:
            file_path = "../checkpoints/geo_median/model_step_"+str(self._checkpoint_step)
            self._load_model(file_path)
            self.cur_step = int(self._checkpoint_step)+1

        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad)
        self.init_model_shapes()
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def start(self):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        # please note that step is start from one here
        self.async_bcast_step()

        # fake test here:
        for i in range(1, self._max_steps):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            print("Master node is entering step: {}".format(i))

            self.async_bcast_step()

            if self.comm_type == "Bcast":
                self.async_bcast_layer_weights_bcast()
            elif self.comm_type == "Async":
                self.async_bcast_layer_weights_async()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()

            # wait for enough gradients to be aggregated:
            while not enough_gradients_received:
                status = MPI.Status()
                if self._compress_grad == "None":
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress":
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    received_grad=decompress(received_msg)

                if status.tag-88 in self.grad_accumulator.model_index_range:
                    if not self._first_grad_received:
                        self._first_grad_received=True
                        grad_gather_start_time = time.time()

                    layer_index = status.tag-88
                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])

                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index)
                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

            if self._update_mode == "normal":
                method_start = time.time()
                self._avg_received_grads()
                method_duration = time.time()-method_start
            elif self._update_mode == "geometric_median":
                method_start = time.time()
                self._get_geo_median()
                method_duration = time.time()-method_start

            # update using SGD method
            update_start = time.time()

            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)

            # update `state_dict` in pytorch modules
            #self.model_update(tmp_module)
            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()
            # save model for validation in a pre-specified frequency
            if self.cur_step%self._eval_freq == 0:
                if "ResNet" not in self.network_config:
                    self._save_model(file_path=self._generate_model_path())
            print("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            self.cur_step += 1

    def init_model_shapes(self):
        for param_idx, param in enumerate(self.network.parameters()):
            self._model_shapes.append(param.size())
            if self._update_mode == "normal":
                self._grad_aggregate_buffer.append(np.zeros(param.size()))
            elif self._update_mode == "geometric_median":
                self._grad_aggregate_buffer.append([])

    def async_bcast_step(self):
        req_list = []
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
        for i in range(len(req_list)):
            req_list[i].wait()

    def async_bcast_layer_weights_async(self):
        request_layers = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            request_workers = []
            layer_to_send = layer.data.numpy().astype(np.float64)
            for i in range(self.world_size):
                if i != 0:
                    req = self.comm.Isend([layer_to_send, MPI.DOUBLE], dest=i, tag=11+layer_idx)
                    request_workers.append(req)

            request_layers.append(request_workers)
        # TODO(hwang): check to see if these `wait` calls are necessary here
        for req_l in request_layers:
            for req_worker in req_l:
                req_worker.wait()

    def async_bcast_layer_weights_bcast(self):
        request_layers = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            request_workers = []
            layer_to_send = layer.data.numpy().astype(np.float64)
            # try to see if collective communication is better here:
            self.comm.Bcast([layer_to_send, MPI.DOUBLE], root=0)

    def async_fetch_gradient_start(self):
        '''
        make gradient fetch requests and return the request list
        '''
        gradient_fetch_requests = [] # `graident_fetch_request` should have length of #fc_layer*num_grad_to_collect
        for layer_idx, layer in enumerate(self.network.parameters()):
            for k in range(self._num_grad_to_collect):
                if self._compress_grad == 'compress':
                    req = self.comm.irecv(self.grad_accumulator.gradient_aggregator[layer_idx][k], source=k+1, tag=88+layer_idx)
                else:
                    req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[layer_idx][k], MPI.DOUBLE], source=k+1, tag=88+layer_idx)
                gradient_fetch_requests.append(req)
        return gradient_fetch_requests

    def aggregate_gradient(self, gradient, layer_idx):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        if self._update_mode == "normal":
            self._grad_aggregate_buffer[layer_idx] += gradient
        elif self._update_mode == "geometric_median":
            _shape = gradient.shape
            if len(_shape)==1:
                self._grad_aggregate_buffer[layer_idx].append(gradient)             
            elif len(_shape)>1:
                self._grad_aggregate_buffer[layer_idx].append(gradient.reshape((reduce(lambda x, y: x * y, _shape),)))

    def model_update(self, tmp_module):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name:
                tmp_dict = {key_name : param}
            else:
                assert param.size() == tmp_module[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(tmp_module[model_counter_])}
                model_counter_+=1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            if self._update_mode == "normal" or self._update_mode == "maj_vote":
                self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape)
            elif self._update_mode == "geometric_median":
                self._grad_aggregate_buffer[i] = []

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network, f_)
        return

    def _load_model(self, file_path):
        model_state_dict=torch.load(file_path)
        self.network.load_state_dict(model_state_dict)
        print("Master Done Loading Checkpoint from {}".format(file_path))

    def _evaluate_model(self, validation_loader):
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        # which indicate an epoch based validation is done
        while validation_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1, 5))
            prec1_counter_ += prec1_tmp
            prec5_counter_ += prec5_tmp
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        self._epoch_counter = validation_loader.dataset.epochs_completed
        print('Testset Performance: Cur Step:{} Prec@1: {} Prec@5: {}'.format(self.cur_step, prec1.numpy()[0], prec5.numpy()[0]))

    def _avg_received_grads(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] /= self._expected_grad_to_recv

    def _get_geo_median(self):
        geo_median_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            geo_median = np.array(hd.geomedian(np.array(grads), axis=0))
            self._grad_aggregate_buffer[g_idx] = geo_median
        print("Master Step: {} Found Geo Median Cost: {:.4f}".format(self.cur_step, time.time()-geo_median_start))


class CodedMaster(SyncReplicasMaster_NN):
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._timeout_threshold = kwargs['timeout_threshold']

        self._num_grad_to_collect = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._coded_grads_buffer = {}
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._expected_grad_to_recv = kwargs['kill_threshold']
        self._update_mode = kwargs['update_mode']
        self._max_steps = kwargs['max_steps']
        self._group_list = kwargs['group_list']
        self._compress_grad = kwargs['compress_grad']
        self._group_size = len(self._group_list[0])

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNetSplit()
        elif self.network_config == "ResNet18":
            self.network=ResNetSplit18()
        elif self.network_config == "ResNet34":
            self.network=ResNetSplit34()
        elif self.network_config == "ResNet50":
            self.network=ResNetSplit50()
        elif self.network_config == "FC":
            self.network=FC_NN_Split()

        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad)
        self.init_model_shapes()
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def init_model_shapes(self):
        tmp_aggregate_buffer = []
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            self._model_shapes.append(shape)
            self._grad_aggregate_buffer.append(np.zeros(shape))
            tmp_aggregate_buffer.append(np.zeros(shape))

        if self._update_mode == "maj_vote":
            for k, v in self._group_list.iteritems():
                for i, l in enumerate(v):
                    if k not in self._coded_grads_buffer.keys():
                        self._coded_grads_buffer[k] = [copy.deepcopy(tmp_aggregate_buffer)]
                    else:
                        self._coded_grads_buffer[k].append(copy.deepcopy(tmp_aggregate_buffer))

    def start(self):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure value fetched from ps is 1
        self.async_bcast_step()

        # fake test here:
        for i in range(1, self._max_steps):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            print("Master node is entering step: {}".format(i))
            self.async_bcast_step()

            if self.comm_type == "Bcast":
                self.async_bcast_layer_weights_bcast()
            elif self.comm_type == "Async":
                self.async_bcast_layer_weights_async()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()
            # wait for enough gradients to be aggregated:
            while not enough_gradients_received:
                status = MPI.Status()
                if self._compress_grad == "None":
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress":
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    received_grad=decompress(received_msg)

                if status.tag-88 in self.grad_accumulator.model_index_range:
                    if not self._first_grad_received:
                        self._first_grad_received=True
                        grad_gather_start_time = time.time()

                    layer_index = status.tag-88

                    # BUG: sometimes this can be zero
                    #received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])

                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(received_grad, layer_index, status.source)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)
            
            if self._update_mode == "normal":
                method_start = time.time()
                self._avg_received_grads()
                method_duration = time.time() - method_start
            elif self._update_mode == "maj_vote":
                # under development, stay tunned
                method_start = time.time()
                self._grad_majority_vote()
                method_duration = time.time() - method_start

            update_start = time.time()
            # update using SGD method
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)
            # update `state_dict` in pytorch modules
            #self.model_update(tmp_module)
            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()
            # save model for validation in a pre-specified frequency
            if self.cur_step%self._eval_freq == 0:
                self._save_model(file_path=self._generate_model_path())
            print("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            self.cur_step += 1

    def aggregate_gradient(self, gradient, layer_idx, source):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        if self._update_mode == "normal":
            self._grad_aggregate_buffer[layer_idx] += gradient
        elif self._update_mode == "maj_vote":
            # under development, stay tunned
            for k, v in self._group_list.iteritems():
                if source in v:
                    assert self._coded_grads_buffer[k][v.index(source)][layer_idx].shape == gradient.shape
                    self._coded_grads_buffer[k][v.index(source)][layer_idx] = gradient

    def _grad_majority_vote(self):
        for k, v in self._coded_grads_buffer.iteritems():
            for j, _ in enumerate(self.network.parameters()):
                _maj_counter = 0
                _maj_found = False
                _maj_search_index = 0
                _maj_grad = v[_maj_search_index][j]
                while not _maj_found:
                    for i, elem in enumerate(v):
                        if np.array_equal(elem[j], _maj_grad):
                            _maj_counter += 1
                    if _maj_counter > self._group_size/2:
                        _maj_found=True
                    else:
                        _maj_counter = 0
                        _maj_search_index += 1
                        _maj_grad = v[_maj_search_index][j]
                # write maj grad into grad aggregate buffer
                assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                self._grad_aggregate_buffer[j] += _maj_grad
        # average among groups
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] /= len(self._group_list)


class CyclicMaster(SyncReplicasMaster_NN):
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.num_workers = self.world_size-1
        self.s = kwargs['worker_fail']
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._num_grad_to_collect = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._coded_grads_buffer = {}
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._update_mode = "normal"
        self._max_steps = kwargs['max_steps']
        self._compress_grad = kwargs['compress_grad']
        self._W_perp = kwargs['W_perp']
        self._W = kwargs['W']
        self._S = kwargs['decoding_S']

        self._C_1 = kwargs['C_1']

        self._estimator = self._estimator_generator(self.num_workers, self.s) # n by s+1 complex matrix
        self._poly_a = np.zeros(self.s+1, dtype=complex)
        self._poly_a[-1] = 1+0j
        # 1 by n-2s
        self._row_vec = np.zeros((1, self.num_workers-2*self.s))
        self._row_vec[0][0]=1

        self.vec = np.zeros(self.num_workers-2*self.s)
        self.vec[0] = 1

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNetSplit()
        elif self.network_config == "ResNet18":
            self.network=ResNetSplit18()
        elif self.network_config == "ResNet34":
            self.network=ResNetSplit34()
        elif self.network_config == "FC":
            self.network=FC_NN_Split()

        # assign a gradient accumulator to collect gradients from workers
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad)
        self.init_model_shapes()
        self._rand_factors = []
        for param in self.network.parameters():
            _dim = reduce(lambda x, y: x * y, param.size())
            self._rand_factors.append(np.random.normal(loc=1.0, size=_dim))

    def init_model_shapes(self):
        tmp_aggregate_buffer = []
        # received gradient matrix
        self._R = []
        for param_idx, param in enumerate(self.network.parameters()):
            _shape = param.size()
            self._model_shapes.append(_shape)
            self._grad_aggregate_buffer.append(np.zeros(_shape))
            tmp_aggregate_buffer.append(np.zeros(_shape))
            # construct R
            self._R.append(np.zeros((self.num_workers, reduce(lambda x, y: x * y, _shape)),dtype=complex))

    def start(self):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure value fetched from ps is 1
        self.async_bcast_step()
        # fake test here:
        for i in range(1, self._max_steps):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            print("Master node is entering step: {}".format(i))

            self.async_bcast_step()
            self.async_bcast_layer_weights_bcast()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()
            # wait for enough gradients to be aggregated:
            while not enough_gradients_received:
                status = MPI.Status()
                if self._compress_grad == "None":
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress":
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    received_grad=decompress(received_msg)


                if status.tag-88 in self.grad_accumulator.model_index_range:
                    if not self._first_grad_received:
                        self._first_grad_received=True
                        grad_gather_start_time = time.time()

                    layer_index = status.tag-88

                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])
                    
                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self._fill_R(layer_index, status.source, received_grad)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)
            
            method_start = time.time()
            for layer_index, R in enumerate(self._R):
                decoded_grad=self._decoding(R, self._rand_factors[layer_index])
                self._grad_aggregate_buffer[layer_index] = np.real(decoded_grad)/self.num_workers
            method_duration = time.time()-method_start

            update_start = time.time()

            # update `state_dict` in pytorch modules
            #self.model_update(tmp_module)
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode="cyclic")
            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            # save model for validation in a pre-specified frequency
            if self.cur_step%self._eval_freq == 0:
                self._save_model(file_path=self._generate_model_path())
            print("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            self.cur_step += 1

    def _fill_R(self, layer_index, src, recv_grad):
        recv_grad = recv_grad.reshape((reduce(lambda x, y: x * y, recv_grad.shape),))
        # sanity check
        assert self._R[layer_index][src-1].shape == recv_grad.shape
        self._R[layer_index][src-1] = recv_grad

    def _decoding(self, R, random_factor):
        _recover_final = np.zeros((1, self.num_workers), dtype=complex)
        E_combined = np.dot(R, random_factor)

        # move this part to wrapped C code:
        alpha = c_coding.solve_poly_a(n=self.num_workers, s=self.s, R=E_combined)

        self._poly_a[0:self.s] = -alpha.reshape(-1)
        estimation = np.dot(self._estimator, self._poly_a)

        err_indices = [i for i, elem in enumerate(estimation) if (np.absolute(elem.real) > 1e-9 or np.absolute(elem.imag) > 1e-9)]

        recover=self._C_1.take(err_indices, axis=0).take(np.arange(self.num_workers-2*self.s),axis=0)

        res = lsq_linear(np.transpose(recover), self.vec)
        new_v = np.transpose(res.x)
        
        remaining_indices = err_indices[0:self.num_workers-2*self.s]

        _recover_final[0][[remaining_indices]] = new_v
        decoded_grad = np.dot(_recover_final, R)
        return decoded_grad[0]

    def _obtain_E(self, alpha, E_2, s):
        # obtain E_1 in shape of n-2s by d
        self._tmp_y = np.zeros((E_2.shape[1], self.num_workers-s), dtype=complex)

        #self._processing_y = np.transpose(E_2)[:, -s:]

        self._tmp_y[:,0:s] = np.transpose(E_2)[:, -s:]
        [self._process(s, alpha, i) for i in range(self.num_workers-2*s)]
        return np.transpose(self._tmp_y[:, s:])

    def _process(self, s, alpha, i):
        tmp = np.dot(self._tmp_y[:,i:s+i], alpha)
        self._tmp_y[:,s+i] = tmp.reshape(-1)

    def _obtain_epsilon(self, E):
        return FT.ifft(E, axis=0)

    def _estimator_generator(self, n, s):
        estimator = np.zeros((n, s+1), dtype=complex)
        #z_gen_func = np.vectorize(lambda t: np.exp(-2*np.pi*t*1j/n))
        z_gen_func = np.vectorize(lambda t: np.exp(2*np.pi*t*1j/n))
        col1 = z_gen_func(np.arange(n))
        for i in range(s+1):
            estimator[:, i] = np.power(col1, i)
        return estimator

def _cls_solver(A, b):
    return np.dot(np.dot(np.linalg.inv(np.dot(_array_getH(A), A)), _array_getH(A)),b)

def _array_getH(ndarray):
    # get conjugate transpose of a np.ndarray
    return ndarray.conj().T