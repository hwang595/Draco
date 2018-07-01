from .utils import *
from .baseline_worker import DistributedWorker

_FACTOR = 23

class CyclicWorker(DistributedWorker):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.num_workers = self.world_size-1
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._train_dir = kwargs['train_dir']
        self._compress_grad = kwargs['compress_grad']
        self._W = kwargs['encoding_matrix']
        self._fake_W = kwargs['fake_W']
        self._seed = kwargs['seed']
        self._num_fail = kwargs['worker_fail']
        self._eval_freq = kwargs['eval_freq']
        self._hat_s = int(2*self._num_fail+1)
        self._err_mode = kwargs['err_mode']
        self._max_steps = kwargs['max_steps']
        self._fail_workers = kwargs['adversaries']

        # only for test
        # this one is going to be used to avoid fetch the weights for multiple times randomly generate fail worker index
        #self._fail_workers = np.random.choice(np.arange(1, self.num_workers+1), size=self._num_fail, replace=False)
        #self._fail_workers = np.arange(1, self._num_fail+1)
        #self._fail_workers = []
        
        self._layer_cur_step = []
        self._checkpoint_step = 0

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

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()
        #self._param_idx = len(self.network.full_modules)*2-1
        self._param_idx = self.network.fetch_init_channel_index-1

    def train(self, training_set, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        assert(self.cur_step == STEP_START_)
        # for debug print
        np.set_printoptions(precision=4,linewidth=200.0)

        # number of batches in one epoch
        num_batch_per_epoch = len(training_set) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step = 0
        iter_start_time = 0
        first = True
        # use following flags to achieve letting each worker compute more batches
        should_enter_next = False

        print("Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            # after each epoch we need to make sure workers in the same group re-shuffling using the same seed
            torch.manual_seed(self._seed+(_FACTOR*num_epoch))
            batch_bias = 0
            batch_idx = 0
            while batch_bias <= len(training_set):
                if batch_bias+self.batch_size*self.num_workers >= len(training_set):
                    break
                gloabl_image_batch, gloabl_label_batch = get_batch(training_set, np.arange(batch_bias, batch_bias+self.batch_size*self.num_workers))
                batch_bias += self.batch_size*self.num_workers
                batch_idx += 1
                grad_collector = {}
                _precision_counter = 0
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                # iteration start here:
                while True:
                    # the worker shouldn't know the current global step except received the message from parameter server
                    self.async_fetch_step()
                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step()
                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue
                    # the real start point of this iteration
                    iter_start_time = time.time()
                    first = False
                    should_enter_next = False
                    print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))
                    # fetch weight
                    fetch_weight_start_time = time.time()
                    self.async_fetch_weights_bcast()
                    fetch_weight_duration = time.time() - fetch_weight_start_time
                    # calculating on coded batches
                    comp_start = time.time()
                    for b in range(self._hat_s):
                        local_batch_indices = np.where(self._fake_W[self.rank-1]!=0)[0]
                        _batch_bias = local_batch_indices[b]*self.batch_size
                        train_image_batch = gloabl_image_batch[_batch_bias:_batch_bias+self.batch_size,:]
                        train_label_batch = gloabl_label_batch[_batch_bias:_batch_bias+self.batch_size]

                        X_batch, y_batch = Variable(train_image_batch), Variable(train_label_batch)
                        self.network.train()
                        self.optimizer.zero_grad()
                        # forward step
                        logits = self.network(X_batch)
                        logits_1 = Variable(logits.data, requires_grad=True)
                        loss = self.criterion(logits_1, y_batch)

                        # backward step
                        backward_start_time = time.time()
                        loss.backward()

                        init_grad_data = logits_1.grad.data.numpy()
                        init_grad_data = np.sum(init_grad_data, axis=0).astype(np.float64)
                        grads=self.network.backward_coded(logits_1.grad, self.cur_step)
                        # debug settings for resnet
                        if "ResNet" in self.network_config:
                            grads.insert(0,init_grad_data)
                        # gather each batch calculated by this worker
                        grad_collector[_batch_bias/self.batch_size] = grads
                        _prec1, _ = accuracy(logits.data, train_label_batch.long(), topk=(1, 5))
                        _precision_counter += _prec1.numpy()[0]
                    comp_duration = time.time() - comp_start
                    # send linear combinations of gradients of multiple batches
                    encode_counter = 0
                    comm_counter = 0
                    encode_cost, comm_cost=self._send_grads(grad_collector, encode_counter, comm_counter)
                    print('Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, Comp: {:.4f}, Comm: {:.4f}, Encode: {:.4f}, Prec@1: {}'.format(self.rank,
                        self.cur_step, num_epoch, batch_idx * self.batch_size, len(training_set), 
                        (100. * (batch_idx * self.batch_size) / len(training_set)), loss.data[0], time.time()-iter_start_time, comp_duration, comm_cost, encode_cost, _precision_counter/self._hat_s))
                    if self.cur_step%self._eval_freq == 0 and self.rank==1:
                        if "ResNet" in self.network_config:
                            self._evaluate_model(test_loader)
                            self._save_model(file_path=self._generate_model_path())
                        else:
                            pass
                    break

    def _send_grads(self, grad_collector, encode_counter, comm_counter):
        '''
        note that at here we're not sending anything about gradient but linear combination of gradients
        '''
        req_send_check = []
        for i, param in enumerate(reversed(grad_collector[grad_collector.keys()[0]])):
            tmp_encode_start = time.time()
            aggregated_grad = np.zeros(param.shape, dtype=complex)
            # calculate combined gradients
            for k, v in grad_collector.iteritems():
                aggregated_grad = np.add(aggregated_grad, np.dot(self._W[self.rank-1][k], v[len(v)-i-1]))
            encode_counter += (time.time() - tmp_encode_start)
            tmp_comm_start = time.time()
            # send grad to master
            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self.rank in self._fail_workers[self.cur_step]:
                simulation_grad = err_simulation(aggregated_grad, self._err_mode, cyclic=True)
                _compressed_grad = compress(simulation_grad)
                req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                _compressed_grad = compress(aggregated_grad)
                req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                req_send_check.append(req_isend)
            comm_counter += (time.time() - tmp_comm_start)
        tmp_comm_start = time.time()
        req_send_check[-1].wait()
        comm_counter += time.time() - tmp_comm_start
        return encode_counter, comm_counter