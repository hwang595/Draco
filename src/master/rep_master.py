from .utils import *
from .baseline_master import SyncReplicasMaster_NN

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
                for i, elem in enumerate(v):
                    if _maj_counter == 0:
                        _maj_grad = elem[j]
                        _maj_counter = 1
                    elif np.array_equal(elem[j], _maj_grad):
                        _maj_counter += 1
                    else:
                        _maj_counter -= 1
                assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                self._grad_aggregate_buffer[j] += _maj_grad
        self._grad_aggregate_buffer = map(lambda x:x/float(len(self._group_list)), self._grad_aggregate_buffer)