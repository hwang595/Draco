from .utils import *

class SyncReplicasMaster_NN(NN_Trainer):
    def __init__(self, comm, **kwargs):
        '''
        master node here, no rank needed since the rank will always be 0 for master node
        '''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.num_workers = self.world_size-1
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
        self._s = kwargs['worker_fail']

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
            elif self._update_mode == "krum":
                method_start = time.time()
                self._krum()
                method_duration = time.time()-method_start

            # update using SGD method
            update_start = time.time()

            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)

            # update `state_dict` in pytorch modules
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
            elif self._update_mode in ("geometric_median", "krum"):
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
        elif self._update_mode in ("geometric_median", "krum"):
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
            elif self._update_mode in ("geometric_median", "krum"):
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

    def _krum(self):
        def __krum(grad_list, s):
            '''
            Method introduced by: https://arxiv.org/abs/1703.02757
            '''
            score = []
            for i, g_i in enumerate(grad_list):
                neighbor_distances = []
                for j, g_j in enumerate(grad_list):
                    if i != j:
                        neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            i_star = score.index(min(score))
            return grad_list[i_star]
        krum_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            krum_median = __krum(grads, self._s)
            self._grad_aggregate_buffer[g_idx] = krum_median
        print("Master Step: {} Krum Cost: {:.4f}".format(self.cur_step, time.time()-krum_start))