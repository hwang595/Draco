from .utils import *
from .baseline_master import SyncReplicasMaster_NN

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
        err_indices = [i for i in range(self.s, self.num_workers)]

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