import numpy as np

ADVERSARY_=-100
CONST_ = -10

def grad_simulation(grad, mode):
	if mode == "rev_grad":
		return ADVERSARY_*grad
	elif mode == "constant":
		return np.ones(grad.shape, dtype=np.float64)*CONST_
	elif mode == "random":
		# TODO(hwang): figure out if this if necessary
		return grad