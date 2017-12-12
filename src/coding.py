import numpy as np
from math import exp, pi

def search_w(n, s):
    # params: n: number of workers
    # params: s: number of fail workers
    C = _construct_c(n)
    _hat_s = int(2*s+1)
    W = _construct_w(n, _hat_s)
    C_1 = C[:, 0:_hat_s]
    C_2 = C[:, _hat_s:]
    W, fake_W = _cls_solving(C_1, W)
    return W, fake_W, C_2


def _construct_c(n):
    # complex matrix here
    _shape = (n, n)
    C = np.zeros(_shape, dtype=complex)
    for p in range(_shape[0]):
        for q in range(_shape[1]):
            if q>=p:
                if p == 0 or q == 0:
                    C[p, q] = 1+0j
                else:
                    C[p, q]=0+exp(-2*pi*p*q/n)*1j
            else:
                C[p, q] = C[q, p]
    return C


def _construct_w(n, hat_s):
    _shape=(n, n)
    W = np.ones(_shape)
    for i in range(_shape[0]):
        if (i+hat_s) <= _shape[0]:
            _valid_range = range(i,i+hat_s)
        else:
            _valid_range = range(i,n)
            for t in range(i+hat_s-_shape[0]):
                _valid_range.append(t)
        for j in range(_shape[1]):
            if j not in _valid_range:
                W[i, j] = 0
    return W


def _cls_solving(C_1, fake_W):
    # return Q here:
    _shape = np.transpose(C_1).shape
    Q = np.ones(_shape,dtype=complex)
    for i in range(_shape[1]):
        indices = np.where(np.transpose(fake_W)[i]==0)[0]
        _A = np.zeros((len(indices),C_1.shape[1]-1),dtype=complex)
        _b = np.zeros((len(indices),1),dtype=complex)
        for j, index in enumerate(indices):
            _A[j] = C_1[index,1:]
            _b[j] = -C_1[index,0]
        _q=_cls_solver(_A, _b)
        Q[1:,i] = _q.reshape(Q[1:,i].shape)
    W = np.dot(C_1, Q)
    return W, fake_W


def _cls_solver(A, b):
    return np.dot(np.dot(np.linalg.inv(np.dot(_array_getH(A), A)), _array_getH(A)),b)


def _array_getH(ndarray):
    # get conjugate transpose of a np.ndarray
    return ndarray.conj().T


if __name__ == "__main__":
    np.set_printoptions(precision=4,linewidth=200.0)
    W, fake_W, C_2 = search_w(5, 1)
    W_perp = _array_getH(C_2)
    print(np.dot(W_perp, W))