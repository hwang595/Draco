import numpy as np
from scipy.optimize import lsq_linear

def search_w(n, s):
    # params: n: number of workers
    # params: s: number of fail workers
    C = _construct_c(n)
    C = np.dot(1/np.sqrt(n), C)
    _hat_s = int(2*s+1)
    W = _construct_w(n, _hat_s)
    C_1 = C[:, 0:n-_hat_s+1]
    C_2 = C[:, n-_hat_s+1:]
    W, fake_W = _cls_solving(C_1, W)
    W_perp = _array_getH(C_2)
    # prepare matrix S
    s_tmp = np.zeros((1, n-_hat_s+1),dtype=complex)
    s_tmp[0][0] = 1.0+0.0j
    S = np.dot(s_tmp, _array_getH(C_1))
    return W, fake_W, W_perp, S, C_1


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
                    C[p, q]=0+np.exp(-2*np.pi*p*q*1j/n)
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
        _q=lsq_linear(_A, _b.reshape(-1)).x
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
    W, fake_W, W_perp, S = search_w(7, 2)
    print(np.dot(W_perp, W))
    print
    print(S)