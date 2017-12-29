from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

#DTYPE = np.complex64_t
ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
def decoding(np.ndarray[np.complex128_t, ndim=2] W_perp, np.ndarray[np.complex128_t, ndim=2]R, int num_workers, int s):
    cdef int _n = num_workers
    cdef int _s = s
    cdef np.ndarray[np.complex128_t, ndim=2] epsilon=_epsilon_searching(W_perp, R, _n, _s)
    return epsilon


@cython.boundscheck(False)
def _epsilon_searching(np.ndarray[np.complex128_t, ndim=2] W_perp, np.ndarray[np.complex128_t, ndim=2] R, int _n, int _s):
    cdef np.ndarray[np.complex128_t, ndim=2] E_2 = np.dot(W_perp, R)
    cdef int _d = E_2.shape[1]
    cdef np.ndarray[np.complex128_t, ndim=2] _X = np.take(E_2, np.array([range(-i-(_s+1), -i-2+1) for i in range(_s)]).reshape(-1), axis=0).reshape((_s, _s*_d), order='F')
    # we use tmp_y as the start point to obtain the full E matrix
    cdef np.ndarray[np.complex128_t, ndim=2] tmp_y = np.take(E_2, np.array([-i-1 for i in range(_s)]), axis=0)
    cdef np.ndarray[np.complex128_t, ndim=1] _y = tmp_y.reshape(-1, order='F')
    cdef np.ndarray[np.complex128_t, ndim=2] alpha = _cls_solver(np.transpose(_X), _y.reshape(_y.shape[0], 1))
    # we want E_1 and E_2 n by d here:
    cdef np.ndarray[np.complex128_t, ndim=2] E_1=_obtain_E1(alpha, tmp_y, _s, _n)
    # concatenate E_1 and E_2 to obtain E
    cdef np.ndarray[np.complex128_t, ndim=2] E = np.concatenate((E_1, E_2), axis=0)
    # obtain epsilon by taking IFT of E:
    cdef np.ndarray[np.complex128_t, ndim=2] epsilon = _obtain_epsilon(E)
    return epsilon


@cython.boundscheck(False)
def _obtain_E1(np.ndarray[np.complex128_t, ndim=2] alpha, np.ndarray[np.complex128_t, ndim=2] y, int s, int n):
    # obtain E_1 in shape of n-2s by d
    cdef np.ndarray[np.complex128_t, ndim=2] _processing_y = np.transpose(y)
    for i in range(n-2*s):
        tmp = np.dot(_processing_y[:,-s:], alpha)
        _processing_y = np.concatenate((_processing_y, tmp), axis=1)
    return np.transpose(_processing_y[:, s:])


@cython.boundscheck(False)
def _obtain_epsilon(np.ndarray[np.complex128_t, ndim=2] E):
    cdef np.ndarray[np.complex128_t, ndim=2] ret = np.fft.ifft(a=E, axis=1)
    return ret


@cython.boundscheck(False)
def _cls_solver(np.ndarray[np.complex128_t, ndim=2] A, np.ndarray[np.complex128_t, ndim=2] b):
    cdef np.ndarray[np.complex128_t, ndim=2] ret = np.dot(np.dot(np.linalg.inv(np.dot(_array_getH(A), A)), _array_getH(A)),b)
    return ret


@cython.boundscheck(False)
def _array_getH(np.ndarray[np.complex128_t, ndim=2] ndarray):
    # get conjugate transpose of a np.ndarray
    cdef np.ndarray[np.complex128_t, ndim=2] ret = ndarray.conj().T
    return ret