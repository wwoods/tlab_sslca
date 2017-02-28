
r"""Optimal learning rates via Zeiler et al. 2012's ADADELTA algorithm.
"""

cimport cython
from libc.math cimport sqrt

from . import modelBaseUnpickler as mbu
import numpy as np

cdef class AdaDelta:
    """ADADELTA optimizer based on Zeiler et al. 2012, with an addition
    for sparsity.
    """

    PICKLE_INIT = [ 'w', 'h', 'rho', 'epsilon', 'sparse_update_threshold',
            'sparse_rho' ]
    PICKLE_STATE = [ '_edX', '_eG2' ]
    __reduce_ex__ = mbu.__reduce_ex__
    __setstate__ = mbu.__setstate__


    def __init__(self, int w, int h, double rho=0.94, double epsilon=1e-6,
            double sparse_update_threshold=0.1, double sparse_rho=1.0):
        """Initializes a 2-d matrix of individually optimized parameters
        controlled by ADADELTA (Zeiler 2012).  A gradient of the overall score
        for each parameter must be estimable.

        rho - Controls the window size for ADADELTA.  Weight for moving average
            stability.  Basically a momentum parameter.

        epsilon - Controls divide-by-zero and keeps learning moving.
        """
        self.w = w
        self.h = h
        self.rho = rho
        self.epsilon = epsilon
        self.sparse_update_threshold = sparse_update_threshold
        self.sparse_rho = sparse_rho
        self._edX = np.zeros((w, h))
        self._eG2 = np.zeros((w, h))


    cpdef convergerProps(self):
        return [ self._edX, self._eG2 ]


    cpdef double getDelta(self, int i, int j, double gradient) except? -1:
        """Given a gradient (increasing this parameter by 1.0 is expected to
        increase the overall score by this much), adjust the parameter in such
        a way that score is minimized.

        Returns the recommended delta for the parameter."""
        cdef double edX = self._edX[i, j]
        cdef double eG2 = self._eG2[i, j]
        cdef double rho = self.rho, urho = 1. - rho, e = self.epsilon
        cdef double srho = self.sparse_rho, \
                sthresh = self.sparse_update_threshold
        cdef double gScalar, delta
        cdef double gradientSqr = gradient*gradient
        # See if the variable is changing significantly enough to count
        if gradientSqr < sthresh * eG2:
            self._edX[i, j] *= srho
            self._eG2[i, j] *= srho
            return 0.

        eG2 = rho * eG2 + urho * gradientSqr
        gScalar = sqrt((edX + e) / (eG2 + e))
        delta = -gScalar * gradient
        self._edX[i, j] = rho * edX + urho * delta * delta
        self._eG2[i, j] = eG2

        return delta
