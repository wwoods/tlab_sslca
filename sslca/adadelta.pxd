
cdef class AdaDelta:
    cdef public int w, h
    cdef public double rho, initial, epsilon, sparse_update_threshold, \
            sparse_rho
    cdef public double[:, ::1] _edX, _eG2

    cpdef convergerProps(self)
    cpdef double getDelta(self, int i, int j, double gradient) except? -1

