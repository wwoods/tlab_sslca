cdef class FastRandom:
    cpdef double get(self) except? -1.
    cpdef int geti(self, int) except? -1

    cdef public double[:] _randBuffer
    """Up-next random values"""
    cdef public int _randIndex
    """The current index into :attr:`_randBuffer`"""
    cdef public int _randLen
    """The length of :attr:`_randBuffer`"""
    cdef public object _randState
    """The np.random.RandomState object responsible for yielding more random
    numbers."""

