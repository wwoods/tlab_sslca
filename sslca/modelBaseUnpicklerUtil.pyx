
import numpy as np


def _fixMemoryview(v):
    """Given an object v, convert it and any memory views in it to numpy arrays
    which can be pickled.
    """
    vt = type(v)
    vn = None
    if getattr(vt, '__name__', None) == '_memoryviewslice':
        vn = np.asarray(v)
    elif vt == list:
        vn = [ _fixMemoryview(e) for e in v ]
    elif vt == tuple:
        vn = tuple(_fixMemoryview(e) for e in v)

    if vn is not None:
        return { '__fixedMemoryview__': vn }

    return v


def _unfixMemoryview(v):
    """Basically, due to weird assignment issues with std::vector in c++ cython
    code, we have to re-cast anything that was a memoryview as a memoryview.
    """
    if not (type(v) == dict and len(v) == 1 and '__fixedMemoryview__' in v):
        return v

    v = v['__fixedMemoryview__']
    if isinstance(v, list):
        return [ _unfixMemoryview(e) for e in v ]
    elif isinstance(v, tuple):
        return tuple(_unfixMemoryview(e) for e in v)

    cdef double[::1] dmem1
    cdef double[:, ::1] dmem2

    if v.dtype == np.double:
        if len(v.shape) == 1:
            dmem1 = v
            return dmem1
        elif len(v.shape) == 2:
            dmem2 = v
            return dmem2

    raise NotImplementedError(v)

