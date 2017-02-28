"""This method must be in a python, not cython, file.  Otherwise, the
__safe_for_unpickling__ attribute can't be set."""

from .modelBaseUnpicklerUtil import _fixMemoryview, _unfixMemoryview


def _unpickle(cls, paramsDict):
    return cls(**paramsDict)
_unpickle.__safe_for_unpickling__ = True


def __reduce_ex__(self, protocol):
    """A canned __reduce_ex__ for Cython extension classes.  Uses class list
    variables PICKLE_INIT and PICKLE_STATE to define properties that need to go
    to the constructor (INIT) and properties set afterwards (STATE).

    Must also set __setstate__!
    """
    if not hasattr(self, '__setstate__'):
        raise ValueError("If __reduce_ex__ is used from modelBaseUnpickler, "
                "must also use __setstate__ from modelBaseUnpickler!")

    # Constructor kwargs
    c = {}
    for k in self.PICKLE_INIT:
        c[k] = getattr(self, k, None)

    # Runtime information
    d = {}
    for p in self.PICKLE_STATE:
        d[p] = getattr(self, p, None)
    if hasattr(self, '__dict__'):
        for k, v in self.__dict__.items():
            if k in self.PICKLE_STATE or k in self.PICKLE_INIT:
                continue
            d[k] = v

    # Convert memoryviews to arrays, which can be pickled
    for k, v in d.items():
        d[k] = _fixMemoryview(v)

    return (_unpickle, (self.__class__, c), d)


def __setstate__(self, state):
    for k, v in state.items():
        setattr(self, k, _unfixMemoryview(v))

