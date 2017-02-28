
#cython: embedsignature=True
#cython: profile=True

r"""This module contains utilities for any learners implemented in :mod:`mr.learn`.  New models should inherit from :class:`SklearnModelBase`, following the semantics laid out by that class.
"""

cimport cython
cimport numpy as np

from . import modelBaseUnpickler as mbu

import sklearn.base
import sklearn.metrics

import pickle
import cython
import inspect
import math
import numpy as np
import scipy
import sys
import time

cdef class SklearnModelBaseStats:
    """Class for debug information.  Arrays are per-patch / experiment."""

    def __str__(self):
        return ("SklearnModelBaseStats after {} samples ({} avg activity, "
                + "{} avg sqr activity, {}% used outputs, {} avg power)"
                ).format(
                    self.index,
                    np.asarray(self.activityCount).mean(),
                    np.asarray(self.activitySqr).mean(),
                    np.asarray(self.outputCount).nonzero()[0].shape[0] * 100.
                        / self.outputCount.shape[0],
                    (np.asarray(self.energy)
                        / np.maximum(1e-300, np.asarray(self.simTime))).mean())


cdef class SklearnModelBase:
    """Class that exposes an interface that is consistent with :mod:`sklearn`'s
    interfaces.  Suitable for both supervised and unsupervised learning.

    .. rubric:: Variables to be used by derived classes

    :_bufferIn: An array of shape (nInputs,) for storing intermediate
            input reconstructions.
    :_bufferOut: An array of shape (nOutputs,) for storing intermediate
            outputs.
    :nInputs: The width of each input vector.
    :nOutputs: The expected width out.

    .. rubric:: Requirements for an implementation

    .. autosummary::

        PARAMS
        PICKLE_VARS
        UNSUPERVISED
        _init
        _partial_fit
        _predict
        _reconstruct

    If an algorithm is offline (access to all training data), then the deriving
    class should override fit instead of _partial_fit.  Be sure to call
    self._checkDatasets(X, y), and self.init(). Everything else should be the
    same.

    .. rubric:: Members

    Attributes:
        PARAMS: List of attributes that constitute the parameters of the model,
                irrespective of learned information from :meth:`fit`.  Anything
                specified in this list will work with :meth:`set_params` and
                :meth:`get_params`. Should be set as
                ``(super class).PARAMS + [ 'this', 'class', 'parms' ]``.
        PICKLE_VARS: Variables that AREN'T in PARAMS or __dict that need to be
                saved to preserve this class' state need to go in
                :attr:`PICKLE_VARS`.  Note that if the deriving class is a
                Python class, not a cdef class, no action is needed.
                Otherwise, should be defined as
                ``(super class).PICKLE_VARS + [ 'this', 'class', 'vars' ]``.
        UNSUPERVISED: True if this learner is unsupervised (has no
                problem-specific output).  The LCA or STDP are both examples of
                unsupervised.  If False, this is a supervised learner.
                Perceptron networks using backpropagation are an example of
                supervised learning.

    """

    DEBUG_CONTINUE = "continue"
    r"""A constant for debug kwarg to functions that indicates the debug table
    should NOT be reset, but debugging should occur."""

    # Limitation of Cython + Sphinx: these are documented in class docstring
    PARAMS = [ 'nOutputs' ]
    PICKLE_VARS = [ '_bufferIn', '_bufferOut', 'nInputs', '_isInit', 't_' ]
    UNSUPERVISED = None

    # Pickling support
    @property
    def PICKLE_INIT(self):
        return self.PARAMS
    @property
    def PICKLE_STATE(self):
        return self.PICKLE_VARS
    __reduce_ex__ = mbu.__reduce_ex__
    __setstate__ = mbu.__setstate__

    property nOutputsConvolved:
        """The number of outputs that are convolved; this is used purely for
        visualizing a network."""
        def __get__(self):
            return self.nOutputs


    def __init__(self, **kwargs):
        self._isInit = False
        self._debug = False
        self.nInputs = 0
        self.debugInfo_ = SklearnModelBaseStats()
        self.set_params(**kwargs)


    @classmethod
    def _get_param_names(cls):
        """Stolen from scipy.base.BaseEstimator."""
        return sorted(cls.PARAMS)


    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)

            if deep:
                if hasattr(value, 'get_params'):
                    deep_items = value.get_params().items()
                    out.update((key + '__' + k, val) for k, val in deep_items)
                    value = {'__estimator__': type(value)}
                elif isinstance(value, list):
                    # We have a list as a parameter, see if its members have
                    # get_value
                    valueOut = []
                    for i, v in enumerate(value):
                        if hasattr(v, 'get_params'):
                            deep_items = v.get_params().items()
                            out.update(('{}__{}__{}'.format(key, i, k), sv)
                                    for k, sv in deep_items)
                            valueOut.append({'__estimator__': type(v)})
                        else:
                            valueOut.append(v)
                    value = valueOut
            out[key] = value
        return out


    def set_params(self, **params):
        if not params:
            return self
        valid_params = self.get_params(deep = False)
        for key, value in sorted(params.iteritems()):
            split = key.split('__', 1)
            if len(split) > 1:
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError("{} in {}?".format(name, self))
                sub_object = getattr(self, name)
                if hasattr(sub_object, 'get_params'):
                    sub_object.set_params(**{sub_name:value})
                elif isinstance(sub_object, list):
                    index, value_name = sub_name.split('__', 1)
                    sub_object[int(index)].set_params(**{value_name:value})
                else:
                    raise ValueError("Param type not recognized!")
            else:
                if not key in valid_params:
                    raise ValueError("{} in {}?".format(key, self))

                if (isinstance(value, dict) and len(value) == 1
                        and '__estimator__' in value):
                    value = value['__estimator__']()
                elif isinstance(value, list):
                    valueOut = []
                    for v in value:
                        if (isinstance(v, dict) and len(v) == 1
                                and '__estimator__' in v):
                            valueOut.append(v['__estimator__']())
                        else:
                            valueOut.append(v)
                    value = valueOut
                setattr(self, key, value)
        return self


    def fit(self, X, y=None, minIters=1, maxIters=0,
            solveAbortWithoutProgress=2, solveTime=0.0, debug=False,
            printProgress=True):
        """Initializes this predictor's learned values so that they fit the
        dimensionality of X, and trains the predictor on X.

        minIters [int, default 1] - The minimum number of iterations to train.

        maxIters [int, default 0] - The maximum number of iterations to train
                on.

        solveAbortWithoutProgress [int, default 2] - The number of consecutive
                trials required during which performance has not improved
                before training stops.

                Set to 0 to disable (will run until maxIters or solveTime).

        solveTime - For smaller datasets, it probably makes sense to run fit()
                across the dataset multiple times by default.  solveTime
                specifies, in seconds, the amount of time that an
                UnsupervisedPredictor is allowed to use on trying to better
                output by multiple partial_fits.
        """
        self._checkDatasets(X, y, isInit=True)

        cdef double score, oldScore, tdur, tstart
        self.init(len(X[0]), None if y is None else len(y[0]))

        self._resetDebug(debug, 0)
        if debug:
            # We want to record all fit iterations, so set it to DEBUG_CONTINUE
            # to preserve each fit's records.
            debug = self.DEBUG_CONTINUE

        tstart = time.time()
        score = self.score(X, y)
        if printProgress:
            print("Initial score of {}".format(score))
        self.partial_fit(X, y, debug = debug)
        tdur = time.time() - tstart

        noProgress = 0
        while True:
            # Breaking reasons:
            # 1. Trained enough (max iterations
            # 2. No good progress
            nscore = self.score(X, y)
            # unsupervised is minimization, supervised is maximization
            smod = -1 if self.UNSUPERVISED else 1

            if smod * (nscore - score) / score >= 1e-6:
                # Progress
                noProgress = 0
                score = nscore
            else:
                noProgress += 1

            if self.fitIters_ >= minIters and (
                    # More than requested iters
                    (maxIters > 0 and self.fitIters_ >= maxIters)
                    # Out of time
                    or (solveTime > 0
                        and time.time() + tdur > tstart + solveTime)
                    # No progress on solution
                    or (solveAbortWithoutProgress > 0
                        and noProgress >= solveAbortWithoutProgress)
                    ):
                if printProgress:
                    print("Finished after {} iterations with score of {}"
                            .format(self.fitIters_, nscore))
                break

            if printProgress:
                print("After iteration {}, score of {}".format(self.fitIters_,
                        nscore))
            self.partial_fit(X, y, debug=debug)


    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


    def init(self, nInputs, nOutputs):
        """nInputs and nOutputs comes from data, so init()
        comes after constructor.  Note that nInputs and nOutputs as passed to
        this function do not necessarily override the actual nInputs and
        nOutputs.

        Calls self._init(self.nInputs, self.nOutputs).

        Args:
            nInputs (int): The number of inputs that this model should expect
                    in each data record.
            nOutputs (int): Either None for "the model will fill this in" or
                    the number of expected outputs, assumed to come from data.
        """
        self.nInputs = nInputs
        if self.nOutputs <= 0:
            if nOutputs is not None:
                self.nOutputs = nOutputs
        self._init(self.nInputs, self.nOutputs)
        if self.UNSUPERVISED is None:
            raise ValueError("Class {} must have UNSUPERVISED defined as True "
                    "or False after _init()".format(self.__class__.__name__))
        if self.nOutputs <= 0:
            raise ValueError("self.nOutputs must be set >= 0 in _init if not "
                    "before")
        self._bufferIn = np.zeros(self.nInputs, dtype = float)
        self._bufferOut = np.zeros(self.nOutputs, dtype = float)

        # Learning scalars - final scalar is:
        # 1.0 / pow(t_, power_t)
        self.t_ = 1.0

        self.fitIters_ = 0

        self._isInit = True


    def partial_fit(self, X, y=None, debug = False):
        self._checkDatasets(X, y)

        if not self._isInit:
            # First time initialization
            self.init(len(X[0]), len(y[0]) if y is not None else None)

        self.fitIters_ += 1

        cdef int i, j
        cdef double t
        cdef double[::1] Xi, yi
        cdef double[::1] _bufferOut = self._bufferOut
        cdef double noutInv = 1. / self.nOutputs

        self._resetDebug(debug, len(X))

        for i in range(len(X)):
            Xi, yi = self._makeArrays(X[i], None if y is None else y[i])
            self._partial_fit(Xi, yi)
            self.t_ += 1.

            if self._debug:
                # Assumes self._bufferOut was populated!
                for j in range(self.nOutputs):
                    t = _bufferOut[j]
                    if abs(t) >= 0.01:
                        self.debugInfo_.activityCount[
                                self.debugInfo_.index] += noutInv
                        self.debugInfo_.outputCount[j] += 1
                    self.debugInfo_.activitySqr[self.debugInfo_.index] += t*t
                    self.debugInfo_.outputSqr[j] += t*t
                self.debugInfo_.index += 1


    def predict(self, X, debug = False):
        """Given X, calculate our output values Y (which can be thought of as
        the strengths of different clusters / classification elements)"""
        self._checkDatasets(X, None, True)

        if not self._isInit:
            raise ValueError("{} must be initialized before predict()".format(
                    self))

        cdef double[:, ::1] Y = np.zeros((len(X), self.nOutputs), dtype = float)
        cdef int i, j
        cdef double t
        cdef double noutInv = 1. / self.nOutputs
        cdef double[::1] Xi

        self._resetDebug(debug, len(X))

        for i in range(len(X)):
            Xi, _ = self._makeArrays(X[i], None)
            self._predict(Xi, Y[i])

            if self._debug:
                # Assumes self._bufferOut was populated!
                for j in range(self.nOutputs):
                    t = Y[i, j]
                    if abs(t) >= 0.01:
                        self.debugInfo_.activityCount[
                                self.debugInfo_.index] += noutInv
                        self.debugInfo_.outputCount[j] += 1
                    self.debugInfo_.activitySqr[self.debugInfo_.index] += t*t
                    self.debugInfo_.outputSqr[j] += t*t
                self.debugInfo_.index += 1
        return np.asarray(Y)


    def reconstruct(self, X):
        """Reconstruct each member of X and return the reconstructions"""
        self._checkDatasets(X, None, True)
        if not self._isInit:
            raise ValueError("{} must be initialized before predict()".format(
                    self))
        cdef int i
        cdef double[:, ::1] Y = np.zeros((len(X), self.nInputs), dtype = float)
        for i in range(len(X)):
            self._predict(X[i], self._bufferOut)
            self._reconstruct(self._bufferOut, Y[i])
        return Y


    def reconstructFromPredict(self, y):
        """Reconstruct each member of the original X given a result from
        predict()."""
        self._checkDatasets(y, None, True)
        if not self._isInit:
            raise ValueError("{} must be initialized before predict()".format(
                    self))
        cdef int i
        cdef double[:, ::1] X = np.zeros((len(y), self.nInputs), dtype=float)
        for i in range(len(y)):
            self._reconstruct(y[i], X[i])
        return X


    def score(self, X, y=None, debug = False):
        """Returns the root-mean-squared-error for the distance from the
        reconstruction of X to X, divided by the number of elements in X.
        """
        self._checkDatasets(X, y)

        cdef int i, j
        cdef double[:, ::1] pX
        if self.UNSUPERVISED and y is not None:
            raise ValueError("Cannot give y with UNSUPERVISED")

        if not self.UNSUPERVISED:
            # Return the r2 error
            pX = self.predict(X, debug=debug)
            return sklearn.metrics.r2_score(y, pX)

        # Due to data augmentation, we need to predict and score immediately
        mse = 0.0
        mseDivisor = len(X) * self.nInputs
        cdef double[::1] vX

        for i in range(len(X)):
            vX, _ = self._makeArrays(X[i], None)
            pX = self.predict([vX], debug=debug)
            self._reconstruct(pX[0, :], self._bufferIn)
            for j in range(self.nInputs):
                mse += (self._bufferIn[j] - vX[j]) ** 2
        return (mse / mseDivisor) ** 0.5


    def visualize(self, params, path = None, inputs = None):
        """Dumps an image at path, based on the visual params (width, height,
        channels).  Essentially, gives the input map that is the most likely
        estimator to produce a single output.

        inputs [None] - Inputs suitable for self.predict.  Determines the
                inputs to visualize
        """
        cdef int i, j
        cdef double[::1] outputVals, _bufferIn = self._bufferIn

        w, h, channels = params
        if w * h * channels != self.nInputs:
            raise ValueError("Bad visualization params")

        def split(layerX, layerY, j):
            """returns x, y, c for x coord, y coord, and color channel of
            input j."""
            c = j % channels
            j //= channels
            ox = layerX + (j % w)
            oy = layerY + (j // w)
            return (ox, oy, c)

        imh = h
        imw = w * self.nOutputsConvolved
        outputVals = np.ones(self.nOutputs, dtype = float)
        if inputs is not None:
            imh += h
        imdata = np.zeros((imh, imw, channels), dtype = np.uint8)

        if inputs is not None:
            self._resetDebug(False, 0)
            self._predict(np.asarray(inputs), outputVals)
            # Print the reconstruction
            self._reconstruct(outputVals, _bufferIn)
            for j in range(self.nInputs):
                ox, oy, c = split(w, h, j)
                imdata[oy, ox, c] = int(
                        255 * max(0.0, min(1.0, _bufferIn[j])))

        for i in range(self.nOutputsConvolved):
            self._bufferOut[:] = 0
            for xi in range(i, self.nOutputs, self.nOutputsConvolved):
                self._bufferOut[xi] = 1.0 * outputVals[xi]
                if inputs is None:
                    break
            self._reconstruct(self._bufferOut, self._bufferIn)
            for j in range(self.nInputs):
                ox, oy, c = split(i * w, 0, j)
                imdata[oy, ox, c] = int(255 * max(0.0, min(1.0,
                        self._bufferIn[j])))

        if inputs is not None:
            # Print the inputs
            for j in range(self.nInputs):
                ox, oy, c = split(0, h, j)
                imdata[oy, ox, c] = int(255 * max(0.0, min(1.0, inputs[j])))

        if channels == 1:
            imdata = imdata[:, :, 0]
        scipy.misc.imsave(path, imdata)


    def _checkDatasets(self, X, y, noYIsOk = False, isInit=False):
        """Checks input / output arrays X and y to make sure they have two
        dimensions and can be indexed via [].

        Raises a ValueError if the parameter is invalid.

        :param isInit: If True, then self.nInputs and self.nOutputs will be set
                after this method is called, and those checks will be skipped.
        """
        if not noYIsOk:
            if self.UNSUPERVISED and y is not None:
                raise ValueError("Unsupervised, cannot have y!")
            elif not self.UNSUPERVISED and y is None:
                raise ValueError("Supervised, must have y!")
        elif y is not None:
            raise ValueError("When noYIsOk, there should be no y!")

        if X is None:
            raise ValueError("X must be specified")

        for name, obj in [ ("X", X), ("y", y) ]:
            if obj is None:
                # None checking is done above
                continue

            for attr in [ '__len__', '__getitem__' ]:
                if not hasattr(obj, attr):
                    raise ValueError("{} does not support {}}".format(name,
                            attr))
            if not hasattr(obj[0], '__len__'):
                raise ValueError("{} does not look like a 2-d array".format(
                        name))

        if not isInit:
            if len(X[0]) != self.nInputs:
                raise ValueError("Wrong input dimension: {}, not {}".format(
                        len(X[0]), self.nInputs))


    cdef _makeArrays(self, x, y):
        cdef double[::1] xi, yi = None
        xi = np.asarray(x, dtype=float)
        if y is not None:
            yi = np.asarray(y, dtype=float)
        return xi, yi


    cpdef _init(self, int nInputs, int nOutputs):
        """Initializes this online algorithm for ``nInputs`` inputs and
        ``nOutputs`` outputs.

        May set self.nOutputs if needed, in which case it should ignore
        nOutputs.

        If :attr:`UNSUPERVISED` is not set at the class level, it MUST be set
        after this function is called.

        Args:
            nInputs (int): The number of inputs that this layer should expect.
            nOutputs (int): The number of outputs that this layer should
                    output, or 0 to indicate that this layer should
                    self-populate the number of outputs (self.nOutputs).
        """
        raise NotImplementedError("_init() in {}".format(self.__class__))


    cpdef _partial_fit(self, double[::1] x, double[::1] y):
        r"""Aggregates new data ``x`` of shape ``self.nInputs`` into the
        learned parameters.  ``x`` and ``y`` have already been type checked.
        ``y`` will always be ``None`` if :attr:`UNSUPERVISED` is ``True``.

        Args:
            x: The input vector, of size :attr:`nInputs`.
            y: If this layer is unsupervised, ``None``.  Otherwise, the output
                    vector, of size :attr:`nOutputs`.

        Returns:
            None
        """
        raise NotImplementedError("_partial_fit() in {}".format(self.__class__))


    cpdef _predict(self, double[::1] x, double[::1] y):
        r"""Fills ``y`` with the outputs for ``x``.  That is, runs this
        algorithm.

        Args:
            x: The input vector, of size :attr:`nInputs`.
            y: Where the result should be stored.

        Returns:
            None: The result is stored in ``y``.
        """
        raise NotImplementedError("_predict() in {}".format(self.__class__))


    cpdef _reconstruct(self, double[::1] y, double[::1] r):
        r"""Fills ``r`` (which is an np.array of dtype float with length
        :attr:`nInputs`) with the most likely reconstruction of inputs for
        given output vector ``y``.

        Args:
            y: Previous result from :meth:`_predict`.
            r: Most likely set of inputs that would lead to ``y``.

        Returns:
            None: Result is stored in ``r``.
        """
        raise NotImplementedError("_reconstruct() in {}".format(self.__class__))


    cpdef _resetDebug(self, debug, int lenX):
        """We're about to do some operations on input array X.  Set up the
        debugInfo_ struct if self.debug is set.
        """
        cdef double[::1] new, tmp

        if not debug:
            self._debug = False
            self.debugInfo_.index = -1
            return

        self._debug = True
        s1 = [ 'activityCount', 'activitySqr', 'energy', 'simTime' ]
        if debug != self.DEBUG_CONTINUE:
            self.debugInfo_.index = 0
            for aname in s1:
                setattr(self.debugInfo_, aname, np.zeros(lenX))

            # Reset output stuff
            self.debugInfo_.outputCount = np.zeros(self.nOutputs)
            self.debugInfo_.outputSqr = np.zeros(self.nOutputs)
        else:
            tmp = self.debugInfo_.activity
            for aname in s1:
                tmp = getattr(self.debugInfo_, aname)
                new = np.zeros(len(tmp) + lenX)
                new[:len(tmp)] = tmp
                setattr(self.debugInfo_, aname, new)
