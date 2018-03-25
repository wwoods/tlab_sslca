
# cython: profile=True
# notcython: linetrace=True, binding=True

r""":mod:`lcaSpikingWoodsAnalytical` provides an analytical (non-SPICE) implementation of the simple spiking architecture.

Most up-to-date class:

.. autosummary::
    :toctree: _autosummary

    LcaSpikingWoodsAnalyticalInhibition

.. image:: /../mr/learn/unsupervised/docExtra/lcaWoodsUninhibited.svg

The *Simple, Spiking, Locally Competitive Algorithm* (SSLCA) consists of rows with a set voltage (either :math:`V_{high}` or :math:`V_{low}` depending on input spike activity) and columns that contain a capacitor which is directly connected to the crossbar.

The design can be broken down into two versions:

* :ref:`sec-non-inhibit`
* :ref:`sec-inhibit`

.. _sec-non-inhibit:

Non-Inhibiting Version
======================

When inhibition is not a factor, the behavior is exactly as described above: a capacitor directly on the crossbar either charges or discharges through the rows.  When that capacitor reaches a threshold voltage, all capacitors are reset, and the process continues.  Mathematically, this can be viewed as the sum of each row's current into the node:

.. math::

    C\frac{\partial V_{neuron}}{\partial t} &= \sum_i{(V_i - V_{neuron})G_i}.

Then, assuming an input row :math:`i` spikes to voltage :math:`V_{set}` with a mean of :math:`K_i` activity (on for :math:`K_i`, off for :math:`1 - K_i`), and is grounded the rest of the time, this becomes:

.. math::

    C\frac{\partial V_{neuron}}{\partial t} &= \sum_i \biggl( K_i(V_{set} - V_{neuron})G_i\\
            &\hspace{5em}+ (1 - K_i)(0 - V_{neuron})G_i \biggr),\\
    &= \sum_i{(K_iV_{set} - V_{neuron})G_i}, \\
    &= V_{set}\sum_i{K_iG_i} - V_{neuron}\sum_i{G_i}, \\
    Q_1 &= \sum_i{G_i}, \\
    Q_2 &= V_{set}\sum_i{K_iG_i}, \\
    \mathcal{L}\{V_{neuron}\}s(Cs + Q_1) &= CsV_{neuron, t=0} + Q_2, \\
    V_{neuron}(t) &= \frac{Q_2}{Q_1}(1 - e^\frac{-tQ_1}{C}) + V_{neuron, t=0} e^\frac{-tQ_1}{C}.

Solving that equation for :math:`t`, letting :math:`G_i = G_{max}g_i, K_i = K_{max}k_i`, assuming the network has :math:`M` inputs, and that there is a distribution :math:`\chi` that matches the empirical data set that the network is exposed to, then the network may be specified by assuming that the network is fully trained with a matching element for each input:

.. math::

    Q_1 &= MG_{max}\overline{\chi}, \\
    Q_2 &= MV_{set}K_{max}G_{max}\overline{\chi^2}, \\
    t &= \frac{-C}{Q_1}ln\left(\frac{\frac{Q_2}{Q_1} - V_{neuron}(t)}{\frac{Q_2}{Q_1} - V_{neuron,t=0}}\right).

Thus, the anticipated fire time (desired spikes per second) can be calibrated based on the desired capacitance and threshold voltage.  Alternatively, by using the product of two different distributions instead of the square of a single for :math:`Q2`, :math:`\overline{\chi\chi}`, a ratio of firing times may be calibrated based on :math:`V_{neuron}`.

.. note:: Fire Voltage Versus Network Size

    Since the storage devices we are using have limited range, it is ideal that the output patches consist of maximally
    conductive elements.

    If the network's size is :math:`M`, and :math:`P` of the outputs are active for a given receptive field, then:

    .. math::

        Q_1 &= M(PG_{max} + (1-P)G_{min}), \\
        Q_2 &= MV_{set}PK_{max}G_{max}, \\
        V_{fire} &= (1 - e^{-1})\frac{V_{set}PK_{max}G_{max}}{P(G_{max} - G_{min}) + G_{min}}.

    .. todo:: Changed to calc fire time from and rInhib_ from k=0.3...

    The take-away from this is that scaling the network size does not affect the firing threshold, but if the desired number of input events remains constant, then :math:`P` will decrease with a larger :math:`M`, leading to a lower firing threshold.  This might be more difficult to implement in hardware.


.. _sec-inhibit:

Inhibiting Version
==================

The inhibiting version has an identical layout to the non-inhibiting version.  However, it has more complicated row and column headers.  The CMOS layout is as follows:

.. image:: /../mr/learn/unsupervised/docExtra/lcaWoodsInhibited.svg

If, in the *Inhibition Logic Module* (ILM), :math:`CBLOW = \overline{SPIKE}`, then the non-inhibiting architecture is realized.  However, if the ILM is implemented as above, then:

Derivation
----------

.. note:: Later derivation shows that the below math is only useful through :math:`V_{i,0} = ...`.

.. math::

    \text{As with uninhibited, a row has $K_i$ spike activity.} \span\\
    \text{Tracking the inhibition voltage $V_{i}$ (pre/post is relative to spike):} \span\\
    A &= \frac{1}{R_{cb}C}, \\
    B &= \frac{1}{RC}, \\
    V_{i,max} &= V_{cc}\frac{R}{R + R_{cb}}, \\
    V_{i,pre} &= V_{i,0} e^{-TB}, \\
    V_{i,post} &= V_{i,max} + (V_{i,pre} - V_{i,max})e^{-T_{spike}A}. \\
    \text{Making the assumption that spike activity is nullified when:} \span\\
    V_i &> V_{i,thresh}, \\
    \text{Then the time that spike activity is nullified is:} \span\\
    V_{i,thresh} &= V_{i,0} e^{-T_{run}B}, \\
    T_{inhib} &= \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B}\text{, bounded on $[0, \infty)$}. \\
    \text{Since $V_{i,post}$ can be calculated, the effective block time added by } \span\\
    \text{a spiking event can be calculated by taking the difference of $T$ values.} \span\\
    \text{If we want constant inhibition, then $V_{post}$ after $V_{pre}$ needs to be $V_{i,0}$:} \span\\
    V_{i,0} &= V_{i,max} + (V_{i,0}e^{-T_{run}B} - V_{i,max})e^{-T_{spike}A}, \\
    T_{run} &= \frac{-ln\left( \frac{V_{i,0} - V_{i,max} + V_{i,max}e^{-T_{spike}A}}{V_{i,0}e^{-T_{spike}A}} \right)}{B}. \\
    \text{Back to the $K_i$ business, an inhibited network will have $K_i = 0$ for $T < T_{inhib}$.} \span\\
    \text{In a $1\times 1$ network, this means that the fire time will be delayed by $T_{inhib}$:} \span\\
    T_{run} &= \frac{-C}{Q_1}ln\left( 1 - V_{neuron}(t)\frac{Q_1}{Q_2} \right) + max\left(0, \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B}\right), \\
    \text{Subsituting in the original design parameters: } \span\\
    T_{run} &= T_{planned} + max\left( 0, \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B} \right). \\
    \text{$T_{run}$ is now the actual time between spikes; the frequency is thus $\frac{1}{T_{run} + T_{spike}}$.} \span\\
    \text{Substituting the $V_{i,0}$ stability equation into the $1\times 1$ equation yields: }\span\\
    \frac{V_{i,0} - V_{i,max} + V_{i,max}e^{-T_{spike}A}}{V_{i,0}e^{-T_{spike}A}} &= e^{-B\left[ ln(e^{T_{planned}}) + min\left(0, ln\left( \frac{V_{i,thresh}}{V_{i,0}} \right) \right) \right]} \\
            &= \begin{cases}
                e^{-BT_{planned}} & \text{if } V_{i,thresh} \ge V_{i,0} \\
                \left(\frac{V_{i,thresh}}{V_{i,0}}\right)^{-B}e^{-BT_{planned}} & \text{otherwise}
                \end{cases}


.. note:: **This is old, and probably bad**

    Consider now a niche problem for investigating this architecture:

    * There are two dictionary elements with M inputs
    * The first element has M-1 elements at max weight (other at K_{min})
    * The second element has only the last element at K_B weight (others at K_{min})
    * The input is a solid bar of K_{in} weight

    Thus the ratio of firing should work out to 1:K/B between the two elements if inhibition is doing its job.  Essentially, we will run the analytical algorithm to determine the actual firing ratio.

    .. math::
        :label: eq-inhib-cap

        \text{Column parameters denoted as $A\{Q_1\}$ and $B\{Q_1\}$, for instance.} \span\\
        A\{Q_1\} &= G_{max}(M - 1 + K_{min}), \\
        A\{Q_2\} &= V_{set}K_{max}G_{max}K_{in}(M - 1 + K_{min}), \\
        B\{Q_1\} &= G_{max}((M - 1)K_{min} + K_B), \\
        B\{Q_2\} &= V_{set}K_{max}G_{max}K_{in}((M - 1)K_{min} + K_B), \\
        \text{For sensitivity, $V_{fire} = (1 - e^{-1})\frac{B\{Q_1\}}{B\{Q_2\}}$}, \span\\
        Q_1 &= MG_{max}K_{in}, \\
        Q_2 &= MV_{set}K_{max}G_{max}K_{in}^2, \\
        C &= \frac{-T_{planned}MG_{max}K_{in}}{ln\left(1 - V_{fire}\frac{Q_1}{Q_2}\right)}. \\
        \text{We have two distinct input inhibition states: $V_{i,A}$ and $V_{i,B}$.} \span\\
        \text{At each step, re-compute all $Q_2$ according to inhibition terms.} \span\\
        \text{Measure times: $T_{inhib,A}$, $T_{inhib,B}$, $t_{fire,A}$, $t_{fire,B}$.} \span\\
        \text{Take the smallest time, re-up $V_{i,A}$, $V_{i,B}$, $V_A$, $V_B$.} \span\\
        \text{Rinse and repeat to 20 spike events.} \span

Now, inhibition works best when it's linear: the charge through the crossbar is exponential between the current inhibition and VCC.  The drain is from a max value of VCC down to ground.  Therefore, the optimal inhibition threshold is :math:`\frac{Vcc}{2}`, as the midpoint is the most linear range of an exponential for both towards VCC and towards ground.

The most important design decisions for inhibition are R and C.  Too large of an R produces good results, but makes the network run for far too long.  Too large of a C can cause performance issues as well.  The goal is to balance these for accuracy against speed; it is imperative that the network still process quickly.

Thus, design for uninhibited network:

.. note::

    Actually:

    #. Choose the minimal RF that will be stored; it is ideal to use as much of the device's range as possible, so calculate this RF by choosing a value :math:`g_{avg}` between the minimum conductance, :math:`g_{min}`, and the maximum, :math:`1`.  Next, select a minimum input stimulus strength that will trigger a firing event on this average conductance; call it :math:`g_{input}`.  Then:

        .. math::

            Q_1 &= MG_{max}g_{avg}, \\
            M_{high} &= \frac{g_{avg} - g_{min}}{1 - g_{min}}, \\
            M_{low} &= 1 - M_{high}, \\
            Q_2 &= V_{cc}MG_{max}K_{max}\left( M_{high} + M_{low}g_{min}^2 \right) \frac{g_{input}}{g_{avg}}.

    #. The fire voltage :math:`V_{fire}` should be based upon the above :math:`Q_1\and Q_2`.
    #. To calculate :math:`C`, use the above process but with :math:`g_{input} = g_{avg}`.

#. Choose the minimal RF of interest, calculate :math:`Q_{1L}\text{ and }Q_{2L}`.
#. Choose the average RF of interest, calculate :math:`Q_{1M}\text{ and }Q_{2M}`.
#. Calculate :math:`V_{fire}` based on :math:`Q_{1L}\and Q_{2L}`.
#. Choose the desired run-time length of algorithm, spike density, and spike resolution.
#. Calculate :math:`C` based on these parameters and :math:`Q_{1M}\and Q_{2M}`.

For an inhibited network (implemented as :meth:`LcaSpikingWoodsAnalyticalInhibition._init`):

#. Choose the minimal RF of interest, calculate :math:`Q_{1L}\text{ and }Q_{2L}`.
#. Choose the average RF of interest, calculate :math:`Q_{1M}\text{ and }Q_{2M}`.
#. Calculate :math:`V_{fire}` based on :math:`Q_{1L}\and Q_{2L}` (:math:`V_{fire} = \alpha\frac{Q_{2L}}{Q_{1L}}`, where :math:`\alpha` is a scaling factor.  Currently :math:`1 - e^{-1}`).
#. Choose the desired input spike maximum density, input spike period, output spike maximum density, and output spike period; :math:`K_{maxIn}, T_{spikePeriodIn}, K_{maxOut}, \and T_{spikePeriodOut}`, respectively.
#. Calculate :math:`C_{max}`, the maximum neuron capacitance (not accounting for inhibition) based on :math:`Q_{1M}\and Q_{2M}` and the time between output spikes :math:`T_{spikePeriodOut}(1 - K_{maxOut})`.  This will use :eq:`eq-inhib-cap`.
#. Modify that :math:`C_{max}` to make room for inhibition effects, and manipulate :math:`R_{inhib}\and C_{inhib}` such that the average RF will fire at the correct interval.  This is accomplished as follows:
    #. Want asymptotic, stable fire time; the reconstruction is accurate for an input :math:`i` when :math:`F(g_i, k_i)`, the time it takes for an output neuron to charge and begin to fire, is balanced with the inhibitory forces created during the firing event:

        .. math::

            F(g_i, k_i) &= T_{spikeGapOut} = \frac{g_iT_{spikePeriodOut}}{k_i} - T_{spikeOut}, \\
            T_{spikeOut} &= T_{spikePeriodOut}K_{maxOut}, \\
            T_{spikeGapOutInhib} &= \text{Portion of $T_{spikeGapOut}$ where inhibition is in play}, \\
                    &= T_{spikeGapOut} - T_{fire}, \\
            T_{fire} &= \text{From uninhibited version:} \\
                    &= \frac{-C}{Q_{1M}}ln\left( \frac{\frac{Q_{2M}}{Q_{1M}} - V_{fire}}{\frac{Q_{2M}}{Q_{1M}} - 0} \right).

        From :eq:`eq-inhib-cap`, can solve for :math:`V_{i,0}` with :math:`V_{i,max} = V_{cc}`:

        .. math::

            K_i &= K_{maxIn}k_i, \\
            A &= \frac{1}{R_{cb}C_{inhib}}, \\
            R_{cb} &= \text{Memristive device resistance}, \\
            B &= \frac{1}{R_{inhib}C_{inhib}}, \\
            V_{i,0} &= \frac{V_{cc}\left( 1 - e^{-T_{spikeOut}A} \right)}{1 - e^{-K_iT_{spikeOutGap}B - T_{spikeOut}A}}, \\
            V_{i,thresh} &= V_{i,0}e^{-K_iT_{spikeGapOutInhib}B}, \\
            V_{i,thresh} &= \frac{V_{cc}}{2} \text{(to try to stay linear)}, \\
            V_{i,0} &= \frac{V_{cc}}{2e^{-K_iT_{spikeGapOutInhib}B}}, \\
            \frac{V_{cc}}{2e^{-K_iT_{spikeGapOutInhib}B}}
                    &= \frac{V_{cc}\left( 1 - e^{-T_{spikeOut}A} \right)}{1 - e^{-K_iT_{spikeGapOut}B - T_{spikeOut}A}}.

        Since all parameters except for :math:`R_{inhib}\and C_{inhib}` are known, the above non-linear equation can be used to solve for whichever of the two is unspecified.  Empirical testing has shown that since fire time and neuron capacitance have a linear relationship, halving the neuron's capacitance and using the same value for :math:`C_{inhib}` is reasonable, though there might be a better approach.

.. _sec-power:

Power Considerations
====================

Discharging a capacitor by :math:`V` volts dissipates :math:`\frac{1}{2}CV^2` joules.

Charging through an RC circuit (should mirror above equation):

.. math::

    \text{Voltage is $V_{supply} - V_{cap}$}, \span\\
    V(t) &= Ve^{\frac{-t}{RC}}, \\
    \frac{V^2(t)}{R} &= \frac{V^2}{R}e^{\frac{-2t}{RC}}, \\
    \int \frac{V^2(t)}{R} dt &= \frac{-CV^2}{2}e^{\frac{-2t}{RC}}, \\
    \int_0^t \frac{V^2(t)}{R} dt &= \frac{CV^2}{2}\left[ 1 - e^{\frac{-2t}{RC}} \right].

Great.  Now, need to compute the power through each junction in the memristive crossbar:

.. math::

    \text{One side is the input, $V_{in}$, other is $V_C(t)$}, \span\\
    \text{we have $V_C(t)$, defined as}: \span\\
    V_C(t) &= \frac{Q_2}{Q_1} + (V_C(0) - \frac{Q_2}{Q_1})e^{\frac{-tQ_1}{C}}, \\
    Z_1 &= \frac{Q_2}{Q_1}, \\
    Z_2 &= \frac{-Q_1}{C}, \\
    (V_C(t) - V_{in})^2 &= \left[ Z_1 - V_{in} + (V_C(0) - Z_1)e^{tZ_2} \right]^2 \\
            &= Z_1^2 - 2Z_1V_{in} + V_{in}^2 + 2e^{tZ_2}(Z_1V_C(0) - Z_1^2 - V_{in}V_C(0) + V_{in}Z_1)
                    + e^{2tZ_2}(V_C(0)^2 - 2V_C(0)Z_1 + Z_1^2), \\
    \int (V_C(t) - V_{in})^2 dt &= \left[
            t\left(Z_1^2 - 2Z_1V_{in} + V_{in}^2\right)
            + \frac{2}{Z_2}e^{tZ_2}\left(Z_1V_C(0) - Z_1^2 - V_{in}V_C(0) + V_{in}Z_1\right)
            + \frac{1}{2Z_2}e^{2tZ_2}\left(V_C(0)^2 - 2V_C(0)Z_1 + Z_1^2\right) \right], \\
    \int_0^t (V_C(t) - V_{in})^2 dt &= \left[
            \splitfrac{
                t\left(Z_1^2 - 2Z_1V_{in} + V_{in}^2\right)
                + \frac{2}{z_2}e^{tZ_2}\left(Z_1V_C(0) - Z_1^2 - V_{in}V_C(0) + V_{in}Z_1\right)
                + \frac{1}{2Z_2}e^{2tZ_2}\left(V_C(0)^2 - 2V_C(0)Z_1 + Z_1^2\right)
                }{
                - \frac{1}{Z_2}\left( Z_1V_C(0) - 1.5Z_1^2 - 2V_{in}V_C(0) + 2V_{in}Z_1 + 0.5V_C(0)^2 \right)
                }
            \right], \\
    E &= \frac{1}{R}\int_0^t (V_C(t) - V_{in})^2 dt.

"""

cimport cython
cimport numpy as np

cdef extern from "math.h":
    double exp(double)
    double log(double)
    double pow(double, double)
    double sqrt(double)

from .adadelta cimport AdaDelta
from .modelBase cimport SklearnModelBase
from .util cimport FastRandom

import collections
import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import scipy
import scipy.optimize
import sklearn
import sys

cdef class LcaSpikingWoodsAnalyticalInhibition(SklearnModelBase):
    """See the module documentation for the design of this class.  Takes two
    receptive fields, one the "most receptive" and another the "least
    receptive" and generates a network that does sparse approximation
    correctly.

    Example usage:

    .. code-block:: python

        physics = dict(phys_rMax=183e3, phys_rMin=53e3)
        lca = LcaSpikingWoodsAnalyticalInhibition(
                10, rfAvg=0.1,
                **physics)

    rfAvg is the target average value of a receptive field, and must be greater
    than ``phys_rMin / phys_rMax``.

    For an uninhibited implementation using this class, use
    :attr:`noInhibition`.

    To debug spike patterns, use :attr:`debugSpikesTo`.
    """

    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [
            'algSpikes',
            'algTInSpike',
            'algTInMinGap',
            'algTOutAccum',
            'algTOutSpike',
            'homeoRate',
            'inhibCScale',
            'noInhibition',
            'rfAvg', 'rfLeast',
            'phys_rMin',
            'phys_rMax',
            'phys_vcc',
            'trainWithZero',
            'variabilityRead',
            'variabilityWrite',
    ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [
            'ada_',
            '_lcaInternal_crossbar',
            '_homeostasis_train',
            'homeostasis_sinceFire_',
            'vFire_',
            'cNeuron_',
            'cInhib_',
            'rInhib_',

            'debugSpikesTo',
    ]


    ## Design parameters
    cdef public double algSpikes, algTInSpike, algTInMinGap
    cdef public double algTOutAccum, algTOutSpike
    cdef public double homeoRate
    """Set to <= 0 to disable; otherwise, the number of epochs before the
    threshold voltage is halved.  E.g., if homeoRate = 1, and there are 10
    neurons, then every 10 samples, if an output has not spiked, its trigger
    voltage will be 0.5 what it normally would be.
    """
    cdef public double inhibCScale
    cdef public bint noInhibition
    """If set to ``True``, disables the simulation of inhibition.  In other
    words, the non-inhibited network will be simulated.
    """
    cdef public double rfAvg, rfLeast

    cdef public bint trainWithZero
    """Set to True to train with zero being an option, otherwise uses the
    minimum conductance (g_min).  Either way, the enforced limit is g_min when
    running the network.  Incompatible with variabilityWrite.
    """

    ## Physics parameters
    cdef public double phys_rMin, phys_rMax, phys_vcc
    cdef public double variabilityRead, variabilityWrite;

    ## Model values initialized by :meth:`_init`
    cdef public AdaDelta ada_
    cdef public double[:, ::1] _lcaInternal_crossbar
    property crossbar_:
        """Returns or sets the crossbar used for sparse representation.
        """
        def __get__(self):
            return self._lcaInternal_crossbar
        def __set__(self, double[:, ::1] crossbar):
            cbShape = (crossbar.shape[0], crossbar.shape[1])
            if crossbar.shape[0] != self.nInputs:
                raise ValueError("Bad shape: {}, not {}".format(cbShape,
                        (self.nInputs, self.nOutputs)))
            if crossbar.shape[1] != self.nOutputs:
                raise ValueError("Bad shape: {}, not {}".format(cbShape,
                        (self.nInputs, self.nOutputs)))
            self._lcaInternal_crossbar = crossbar
    cdef public bint _homeostasis_train
    """True if _predict is happening during training; False otherwise."""
    cdef public double[:] homeostasis_sinceFire_
    """The time (s) since each neuron last fired during training runs."""
    cdef public double vFire_
    """Fire threshold (V)"""
    cdef public double cNeuron_
    """Capacitance of neuron (F)"""
    cdef public double cInhib_
    """Capacitance of inhibition row (F)"""
    cdef public double rInhib_
    """Resistance of RC circuit for inhibition (Ohms)"""

    ## Debug stuff
    cdef public tuple debugSpikesTo
    """Leave as None to disable.  Otherwise, when ``debug=True`` is
    passed to :meth:`predict`, this tuple will consist of:

        visualParams: The visual parameters to use for rendering the images.
        path: A string that will be formatted with the input number and an
                image detailing which spike patterns were seen will be
                dumped to this path.
    """


    def __repr__(self):
        if not self._isInit:
            return "{}<uninitialized>()".format(self.__class__.__name__)

        return "{}<vFire_={}, cNeuron_={}, cInhib_={}, rInhib_={}>()".format(
                self.__class__.__name__, self.vFire_, self.cNeuron_,
                self.cInhib_, self.rInhib_)


    ## Temporaries that are not saved
    cdef public FastRandom _rand


    def __init__(self, nOutputs=10, **kwargs):
        defaults = {
                'nOutputs': nOutputs,
                'algSpikes': 10.,
                'algTInSpike': 0.4e-9,
                'algTInMinGap': 0.4e-9,  # A bit of a misnomer; the average gap
                                         # between spikes for an input of 1.0
                'algTOutAccum': 0.8e-9,
                'algTOutSpike': 0.2e-9,
                'homeoRate': 5.,  # Set to <= 0 to disable homeostasis
                'inhibCScale': 0.5,
                'noInhibition': False,
                'rfLeast': -1,  # -1 means auto guess based on rfAvg
                'rfAvg': -1,  # defaults to 0.5 * (1. + phys_rMin / phys_rMax)
                'trainWithZero': True,

                'phys_rMin': 52e3,  # Memristor Panic Woods et al., at 0.7V
                'phys_rMax': 207e3,
                'phys_vcc': 0.7,
                'variabilityRead': 0.,
                'variabilityWrite': 0.,
        }
        defaults.update(**kwargs)
        super().__init__(**defaults)

        self.debugSpikesTo = None
        self._rand = FastRandom()


    cpdef variabilityWriteShakeup(self, double amt):
        """Initiates a one-off crossbar rewrite that jiggers conductances."""
        cdef FastRandom rand = self._rand
        cdef double[:, :] cb = self.crossbar_
        cdef int i, j
        for i in range(cb.shape[0]):
            for j in range(cb.shape[1]):
                cb[i, j] *= (
                        (1. - amt)
                        + rand.get() * (2 * amt))
                # Don't allow negative conductances, that's silly
                cb[i, j] = max(0., cb[i, j])


    cdef _debugSpikesBlit(self, np.uint8_t[:, :, :] imgData, int y, int x,
            double[:] buffer, bint rescale=True):
        """Utility function for :attr:`debugSpikesTo`` used to blit from a
        buffer onto an image.
        """
        cdef int i, j, z, q
        cdef int w, h, c
        cdef double s = 255.

        w, h, c = self.debugSpikesTo[0]
        if w*h*c != buffer.shape[0]:
            raise ValueError("Expected buffer of size {}, got {}".format(
                    w*h*c, buffer.shape[0]))

        if rescale:
            s = -1e300
            for i in range(w*h*c):
                s = max(s, buffer[i])
            s = 255. / max(1e-300, abs(s))

        q = 0
        x *= w
        y *= h
        for j in range(h):
            for i in range(w):
                for z in range(c):
                    imgData[y+j, i+x, z] = max(0, min(255, int(s*buffer[q])))
                    q += 1


    cpdef _init(self, int nInputs, int nOutputs):
        """Responsible for setting up a new instance of this network.  This
        function performs the design steps for an inhibited network mentioned
        in :ref:`sec-inhibit`.
        """
        cdef int i, j

        self.ada_ = AdaDelta(nInputs, nOutputs)
        self._lcaInternal_crossbar = np.random.uniform(size=(nInputs, nOutputs))
        self._homeostasis_train = False
        self.homeostasis_sinceFire_ = np.zeros(nOutputs)

        ## Parameter auto-detection!
        cdef double Q1M = 0., Q2M = 0., Q1L = 0., Q2L = 0., Q, T
        cdef double rfAvg = self.rfAvg
        cdef double rfLeast = self.rfLeast
        cdef double gMin = self.phys_rMin / self.phys_rMax

        if rfAvg < 0:
            rfAvg = 0.5 * (1. + gMin)
        elif rfAvg <= gMin:
            raise ValueError("rfAvg must be greater than device minimum "
                    "conductance ratio: {}".format(gMin))

        if rfLeast < 0:
            # From empirical evidence, this works well.  Try a 2-d sweep of
            # rfAvg vs rfLeast if you don't believe me.
            rfLeast = rfAvg * (1. - np.exp(-1))
        elif rfLeast == 0:
            raise ValueError("rfLeast cannot be zero")

        # vFire is determined off the least receptive field's Q ratio
        Q1L, Q2L = self._init_getQ(rfAvg, rfLeast)
        Q1M, Q2M = self._init_getQ(rfAvg, rfAvg)
        Q = Q2L / Q1L
        self.vFire_ = Q * (1. - np.exp(-1))

        if False:
            # NOTE: This is False to enable better spike-count-stability across
            # different rfLeast and rfAvg settings.
            # Calculate vFire_ according to ratio....
            # Q1M/Q1L * ln(1 - Q1L/Q2L * vFire) / ln(1 - Q1M/Q2M * vFire) == self.algSpikes
            # Note that initial guess is the maximum allowed so that the trivial
            # solution of 0 is avoided (see _init_solveForVFire_inner).
            guess = min(Q2L / Q1L, Q2M / Q1M)
            opts = {
                    'disp': True,
            }
            r = scipy.optimize.minimize(self._init_solveForVFire_inner, [guess*.5],
                    args=(float(self.algSpikes)*0+2., Q1L, Q2L, Q1M, Q2M),
                    options=opts, tol=1e-8,
                    bounds=[(1e-8, guess)])
            if not r.success:
                raise ValueError("Could not find vFire: {}".format(r.message))
            if abs(r.fun) >= 1e-4:
                raise ValueError("Bad vFire found: {} / {}".format(r.x[0], r.fun))
            if r.x[0] <= 1e-8:
                raise ValueError("Found trivial vFire solution: {}".format(r.x[0]))
            self.vFire_ = r.x[0]

        cdef double T_outSpike = self.algTOutSpike
        cdef double T_outSpikeGap = self.algTOutAccum
        if Q2M == 0 or Q1M == 0 or self.vFire_ == 0:
            raise ValueError("WTF: {}, {}, {}".format(Q1M, Q2M, self.vFire_))
        cdef double C = (-T_outSpikeGap * Q1M
                / log(1. - self.vFire_ * Q1M / Q2M))
        if np.isnan(C):
            raise ValueError("Could not get capacitance: {}, {}".format(
                    self.vFire_, Q2M / Q1M))

        if not self.noInhibition:
            # Halve that capacitance to make room for inhibition
            self.cNeuron_ = C * self.inhibCScale

            ## Now solve for the inhibition parameters, assuming we want the
            # capacitance of the inhibition to match the neurons
            self.cInhib_ = C * (1. - self.inhibCScale)

            r = self._init_solveForR(self.cInhib_, k=rfAvg, g=rfAvg, Q1=Q1M,
                    Q2=Q2M)
            self.rInhib_ = r
        else:
            self.cNeuron_ = C
            self.cInhib_ = -1.
            self.rInhib_ = -1.


    cpdef _init_getQ(self, double rfStored, double rfInput):
        """For a given ``rf``, returns (Q1, Q2).

        The stored rf has average value rfStored; the input has average value
        rfInput.

        The Q values chosen are for a receptive field; generally speaking, we
        want that receptive field to be as bright as possible, so as to use
        as much of the memristive device's range as we can.  Thus, the stored
        value is assumed to be maximum intensity inputs with a few at g_min.
        The input pattern is assumed to match, as closely as possible, the
        stored.
        """
        if self.nInputs <= 0:
            raise ValueError("Must init() first")

        cdef double G_max = 1. / self.phys_rMin
        cdef double K_max = self.algTInSpike / (self.algTInSpike
                + self.algTInMinGap)
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double Q1 = 0., Q2 = 0.

        if rfStored < g_min:
            raise ValueError("Minimum conductance is {}, but rfStored was {}"
                    .format(g_min, rfStored))

        # Q1, sum of conductances, will just be rfStored scaled
        Q1 = self.nInputs * G_max * rfStored

        # Q2, sum of input intensity times stored intensity, will be calculated
        # from the square of the actual rfStored configuration multiplied by
        # rfInput / rfStored.
        # rfStored = 1. * inputsHigh + g_min * (1 - inputsHigh)
        # inputsHigh = (rfStored - g_min) / (1 - g_min)
        cdef double inputsHigh = (rfStored - g_min) / (1 - g_min)
        cdef double inputsLow = 1. - inputsHigh
        Q2 = self.phys_vcc * self.nInputs * G_max * K_max * (
                inputsHigh * 1. * 1.
                + inputsLow * g_min * g_min
                ) * rfInput / rfStored
        return (Q1, Q2)


    cpdef double _init_solveForR(self, double C, double k, double g,
            double Q1, double Q2) except? -1.:
        if self.nInputs <= 0:
            raise RuntimeError("Must init() first")

        cdef double Q = Q2 / Q1
        cdef double T_fire = -self.cNeuron_ / Q1 * log((Q - self.vFire_) / Q)
        cdef double ff = np.finfo(float).eps * 1e1
        opts = {
                'eps': 1e-1,  # Basically the tolerance of the solved variable
                'ftol': ff,
                'gtol': 1e-20,
                'disp': False,
        }
        r = scipy.optimize.minimize(self._init_solveForR_inner, [1e3],
                args=(C, k, g, T_fire), options=opts,
                bounds=[ (1e-6, None) ])
        if not r.success:
            raise ValueError("Inhibition resistance failure for cInhib_ {}: {}"
                    "\n\n{}".format(self.cInhib_, r.message, r))
        if abs(r.fun) >= 1e-4:
            raise ValueError("Inhibition resistance failure: {} -> {}.  {}, "
                    "{}, {}, {}, {}, {}.\n\n{}".format(r.x, r.fun, C, k, g, Q1,
                        Q2, self.vFire_, r))
        return r.x[0]


    cpdef double _init_solveForR_inner(self, double[:] R0, double C_i,
            double k, double g, double T_fire) except? -1.:
        """Since a viable RC pair is difficult to compute analytically,
        this method's roots are the solutions to (self.cInhib_, R) that
        optimize the inhibition response in this network.

        :param R0: The guess from scipy.optimize.
        :param C_i: The inhibition capacitance to use.
        :param k: The actual input k to use
        :param g: The stored crossbar value to represent k.
        :param T_fire: The time calculation based on Q1 and Q2 for the neuron
                being balanced.
        """
        cdef double R = R0[0]

        cdef double vcc = self.phys_vcc
        cdef double G_max = 1. / self.phys_rMin
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double Rcb = 1. / (G_max * max(g, g_min))

        cdef double A = 1. / (Rcb * C_i)
        cdef double B = 1. / (R * C_i)

        # Desired spike period
        cdef double T_outSpikePeriod = (self.algTOutAccum
                + self.algTOutSpike) * g / k
        cdef double T_outSpike = self.algTOutSpike
        cdef double T_outSpikeGap = T_outSpikePeriod - T_outSpike
        cdef double T_outSpikeGapInhib = T_outSpikeGap - T_fire

        cdef double ke = k * self.algTInSpike / (self.algTInSpike
                + self.algTInMinGap)
        r = (0.5 * exp(ke * T_outSpikeGapInhib * B)
                - (1 - exp(-T_outSpike * A))
                    / (1 - exp(-ke * T_outSpikeGap * B - T_outSpike * A)))

        # The log version is more numerically stable
        r = log(0.5) + ke * T_outSpikeGapInhib * B - (
                log(1 - exp(-T_outSpike * A))
                - log(1 - exp(-ke * T_outSpikeGap * B - T_outSpike * A)))

        #print("INHIB R {} -> {}... {}".format(R, r, r*r))
        return r*r


    cpdef double _init_solveForVFire_inner(self, double[:] R0, double ratio,
            double Q1L, double Q2L, double Q1M, double Q2M) except? -1.:
        cdef double R = R0[0]
        #r = (ratio
        #        - Q1M / Q1L * np.log(1. - Q1L / Q2L * R)
        #            / np.log(1. - Q1M / Q2M * R))
        # The above equation, which is supposed to be equal to zero, can be
        # rearranged for better numerical accuracy:
        # ratio * np.log(1. - Q1M / Q2M * R)
        #     = Q1M / Q1L * np.log(1. - Q1L / Q2L * R)
        # But, np.log(1.) == 0, making R == 0 an implicit success.  This is
        # why R (vFire) was bounded previously to 1e-8 as a minimum.
        cdef double r = (pow(1. - Q1M / Q2M * R, ratio)
                - pow(1. - Q1L / Q2L * R, Q1M / Q1L))
        return r*r


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef _partial_fit(self, double[::1] x, double[::1] y):
        cdef int i, j
        cdef double u
        cdef FastRandom rand = self._rand
        cdef double[:, ::1] cb = self._lcaInternal_crossbar
        cdef double[::1] bo = self._bufferOut
        cdef double variabilityWrite = self.variabilityWrite
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef bint trainWithZero = self.trainWithZero

        if trainWithZero and variabilityWrite > 0:
            raise ValueError("Must not trainWithZero if variabilityWrite")

        if self.homeoRate > 0:
            self._homeostasis_train = True
        self._predict(x, bo)
        self._homeostasis_train = False
        self._reconstruct(bo, self._bufferIn)

        cdef np.ndarray[np.double_t, ndim=2] r = np.asmatrix(x) - self._bufferIn
        for i in range(cb.shape[0]):
            for j in range(cb.shape[1]):
                u = self.ada_.getDelta(i, j, -2. * bo[j] * r[0, i])
                if abs(u) <= 1e-3:
                    continue

                if not trainWithZero:
                    cb[i, j] = max(g_min, min(1.0, cb[i, j] + u))
                else:
                    cb[i, j] = max(0., min(1., cb[i, j] + u))

        if variabilityWrite > 0:
            self.variabilityWriteShakeup(variabilityWrite)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef _predict(self, double[::1] x, double[::1] y):
        ## Local temporaries
        cdef int i, j, k, m
        cdef double Q, u, u2, u3, u4
        cdef double inhibMult, inhibE
        cdef double dt
        cdef int hadSpike
        cdef FastRandom rand = self._rand

        ## Debug variables
        cdef bint DEBUG = 0  # Hardcoded debug flag
        cdef bint debug = self._debug
        cdef bint debugSpikes = (debug and self.debugSpikesTo is not None)
        cdef int debugSpikesN = 0
        cdef list debugSpikesInhibs
        cdef list debugSpikesInputs
        cdef list debugSpikesInputsW
        cdef double[::1] debugBufInputs, debugBufInputsW
        cdef np.uint8_t[:, :, :] debugImgData
        cdef double debugEventResolution
        cdef double debugEventLast
        cdef double[::1] debugSpikesDt
        cdef double[:, ::1] debugSpikesVolts

        cdef float debugE = 0.  # Debug energy
        cdef int debugNSteps = 0

        if debugSpikes:
            if (not isinstance(self.debugSpikesTo, tuple)
                    or len(self.debugSpikesTo) != 2):
                raise ValueError("debugSpikesTo must be tuple: (visualParams, "
                        "path); got {}".format(self.debugSpikesTo))
            debugSpikesInhibs = []
            debugSpikesInputs = []
            debugSpikesInputsW = []
            debugBufInputs = np.zeros(self.nInputs)
            debugBufInputsW = np.zeros(self.nInputs)

            # For simulation speed, voltages are recorded at the given resolution
            # only
            # Events:
            #  in{}spike: Times of input spikes.
            #  out{}spike: Times of output spikes.
            #  volt: Voltage of input inhibition / output state capacitors.
            debugEventResolution = 0.1 * min(self.algTOutAccum,
                    self.algTOutSpike, self.algTInSpike)
            debugEventLast = -debugEventResolution
            debugEvents = collections.defaultdict(list)

        ## Local mirror of relevant physical constants
        cdef int nInputs = self.nInputs
        cdef int nOutputs = self.nOutputs
        cdef double algTime = ((self.algTOutAccum + self.algTOutSpike)
                * self.algSpikes)
        cdef bint useInhib = not self.noInhibition
        cdef double[:, ::1] crossbar = self._lcaInternal_crossbar
        cdef bint updateReadConductances = True
        cdef double[:, ::1] gCrossbar = np.zeros((nInputs, nOutputs))
        cdef double vcc = self.phys_vcc
        cdef double vFire = self.vFire_, vThresh = vFire
        cdef double inhib_vThresh = 0.5 * vcc
        cdef bint trainWithZero = self.trainWithZero
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double G_max = 1. / self.phys_rMin
        cdef double K_maxIn = self.algTInSpike / (self.algTInSpike
                + self.algTInMinGap)
        cdef double C = self.cNeuron_
        cdef double C_i = self.cInhib_
        cdef double R_i = self.rInhib_
        cdef double onegBI = -1. * R_i * C_i
        cdef double negBI = 1. / onegBI
        cdef double variabilityRead = self.variabilityRead
        cdef bint homeoInUse = self._homeostasis_train
        cdef double[:] homeoTime = self.homeostasis_sinceFire_
        cdef double homeoTimeMin = algTime
        cdef double homeoRate = 1. / (algTime * nOutputs * self.homeoRate)

        ## Timing calculations
        # Bound the minimum time step due to input spiking events.  Otherwise,
        # the simulation takes ridiculous amounts of time for larger numbers
        # of inputs.  Remember, event-based is not always faster.
        cdef double dtMinInput = self.algTInSpike*0.25
        cdef double T_outSpike = self.algTOutSpike
        cdef double T_inSpike = self.algTInSpike
        # Calculate the scalar for the random inbound spikes for each input;
        # note that this is doubled since the mean of a uniform random is 0.5
        cdef double[:] T_inGapRand = np.zeros(nInputs)
        for i in range(nInputs):
            # Period of one input spike event at full capacity
            # Want T_inSpike / (T_inSpike + gap) = K_maxIn * x[i]
            # gap = T_inSpike / (K_maxIn * x[i]) - T_inSpike
            u = T_inSpike / (K_maxIn * max(1e-4, x[i]))
            T_inGapRand[i] = 2. * (u - T_inSpike)

        ## States
        cdef double[:] Vnode = np.zeros(nOutputs)
        cdef double[:] Q1 = np.zeros(nOutputs)
        cdef double[:] Q2 = np.zeros(nOutputs)
        cdef double[:] Vinhib = np.ones(nInputs) * inhib_vThresh
        cdef double[:] inState = np.zeros(nInputs)
        cdef double[:] fireCond = np.zeros(nInputs)
        cdef double t = 0., tLastSpike = 0.
        y[:] = 0.

        ## Initialize input spikes (initial phase unknown)
        for i in range(nInputs):
            inState[i] = rand.get() * (T_inGapRand[i]*0.5 + T_inSpike)

        if DEBUG:
            dbgK = np.zeros(nInputs)
        while t < algTime:
            debugNSteps += 1
            dt = algTime - t

            ## Reload conductances due to read variation or init?
            if updateReadConductances:
                updateReadConductances = False
                ## Calculate the actual crossbar conductances
                for i in range(nInputs):
                    for j in range(nOutputs):
                        # NOTE - g_min is enforced in partial_fit.  We do not
                        # re-inforce it here so that variability works.
                        if not trainWithZero:
                            u = G_max * crossbar[i, j]
                        else:
                            u = G_max * max(g_min, crossbar[i, j])
                        if variabilityRead > 0:
                            # is percentage (1 = 100%) of resistance
                            # variability... goes from 1. - var to 1 + var.
                            # Note that the standard deviation of this new
                            # random variable is significant; see Soudry et al.
                            # Retains same average.
                            u *= (
                                    (1. - variabilityRead)
                                    + rand.get() * (2 * variabilityRead))
                        gCrossbar[i, j] = u

                ## Initialize Q1 values which are constant
                Q1[:] = 0.
                for i in range(nInputs):
                    for j in range(nOutputs):
                        Q1[j] += gCrossbar[i, j]

            ## Update Q values
            Q2[:] = 0.
            for i in range(nInputs):
                if (Vinhib[i] > inhib_vThresh and useInhib
                        or inState[i] > T_inSpike):
                    # Inhibited or not spiking
                    continue

                for j in range(nOutputs):
                    Q2[j] += gCrossbar[i, j]
            for i in range(nOutputs):
                # K_max and k_i are both 1 here, since we know anything logged
                # is currently spiking.  Note that G_max was already rolled
                # into gCrossbar
                Q2[i] *= vcc

            if DEBUG:
                allTimes = []

            ## Calculate the next phase change
            # For input lines
            for i in range(nInputs):
                # Input line
                if inState[i] > T_inSpike:
                    u = inState[i] - T_inSpike
                else:
                    u = inState[i]

                # Input inhibition
                if (useInhib and Vinhib[i] > inhib_vThresh
                        and inState[i] <= T_inSpike):
                    u = min(u, log(inhib_vThresh / Vinhib[i]) * onegBI)
                u = max(u, dtMinInput)

                # Apply
                dt = min(dt, u)

            # For output neurons
            for i in range(nOutputs):
                Q = Q2[i] / Q1[i]

                if homeoInUse:
                    vThresh = vFire
                    if homeoTime[i] > homeoTimeMin:
                        vThresh *= 0.5 ** ((homeoTime[i] - homeoTimeMin)
                                * homeoRate)
                        # Since we sloppily integrate the homeostasis threshold,
                        # a negative time delta might be detected since the new
                        # homeotime allows a spike that would not have
                        # otherwise occurred.  Correct that by setting the
                        # vThresh to the maximum of the current voltage and
                        # the desired threshold.
                        vThresh = max(vThresh, Vnode[i])

                if Q <= vThresh:
                    continue

                u = -C / Q1[i] * log((Q - vThresh) / (Q - Vnode[i]))
                if DEBUG:
                    allTimes.append(u)
                dt = min(dt, u)

            if DEBUG:
                print("At {}: {}, {}.  {}.  {}.  {}.".format(t, dt, allTimes,
                        np.asarray(y), np.asarray(Vnode), np.asarray(Vinhib)))

            ##  Apply time update
            if dt < -1e-300:
                raise ValueError("BAD TIME UPDATE: {}.\n\nIn states: {}\n\nT_inSpike: {}"
                        .format(dt, inState, T_inSpike))

            # Ensure there are no zero-time updates
            dt += min(T_inSpike, T_outSpike) * 1e-8

            if debugSpikes:
                # Determine which time steps require a voltage trace.
                a = []
                u = debugEventLast
                while True:
                    u += debugEventResolution
                    if u - t <= dt:
                        a.append(u - t)
                        debugEventLast = u
                    else:
                        break
                if len(a) > 0:
                    debugSpikesDt = np.asarray(a)
                else:
                    debugSpikesDt = None
                debugSpikesVolts = np.zeros((len(a), nInputs + nOutputs))
                debugEvents['volt'].append(debugSpikesVolts)

            ### Input spikes / inhibition behavior
            if useInhib:
                # Update inhibition first as regardless of spike, the fall time
                # is the same.
                inhibMult = exp(dt * negBI)
                if debug:
                    inhibE = C_i * 0.5 * (1. - exp(2*dt*negBI))
                for i in range(nInputs):
                    if debugSpikes:
                        # Input voltage trace.
                        for j in range(debugSpikesVolts.shape[0]):
                            u = 1.
                            if inState[i] <= T_inSpike:
                                # spiking
                                u = exp(debugSpikesDt[j] * negBI)
                            debugSpikesVolts[j, i] = Vinhib[i] * u

                    if inState[i] > T_inSpike:
                        # Not spiking
                        continue

                    if DEBUG:
                        dbgK[i] += dt

                    if debugSpikes:
                        debugBufInputs[i] += dt
                        if Vinhib[i] <= inhib_vThresh:
                            debugBufInputsW[i] += dt

                    if debug:
                        debugE += inhibE * Vinhib[i] ** 2
                    Vinhib[i] *= inhibMult
            elif debugSpikes:
                for i in range(nInputs):
                    if inState[i] > T_inSpike:
                        # Not spiking
                        continue
                    debugBufInputs[i] += dt

                    if debugSpikes:
                        # No voltage trace.
                        for j in range(debugSpikesVolts.shape[0]):
                            debugSpikesVolts[j, i] = 0.

            # Charge all neurons according to dt, log spikes
            hadSpike = -1
            for i in range(nOutputs):
                u = Q2[i] / Q1[i]
                u2 = -Q1[i] / C
                if debug:
                    for j in range(nInputs):
                        # spiking?
                        if inState[j] <= T_inSpike and (not useInhib
                                or Vinhib[j] <= inhib_vThresh):
                            u3 = vcc
                        else:
                            u3 = 0.

                        # To find the power across this resistor, we need to
                        # find the voltage.  One end is our capacitor, whose
                        # instantaneous voltage is Vnode.  The other is the
                        # source junction, which is u3:
                        #
                        # v = u2 + (Vnode - u2) * u(t) - u3
                        #
                        # This derivation is detailed in the module
                        # documentation under :ref:`sec-power`.
                        #
                        # From that formula, Z_1 = u; Z_2 = u2; V_{in} = u3;
                        # t = dt; V_C(0) = u4;
                        u4 = Vnode[i]
                        debugE += gCrossbar[j, i] * (
                                dt * (u**2 - 2*u*u3 + u3**2)
                                + 2. / u2 * exp(dt*u2) * (
                                    u*u4 - u**2 - u3*u4 + u3*u)
                                + 0.5 / u2 * exp(2*dt*u2) * (
                                    u4**2 - 2*u4*u + u**2)
                                - 1./u2 * (u*u4 - 1.5*u**2 - 2*u3*u4 + 2*u3*u
                                    + 0.5*u4**2))

                if debugSpikes:
                    # Log the voltage trace for this node
                    for j in range(debugSpikesVolts.shape[0]):
                        u3 = exp(u2 * debugSpikesDt[j])
                        debugSpikesVolts[j, nInputs + i] = (
                                u * (1. - u3) + Vnode[i] * u3)

                # Out of debugging, change u2 to be its exponent
                u2 = exp(u2 * dt)
                Vnode[i] = u * (1. - u2) + Vnode[i] * u2
                if homeoInUse:
                    vThresh = vFire
                    if homeoTime[i] > homeoTimeMin:
                        vThresh *= 0.5 ** ((homeoTime[i] - homeoTimeMin)
                                * homeoRate)

                    # Finally, add time not fired (will immediately be reset if
                    # a fire was triggered
                    homeoTime[i] += dt
                if Vnode[i] < vThresh:
                    continue

                # Spiking event; tally up inhibition.  We have a flag for
                # spiking because when multiple spikes happen, inhibition
                # should still only be updated once
                hadSpike = i

                if debugSpikes:
                    debugEvents['out' + str(i) + 'spike'].append(t + dt)

                y[i] += 1.
                if homeoInUse:
                    homeoTime[i] = 0.

                for j in range(nInputs):
                    # Conductance stacks
                    fireCond[j] += gCrossbar[j, i]

            # Update inhibition of each row based on spiking events and
            # add spike time to dt
            if hadSpike >= 0:
                # Debug first; needs fireCond
                if debugSpikes:
                    debugSpikesN += 1
                    debugSpikesInputs.append(debugBufInputs.copy())
                    debugSpikesInputsW.append(debugBufInputsW.copy())
                    debugSpikesInhibs.append(fireCond.copy())

                    debugBufInputs[:] = 0.
                    debugBufInputsW[:] = 0.

                # Energy from draining neuron capacitors
                if debug:
                    for i in range(nOutputs):
                        debugE += 0.5 * C * Vnode[i]**2

                # If we're using read variability, we need to re-up the
                # conductances
                if variabilityRead > 0:
                    updateReadConductances = True

                if debugSpikes:
                    # Need to count out T_outSpike more time.
                    a = []
                    u = debugEventLast
                    while True:
                        u += debugEventResolution
                        if u - (t + dt) <= T_outSpike:
                            a.append(u - (t + dt))
                            debugEventLast = u
                        else:
                            break
                    if len(a) > 0:
                        debugSpikesDt = np.asarray(a)
                    else:
                        debugSpikesDt = None
                    debugSpikesVolts = np.zeros((len(a), nInputs + nOutputs))
                    debugEvents['volt'].append(debugSpikesVolts)

                # No more voltages; they were all reset
                Vnode[:] = 0.
                for j in range(nInputs):
                    if useInhib:
                        u2 = 1. / fireCond[j]  # R_{cb}

                        if debug:
                            debugE += 0.5 * C_i * (vcc - Vinhib[j]) ** 2 * (
                                    1. - exp(
                                        -2*T_outSpike * fireCond[j] / C_i))
                        if debugSpikes:
                            for i in range(debugSpikesDt.shape[0]):
                                u = exp(-debugSpikesDt[i] / (C_i * u2))
                                debugSpikesVolts[i, j] = vcc * (1. - u) + Vinhib[j] * u

                        u = exp(-T_outSpike / (C_i * u2))  # e^{-T_{spike}A}
                        Vinhib[j] = vcc * (1. - u) + Vinhib[j] * u

                    # Also reset fireCond so it's ready for the next spike
                    # event.
                    fireCond[j] = 0.

                # The time delta for this step now includes the spike
                dt += T_outSpike
                tLastSpike = t + dt

            # Finally, update input lines according to dt, including any time
            # added due to spiking
            for i in range(nInputs):
                inState[i] -= dt
                # MUST be a while loop - T_outSpike can be comparatively large
                while inState[i] <= 0.:
                    if debugSpikes:
                        debugEvents['in' + str(i) + 'spike'].append(t + dt + inState[i] - T_inSpike)
                    inState[i] += rand.get() * T_inGapRand[i] + T_inSpike

            # And update sim time so far
            t += dt

        u = 1. / self.algSpikes
        for i in range(nOutputs):
            y[i] *= u

        if self._debug:
            self.debugInfo_.simTime[self.debugInfo_.index] = t
            self.debugInfo_.energy[self.debugInfo_.index] = debugE

        if DEBUG:
            print("K: {} / {}".format(dbgK / algTime,
                    np.asarray(x) * self.algTInSpike / (self.algTInSpike
                        + self.algTInMinGap)))
        if debugSpikes:
            # Reconstruct, populate image
            debugImgData = np.zeros((
                    self.debugSpikesTo[0][1]*(2 + debugSpikesN),
                    self.debugSpikesTo[0][0]*3,
                    self.debugSpikesTo[0][2]), dtype=np.uint8)

            # First row: Input and reconstruction, real magnitude
            self._debugSpikesBlit(debugImgData, 0, 0, x, False)
            self._reconstruct(y, self._bufferIn)
            self._debugSpikesBlit(debugImgData, 0, 1, self._bufferIn, False)

            # Second row: Input and reconstruction, scaled
            self._debugSpikesBlit(debugImgData, 1, 0, x)
            self._debugSpikesBlit(debugImgData, 1, 1, self._bufferIn)

            # Subsequent rows: seen inputs w/o inhibition, seen inputs w/
            # inhib, response
            for i in range(debugSpikesN):
                self._debugSpikesBlit(debugImgData, i+2, 0,
                        debugSpikesInputs[i])
                self._debugSpikesBlit(debugImgData, i+2, 1,
                        debugSpikesInputsW[i])
                self._debugSpikesBlit(debugImgData, i+2, 2,
                        debugSpikesInhibs[i])

            imgArr = debugImgData
            if self.debugSpikesTo[0][2] == 1:
                imgArr = np.reshape(imgArr, imgArr.shape[:2])
            scipy.misc.imsave(self.debugSpikesTo[1].format(
                    self.debugInfo_.index), imgArr)

            # Now plot the signal traces, with a helpful visualization.
            show_img = True if nInputs == 4 else False
            tot_ax = nInputs + nOutputs + (2 if show_img else 0)
            norm_ax = nInputs + nOutputs
            f, ax = plt.subplots(tot_ax, 1, sharex='all',
                    sharey=False, figsize=(3.6, 0.5 + 0.35 * tot_ax),
                    gridspec_kw=dict(wspace=0, hspace=0))

            n_slots = 0
            for v in debugEvents['volt']:
                n_slots += v.shape[0]
            times = np.arange(n_slots) * debugEventResolution
            data = np.zeros((nInputs + nOutputs, n_slots))
            n_slot = 0
            for v in debugEvents['volt']:
                for i in range(nInputs):
                    data[i, n_slot:n_slot + v.shape[0]] = v[:, i]
                for j in range(nOutputs):
                    data[nInputs + j, n_slot:n_slot + v.shape[0]] = v[:, nInputs + j]
                n_slot += v.shape[0]
            spikes = np.zeros((nInputs + nOutputs, n_slots))
            spikes_cnt = np.zeros((nInputs + nOutputs + 1, n_slots))  # last is any output
            for i in range(nInputs):
                for t in debugEvents['in' + str(i) + 'spike']:
                    spikes[i, (times >= t) & (times < t + self.algTInSpike)] = 1.
                    spikes_cnt[i, np.searchsorted(times, t)] += 1
            spikesAny = np.zeros(n_slots)
            for j in range(nOutputs):
                for t in debugEvents['out' + str(j) + 'spike']:
                    spikes[nInputs + j, (times >= t) & (times < t + self.algTOutSpike)] = 1.
                    spikes_cnt[nInputs + j, np.searchsorted(times, t)] += 1
                    spikes_cnt[nInputs + nOutputs, np.searchsorted(times, t)] += 1
                spikesAny += spikes[nInputs + j]

            # Mask out any input spikes occurring during an output spike
            for i in range(nInputs):
                spikes[i] *= 1 - np.clip(spikesAny, 0, 1)

            # Change spikes to a boolean array so matplotlib can render
            spikes = np.asarray(spikes, dtype=bool)

            # Plot in ns
            times *= 1e9
            for axx in ax:
                xmn = 0
                if show_img:
                    xmn = -1
                axx.set_xlim(xmn, times[times.shape[0]-1])
                axx.set_yticks([])
                axx.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
                axx.grid('on', linestyle='--', axis='x')

            ax[1].set_yticks([inhib_vThresh])
            ax[norm_ax-1].set_yticks([vFire])

            ax[tot_ax-1].set_xlabel('Time (ns)')
            ax[norm_ax//2].set_ylabel('Voltage (V)')
            for i in range(nInputs):
                ax[i].plot(times, data[i])
                ax[i].plot(times, (inhib_vThresh,) * len(times), linestyle='--')
                ax[i].set_ylim(*ax[i].get_ylim())  # bug workaround
                mn, mx = ax[i].get_ylim()
                mn -= 0.01
                mx += 0.01
                ax[i].set_ylim(mn, mx)
                trans = ax[i].get_xaxis_transform()
                ax[i].fill_between(times, 0, 1, where=spikes[i], facecolor='black', alpha=0.25, transform=trans)
                for j in range(nOutputs):
                    ax[i].fill_between(times, 0, 1, where=spikes[nInputs + j], facecolor='red', alpha=0.25, transform=trans)
            for j in range(nOutputs):
                i = nInputs + j
                ax[i].plot(times, data[i])
                ax[i].plot(times, (vFire,) * len(times), linestyle='--', color='green')
                ax[i].set_ylim(*ax[i].get_ylim())  # bug workaround
                mn, mx = ax[i].get_ylim()
                mn = 0.
                mx += 0.01
                ax[i].set_ylim(mn, mx)
                trans = ax[i].get_xaxis_transform()
                ax[i].fill_between(times, 0, 1, where=spikes[i], facecolor='black', alpha=0.25, transform=trans)

            if show_img:
                # Use images to demonstrate functionality
                ims = [matplotlib.image.imread('debug_gfx/{}.png'.format(i))
                        for i in range(nInputs)]
                im1 = ax[len(ax) - 2]
                im2 = ax[len(ax) - 1]

                imargs = dict(interpolation='none', aspect='auto')

                # Show templates
                for i in range(nInputs):
                    _add_text(ax[i], 'In{}'.format(i))
                    ymn, ymx = ax[i].get_ylim()
                    im = ax[i].imshow(ims[i], extent=(-1, 0, ymn, ymx),
                            **imargs)
                    ax[i].set_ylim(ymn, ymx)
                for j in range(nOutputs):
                    _add_text(ax[nInputs + j], 'Out{}'.format(j))
                    ymn, ymx = ax[nInputs + j].get_ylim()
                    for i in range(nInputs):
                        ax[nInputs + j].imshow(ims[i], extent=(-1, 0, ymn, ymx),
                                alpha=crossbar[i, j], **imargs)

                _add_text(ax[nInputs + nOutputs], 'Reconstruction')
                for i in range(nInputs):
                    ax[nInputs + nOutputs].imshow(ims[i], extent=(-1, 0, 0, 1),
                            alpha=x[i], **imargs)

                _add_text(ax[nInputs + nOutputs + 1], 'Inputs Seen')

                # Assuming linear, know that a value of 1. expects a spike
                # every T_inSpike / K_maxIn.
                spikes_cnt = spikes_cnt.cumsum(1)

                t = times[0]
                img_times = np.linspace(
                        (len(times)-1) * 0.5 / 10,
                        (len(times)-1) * 9.5 / 10,
                        10, dtype=int)
                last_ins = np.zeros(nInputs)
                for m, j in enumerate(img_times):
                    # left, right, bottom, top
                    t = times[j]
                    u = t - times[img_times[0]]
                    extent = (u, u + times[img_times[1]] - times[img_times[0]], 0, 1)

                    # Data going in to each output spike.
                    j_last = np.searchsorted(spikes_cnt[nInputs + nOutputs], m)
                    # Skip over activity
                    j_last += int(self.algTOutSpike / debugEventResolution)
                    j_this = np.searchsorted(spikes_cnt[nInputs + nOutputs], m+1)
                    for i in range(nInputs):
                        dt = (j_this - j_last) * debugEventResolution
                        # seconds of spike activity
                        act = (spikes[i, j_last:j_this] * debugEventResolution).sum()
                        # expected seconds of spike activity for a 1.0 input
                        ex = max(1e-16, T_inSpike * dt / (T_inSpike / K_maxIn))
                        #print(f'{i} @ {m} / {dt}: {act} / {ex}')
                        im2.imshow(ims[i], alpha=act / ex, extent=extent,
                                **imargs)

                    # Reconstruction at time t.
                    for i in range(nInputs):
                        u = 0.
                        for k in range(nOutputs):
                            u += spikes_cnt[nInputs + k, j] * crossbar[i, k]
                        u *= (self.algTOutAccum + self.algTOutSpike) / (1e-20 + t * 1e-9)
                        im1.imshow(ims[i], alpha=u, extent=extent, **imargs)

            fname = self.debugSpikesTo[1].format(str(self.debugInfo_.index) + '_signal')
            if fname.endswith('.png'):
                fname = fname[:len(fname)-4] + '.pdf'
            plt.tight_layout(pad=0)
            plt.savefig(fname)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[::1] y, double[::1] r):
        cdef int i, j
        cdef int nInputs = self.nInputs, nOutputs = self.nOutputs
        r[:] = 0.0
        for j in range(nOutputs):
            for i in range(nInputs):
                r[i] += self._lcaInternal_crossbar[i, j] * y[j]


def _add_text(ax, text):
    """Helper."""
    ax.text(-0.93, 0.935, text, color='white',
        verticalalignment='top', transform=ax.get_xaxis_transform())
    ax.text(-0.95, 0.95, text, color='black',
            verticalalignment='top', transform=ax.get_xaxis_transform())

