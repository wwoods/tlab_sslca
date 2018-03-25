
from sslca import LcaSpikingWoodsAnalyticalInhibition

import matplotlib.pyplot
import numpy as np

FM_FONT_SIZE = 8
matplotlib.rc('font', size=FM_FONT_SIZE)
matplotlib.pyplot.rcParams.update({
        'figure.dpi': 300,
        'font.size': FM_FONT_SIZE,
        'image.cmap': 'viridis',
        'legend.fontsize': FM_FONT_SIZE,
        'text.latex.unicode': True,
        'text.usetex': True,
})

def main():
    """The SSLCA implementation expects a dataset which can be indexed as
    dataset[sample index, input index].  We'll use a series of 4x1 "images",
    with known sparse elements [1., 1, 0, 0], [0, 1, 1, 0], and
    [0, 0, 1, 1].
    """
    train = np.asarray([
            [1., 1., 0., 0.],
            [0., 0., 1., 1.],
            [1., 1., 1., 1.],
            [0.5, 1., 0.5, 0.],
            [0., 0.5, 1., 0.5],
            [0., 0.5, 0.5, 0.],
    ])
    test = np.asarray([
            [1., 1., 0., 0.],
            [0., 1., 1., 0.],
            [0., 0., 1., 1.],
    ])

    net = LcaSpikingWoodsAnalyticalInhibition(
            nOutputs=3,
            algSpikes=10,
            trainWithZero=True,
            #phys_rMax=207e6,
    )
    net.init(4, 3)

    for epoch in range(1000):
        net.partial_fit(train)

    print("Neurons (each column):\n{}".format(np.asarray(net.crossbar_)))
    print("Train RMSE: {}".format(((((net.reconstruct(train) - train) ** 2).sum()
            / (train.shape[0] * train.shape[1])) ** 0.5)))
    print("Test RMSE: {}".format(((((net.reconstruct(test) - test) ** 2).sum()
            / (test.shape[0] * test.shape[1])) ** 0.5)))

    # Demonstration of spike timing
    net = LcaSpikingWoodsAnalyticalInhibition(
            nOutputs=2,
            algSpikes=10,
            rfAvg=0.5,  # Best set to true data set average (including any bias;
                        # should not be close to rMin / rMax!)
            trainWithZero=True,
    )
    net.init(4, 2)
    net.crossbar_ = np.asarray([[1., 1., 0, 0], [0, 0, 1, 1]]).swapaxes(0, 1).copy()
    net.debugSpikesTo = ((4,1,1), 'spiketest{}.png')
    net.predict(np.asarray([[0., 1, 0.5, 0.5]]), debug=True)
    net.debugSpikesTo = None



if __name__ == '__main__':
    main()

