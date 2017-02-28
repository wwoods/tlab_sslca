
from sslca import LcaSpikingWoodsAnalyticalInhibition

import numpy as np

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
            [1., 2., 1., 0.],
            [0., 1., 2., 1.],
            [0., 2., 2., 0.],
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



if __name__ == '__main__':
    main()

