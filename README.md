This repository contains the simulation implementation used by Woods et al., "Fast and Accurate Sparse Coding of Visual Stimuli with a Simple, Ultra-Low-Energy Spiking Architecture," 2017 (submitted).

Example usage can be found in `example.py`.  Main algorithms contained in `sslca/sslca.pyx`.  To get results like in the paper, plug in your favorite CIFAR-10 loader rather than the hand-crafted train / test samples.

Requires at least cython, matploblib, numpy, scikit-learn, and scipy.

