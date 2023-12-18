import os

import numpy as np
import matplotlib.pyplot as plt

from common import multivariate_normal, gmm


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
    orig_xs = np.loadtxt(path)

    phis = np.array([0.35589156, 0.64410844])
    mus = np.array([[2.03643399, 54.47897503], [4.28970224, 79.9686019]])
    covs = np.array([[[0.06920385, 0.43554569], [0.43554569, 33.6998689]],
                     [[0.16991733, 0.93995968], [0.93995968, 36.0389029]]])

    N = 500
    new_xs = np.zeros((N, 2))
    for n in range(N):
        k = np.random.choice(2, p=phis)
        mu, cov = mus[k], covs[k]
        new_xs[n] = np.random.multivariate_normal(mu, cov)

    plt.scatter(orig_xs[:, 0], orig_xs[:, 1], alpha=0.7, label='Original')
    plt.scatter(new_xs[:, 0], new_xs[:, 1], alpha=0.7, label='Generated')
    plt.legend()
    plt.show()
    plt.close()
