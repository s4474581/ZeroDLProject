import os
import numpy as np
import matplotlib.pyplot as plt

from common import multivariate_normal, gmm, plot_contour


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
    xs = np.loadtxt(path)
    print('xs.shape:', xs.shape)

    phis = np.array([0.5, 0.5])
    mus = np.array([[0., 50.], [0., 100.]])
    covs = np.array([np.eye(2), np.eye(2)])

    K = len(phis)
    N = len(xs)
    ITERS = 10

    for iter in range(ITERS):
        qs = np.zeros((N, K))
        for n in range(N):
            x = xs[n]
            for k in range(K):
                phi, mu, cov = phis[k], mus[k], covs[k]
                qs[n, k] = phi * multivariate_normal(x, mu, cov)
            qs[n] /= gmm(x, phis, mus, covs)

        qs_sum = qs.sum(axis=0)
        for k in range(K):
            # phis
            phis[k] = qs_sum[k] / N

            # mus
            c = 0
            for n in range(N):
                c += qs[n, k] * xs[n]
            mus[k] = c / qs_sum[k]

            # covs
            c = 0
            for n in range(N):
                z = xs[n] - mus[k]
                z = z[:, np.newaxis]
                c += qs[n, k] * z @ z.T
            covs[k] = c / qs_sum[k]

    plt.scatter(xs[:, 0], xs[:, 1])
    plot_contour(phis, mus, covs)
    plt.show()
    plt.close()
