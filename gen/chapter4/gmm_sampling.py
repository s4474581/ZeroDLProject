# GMM Sample 混合ガウスモデルのサンプル
import matplotlib.pyplot as plt
import numpy as np

phis = np.array([0.2, 0.5, 0.3])
mus = np.array([[-3., -1.5], [0., 0.], [3., 1.5]])
covs = np.array([
    [[0.4, 0.5], [0.5, 0.8]], [[0.4, 0.5], [0.5, 0.8]], [[0.4, 0.5], [0.5, 0.8]]])

N = 1000


def sample():
    k = np.random.choice(3, p=phis)
    mu, cov = mus[k], covs[k]
    x = np.random.multivariate_normal(mu, cov)
    return x


if __name__ == '__main__':
    xs = np.zeros((N, 2))
    for i in range(N):
        x = sample()
        xs[i] = x

    # plot
    plt.scatter(xs[:, 0], xs[:, 1], s=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close()
