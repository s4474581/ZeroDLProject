# GMM 混合ガウスモデル
import matplotlib.pyplot as plt
import numpy as np

from common import multivariate_normal

phis = np.array([0.2, 0.5, 0.3])
mus = np.array([[-3., -1.5], [0., 0.], [3., 1.5]])
covs = np.array([
    [[0.4, 0.5], [0.5, 0.8]], [[0.4, 0.5], [0.5, 0.8]], [[0.4, 0.5], [0.5, 0.8]]])


# GMM⇨混合ガウスモデル
def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


if __name__ == '__main__':
    x = np.array([2., 0.])
    y = gmm(x, phis, mus, covs)
    print(y)

    # plot
    xs = ys = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gmm(x, phis, mus, covs)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    # axes.plot_surface()⇨3次元グラフ描画
    ax1.plot_surface(X, Y, Z, cmap='jet')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    # axes.contour()⇨等高線グラフ描画
    ax2.contour(X, Y, Z)
    plt.show()
    plt.close()


