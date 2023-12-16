import numpy as np
import matplotlib.pyplot as plt

from numpy_matrix import multivariate_normal


if __name__ == '__main__':
    mu = np.array([0.5, -0.2])
    cov = np.array([[2.0, 0.3], [0.3, 0.5]])
    xs = ys = np.arange(-5, 5, 0.1)
    # np.meshgrid()⇨格子点を作る
    X, Y = np.meshgrid(xs, ys)
    # np.zeros_like()⇨データ型を受け継ぐ
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = multivariate_normal(x, mu, cov)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    # axes.plot_surface()⇨3次元グラフ描画
    ax1.plot_surface(X, Y, Z, cmap='viridis')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    # axes.contour()⇨等高線グラフ描画
    ax2.contour(X, Y, Z)
    plt.show()
    plt.close()

