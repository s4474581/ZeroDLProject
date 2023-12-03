import os
import numpy as np
import matplotlib.pyplot as plt


def multivariate_normal(x, mu, cov):
    # 行列
    det = np.linalg.det(cov)
    # 逆行列⇨inverse
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'height_weight.txt')
    xs = np.loadtxt(path)

    # MLE
    mu = np.mean(xs, axis=0)
    # cov⇨今日分散(covariance)
    cov = np.cov(xs, rowvar=False)
    print(cov)

    small_xs = xs[:500]
    x = np.arange(150, 195, 0.5)
    y = np.arange(45, 75, 0.5)
    # 格子列
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = multivariate_normal(x, mu, cov)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot_surface(X, Y, Z, cmap='jet')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(small_xs[:, 0], small_xs[:, 1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(156, 189)
    ax2.set_ylim(36, 79)
    ax2.contour(X, Y, Z)
    plt.show()
    plt.close()