import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # データ少なめの3次元グラフ
    X = np.array([[-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2]])
    Y = np.array([[-2, -2, -2, -2, -2],
                  [-1, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2]])
    Z = X ** 2 + Y ** 2

    ax = plt.axes(projection='3d')
    # plot_surface()⇨3次元グラフ描画
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()

    # データ多めの3次元グラフ
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()

    # 等高線グラフ
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    ax = plt.axes()
    # contour()⇨等高線グラフ
    ax.contour(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    plt.close()




