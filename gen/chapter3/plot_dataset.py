import os

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'height_weight.txt')
    # txtファイルのデータをnp配列に読み込む
    xs = np.loadtxt(path)

    print('xs.shape:', xs.shape)
    print('xs[0]:', xs[0])

    small_xs = xs[:500]
    plt.scatter(small_xs[:, 0], small_xs[:, 1])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()
    plt.close()
