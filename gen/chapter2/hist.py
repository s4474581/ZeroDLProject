import os

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    print('xs.shape:', xs.shape)

    plt.hist(xs, bins='auto', density=True)
    plt.show()
    plt.close()


