import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
    xs = np.loadtxt(path)

    print('xs.shape:', xs.shape)
    print('xs[0]:', xs[0])
    print('xs[1]:', xs[1])

    plt.scatter(xs[:, 0], xs[:, 1])
    # Eruption⇨噴火？
    plt.xlabel('Duration of eruption')
    plt.ylabel('Interval between eruptions')
    plt.show()
    plt.close()