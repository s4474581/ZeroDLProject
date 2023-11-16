import os

import numpy as np
import matplotlib.pyplot as plt

from common_gen.normalize import normal


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)

    mu = np.mean(xs)
    sigma = np.std(xs)

    x = np.arange(150, 190)
    y = normal(x, mu, sigma)

    plt.hist(xs, bins='auto', density=True)
    plt.plot(x, y)
    plt.show()
    plt.close()


