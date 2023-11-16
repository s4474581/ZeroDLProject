import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from common_gen.normalize import normal


if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)
    y = normal(x)

    plt.plot(x, y)
    plt.show()
    plt.close()