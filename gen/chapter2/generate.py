import os

import  numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    mu = np.mean(xs)
    sigma = np.std(xs)

    samples = np.random.normal(mu, sigma, 10000)

    # density⇨密度
    plt.hist(xs, bins='auto', density=True, alpha=0.7, label='original')
    plt.hist(samples, bins='auto', density=True, alpha=0.7, label='generated')
    plt.legend()
    plt.show()
    plt.close()

