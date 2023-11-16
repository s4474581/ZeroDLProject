# Central limit⇨中心限界
import numpy as np
import matplotlib.pyplot as plt

from common_gen.normalize import normal


if __name__ == '__main__':
    x_means = []
    N = 3

    for _ in range(10000):
        xs = []
        for i in range(N):
            x = np.random.rand()
            xs.append(x)
        mean = np.sum(xs)
        x_means.append(mean)

    x_norm = np.linspace(-5, 5, 1000)
    mu = 0.5
    sigma = np.sqrt(1 / 12 / N)
    y_norm = normal(x_norm, mu, sigma)

    # plot
    plt.hist(x_means, bins='auto', density=True)
    plt.plot(x_norm, y_norm)
    plt.title(f'N={N}')
    plt.xlim(-1, 2)
    plt.show()
    plt.close()
