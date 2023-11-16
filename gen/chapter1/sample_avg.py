import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x_means = []
    N = 1

    for _ in range(1000):
        xs = []
        for i in range(N):
            x = np.random.rand()
            xs.append(x)
        mean = np.mean(xs)
        x_means.append(mean)

    plt.hist(x_means, bins='auto', density=True)
    plt.title(f'N={N}')
    plt.show()
    plt.close()
