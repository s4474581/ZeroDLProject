import numpy as np


def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * (np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    return y
