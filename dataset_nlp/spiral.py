# スパイラルデータセット
import numpy as np

SEED = 1984


def load_data(seed=SEED):
    np.random.seed(seed)
    N = 100
    DIM = 2
    CLS_NUM = 3

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int32)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1. * rate
            theta = j * 4. + 4. * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

