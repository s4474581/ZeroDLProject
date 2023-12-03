import os
import numpy as np
import scipy


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), ('height.txt'))
    xs = np.loadtxt(path)
    mu = np.mean(xs)
    sigma = np.std(xs)

    # 累積分布関数(sigmoidっぽい)
    p1 = scipy.stats.norm.cdf(160, mu, sigma)
    print('p(x <= 160)', p1)
    p2 = scipy.stats.norm.cdf(180, mu, sigma)
    print('p(x > 180)', 1 - p2)