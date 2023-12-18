import numpy as np


# KL⇨KLダイバージェンス
def kl(p, q):
    return p[0] * np.log(p[0] / q[0]) + p[1] * np.log(p[1] / q[1])


if __name__ == '__main__':
    p = [0.7, 0.3]
    q = [0.6, 0.4]
    print(kl(p, q))

    p = [0.7, 0.3]
    q = [0.2, 0.8]
    print(kl(p, q))

    p = [0.7, 0.3]
    q = [0.7, 0.3]
    print(kl(p, q))
