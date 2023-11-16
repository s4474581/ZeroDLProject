# スパイラルデータセットの可視化
import sys
# 親ディレクトリのインポート⇨dataset_nlpディレクトリのインポート
sys.path.append('..')
from dataset_nlp import spiral
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x, t = spiral.load_data()
    print('x:', x.shape)
    print('t:', t.shape)

    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
    plt.show()
    plt.close()

