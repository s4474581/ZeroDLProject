import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from common_nlp.util import preprocess, create_co_matrix, ppmi


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    # np.linalg.svg()⇨特異値分解
    U, S, V = np.linalg.svd(W)

    np.set_printoptions(precision=3)
    print('C:', C[0])
    print('W:', W[0])
    print('U:', U[0])

    # Plot
    for word, word_id in word_to_id.items():
        # plt.annotate()⇨注釈
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()
    plt.close()
