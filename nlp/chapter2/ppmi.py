# 正の相互情報量
import sys
sys.path.append('..')

import numpy as np
from common_nlp.util import preprocess, create_co_matrix, cos_similarity, ppmi


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    # print('corpus:', corpus, ' word_to_id: ', word_to_id)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print('Co-occurrence matrix')
    print(C)
    print('--' * 25)
    print('PPMI')
    print(W)
