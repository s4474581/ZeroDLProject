# 類似単語ランキング
import sys
sys.path.append('..')

from common_nlp.util import preprocess, create_co_matrix, most_similar


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, C, top=5)
