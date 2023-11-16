# import sys
# sys.apth.append('..')
import pickle

from common_nlp.util import most_similar, analogy


if __name__ == '__main__':
    pkl_file = 'cbow_params.pkl'
    # pkl_file = 'skipgram_params.pkl'

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    print('--' * 25)
    analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
    analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
    analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
    analogy('good', 'better', 'bed', word_to_id, id_to_word, word_vecs)

