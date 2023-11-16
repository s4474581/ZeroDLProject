import sys
sys.path.append('..')

from dataset_nlp import ptb


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print('corpus size:', len(corpus))
    print('corpus[:30]:', corpus[:30])
    print('id_to_word[0]:', id_to_word[0])
    word = input('search word:')
    if word == '':
        word = 'car'
    print(f'word_to_id[{word}]:', word_to_id[word])
