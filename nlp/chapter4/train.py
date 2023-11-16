import pickle
import sys
sys.path.append('..')

from common_nlp.np import *
from common_nlp.trainer import Trainer
from common_nlp.optimizer import Adam
from cbow import CBOW
from common_nlp.util import preprocess, create_contexts_target, convert_one_hot
from dataset_nlp import ptb


if __name__ == '__main__':
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)

    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    word_vecs = model.word_vecs

    # Save parameter
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)

    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
