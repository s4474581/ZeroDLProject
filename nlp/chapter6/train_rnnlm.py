# Rnnlmの訓練
import sys
sys.path.append('..')

from common_nlp.optimizer import SGD
from common_nlp.trainer import RnnlmTrainer
from common_nlp.util import eval_perplexity
from dataset_nlp import ptb
from rnnlm import Rnnlm


if __name__ == '__main__':
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100
    time_size = 35
    lr = 20.0
    max_epochs = 4
    max_grads = 0.25

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 勾配クリッピング適用学習
    trainer.fit(xs, ts, max_epochs, batch_size, time_size, max_grads, eval_interval=20)
    trainer.plot(ylim=(0, 500))

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('Test perplexity: ', ppl_test)

    model.save_params()

