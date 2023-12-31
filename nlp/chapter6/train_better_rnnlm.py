# Rnnlmの訓練
import sys
sys.path.append('..')

from common_nlp import config
# config.GPU = True

from common_nlp.optimizer import SGD
from common_nlp.trainer import RnnlmTrainer
from common_nlp.util import eval_perplexity
from dataset_nlp import ptb
from better_rnnlm import BetterRnnlm


if __name__ == '__main__':
    batch_size = 20
    wordvec_size = 650
    hidden_size = 650
    time_size = 35
    lr = 20.0
    max_epochs = 40
    max_grads = 0.25
    dropput = 0.5

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('val')
    corpus_test, _, _ = ptb.load_data('test')

    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = BetterRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    best_ppl = float('inf')
    for epoch in range(max_epochs):
        trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grads)

        model.reset_state()
        ppl = eval_perplexity(model, corpus_test)
        print('Valid perplexity: ', ppl)

        if best_ppl > ppl:
            best_ppl = ppl
            model.save_params()
        else:
            lr /= 4.0
            optimizer.lr = lr

        model.reset_state()
        print('--' * 25)

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('Test perplexity: ', ppl_test)

