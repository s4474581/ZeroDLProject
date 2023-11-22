import sys
sys.path.append('..')

from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm
from dataset_nlp import ptb
from common_nlp.util import eval_perplexity


if __name__ == '__main__':
    model = Rnnlm()
    # model = BetterRnnlm()

    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('Test perplexity: ', ppl_test)