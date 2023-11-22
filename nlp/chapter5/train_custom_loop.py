import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common_nlp.optimizer import SGD
from dataset_nlp import ptb
from simple_rnnlm import SimpleRnnlm


if __name__ == '__main__':
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    max_epochs = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)

    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    jump = (corpus_size - 1) // batch_size
    offsets = [i * jump for i in range(batch_size)]

    for epoch in range(max_epochs):
        # iter⇨iteration・繰り返し
        for iter in range(max_iters):
            # Mini batch
            batch_x = np.empty((batch_size, time_size), dtype='int')
            batch_t = np.empty((batch_size, time_size), dtype='int')
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    batch_x[i, t] = xs[(offset + time_idx) % data_size]
                    batch_t[i, t] = ts[(offset + time_idx) % data_size]
                time_idx += 1

            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # Perplexity(ppl)⇨確率の逆数・低いほどいい
        ppl = np.exp(total_loss / loss_count)
        print(f'| epoch {epoch+1} | perplexity {ppl:.2f}%')
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    # Plot
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()
    plt.close()
