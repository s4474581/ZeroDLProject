#
import sys
sys.path.append('..')

from common_nlp.optimizer import SGD
from common_nlp.trainer import Trainer
from dataset_nlp import spiral
from two_layer_net import TwoLayerNet


if __name__ == '__main__':
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.

    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=3, output_size=3)
    optimizer = SGD(lr=learning_rate)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
    trainer.plot()


'''
perplexity(ppl)
言語モデルの評価指標
小さいと正しい予測ができている

$$ ppl = \exp\biggl-\frac{1}{N}\sum_{n}\sum_{k}t_{n, k}\log p_{model}(y_{n, k})\giggr $$

[参考](https://data-analytics.fun/2022/01/15/understanding-perplexity/)


TeXのコマンドサンプルサイト
https://docs.moodle.org/dev/MediaWiki_TeX_test
'''