# 回帰
import torch
import matplotlib.pyplot as plt

SEED = 0


def predict(x):
    y = x @ W + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N


torch.manual_seed(SEED)
x = torch.rand(100, 1)
y = 5 + 2 * x + torch.rand(100, 1)

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

if __name__ == '__main__':
    lr = 0.1
    iters = 100

    for i in range(iters):
        y_hat = predict(x)
        loss = mean_squared_error(y, y_hat)

        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        W.grad.zero_()
        b.grad.zero_()

        if i % 10 == 0:
            print('loss:', loss.item())

    print('loss.item():', loss.item())
    print('W:', W.item())
    print('b:', b.item())

    plt.scatter(x.detach().numpy(), y.detach().numpy(), s=10)
    x = torch.tensor([[0.], [1.]])
    y = W.detach().numpy() * x.detach().numpy() + b.detach().numpy()
    plt.plot(x, y, color='red')
    plt.show()
    plt.close()




