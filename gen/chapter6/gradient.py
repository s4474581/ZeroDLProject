import torch


# ローゼンブロック関数⇨2変数での最適化アルゴリズムのベンチマークで使われる
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


if __name__ == '__main__':
    x0 = torch.tensor(0., requires_grad=True)
    x1 = torch.tensor(2., requires_grad=True)

    y = rosenbrock(x0, x1)
    y.backward()
    print('x0.grad:', x0.grad, ' x1.grad:', x1.grad)

    lr = 0.001
    iters = 10000

    for i in range(iters):
        if i % 1000 == 0:
            print('x0.item:', x0.item(), ' x1.item:', x1.item())

        y = rosenbrock(x0, x1)

        y.backward()

        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data

        x0.grad.zero_()
        x1.grad.zero_()

    print('x0.item:', x0.item(), ' x1.item:', x1.item())