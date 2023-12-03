import numpy as np


if __name__ == '__main__':
    x = np.array([1, 2, 3])
    print('x.__class__:', x.__class__)
    print('x.shape:', x.shape)
    print('x.ndim:', x.ndim)
    W = np.array([[1, 2, 3], [4, 5, 6]])
    print('W.shape:', W.shape)
    print('W.ndim:', W.ndim)

    # Element-wise operation⇨アダマール積(要素ごとの積)
    W = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[0, 1, 2], [3, 4, 5]])
    print('W + X:', W + X)
    print('W * X:', W * X)

    # Inner product(内積, a @ b)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    y = np.dot(a, b)
    print(y)

    # Matrix multiplication(A @ B)
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    Y = np.dot(A, B)
    print(Y)
