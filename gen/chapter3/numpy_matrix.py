import numpy as np


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


if __name__ == '__main__':

    # Transpose
    A = np.array([[1, 2, 3], [4, 5, 6]])
    print(A)
    print(A.T)

    # Determinant(加算)⇨np.linalg.det()
    A = np.array([[3, 4], [5, 6]])
    d = np.linalg.det(A)
    print(d)

    # Inverse Matrix(逆行列)⇨np.linalg.inv()
    A = np.array([[0.5, 1.0], [2.0, 3.0]])
    B = np.linalg.inv(A)
    print(B)
    print(A @ B)

    x = np.array([0, 0])
    mu = np.array([1, 2])
    cov = np.array([[1, 0], [0, 1]])
    y = multivariate_normal(x, mu, cov)
    print(y)



