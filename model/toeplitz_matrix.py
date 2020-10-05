import numpy as np


def D_t(m: int, dt: float):
    diff = np.identity(m)
    diff[0][0] = 0
    for i in range(1, m):
        diff[i][i - 1] = -1
    matrix = np.kron(diff, np.identity(m)) / dt
    return matrix


def D_ss(m: int, ds: float):
    diff = -2 * np.identity(m)
    diff[0][0] = 0
    diff[-1][-1] = 0
    for i in range(1, m-1):
        diff[i][i - 1] = -1
        diff[i][i + 1] = -1
    matrix = np.kron(np.identity(m), diff) / (ds ** 2)
    return matrix


