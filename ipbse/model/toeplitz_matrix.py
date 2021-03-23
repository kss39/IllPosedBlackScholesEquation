import numpy as np


def D_t_tshape(m: int, dt: float):
    diff = np.identity(m)
    diff[0][0] = 0
    for i in range(1, m):
        diff[i][i - 1] = -1
    identity_except_bound = np.identity(m)
    identity_except_bound[0, 0] = 0
    identity_except_bound[-1, -1] = 0
    matrix = np.kron(diff, identity_except_bound) / dt
    return matrix


def D_ss_tshape(m: int, ds: float):
    diff = -2 * np.identity(m)
    diff[0][0] = 0
    diff[-1][-1] = 0
    for i in range(1, m-1):
        diff[i][i - 1] = 1
        diff[i][i + 1] = 1
    identity_except_init = np.identity(m)
    identity_except_init[0, 0] = 0
    matrix = np.kron(identity_except_init, diff) / (ds ** 2)
    return matrix


# nt stands for "no t-shape".
def D_t_nt(m: int, dt: float):
    d = np.zeros((m,m))
    d[0][0] = -1
    d[0][1] = 1
    d[m-1][m-2] = -1
    d[m-1][m-1] = 1
    for i in range(m-2):
        d[i+1][i] = -0.5
        d[i+1][i+2] = 0.5
    return np.kron(d/dt, np.identity(m))


def D_ss_nt(m: int, ds: float):
    d = np.zeros((m,m))
    d[0][0] = 0.5
    d[0][1] = -1
    d[0][2] = 0.5
    d[m-1][m-3] = 0.5
    d[m-1][m-2] = -1
    d[m-1][m-1] = 0.5
    for i in range(m-2):
        d[i+1][i] = 1
        d[i+1][i+1] = -2
        d[i+1][i+2] = 1
    return np.kron(d/(ds*ds), np.identity(m))
