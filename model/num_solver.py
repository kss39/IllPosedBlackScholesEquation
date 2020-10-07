import numpy as np
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize

from scipy.linalg import inv

from . import toeplitz_matrix as tm

tau = 1 / 255
m = 100


class QuadraticIE:
    """
    Quadratic Interpolation / Extrapolation object. Used for option ask/bid price and implied volatility extrapolation
    in DataBlock.
    """

    def __init__(self, day_one, day_two, day_three, stock_price=0.0):
        self.fit_function = np.poly1d(np.polyfit([-2 * tau, -1 * tau, 0], [day_one, day_two, day_three], 2))
        self.stock = stock_price

    def at_day(self, day: int):
        """
        Evaluate the function at the given day.
        :param day: the given day. Today is 0, tomorrow is 1, etc.
        :return: the value at the given day.
        """
        return np.polyval(self.fit_function, day * tau)

    def at_time(self, time):
        """
        Evaluate the function at the given time. Note the difference with at_day.
        :param time: the given time. Right now is 0, the next year is 1.
        :return: the value at the given time.
        """
        return np.polyval(self.fit_function, time)


class DataBlock:
    """
    DataBlock is a block of information containing three consecutive days of stock/option price used
    for solving the Black-Scholes Equations.
    """

    def __init__(self, today, option_ask, option_bid, volatility, stock_ask, stock_bid):
        """
        Initializes the DataBlock with given data. Asserts all given data format is correct.

        :param option_ask: option ask price for 3 days
        :param option_bid: option bid price for 3 days
        :param volatility: implied volatility for 3 days
        :param stock_ask: current stock ask price
        :param stock_bid: current stock bid price
        """
        self.date = today
        for i in [option_ask, option_bid, volatility]:
            assert len(i) == 3
        for i in {stock_ask, stock_bid}:
            assert type(i) == float
        self.s_a = stock_ask
        self.s_b = stock_bid
        self.u_a = QuadraticIE(*option_ask, self.s_a)
        self.u_b = QuadraticIE(*option_bid, self.s_b)
        self.volatility = QuadraticIE(*volatility)
        # Construct the auxiliary function F
        self.func_aux = construct_F_cont(self.u_a, self.u_b)
        # self.func_ax = construct_A(self.s_a, self.s_b)
        self.func_sigma2 = construct_sigma_squared(self.volatility)
        # self.system is the system of equation Ax=b.
        self.m = None
        self.beta = None
        self.system = None

    def create_system(self, m: int, beta: float):
        """
        Creates the system of equation with grid count m.

        :param m: the grid count
        :param beta: the regularization parameter
        :return: the system (A, b) which is in the equation Ax = b.
        """
        if self.m == m and self.beta == beta:
            return self.system
        self.m = m
        self.beta = beta

        f_cont = construct_F_cont(self.u_a, self.u_b)
        f_matrix = construct_F_matrix(f_cont, m, self.s_b, self.s_a)
        f_vector = f_matrix.reshape(-1)

        u_bd = np.zeros((m, m))
        u_bd[0] = f_matrix[0]
        u_bd[:, 0] = f_matrix[:, 0]
        u_bd[:, -1] = f_matrix[:, -1]
        u_bd = u_bd.reshape(-1)

        boundary_indices = np.ones((m, m), dtype=bool)
        boundary_indices[0] = False
        boundary_indices[:, 0] = False
        boundary_indices[:, -1] = False
        boundary_indices = boundary_indices.reshape(-1)

        s = np.linspace(self.s_b, self.s_a, self.m)
        t = np.linspace(0, 2 * tau, m)
        meshgrid = np.meshgrid(s, t)
        lu = A(tm.D_t(m, 2 * tau / m), tm.D_ss(m, self.s_a - self.s_b), R(meshgrid, self.func_sigma2))

        b_rhs = - lu @ u_bd

        lu = lu[:, boundary_indices]
        lu = lu[boundary_indices]
        b_rhs = b_rhs[boundary_indices]
        f_vector = f_vector[boundary_indices]

        j_beta = lambda u: np.linalg.norm(lu @ u - b_rhs) ** 2 + beta * np.linalg.norm(u - f_vector) ** 2

        self.j_beta = j_beta

    def solve(self):
        result_u = np.zeros(self.m ** 2 - 3 * self.m + 2)
        return minimize(self.j_beta, result_u, method='CG')

        # lu, f_matrix = self.system
        # result = inv(lu.T @ lu + np.identity(self.m**2)) @ (self.beta * f_matrix)
        # # result = linalg.cg(lu.T @ lu, self.beta * f_matrix)
        # return result


"""
Below are auxiliary functions.
"""


#
# def construct_F_cont(option_ask: QuadraticIE, option_bid: QuadraticIE):
#     """
#     Returns a function to evaluate the function F(x, t) defined in Paper pg. 7
#     :param option_ask: the option ask price
#     :param option_bid: the option bid price
#     :return: a function to evaluate the function F(x, t)
#     """
#     return lambda x, t: x * (option_ask.at_time(t) - option_bid.at_time(t)) + option_bid.at_time(t)


def construct_F_cont(option_ask: QuadraticIE, option_bid: QuadraticIE):
    """
    Returns a function to evaluate the function F(x, t) defined in new paper pg. 11
    :param option_ask: the option ask price
    :param option_bid: the option bid price
    :return: a function to evaluate the function F(x, t)
    """
    s_a = option_ask.stock
    s_b = option_bid.stock
    return lambda s, t: \
        s * (option_bid.at_time(t) - option_ask.at_time(t)) / (s_b - s_a) + \
        (option_ask.at_time(t) * s_b - option_bid.at_time(t) * s_a) / (s_b - s_a)


def construct_F_matrix(func, m: int, s_b: float, s_a: float):
    """
    Evaluates the auxiliary function F in the mesh grid space Q_2tau.

    :param func: the continuous function F
    :param m: the parameter M
    :param s_b: stock bid price
    :param s_a: stock ask price
    :return: a m*m dimension vector representing the F in finite elements
    """
    s = np.linspace(s_b, s_a, m)
    t = np.linspace(0, 2 * tau, m)
    grid = np.stack(np.meshgrid(s, t))
    return func(*grid)


# def construct_A(s_a: float, s_b: float):
#     """Returns a function evaluating A(x) on the given day.
#     """
#     diff = s_a - s_b
#     return lambda x: (255 / 2) * (((x * diff) + s_b) ** 2) / (diff ** 2)


def construct_sigma_squared(volatility: QuadraticIE):
    return lambda t: volatility.at_time(t) ** 2


def R(meshgrid, sigma_squared_func):
    x_values, t_values = meshgrid
    original_matrix = (x_values ** 2) * sigma_squared_func(t_values) / 2
    reshaped = original_matrix.reshape(-1)
    return np.diag(reshaped)


def A(D_t, D_ss, R):
    return D_t + R @ D_ss

# For test only
# block = DataBlock(today='10/19/2016',\
#                        option_ask = [0.86, 0.86, 0.86],\
#                        option_bid = [0.84, 0.85, 0.85],\
#                        volatility = [0.39456, 0.38061, 0.37096],\
#                        stock_ask = 4.66,\
#                        stock_bid = 4.65)
#
# print(block.create_system(5, 0.01))
# print(block.solve())

# test_block = DataBlock(today='10/19/2016',\
#                        option_ask = [8.44999981, 8.55000019, 9.10000038],\
#                        option_bid = [7.05000019, 7.8499999, 8.5],\
#                        volatility = [0.39456, 0.38061, 0.37096],\
#                        stock_ask = 40.66,\
#                        stock_bid = 40.65)

# TODO:
# 1. volatility should divide by 100
# 2. change tau to 1 instead of 1/255
