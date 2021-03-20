import numpy as np
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize

from ipbse.model import toeplitz_matrix as tm

tau = 1 / 255


class QuadraticIE:
    """
    Quadratic Interpolation / Extrapolation object. Used for option ask/bid price and implied volatility extrapolation
    in DataBlock.
    """

    def __init__(self, day_one, day_two, day_three, stock_price=0.0):
        self.to_list = [day_one, day_two, day_three]
        self.fit_function = np.poly1d(np.polyfit([-2 * tau, -1 * tau, 0], self.to_list, 2))
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

        # Used to store accurate historical data for later references.
        self.u_a_list = option_ask
        self.u_b_list = option_bid
        self.ivol_list = volatility
        # ----------------------------------

        self.u_a = QuadraticIE(*option_ask, self.s_a)
        self.u_b = QuadraticIE(*option_bid, self.s_b)
        self.volatility = QuadraticIE(*volatility)
        self.func_aux = construct_F_cont(self.u_a, self.u_b)
        self.func_sigma2 = construct_sigma_squared(self.volatility)
        self.m = None
        self.beta = None
        self.j_beta = None

    def data(self):
        """
        :return: A string version of the input data.
        """
        string_dict = {'today': self.date,
                       'option_ask': self.u_a.to_list,
                       'option_bid': self.u_b.to_list,
                       'stock_ask': self.s_a,
                       'stock_bid': self.s_b,
                       'volatility': self.volatility.to_list}
        return str(string_dict)

    def create_system(self, grid_count: int, beta: float, dt, dss):
        """
        Creates the system of equation with grid count m.
        Returns the Tikhonov-like functional J_beta which needs minimization.

        :param grid_count: the grid count
        :param beta: the regularization parameter
        :return: the Tikhonov-like functional J_beta
        """
        if self.m == grid_count and self.beta == beta:
            return self.j_beta
        self.m = grid_count
        self.beta = beta

        f_cont = construct_F_cont(self.u_a, self.u_b)
        f_matrix = construct_F_matrix(f_cont, grid_count, self.s_b, self.s_a)
        f_vector = f_matrix.reshape(-1)

        u_bd = np.zeros((grid_count, grid_count))
        u_bd[0] = f_matrix[0]
        u_bd[:, 0] = f_matrix[:, 0]
        u_bd[:, -1] = f_matrix[:, -1]
        u_bd = u_bd.reshape(-1)

        boundary_indices = np.ones((grid_count, grid_count), dtype=bool)
        boundary_indices[0] = False
        boundary_indices[:, 0] = False
        boundary_indices[:, -1] = False
        boundary_indices = boundary_indices.reshape(-1)

        s = np.linspace(self.s_b, self.s_a, self.m)
        t = np.linspace(0, 2 * tau, grid_count)
        meshgrid = np.meshgrid(s, t)
        lu = A(dt(grid_count, 2 * tau / grid_count),
               dss(grid_count, (self.s_a - self.s_b) / grid_count),
               R(meshgrid, self.func_sigma2))

        b_rhs = - lu @ u_bd

        lu = lu[:, boundary_indices]
        lu = lu[boundary_indices]
        b_rhs = b_rhs[boundary_indices]
        f_vector = f_vector[boundary_indices]

        # Normalize each row of the system Ax = b, to prevent float overflowing
        norm = np.linalg.norm(lu)
        lu = lu / norm
        b_rhs /= norm

        # Construct the Tikhonov-like functional
        def j_beta(u):
            return np.linalg.norm(lu @ u - b_rhs) ** 2 + beta * np.linalg.norm(u - f_vector) ** 2
        self.j_beta = j_beta

        return j_beta

    def solve(self):
        """
        Solves the DataBlock. If the J_beta hasn't been created yet, it will throw an error.

        :return: a scipy.optimize.OptimizeResult representing the minimizer
        """
        result_u = np.zeros(self.m ** 2 - 3 * self.m + 2)
        return minimize(self.j_beta, result_u, method='CG')


"""
Below are auxiliary functions.
"""


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


def construct_sigma_squared(volatility: QuadraticIE):
    """
    Returns a function to evaluate the function sigma^2(t).

    :param volatility: the extrapolated volatility.
    :return: the sigma^2(t) function
    """
    return lambda t: volatility.at_time(t) ** 2


def R(meshgrid, sigma_squared_func):
    """
    Returns a diagonal matrix with elements corresponding to the factor sigma^2(t) * s^2 / 2.

    :param meshgrid: Provided meshgrid of Q_2tau.
    :param sigma_squared_func: The sigma^2(t) function
    :return: the R diagonal matrix.
    """
    x_values, t_values = meshgrid
    original_matrix = (x_values ** 2) * sigma_squared_func(t_values) / 2
    reshaped = original_matrix.reshape(-1)
    return np.diag(reshaped)


def A(D_t, D_ss, R):
    """
    Construct the differential operator in matrix (discrete approximation) form.

    :param D_t: D_t toeplitz matrix
    :param D_ss: D_ss toeplitz matrix
    :param R: R diagonal matrix
    :return: A (Lu, the differential operator)
    """
    return D_t + R @ D_ss
