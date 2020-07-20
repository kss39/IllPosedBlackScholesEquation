import numpy as np
import scipy.sparse.linalg as linalg

tau = 1 / 255


class QuadraticIE:
    """
    Quadratic Interpolation / Extrapolation object. Used for option ask/bid price and implied volatility extrapolation
    in DataBlock.
    """

    def __init__(self, day_one, day_two, day_three):
        self.fit_function = np.poly1d(np.polyfit([-2 * tau, -1 * tau, 0], [day_one, day_two, day_three], 2))

    def at_day(self, day: int):
        """
        Evaluate the function at the given day.
        :param day: the given day. Today is 0, tomorrow is 1, etc.
        :return: the value at the given day.
        """
        return np.polyval(self.fit_function, day * tau)

    def at_time(self, time):
        """
        Evalue the function at the given time. Note the difference with at_day.
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
        for i in {option_ask, option_bid, volatility}:
            assert len(i) == 3
        for i in {stock_ask, stock_bid}:
            assert type(i) == float
        self.u_a = QuadraticIE(*option_ask)
        self.u_b = QuadraticIE(*option_bid)
        self.volatility = QuadraticIE(*volatility)
        self.s_a = stock_ask
        self.s_b = stock_bid
        # Construct the auxiliary function F
        self.func_aux = construct_F(self.u_a, self.u_b)
        self.func_ax = construct_A(self.s_a, self.s_b)
        self.func_sigma2 = construct_sigma_squared(self.volatility)
        # self.m is the grid_count.
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
        matrix_A = np.zeros((m*m, m*m))
        vector_b = np.zeros(m*m)

        # Step sizes of x and t.
        step_x = 1 / (m - 1)
        step_t = 2 * tau / (m - 1)
        for i_t in range(m):
            t_value = i_t * step_t
            for j_x in range(m):
                x_value = j_x * step_x
                serial = i_t * m + j_x
                # vector b
                vector_b[serial] = 2 * beta * self.func_aux(x_value, t_value)
                # matrix A, if not at boundary
                if i_t != 0 and (j_x != 0 and j_x != m-1):
                    # dt^2 term
                    matrix_A[serial, serial] += 1 / step_t
                    matrix_A[serial-m, serial] -= 2 / step_t
                    matrix_A[serial-m, serial-m] += 1 / step_t
                    # dxdt term
                    coeff = self.func_sigma2(t_value) * self.func_ax(x_value) / step_t / (step_x ** 2)
                    matrix_A[serial, serial+1] += coeff
                    matrix_A[serial, serial] -= 2 * coeff
                    matrix_A[serial-1, serial] += coeff
                    matrix_A[serial-m, serial+1] -= coeff
                    matrix_A[serial-m, serial] += 2 * coeff
                    matrix_A[serial-m, serial-1] -= coeff
                    # dx^2 term
                    sq_coeff = (self.func_sigma2(t_value) * self.func_ax(x_value) / step_x) ** 2
                    matrix_A[serial+1, serial+1] += sq_coeff
                    matrix_A[serial, serial] += 4 * sq_coeff
                    matrix_A[serial - 1, serial - 1] += sq_coeff
                    matrix_A[serial, serial+1] -= 4 * sq_coeff
                    matrix_A[serial-1, serial] -= 4 * sq_coeff
                    matrix_A[serial-1, serial+1] += 2 * sq_coeff

        # TODO: ask Kirill about the *2 thing.
        matrix_A *= 2

        self.system = (matrix_A, vector_b)
        return self.system

    def solve(self):
        assert self.system is not None
        A, b = self.system
        return linalg.cg(A, b)


"""
Below are auxiliary functions.
"""


def construct_F(option_ask: QuadraticIE, option_bid: QuadraticIE):
    """
    Returns a function to evaluate the function F(x, t) defined in Paper pg. 7
    :param option_ask: the option ask price
    :param option_bid: the option bid price
    :return: a function to evaluate the function F(x, t)
    """
    return lambda x, t: x * (option_ask.at_time(t) - option_bid.at_time(t)) + option_bid.at_time(t)


def construct_A(s_a: float, s_b: float):
    """Returns a function evaluating A(x) on the given day.
    """
    diff = s_a - s_b
    return lambda x: (255 / 2) * (((x * diff) + s_b) ** 2) / (diff ** 2)


def construct_sigma_squared(volatility: QuadraticIE):
    return lambda t: volatility.at_time(t) ** 2

