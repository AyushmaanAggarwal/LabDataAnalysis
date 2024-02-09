# Data Analysis for Physics 111A Lab
# Written by Ayushmaan Aggarwal
# Date Created: 9/13/2022

# Currently implemented features:
# Covariance, Variance, Standard Deviation, Correlation Coefficents, more

import numpy as np
import matplotlib.colors as mcolors
from uncertainties import ufloat, unumpy as unp

ufloat_types = [type(ufloat(0, 0)), type(ufloat(0, 0) / ufloat(1, 1))]
colors = list(mcolors.TABLEAU_COLORS)


def covariance(x, y):
    """
    Computes the covariance between 2 variables

    >>> covariance([1,2,3],[1,2,3])==1
    True
    >>> covariance([1,2,3],[3,2,1])==-1
    True
    >>> covariance([5,6,7],[3,2,1])==-1
    True
    """
    assert len(x) == len(y)
    u_x, u_y = np.mean(x), np.mean(y)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i] - u_x) * (y[i] - u_y)

    return sum_covar / (len(x) - 1)


def variance(x):
    """
    Computes variance across a variable 

    >>> variance([1,2,3])
    1.0
    """
    u_x = np.mean(x)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i] - u_x) ** 2

    return sum_covar / (len(x) - 1)


def std(x):
    return (variance(x)) ** 0.5


def quartrature_sum(x):
    """
    Equivalent to calculating the euclidian distance between points in a list given a list of residuals/errors
    
    >>> quartrature_sum([40, 20, 20])==np.sqrt(2400)
    True
    """
    sum_quart = 0
    for val in x:
        sum_quart += val**2
    return sum_quart**0.5


def correlation_coefficients(x, y):
    sigma_xy = covariance(x, y)
    sigma_x = np.sqrt(variance(x))
    sigma_y = np.sqrt(variance(y))
    return sigma_xy / (sigma_x * sigma_y)


def linear_fit_error(x, y, m, c, yerr):
    """
    Calculates error in the slope and intercept of a linear fit
    """
    assert len(x) == len(y)
    N = len(x)
    y_pred = np.array(x) * m + c
    alpha_cu = common_uncertainty(y_pred, y, m, c)
    alpha_cu = alpha_cu if alpha_cu > yerr else yerr
    sigma_x = variance(x)

    alpha_m = alpha_cu / (N * sigma_x**2)

    alpha_c = np.mean(np.array(x) ** 2) * alpha_m
    return alpha_m, alpha_c


def common_uncertainty(y_pred, y, m, c):
    assert len(y_pred) == len(y)
    summation = sum((np.array(y) - np.array(y_pred)) ** 2)
    return np.sqrt(summation / (len(y_pred) - 2))


def simple_least_squares_linear(x, y):
    """
    Calculates a simple linear fit

    >>> simple_least_squares_linear([1,2,3,4], [1,2,3,4])
    (1.0, 0.0)
    >>> simple_least_squares_linear([1,2,3,4], [4,3,2,1])
    (-1.0, 5.0)
    >>> simple_least_squares_linear([1,2,3,4], [10, 10, 10, 10])
    (0.0, 10.0)
    """
    sigma_xy = covariance(x, y)
    sigma_2 = variance(x)
    m = sigma_xy / sigma_2

    y_mean = np.mean(y)
    x_mean = np.mean(x)
    c = y_mean - m * x_mean

    return m, c


def weighted_least_squares_linear(x, y, err=[]):
    """
    Calculates an error weighted least squares linear fit
    Can take in uncertaintes.ufloat or a simple list with errors
    Input: x, y -> type ufloat
    Input 2: x, y, err -> type list
    Returns: [m, c], [y_pred, res], [chi_squared]
    """
    print("Warning: the output for this function has changed to support ufloat")
    print("the output is now")
    if err == []:
        err = combine_linear_uncertainties(x, y)

    sum_mult2 = lambda x, y: sum(np.multiply(x, y))
    sum_mult3 = lambda x, y, z: sum(np.multiply(np.multiply(x, y), z))

    w = np.divide(1, np.power(err, 2))
    x2 = np.power(x, 2)

    delta = sum(w) * sum_mult2(w, x2) - np.power(sum_mult2(w, x), 2)
    m = (sum(w) * sum_mult3(w, x, y) - sum_mult2(w, x) * sum_mult2(w, y)) / delta
    c = (sum_mult2(w, y) - m * sum_mult2(w, x)) / sum(w)

    m_err = np.sqrt(sum(w) / delta)
    c_err = np.sqrt(sum_mult2(w, x2) / delta)

    y_pred = np.add(np.multiply(m, x), c)
    res = np.subtract(y_pred, y)
    chi_squared = sum_mult2(w, np.power(res, 2))
    # print(f"m = {m:.2}±{m_err:.2}, c = {c:.2}±{c_err:.2}, Χ² = {chi_squared:.2}")
    # print(f"Equation: y = ({m:.2}±{m_err:.2})*x + ({c:.2}±{c_err:.2})")
    return [ufloat(m, m_err), ufloat(c, c_err)], [y_pred, res], [chi_squared]

def seperate_uncertainty_array(x):
    """
    Seperates the nominal values and uncertainties of an ufloat array
    Arguments:
    - x is the ufloat value with error
    >>> seperate_uncertainty_array(np.array([ufloat(0.0,2.0), ufloat(1.0, 1.0)]))
    (array([0., 1.]), array([2., 1.]))
    """
    return unp.nominal_values(x), unp.std_devs(x)

def weighted_average(x):
    """
    Computes the weighted average of a ufloat array
    Assumes a roughly normal distribution of error to weigh each error by the
    inverse of the error squared
    Arguments:
    - x is the ufloat value with error

    >>> weighted_average([ufloat(0.0,2.0), ufloat(1.0, 1.0)]).n 
    0.8
    >>> weighted_average([ufloat(0.0,1.0), ufloat(1.0, 1.0)]).n
    0.5
    """
    _, err = seperate_uncertainty_array(x)
    weights = 1 / np.square(err)
    weights = weights / np.sum(weights)
    return np.sum(np.multiply(weights, x))


def combine_linear_uncertainties(x, y, x_err=[], y_err=[]):
    """
    Combines the error in x and in y assuming a linear relationship between x and y
    """
    if x_err == [] and y_err == []:
        _, x_err = seperate_uncertainty_array(x)
        _, y_err = seperate_uncertainty_array(y)

    m, _ = simple_least_squares_linear(x, y)
    return quartrature_sum([y_err, m * x_err])


def agreement_test(x, y):
    """
    Returns output of agreement test by calculating if the x and y value passes a
    2 sigma agreement test
    Arugments: x, y -> ufloat
    Returns: boolean

    >>> agreement_test(ufloat(0,0), ufloat(0,0))
    True
    >>> agreement_test(ufloat(1,1), ufloat(0,1))
    True
    >>> agreement_test(ufloat(10,1), ufloat(0,1))
    False
    """
    assert type(x) in ufloat_types, "Agreement test only takes in ufloats"
    assert type(y) in ufloat_types, "Agreement test only takes in ufloats"

    difference = abs(abs(x.n) - abs(y.n))
    if difference == 0:
        return True

    error = np.sqrt(np.square(x.s) + np.square(y.s))
    if error == 0:
        return False

    agreement_value = difference / (error * 2)
    return agreement_value < 1


def get_uncertain_array(x, error):
    if type(error) in [type([]), type(np.array([]))]:
        return [ufloat(val, abs(err)) for val, err in zip(x, error)]
    return [ufloat(val, abs(error)) for val in x]


def gen_ufloat(x, key):
    return ufloat(x, abs(key(x)))


def gen_ufloat_array(x, key):
    return [ufloat(val, key(abs(val))) for val in x]


def seperate_ufloat_array(x):
    return [val.nominal_value for val in x], [val.std_dev for val in x]


def print_ufloats(x, digits=3):
    if type(x) in [type(np.array([])), type([])]:
        print("[", end="")
        for val in x:
            print(
                str(np.round(val.n, digits)) + " ± " + str(np.round(val.s, digits)),
                end=", ",
            )
        print("\b\b]")
    else:
        print(np.round(val.n, digits) + " ± " + np.round(val.s, digits))


# Unit Conversions (used to determine error bounds)
def metric_converter(x, prefix):
    """
    Converts Metric Prefix to No Prefix

    >>> metric_converter(1, "T") == 1e12
    True
    >>> metric_converter(10, "n") == 1e-8
    True
    """
    conversion_rate = 1
    match prefix:
        case "T":
            conversion_rate = 1e12
        case "G":
            conversion_rate = 1e9
        case "M":
            conversion_rate = 1e6
        case "k":
            conversion_rate = 1e3
        case "h":
            conversion_rate = 1e2
        case "da":
            conversion_rate = 1e1
        case "d":
            conversion_rate = 1e-1
        case "c":
            conversion_rate = 1e-2
        case "m":
            conversion_rate = 1e-3
        case "u":
            conversion_rate = 1e-6
        case "n":
            conversion_rate = 1e-9
        case "p":
            conversion_rate = 1e-12
        case _:
            raise ValueError("Must be a metric prefix")

    return x * conversion_rate


if __name__ == "__main__":
    import doctest

    doctest.testmod()
