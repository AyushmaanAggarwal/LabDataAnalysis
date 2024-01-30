# Data Analysis for Physics 111A Lab
# Written by Ayushmaan Aggarwal
# Date Created: 9/13/2022

# Currently implemented features:
# Covariance, Variance, Standard Deviation, Correlation Coefficents, more

import numpy as np
import matplotlib.colors as mcolors
from uncertainties import ufloat

ufloat_types = [type(ufloat(0, 0)), type(ufloat(0, 0) / ufloat(1, 1))]
colors = list(mcolors.TABLEAU_COLORS)


def covariance(x, y):
    assert len(x) == len(y)
    u_x, u_y = np.mean(x), np.mean(y)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i] - u_x) * (y[i] - u_y)

    return sum_covar / (len(x) - 1)


def variance(x):
    u_x = np.mean(x)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i] - u_x) ** 2

    return sum_covar / (len(x) - 1)


def std(x):
    return (variance(x)) ** 0.5


def quartrature_sum(x):
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
    sigma_xy = covariance(x, y)
    sigma_2 = variance(x)
    m = sigma_xy / sigma_2

    y_mean = np.mean(y)
    x_mean = np.mean(x)
    c = y_mean - m * x_mean

    return m, c


def weighted_least_squares_linear(x, y, err):
    """
    Returns: [m, c], [m_err, c_err], [y_pred, res], [chi_squared]
    """
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
    return [m, c], [m_err, c_err], [y_pred, res], [chi_squared]


def weighted_average(x):
    """
    Assumes a roughly normal distribution of error to weigh each error by the
    inverse of the error squared
    Arguments:
    - x is the ufloat value with error

    """
    _, err = seperate_uncertainty_array(x)
    weights = 1 / np.square(err)
    weights = weights / sum(weights)
    return np.sum(np.multiply(weights, x))


def combine_linear_uncertainties(x, y, x_err, y_err):
    m, _ = simple_least_squares_linear(x, y)
    return quartrature_sum([y_err, m * x_err])


def agreement_test(x, y):
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


def gen_uncertain_array(x, key):
    return [ufloat(val, key(abs(val))) for val in x]


def seperate_uncertainty_array(x):
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
    """
    conversion_rate = 1
    match prefix:
        case "T":
            conversion_rate = 10e12
        case "G":
            conversion_rate = 10e9
        case "M":
            conversion_rate = 10e6
        case "k":
            conversion_rate = 10e3
        case "h":
            conversion_rate = 10e2
        case "da":
            conversion_rate = 10e1
        case "d":
            conversion_rate = 10e-1
        case "c":
            conversion_rate = 10e-2
        case "m":
            conversion_rate = 10e-3
        case "u":
            conversion_rate = 10e-6
        case "n":
            conversion_rate = 10e-9
        case "p":
            conversion_rate = 10e-12
        case _:
            raise ValueError("Must be a metric prefix")
