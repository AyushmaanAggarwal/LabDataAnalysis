# Data Analysis for Physics 

import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, "same")

def line(x, m, b):
    return np.multiply(x, m) + b

def gaussian(x, mu, sigma):
    assert sigma >= 0
    exponent = -0.5*np.square(np.subtract(x, mu)/sigma)
    scale = 1/(np.sqrt(2*np.pi)*sigma)
    return scale*np.exp(exponent)