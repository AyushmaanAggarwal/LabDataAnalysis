# Error Calculations for Physics 111A Lab
# Written by Ayushmaan Aggarwal
# Date Created: 1/20/2024

# Currently implemented features:
# Calculating errors from ADS and DMM

import numpy as np
from uncertainties import ufloat


# DMM Errors
def dmm_err_dc_voltage(voltage, digits=0.01):
    """
    Calculates the error in the DC voltage and returns a ufloat
    """
    assert voltage >= 200e-3, "Voltage should be >= 200mV"
    assert voltage <= 1e3, "Voltage should be <= 1000V"

    return ufloat(voltage, (0.5 / 100.0) * voltage + digits)


def dmm_err_ac_voltage(voltage, digits=0.01):
    """
    Calculates the error in the AC voltage and returns a ufloat
    """
    assert voltage >= 200e-3, "Voltage should be >= 200mV"
    assert voltage <= 750, "Voltage should be <= 750V"

    return ufloat(voltage, (0.8 / 100.0) * voltage + 3 * digits)


def dmm_err_dc_current(current, digits=0.01):
    """
    Calculates the error in the DC current and returns a ufloat
    """
    assert current >= 20e-3, "Current should be >= 20mA"
    assert current <= 20, "Current should be <= 20A"

    return ufloat(current, (0.8 / 100.0) * current + digits)


def dmm_err_ac_current(current, digits=0.01):
    """
    Calculates the error in the AC current and returns a ufloat
    """
    assert current >= 20e-3, "Current should be >= 20mA"
    assert current <= 20, "Current should be <= 20A"

    return ufloat(current, (1.0 / 100.0) * current + 3 * digits)


def dmm_err_resistance(resistance, digits=0.01):
    """
    Calculates the error in the resistance and returns a ufloat
    """
    assert resistance >= 200, "Resistance should be >= 200 Ohm"
    assert resistance <= 2000e6, "Resistance should be <= 2000 M Ohm"

    return ufloat(resistance, (2.5 / 100.0) * resistance + 3 * digits)


def dmm_err_capacitance(capacitance, digits=0.01):
    """
    Calculates the error in the capacitance and returns a ufloat
    """
    assert capacitance >= 2e-9, "Capacitance should be >= 2nF"
    assert capacitance <= 200e-6, "Capacitance should be <= 200uF"

    return ufloat(capacitance, (2.5 / 100.0) * capacitance + 5 * digits)


def resistance_err(resistance):
    return ufloat(resistance, 0.01 * resistance)


# ADS Errors
def ads_err_volt_voltmeter(voltage, digits=0.001):
    return ufloat(voltage, digits)


def ads_err_volt_oscilliscope(voltage, scale=0.5):
    if scale >= 1:
        return ufloat(voltage, 0.1 + (0.5 / 100.0) * voltage)
    elif scale <= 0.5:
        return ufloat(voltage, 0.01 + (0.5 / 100.0) * voltage)


def ads_err_voltage_output(voltage):
    ### WHAT IS V/div
    if voltage <= 1:
        return ufloat(voltage, 0.01 + 0.5 * voltage)
    else:
        return ufloat(voltage, 0.025 + 0.5 * voltage)
