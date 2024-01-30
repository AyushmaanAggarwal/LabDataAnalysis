# Error Calculations for Physics 111A Lab
# Written by Ayushmaan Aggarwal
# Date Created: 1/20/2024

# Currently implemented features:
# Calculating errors from ADS and DMM

# DMM Errors


def dmm_err_dc_voltage(voltage, digits=0.01):
    # print(f"Voltages should be between 200mV and 1000V")
    return (0.5 / 100.0) * voltage + digits


def dmm_err_ac_voltage(voltage, digits=0.01):
    # print(f"Voltages should be between 200mV and 750V")
    return (0.8 / 100.0) * voltage + 3 * digits


def dmm_err_dc_current(current, digits=0.01):
    # print(f"Currents should be between 20mA and 20A")
    return (0.8 / 100.0) * current + digits


def dmm_err_ac_current(current, digits=0.01):
    # print(f"Currents should be between 2mA and 20A")
    return (1.0 / 100.0) * current + 3 * digits


def dmm_err_resistance(resistance, digits=0.01):
    # print(f"Resistance should be between 200 Ohm and 2000M Ohm")
    return (2.5 / 100.0) * resistance + 3 * digits


def dmm_err_capacitance(capacitance, digits=0.01):
    # print(f"Capacitance should be between 2nF and 200uF")
    return (2.5 / 100.0) * capacitance + 5 * digits


def resistance_err(resistance):
    return 0.01 * resistance


# ADS Errors
def ads_err_volt_voltmeter(voltage, digits=0.001):
    return digits


def ads_err_volt_oscilliscope(voltage, scale=0.5):
    ### WHAT IS V/div
    if scale >= 1:
        return 0.1 + (0.5 / 100.0) * voltage
    elif scale <= 0.5:
        return 0.01 + (0.5 / 100.0) * voltage


def ads_err_voltage_output(voltage):
    ### WHAT IS V/div
    if voltage <= 1:
        return 0.01 + 0.5 * voltage
    else:
        return 0.025 + 0.5 * voltage
