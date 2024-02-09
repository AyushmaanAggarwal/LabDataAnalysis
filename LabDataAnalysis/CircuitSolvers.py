# Solving Circits for Physics 111A Lab
# Written by Ayushmaan Aggarwal
# Date Created: 1/21/2024


def voltage_divider(v_top, r_top, v_bottom, r_bottom):
    """
    Takes in top and bottom voltage and resistor and solves for
    - v_out - output voltage
    - v_r1 - voltage drop across the top resistor
    - v_r2 - voltage drop across the bottom resistor
    - I - current flowing between the 2 circuts

    >>> voltage_divider(1,1,1,1)
    (1.0, 0.0, 0.0, 0.0)
    >>> voltage_divider(1,1,0,1)
    (0.5, 0.5, -0.5, 0.5)
    """
    v1, r1, v2, r2 = v_bottom, r_top, v_top, r_bottom
    v_out = (v1 * r1 + v2 * r2) / (r1 + r2)
    v_r1 = v2 - v_out
    v_r2 = v1 - v_out
    I = v_r1 / r1
    # r_1_n = v_r1 / I
    # r_2_n = v_r2 / I

    return v_out, v_r1, v_r2, I  # , r_1_n, r_2_n


def resistor_in_parallel(r1, r2):
    return 1 / (1 / r1 + 1 / r2)


def resistor_in_series(r1, r2):
    return r1 + r2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
