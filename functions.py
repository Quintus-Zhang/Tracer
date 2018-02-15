from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    return values**(1-gamma) / (1-gamma)


def read_input_data(income_fp, mortal_fp):
    income_var = pd.ExcelFile(income_fp)
    # labor income process
    income = pd.read_excel(income_var, sheet_name='Income profiles')
    income.rename(columns={'TotYwUnempwSS': 'f'}, inplace=True)

    # decomposed variance
    std = pd.read_excel(income_var, sheet_name='Variance', skiprows=2)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

    # conditional survival probabilities
    surviv_prob = pd.read_excel(mortal_fp)
    surviv_prob.set_index('AGE', inplace=True)
    # surviv_prob.rename(columns={'CSP': 'p'}, inplace=True)

    return income, std, surviv_prob


def exp_val(inc_with_shk_tran, exp_inc_shk_perm, savings_incr, grid_w, v, weight):
    """

    :param inc_with_shk_tran:
    :param exp_inc_shk_perm:
    :param savings_incr:
    :param grid_w:
    :param v:
    :param weight:
    :return:
    """
    ev = 0.0
    for j in range(3):
        for k in range(3):
            inc = inc_with_shk_tran[j] * exp_inc_shk_perm[k]
            wealth = savings_incr + inc

            wealth[wealth > grid_w[-1]] = grid_w[-1]
            wealth[wealth < grid_w[0]] = grid_w[0]

            spline = interp1d(grid_w, v, kind='cubic')

            v_w = spline(wealth)
            temp = weight[j] * weight[k] * v_w
            ev = ev + temp
    ev = ev / np.pi   # quadrature
    return ev
