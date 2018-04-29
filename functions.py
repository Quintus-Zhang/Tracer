from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from constants import *

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(C, gamma):
    """ Constant Relative Risk Aversion - Utility Function

    :param C: array or scalar, consumption
    :param gamma: scalar, risk preference parameter
    :return: array or scalar, utility
    """
    if gamma == 1:
        return np.log(C)   # TODO: add try-except block to catch the error
    else:
        try:
            return C**(1-gamma) / (1-gamma)
        except ZeroDivisionError as e:
            raise ValueError('Consumption cannot be zero.') from e


def cal_income(coeffs):
    """ Calculating income over age by age polynomials

    :param coeffs: DataFrame, containing coefficients of the age polynomials for all 3 education groups
    :return: array, income from age 22 to 65 for education group AltDeg
    """
    coeff_this_group = coeffs.loc[education_level[AltDeg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(START_AGE, RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    return income


def read_input_data(income_fp, mortal_fp):
    """ Read data - Income(coefficients of age polynomials), Income Shocks(std), Survival Probability(conditional prob)

    :param income_fp: file path of the income data
    :param mortal_fp: file path of the survival prob data
    :return:
        age_coeff: DataFrame, containing coefficients of the age polynomials for all 3 education groups
        std      : DataFrame, two groups of income shocks
        cond_prob: DataFrame, conditional survival prob from age 21 to 100
    """
    age_coeff_and_var = pd.ExcelFile(income_fp)
    age_coeff = pd.read_excel(age_coeff_and_var, sheet_name='Coefficients')       # - age coefficients

    std = pd.read_excel(age_coeff_and_var, sheet_name='Variance', header=[1, 2])  # - income shocks
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)

    return age_coeff, std, cond_prob


def exp_val(inc_with_shk_tran, exp_inc_shk_perm, savings_incr, grid_w, v, weight, age, flag):
    # ev = 0.0
    # for j in range(3):
    #     for k in range(3):
    #         inc = inc_with_shk_tran[j] * exp_inc_shk_perm[k]
    #
    #         wealth = savings_incr + inc
    #
    #         wealth[wealth > grid_w[-1]] = grid_w[-1]
    #         wealth[wealth < grid_w[0]] = grid_w[0]
    #
    #         spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends
    #
    #         v_w = spline(wealth)
    #         temp = weight[j] * weight[k] * v_w
    #         ev = ev + temp
    # ev = ev / np.pi   # quadrature
    # return ev

    ev_list = []
    for unemp_flag in [True, False]:
        ev = 0.0
        for j in range(3):
            for k in range(3):
                inc = inc_with_shk_tran[j] * exp_inc_shk_perm[k]
                inc = inc * unemp_frac[AltDeg] if unemp_flag else inc         # theta

                if age < START_AGE + TERM:
                    if flag == 'rho':
                        inc *= rho
                    elif flag == 'ppt':
                        inc -= ppt
                    else:
                        pass

                wealth = savings_incr + inc

                wealth[wealth > grid_w[-1]] = grid_w[-1]
                wealth[wealth < grid_w[0]] = grid_w[0]

                spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

                v_w = spline(wealth)
                temp = weight[j] * weight[k] * v_w
                ev = ev + temp
        ev = ev / np.pi   # quadrature
        ev_list.append(ev)
    ev_all_include = unempl_rate[AltDeg] * ev_list[0] + (1 - unempl_rate[AltDeg]) * ev_list[1]      # include income risks and unemployment risk
    return ev_all_include


def exp_val_r(inc, exp_inc_shk_perm, savings_incr, grid_w, v, weight):
    ev = 0.0
    for k in range(3):
        wealth = savings_incr + inc * exp_inc_shk_perm[k] * ret_frac[AltDeg]

        wealth[wealth > grid_w[-1]] = grid_w[-1]
        wealth[wealth < grid_w[0]] = grid_w[0]

        spline = CubicSpline(grid_w, v, bc_type='natural')

        v_w = spline(wealth)
        temp = weight[k] * v_w
        ev = ev + temp
    ev = ev / np.sqrt(np.pi)
    return ev


# def exp_val_r(inc, savings_incr, grid_w, v):
#     wealth = savings_incr + inc
#
#     wealth[wealth > grid_w[-1]] = grid_w[-1]
#     wealth[wealth < grid_w[0]] = grid_w[0]
#
#     spline = CubicSpline(grid_w, v, bc_type='natural')
#
#     v_w = spline(wealth)
#
#     ev = v_w
#     return ev

