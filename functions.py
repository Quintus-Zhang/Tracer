from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from constants import *

def utility(C, gamma):
    """ Constant Relative Risk Aversion - Utility Function

    :param C: array or scalar, consumption
    :param gamma: scalar, risk preference parameter
    :return: array or scalar, utility
    """
    if gamma == 1:
        try:
            return np.log(C)
        except ValueError:
            raise ValueError('Consumption cannot be negative.')
    else:
        try:
            return C**(1-gamma) / (1-gamma)
        except ZeroDivisionError as e:
            raise ValueError('Consumption cannot be zero.') from e


def cal_income(coeffs):
    """ Calculating income over age by age polynomials

    :param coeffs: DataFrame, containing coefficients of the age polynomials for all 3 education groups
    :return: array, income from age 22 to 65 for education group {AltDeg}
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

    cond_prob = pd.read_excel(mortal_fp)                                          # - conditional survival probability
    cond_prob.set_index('AGE', inplace=True)

    return age_coeff, std, cond_prob


def exp_val(inc_with_shk_tran, exp_inc_shk_perm, savings_incr, grid_coh, v, weight, age, flag):
    """ Calculate the expected value of the backward cumulative utility within the working period using Gaussian
    Quadrature and Cubic Spline Interpolation

    :param inc_with_shk_tran:
    :param exp_inc_shk_perm : array with size 3-by-1, exponentiated permanent income shock
    :param savings_incr     : array with size N_C-by-1, savings increment
    :param grid_coh         : array with size (N_W, ), grid of cash-on-hand
    :param v                : array with size (N_W, ), backward cumulative utility(value function)
    :param weight           : array with size 3-by-1, weight from Gaussian Quadrature
    :param age              : int, actual age
    :param flag             : string, flag var to choose adding details of ISA or Loan or nothing(just the original model)
    :return: float64, expected value of the backward cumulative utility
    """
    ev_list = []
    for unemp_flag in [True, False]:
        ev = 0.0
        for j in range(3):
            for k in range(3):
                inc = inc_with_shk_tran[j] * exp_inc_shk_perm[k]
                inc = inc * unemp_frac[AltDeg] if unemp_flag else inc    # unemployment risk

                # MARK: ISA / Loan
                if age < START_AGE + TERM:
                    if flag == 'rho':
                        inc *= rho
                    elif flag == 'ppt':
                        inc -= ppt
                    else:
                        pass

                coh = savings_incr + inc

                coh[coh > grid_coh[-1]] = grid_coh[-1]  # If coh go across the boundary, then we set it to the boundary value.
                coh[coh < grid_coh[0]] = grid_coh[0]    # This makes sure that extrapolation won't happen here.

                spline = CubicSpline(grid_coh, v, bc_type='natural')  # 'natural' requires minimum curvature in the both ends

                v_coh = spline(coh)
                temp = weight[j] * weight[k] * v_coh
                ev = ev + temp
        ev = ev / np.pi     # quadrature
        ev_list.append(ev)
    ev_all_include = unempl_rate[AltDeg] * ev_list[0] + (1 - unempl_rate[AltDeg]) * ev_list[1]  # include the unemployment risk
    return ev_all_include


def exp_val_r(inc, exp_inc_shk_perm, savings_incr, grid_coh, v, weight):
    """ Calculate the expected value of the backward cumulative utility within the retirement period using Gaussian
    Quadrature and Cubic Spline Interpolation

    :param inc             : float64, deterministic component at certain retirement age
    :param exp_inc_shk_perm: array with size 3-by-1, exponentiated permanent income shock
    :param savings_incr    : array with size N_C-by-1, savings increment
    :param grid_coh        : array with size (N_W, ), grid of cash-on-hand
    :param v               : array with size (N_W, ), backward cumulative utility(value function)
    :param weight          : array with size 3-by-1, weight from Gaussian Quadrature
    :return: float64, expected value of the backward cumulative utility
    """
    ev = 0.0
    for k in range(3):
        coh = savings_incr + inc * exp_inc_shk_perm[k] * ret_frac[AltDeg]  # cash-on-hand cumulation equation

        coh[coh > grid_coh[-1]] = grid_coh[-1]  # If coh go across the boundary, then we set it to the boundary value.
        coh[coh < grid_coh[0]] = grid_coh[0]    # This makes sure that extrapolation won't happen here.

        spline = CubicSpline(grid_coh, v, bc_type='natural')  # Interpolation
        v_coh = spline(coh)

        temp = weight[k] * v_coh
        ev = ev + temp
    ev = ev / np.sqrt(np.pi)
    return ev
