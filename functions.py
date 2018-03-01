from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from constants import unemp_frac, education_level, START_AGE, RETIRE_AGE

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    return values**(1-gamma) / (1-gamma)


def cal_income(coeffs, AltDeg, labor_inc_only):
    coeff_this_group = coeffs.loc[education_level[AltDeg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(START_AGE, RETIRE_AGE+1)      # 22 to 65

    if labor_inc_only:
        income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    else:
        income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3) * unemp_frac[AltDeg]
    return income


def read_input_data(income_fp, mortal_fp):
    age_coeff_and_var = pd.ExcelFile(income_fp)
    # age coefficients
    age_coeff = pd.read_excel(age_coeff_and_var, sheet_name='Coefficients')

    # decomposed variance
    std = pd.read_excel(age_coeff_and_var, sheet_name='Variance', header=[1, 2])
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

    # conditional survival probabilities
    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)

    return age_coeff, std, cond_prob


def exp_val(inc_with_shk_tran, exp_inc_shk_perm, savings_incr, grid_w, v, weight):
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


def exp_val_r(inc, savings_incr, grid_w, v):
    wealth = savings_incr + inc

    wealth[wealth > grid_w[-1]] = grid_w[-1]
    wealth[wealth < grid_w[0]] = grid_w[0]

    spline = interp1d(grid_w, v, kind='cubic')

    v_w = spline(wealth)

    ev = v_w
    return ev
