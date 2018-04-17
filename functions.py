from scipy.interpolate import interp2d, RectBivariateSpline
import numpy as np
import pandas as pd
from constants import *
import matplotlib.pyplot as plt



###########################################################################
#                              Functions                                  #
###########################################################################
def utility(values, gamma):
    if gamma == 1:
        return np.log(values)
    else:
        return values**(1-gamma) / (1-gamma)


def cal_income(coeffs):
    coeff_this_group = coeffs.loc[education_level[AltDeg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(START_AGE, RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
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


def exp_val(inc_with_shk_tran, exp_inc_shk_perm, savings_incr, debt, grid_w, grid_d, v, weight, age, flag):
    # ev = 0.0
    # for j in range(3):
    #     for k in range(3):
    #         inc = inc_with_shk_tran[j] * exp_inc_shk_perm[k]
    #
    #         COH = savings_incr + inc
    #
    #         COH[COH > grid_w[-1]] = grid_w[-1]
    #         COH[COH < grid_w[0]] = grid_w[0]
    #
    #         spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends
    #
    #         v_w = spline(COH)
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

                COH = savings_incr + inc

                COH[COH > grid_w[-1]] = grid_w[-1]
                COH[COH < grid_w[0]] = grid_w[0]

                # spline = interp2d(grid_w, grid_d, v, kind='cubic')  # minimum curvature in both ends
                spline = RectBivariateSpline(grid_d, grid_w, v)

                v_w = spline.ev(debt, COH)

                temp = weight[j] * weight[k] * v_w
                ev = ev + temp
        ev = ev / np.pi   # quadrature
        ev_list.append(ev)
    ev_all_include = unempl_rate[AltDeg] * ev_list[0] + (1 - unempl_rate[AltDeg]) * ev_list[1]      # include income risks and unemployment risk
    return ev_all_include


def exp_val_r(inc, exp_inc_shk_perm, savings_incr, debt, grid_w, grid_d, v, weight):
    ev = 0.0
    for k in range(3):
        COH = savings_incr + inc * exp_inc_shk_perm[k] * ret_frac[AltDeg]

        COH[COH > grid_w[-1]] = grid_w[-1]
        COH[COH < grid_w[0]] = grid_w[0]

        # spline = interp2d(grid_w, grid_d, v, kind='cubic')  # minimum curvature in both ends
        spline = RectBivariateSpline(grid_d, grid_w, v)

        # plt.scatter(debt, COH)
        # plt.show()

        v_w = spline.ev(debt, COH)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x, y = np.meshgrid(COH[:1000], debt[:1000])
        # ax.scatter(x, y, v_w)
        # plt.show()

        temp = weight[k] * v_w
        ev = ev + temp
    ev = ev / np.sqrt(np.pi)
    return ev
