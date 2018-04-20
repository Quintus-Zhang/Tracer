from scipy.interpolate import CubicSpline
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
from constants import *
import multiprocessing as mp

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


def adj_income_process(income, sigma_perm, sigma_tran):
    # generate random walk and normal r.v.
    rn_perm = np.random.normal(MU, sigma_perm, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    rn_tran = np.random.normal(MU, sigma_tran, (N_SIM, RETIRE_AGE - START_AGE + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income)

    # - retirement
    ret_income_vec = ret_frac[AltDeg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # # unemployment risk
    # Y_list = []
    # for unemp_flag in [True, False]:
    #     Y = inc_with_inc_risk * unemp_frac[AltDeg] if unemp_flag else inc_with_inc_risk
    #     Y_list.append(Y)
    # Y = unempl_rate[AltDeg] * Y_list[0] + (1 - unempl_rate[AltDeg]) * Y_list[1]      # include income risks and unemployment risk

    # unemployment risk
    # generate bernoulli random variable
    p = 1 - unempl_rate[AltDeg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - START_AGE + 1, N_SIM)).astype(float)
    r[r == 0] = unemp_frac[AltDeg]
    ones = np.ones((END_AGE - RETIRE_AGE, N_SIM))
    bern = np.append(r, ones, axis=0)
    Y = np.multiply(inc_with_inc_risk, bern.T)

    # # adjust income with debt repayment
    # D = np.zeros(Y.shape)
    # D[:, 0] = INIT_DEBT
    # P = np.zeros(Y.shape)
    #
    # for t in range(END_AGE - START_AGE):
    #     cond1 = np.logical_and(Y[:, t] >= 2 * P_BAR, D[:, t] >= P_BAR)
    #     cond2 = np.logical_and(Y[:, t] >= 2 * D[:, t], D[:, t] < P_BAR)
    #     cond3 = np.logical_and(Y[:, t] < 2 * P_BAR, D[:, t] >= P_BAR)
    #     cond4 = np.logical_and(Y[:, t] < 2 * D[:, t], D[:, t] < P_BAR)
    #
    #     P[cond1, t] = P_BAR
    #     P[cond2, t] = D[cond2, t]
    #     P[cond3, t] = Y[cond3, t] / 2
    #     P[cond4, t] = Y[cond4, t] / 2
    #
    #     D[:, t + 1] = D[:, t] * (1 + rate) - P[:, t]
    # adj_Y = Y - P

    adj_Y = Y
    adj_Y[:, :TERM] *= rho

    return adj_Y


def exp_val_new(y, savings_incr, grid_w, v):

    COH = np.zeros((N_SIM, N_C))
    COH[:] = np.squeeze(savings_incr)
    COH += y[None].T

    COH[COH > grid_w[-1]] = grid_w[-1]
    COH[COH < grid_w[0]] = grid_w[0]

    spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

    p = mp.Pool(processes=mp.cpu_count())
    v_w = p.apply(spline, args=(COH,))
    p.close()

    # for i in range(N_SIM):
    #     v_w[i, :] = spline(COH[i, :])

    ev = v_w.mean(axis=0)
    return ev[None].T



