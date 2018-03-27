from scipy.interpolate import CubicSpline
import pandas as pd
import os
import numpy as np
from functions import utility, cal_income
from constants import START_AGE, END_AGE, RETIRE_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, N_SIM, MU, ret_frac, unempl_rate, unemp_frac, rho, ppt, INIT_WEALTH
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import sys

# policy functions: C_t(W_t)
def c_func(c_df, w, age):
    """ Given the consumption functions and wealth at certain age, return the corresponding consumption
    """
    w = np.where(w < UPPER_BOUND_W, w, UPPER_BOUND_W)
    w = np.where(w > 1, w, 1)
    spline = CubicSpline(c_df[str(END_AGE)], c_df[str(age)], bc_type='natural')
    c = spline(w)
    # MARK: check if there exists consumption < 0
    if any(c < 0):
        # set coeffs of 2nd and 3rd order term to 0, 1st order term to the slope between the first two points
        spline.c[:2, 0] = 0
        spline.c[2, 0] = (c_df.loc[1, str(age)] - c_df.loc[0, str(age)]) / (c_df.loc[1, str(END_AGE)] - c_df.loc[0, str(END_AGE)])
    c = spline(w)
    return c


def generate_consumption_process(income_bf_ret, sigma_perm_shock, sigma_tran_shock, c_func_df, AltDeg, flag):
    """ Calculating the certainty equivalent annual consumption and life time wealth

    :param age_coeff: a DataFrame
    :param std: a DataFrame
    :param AltDeg: ID for three education groups, e.g. 1 refers to No high schools
    :param c_func_dir: a string, stores the file path of 'consumption_*.xlsx'
    :return: None, but generates a 'ce.xlsx'
    """
    YEARS = END_AGE - START_AGE + 1


    # MARK: retirement income differs across individuals
    # income
    # before retirement
    rn_perm = np.random.normal(MU, sigma_perm_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    rn_tran = np.random.normal(MU, sigma_tran_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))

    # retirement
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income_bf_ret)
    ret_income_vec = ret_frac[AltDeg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)


    # add unemployment risks - generate bernoulli random variable
    p = 1 - unempl_rate[AltDeg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - START_AGE + 1, N_SIM)).astype(float)
    r[r == 0] = unemp_frac[AltDeg]

    ones = np.ones((END_AGE - RETIRE_AGE, N_SIM))
    bern = np.append(r, ones, axis=0)

    inc = np.multiply(inc_with_inc_risk, bern.T)

    # # ISA, Loan or origin
    # if flag == 'rho':
    #     inc[:, :10] *= rho
    # elif flag == 'ppt':
    #     inc[:, :10] -= ppt
    # else:
    #     pass

    ###########################################################################
    #                      COH_t+1 = R(COH_t - C_t) + Y_t+1                   #
    #                       wealth = R(COH_t - C_t)                           #
    ###########################################################################

    cash_on_hand = np.zeros((N_SIM, YEARS))
    c = np.zeros((N_SIM, YEARS))

    cash_on_hand[:, 0] = INIT_WEALTH + inc[:, 0]   # cash on hand at age 22

    # 0-77, calculate consumption from 22 to 99, cash on hand from 23 to 100
    for t in range(YEARS - 1):
        c[:, t] = c_func(c_func_df, cash_on_hand[:, t], t + START_AGE)
        cash_on_hand[:, t+1] = (1 + R) * (cash_on_hand[:, t] - c[:, t]) + inc[:, t+1]  # 1-78
    c[:, -1] = c_func(c_func_df, cash_on_hand[:, -1], END_AGE)   # consumption at age 100

    plt.plot(cash_on_hand.mean(axis=0), label='cash-on-hand')
    plt.plot(c.mean(axis=0), label='consumption')
    plt.title('Average Cash-on-hand and Consumption over the life cycle\n UPPER_BOUND_W = 3,000,000')
    plt.xlabel('Age')
    plt.ylabel('Dollar')
    plt.legend()
    plt.grid()
    plt.show()

    # c_process_df = pd.DataFrame(c)
    # c_process_df.to_excel(c_proc_fp, index=False)
    return c


def cal_certainty_equi(prob, c):

    # discount factor
    YEARS = END_AGE - START_AGE + 1
    delta = np.ones((YEARS, 1)) * DELTA
    delta[0] = 1
    delta = np.cumprod(delta)

    util_c = np.apply_along_axis(utility, 1, c, GAMMA)
    simu_util = np.sum(np.multiply(util_c[:, :44], (delta * prob)[:44]), axis=1)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # x = np.arange(N_SIM)
    # plt.scatter(x, simu_util)
    # plt.ylim(-10**(-3), 10**(-3))
    # plt.show()
    #
    # plt.figure()
    # b = -0.1**(np.arange(4, 16))
    # n, bins, patches = plt.hist(simu_util, bins=b)

    # MARK: ce calculation when gamma = 1
    if GAMMA == 1:
        c_ce = np.exp(np.mean(simu_util) / np.sum((delta * prob)[:44]))
    else:
        c_ce = ((1 - GAMMA) * np.mean(simu_util) / np.sum((delta * prob)[:44]))**(1 / (1-GAMMA))  # MARK: 44 periods
    total_w_ce = prob[:44].sum() * c_ce   # 42.7
    # print(c_ce, ', ', total_w_ce)

    return c_ce, total_w_ce
