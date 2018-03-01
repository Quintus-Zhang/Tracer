from scipy.interpolate import interp1d
import pandas as pd
import os
import numpy as np
from functions import utility, cal_income
from constants import START_AGE, END_AGE, RETIRE_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, N_SIM, MU, ret_frac, unempl_rate


# policy functions: C_t(W_t)
def c_func(c_df, w, age):
    """ Given the consumption functions and wealth at certain age, return the corresponding consumption

    :param c_df:
    :param w:
    :param age:
    :return:
    """
    w = np.where(w < c_df.loc[c_df.index[-1], str(END_AGE)], w, c_df.loc[c_df.index[-1], str(END_AGE)])  # TODO: messy
    w = np.where(w > c_df.loc[c_df.index[0], str(END_AGE)], w, c_df.loc[c_df.index[0], str(END_AGE)])
    spline = interp1d(c_df[str(END_AGE)], c_df[str(age)], kind='cubic')
    c = spline(w)
    return c


def generate_consumption_process(income_bf_ret, income_ret, sigma_perm_shock, sigma_tran_shock, c_func_df, c_proc_fp):
    """ Calculating the certainty equivalent annual consumption and life time wealth

    :param age_coeff: a DataFrame
    :param std: a DataFrame
    :param AltDeg: ID for three education groups, e.g. 1 refers to No high schools
    :param c_func_dir: a string, stores the file path of 'consumption_*.xlsx'
    :return: None, but generates a 'ce.xlsx'
    """
    YEARS = END_AGE - START_AGE + 1

    ret_income_vec = income_ret * np.ones(END_AGE - RETIRE_AGE)
    income = np.append(income_bf_ret, ret_income_vec)

    rn_perm = np.random.normal(MU, sigma_perm_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    rn_tran = np.random.normal(MU, sigma_tran_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))

    zeros = np.zeros((N_SIM, END_AGE - RETIRE_AGE))
    perm = np.append(rand_walk, zeros, axis=1)
    tran = np.append(rn_tran, zeros, axis=1)

    inc = np.multiply(np.exp(perm) * np.exp(tran), income)  # inc.shape: (simu_N x 79)

    w = np.zeros((N_SIM, 80))
    c = np.zeros((N_SIM, 80))

    w[:, 0] = 0

    for t in range(YEARS):
        try:
            if t == 0:
                c[:, t] = 0
            else:
                c[:, t] = c_func(c_func_df, w[:, t], t-1+START_AGE)
        except:
            print(w[:, t])
        w[:, t+1] = R * (w[:, t] - c[:, t]) + inc[:, t]
    c[:, -1] = c_func(c_func_df, w[:, -1], END_AGE)

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
    simu_util = np.sum(np.multiply(util_c[:, 1:45], (delta * prob)[:44]), axis=1)

    c_ce = -np.sum(delta * prob) / np.mean(simu_util)
    total_w_ce = prob[:44].sum() * c_ce   # 42.7
    # print(c_ce, ', ', total_w_ce)

    return c_ce, total_w_ce
