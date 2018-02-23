from scipy.interpolate import interp1d
import pandas as pd
import os
import numpy as np
from functions import utility, cal_income
from constants import START_AGE, END_AGE, RETIRE_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, N_SIM, MU, ret_frac


# policy functions: C_t(W_t)
def c_func(c_df, w, age):
    """ Given the consumption functions and wealth at certain age, return the corresponding consumption

    :param c_df:
    :param w:
    :param age:
    :return:
    """
    # w = min(w, c_df.loc[c_df.index[-1], str(END_AGE)])
    # w = max(w, c_df.loc[c_df.index[0], str(END_AGE)])
    w = np.where(w < c_df.loc[c_df.index[-1], str(END_AGE)], w, c_df.loc[c_df.index[-1], str(END_AGE)])  # TODO: messy
    w = np.where(w > c_df.loc[c_df.index[0], str(END_AGE)], w, c_df.loc[c_df.index[0], str(END_AGE)])
    spline = interp1d(c_df[str(END_AGE)], c_df[str(age)], kind='cubic')
    c = spline(w)
    return c


def cal_certainty_equi(age_coeff, std, surviv_prob, AltDeg, c_func_dir=''):
    """ Calculating the certainty equivalent annual consumption and life time wealth

    :param age_coeff: a DataFrame
    :param std: a DataFrame
    :param surviv_prob: a DataFrame, conditional survival probability
    :param AltDeg: ID for three education groups, e.g. 1 refers to No high schools
    :param c_func_dir: a string, stores the file path of 'consumption_*.xlsx'
    :return: None, but generates a 'ce.xlsx'
    """
    YEARS = END_AGE - START_AGE + 1

    # read consumption data
    base_path = os.path.dirname(__file__)
    consmp_fp = os.path.join(base_path, 'results', c_func_dir, 'consumption_' + education_level[AltDeg] +'.xlsx')
    c_df = pd.read_excel(consmp_fp)

    # income
    coeff_this_group = age_coeff.loc[education_level[AltDeg]]
    ages = np.arange(START_AGE, RETIRE_AGE+1)      # 22 to 65
    income_bef_ret = cal_income(coeff_this_group, ages)    # 0:43, 22:65

    ret_income = ret_frac[AltDeg] * income_bef_ret[-1]
    ret_income_vec = ret_income * np.ones(END_AGE - RETIRE_AGE)

    income = np.append(income_bef_ret, ret_income_vec)

    # variance
    sigma_perm_shock = std.loc['sigma_permanent', education_level[AltDeg]]
    sigma_tran_shock = std.loc['sigma_transitory', education_level[AltDeg]]

    # conditional survival probabilities
    cond_prob = surviv_prob.loc[START_AGE:END_AGE, 'CSP']
    prob = cond_prob.cumprod().values

    # discount factor
    delta = np.ones((YEARS, 1)) * DELTA
    delta[0] = 1
    delta = np.cumprod(delta)

    rn_perm = np.random.normal(MU, sigma_perm_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    rn_tran = np.random.normal(MU, sigma_tran_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))

    zeros = np.zeros((N_SIM, END_AGE - RETIRE_AGE))
    perm = np.append(rand_walk, zeros, axis=1)
    tran = np.append(rn_tran, zeros, axis=1)

    inc = np.multiply(np.exp(perm) * np.exp(tran), income)  # inc.shape: (simu_N x 79)

    w = np.zeros_like(inc)
    c = np.zeros_like(inc)

    w[:, 0] = 0

    for t in range(YEARS-1):
        try:
            c[:, t] = c_func(c_df, w[:, t], t+START_AGE)
        except:
            print(w[:, t])
        w[:, t+1] = R * (w[:, t] - c[:, t]) + inc[:, t+1]
    c[:, -1] = c_func(c_df, w[:, -1], END_AGE)

    util_c = np.apply_along_axis(utility, 1, c, GAMMA)
    simu_util = np.sum(np.multiply(util_c, delta * prob)[:, 1:], axis=1)

    c_ce = -np.sum(delta * prob) / np.mean(simu_util)   # TODO: this formula should change with GAMMA
    total_w_ce = prob[:44].sum() * c_ce   # 42.7

    return c_ce, total_w_ce
