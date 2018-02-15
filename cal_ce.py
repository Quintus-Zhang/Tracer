from scipy.interpolate import interp1d
import pandas as pd
import os
import numpy as np
from functions import utility
from constants import START_AGE, END_AGE, RETIRE_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, N_SIM, MU


# policy functions: C_t(W_t)
def c_func(c_df, w, age):
    """Given the consumption functions and wealth at certain age, return the corresponding consumption"""
    # w = min(w, c_df.loc[c_df.index[-1], str(END_AGE)])
    # w = max(w, c_df.loc[c_df.index[0], str(END_AGE)])
    w = np.where(w < c_df.loc[c_df.index[-1], str(END_AGE)], w, c_df.loc[c_df.index[-1], str(END_AGE)])
    w = np.where(w > c_df.loc[c_df.index[0], str(END_AGE)], w, c_df.loc[c_df.index[0], str(END_AGE)])
    spline = interp1d(c_df[str(END_AGE)], c_df[str(age)], kind='cubic')
    c = spline(w)
    return c


def cal_certainty_equi(income, std, surviv_prob, AltDeg, c_func_dir):

    YEARS = END_AGE - START_AGE + 1

    # read consumption data
    base_path = os.path.dirname(__file__)
    consmp_fp = os.path.join(base_path, 'results', c_func_dir, 'consumption_' + education_level[AltDeg] +'.xlsx')
    c_df = pd.read_excel(consmp_fp)

    # income
    income = income.loc[income['AltDeg'] == AltDeg]
    income.set_index('Age', inplace=True)
    income = income.loc[START_AGE:END_AGE]
    income.reset_index(inplace=True, drop=True)

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

    inc = np.multiply(np.exp(perm) * np.exp(tran), income['f'].values)  # inc.shape: (simu_N x 79)

    w = np.zeros_like(inc)
    c = np.zeros_like(inc)

    w[:, 0] = 1.1

    for t in range(YEARS-1):
        try:
            c[:, t] = c_func(c_df, w[:, t], t+START_AGE)
        except:
            print(w[:, t])
        w[:, t+1] = R * (w[:, t] - c[:, t]) + inc[:, t+1]
    c[:, -1] = c_func(c_df, w[:, -1], END_AGE)

    util_c = np.apply_along_axis(utility, 1, c, GAMMA)
    simu_util = np.sum(np.multiply(util_c, delta * prob)[:, 1:], axis=1)

    c_ce = -np.sum(delta * prob) / np.mean(simu_util)
    total_w_ce = np.sum(delta * c_ce)                  # TODO change

    return c_ce, total_w_ce


# def cal_certainty_equi(income, std, surviv_prob, AltDeg):
#
#     YEARS = END_AGE - START_AGE + 1
#
#     # read consumption data
#     base_path = os.path.dirname(__file__)
#     consmp_fp = os.path.join(base_path, 'results', 'c_v_2m_income_test', 'consumption_' + education_level[AltDeg] +'.xlsx')
#     c_df = pd.read_excel(consmp_fp)
#
#     # income
#     income = income.loc[income['AltDeg'] == AltDeg]
#     income.set_index('Age', inplace=True)
#     income = income.loc[START_AGE:END_AGE]
#     income.reset_index(inplace=True, drop=True)
#
#     # variance
#     sigma_perm_shock = std.loc['sigma_permanent', education_level[AltDeg]]
#     sigma_tran_shock = std.loc['sigma_transitory', education_level[AltDeg]]
#
#     # conditional survival probabilities
#     cond_prob = surviv_prob.loc[START_AGE:END_AGE, 'p']
#     prob = cond_prob.cumprod().values[None].T
#
#     # discount factor
#     delta = np.ones((YEARS, 1)) * DELTA
#     delta[0] = 1
#     delta = np.cumprod(delta)[None].T
#
#     simu = np.zeros((10000, 1))
#     for i in range(10000):
#         print(i)
#         rn_perm = np.random.normal(0, sigma_perm_shock, (RETIRE_AGE - START_AGE + 1, 1))
#         rn_tran = np.random.normal(0, sigma_tran_shock, (RETIRE_AGE - START_AGE + 1, 1)).T
#         rand_walk = np.cumsum(rn_perm)
#         zeros = np.zeros(END_AGE - RETIRE_AGE)
#
#         perm = np.append(rand_walk, zeros)
#         tran = np.append(rn_tran, zeros)
#
#         inc = income['f'].multiply(np.exp(perm) * np.exp(tran))
#
#         # calculating the wealth and consumption at each age
#         w = np.zeros((YEARS, 1))
#         c = np.zeros((YEARS, 1))
#         w[0] = 1.1
#         for t in range(YEARS-1):   # t: 0 to 77
#             c[t] = c_func(c_df, w[t], t+START_AGE)
#             w[t+1] = R * (w[t] - c[t]) + inc[t+1]  # income.loc[t+1, 'f']  #
#         c[-1] = c_func(c_df, w[-1], END_AGE)
#
#         # if any(np.isnan(c)):
#         #     print('\n')
#         #     print(i)
#         #     print('rn_perm')
#         #     print(rn_perm)
#         #     print('rand_walk')
#         #     print(rand_walk)
#         #     print('np.exp(perm)')
#         #     print(np.exp(perm))
#         #     print('inc')
#         #     print(inc)
#         #     print('wealth')
#         #     print(w)
#         #     print('consumption')
#         #     print(c)
#         #     return
#
#         simu[i, 0] = np.sum((delta * prob * utility(c, GAMMA))[1:])
#
#     c_ce = -np.sum(delta * prob) / np.mean(simu[:, 0])  # TODO prob not right
#     total_w_ce = np.sum(delta * c_ce)                   # TODO change
#
#     return c_ce, total_w_ce


# labor income
#                 average of c     sum of wealth
# no high school  27547.6864112    1375105.73210829
# high school     35067.4213978    1651169.71330568
# college         57223.8679602    2683333.35554625

# total income
#                 average of c     sum of wealth
# no high school  22094.2930475    1103415.7694
# high school     30079.5150547    1416768.00147
# college         53030.2787226    2487258.37053

# 100
#                 average of c     sum of wealth
# no high school  20049.0271382    1935226.66268
# high school     26184.96834      2391129.44889
# college         41982.2302443    4117647.08556

# discounted
#                 average of c     sum of wealth
# no high school   7595.1142992     808599.42599
# high school     10353.5293365    1018738.76052
# college         18193.286268     1798940.20439

# time seperable utility formula
# def cal_certainty_equi(income, std, surviv_prob, AltDeg):
#
#     YEARS = END_AGE - START_AGE + 1
#
#     # read consumption data
#     base_path = os.path.dirname(__file__)
#     consmp_fp = os.path.join(base_path, 'results', 'consumption_' + education_level[AltDeg] +'.xlsx')
#     c_df = pd.read_excel(consmp_fp)
#
#     # income
#     income = income.loc[income['AltDeg'] == AltDeg]
#     income.set_index('Age', inplace=True)
#     income = income.loc[START_AGE:END_AGE]
#     income.reset_index(inplace=True, drop=True)
#
#     # conditional survival probabilities
#     cond_prob = surviv_prob.loc[START_AGE:END_AGE, 'p']
#     prob = cond_prob.cumprod().values[None].T
#
#     # discount factor
#     delta = np.ones((YEARS, 1)) * DELTA
#     delta[0] = 1
#     delta = np.cumprod(delta)[None].T
#
#     # calculating the wealth and consumption at each age
#     w = np.zeros((YEARS, 1))
#     c = np.zeros((YEARS, 1))
#     w[0] = 0.1
#     for t in range(YEARS-1):   # t: 0 to 77
#         c[t] = c_func(c_df, w[t], t+START_AGE)
#         w[t+1] = R * (w[t] - c[t]) + income.loc[t+1, 'f']
#     c[-1] = c_func(c_df, w[-1], END_AGE)
#
#     util_tot_w = np.sum((delta * prob * utility(w, GAMMA))[1:])
#
#     ce_tot_w = -np.sum(delta * prob) / util_tot_w
#
#     util_tot_c = np.sum((delta * prob * utility(c, GAMMA))[1:])
#     ce_tot_c = -np.sum(delta * prob) / util_tot_c
#
#     return ce_tot_c, ce_tot_w


# mortal probability random source
# def cal_certainty_equi(income, std, surviv_prob, AltDeg):
#
#     YEARS = END_AGE - START_AGE + 1
#
#     # read consumption data
#     base_path = os.path.dirname(__file__)
#     consmp_fp = os.path.join(base_path, 'results', 'consumption_' + education_level[AltDeg] +'.xlsx')
#     c_df = pd.read_excel(consmp_fp)
#
#     # income
#     income = income.loc[income['AltDeg'] == AltDeg]
#     income.set_index('Age', inplace=True)
#     income = income.loc[START_AGE:END_AGE]
#     income.reset_index(inplace=True, drop=True)
#
#     # conditional survival probabilities
#     cond_prob = surviv_prob.loc[START_AGE-1:END_AGE-1, 'p']
#     prob = cond_prob.cumprod().values[None].T
#
#     # discount factor
#     delta = np.ones((YEARS, 1)) * DELTA
#     delta[0] = 1
#     delta = np.cumprod(delta)[None].T
#
#     # calculating the wealth and consumption at each age
#     w = np.zeros((YEARS, 1))
#     c = np.zeros((YEARS, 1))
#     w[0] = 0.1
#     for t in range(YEARS-1):   # t: 0 to 77
#         c[t] = c_func(c_df, w[t], t+START_AGE)
#         w[t+1] = R * (w[t] - c[t]) + income.loc[t+1, 'f']
#     c[-1] = c_func(c_df, w[-1], END_AGE)
#
#     # adjust the wealth and consumption with mort and disc fac
#     # w_adj = prob * delta * w
#     # c_adj = prob * delta * c
#
#     c_dis = delta * c
#     c_cumsum = np.cumsum(c_dis)
#     c_cummean = c_cumsum / np.arange(1, 80)
#     c_cummean = np.delete(c_cummean, 0)
#
#     w_dis = delta * w
#     w_cumsum = np.cumsum(w_dis)
#     w_cumsum = np.delete(w_cumsum, 0)
#
#     prob = cond_prob.cumprod()
#     shifted_cond_prob = cond_prob.shift(-1)
#     shifted_cond_prob.loc[prob.index[-1]] = 0
#     prob = prob * (1-shifted_cond_prob)
#     prob.drop(prob.index[0], inplace=True)
#
#     exp_util_c = np.sum(prob.values * utility(c_cummean, GAMMA))
#     exp_util_w = np.sum(prob.values * utility(w_cumsum, GAMMA))
#
#     mean_c_ce = - 1 / exp_util_c
#     sum_w_ce = - 1 / exp_util_w
#
#     return mean_c_ce, sum_w_ce


# Y random source
# def cal_certainty_equi(income, std, surviv_prob, AltDeg):
#
#     YEARS = END_AGE - START_AGE + 1
#
#     # read consumption data
#     base_path = os.path.dirname(__file__)
#     consmp_fp = os.path.join(base_path, 'results', 'consumption_' + education_level[AltDeg] +'.xlsx')
#     c_df = pd.read_excel(consmp_fp)
#
#     # income
#     income = income.loc[income['AltDeg'] == AltDeg]
#     income.set_index('Age', inplace=True)
#     income = income.loc[START_AGE:END_AGE]
#     income.reset_index(inplace=True, drop=True)
#
#     # variance
#     sigma_perm_shock = std.loc['sigma_permanent', education_level[AltDeg]]
#     sigma_tran_shock = std.loc['sigma_transitory', education_level[AltDeg]]
#
#     # conditional survival probabilities
#     cond_prob = surviv_prob.loc[START_AGE:END_AGE, 'p']
#     prob = cond_prob.cumprod().values[None].T
#
#     # discount factor
#     delta = np.ones((YEARS, 1)) * DELTA
#     delta[0] = 1
#     delta = np.cumprod(delta)[None].T
#
#
#     z = np.zeros((1000, 2))
#     for i in range(1000):
#         print(i)
#         rn_perm = np.random.normal(0, sigma_perm_shock, (RETIRE_AGE - START_AGE + 1, 1))
#         rn_tran = np.random.normal(0, sigma_tran_shock, (RETIRE_AGE - START_AGE + 1, 1)).T
#         rand_walk = np.cumsum(rn_perm)
#         zeros = np.zeros(END_AGE - RETIRE_AGE)
#
#         perm = np.append(rand_walk, zeros)
#         tran = np.append(rn_tran, zeros)
#
#         inc = income['f'].multiply(np.exp(perm) * np.exp(tran))
#
#         # calculating the wealth and consumption at each age
#         w = np.zeros((YEARS, 1))
#         c = np.zeros((YEARS, 1))
#         w[0] = 1.1
#         for t in range(YEARS-1):   # t: 0 to 77
#             c[t] = c_func(c_df, w[t], t+START_AGE)
#             w[t+1] = R * (w[t] - c[t]) + inc[t+1]  # income.loc[t+1, 'f']  #
#         c[-1] = c_func(c_df, w[-1], END_AGE)
#
#         # adjust the wealth and consumption with mort and disc fac
#         w_adj = prob * delta * w
#         c_adj = prob * delta * c
#
#         if any(np.isnan(c_adj)):
#             print('\n')
#             print(i)
#             print('rn_perm')
#             print(rn_perm)
#             print('rand_walk')
#             print(rand_walk)
#             print('np.exp(perm)')
#             print(np.exp(perm))
#             print('inc')
#             print(inc)
#             print('wealth')
#             print(w)
#             print('consumption')
#             print(c)
#             print('adjusted consumption')
#             print(c_adj)
#             return
#
#         z[i, 0] = utility(np.sum(w_adj), GAMMA)
#         z[i, 1] = utility(np.mean(c_adj), GAMMA)
#     total_w_ce = -np.mean(z[:, 0])**(-1)
#     mean_c_ce = -np.mean(z[:, 1])**(-1)
#
#     # return np.mean(c_adj), np.sum(w_adj)
#     return mean_c_ce, total_w_ce